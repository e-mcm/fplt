"""Append-only prediction ledger. Actuals are filled once a gameweek is checked."""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from fplt.data.store import DataStore
from fplt.schemas.ledger import LEDGER_COLUMNS, LedgerSchema

logger = logging.getLogger(__name__)


def ledger_path(root: Path) -> Path:
    """Parquet path of the prediction ledger."""
    return root / "ledger" / "predictions" / "part.parquet"


def empty_ledger() -> pl.DataFrame:
    """An empty frame that still matches ``LedgerSchema``."""
    return pl.DataFrame(schema=LEDGER_COLUMNS)


def append_predictions(root: Path, frame: pl.DataFrame) -> None:
    """Append ``frame`` to ``ledger/predictions/part.parquet``.

    Args:
        root: Data directory that contains ``ledger/``.
        frame: Rows matching ``LedgerSchema``.
    """
    if frame.is_empty():
        logger.debug("append_predictions skipped empty frame")
        return
    checked = LedgerSchema.validate(align_ledger_columns(frame))
    assert isinstance(checked, pl.DataFrame)
    path = ledger_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = align_ledger_columns(pl.read_parquet(path))
        checked = align_ledger_columns(
            pl.concat([existing, checked], how="vertical_relaxed")
        )
        validated = LedgerSchema.validate(checked)
        assert isinstance(validated, pl.DataFrame)
        checked = validated
    checked.write_parquet(path)
    logger.info("appended %s rows to %s (total=%s)", frame.height, path, checked.height)


def read_predictions(root: Path) -> pl.DataFrame:
    """Read the prediction ledger, or an empty typed frame."""
    path = ledger_path(root)
    if not path.exists():
        return empty_ledger()
    return align_ledger_columns(pl.read_parquet(path))


def align_ledger_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Pad missing event-prob / actual columns so older ledgers still load.

    Args:
        frame: Ledger-shaped rows that may omit newer nullable columns.

    Returns:
        Frame with every ``LEDGER_COLUMNS`` key present (null-filled when absent).
    """
    extras: list[pl.Expr] = []
    for name, dtype in LEDGER_COLUMNS.items():
        if name in frame.columns:
            continue
        extras.append(pl.lit(None).cast(dtype).alias(name))
    if extras:
        frame = frame.with_columns(extras)
    return frame.select(list(LEDGER_COLUMNS))


def fill_actuals(root: Path, store: DataStore) -> None:
    """Fill actuals for target gameweeks whose ``data_checked`` flag is set.

    Unchecked gameweeks stay null even when a stat row is already stored.
    The latest stat row is used once the gameweek is checked.

    Args:
        root: Data directory that contains the ledger.
        store: Staged tables, including ``player_fixture_stats`` and ``gameweeks``.
    """
    path = ledger_path(root)
    if not path.exists():
        logger.debug("fill_actuals: no ledger at %s", path)
        return
    ledger = align_ledger_columns(pl.read_parquet(path))
    logger.info("fill_actuals: scoring %s ledger rows", ledger.height)
    stats = (
        store.read("player_fixture_stats")
        .sort("available_at")
        .unique(subset=["season", "gw", "player_key", "fixture_id"], keep="last")
        .select(
            pl.col("season").alias("target_season"),
            pl.col("gw").alias("target_gw"),
            "player_key",
            "fixture_id",
            pl.col("total_points").alias("scored_pts"),
            pl.col("minutes").alias("scored_minutes"),
            pl.col("goals_scored").alias("scored_goals"),
            pl.col("assists").alias("scored_assists"),
            pl.col("clean_sheets").alias("scored_cs"),
        )
    )
    checked = (
        store.read("gameweeks")
        .filter(pl.col("data_checked"))
        .select(
            pl.col("season").alias("target_season"),
            pl.col("gw").alias("target_gw"),
        )
        .unique()
        .with_columns(pl.lit(True).alias("checked"))
    )
    drop_actuals = [
        col
        for col in (
            "actual_pts",
            "actual_minutes",
            "actual_goals",
            "actual_assists",
            "actual_cs",
        )
        if col in ledger.columns
    ]
    filled = (
        ledger.drop(*drop_actuals)
        .join(
            stats,
            on=["target_season", "target_gw", "player_key", "fixture_id"],
            how="left",
        )
        .join(checked, on=["target_season", "target_gw"], how="left")
        .with_columns(
            pl.when(pl.col("checked").fill_null(False))
            .then(pl.col("scored_pts"))
            .otherwise(None)
            .cast(pl.Int64)
            .alias("actual_pts"),
            pl.when(pl.col("checked").fill_null(False))
            .then(pl.col("scored_minutes"))
            .otherwise(None)
            .cast(pl.Int64)
            .alias("actual_minutes"),
            pl.when(pl.col("checked").fill_null(False))
            .then(pl.col("scored_goals"))
            .otherwise(None)
            .cast(pl.Int64)
            .alias("actual_goals"),
            pl.when(pl.col("checked").fill_null(False))
            .then(pl.col("scored_assists"))
            .otherwise(None)
            .cast(pl.Int64)
            .alias("actual_assists"),
            pl.when(pl.col("checked").fill_null(False))
            .then(pl.col("scored_cs"))
            .otherwise(None)
            .cast(pl.Int64)
            .alias("actual_cs"),
        )
        .drop(
            "checked",
            "scored_pts",
            "scored_minutes",
            "scored_goals",
            "scored_assists",
            "scored_cs",
        )
        .select(list(LEDGER_COLUMNS))
    )
    validated = LedgerSchema.validate(filled)
    assert isinstance(validated, pl.DataFrame)
    scored = validated.filter(pl.col("actual_pts").is_not_null()).height
    validated.write_parquet(path)
    logger.info(
        "fill_actuals complete: %s/%s rows have actual_pts",
        scored,
        validated.height,
    )
