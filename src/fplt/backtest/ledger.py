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
    checked = LedgerSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    path = ledger_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        checked = pl.concat([pl.read_parquet(path), checked], how="vertical_relaxed")
    checked.write_parquet(path)
    logger.info("appended %s rows to %s (total=%s)", frame.height, path, checked.height)


def read_predictions(root: Path) -> pl.DataFrame:
    """Read the prediction ledger, or an empty typed frame."""
    path = ledger_path(root)
    if not path.exists():
        return empty_ledger()
    return pl.read_parquet(path)


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
    ledger = pl.read_parquet(path)
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
    filled = (
        ledger.drop("actual_pts", "actual_minutes")
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
        )
        .drop("checked", "scored_pts", "scored_minutes")
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
