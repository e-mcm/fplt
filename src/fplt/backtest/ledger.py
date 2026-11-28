"""Ledger helpers: column alignment and the read-only view entry point.

Predictions live under ``runs/``; actuals live in curated/staged outcomes.
``Store.ledger()`` joins them at read time. ``read_predictions`` is a thin
alias for that view so older call sites keep working. The former
``append_predictions`` / ``fill_actuals`` mutators are gone (B9 PR2).

Archived ``benchmark_projections`` (ep_next / OpenFPL) are projected into the
same view as mean-only pseudo-models (``model_id=archive:<source>``, #390).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from fplt.backtest.eval_set import OPENFPL_COMPARE_FROM
from fplt.schemas.benchmark_projections import (
    SOURCE_EP_NEXT,
    SOURCE_OPENFPL,
)
from fplt.schemas.ledger import LEDGER_COLUMNS

# Keep in sync with ``fplt.backtest.archive_compare.ARCHIVE_MODEL_PREFIX``.
ARCHIVE_MODEL_PREFIX = "archive:"
BENCHMARK_RUN_KIND = "benchmark"
LEDGER_BENCHMARK_SOURCES: tuple[str, ...] = (SOURCE_EP_NEXT, SOURCE_OPENFPL)


def ledger_path(root: Path) -> Path:
    """Legacy parquet path (read-only archive until B10 import)."""
    return root / "ledger" / "predictions" / "part.parquet"


def empty_ledger() -> pl.DataFrame:
    """An empty frame that still matches ``LedgerSchema``."""
    return pl.DataFrame(schema=LEDGER_COLUMNS)


def read_predictions(root: Path) -> pl.DataFrame:
    """Return the ledger view (predictions ⋈ outcomes).

    Prefer ``Store(root).ledger()`` at new call sites. This alias exists so
    boards and optimisers that still import ``read_predictions`` see the same
    view without rewriting every reader in B9.

    Args:
        root: Local data root (``$FPLT_DATA_DIR``).

    Returns:
        ``LedgerSchema`` rows (possibly empty).
    """
    from fplt.store import Store

    return Store(root).ledger()


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


def archive_model_id(source: str) -> str:
    """Ledger ``model_id`` for an archived benchmark ``source``."""
    return f"{ARCHIVE_MODEL_PREFIX}{source}"


def guard_openfpl_benchmarks(
    benchmarks: pl.DataFrame,
    *,
    compare_from: str = OPENFPL_COMPARE_FROM,
) -> pl.DataFrame:
    """Drop OpenFPL rows for seasons before ``compare_from``.

    AGENTS.md §6 / OpenFPL training window: never score OpenFPL on
    2020-21…2023-24. ``ep_next`` (and other sources) are unchanged.

    Args:
        benchmarks: ``BenchmarkProjectionSchema`` rows (or empty).
        compare_from: Inclusive season floor (default ``2024-25``).

    Returns:
        Filtered frame; empty input stays empty.
    """
    if benchmarks.is_empty() or "source" not in benchmarks.columns:
        return benchmarks
    floor = int(compare_from.split("-", 1)[0])
    as_of_year = pl.col("as_of_season").str.split("-").list.first().cast(pl.Int64)
    target_year = pl.col("target_season").str.split("-").list.first().cast(pl.Int64)
    early = (as_of_year < floor) | (target_year < floor)
    return benchmarks.filter((pl.col("source") != SOURCE_OPENFPL) | (~early))


def predictions_from_benchmarks(
    data_dir: Path,
    *,
    benchmarks: pl.DataFrame | None = None,
    players: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Project archived ep_next / OpenFPL rows onto ledger prediction shape.

    Mean only: ``xpts`` filled, event probabilities and ``xpts_sd`` null.
    ``model_id`` is ``archive:<source>``; ``run_kind`` is ``benchmark``.
    OpenFPL rows for seasons before ``OPENFPL_COMPARE_FROM`` are dropped.

    Args:
        data_dir: Root with ``benchmark_projections/`` and staged ``players``.
        benchmarks: Optional injected archive frame (tests); default reads disk.
        players: Optional staged players for ``position``; default reads staged.

    Returns:
        Ledger-shaped prediction rows with null actuals (possibly empty).
    """
    from fplt.data.benchmark_projections import read_benchmark_archive
    from fplt.data.store import DataStore

    archive = read_benchmark_archive(data_dir) if benchmarks is None else benchmarks
    if archive.is_empty():
        return empty_ledger()
    archive = archive.filter(pl.col("source").is_in(list(LEDGER_BENCHMARK_SOURCES)))
    archive = guard_openfpl_benchmarks(archive)
    if archive.is_empty():
        return empty_ledger()

    positions = _player_positions(
        DataStore(data_dir).read("players") if players is None else players
    )
    frame = archive.join(positions, on=["target_season", "player_key"], how="left")
    out = align_ledger_columns(
        frame.select(
            pl.concat_str(pl.lit(ARCHIVE_MODEL_PREFIX), pl.col("source")).alias(
                "model_id"
            ),
            pl.lit(BENCHMARK_RUN_KIND).alias("run_kind"),
            "as_of_season",
            "as_of_gw",
            "target_season",
            "target_gw",
            (pl.col("target_gw") - pl.col("as_of_gw") + 1)
            .clip(lower_bound=1)
            .alias("horizon"),
            "player_key",
            pl.col("position").fill_null("UNK"),
            "fixture_id",
            pl.col("xpts").cast(pl.Float64),
            pl.lit(None).cast(pl.Float64).alias("xpts_sd"),
            pl.lit(None).cast(pl.Float64).alias("p_haul"),
            pl.lit(None).cast(pl.Float64).alias("p_60"),
        )
    )
    return out.with_columns(
        *[pl.lit(None).cast(pl.Int64).alias(col) for col in _ACTUAL_COLS]
    ).select(list(LEDGER_COLUMNS))


def _player_positions(players: pl.DataFrame) -> pl.DataFrame:
    """Latest position per ``(season, player_key)`` for target joins."""
    schema = {
        "target_season": pl.Utf8(),
        "player_key": pl.Int64(),
        "position": pl.Utf8(),
    }
    if players.is_empty():
        return pl.DataFrame(schema=schema)
    needed = {"season", "player_key", "position", "available_at"}
    if not needed.issubset(players.columns):
        return pl.DataFrame(schema=schema)
    return (
        players.sort("available_at")
        .unique(subset=["season", "player_key"], keep="last")
        .select(
            pl.col("season").alias("target_season"),
            "player_key",
            "position",
        )
    )


_ACTUAL_COLS = (
    "actual_pts",
    "actual_minutes",
    "actual_goals",
    "actual_assists",
    "actual_cs",
)
