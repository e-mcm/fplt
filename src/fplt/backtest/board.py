"""Leaderboard slices over the prediction ledger."""

from __future__ import annotations

import polars as pl

from fplt.backtest.metrics import summarise


def leaderboard(
    ledger: pl.DataFrame,
    *,
    season: str | None = None,
    horizon: int | None = None,
    position: str | None = None,
) -> pl.DataFrame:
    """Score one slice of the prediction ledger.

    Args:
        ledger: Rows from ``read_predictions``.
        season: Target season to keep. ``None`` keeps every season.
        horizon: Horizon to keep. ``None`` keeps every horizon.
        position: Position to keep. ``None`` keeps every position.

    Returns:
        The ``summarise`` frame for the rows that remain.
    """
    frame = ledger
    if season is not None:
        frame = frame.filter(pl.col("target_season") == season)
    if horizon is not None:
        frame = frame.filter(pl.col("horizon") == horizon)
    if position is not None:
        frame = frame.filter(pl.col("position") == position)
    return summarise(frame)


def side_by_side(
    ledger: pl.DataFrame,
    *,
    season: str,
    gw: int,
    as_of_gw: int,
) -> pl.DataFrame:
    """Expected points for one gameweek, one column per model.

    Args:
        ledger: Rows from ``read_predictions``.
        season: Target season.
        gw: Target gameweek.
        as_of_gw: Deadline the predictions were made at.

    Returns:
        One row per player and fixture. Model ids are the value columns.
    """
    rows = ledger.filter(
        (pl.col("target_season") == season)
        & (pl.col("target_gw") == gw)
        & (pl.col("as_of_gw") == as_of_gw)
    )
    if rows.is_empty():
        return pl.DataFrame(
            schema={
                "player_key": pl.Int64(),
                "position": pl.Utf8(),
                "fixture_id": pl.Int64(),
            }
        )
    return rows.pivot(
        on="model_id",
        index=["player_key", "position", "fixture_id"],
        values="xpts",
        aggregate_function="first",
    ).sort("player_key", "fixture_id")
