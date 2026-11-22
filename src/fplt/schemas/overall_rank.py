"""Pandera schema for cached overall-league points-at-rank ladders."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl


class OverallRankLadderSchema(pa.DataFrameModel):
    """One rung on an overall-league points-at-rank ladder.

    ``percentile`` helpers consume these rows via
    ``fplt.backtest.live_percentile``; this schema validates frames that cross
    module boundaries (e.g. reports).
    """

    season: str
    captured_at: str
    league_id: int = pa.Field(ge=1)
    total_players: int = pa.Field(ge=1)
    target_rank: int = pa.Field(ge=1)
    observed_rank: int = pa.Field(ge=1)
    points: int = pa.Field(ge=0)
    page: int = pa.Field(ge=1)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


OVERALL_RANK_LADDER_COLUMNS: dict[str, pl.DataType] = {
    "season": pl.Utf8(),
    "captured_at": pl.Utf8(),
    "league_id": pl.Int64(),
    "total_players": pl.Int64(),
    "target_rank": pl.Int64(),
    "observed_rank": pl.Int64(),
    "points": pl.Int64(),
    "page": pl.Int64(),
}
