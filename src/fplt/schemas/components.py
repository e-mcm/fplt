"""Pandera schemas for component outputs that cross a module boundary."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl

MINUTE_BUCKETS = ("0", "1-59", "60+")


class InvolvementSchema(pa.DataFrameModel):
    """Per-90 goal and assist weights for one player in one fixture.

    Within a team and fixture the goal weights sum to 1, and so do the
    assist weights. The simulator turns those shares into scorers.
    """

    player_key: int
    fixture_id: int
    goal_weight: float = pa.Field(ge=0)
    assist_weight: float = pa.Field(ge=0)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["player_key", "fixture_id"]  # noqa: RUF012


class OtherPointsSchema(pa.DataFrameModel):
    """Expected points outside goals, assists, and clean sheets.

    One row per player, fixture, and minutes bucket. ``cards`` and ``other``
    are already in points, so a yellow card is negative. Defensive
    contribution is non-zero when staged clearances, tackles, and recoveries
    are present and the season awards those points.
    """

    player_key: int
    fixture_id: int
    minutes_bucket: str = pa.Field(isin=list(MINUTE_BUCKETS))
    bonus: float
    saves: float
    cards: float
    defensive_contribution: float
    other: float

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["player_key", "fixture_id", "minutes_bucket"]  # noqa: RUF012


INVOLVEMENT_COLUMNS: dict[str, pl.DataType] = {
    "player_key": pl.Int64(),
    "fixture_id": pl.Int64(),
    "goal_weight": pl.Float64(),
    "assist_weight": pl.Float64(),
}

OTHER_POINTS_COLUMNS: dict[str, pl.DataType] = {
    "player_key": pl.Int64(),
    "fixture_id": pl.Int64(),
    "minutes_bucket": pl.Utf8(),
    "bonus": pl.Float64(),
    "saves": pl.Float64(),
    "cards": pl.Float64(),
    "defensive_contribution": pl.Float64(),
    "other": pl.Float64(),
}
