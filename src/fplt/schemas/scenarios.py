"""Pandera schemas for shared prediction scenarios used by lineup SAA."""

from __future__ import annotations

import pandera.polars as pa


class ScenarioOutcomeSchema(pa.DataFrameModel):
    """One player outcome in one shared simulation draw.

    Scenarios are joint across the candidate pool for a single gameweek:
    the same ``sim`` index is a common random number for every player.
    Points and minutes are the realised FPL totals for that draw (DGW
    already aggregated to the player-gameweek).
    """

    player_key: int
    sim: int = pa.Field(ge=0)
    points: float
    minutes: int = pa.Field(ge=0)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["player_key", "sim"]
