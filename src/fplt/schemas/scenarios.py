"""Pandera schemas for shared prediction scenarios used by lineup SAA."""

from __future__ import annotations

import pandera.polars as pa


class PredictionSampleSchema(pa.DataFrameModel):
    """One player–fixture outcome in one joint Monte Carlo draw.

    Rows that share ``sim`` within a fixture are scoreline-correlated
    (teammate stacks). Different fixtures use independent streams; the
    same ``sim`` index is still the CRN key when aggregating a DGW to
    player–gameweek grain for :class:`ScenarioOutcomeSchema`.
    """

    player_key: int
    fixture_id: int
    target_season: str
    target_gw: int = pa.Field(ge=1, le=38)
    sim: int = pa.Field(ge=0)
    points: float
    minutes: int = pa.Field(ge=0)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["player_key", "fixture_id", "sim"]


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
