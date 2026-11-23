"""Pandera schemas for snapshot-backed minutes measurement reports."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl


class MinutesMeasurementSummarySchema(pa.DataFrameModel):
    """One metric row from ``fplt minutes-measure``.

    ``slice`` encodes filters such as ``source=fpl_api|regime=rotation`` or
    ``decision_pool|weighted``.
    """

    model_name: str
    metric: str
    slice: str
    value: float = pa.Field(nullable=True)
    n: int = pa.Field(ge=0)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


class MinutesMissSchema(pa.DataFrameModel):
    """One player–fixture miss row ranked by ``|ΔP(start)| × xPts``."""

    season: str
    gw: int = pa.Field(ge=1, le=38)
    player_key: int
    fixture_id: int
    position: str
    regime: str
    fpl_flag: str
    returning_from_absence: bool
    p_start: float = pa.Field(ge=0, le=1)
    started: float = pa.Field(ge=0, le=1)
    abs_delta_p_start: float = pa.Field(ge=0)
    xpts_weight: float = pa.Field(ge=0)
    impact: float = pa.Field(ge=0)
    in_decision_pool: bool

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


MINUTES_MEASUREMENT_SUMMARY_COLUMNS: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "metric": pl.Utf8(),
    "slice": pl.Utf8(),
    "value": pl.Float64(),
    "n": pl.Int64(),
}

MINUTES_MISS_COLUMNS: dict[str, pl.DataType] = {
    "season": pl.Utf8(),
    "gw": pl.Int64(),
    "player_key": pl.Int64(),
    "fixture_id": pl.Int64(),
    "position": pl.Utf8(),
    "regime": pl.Utf8(),
    "fpl_flag": pl.Utf8(),
    "returning_from_absence": pl.Boolean(),
    "p_start": pl.Float64(),
    "started": pl.Float64(),
    "abs_delta_p_start": pl.Float64(),
    "xpts_weight": pl.Float64(),
    "impact": pl.Float64(),
    "in_decision_pool": pl.Boolean(),
}
