"""Pandera schemas for layered-component eval summaries."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl


class ComponentEvalSummarySchema(pa.DataFrameModel):
    """One metric row from ``fplt eval-components``.

    ``slice`` holds filters such as ``overall`` or ``source=vaastav``.
    """

    component: str
    metric: str
    model_name: str
    slice: str
    value: float = pa.Field(nullable=True)
    n: int = pa.Field(ge=0)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


class SimsSweepSchema(pa.DataFrameModel):
    """One ``n_sims`` setting from the Monte Carlo stability sweep."""

    n_sims: int = pa.Field(ge=1)
    wall_seconds: float = pa.Field(ge=0)
    mean_abs_delta_xpts: float = pa.Field(nullable=True, ge=0)
    max_abs_delta_xpts: float = pa.Field(nullable=True, ge=0)
    mean_xpts_sd: float = pa.Field(nullable=True, ge=0)
    analytic_se: float = pa.Field(nullable=True, ge=0)
    n_players: int = pa.Field(ge=0)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


class ComponentEvalPerMatchSchema(pa.DataFrameModel):
    """One fixture × arm × metric row from a scoreline component board."""

    season: str
    gw: int = pa.Field(ge=1)
    fixture_id: int
    arm: str
    metric: str
    value: float = pa.Field(nullable=True)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


COMPONENT_EVAL_COLUMNS: dict[str, pl.DataType] = {
    "component": pl.Utf8(),
    "metric": pl.Utf8(),
    "model_name": pl.Utf8(),
    "slice": pl.Utf8(),
    "value": pl.Float64(),
    "n": pl.Int64(),
}

SIMS_SWEEP_COLUMNS: dict[str, pl.DataType] = {
    "n_sims": pl.Int64(),
    "wall_seconds": pl.Float64(),
    "mean_abs_delta_xpts": pl.Float64(),
    "max_abs_delta_xpts": pl.Float64(),
    "mean_xpts_sd": pl.Float64(),
    "analytic_se": pl.Float64(),
    "n_players": pl.Int64(),
}

PER_MATCH_COLUMNS: dict[str, pl.DataType] = {
    "season": pl.Utf8(),
    "gw": pl.Int64(),
    "fixture_id": pl.Int64(),
    "arm": pl.Utf8(),
    "metric": pl.Utf8(),
    "value": pl.Float64(),
}
