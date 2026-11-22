"""Pandera schemas for attainable-share and component-regret tables."""

from __future__ import annotations

import pandera.polars as pa


class ComponentRegretSchema(pa.DataFrameModel):
    """Per model × season × gameweek regret against hindsight components."""

    model_id: str
    season: str
    gw: int = pa.Field(ge=1, le=38)
    realised_points: int
    captaincy_regret: int = pa.Field(ge=0)
    xi_bench_regret: int = pa.Field(ge=0)
    transfer_regret: int = pa.Field(ge=0)
    chip_timing_regret: int = pa.Field(nullable=True, ge=0)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["model_id", "season", "gw"]


class AttainableShareSchema(pa.DataFrameModel):
    """Season totals with share of constrained-hindsight attainable points."""

    model_id: str
    season: str
    total_points: int
    attainable_points: int = pa.Field(ge=0)
    share_of_attainable: float = pa.Field(ge=0)
    gameweeks: int = pa.Field(ge=0)
    captaincy_regret: int = pa.Field(ge=0)
    xi_bench_regret: int = pa.Field(ge=0)
    transfer_regret: int = pa.Field(ge=0)
    chip_timing_regret: int = pa.Field(nullable=True, ge=0)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["model_id", "season"]
