"""Pandera schema for forecast skill-by-horizon curves."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl


class HorizonSkillSchema(pa.DataFrameModel):
    """One MAE/RMSE row per model × lead-time horizon.

    ``mae_vs_h1`` / ``rmse_vs_h1`` are ratios vs the same model's horizon-1
    row (null when H=1 is missing or zero).
    """

    model_name: str
    model_id: str
    horizon: int = pa.Field(ge=1)
    mae: float = pa.Field(nullable=True)
    rmse: float = pa.Field(nullable=True)
    n: int = pa.Field(ge=0)
    mae_vs_h1: float = pa.Field(nullable=True)
    rmse_vs_h1: float = pa.Field(nullable=True)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


HORIZON_SKILL_COLUMNS: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "model_id": pl.Utf8(),
    "horizon": pl.Int64(),
    "mae": pl.Float64(),
    "rmse": pl.Float64(),
    "n": pl.Int64(),
    "mae_vs_h1": pl.Float64(),
    "rmse_vs_h1": pl.Float64(),
}
