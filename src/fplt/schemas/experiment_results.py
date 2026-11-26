"""Pandera schemas for experiment ``results/`` tables (Stage C §3.8)."""

from __future__ import annotations

import pandera.polars as pa


class ArmMetricsSchema(pa.DataFrameModel):
    """Arm × window × metric values under ``results/arm_metrics.parquet``."""

    arm: str
    window: str
    metric: str
    value: float
    se: float = pa.Field(nullable=True)
    n: int = pa.Field(ge=0)
    stage: str
    schema_version: int = pa.Field(ge=1)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


class PairedMetricsSchema(pa.DataFrameModel):
    """Treatment − control diffs under ``results/paired.parquet``."""

    treatment_arm: str
    control_arm: str
    window: str
    metric: str
    diff: float
    ci_lo: float = pa.Field(nullable=True)
    ci_hi: float = pa.Field(nullable=True)
    p: float = pa.Field(nullable=True)
    method: str = pa.Field(nullable=True)
    stage: str
    schema_version: int = pa.Field(ge=1)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


class PerUnitDiffSchema(pa.DataFrameModel):
    """Per-gameweek (or per-player) diffs under ``results/per_unit.parquet``."""

    treatment_arm: str
    control_arm: str
    window: str
    metric: str
    unit_season: str
    unit_gw: int = pa.Field(ge=1, le=38)
    player_key: int = pa.Field(nullable=True)
    diff: float
    stage: str
    schema_version: int = pa.Field(ge=1)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
