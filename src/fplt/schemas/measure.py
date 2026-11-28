"""Pandera schema for the common Measure metrics table (toolkit §2.3)."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl

from fplt.schemas.l2 import SCHEMA_VERSION

MEASURE_STAGES: tuple[str, ...] = (
    "components",
    "prediction",
    "decision",
    "season",
)

MEASURE_METRICS_COLUMNS: dict[str, pl.DataType] = {
    "arm": pl.Utf8(),
    "window": pl.Utf8(),
    "metric": pl.Utf8(),
    "value": pl.Float64(),
    "se": pl.Float64(),
    "n": pl.Int64(),
    "stage": pl.Utf8(),
    "slice": pl.Utf8(),
    "schema_version": pl.Int64(),
}


class MeasureMetricsSchema(pa.DataFrameModel):
    """Standard metrics table returned by every ``Measure`` adapter.

    Aligns with experiment ``arm_metrics`` keys (arm × window × metric → value,
    se, n, stage) plus ``slice`` so component / forecast / board scorers share
    one shape for screens and dashboards
    (``docs/proposals/model-improvement-toolkit.md`` §2.3).
    """

    arm: str
    window: str
    metric: str
    value: float = pa.Field(nullable=True)
    se: float = pa.Field(nullable=True)
    n: int = pa.Field(ge=0)
    stage: str
    slice: str
    schema_version: int = pa.Field(ge=1)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


__all__ = [
    "MEASURE_METRICS_COLUMNS",
    "MEASURE_STAGES",
    "SCHEMA_VERSION",
    "MeasureMetricsSchema",
]
