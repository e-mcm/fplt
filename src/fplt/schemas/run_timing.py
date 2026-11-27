"""Pandera schemas for ``fplt.analysis.run_timing`` UI tables (#291 PR3)."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl


class RunTimingPhaseSchema(pa.DataFrameModel):
    """One top-level or nested timing phase row for the This-week panel."""

    phase: str
    seconds: float = pa.Field(ge=0)
    share_pct: float = pa.Field(ge=0)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["phase"]  # noqa: RUF012


class RunTimingTransferSchema(pa.DataFrameModel):
    """One R2 transfer row (pull/push) with optional throughput."""

    op: str
    prefix: str = pa.Field(nullable=True)
    n_objects: int = pa.Field(nullable=True, ge=0)
    bytes: int = pa.Field(nullable=True, ge=0)
    seconds: float = pa.Field(ge=0)
    throughput_mib_s: float = pa.Field(nullable=True, ge=0)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


class RunTimingTrendSchema(pa.DataFrameModel):
    """One gameweek row for the ops timing trend chart."""

    season: str
    gw: int = pa.Field(ge=1, le=38)
    checkpoint: str
    total_s: float = pa.Field(ge=0)

    class Config(pa.DataFrameModel.Config):
        strict = False
        coerce = True
        unique = ["season", "gw", "checkpoint"]  # noqa: RUF012


RUN_TIMING_PHASE_COLUMNS: dict[str, pl.DataType] = {
    "phase": pl.Utf8(),
    "seconds": pl.Float64(),
    "share_pct": pl.Float64(),
}

RUN_TIMING_TRANSFER_COLUMNS: dict[str, pl.DataType] = {
    "op": pl.Utf8(),
    "prefix": pl.Utf8(),
    "n_objects": pl.Int64(),
    "bytes": pl.Int64(),
    "seconds": pl.Float64(),
    "throughput_mib_s": pl.Float64(),
}

RUN_TIMING_TREND_COLUMNS: dict[str, pl.DataType] = {
    "season": pl.Utf8(),
    "gw": pl.Int64(),
    "checkpoint": pl.Utf8(),
    "total_s": pl.Float64(),
}
