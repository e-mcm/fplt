"""Pandera schemas for predicted-XI accuracy audit reports."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl


class PredictedXiAuditSummarySchema(pa.DataFrameModel):
    """One metric row from ``fplt predicted-xi-audit``.

    ``slice`` encodes filters such as ``source=touchline_wire`` or
    ``hours_to_ko=0-6`` or ``confidence=0.8-1.0``.
    """

    metric: str
    slice: str
    value: float = pa.Field(nullable=True)
    n: int = pa.Field(ge=0)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


class PredictedXiAuditDetailSchema(pa.DataFrameModel):
    """One predicted-start row joined to the official start label."""

    season: str
    gw: int = pa.Field(nullable=True, ge=1, le=38)
    fixture_id: int
    team_key: int
    player_key: int
    source: str
    role: str
    confidence: float = pa.Field(nullable=True, ge=0.0, le=1.0)
    hours_to_ko: float = pa.Field(nullable=True)
    actual_started: bool
    correct: bool

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


PREDICTED_XI_AUDIT_SUMMARY_COLUMNS: dict[str, pl.DataType] = {
    "metric": pl.Utf8(),
    "slice": pl.Utf8(),
    "value": pl.Float64(),
    "n": pl.Int64(),
}

PREDICTED_XI_AUDIT_DETAIL_COLUMNS: dict[str, pl.DataType] = {
    "season": pl.Utf8(),
    "gw": pl.Int64(),
    "fixture_id": pl.Int64(),
    "team_key": pl.Int64(),
    "player_key": pl.Int64(),
    "source": pl.Utf8(),
    "role": pl.Utf8(),
    "confidence": pl.Float64(),
    "hours_to_ko": pl.Float64(),
    "actual_started": pl.Boolean(),
    "correct": pl.Boolean(),
}
