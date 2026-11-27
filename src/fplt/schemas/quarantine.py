"""Pandera schema for ops-06 capture quarantine records.

A failed sanity gate writes one row per ``(capture_path, gate)``. Raw stays
write-once; staging later skips quarantined captures
(``docs/proposals/ops-hardening-issues.md`` §ops-06).
"""

from __future__ import annotations

from datetime import datetime

import pandera.polars as pa


class QuarantineRecordSchema(pa.DataFrameModel):
    """One failed capture sanity gate (ops-06).

    ``available_at`` is when the quarantine row was written (PIT).
    ``captured_at`` is the raw capture stamp being quarantined.
    """

    season: str
    source: str
    capture_path: str
    captured_at: datetime
    gate: str
    detail: str
    previous_capture_path: str = pa.Field(nullable=True)
    available_at: datetime
    schema_version: int = pa.Field(ge=1)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["capture_path", "gate", "available_at"]  # noqa: RUF012
