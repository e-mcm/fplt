"""Pydantic schema for nightly performance bench results (#312).

Write-once under ``fplt-lab/bench/date=<UTC>/results.json``. Rolling baselines
are computed at read time from prior nights — not stored in this document.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, cast

from pydantic import BaseModel, ConfigDict, Field, field_validator

BENCH_RESULTS_SCHEMA_VERSION = 1


class BenchCaseDuration(BaseModel):
    """One pytest case duration captured for the nightly top-N list."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    nodeid: str = Field(min_length=1)
    seconds: float = Field(ge=0.0)


class BenchResults(BaseModel):
    """One night's fixed-fixture performance snapshot (``schema_version`` 1)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=BENCH_RESULTS_SCHEMA_VERSION, ge=1)
    date: str = Field(
        min_length=10,
        max_length=10,
        pattern=r"^\d{4}-\d{2}-\d{2}$",
        description="UTC calendar date of the run (YYYY-MM-DD).",
    )
    recorded_at: datetime
    git_sha: str | None = None
    phases: dict[str, float] = Field(default_factory=dict)
    mc_bench: dict[str, float] = Field(
        default_factory=dict,
        description="Per-step mc-bench wall times in milliseconds.",
    )
    pytest_top25: list[BenchCaseDuration] = Field(
        default_factory=list[BenchCaseDuration]
    )
    meta: dict[str, Any] = Field(default_factory=dict)

    @field_validator("phases", "mc_bench", mode="before")
    @classmethod
    def _positive_named_seconds(cls, value: object) -> dict[str, float]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise ValueError("phases/mc_bench must be a mapping")
        cleaned: dict[str, float] = {}
        for name, seconds in cast(dict[object, object], value).items():
            if not str(name).strip():
                raise ValueError("phase names must be non-empty")
            if isinstance(seconds, bool):
                raise ValueError(f"{name!r} value must be numeric")
            as_float = float(cast(float, seconds))
            if as_float <= 0:
                continue
            cleaned[str(name)] = as_float
        return cleaned

    def model_dump_json_doc(self) -> dict[str, Any]:
        """Return a JSON-ready dict (``recorded_at`` as ISO-8601 with ``Z``)."""
        payload = self.model_dump(mode="json")
        recorded = payload.get("recorded_at")
        if isinstance(recorded, str) and recorded.endswith("+00:00"):
            payload["recorded_at"] = recorded.replace("+00:00", "Z")
        return payload
