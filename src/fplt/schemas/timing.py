"""Pydantic schema for wall-clock timing JSON records.

Shared by ``data_dir/timing/*.json``, run-folder ``logs/timing.json``, and
official ``ops_timing.json`` (issue #291). Schema version stays aligned with
``fplt.utils.timing.TIMING_SCHEMA_VERSION``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator

TIMING_RECORD_SCHEMA_VERSION = 1


class TimingRecord(BaseModel):
    """One wall-clock timing artifact (``schema_version`` 1).

    ``phases`` maps phase name → seconds. Nested names (``fit/score``) sit
    beside their parents so ``perf-report`` can aggregate across surfaces.
    ``total_s`` is the kind's top-level sum (not the sum of every nested key).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=TIMING_RECORD_SCHEMA_VERSION, ge=1)
    kind: str = Field(min_length=1)
    recorded_at: datetime
    meta: dict[str, Any] = Field(default_factory=dict)
    phases: dict[str, float] = Field(default_factory=dict)
    total_s: float = Field(ge=0.0)

    @field_validator("phases")
    @classmethod
    def _positive_phases(cls, value: dict[str, float]) -> dict[str, float]:
        cleaned: dict[str, float] = {}
        for name, seconds in value.items():
            if not name.strip():
                raise ValueError("phase names must be non-empty")
            # Reject bool (subclass of int) before coercing.
            if isinstance(seconds, bool):
                raise ValueError(f"phase {name!r} seconds must be numeric")
            as_float = float(seconds)
            if as_float <= 0:
                continue
            cleaned[name] = as_float
        return cleaned

    def model_dump_json_doc(self) -> dict[str, Any]:
        """Return a JSON-ready dict (``recorded_at`` as ISO-8601 with ``Z``)."""
        payload = self.model_dump(mode="json")
        recorded = payload.get("recorded_at")
        if isinstance(recorded, str) and recorded.endswith("+00:00"):
            payload["recorded_at"] = recorded.replace("+00:00", "Z")
        return payload
