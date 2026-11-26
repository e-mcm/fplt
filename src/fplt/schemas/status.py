"""Pydantic schema for the ops ``status.json`` document (ops-10).

Written under ``$FPLT_DATA_DIR/status/status.json`` and pushed to
``fplt-data/status/status.json`` (mutable overwrite; excluded from R2 locks).
Sibling ops fields that are not yet merged report ``null`` plus a ``reason``.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

STATUS_SCHEMA_VERSION = 1

# Capture GHA / ops jobs whose freshness feeds the heartbeat block.
STATUS_SOURCES: tuple[str, ...] = (
    "fpl_api",
    "understat",
    "api_football",
    "football_data",
)

HEARTBEAT_NAMES: tuple[str, ...] = (
    "snapshot",
    "understat",
    "api_football",
    "football_data",
    "ops_tick",
)

# Fields that must never appear on a public / website copy.
PRIVATE_STATUS_FIELDS: frozenset[str] = frozenset({"token_expiry", "last_error"})


class CaptureSourceStatus(BaseModel):
    """Last raw capture for one source under ``raw/<source>/``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(min_length=1)
    captured_at: datetime | None = None
    path: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _null_needs_reason(self) -> CaptureSourceStatus:
        if self.captured_at is None and not self.reason:
            raise ValueError("captured_at null requires reason")
        if self.captured_at is not None and self.reason:
            raise ValueError("reason must be null when captured_at is set")
        return self


class PlanDeadlineStatus(BaseModel):
    """One production model's plan at one deadline (ops horizon)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(min_length=1)
    model_id: str | None = None
    season: str | None = None
    gw: int | None = Field(default=None, ge=1, le=38)
    horizon: int | None = Field(default=None, ge=1)
    written_at: datetime | None = None
    path: str | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _null_needs_reason(self) -> PlanDeadlineStatus:
        missing = self.written_at is None or self.season is None or self.gw is None
        if missing and not self.reason:
            raise ValueError("missing plan fields require reason")
        if not missing and self.reason:
            raise ValueError("reason must be null when a plan is present")
        return self


class NextDeadlineStatus(BaseModel):
    """Next unchecked deadline plus ops-02 checkpoint state when available."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    season: str | None = None
    gw: int | None = Field(default=None, ge=1, le=38)
    deadline_utc: datetime | None = None
    checkpoints_done: list[str] | None = None
    checkpoints_due: list[str] | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _null_needs_reason(self) -> NextDeadlineStatus:
        if self.deadline_utc is None and not self.reason:
            raise ValueError("deadline_utc null requires reason")
        return self


class TokenExpiryStatus(BaseModel):
    """FPL API token expiry (private; stripped from the public copy)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    expires_at: datetime | None = None
    expired: bool | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _null_needs_reason(self) -> TokenExpiryStatus:
        if self.expires_at is None and not self.reason:
            raise ValueError("expires_at null requires reason")
        if self.expires_at is not None and self.reason:
            raise ValueError("reason must be null when expires_at is set")
        return self


class QuarantineStatusItem(BaseModel):
    """One active ops-06 quarantine row."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    source: str = Field(min_length=1)
    capture_path: str = Field(min_length=1)
    gate: str = Field(min_length=1)
    detail: str = ""
    captured_at: datetime
    available_at: datetime


class LastErrorStatus(BaseModel):
    """Most recent ops failure recorded under ``status/last_error.json``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    at: datetime
    message: str = Field(min_length=1)
    source: str | None = None


class HeartbeatStamp(BaseModel):
    """Heartbeat-relevant freshness for one scheduled job."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    last_at: datetime | None = None
    reason: str | None = None

    @model_validator(mode="after")
    def _null_needs_reason(self) -> HeartbeatStamp:
        if self.last_at is None and not self.reason:
            raise ValueError("last_at null requires reason")
        if self.last_at is not None and self.reason:
            raise ValueError("reason must be null when last_at is set")
        return self


class StatusDocument(BaseModel):
    """Full private ops status document written each ops tick."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: int = Field(default=STATUS_SCHEMA_VERSION, ge=1)
    generated_at: datetime
    last_captures: list[CaptureSourceStatus]
    last_plans: list[PlanDeadlineStatus]
    next_deadline: NextDeadlineStatus
    token_expiry: TokenExpiryStatus
    quarantines: list[QuarantineStatusItem] = Field(
        default_factory=list[QuarantineStatusItem]
    )
    prod_sha: str | None = None
    prod_sha_reason: str | None = None
    last_error: LastErrorStatus | None = None
    last_error_reason: str | None = None
    heartbeats: list[HeartbeatStamp]

    @model_validator(mode="after")
    def _prod_and_error_reasons(self) -> StatusDocument:
        if self.prod_sha is None and not self.prod_sha_reason:
            raise ValueError("prod_sha null requires prod_sha_reason")
        if self.prod_sha is not None and self.prod_sha_reason:
            raise ValueError("prod_sha_reason must be null when prod_sha is set")
        if self.last_error is None and not self.last_error_reason:
            raise ValueError("last_error null requires last_error_reason")
        if self.last_error is not None and self.last_error_reason:
            raise ValueError("last_error_reason must be null when last_error is set")
        return self

    def model_dump_json_doc(self) -> dict[str, Any]:
        """Return a JSON-ready dict (datetimes as ISO-8601)."""
        return self.model_dump(mode="json")
