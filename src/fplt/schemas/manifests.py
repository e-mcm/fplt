"""Pydantic manifests for runs, curated builds, and official pointers.

These are the Stage B2 JSON contracts from ``docs/proposals/fplt-data-model.md``
step 2. Experiment / batch manifests wait for Stage C.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RunKind = Literal["official", "adhoc", "experiment", "benchmark"]
RunMode = Literal["live", "backtest"]
RunType = Literal["predict", "plan", "eval"]
RunStatus = Literal["running", "succeeded", "failed"]
SquadMode = Literal["real", "virtual"]
LineRole = Literal["primary", "candidate"]


class AsOfDeadline(BaseModel):
    """One deadline a run predicted or planned against."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    season: str = Field(min_length=1)
    gw: int = Field(ge=1, le=38)
    deadline_utc: datetime


class RunManifest(BaseModel):
    """Immutable identity and provenance for one analytic run.

    Written as ``runs/run=<run_id>/manifest.json``. Identity for reuse is
    ``(spec_hash, code_sha, lockfile_hash, build_id, seeds)``. ``build_id``
    names the curated build the PITView was made from when one exists;
    ``raw_manifest_hash`` / ``staged_digests`` remain for migration / debug.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    run_id: str = Field(min_length=1)
    kind: RunKind
    mode: RunMode
    purpose: str = ""
    experiment_id: str | None = None
    arm: str | None = None
    run_type: RunType
    spec_hash: str = Field(min_length=1)
    spec_label: str = Field(min_length=1)
    model_id: str = Field(min_length=1)
    config_hashes: dict[str, str] = Field(default_factory=dict)
    line: str | None = None
    plan_horizon: int | None = Field(default=None, ge=1)
    squad_mode: SquadMode | None = None
    prev_run_id: str | None = None
    code_sha: str = Field(min_length=1)
    code_dirty: bool = False
    prod_tag: str | None = None
    fplt_mc_version: str | None = None
    lockfile_hash: str = Field(min_length=1)
    build_id: str | None = None
    raw_manifest_hash: str | None = None
    staged_digests: dict[str, str] = Field(default_factory=dict)
    as_of: list[AsOfDeadline] = Field(min_length=1)
    horizon: int = Field(ge=1)
    checkpoint: str | None = None
    seeds: dict[str, int] = Field(default_factory=dict)
    host: str = Field(min_length=1)
    started_at: datetime
    finished_at: datetime | None = None
    status: RunStatus
    parent_run_id: str | None = None
    tags: dict[str, str] = Field(default_factory=dict)

    @field_validator("config_hashes", "staged_digests", "tags")
    @classmethod
    def _non_empty_keys(cls, value: dict[str, str]) -> dict[str, str]:
        for key in value:
            if not key.strip():
                raise ValueError("dict keys must be non-empty")
        return value

    @model_validator(mode="after")
    def _status_timestamps(self) -> RunManifest:
        if self.status == "running" and self.finished_at is not None:
            raise ValueError("running manifests must not set finished_at")
        if self.status in ("succeeded", "failed") and self.finished_at is None:
            raise ValueError(f"{self.status} manifests require finished_at")
        if self.run_type == "plan" and self.plan_horizon is None:
            raise ValueError("plan runs require plan_horizon")
        if self.squad_mode == "virtual" and self.run_type == "plan":
            pass  # prev_run_id may be null on the first week of a line
        return self


class BuildTableEntry(BaseModel):
    """One curated table pointer inside a build manifest."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sha: str = Field(min_length=1)
    rows: int = Field(ge=0)
    schema_hash: str = Field(min_length=1)


class BuildManifest(BaseModel):
    """Content-addressed curated build: table digests plus raw provenance.

    Written as ``curated/builds/build=<build_id>.json``. ``build_id`` is
    ``hash(raw manifest, staging_version, schema versions)``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    build_id: str = Field(min_length=1)
    created_at: datetime
    staging_version: int = Field(ge=1)
    code_sha: str = Field(min_length=1)
    raw_manifest_hash: str = Field(min_length=1)
    raw_cutoff: datetime
    tables: dict[str, BuildTableEntry] = Field(min_length=1)
    quarantined_captures: list[str] = Field(default_factory=list)

    @field_validator("tables")
    @classmethod
    def _table_names(
        cls, value: dict[str, BuildTableEntry]
    ) -> dict[str, BuildTableEntry]:
        for name in value:
            if not name.strip():
                raise ValueError("table names must be non-empty")
        return value


class OfficialPointer(BaseModel):
    """Write-once official line pointer for one season/GW/checkpoint/line.

    Path: ``official/season=<s>/gw=<gw>/checkpoint=<cp>/line=<line>.json``.
    Written only after the run's ``_SUCCESS``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    predict_run_id: str = Field(min_length=1)
    plan_run_id: str = Field(min_length=1)
    role: LineRole
    squad_mode: SquadMode
    written_at: datetime


class AppliedRecord(BaseModel):
    """Which official line was actually applied for a gameweek checkpoint.

    Path: ``official/season=<s>/gw=<gw>/checkpoint=<cp>/applied.json``.
    Separate from the per-line official pointers so the record keeps both the
    system's official output and what was acted on.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    line: str = Field(min_length=1)
    plan_run_id: str = Field(min_length=1)
    submitted_at: datetime


class InterventionRecord(BaseModel):
    """Manual veto, edit, or chip override beside an applied record.

    Path: ``official/…/intervention.json`` (or a timestamped sibling). Without
    this, the official record would claim the model made decisions it did not.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    who: str = Field(min_length=1)
    intervened_at: datetime
    what_changed: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    line: str | None = None
    plan_run_id: str | None = None
