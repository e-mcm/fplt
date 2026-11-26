"""Pydantic manifests for runs, curated builds, official pointers, experiments.

Stage B2 contracts plus Stage C experiment / batch manifests and write-once
events from ``docs/proposals/fplt-data-model.md`` §3.8.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

RunKind = Literal["official", "adhoc", "experiment", "benchmark", "legacy"]
RunMode = Literal["live", "backtest"]
RunType = Literal["predict", "plan", "eval"]
RunStatus = Literal["running", "succeeded", "failed"]
SquadMode = Literal["real", "virtual"]
LineRole = Literal["primary", "candidate"]
ExperimentFlavour = Literal["comparison", "ab"]
ArmRole = Literal["control", "treatment", "baseline"]
ExperimentEventKind = Literal[
    "submitted",
    "arm_started",
    "arm_succeeded",
    "arm_failed",
    "analysed",
    "aborted",
]
ExperimentStatusName = Literal["planned", "running", "analysed", "failed", "aborted"]


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


class ExperimentArm(BaseModel):
    """One arm declared on an experiment manifest at submit time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    arm: str = Field(min_length=1)
    role: ArmRole
    spec_hash: str = Field(min_length=1)
    spec_label: str = Field(min_length=1)
    env: dict[str, str] = Field(default_factory=dict)
    run_id: str | None = None

    @field_validator("env")
    @classmethod
    def _env_keys(cls, value: dict[str, str]) -> dict[str, str]:
        for key in value:
            if not key.strip():
                raise ValueError("env keys must be non-empty")
        return value


class ExperimentWindow(BaseModel):
    """One primary / confirm / holdout window pinned at experiment submit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    role: str = Field(min_length=1)
    seasons: list[str] = Field(min_length=1)
    from_gw: int = Field(ge=1, le=38)
    to_gw: int = Field(ge=1, le=38)

    @field_validator("seasons")
    @classmethod
    def _seasons_nonempty(cls, value: list[str]) -> list[str]:
        cleaned = [s.strip() for s in value]
        if any(not s for s in cleaned):
            raise ValueError("seasons entries must be non-empty")
        return cleaned

    @model_validator(mode="after")
    def _gw_order(self) -> ExperimentWindow:
        if self.from_gw > self.to_gw:
            raise ValueError("from_gw must be <= to_gw")
        return self


class ExperimentManifest(BaseModel):
    """Immutable experiment identity written at submit under ``experiments/``.

    Gates, metrics, windows and the EvalSpec hash are frozen here so later
    edits to the YAML cannot change a submitted experiment's verdict rule.
    Status is derived from write-once event files, not a mutable field.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    experiment_id: str = Field(min_length=1)
    flavour: ExperimentFlavour
    question: str = Field(min_length=1)
    batch_id: str | None = None
    eval_spec: str | None = None
    eval_spec_hash: str | None = None
    arms: list[ExperimentArm] = Field(min_length=1)
    build_id: str | None = None
    windows: list[ExperimentWindow] = Field(min_length=1)
    as_of: list[AsOfDeadline] = Field(default_factory=list[AsOfDeadline])
    horizon: int = Field(ge=1)
    seeds: dict[str, int] = Field(default_factory=dict)
    metrics: list[str] = Field(default_factory=list)
    gates: dict[str, Any] = Field(default_factory=dict)
    code_sha: str = Field(min_length=1)
    lockfile_hash: str = Field(min_length=1)
    submitted_by: str = Field(min_length=1)
    host: str = Field(min_length=1)
    submitted_at: datetime

    @field_validator("question")
    @classmethod
    def _question_strip(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must be non-blank")
        return cleaned

    @field_validator("seeds")
    @classmethod
    def _seed_keys(cls, value: dict[str, int]) -> dict[str, int]:
        for key in value:
            if not key.strip():
                raise ValueError("seed keys must be non-empty")
        return value

    @model_validator(mode="after")
    def _ab_requires_eval(self) -> ExperimentManifest:
        if self.flavour == "ab" and not (self.eval_spec and self.eval_spec_hash):
            raise ValueError("ab experiments require eval_spec and eval_spec_hash")
        names = [arm.arm for arm in self.arms]
        if len(names) != len(set(names)):
            raise ValueError("arm names must be unique")
        return self


class ExperimentEvent(BaseModel):
    """One write-once progress event under ``experiments/exp=…/events/``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    event: ExperimentEventKind
    at: datetime
    arm: str | None = None
    run_id: str | None = None
    detail: str | None = None

    @model_validator(mode="after")
    def _arm_events_need_arm(self) -> ExperimentEvent:
        if self.event.startswith("arm_") and not (self.arm and self.arm.strip()):
            raise ValueError(f"{self.event} events require arm")
        return self


class BatchManifest(BaseModel):
    """Immutable batch of experiments submitted together.

    Path: ``batches/batch=<batch_id>/manifest.json``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    batch_id: str = Field(min_length=1)
    label: str = Field(min_length=1)
    experiment_ids: list[str] = Field(min_length=1)
    submitted_at: datetime
    submitted_by: str = Field(min_length=1)
    host: str = Field(min_length=1)

    @field_validator("experiment_ids")
    @classmethod
    def _unique_experiments(cls, value: list[str]) -> list[str]:
        cleaned = [item.strip() for item in value]
        if any(not item for item in cleaned):
            raise ValueError("experiment_ids must be non-empty")
        if len(cleaned) != len(set(cleaned)):
            raise ValueError("experiment_ids must be unique")
        return cleaned


class ExperimentVerdict(BaseModel):
    """Computed verdict under ``experiments/exp=…/results/verdict.json``.

    Distinct from the human keep/drop decision in ``registry.yaml``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    overall: str = Field(min_length=1)
    gates: list[dict[str, Any]] = Field(default_factory=list[dict[str, Any]])
    computed_at: datetime
    eval_spec_hash: str | None = None
