"""Standard A/B promote-eval: control vs treatment ModelSpec.

Any lever — new staged data, feature version, hyperparam fork, layered
component registry swap, full predictor fork, lockfile upgrade, or no-behaviour
refactor — is expressed as two YAML specs plus an ``EvalSpec``. P1 validates
orientation, one lever, coverage, and leakage before fit. P2 runs
``compare_models`` on both arms and writes ``recommendation.md``. P2.5 produces
each arm via the content-addressed ledger cache. P7 adds ``kind: environment``
and ``kind: refactor`` (worktree + ``uv sync --frozen`` + subprocess) with the
§3.2 equivalence verdict. Improvement verdicts are ``drop`` (coverage / leak)
or ``inconclusive`` until FDB-73. Paired intervals (HAC + stationary bootstrap)
are descriptive for improvement levers; equivalence may ``keep`` on identical
predictions or ``drop`` when an interval lies entirely on the bad side.
"""

from __future__ import annotations

import hashlib
import json
import logging
import platform
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from importlib.metadata import version as package_version
from pathlib import Path
from typing import Any, Literal, cast

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fplt.backtest.compare import CompareResult, compare_models
from fplt.backtest.component_eval import ComponentEvalResult, run_component_eval
from fplt.backtest.decision_pool import (
    HEADLINE_METRIC,
    POOL_FILTER_DESCRIPTION,
    filter_decision_pool,
)
from fplt.backtest.equivalence import (
    EquivalenceDiff,
    EquivalenceSpec,
    add_worktree,
    compare_prediction_frames,
    diff_lockfiles,
    equivalence_verdict,
    interval_drop_for_frames,
    is_equivalence_kind,
    produce_ledger_in_worktree,
    remove_worktree,
    resolve_git_sha,
    sync_frozen,
)
from fplt.backtest.intervals import (
    equivalence_drop_from_intervals,
    interval_to_dict,
    intervals_for_pair,
    multi_arm_selection,
    per_gw_losses,
    per_gw_wins_from_losses,
)
from fplt.backtest.ledger import read_predictions
from fplt.backtest.produce_ledger import (
    StaleLedgerCacheError,
    repo_root,
    resolved_spec_sha256,
    uv_lock_sha256,
)
from fplt.data.store import DataStore
from fplt.models.spec import (
    ModelSpec,
    git_sha,
    load_model_specs,
    registered_components,
    registered_kinds,
)
from fplt.optimise.plan import DEFAULT_PLAN_HORIZON
from fplt.optimise.realised import DEFAULT_MAX_MEAN_PPG
from fplt.scoring.rules import Rules

logger = logging.getLogger(__name__)

EvalKind = Literal[
    "data",
    "feature",
    "config",
    "component",
    "model",
    "environment",
    "refactor",
]
Lever = Literal[
    "data_table",
    "feature_version",
    "hyperparam",
    "component_swap",
    "model_fork",
    "dependency_upgrade",
    "no_behaviour_change",
]
Tier = Literal["S0", "S1", "S2", "S3", "S4"]
CompareModeName = Literal["walk_forward", "fixed"]
Orientation = Literal["opt_in", "opt_out"]
CoverageMetric = Literal[
    "odds_join",
    "odds_history_join",
    "penalties_order_non_null",
    "dc_season_eligible",
]
CoverageRead = Literal["strict_pit", "closing_proxy_labelled", "prematch_scheduled"]
Verdict = Literal["keep", "tune", "drop", "inconclusive"]

_TIER_TO_GW: dict[Tier, int] = {
    "S0": 1,
    "S1": 10,
    "S2": 20,
    "S3": 10,
    "S4": 38,
}

# Absent key → code default (design §2 / §3).
_BOOL_DEFAULTS: dict[str, bool] = {
    "use_penalties_order": True,
    "use_defensive_contribution": True,
}
_FLOAT_DEFAULTS: dict[str, float] = {
    "market_prior_strength": 0.0,
    "market_cs_strength": 0.0,
}
_ORIENTATION_PARAMS = frozenset({*_BOOL_DEFAULTS, *_FLOAT_DEFAULTS})

_ODDS_COLS = ("home_odds", "draw_odds", "away_odds")
_RAW_DE_VIG_ARM = "raw_de_vig_1x2"
_SLOT_CLI_FLAGS = {
    "score": "--score-models",
    "minutes": "--minutes-model",
    "involvement": "--involvement-model",
}


class ChangeSpec(BaseModel):
    """What differs between control and treatment (one lever)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lever: Lever
    orientation: Orientation | None = None
    slot: str | None = None
    control_kind: str | None = None
    treatment_kind: str | None = None
    tables: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
    packages: list[str] = Field(default_factory=list)
    control_ref: str | None = None
    treatment_ref: str | None = None
    allow_multi_lever: bool = False
    diff: list[str] = Field(default_factory=list)
    notes: str = ""

    @model_validator(mode="after")
    def _slot_rules(self) -> ChangeSpec:
        if self.lever == "component_swap" and not self.slot:
            raise ValueError("component_swap requires change.slot")
        if self.lever == "model_fork" and self.slot is not None:
            raise ValueError("model_fork must leave change.slot null")
        if self.lever == "data_table" and not self.tables:
            raise ValueError("data_table requires change.tables")
        if self.lever == "feature_version" and not self.features:
            raise ValueError("feature_version requires change.features")
        if self.lever == "dependency_upgrade":
            if not self.packages:
                raise ValueError("dependency_upgrade requires change.packages")
            if not self.control_ref or not self.treatment_ref:
                raise ValueError(
                    "dependency_upgrade requires change.control_ref and "
                    "change.treatment_ref (lock_ref per arm)"
                )
            if self.orientation is not None:
                raise ValueError("dependency_upgrade must leave orientation null")
        if self.lever == "no_behaviour_change":
            if not self.control_ref or not self.treatment_ref:
                raise ValueError(
                    "no_behaviour_change requires change.control_ref and "
                    "change.treatment_ref"
                )
            if self.orientation is not None:
                raise ValueError("no_behaviour_change must leave orientation null")
            if self.packages:
                raise ValueError("no_behaviour_change must leave packages empty")
        return self


class WindowSpec(BaseModel):
    """Shared walk-forward window for both arms."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    seasons: list[str] = Field(default_factory=lambda: ["2024-25"])
    from_gw: int = Field(default=1, ge=1, le=38)
    to_gw: int = Field(default=20, ge=1, le=38)
    seasons_back: int = Field(default=4, ge=1)
    fit_gw: int | None = Field(default=None, ge=1, le=38)
    modes: list[CompareModeName] = Field(default_factory=lambda: ["walk_forward"])

    @model_validator(mode="after")
    def _gw_order(self) -> WindowSpec:
        if self.from_gw > self.to_gw:
            raise ValueError("window.from_gw must be <= window.to_gw")
        if not self.seasons:
            raise ValueError("window.seasons must be non-empty")
        if not self.modes:
            raise ValueError("window.modes must be non-empty")
        return self


class SuitesSpec(BaseModel):
    """Which metric suites to run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prediction: bool = True
    components: bool = False
    decisions: bool = False
    steps: int = Field(default=1, ge=1)
    decisions_horizon: int = Field(default=DEFAULT_PLAN_HORIZON, ge=1)
    max_mean_ppg: float = Field(default=DEFAULT_MAX_MEAN_PPG, ge=0)


class KeepIfSpec(BaseModel):
    """Optional metric-name hints for the recommendation phase."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prediction_any: list[str] = Field(default_factory=list)
    components_any: list[str] = Field(default_factory=list)


class DropIfSpec(BaseModel):
    """Hard gate flags."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    coverage_below_min: bool = True
    mean_ppg_gate_fail: bool = True


class CoverageSpec(BaseModel):
    """Named coverage formula (numerator is not free text)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    metric: CoverageMetric | None = None
    read: CoverageRead | None = None


class GatesSpec(BaseModel):
    """Coverage and keep/drop hints."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    allow_low_coverage: bool = False
    coverage: CoverageSpec = Field(default_factory=CoverageSpec)
    keep_if: KeepIfSpec = Field(default_factory=KeepIfSpec)
    drop_if: DropIfSpec = Field(default_factory=DropIfSpec)


class ReportSpec(BaseModel):
    """Human-facing report metadata."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    title: str = ""
    store_doc: str | None = None


class EvalSpec(BaseModel):
    """One control-vs-treatment promote evaluation."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str
    kind: EvalKind
    question: str
    description: str = ""
    change: ChangeSpec
    control: str
    treatment: str
    arms: list[str] = Field(
        default_factory=list,
        description=(
            "Optional extra ModelSpec names for a dose / multi-arm readout. "
            "Control and treatment stay the promote pair; arms share the "
            "compare window so SPA/MCS can run. Descriptive — not a keep."
        ),
    )
    window: WindowSpec = Field(default_factory=WindowSpec)
    suites: SuitesSpec = Field(default_factory=SuitesSpec)
    gates: GatesSpec = Field(default_factory=GatesSpec)
    report: ReportSpec = Field(default_factory=ReportSpec)
    equivalence: EquivalenceSpec = Field(default_factory=EquivalenceSpec)

    @field_validator("question")
    @classmethod
    def _question_nonblank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("question must be non-blank")
        return cleaned

    @model_validator(mode="after")
    def _kind_lever_align(self) -> EvalSpec:
        expected: dict[EvalKind, set[Lever]] = {
            "data": {"data_table"},
            "feature": {"feature_version"},
            "config": {"hyperparam"},
            "component": {"component_swap"},
            "model": {"model_fork"},
            "environment": {"dependency_upgrade"},
            "refactor": {"no_behaviour_change"},
        }
        allowed = expected[self.kind]
        if self.change.lever not in allowed:
            raise ValueError(
                f"kind {self.kind!r} expects change.lever in {sorted(allowed)}, "
                f"got {self.change.lever!r}"
            )
        if is_equivalence_kind(self.kind):
            if self.control != self.treatment:
                raise ValueError(
                    "equivalence kinds require control == treatment ModelSpec name"
                )
            if self.arms:
                raise ValueError("equivalence kinds must leave arms empty")
        elif self.control == self.treatment:
            raise ValueError("control and treatment must differ")
        for arm in self.arms:
            if arm in {self.control, self.treatment}:
                raise ValueError(
                    f"arms entry {arm!r} duplicates control or treatment; "
                    "omit it from arms"
                )
        if len(set(self.arms)) != len(self.arms):
            raise ValueError("arms must be unique")
        return self


def compare_arm_names(spec: EvalSpec) -> list[str]:
    """Return ModelSpec names passed to ``compare`` / ``decisions``.

    Control and treatment first, then any dose ``arms`` (feature-eval §9).
    """
    names = [spec.control, spec.treatment, *spec.arms]
    # Preserve order while dropping accidental duplicates.
    seen: set[str] = set()
    out: list[str] = []
    for name in names:
        if name not in seen:
            seen.add(name)
            out.append(name)
    return out


class PromoteEvalPlan(BaseModel):
    """Resolved dry-run plan for one EvalSpec."""

    model_config = ConfigDict(frozen=True)

    eval_name: str
    kind: EvalKind
    lever: Lever
    control: str
    treatment: str
    arms: list[str] = Field(default_factory=list)
    seasons: list[str]
    from_gw: int
    to_gw: int
    seasons_back: int
    modes: list[CompareModeName]
    suites: SuitesSpec
    commands: list[str]
    diff: list[str]
    notes: str
    warnings: list[str] = Field(default_factory=list)
    dry_run: bool = True


@dataclass
class CoverageResult:
    """One named coverage measurement."""

    metric: str
    numerator: int
    denominator: int
    rate: float
    min_coverage: float | None
    passed: bool
    extras: dict[str, Any] = field(default_factory=dict[str, Any])


@dataclass
class PreflightResult:
    """Outcome of promote-eval preflight checks."""

    ok: bool
    failures: list[str] = field(default_factory=list[str])
    warnings: list[str] = field(default_factory=list[str])
    coverage: CoverageResult | None = None
    leak: bool = False
    force_inconclusive: bool = False
    git_dirty: bool = False
    git_sha: str = "unknown"
    control_spec: ModelSpec | None = None
    treatment_spec: ModelSpec | None = None
    effective_flags: dict[str, Any] = field(default_factory=dict[str, Any])
    pit_caveat: str | None = None
    control_lock_sha256: str | None = None
    treatment_lock_sha256: str | None = None
    control_ref: str | None = None
    treatment_ref: str | None = None
    control_ref_sha: str | None = None
    treatment_ref_sha: str | None = None


@dataclass
class PromoteEvalResult:
    """Artifacts and verdict from an execute run."""

    eval_name: str
    verdict: Verdict
    exit_code: int
    output_dir: Path
    plan: PromoteEvalPlan
    preflight: PreflightResult
    recommendation: str
    compare: CompareResult | None = None
    summary: pl.DataFrame | None = None
    components: ComponentEvalResult | None = None
    per_match: pl.DataFrame | None = None
    equivalence_diff: EquivalenceDiff | None = None


def default_evals_dir() -> Path:
    """Return ``configs/evals`` at the repo root."""
    return Path(__file__).resolve().parents[3] / "configs" / "evals"


def load_eval_spec(name: str, directory: Path | None = None) -> EvalSpec:
    """Load one EvalSpec YAML by ``name``.

    Args:
        name: Stem of ``configs/evals/<name>.yaml`` (or a path stem).
        directory: Override evals folder (tests).

    Returns:
        Validated ``EvalSpec``.

    Raises:
        FileNotFoundError: No matching YAML.
        ValueError: YAML is empty, not a mapping, or fails validation.
    """
    folder = directory if directory is not None else default_evals_dir()
    path = folder / f"{name}.yaml"
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path.name} must contain a mapping")
    spec = EvalSpec.model_validate(raw)
    if spec.name != name:
        raise ValueError(
            f"EvalSpec name {spec.name!r} does not match file stem {name!r}"
        )
    return spec


def plan_promote_eval(
    spec: EvalSpec,
    *,
    dry_run: bool = True,
    tier: Tier | None = None,
    from_gw: int | None = None,
    to_gw: int | None = None,
    seasons: list[str] | None = None,
    skip_decisions: bool = False,
    models_dir: Path | None = None,
) -> PromoteEvalPlan:
    """Build the command plan for an EvalSpec (dry-run safe).

    Args:
        spec: Loaded evaluation.
        dry_run: When true, plan only.
        tier: Optional S0–S4 clamp on ``to_gw`` / decisions.
        from_gw: Override window start.
        to_gw: Override window end.
        seasons: Override season list.
        skip_decisions: Force decisions suite off.
        models_dir: Optional ``configs/models`` override for existence checks.

    Returns:
        Resolved ``PromoteEvalPlan`` with printable commands and warnings.
    """
    window = spec.window
    start = from_gw if from_gw is not None else window.from_gw
    end = to_gw if to_gw is not None else window.to_gw
    season_list = list(seasons) if seasons is not None else list(window.seasons)
    suites = spec.suites

    warnings: list[str] = []
    if tier is not None:
        end = min(end, _TIER_TO_GW[tier])
        if tier in {"S0", "S1", "S2"} and suites.decisions:
            warnings.append(f"tier {tier} disables decisions suite")
            suites = suites.model_copy(update={"decisions": False})
    if skip_decisions and suites.decisions:
        suites = suites.model_copy(update={"decisions": False})
        warnings.append("decisions skipped via --skip-decisions")
    if is_equivalence_kind(spec.kind):
        if suites.components:
            warnings.append(
                f"kind {spec.kind} forces components suite off (prediction ledger only)"
            )
            suites = suites.model_copy(update={"components": False})
        if suites.decisions:
            warnings.append(
                f"kind {spec.kind} forces decisions suite off (not a bakeoff)"
            )
            suites = suites.model_copy(update={"decisions": False})
        if not suites.prediction:
            warnings.append(
                f"kind {spec.kind} enables prediction suite for ledger equivalence"
            )
            suites = suites.model_copy(update={"prediction": True})
    if start > end:
        raise ValueError(f"from_gw {start} > to_gw {end}")

    model_names = set(load_model_specs(models_dir))
    for arm in compare_arm_names(spec):
        if arm not in model_names:
            warnings.append(f"ModelSpec {arm!r} not found under configs/models")
    if spec.arms:
        warnings.append(
            "several arms on one window: SPA/MCS are descriptive; "
            "keep needs a fresh confirmation window (§9)"
        )

    commands = _commands_for(
        spec,
        seasons=season_list,
        from_gw=start,
        to_gw=end,
        suites=suites,
        modes=list(window.modes),
        seasons_back=window.seasons_back,
        fit_gw=window.fit_gw,
        models_dir=models_dir,
        warnings=warnings,
    )

    return PromoteEvalPlan(
        eval_name=spec.name,
        kind=spec.kind,
        lever=spec.change.lever,
        control=spec.control,
        treatment=spec.treatment,
        arms=list(spec.arms),
        seasons=season_list,
        from_gw=start,
        to_gw=end,
        seasons_back=window.seasons_back,
        modes=list(window.modes),
        suites=suites,
        commands=commands,
        diff=list(spec.change.diff),
        notes=spec.change.notes,
        warnings=warnings,
        dry_run=dry_run,
    )


def format_plan(plan: PromoteEvalPlan) -> str:
    """Render a promote-eval plan for stdout."""
    lines = [
        f"promote-eval {plan.eval_name}  kind={plan.kind}  lever={plan.lever}",
        f"  control={plan.control}  treatment={plan.treatment}",
    ]
    if plan.arms:
        lines.append(f"  arms (dose): {' '.join(plan.arms)}")
    lines.extend(
        [
            f"  seasons={','.join(plan.seasons)}  gw={plan.from_gw}-{plan.to_gw}  "
            f"seasons_back={plan.seasons_back}  modes={','.join(plan.modes)}",
            (
                f"  suites: prediction={plan.suites.prediction} "
                f"components={plan.suites.components} "
                f"decisions={plan.suites.decisions}"
            ),
        ]
    )
    if plan.diff:
        lines.append("  diff:")
        lines.extend(f"    - {item}" for item in plan.diff)
    if plan.notes:
        lines.append(f"  notes: {plan.notes}")
    lines.append("  commands:")
    lines.extend(f"    {cmd}" for cmd in plan.commands)
    if plan.warnings:
        lines.append("  warnings:")
        lines.extend(f"    - {item}" for item in plan.warnings)
    if plan.dry_run:
        lines.append("  (dry-run: no fits executed)")
    return "\n".join(lines)


def plan_to_json(plan: PromoteEvalPlan) -> str:
    """Serialize a plan as sorted JSON."""
    payload: dict[str, Any] = plan.model_dump(mode="json")
    return json.dumps(payload, sort_keys=True, indent=2) + "\n"


def git_is_dirty(*, cwd: Path | None = None) -> bool:
    """Return whether the working tree has uncommitted changes."""
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd or repo_root(),
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return bool(completed.stdout.strip())


def effective_param(spec: ModelSpec, slot: str | None, key: str) -> Any:
    """Return a component (or top-level) param with code defaults applied.

    Args:
        spec: Loaded ModelSpec.
        slot: Component slot, or ``None`` for top-level ``params``.
        key: Parameter name.

    Returns:
        The effective value (YAML value or code default).
    """
    params = (
        dict(spec.params)
        if slot is None
        else dict(spec.components[slot].params)
        if slot in spec.components
        else {}
    )
    if key in params:
        return params[key]
    if key in _BOOL_DEFAULTS:
        return _BOOL_DEFAULTS[key]
    if key in _FLOAT_DEFAULTS:
        return _FLOAT_DEFAULTS[key]
    return None


def is_effectively_on(key: str, value: Any) -> bool:
    """Whether an orientation param is in the on state."""
    if key in _BOOL_DEFAULTS:
        return bool(value)
    if key in _FLOAT_DEFAULTS:
        return float(value) > 0.0
    return bool(value)


def preflight_promote_eval(
    spec: EvalSpec,
    plan: PromoteEvalPlan,
    *,
    data_dir: Path,
    allow_dirty: bool = False,
    models_dir: Path | None = None,
    argv: list[str] | None = None,
) -> PreflightResult:
    """Run P1 preflight checks before any fit.

    Args:
        spec: Loaded EvalSpec.
        plan: Resolved window / suites plan.
        data_dir: Staged data root (``$FPLT_DATA_DIR``).
        allow_dirty: Continue on a dirty tree; forces inconclusive later.
        models_dir: Optional models override.
        argv: Original CLI argv for audit (unused here; kept for callers).

    Returns:
        ``PreflightResult`` with failures, warnings, and optional coverage.
    """
    del argv  # recorded by execute, not preflight
    _ensure_model_registry()
    failures: list[str] = []
    warnings: list[str] = list(plan.warnings)
    dirty = git_is_dirty()
    sha = git_sha()
    force_inconclusive = False
    if dirty and not allow_dirty:
        failures.append("dirty working tree (pass --allow-dirty to continue)")
    elif dirty and allow_dirty:
        force_inconclusive = True
        warnings.append("--allow-dirty set; verdict forced to inconclusive")

    specs = load_model_specs(models_dir)
    control = specs.get(spec.control)
    treatment = specs.get(spec.treatment)
    if control is None:
        failures.append(f"control ModelSpec {spec.control!r} missing")
    if treatment is None:
        failures.append(f"treatment ModelSpec {spec.treatment!r} missing")
    if control is None or treatment is None:
        return PreflightResult(
            ok=False,
            failures=failures,
            warnings=warnings,
            git_dirty=dirty,
            git_sha=sha,
            force_inconclusive=force_inconclusive,
        )

    failures.extend(_check_registered(control, treatment))
    failures.extend(_check_kinds(spec, control, treatment))
    failures.extend(_check_one_lever(spec, control, treatment))
    orient_fails, effective = _check_orientation(spec, control, treatment)
    failures.extend(orient_fails)

    control_lock_sha: str | None = None
    treatment_lock_sha: str | None = None
    control_ref = spec.change.control_ref
    treatment_ref = spec.change.treatment_ref
    control_ref_sha: str | None = None
    treatment_ref_sha: str | None = None
    if is_equivalence_kind(spec.kind):
        eq_fails, eq_warns, lock_meta = _check_equivalence_preflight(
            spec, control, treatment
        )
        failures.extend(eq_fails)
        warnings.extend(eq_warns)
        control_lock_sha = lock_meta.get("control_lock_sha256")
        treatment_lock_sha = lock_meta.get("treatment_lock_sha256")
        control_ref_sha = lock_meta.get("control_ref_sha")
        treatment_ref_sha = lock_meta.get("treatment_ref_sha")

    store = DataStore(data_dir)
    failures.extend(_check_tables(spec, store))

    coverage: CoverageResult | None = None
    pit_caveat: str | None = None
    leak = False
    if not failures and not is_equivalence_kind(spec.kind):
        try:
            coverage, extra_warnings, pit_caveat = _measure_coverage(
                spec, plan, store, treatment
            )
            warnings.extend(extra_warnings)
        except ValueError as exc:
            failures.append(str(exc))
        else:
            if (
                coverage is not None
                and not coverage.passed
                and not spec.gates.allow_low_coverage
                and spec.gates.drop_if.coverage_below_min
            ):
                # Coverage miss is handled as drop by execute (exit 1), not a
                # structural preflight failure — still mark for the runner.
                pass
            elif (
                coverage is not None
                and not coverage.passed
                and spec.gates.allow_low_coverage
            ):
                force_inconclusive = True
                warnings.append(
                    f"coverage {coverage.rate:.3f} < min_coverage "
                    f"{coverage.min_coverage}; allow_low_coverage → inconclusive"
                )
            leak_msg = _check_leak(spec, plan, data_dir, store)
            if leak_msg is not None:
                leak = True
                failures.append(leak_msg)

    return PreflightResult(
        ok=not failures and not leak,
        failures=failures,
        warnings=warnings,
        coverage=coverage,
        leak=leak,
        force_inconclusive=force_inconclusive,
        git_dirty=dirty,
        git_sha=sha,
        control_spec=control,
        treatment_spec=treatment,
        effective_flags=effective,
        pit_caveat=pit_caveat,
        control_lock_sha256=control_lock_sha,
        treatment_lock_sha256=treatment_lock_sha,
        control_ref=control_ref,
        treatment_ref=treatment_ref,
        control_ref_sha=control_ref_sha,
        treatment_ref_sha=treatment_ref_sha,
    )


def execute_promote_eval(
    spec: EvalSpec,
    *,
    data_dir: Path,
    dry_run: bool = False,
    tier: Tier | None = None,
    from_gw: int | None = None,
    to_gw: int | None = None,
    seasons: list[str] | None = None,
    skip_decisions: bool = False,
    allow_dirty: bool = False,
    models_dir: Path | None = None,
    output_dir: Path | None = None,
    argv: list[str] | None = None,
) -> PromoteEvalResult:
    """Plan, preflight, optionally compare, and write promote-eval artifacts.

    Args:
        spec: Loaded EvalSpec.
        data_dir: Staged data root.
        dry_run: When true, plan only (no preflight store reads beyond plan).
        tier: Optional S0–S4 clamp.
        from_gw: Override window start.
        to_gw: Override window end.
        seasons: Override seasons.
        skip_decisions: Force decisions off.
        allow_dirty: Continue on dirty tree; forces ``inconclusive``.
        models_dir: Optional models override.
        output_dir: Artifact directory override.
        argv: CLI argv recorded in ``audit.json``.

    Returns:
        ``PromoteEvalResult`` including exit code for the CLI.
    """
    plan = plan_promote_eval(
        spec,
        dry_run=dry_run,
        tier=tier,
        from_gw=from_gw,
        to_gw=to_gw,
        seasons=seasons,
        skip_decisions=skip_decisions,
        models_dir=models_dir,
    )
    if dry_run:
        out = output_dir or _default_output_dir(data_dir, spec.name)
        out.mkdir(parents=True, exist_ok=True)
        (out / "plan.json").write_text(plan_to_json(plan), encoding="utf-8")
        text = format_plan(plan)
        return PromoteEvalResult(
            eval_name=spec.name,
            verdict="inconclusive",
            exit_code=0,
            output_dir=out,
            plan=plan,
            preflight=PreflightResult(ok=True),
            recommendation=text,
        )

    out = output_dir or _default_output_dir(data_dir, spec.name)
    out.mkdir(parents=True, exist_ok=True)
    (out / "plan.json").write_text(plan_to_json(plan), encoding="utf-8")

    preflight = preflight_promote_eval(
        spec,
        plan,
        data_dir=data_dir,
        allow_dirty=allow_dirty,
        models_dir=models_dir,
        argv=argv,
    )
    coverage_drop = _coverage_is_drop(spec, preflight)
    if preflight.failures and not (coverage_drop and not preflight.leak):
        # Structural / leak failures: exit 2. Leak still writes a drop report.
        if preflight.leak:
            verdict: Verdict = "drop"
            recommendation = render_recommendation(
                spec,
                plan,
                preflight,
                verdict=verdict,
                allow_dirty=allow_dirty,
                compare=None,
                deltas=None,
            )
            _write_recommendation(out, recommendation)
            _write_audit(
                out,
                spec,
                plan,
                preflight,
                data_dir=data_dir,
                argv=argv,
                wall_clock={},
                peak_rss={},
                folds=[],
                verdict=verdict,
            )
            return PromoteEvalResult(
                eval_name=spec.name,
                verdict=verdict,
                exit_code=2,
                output_dir=out,
                plan=plan,
                preflight=preflight,
                recommendation=recommendation,
            )
        # Pure structural fail: no recommendation required.
        msg = "; ".join(preflight.failures)
        recommendation = f"preflight failed: {msg}\n"
        _write_audit(
            out,
            spec,
            plan,
            preflight,
            data_dir=data_dir,
            argv=argv,
            wall_clock={},
            peak_rss={},
            folds=[],
            verdict="drop",
        )
        return PromoteEvalResult(
            eval_name=spec.name,
            verdict="drop",
            exit_code=2,
            output_dir=out,
            plan=plan,
            preflight=preflight,
            recommendation=recommendation,
        )

    if coverage_drop:
        verdict = "drop"
        recommendation = render_recommendation(
            spec,
            plan,
            preflight,
            verdict=verdict,
            allow_dirty=allow_dirty,
            compare=None,
            deltas=None,
        )
        _write_recommendation(out, recommendation)
        _write_audit(
            out,
            spec,
            plan,
            preflight,
            data_dir=data_dir,
            argv=argv,
            wall_clock={},
            peak_rss={},
            folds=[],
            verdict=verdict,
        )
        _write_summary_json(out, spec, plan, preflight, verdict=verdict, deltas=None)
        return PromoteEvalResult(
            eval_name=spec.name,
            verdict=verdict,
            exit_code=1,
            output_dir=out,
            plan=plan,
            preflight=preflight,
            recommendation=recommendation,
        )

    if is_equivalence_kind(spec.kind) and tier != "S0":
        return _execute_equivalence(
            spec,
            plan,
            preflight,
            data_dir=data_dir,
            output_dir=out,
            models_dir=models_dir,
            allow_dirty=allow_dirty,
            argv=argv,
        )

    compare_result: CompareResult | None = None
    deltas: dict[str, Any] | None = None
    wall_clock: dict[str, float] = {}
    peak_rss: dict[str, int] = {}
    folds: list[dict[str, Any]] = []
    components_result: ComponentEvalResult | None = None
    per_match: pl.DataFrame | None = None
    component_deltas: dict[str, Any] | None = None
    run_prediction = plan.suites.prediction and tier != "S0"
    run_components = plan.suites.components and tier != "S0"
    if run_components:
        components_result, per_match, component_deltas = _run_components(
            spec,
            plan,
            preflight,
            data_dir=data_dir,
            output_dir=out / "components",
            models_dir=models_dir,
        )
        if per_match is not None and not per_match.is_empty():
            per_match.write_parquet(out / "per_match.parquet")
    if run_prediction:
        try:
            compare_result, wall_clock, peak_rss = _run_compare(
                plan, data_dir=data_dir, output_dir=out / "compare"
            )
        except StaleLedgerCacheError as exc:
            # Cached ledger whose stored key ≠ the key just computed (§8).
            msg = f"stale ledger cache: {exc}"
            stale_preflight = PreflightResult(
                ok=False,
                failures=[msg],
                warnings=list(preflight.warnings),
                coverage=preflight.coverage,
                leak=preflight.leak,
                force_inconclusive=preflight.force_inconclusive,
                git_dirty=preflight.git_dirty,
                git_sha=preflight.git_sha,
                control_spec=preflight.control_spec,
                treatment_spec=preflight.treatment_spec,
                effective_flags=dict(preflight.effective_flags),
                pit_caveat=preflight.pit_caveat,
            )
            recommendation = f"preflight failed: {msg}\n"
            _write_audit(
                out,
                spec,
                plan,
                stale_preflight,
                data_dir=data_dir,
                argv=argv,
                wall_clock={},
                peak_rss={},
                folds=[],
                verdict="drop",
            )
            return PromoteEvalResult(
                eval_name=spec.name,
                verdict="drop",
                exit_code=2,
                output_dir=out,
                plan=plan,
                preflight=stale_preflight,
                recommendation=recommendation,
            )
        deltas = _prediction_deltas(
            compare_result,
            plan.control,
            plan.treatment,
            data_dir=data_dir,
            position_specific=_lever_is_position_specific(spec),
        )
        folds = _folds_from_compare(compare_result, data_dir)
        summary = compare_result.summary
        compare_result.summary.write_parquet(out / "summary.parquet")
        (out / "summary.json").write_text(
            json.dumps(
                {
                    "eval_name": spec.name,
                    "verdict": "inconclusive",
                    "summary": summary.to_dicts(),
                    "deltas": deltas,
                    "component_deltas": component_deltas,
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        summary = None
        _write_summary_json(
            out,
            spec,
            plan,
            preflight,
            verdict="inconclusive",
            deltas=None,
            component_deltas=component_deltas,
        )

    verdict = "inconclusive"
    if preflight.force_inconclusive:
        verdict = "inconclusive"
    recommendation = render_recommendation(
        spec,
        plan,
        preflight,
        verdict=verdict,
        allow_dirty=allow_dirty,
        compare=compare_result,
        deltas=deltas,
        components_ran=run_components,
        component_deltas=component_deltas,
        decisions_ran=False,
        prediction_ran=run_prediction,
        per_match=per_match,
    )
    _write_recommendation(out, recommendation)
    _write_audit(
        out,
        spec,
        plan,
        preflight,
        data_dir=data_dir,
        argv=argv,
        wall_clock=wall_clock,
        peak_rss=peak_rss,
        folds=folds,
        verdict=verdict,
    )
    _write_spec_diff(out, preflight)
    return PromoteEvalResult(
        eval_name=spec.name,
        verdict=verdict,
        exit_code=0,
        output_dir=out,
        plan=plan,
        preflight=preflight,
        recommendation=recommendation,
        compare=compare_result,
        summary=summary,
        components=components_result,
        per_match=per_match,
    )


def _execute_equivalence(
    spec: EvalSpec,
    plan: PromoteEvalPlan,
    preflight: PreflightResult,
    *,
    data_dir: Path,
    output_dir: Path,
    models_dir: Path | None,
    allow_dirty: bool,
    argv: list[str] | None,
) -> PromoteEvalResult:
    """Produce both arms in worktrees and apply the §3.2 verdict."""
    root = repo_root()
    control_ref = spec.change.control_ref or "HEAD"
    treatment_ref = spec.change.treatment_ref or "HEAD"
    work_root = output_dir / "worktrees"
    control_wt = work_root / "control"
    treatment_wt = work_root / "treatment"
    wall_clock: dict[str, float] = {}
    peak_rss: dict[str, int] = {}
    eq_diff: EquivalenceDiff | None = None
    deltas: dict[str, Any] | None = None

    try:
        add_worktree(root, control_ref, control_wt)
        add_worktree(root, treatment_ref, treatment_wt)
        sync_frozen(control_wt)
        sync_frozen(treatment_wt)

        mode = plan.modes[0] if plan.modes else "walk_forward"
        # Environment can reuse the ledger cache (lock hash differs). Refactor
        # disables cache so identical lock+spec keys cannot collide across refs.
        use_cache = spec.kind == "environment"
        control_frames: list[pl.DataFrame] = []
        treatment_frames: list[pl.DataFrame] = []
        for season in plan.seasons:
            arms = (("control", control_wt), ("treatment", treatment_wt))
            for role, worktree in arms:
                out_parquet = output_dir / "ledgers" / role / f"{season}_{mode}.parquet"
                started = time.perf_counter()
                before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                produce_ledger_in_worktree(
                    worktree,
                    data_dir=data_dir,
                    model_name=plan.control,
                    season=season,
                    mode=mode,
                    test_from_gw=plan.from_gw,
                    test_to_gw=plan.to_gw,
                    fit_gw=plan.from_gw if mode == "walk_forward" else None,
                    steps=plan.suites.steps,
                    seasons_back=plan.seasons_back,
                    out_parquet=out_parquet,
                    use_cache=use_cache,
                    models_dir=models_dir,
                )
                elapsed = time.perf_counter() - started
                after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
                wall_clock[f"{role}:{season}"] = elapsed
                peak_rss[f"{role}:{season}"] = max(after - before, 0)
                frame = pl.read_parquet(out_parquet)
                if role == "control":
                    control_frames.append(frame)
                else:
                    treatment_frames.append(frame)

        control_frame = pl.concat(control_frames, how="vertical_relaxed")
        treatment_frame = pl.concat(treatment_frames, how="vertical_relaxed")
        eq_diff = compare_prediction_frames(
            control_frame, treatment_frame, spec.equivalence.tolerance
        )
        actuals = _actuals_for_equivalence(data_dir, plan)
        interval_drop, interval_dicts = interval_drop_for_frames(
            control_frame,
            treatment_frame,
            control_id=f"{plan.control}::control",
            treatment_id=f"{plan.control}::treatment",
            actuals=actuals,
        )
        deltas = {
            "equivalence_interval_drop": interval_drop,
            "intervals": interval_dicts,
            "n_rows_compared": eq_diff.n_rows,
            "n_beyond_tolerance": eq_diff.n_beyond_tolerance,
            "max_abs_diff": eq_diff.max_abs_diff,
            "top_offenders": eq_diff.top_offenders,
        }
        verdict = equivalence_verdict(
            eq_diff,
            interval_drop=interval_drop,
            force_inconclusive=preflight.force_inconclusive,
        )
    except (OSError, RuntimeError, subprocess.CalledProcessError, ValueError) as exc:
        msg = f"equivalence produce failed: {exc}"
        logger.exception(msg)
        failed = PreflightResult(
            ok=False,
            failures=[msg],
            warnings=list(preflight.warnings),
            git_dirty=preflight.git_dirty,
            git_sha=preflight.git_sha,
            control_spec=preflight.control_spec,
            treatment_spec=preflight.treatment_spec,
            effective_flags=dict(preflight.effective_flags),
            control_lock_sha256=preflight.control_lock_sha256,
            treatment_lock_sha256=preflight.treatment_lock_sha256,
            control_ref=preflight.control_ref,
            treatment_ref=preflight.treatment_ref,
            control_ref_sha=preflight.control_ref_sha,
            treatment_ref_sha=preflight.treatment_ref_sha,
        )
        recommendation = f"preflight failed: {msg}\n"
        _write_audit(
            output_dir,
            spec,
            plan,
            failed,
            data_dir=data_dir,
            argv=argv,
            wall_clock=wall_clock,
            peak_rss=peak_rss,
            folds=[],
            verdict="drop",
        )
        return PromoteEvalResult(
            eval_name=spec.name,
            verdict="drop",
            exit_code=2,
            output_dir=output_dir,
            plan=plan,
            preflight=failed,
            recommendation=recommendation,
        )
    finally:
        remove_worktree(root, control_wt)
        remove_worktree(root, treatment_wt)

    assert eq_diff is not None
    recommendation = render_recommendation(
        spec,
        plan,
        preflight,
        verdict=verdict,
        allow_dirty=allow_dirty,
        compare=None,
        deltas=deltas,
        components_ran=False,
        decisions_ran=False,
        prediction_ran=True,
        equivalence_diff=eq_diff,
    )
    _write_recommendation(output_dir, recommendation)
    _write_audit(
        output_dir,
        spec,
        plan,
        preflight,
        data_dir=data_dir,
        argv=argv,
        wall_clock=wall_clock,
        peak_rss=peak_rss,
        folds=[],
        verdict=verdict,
    )
    _write_spec_diff(output_dir, preflight)
    _write_summary_json(
        output_dir,
        spec,
        plan,
        preflight,
        verdict=verdict,
        deltas=deltas,
    )
    (output_dir / "summary.json").write_text(
        json.dumps(
            {
                "eval_name": spec.name,
                "verdict": verdict,
                "question_type": "equivalence",
                "deltas": deltas,
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    exit_code = 0
    return PromoteEvalResult(
        eval_name=spec.name,
        verdict=verdict,
        exit_code=exit_code,
        output_dir=output_dir,
        plan=plan,
        preflight=preflight,
        recommendation=recommendation,
        equivalence_diff=eq_diff,
    )


def _actuals_for_equivalence(
    data_dir: Path, plan: PromoteEvalPlan
) -> pl.DataFrame | None:
    """Build actuals rows for interval scoring when staged stats exist."""
    store = DataStore(data_dir)
    try:
        stats = store.read("player_fixture_stats")
        gws = store.read("gameweeks")
    except Exception:
        return None
    if stats.is_empty() or gws.is_empty():
        return None
    checked = gws.filter(
        pl.col("season").is_in(plan.seasons)
        & pl.col("gw").is_between(plan.from_gw, plan.to_gw, closed="both")
        & (pl.col("data_checked") == True)  # noqa: E712
    ).select(["season", "gw"])
    if checked.is_empty():
        return None
    joined = stats.join(
        checked,
        left_on=["season", "gw"],
        right_on=["season", "gw"],
        how="inner",
    )
    if "position" in joined.columns:
        position_expr: pl.Expr = pl.col("position")
    else:
        position_expr = pl.lit("MID").alias("position")
    frame = joined.select(
        pl.col("season").alias("target_season"),
        pl.col("gw").alias("target_gw"),
        "player_key",
        "fixture_id",
        pl.col("total_points").alias("actual_pts"),
        position_expr,
    ).unique(subset=["target_season", "target_gw", "player_key", "fixture_id"])
    return frame


def render_recommendation(
    spec: EvalSpec,
    plan: PromoteEvalPlan,
    preflight: PreflightResult,
    *,
    verdict: Verdict,
    allow_dirty: bool,
    compare: CompareResult | None,
    deltas: dict[str, Any] | None,
    components_ran: bool = False,
    component_deltas: dict[str, Any] | None = None,
    decisions_ran: bool = False,
    prediction_ran: bool = False,
    per_match: pl.DataFrame | None = None,
    equivalence_diff: EquivalenceDiff | None = None,
) -> str:
    """Build ``recommendation.md`` in the §9 section order.

    Args:
        spec: EvalSpec under test.
        plan: Resolved plan.
        preflight: Preflight outcome (coverage, flags, caveat).
        verdict: Final call.
        allow_dirty: Whether ``--allow-dirty`` was set.
        compare: Optional compare result.
        deltas: Optional prediction deltas.
        components_ran: Whether the components suite ran.
        component_deltas: Optional pooled component metric deltas.
        decisions_ran: Whether the decisions suite ran.
        prediction_ran: Whether prediction compare ran.
        per_match: Optional scoreline fixture × arm × metric board.

    Returns:
        Markdown recommendation text.
    """
    title = spec.report.title or spec.name
    orientation = spec.change.orientation or "n/a"
    question_type = "equivalence" if is_equivalence_kind(spec.kind) else "improvement"
    dirty_note = " [--allow-dirty]" if allow_dirty and preflight.git_dirty else ""
    lines: list[str] = [
        f"# {title}{dirty_note}",
        "",
        f"question: {spec.question}",
        f"question_type: {question_type}",
        f"kind: {spec.kind}",
    ]
    if not is_equivalence_kind(spec.kind):
        lines.append(f"orientation: {orientation}")
    lines.extend(
        [
            f"lever: {spec.change.lever}",
            "diff:",
            *[f"  - {item}" for item in spec.change.diff],
            "",
        ]
    )
    # 2. Arms
    lines.append("## Arms")
    for role, name, arm in (
        ("control", plan.control, preflight.control_spec),
        ("treatment", plan.treatment, preflight.treatment_spec),
    ):
        production = _is_production_arm(spec, role)
        flag_bits = ", ".join(
            f"{key}={preflight.effective_flags.get(f'{role}.{key}', 'n/a')}"
            for key in sorted(
                k.split(".", 1)[1]
                for k in preflight.effective_flags
                if k.startswith(f"{role}.")
            )
        )
        lock_sha = (
            preflight.control_lock_sha256
            if role == "control"
            else preflight.treatment_lock_sha256
        )
        ref = preflight.control_ref if role == "control" else preflight.treatment_ref
        lines.append(
            f"- {role}: {name}"
            f"{' (production)' if production else ''}"
            f"{f' [{flag_bits}]' if flag_bits else ''}"
            f"{f' model_id={arm.model_id}' if arm is not None else ''}"
            f"{f' ref={ref}' if ref else ''}"
            f"{f' uv_lock_sha256={lock_sha}' if lock_sha else ''}"
        )
    lines.append("")
    # 3. Audit summary
    audit_lock = (
        f"control={preflight.control_lock_sha256} "
        f"treatment={preflight.treatment_lock_sha256}"
        if is_equivalence_kind(spec.kind)
        else uv_lock_sha256()
    )
    lines.extend(
        [
            "## Audit",
            f"- git_sha: {preflight.git_sha}",
            f"- git_dirty: {preflight.git_dirty}",
            f"- uv_lock_sha256: {audit_lock}",
            "- full record: audit.json",
            "",
        ]
    )
    # 4. Window
    n_sims = _n_sims(preflight.treatment_spec or preflight.control_spec)
    lines.extend(
        [
            "## Window",
            f"- seasons: {', '.join(plan.seasons)}",
            f"- gw: {plan.from_gw}-{plan.to_gw}",
            f"- seasons_back: {plan.seasons_back}",
            f"- modes: {', '.join(plan.modes)}",
            f"- n_sims: {n_sims}",
            "",
        ]
    )
    # 5. Coverage
    lines.append("## Coverage")
    cov = preflight.coverage
    if cov is None:
        lines.append("- (no coverage metric configured)")
    else:
        status = "pass" if cov.passed else "fail"
        lines.append(
            f"- metric={cov.metric} numerator={cov.numerator} "
            f"denominator={cov.denominator} rate={cov.rate:.4f} "
            f"min_coverage={cov.min_coverage} → {status}"
        )
        for key, value in sorted(cov.extras.items()):
            lines.append(f"- {key}: {value}")
    lines.append("")
    # 6. Join / available_at rule
    lines.append("## Join / available_at")
    read = spec.gates.coverage.read or "strict_pit"
    lines.append(f"- coverage.read: {read}")
    lines.append("- rule: available_at <= gameweek deadline (PITView)")
    mps = preflight.effective_flags.get("treatment.market_prior_source")
    if mps is not None:
        lines.append(f"- market_prior_source: {mps}")
    flag_keys = {k.split(".", 1)[-1] for k in preflight.effective_flags}
    if "market_prior_strength" in flag_keys:
        lines.append(
            "- market_prior_strength (treatment): "
            f"{preflight.effective_flags.get('treatment.market_prior_strength')}"
        )
    if "market_cs_strength" in flag_keys:
        lines.append(
            "- market_cs_strength (treatment): "
            f"{preflight.effective_flags.get('treatment.market_cs_strength')}"
        )
    lines.append("")
    # 7. PIT caveat
    lines.append("## PIT caveat")
    if preflight.pit_caveat:
        lines.append(preflight.pit_caveat)
    else:
        lines.append("- (none)")
    lines.append("")
    # 8. Deltas
    lines.append("## Prediction deltas")
    if not prediction_ran or deltas is None:
        lines.append("- prediction: not run")
    else:
        # RMSE is the mean-seeking headline; MAE stays for continuity with old A/Bs.
        for key in ("rmse", "mae", "spearman", "top10_precision"):
            if key in deltas:
                label = "headline" if key == "rmse" else "secondary"
                lines.append(
                    f"- {key}_delta ({label}; treatment better > 0): {deltas[key]}"
                )
        if "per_gw_wins" in deltas:
            wins = deltas["per_gw_wins"]
            lines.append(
                f"- per_gw_wins (RMSE): control={wins.get('control', 0)} "
                f"treatment={wins.get('treatment', 0)} "
                f"ties={wins.get('ties', 0)}"
            )
        if "per_gw_wins_mae" in deltas:
            wins_mae = deltas["per_gw_wins_mae"]
            lines.append(
                "- per_gw_wins (MAE, continuity): "
                f"control={wins_mae.get('control', 0)} "
                f"treatment={wins_mae.get('treatment', 0)} "
                f"ties={wins_mae.get('ties', 0)}"
            )
        if "mean_xpts" in deltas:
            lines.append(f"- mean_xpts: {deltas['mean_xpts']}")
        lines.extend(_format_interval_lines(deltas))
        if "spa" in deltas and deltas["spa"] is not None:
            spa = deltas["spa"]
            lines.append(
                "- SPA (consistent p-value; any arm beat control?): "
                f"{spa.get('pvalue_consistent')}"
            )
        if "mcs" in deltas and deltas["mcs"] is not None:
            mcs = deltas["mcs"]
            lines.append(
                f"- MCS included (size={mcs.get('size')}): {mcs.get('included')}"
            )
            lines.append(f"- MCS excluded: {mcs.get('excluded')}")
        lines.append(
            "- intervals are descriptive for improvement levers "
            "(not a keep rule; equivalence may drop on entirely-bad intervals)"
        )
        lines.append(f"- player pool: {POOL_FILTER_DESCRIPTION}")
        lines.append(
            "- W9-style MAE/RMSE split: RMSE is the mean-seeking headline "
            "(optimiser consumes expected points); MAE alone can prefer "
            "median-seeking forecasts that underprice haul upside — read both, "
            "prefer RMSE when they disagree on a mean-seeking board"
        )
        if (
            is_scoreline_lever(spec)
            and per_match is not None
            and not per_match.is_empty()
        ):
            market = _pooled_arm_metric(per_match, _RAW_DE_VIG_ARM, "rps")
            lines.append(
                f"- raw de-vig 1X2 ({_RAW_DE_VIG_ARM}) mean rps: "
                f"{market if market is not None else 'n/a'}"
            )
            lines.append(
                "- odds_poisson_v1 is not the market row "
                "(book = raw proportional de-vig 1X2)"
            )
    lines.append("")
    # 9. Components
    lines.append("## Components")
    if components_ran and component_deltas:
        for key, value in sorted(component_deltas.items()):
            lines.append(f"- {key}: {value}")
        if is_scoreline_lever(spec):
            lines.append(
                f"- market arm: {_RAW_DE_VIG_ARM} "
                "(same clock as coverage.read; not odds_poisson_v1)"
            )
            if per_match is not None and not per_match.is_empty():
                lines.append(
                    f"- per_match rows: {per_match.height} "
                    "(fixture × arm × metric → per_match.parquet)"
                )
    elif components_ran:
        lines.append("- components suite ran (no pooled deltas)")
    else:
        lines.append("- components: not run")
    lines.append("")
    # 10. Decisions
    lines.append("## Decisions")
    if decisions_ran:
        lines.append("- decisions suite ran")
    else:
        lines.append("- decisions: not run")
    lines.append("")
    # 11. Equivalence
    lines.append("## Equivalence")
    if not is_equivalence_kind(spec.kind):
        lines.append("- n/a (improvement lever)")
    elif equivalence_diff is None:
        lines.append("- equivalence suite did not run")
    else:
        lines.append(f"- predictions compared: {equivalence_diff.n_rows}")
        lines.append(f"- beyond tolerance: {equivalence_diff.n_beyond_tolerance}")
        lines.append(f"- max_abs_diff: {equivalence_diff.max_abs_diff}")
        if equivalence_diff.top_offenders:
            lines.append("- top offenders:")
            for row in equivalence_diff.top_offenders[:10]:
                lines.append(f"  - {row}")
        else:
            lines.append("- top offenders: (none)")
    lines.append("")
    # 12. Call
    lines.append("## Call")
    lines.append(f"**{verdict}**")
    lines.append("")
    lines.append(
        _verdict_paragraph(
            spec,
            verdict,
            preflight,
            allow_dirty=allow_dirty,
            equivalence_diff=equivalence_diff,
            deltas=deltas,
        )
    )
    if compare is not None:
        lines.append("")
        lines.append(f"(compare output: {compare.output_dir})")
    return "\n".join(lines) + "\n"


def _verdict_paragraph(
    spec: EvalSpec,
    verdict: Verdict,
    preflight: PreflightResult,
    *,
    allow_dirty: bool,
    equivalence_diff: EquivalenceDiff | None = None,
    deltas: dict[str, Any] | None = None,
) -> str:
    if verdict == "drop" and preflight.leak:
        return (
            "Leak: a predicted-fixture feature has available_at after the "
            "deadline. The lever is not deadline-safe."
        )
    if verdict == "drop" and preflight.coverage is not None:
        cov = preflight.coverage
        return (
            f"Coverage {cov.rate:.3f} is below min_coverage={cov.min_coverage} "
            f"for metric={cov.metric}. Drop until the lever actually runs on "
            "the model's read."
        )
    if allow_dirty and preflight.git_dirty:
        return (
            "Working tree was dirty with --allow-dirty; recommendation is "
            "inconclusive regardless of metric deltas."
        )
    if preflight.force_inconclusive and spec.gates.allow_low_coverage:
        return (
            "Coverage was below the gate with allow_low_coverage; tables are "
            "shown but the call stays inconclusive."
        )
    if is_equivalence_kind(spec.kind):
        if verdict == "keep":
            n = equivalence_diff.n_rows if equivalence_diff is not None else 0
            return (
                f"Predictions are identical within tolerance across {n} rows. "
                "Adopt the candidate lockfile or refactor."
            )
        if verdict == "drop":
            return (
                "Predictions moved and at least one primary-metric interval "
                "lies entirely on the bad side. Keep the control ref/lockfile."
            )
        n_beyond = (
            equivalence_diff.n_beyond_tolerance if equivalence_diff is not None else 0
        )
        _ = deltas
        return (
            f"Predictions differ beyond tolerance on {n_beyond} rows, but no "
            "primary-metric interval is entirely bad. Call stays inconclusive "
            "(different, not worse)."
        )
    _ = deltas
    return (
        "Improvement levers stay inconclusive until FDB-73 sets keep/drop "
        "thresholds. Deltas are descriptive only; keep_if is not applied. "
        f"Headline point metric is {HEADLINE_METRIC} on the decision-relevant pool."
    )


def _coverage_is_drop(spec: EvalSpec, preflight: PreflightResult) -> bool:
    cov = preflight.coverage
    if cov is None or cov.passed:
        return False
    if spec.gates.allow_low_coverage:
        return False
    return bool(spec.gates.drop_if.coverage_below_min)


def _is_production_arm(spec: EvalSpec, role: str) -> bool:
    if is_equivalence_kind(spec.kind):
        # Control is the shipped lockfile / git ref.
        return role == "control"
    orientation = spec.change.orientation
    if orientation == "opt_out":
        return role == "treatment"
    if orientation == "opt_in":
        return role == "control"
    return False


def _n_sims(spec: ModelSpec | None) -> int | None:
    if spec is None:
        return None
    value = spec.params.get("n_sims")
    return int(value) if value is not None else None


def _ensure_model_registry() -> None:
    # Side-effect imports so registered_kinds / components resolve.
    import fplt.models.baselines  # noqa: F401  # pyright: ignore[reportUnusedImport]
    import fplt.models.components.involvement.shares  # noqa: F401  # pyright: ignore[reportUnusedImport]
    import fplt.models.components.involvement.xg  # noqa: F401  # pyright: ignore[reportUnusedImport]
    import fplt.models.components.minutes.empirical  # noqa: F401  # pyright: ignore[reportUnusedImport]
    import fplt.models.components.minutes.empirical_v2  # noqa: F401  # pyright: ignore[reportUnusedImport]
    import fplt.models.components.other.player  # noqa: F401  # pyright: ignore[reportUnusedImport]
    import fplt.models.components.other.priors  # noqa: F401  # pyright: ignore[reportUnusedImport]
    import fplt.models.components.score.dixon_coles  # noqa: F401  # pyright: ignore[reportUnusedImport]
    import fplt.models.components.score.odds_1x2  # noqa: F401  # pyright: ignore[reportUnusedImport]
    import fplt.models.components.score.odds_poisson  # noqa: F401  # pyright: ignore[reportUnusedImport]
    import fplt.models.composite  # noqa: F401  # pyright: ignore[reportUnusedImport]
    import fplt.models.direct  # noqa: F401  # pyright: ignore[reportUnusedImport]
    import fplt.models.external.openfpl  # noqa: F401  # pyright: ignore[reportUnusedImport]


def _check_registered(control: ModelSpec, treatment: ModelSpec) -> list[str]:
    failures: list[str] = []
    kinds = set(registered_kinds())
    comps = set(registered_components())
    for arm in (control, treatment):
        if arm.kind not in kinds:
            failures.append(f"kind {arm.kind!r} not in registered_kinds()")
        for slot, child in arm.components.items():
            if child.kind not in comps and child.kind not in kinds:
                failures.append(
                    f"component {slot}.{child.kind!r} not registered (arm={arm.name})"
                )
    return failures


def _check_kinds(spec: EvalSpec, control: ModelSpec, treatment: ModelSpec) -> list[str]:
    failures: list[str] = []
    slot = spec.change.slot
    if spec.change.control_kind is not None:
        got = (
            control.kind
            if slot is None or spec.change.lever == "model_fork"
            else control.components.get(
                slot, ModelSpec(name="missing", kind="missing")
            ).kind
        )
        if slot is not None and slot not in control.components:
            failures.append(f"control missing components.{slot}")
        elif got != spec.change.control_kind:
            failures.append(
                f"control_kind {spec.change.control_kind!r} != loaded {got!r}"
            )
    if spec.change.treatment_kind is not None:
        got = (
            treatment.kind
            if slot is None or spec.change.lever == "model_fork"
            else treatment.components.get(
                slot, ModelSpec(name="missing", kind="missing")
            ).kind
        )
        if slot is not None and slot not in treatment.components:
            failures.append(f"treatment missing components.{slot}")
        elif got != spec.change.treatment_kind:
            failures.append(
                f"treatment_kind {spec.change.treatment_kind!r} != loaded {got!r}"
            )
    return failures


def _check_equivalence_preflight(
    spec: EvalSpec, control: ModelSpec, treatment: ModelSpec
) -> tuple[list[str], list[str], dict[str, str]]:
    """Preflight checks unique to ``environment`` / ``refactor`` (§8)."""
    failures: list[str] = []
    warnings: list[str] = []
    meta: dict[str, str] = {}
    root = repo_root()
    control_ref = spec.change.control_ref
    treatment_ref = spec.change.treatment_ref
    if not control_ref or not treatment_ref:
        failures.append(
            "equivalence kinds require change.control_ref and treatment_ref"
        )
        return failures, warnings, meta
    if control_ref == treatment_ref:
        failures.append(
            f"control_ref and treatment_ref are identical ({control_ref!r})"
        )

    try:
        control_sha = resolve_git_sha(control_ref, cwd=root)
        treatment_sha = resolve_git_sha(treatment_ref, cwd=root)
    except (OSError, subprocess.CalledProcessError) as exc:
        failures.append(f"could not resolve git refs: {exc}")
        return failures, warnings, meta
    meta["control_ref_sha"] = control_sha
    meta["treatment_ref_sha"] = treatment_sha

    control_hash = resolved_spec_sha256(control)
    treatment_hash = resolved_spec_sha256(treatment)
    specs_differ = control_hash != treatment_hash

    # Materialise lockfiles at each ref via git show (no worktree yet).
    control_lock_text = _git_show_file(control_ref, "uv.lock", cwd=root)
    treatment_lock_text = _git_show_file(treatment_ref, "uv.lock", cwd=root)
    if control_lock_text is None or treatment_lock_text is None:
        failures.append("uv.lock missing on control_ref or treatment_ref")
        return failures, warnings, meta
    pyproject = root / "pyproject.toml"
    with tempfile.TemporaryDirectory(prefix="fplt-equiv-lock-") as tmp_name:
        tmp = Path(tmp_name)
        control_lock_path = tmp / "control.uv.lock"
        treatment_lock_path = tmp / "treatment.uv.lock"
        control_lock_path.write_text(control_lock_text, encoding="utf-8")
        treatment_lock_path.write_text(treatment_lock_text, encoding="utf-8")
        lock_diff = diff_lockfiles(
            control_lock_path,
            treatment_lock_path,
            pyproject=pyproject,
            declared_packages=list(spec.change.packages),
        )
    meta["control_lock_sha256"] = lock_diff.control_sha256
    meta["treatment_lock_sha256"] = lock_diff.treatment_sha256

    if specs_differ and not lock_diff.identical and not spec.change.allow_multi_lever:
        failures.append(
            "spec and environment both differ (two levers); set allow_multi_lever "
            "only for scaffolds"
        )

    if spec.kind == "environment":
        if specs_differ:
            failures.append(
                "kind: environment and the resolved spec hashes differ "
                "(not an environment lever)"
            )
        if lock_diff.identical:
            failures.append(
                "kind: environment and the lockfiles are identical (no contrast)"
            )
        if lock_diff.missing_packages:
            failures.append(
                "kind: environment and a changed direct dependency is missing "
                f"from change.packages: {list(lock_diff.missing_packages)}"
            )
        if lock_diff.transitive_changed:
            warnings.append(
                "transitive dependency changes on environment lever: "
                + "; ".join(lock_diff.transitive_changed[:8])
            )
    elif spec.kind == "refactor":
        if specs_differ:
            failures.append(
                "kind: refactor and the resolved specs differ (not a refactor)"
            )
        if not lock_diff.identical:
            failures.append("kind: refactor and the lockfile differs (not a refactor)")

    return failures, warnings, meta


def _git_show_file(ref: str, path: str, *, cwd: Path) -> str | None:
    try:
        completed = subprocess.run(
            ["git", "show", f"{ref}:{path}"],
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return completed.stdout


def _check_one_lever(
    spec: EvalSpec, control: ModelSpec, treatment: ModelSpec
) -> list[str]:
    if is_equivalence_kind(spec.kind):
        return []
    if spec.change.allow_multi_lever or spec.change.lever == "model_fork":
        return []
    diffs = _component_param_diffs(control, treatment)
    slot = spec.change.slot
    failures: list[str] = []
    if spec.change.lever == "component_swap":
        other = [path for path in diffs if not path.startswith(f"components.{slot}.")]
        if other:
            failures.append(f"multi-lever: diffs outside slot {slot!r}: {other[:5]}")
        return failures
    if spec.change.lever in {"hyperparam", "data_table", "feature_version"}:
        if slot is None:
            # Top-level / multi-slot scaffold without allow_multi_lever.
            if len(diffs) > 1:
                failures.append(f"multi-lever: multiple diffs {diffs[:5]}")
            return failures
        outside = [path for path in diffs if not path.startswith(f"components.{slot}.")]
        if outside:
            failures.append(f"multi-lever: diffs outside slot {slot!r}: {outside[:5]}")
        param_keys = sorted(
            {
                path.split(".params.", 1)[1]
                for path in diffs
                if f"components.{slot}.params." in path
            }
        )
        # Allow component .name to differ alongside one param family.
        if len(param_keys) > 1:
            failures.append(
                f"config diff touches more than one param family: {param_keys}"
            )
        kind_paths = [path for path in diffs if path == f"components.{slot}.kind"]
        if kind_paths and spec.change.lever == "hyperparam":
            failures.append(
                f"hyperparam lever changed components.{slot}.kind (use component_swap)"
            )
    return failures


def _component_param_diffs(control: ModelSpec, treatment: ModelSpec) -> list[str]:
    """Diff paths excluding top-level and nested ``name`` fields."""
    left = control.model_dump(mode="json")
    right = treatment.model_dump(mode="json")
    left.pop("name", None)
    right.pop("name", None)
    return _dict_diff_paths(left, right)


def _dict_diff_paths(left: Any, right: Any, prefix: str = "") -> list[str]:
    if type(left) is not type(right):
        return [prefix or "<root>"]
    if isinstance(left, dict) and isinstance(right, dict):
        left_map = cast(dict[str, Any], left)
        right_map = cast(dict[str, Any], right)
        keys = sorted(set(left_map) | set(right_map))
        paths: list[str] = []
        for key in keys:
            if key == "name":
                continue
            path = f"{prefix}.{key}" if prefix else str(key)
            if key not in left_map or key not in right_map:
                paths.append(path)
            else:
                paths.extend(_dict_diff_paths(left_map[key], right_map[key], path))
        return paths
    if isinstance(left, list):
        if left != right:
            return [prefix or "<root>"]
        return []
    if left != right:
        return [prefix or "<root>"]
    return []


def _check_orientation(
    spec: EvalSpec, control: ModelSpec, treatment: ModelSpec
) -> tuple[list[str], dict[str, Any]]:
    effective: dict[str, Any] = {}
    orientation = spec.change.orientation
    slot = spec.change.slot
    keys = _orientation_keys(control, treatment, slot)
    for key in keys:
        c_val = effective_param(control, slot, key)
        t_val = effective_param(treatment, slot, key)
        effective[f"control.{key}"] = c_val
        effective[f"treatment.{key}"] = t_val
    # market_prior_source is audit-only when present
    for role, arm in (("control", control), ("treatment", treatment)):
        if slot and slot in arm.components:
            src = arm.components[slot].params.get("market_prior_source")
            if src is not None:
                effective[f"{role}.market_prior_source"] = src
        src = arm.params.get("market_prior_source")
        if src is not None:
            effective[f"{role}.market_prior_source"] = src

    if orientation is None or not keys:
        return [], effective
    failures: list[str] = []
    for key in keys:
        c_on = is_effectively_on(key, effective[f"control.{key}"])
        t_on = is_effectively_on(key, effective[f"treatment.{key}"])
        if c_on == t_on:
            failures.append(
                f"no contrast on {key}: both arms effective {'on' if c_on else 'off'}"
            )
        elif c_on and not t_on:
            failures.append(
                f"arms inverted on {key}: control on, treatment off "
                f"(orientation={orientation})"
            )
        # Both orientations require control off / treatment on.
    return failures, effective


def _orientation_keys(
    control: ModelSpec, treatment: ModelSpec, slot: str | None
) -> list[str]:
    found: list[str] = []
    for key in sorted(_ORIENTATION_PARAMS):
        c_params = _params_for(control, slot)
        t_params = _params_for(treatment, slot)
        explicit = key in c_params or key in t_params
        c_val = effective_param(control, slot, key)
        t_val = effective_param(treatment, slot, key)
        if c_val != t_val or explicit:
            # Only track when at least one arm sets the key, or values diverge
            # via defaults (absent vs explicit).
            if explicit or c_val != t_val:
                found.append(key)
    return found


def _params_for(spec: ModelSpec, slot: str | None) -> dict[str, Any]:
    if slot is None:
        return dict(spec.params)
    if slot not in spec.components:
        return {}
    return dict(spec.components[slot].params)


def _check_tables(spec: EvalSpec, store: DataStore) -> list[str]:
    failures: list[str] = []
    from fplt.schemas.staged import STAGED_SCHEMAS

    for name in spec.change.tables:
        if name not in STAGED_SCHEMAS:
            failures.append(f"table {name!r} is not a staged schema")
            continue
        path = store.path(name)
        if not path.is_file():
            failures.append(f"table {name!r} missing under staged storage")
            continue
        frame = store.read(name)
        if "available_at" not in frame.columns:
            failures.append(f"table {name!r} has no available_at column")
        elif frame.is_empty():
            # Empty is a warn-level concern for optional tables; still fail when
            # the consumer listed it as required for the lever.
            failures.append(f"table {name!r} is empty")
    return failures


def _measure_coverage(
    spec: EvalSpec,
    plan: PromoteEvalPlan,
    store: DataStore,
    treatment: ModelSpec,
) -> tuple[CoverageResult | None, list[str], str | None]:
    metric = spec.gates.coverage.metric
    if metric is None:
        if spec.gates.min_coverage is not None:
            raise ValueError("gates.min_coverage set but gates.coverage.metric is null")
        return None, [], None
    warnings: list[str] = []
    pit_caveat: str | None = None
    fixtures = _target_fixtures(store, plan.seasons, plan.from_gw, plan.to_gw)
    if metric == "odds_join":
        coverage, odds_warnings, pit_caveat = _coverage_odds_join(
            spec, plan, store, fixtures, treatment
        )
        warnings.extend(odds_warnings)
        return coverage, warnings, pit_caveat
    if metric == "penalties_order_non_null":
        return (
            _coverage_penalties_order(spec, plan, store, fixtures),
            warnings,
            None,
        )
    if metric == "dc_season_eligible":
        return _coverage_dc_season(spec, fixtures), warnings, None
    if metric == "odds_history_join":
        raise ValueError(
            "odds_history_join is not implemented in P1+P2 (history prior package)"
        )
    raise ValueError(f"unknown coverage metric {metric!r}")


def _target_fixtures(
    store: DataStore, seasons: list[str], from_gw: int, to_gw: int
) -> pl.DataFrame:
    frame = (
        store.table("fixtures")
        .filter(
            pl.col("season").is_in(seasons)
            & pl.col("gw").is_between(from_gw, to_gw, closed="both")
        )
        .sort(["season", "gw", "fixture_id", "available_at"])
        .unique(subset=["season", "fixture_id"], keep="last")
        .collect()
    )
    return frame


def _coverage_odds_join(
    spec: EvalSpec,
    plan: PromoteEvalPlan,
    store: DataStore,
    fixtures: pl.DataFrame,
    treatment: ModelSpec,
) -> tuple[CoverageResult, list[str], str | None]:
    warnings: list[str] = []
    read = spec.gates.coverage.read or "strict_pit"
    use_proxy = False
    score = treatment.components.get("score")
    if score is not None:
        use_proxy = bool(score.params.get("use_closing_proxy", False))
    if read == "closing_proxy_labelled" and not use_proxy:
        raise ValueError(
            "coverage.read=closing_proxy_labelled but treatment "
            "use_closing_proxy is not true"
        )
    if read == "prematch_scheduled" and use_proxy:
        raise ValueError(
            "coverage.read=prematch_scheduled but treatment "
            "use_closing_proxy is true (CS-only / prematch boards must "
            "keep use_closing_proxy: false)"
        )
    odds = store.read("match_odds")
    denom = fixtures.height
    if denom == 0:
        return (
            CoverageResult(
                metric="odds_join",
                numerator=0,
                denominator=0,
                rate=0.0,
                min_coverage=spec.gates.min_coverage,
                passed=spec.gates.min_coverage is None or spec.gates.min_coverage == 0,
                extras={},
            ),
            warnings,
            None,
        )

    # Per-fixture deadline for strict PIT / prematch coverage.
    deadlines = (
        store.table("gameweeks")
        .filter(pl.col("season").is_in(plan.seasons))
        .sort("available_at")
        .unique(subset=["season", "gw"], keep="last")
        .select(["season", "gw", "deadline_time"])
        .collect()
    )
    targets = fixtures.join(deadlines, on=["season", "gw"], how="left")

    fixture_id_rate, team_rate, strict_rate, proxy_rate = _odds_join_rates(
        targets, odds, read=read
    )
    prematch_rate = 0.0
    prematch_dropped: tuple[int, ...] = ()
    if read == "prematch_scheduled":
        from fplt.data.football_data import check_prematch_deadline_coverage

        prematch = check_prematch_deadline_coverage(targets, odds, min_coverage=None)
        prematch_rate = prematch.rate
        prematch_dropped = prematch.dropped_fixture_ids
        if not prematch.passed and prematch.message:
            warnings.append(prematch.message)
    if team_rate - fixture_id_rate > 0.05 + 1e-12:
        warnings.append(
            f"team-key odds rate {team_rate:.3f} exceeds fixture_id rate "
            f"{fixture_id_rate:.3f} by >5pp"
        )
    if not odds.is_empty() and (odds["odds_kind"] == "closing_proxy").any():
        warnings.append("closing-proxy odds rows present")
    if (
        not odds.is_empty()
        and "odds_kind" in odds.columns
        and (odds["odds_kind"] == "prematch_scheduled").any()
    ):
        warnings.append("prematch_scheduled odds rows present")

    if read == "strict_pit":
        rate = strict_rate
    elif read == "closing_proxy_labelled":
        rate = proxy_rate
    else:
        rate = prematch_rate
    numerator = int(round(rate * denom))
    min_cov = spec.gates.min_coverage
    passed = True if min_cov is None else rate + 1e-12 >= min_cov
    mps = "latest"
    mcs = None
    if score is not None:
        mps = str(score.params.get("market_prior_source", "latest"))
        if "market_cs_strength" in score.params:
            mcs = score.params.get("market_cs_strength")
    caveat = (
        "Closing lines are stamped available_at=kickoff, after the FPL deadline. "
        "This run did not move the PIT cutoff. "
        f"Strict-PIT odds join = {strict_rate:.1%}. "
        "Closing-proxy join (labelled; used only by specs with "
        f"use_closing_proxy: true) = {proxy_rate:.1%}. "
        f"Prematch-scheduled deadline coverage = {prematch_rate:.1%}. "
        f"market_prior_source = {mps}."
    )
    if mcs is not None:
        caveat += f" market_cs_strength = {mcs}."
    # Always print caveat when match_odds is involved or closing_proxy exists.
    if "match_odds" in spec.change.tables or (
        not odds.is_empty() and (odds["odds_kind"] == "closing_proxy").any()
    ):
        pit_caveat: str | None = caveat
    else:
        pit_caveat = caveat if read in {"strict_pit", "prematch_scheduled"} else None

    extras: dict[str, Any] = {
        "fixture_id_join_rate": fixture_id_rate,
        "team_key_join_rate": team_rate,
        "strict_pit_rate": strict_rate,
        "closing_proxy_rate": proxy_rate,
        "prematch_scheduled_rate": prematch_rate,
        "read": read,
    }
    if prematch_dropped:
        extras["prematch_dropped_fixture_ids"] = list(prematch_dropped)

    return (
        CoverageResult(
            metric="odds_join",
            numerator=numerator,
            denominator=denom,
            rate=rate,
            min_coverage=min_cov,
            passed=passed,
            extras=extras,
        ),
        warnings,
        pit_caveat,
    )


def _odds_join_rates(
    targets: pl.DataFrame, odds: pl.DataFrame, *, read: CoverageRead
) -> tuple[float, float, float, float]:
    if targets.is_empty():
        return 0.0, 0.0, 0.0, 0.0
    denom = targets.height

    # Fixture-id join (season + fixture_id), ignoring available_at.
    with_id = odds.filter(pl.col("fixture_id").is_not_null())
    via_id = targets.select(["season", "fixture_id"]).join(
        with_id.sort("available_at")
        .unique(subset=["season", "fixture_id"], keep="last")
        .select(["season", "fixture_id", *_ODDS_COLS]),
        on=["season", "fixture_id"],
        how="left",
    )
    fixture_id_rate = (
        via_id.filter(
            pl.col("home_odds").is_not_null()
            & pl.col("draw_odds").is_not_null()
            & pl.col("away_odds").is_not_null()
        ).height
        / denom
    )

    team_keys = ["season", "team_h_key", "team_a_key"]
    via_team = targets.select(["season", "team_h_key", "team_a_key"]).join(
        odds.sort("available_at")
        .unique(subset=team_keys, keep="last")
        .select([*team_keys, *_ODDS_COLS]),
        on=team_keys,
        how="left",
    )
    team_rate = (
        via_team.filter(
            pl.col("home_odds").is_not_null()
            & pl.col("draw_odds").is_not_null()
            & pl.col("away_odds").is_not_null()
        ).height
        / denom
    )

    # Strict PIT: odds.available_at <= deadline_time, prefer fixture_id then teams.
    strict_odds = odds
    proxy_odds = (
        odds.filter(pl.col("odds_kind") == "closing_proxy")
        if "odds_kind" in odds.columns
        else odds.clear()
    )

    strict_joined = _join_odds_for_targets(targets, strict_odds, pit=True)
    proxy_joined = _join_odds_for_targets(targets, proxy_odds, pit=False)
    strict_rate = (
        strict_joined.filter(
            pl.col("home_odds").is_not_null()
            & pl.col("draw_odds").is_not_null()
            & pl.col("away_odds").is_not_null()
        ).height
        / denom
    )
    proxy_rate = (
        proxy_joined.filter(
            pl.col("home_odds").is_not_null()
            & pl.col("draw_odds").is_not_null()
            & pl.col("away_odds").is_not_null()
        ).height
        / denom
    )
    del read  # rate selection happens in caller
    return fixture_id_rate, team_rate, strict_rate, proxy_rate


def _join_odds_for_targets(
    targets: pl.DataFrame, odds: pl.DataFrame, *, pit: bool
) -> pl.DataFrame:
    if targets.is_empty() or odds.is_empty():
        return targets.select(
            pl.col("season"),
            pl.col("fixture_id"),
            pl.lit(None).cast(pl.Float64).alias("home_odds"),
            pl.lit(None).cast(pl.Float64).alias("draw_odds"),
            pl.lit(None).cast(pl.Float64).alias("away_odds"),
        )
    indexed = targets.with_row_index("_row")
    working = odds
    if pit and "deadline_time" in targets.columns:
        # Join deadline onto odds via fixture season/gw through targets later.
        pass
    # fixture_id path
    id_keys = ["season", "fixture_id"]
    with_id = working.filter(pl.col("fixture_id").is_not_null())
    if pit:
        via_id = (
            indexed.select(["_row", "season", "fixture_id", "deadline_time"])
            .join(
                with_id.select(["season", "fixture_id", "available_at", *_ODDS_COLS]),
                on=["season", "fixture_id"],
                how="left",
            )
            .filter(
                pl.col("home_odds").is_null()
                | (pl.col("available_at") <= pl.col("deadline_time"))
            )
            .sort("available_at")
            .unique(subset=["_row"], keep="last")
            .select(["_row", *_ODDS_COLS])
        )
    else:
        via_id = (
            indexed.select(["_row", "season", "fixture_id"])
            .join(
                with_id.sort("available_at")
                .unique(subset=id_keys, keep="last")
                .select([*id_keys, *_ODDS_COLS]),
                on=id_keys,
                how="left",
            )
            .select(["_row", *_ODDS_COLS])
        )
    team_keys = ["season", "team_h_key", "team_a_key"]
    if pit:
        via_teams = (
            indexed.select(
                ["_row", "season", "team_h_key", "team_a_key", "deadline_time"]
            )
            .join(
                working.select([*team_keys, "available_at", *_ODDS_COLS]),
                on=team_keys,
                how="left",
            )
            .filter(
                pl.col("home_odds").is_null()
                | (pl.col("available_at") <= pl.col("deadline_time"))
            )
            .sort("available_at")
            .unique(subset=["_row"], keep="last")
            .select(["_row", *_ODDS_COLS])
        )
    else:
        via_teams = (
            indexed.select(["_row", *team_keys])
            .join(
                working.sort("available_at")
                .unique(subset=team_keys, keep="last")
                .select([*team_keys, *_ODDS_COLS]),
                on=team_keys,
                how="left",
            )
            .select(["_row", *_ODDS_COLS])
        )
    merged = (
        indexed.select("_row")
        .join(via_id, on="_row", how="left")
        .join(via_teams, on="_row", how="left", suffix="_team")
        .sort("_row")
    )
    return merged.select(
        [
            pl.coalesce([pl.col(name), pl.col(f"{name}_team")]).alias(name)
            for name in _ODDS_COLS
        ]
    )


def _coverage_penalties_order(
    spec: EvalSpec,
    plan: PromoteEvalPlan,
    store: DataStore,
    fixtures: pl.DataFrame,
) -> CoverageResult:
    stats = (
        store.table("player_fixture_stats")
        .filter(
            pl.col("season").is_in(plan.seasons)
            & pl.col("gw").is_between(plan.from_gw, plan.to_gw, closed="both")
        )
        .collect()
    )
    # Player–fixtures scored: prefer stats rows; fall back to fixture×players.
    if stats.is_empty():
        denom = 0
        numerator = 0
        extras: dict[str, Any] = {
            "penalties_order_eq_1_rate": 0.0,
            "source_fpl_api_share": 0.0,
        }
    else:
        deadlines = (
            store.table("gameweeks")
            .filter(pl.col("season").is_in(plan.seasons))
            .sort("available_at")
            .unique(subset=["season", "gw"], keep="last")
            .select(["season", "gw", "deadline_time"])
            .collect()
        )
        keyed = stats.select(["season", "gw", "player_key", "fixture_id"]).unique()
        keyed = keyed.join(deadlines, on=["season", "gw"], how="left")
        status = store.read("player_status")
        # Latest status per player with available_at <= deadline.
        joined = (
            keyed.join(
                status.select(
                    [
                        "season",
                        "player_key",
                        "penalties_order",
                        "available_at",
                        "source",
                    ]
                ),
                on=["season", "player_key"],
                how="left",
            )
            .filter(
                pl.col("available_at").is_null()
                | (pl.col("available_at") <= pl.col("deadline_time"))
            )
            .sort("available_at")
            .unique(subset=["season", "fixture_id", "player_key"], keep="last")
        )
        denom = keyed.height
        non_null = joined.filter(pl.col("penalties_order").is_not_null())
        numerator = non_null.height
        eq1 = joined.filter(pl.col("penalties_order") == 1).height
        fpl = joined.filter(pl.col("source") == "fpl_api").height
        extras = {
            "penalties_order_eq_1_rate": (eq1 / denom) if denom else 0.0,
            "source_fpl_api_share": (fpl / denom) if denom else 0.0,
        }
    rate = (numerator / denom) if denom else 0.0
    min_cov = spec.gates.min_coverage
    passed = True if min_cov is None else rate + 1e-12 >= min_cov
    del fixtures
    return CoverageResult(
        metric="penalties_order_non_null",
        numerator=numerator,
        denominator=denom,
        rate=rate,
        min_coverage=min_cov,
        passed=passed,
        extras=extras,
    )


def _coverage_dc_season(spec: EvalSpec, fixtures: pl.DataFrame) -> CoverageResult:
    denom = fixtures.height
    if denom == 0:
        rate = 0.0
        numerator = 0
    else:
        seasons = fixtures["season"].unique().to_list()
        eligible = {
            season
            for season in seasons
            if Rules.for_season(str(season)).defensive_contributions
        }
        mask = fixtures["season"].is_in(sorted(eligible))
        numerator = int(mask.sum())
        rate = numerator / denom
    min_cov = spec.gates.min_coverage
    passed = True if min_cov is None else rate + 1e-12 >= min_cov
    return CoverageResult(
        metric="dc_season_eligible",
        numerator=numerator,
        denominator=denom,
        rate=rate,
        min_coverage=min_cov,
        passed=passed,
        extras={},
    )


def _check_leak(
    spec: EvalSpec,
    plan: PromoteEvalPlan,
    data_dir: Path,
    store: DataStore,
) -> str | None:
    """Fail when a feature for a predicted fixture is after the deadline."""
    feature_names = list(spec.change.features)
    if not feature_names and spec.kind != "feature":
        return None
    fixtures = _target_fixtures(store, plan.seasons, plan.from_gw, plan.to_gw)
    if fixtures.is_empty():
        return None
    deadlines = (
        store.table("gameweeks")
        .filter(pl.col("season").is_in(plan.seasons))
        .sort("available_at")
        .unique(subset=["season", "gw"], keep="last")
        .select(["season", "gw", "deadline_time"])
        .collect()
    )
    targets = fixtures.join(deadlines, on=["season", "gw"], how="left")
    feature_root = data_dir / "features"
    if not feature_root.is_dir():
        return None
    for path in feature_root.glob("**/part.parquet"):
        try:
            frame = pl.read_parquet(path)
        except Exception:
            continue
        if "fixture_id" not in frame.columns or "available_at" not in frame.columns:
            continue
        if "season" not in frame.columns:
            continue
        merged = frame.join(
            targets.select(["season", "fixture_id", "deadline_time"]),
            on=["season", "fixture_id"],
            how="inner",
        )
        bad = merged.filter(pl.col("available_at") > pl.col("deadline_time"))
        if not bad.is_empty():
            return (
                f"leak: {path} has {bad.height} predicted-fixture row(s) with "
                "available_at after the deadline"
            )
    return None


def _run_compare(
    plan: PromoteEvalPlan, *, data_dir: Path, output_dir: Path
) -> tuple[CompareResult, dict[str, float], dict[str, int]]:
    wall: dict[str, float] = {}
    rss: dict[str, int] = {}
    # One compare call scores control, treatment, and any dose arms.
    compare_models_list = [plan.control, plan.treatment, *plan.arms]
    started = time.perf_counter()
    before = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    result = compare_models(
        data_dir,
        plan.seasons[0],
        models=compare_models_list,
        test_from_gw=plan.from_gw,
        test_to_gw=plan.to_gw,
        fit_gw=plan.from_gw if plan.modes == ["walk_forward"] else None,
        seasons_back=plan.seasons_back,
        modes=list(plan.modes),  # type: ignore[arg-type]
        steps=plan.suites.steps,
        output_dir=output_dir,
    )
    # Multi-season: run remaining seasons and concat summaries.
    summaries = [result.summary]
    details = [result.detail]
    for season in plan.seasons[1:]:
        season_out = output_dir / season
        part = compare_models(
            data_dir,
            season,
            models=compare_models_list,
            test_from_gw=plan.from_gw,
            test_to_gw=plan.to_gw,
            seasons_back=plan.seasons_back,
            modes=list(plan.modes),  # type: ignore[arg-type]
            steps=plan.suites.steps,
            output_dir=season_out,
        )
        summaries.append(part.summary)
        details.append(part.detail)
    elapsed = time.perf_counter() - started
    after = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    wall["both"] = elapsed
    rss["both"] = max(after - before, 0)
    if len(summaries) > 1:
        result = CompareResult(
            season=",".join(plan.seasons),
            test_from_gw=result.test_from_gw,
            test_to_gw=result.test_to_gw,
            fit_gw=result.fit_gw,
            seasons_back=result.seasons_back,
            modes=result.modes,
            runs=result.runs,
            summary=pl.concat(summaries, how="vertical_relaxed"),
            detail=pl.concat(details, how="vertical_relaxed"),
            output_dir=output_dir,
            time_decay=result.time_decay,
            skipped_openfpl=result.skipped_openfpl,
        )
    return result, wall, rss


def _prediction_deltas(
    compare: CompareResult,
    control: str,
    treatment: str,
    *,
    data_dir: Path,
    position_specific: bool = False,
) -> dict[str, Any]:
    summary = compare.summary
    if summary.is_empty() or "model_name" not in summary.columns:
        return {}
    # Prefer walk_forward horizon 1 overall rows.
    frame = summary
    if "mode" in frame.columns:
        wf = frame.filter(pl.col("mode") == "walk_forward")
        if not wf.is_empty():
            frame = wf
    if "horizon" in frame.columns:
        h1 = frame.filter(pl.col("horizon") == 1)
        if not h1.is_empty():
            frame = h1
    control_rows = frame.filter(pl.col("model_name") == control)
    treatment_rows = frame.filter(pl.col("model_name") == treatment)
    if control_rows.is_empty() or treatment_rows.is_empty():
        return {}
    c = control_rows.to_dicts()[0]
    t = treatment_rows.to_dicts()[0]
    deltas: dict[str, Any] = {}
    # Lower is better → control - treatment; higher is better → treatment - control.
    # RMSE first (headline); MAE kept for continuity with older A/Bs.
    for key in ("rmse", "mae"):
        if c.get(key) is not None and t.get(key) is not None:
            deltas[key] = float(c[key]) - float(t[key])
    for key in ("spearman", "top10_precision"):
        if c.get(key) is not None and t.get(key) is not None:
            deltas[key] = float(t[key]) - float(c[key])

    id_by_name = _model_ids_by_name(compare)
    control_id = id_by_name.get(control)
    treatment_id = id_by_name.get(treatment)
    if control_id is None or treatment_id is None:
        deltas["per_gw_wins"] = {"control": 0, "treatment": 0, "ties": 0}
        return deltas

    try:
        ledger = read_predictions(data_dir)
    except Exception:
        deltas["per_gw_wins"] = {"control": 0, "treatment": 0, "ties": 0}
        return deltas
    model_ids = {control_id, treatment_id}
    scored = ledger.filter(pl.col("model_id").is_in(sorted(model_ids)))
    if scored.is_empty():
        deltas["per_gw_wins"] = {"control": 0, "treatment": 0, "ties": 0}
        return deltas
    try:
        scored = filter_decision_pool(scored, DataStore(data_dir))
    except Exception:
        logger.warning(
            "decision pool filter failed; intervals use full scored ledger",
            exc_info=True,
        )
    if scored.is_empty():
        deltas["per_gw_wins"] = {"control": 0, "treatment": 0, "ties": 0}
        return deltas

    positions = ("DEF", "MID", "FWD", "GKP") if position_specific else None
    paired = intervals_for_pair(
        scored,
        control_id=control_id,
        treatment_id=treatment_id,
        metrics=("rmse", "mae", "spearman"),
        positions=positions,
    )
    deltas["intervals"] = [interval_to_dict(item) for item in paired]
    # Equivalence-only signal; improvement runners ignore this for the call.
    deltas["equivalence_interval_drop"] = equivalence_drop_from_intervals(paired)

    control_rmse = per_gw_losses(scored, model_id=control_id, metric="rmse")
    treatment_rmse = per_gw_losses(scored, model_id=treatment_id, metric="rmse")
    deltas["per_gw_wins"] = per_gw_wins_from_losses(control_rmse, treatment_rmse)
    control_mae = per_gw_losses(scored, model_id=control_id, metric="mae")
    treatment_mae = per_gw_losses(scored, model_id=treatment_id, metric="mae")
    deltas["per_gw_wins_mae"] = per_gw_wins_from_losses(control_mae, treatment_mae)

    # SPA / MCS only when several arms share the window (dose / multi-arm).
    spa_mcs = _spa_mcs_from_ledger(
        scored, compare=compare, control=control, id_by_name=id_by_name
    )
    if spa_mcs is not None:
        deltas["spa"] = spa_mcs["spa"]
        deltas["mcs"] = spa_mcs["mcs"]
    return deltas


def _spa_mcs_from_ledger(
    scored: pl.DataFrame,
    *,
    compare: CompareResult,
    control: str,
    id_by_name: dict[str, str],
) -> dict[str, Any] | None:
    """Build SPA/MCS payloads when compare scored more than two arms."""
    arm_names = sorted({run.model_name for run in compare.runs})
    if len(arm_names) <= 2 or control not in arm_names:
        return None
    keys: pl.DataFrame | None = None
    per_arm: dict[str, pl.DataFrame] = {}
    for name in arm_names:
        mid = id_by_name.get(name)
        if mid is None:
            return None
        losses = per_gw_losses(scored, model_id=mid, metric=HEADLINE_METRIC)
        if losses.is_empty():
            return None
        per_arm[name] = losses
        gw_keys = losses.select(["target_season", "target_gw"])
        keys = (
            gw_keys
            if keys is None
            else keys.join(gw_keys, on=["target_season", "target_gw"], how="inner")
        )
    if keys is None or keys.is_empty():
        return None
    loss_by_arm: dict[str, Any] = {}
    for name, losses in per_arm.items():
        aligned = keys.join(
            losses, on=["target_season", "target_gw"], how="inner"
        ).sort(["target_season", "target_gw"])
        loss_by_arm[name] = aligned["loss"].to_numpy()
    selection = multi_arm_selection(loss_by_arm, control=control)
    if selection is None:
        return None
    spa, mcs = selection
    return {
        "spa": {
            "pvalue_consistent": spa.pvalue_consistent,
            "pvalue_lower": spa.pvalue_lower,
            "pvalue_upper": spa.pvalue_upper,
            "n_gw": spa.n_gw,
            "n_models": spa.n_models,
        },
        "mcs": {
            "included": list(mcs.included),
            "excluded": list(mcs.excluded),
            "size": mcs.size,
            "n_gw": mcs.n_gw,
        },
    }


def _model_ids_by_name(compare: CompareResult) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for run in compare.runs:
        mapping[run.model_name] = run.spec.model_id
    return mapping


def _lever_is_position_specific(spec: EvalSpec) -> bool:
    return any("defensive_contribution" in line for line in spec.change.diff)


def _format_interval_lines(deltas: dict[str, Any]) -> list[str]:
    raw_obj = deltas.get("intervals")
    if not isinstance(raw_obj, list) or not raw_obj:
        return ["- HAC / bootstrap intervals: unavailable (no paired gameweeks)"]
    raw = cast(list[object], raw_obj)
    lines: list[str] = []
    for entry in raw:
        if not isinstance(entry, dict):
            continue
        item = cast(dict[str, Any], entry)
        metric = item.get("metric")
        position = item.get("position")
        label = f"{metric}" if position is None else f"{metric}[{position}]"
        mean = item.get("mean")
        hac = item.get("hac_ci")
        boot = item.get("bootstrap_ci")
        hac_p = item.get("hac_pvalue")
        hac_se = item.get("hac_se")
        lines.append(
            f"- {label}: mean_d={mean} HAC_se={hac_se} HAC_p={hac_p} "
            f"HAC_CI={hac} bootstrap_CI={boot}"
        )
    return lines


def _folds_from_compare(compare: CompareResult, data_dir: Path) -> list[dict[str, Any]]:
    try:
        ledger = read_predictions(data_dir)
    except Exception:
        return []
    if ledger.is_empty():
        return []
    model_ids = {run.spec.model_id for run in compare.runs}
    scored = ledger.filter(pl.col("model_id").is_in(sorted(model_ids)))
    if scored.is_empty():
        return []
    folds: list[dict[str, Any]] = []
    for (season, gw), group in scored.group_by(["as_of_season", "as_of_gw"]):
        folds.append(
            {
                "season": season,
                "gw": gw,
                "as_of": f"{season}_gw{int(gw):02d}",
                "n_train": None,
                "n_test": group.height,
            }
        )
    return sorted(folds, key=lambda row: (str(row["season"]), int(row["gw"])))


def _default_output_dir(data_dir: Path, eval_name: str) -> Path:
    stamp = datetime.now(tz=UTC).strftime("%Y%m%dT%H%M%SZ")
    return data_dir / "promote_eval" / eval_name / stamp


def _write_recommendation(out: Path, text: str) -> None:
    (out / "recommendation.md").write_text(text, encoding="utf-8")


def _write_spec_diff(out: Path, preflight: PreflightResult) -> None:
    control = preflight.control_spec
    treatment = preflight.treatment_spec
    payload = {
        "control": control.model_dump(mode="json") if control else None,
        "treatment": treatment.model_dump(mode="json") if treatment else None,
        "diff_paths": (
            _component_param_diffs(control, treatment) if control and treatment else []
        ),
        "effective_flags": preflight.effective_flags,
    }
    (out / "spec_diff.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def _write_summary_json(
    out: Path,
    spec: EvalSpec,
    plan: PromoteEvalPlan,
    preflight: PreflightResult,
    *,
    verdict: Verdict,
    deltas: dict[str, Any] | None,
    component_deltas: dict[str, Any] | None = None,
) -> None:
    payload = {
        "eval_name": spec.name,
        "verdict": verdict,
        "control": plan.control,
        "treatment": plan.treatment,
        "coverage": None
        if preflight.coverage is None
        else {
            "metric": preflight.coverage.metric,
            "numerator": preflight.coverage.numerator,
            "denominator": preflight.coverage.denominator,
            "rate": preflight.coverage.rate,
            "min_coverage": preflight.coverage.min_coverage,
            "passed": preflight.coverage.passed,
            "extras": preflight.coverage.extras,
        },
        "deltas": deltas,
        "component_deltas": component_deltas,
        "failures": preflight.failures,
        "warnings": preflight.warnings,
    }
    (out / "summary.json").write_text(
        json.dumps(payload, indent=2, default=str) + "\n", encoding="utf-8"
    )


def _write_audit(
    out: Path,
    spec: EvalSpec,
    plan: PromoteEvalPlan,
    preflight: PreflightResult,
    *,
    data_dir: Path,
    argv: list[str] | None,
    wall_clock: dict[str, float],
    peak_rss: dict[str, int],
    folds: list[dict[str, Any]],
    verdict: Verdict,
) -> None:
    data_hashes = _hash_staged(data_dir, spec)
    config = {}
    if preflight.control_spec is not None:
        config["control"] = {
            "name": preflight.control_spec.name,
            "model_id": preflight.control_spec.model_id,
            "sha256": _spec_sha256(preflight.control_spec),
        }
    if preflight.treatment_spec is not None:
        config["treatment"] = {
            "name": preflight.treatment_spec.name,
            "model_id": preflight.treatment_spec.model_id,
            "sha256": _spec_sha256(preflight.treatment_spec),
        }
    try:
        fplt_ver = package_version("fplt")
    except Exception:
        fplt_ver = "0.0.0"
    lock_payload: str | dict[str, str | None]
    if is_equivalence_kind(spec.kind):
        lock_payload = {
            "control": preflight.control_lock_sha256,
            "treatment": preflight.treatment_lock_sha256,
        }
    else:
        lock_payload = uv_lock_sha256()
    payload = {
        "eval_name": spec.name,
        "verdict": verdict,
        "question_type": (
            "equivalence" if is_equivalence_kind(spec.kind) else "improvement"
        ),
        "git_sha": preflight.git_sha,
        "git_dirty": preflight.git_dirty,
        "uv_lock_sha256": lock_payload,
        "control_ref": preflight.control_ref,
        "treatment_ref": preflight.treatment_ref,
        "control_ref_sha": preflight.control_ref_sha,
        "treatment_ref_sha": preflight.treatment_ref_sha,
        "python": sys.version.split()[0],
        "os": platform.platform(),
        "fplt_version": fplt_ver,
        "argv": argv or [],
        "data": data_hashes,
        "config": config,
        "seed": (
            preflight.treatment_spec.seed
            if preflight.treatment_spec is not None
            else None
        ),
        "n_sims": _n_sims(preflight.treatment_spec or preflight.control_spec),
        "folds": folds,
        "market_prior_source": preflight.effective_flags.get(
            "treatment.market_prior_source"
        ),
        "wall_clock_s": wall_clock,
        "peak_rss": peak_rss,
        "window": {
            "seasons": plan.seasons,
            "from_gw": plan.from_gw,
            "to_gw": plan.to_gw,
            "seasons_back": plan.seasons_back,
            "modes": plan.modes,
        },
        "force_inconclusive": preflight.force_inconclusive,
    }
    (out / "audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str) + "\n",
        encoding="utf-8",
    )


def _spec_sha256(spec: ModelSpec) -> str:
    raw = json.dumps(
        spec.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
    )
    return hashlib.sha256(raw.encode()).hexdigest()


def _hash_staged(data_dir: Path, spec: EvalSpec) -> dict[str, Any]:
    store = DataStore(data_dir)
    names = set(spec.change.tables)
    # Always record core tables when present.
    names.update({"fixtures", "gameweeks", "player_status", "player_fixture_stats"})
    if spec.gates.coverage.metric == "odds_join":
        names.add("match_odds")
    out: dict[str, Any] = {}
    for name in sorted(names):
        path = store.path(name)
        if not path.is_file():
            continue
        frame = store.read(name)
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        max_at = None
        if "available_at" in frame.columns and not frame.is_empty():
            max_at = str(frame["available_at"].max())
        out[name] = {
            "sha256": digest,
            "row_count": frame.height,
            "max_available_at": max_at,
        }
    return out


def _commands_for(
    spec: EvalSpec,
    *,
    seasons: list[str],
    from_gw: int,
    to_gw: int,
    suites: SuitesSpec,
    modes: list[CompareModeName],
    seasons_back: int,
    fit_gw: int | None,
    models_dir: Path | None = None,
    warnings: list[str] | None = None,
) -> list[str]:
    commands: list[str] = []
    arms = " ".join(compare_arm_names(spec))
    warn = warnings if warnings is not None else []
    if is_equivalence_kind(spec.kind):
        control_ref = spec.change.control_ref or "HEAD"
        treatment_ref = spec.change.treatment_ref or "HEAD"
        commands.append(
            f"git worktree add --detach <tmp>/control {control_ref} && uv sync --frozen"
        )
        commands.append(
            f"git worktree add --detach <tmp>/treatment {treatment_ref} && "
            "uv sync --frozen"
        )
        for season in seasons:
            mode_flag = modes[0] if modes else "walk_forward"
            for role in ("control", "treatment"):
                commands.append(
                    f"uv run --directory <tmp>/{role} --frozen "
                    f"python -m fplt.backtest._produce_arm_main  "
                    f"# model={spec.control} season={season} "
                    f"gw={from_gw}-{to_gw} mode={mode_flag} "
                    f"seasons_back={seasons_back} steps={suites.steps}"
                )
        commands.append(
            "compare prediction ledgers (assert_frame_equal + §3.2 intervals)"
        )
        return commands
    for season in seasons:
        if suites.components:
            commands.append(
                _components_command(
                    spec,
                    season=season,
                    from_gw=from_gw,
                    to_gw=to_gw,
                    models_dir=models_dir,
                    warnings=warn,
                )
            )
        if suites.prediction:
            mode_flag = " ".join(modes)
            fit = f" --fit-gw {fit_gw}" if fit_gw is not None else ""
            commands.append(
                "uv run fplt compare "
                f"--season {season} --models {arms} "
                f"--test-from-gw {from_gw} --test-to-gw {to_gw} "
                f"--seasons-back {seasons_back} --modes {mode_flag} "
                f"--steps {suites.steps}{fit}"
            )
        if suites.decisions:
            max_ppg = suites.max_mean_ppg
            commands.append(
                "uv run fplt decisions "
                f"--season {season} --models {arms} "
                f"--from-gw {from_gw} --to-gw {to_gw} "
                f"--horizon {suites.decisions_horizon} "
                f"--max-mean-ppg {max_ppg:g}"
            )
    if not commands:
        commands.append("# no suites enabled")
    return commands


def is_scoreline_lever(spec: EvalSpec) -> bool:
    """Whether this eval should write the scoreline per-match board."""
    return spec.change.slot == "score" or "match_odds" in spec.change.tables


def is_n_sims_lever(spec: EvalSpec) -> bool:
    """Whether ``n_sims`` is the declared lever (keep the sims sweep)."""
    if any("n_sims" in item for item in spec.change.diff):
        return True
    notes = spec.change.notes.lower()
    return "n_sims" in notes and "unchanged" not in notes


def slot_component_names(
    control: ModelSpec,
    treatment: ModelSpec,
    slot: str | None,
) -> tuple[str, str] | None:
    """Return ``(control_slot_name, treatment_slot_name)`` for ``slot``.

    Args:
        control: Control parent ModelSpec.
        treatment: Treatment parent ModelSpec.
        slot: Component slot such as ``score`` / ``minutes`` / ``involvement``.

    Returns:
        The nested ``components.<slot>.name`` values, or ``None`` when the slot
        is missing on either arm.
    """
    if slot is None:
        return None
    if slot not in control.components or slot not in treatment.components:
        return None
    return control.components[slot].name, treatment.components[slot].name


def raw_de_vig_spec(*, coverage_read: CoverageRead | None) -> ModelSpec:
    """Build the raw proportional de-vig 1X2 market arm for scoreline boards.

    Uses the same clock as ``coverage.read``: ``closing_proxy_labelled`` admits
    kickoff-stamped closes; ``strict_pit`` (default) does not. This is not
    ``odds_poisson_v1``.
    """
    use_proxy = coverage_read == "closing_proxy_labelled"
    return ModelSpec(
        name=_RAW_DE_VIG_ARM,
        kind="odds_1x2",
        params={
            "use_closing_proxy": use_proxy,
            "fallback_p_home": 1.0 / 3.0,
            "fallback_p_draw": 1.0 / 3.0,
            "fallback_p_away": 1.0 / 3.0,
        },
        refit="never",
        seed=0,
    )


def _components_command(
    spec: EvalSpec,
    *,
    season: str,
    from_gw: int,
    to_gw: int,
    models_dir: Path | None,
    warnings: list[str],
) -> str:
    base = (
        "uv run fplt eval-components "
        f"--season {season} --from-gw {from_gw} --to-gw {to_gw}"
    )
    parts = [base]
    model_names = load_model_specs(models_dir)
    control = model_names.get(spec.control)
    treatment = model_names.get(spec.treatment)
    slot = spec.change.slot
    names = (
        slot_component_names(control, treatment, slot)
        if control is not None and treatment is not None
        else None
    )
    if names is not None and slot in _SLOT_CLI_FLAGS:
        flag = _SLOT_CLI_FLAGS[slot]
        parts.append(f"{flag} {names[0]} {names[1]}")
    elif suites_needs_slot_flags(spec):
        warnings.append(
            "eval-components omitted arm-specific model flags "
            "(unresolved control/treatment slot names)"
        )
    if not is_n_sims_lever(spec):
        parts.append("--skip-sims")
    return " ".join(parts)


def suites_needs_slot_flags(spec: EvalSpec) -> bool:
    """Whether the components suite should name the two slot specs."""
    return spec.suites.components and spec.change.slot in _SLOT_CLI_FLAGS


def _run_components(
    spec: EvalSpec,
    plan: PromoteEvalPlan,
    preflight: PreflightResult,
    *,
    data_dir: Path,
    output_dir: Path,
    models_dir: Path | None,
) -> tuple[ComponentEvalResult | None, pl.DataFrame | None, dict[str, Any] | None]:
    control = preflight.control_spec
    treatment = preflight.treatment_spec
    if control is None or treatment is None:
        specs = load_model_specs(models_dir)
        control = specs.get(plan.control)
        treatment = specs.get(plan.treatment)
    if control is None or treatment is None:
        logger.warning("skipping components suite: unresolved arm ModelSpecs")
        return None, None, None

    slot = spec.change.slot
    names = slot_component_names(control, treatment, slot)
    score_models: list[str] = []
    minutes_models: list[str] = []
    involvement_models: list[str] = []
    extra_score: list[ModelSpec] = []
    scoreline = is_scoreline_lever(spec)
    if names is not None and slot == "score":
        score_models = [names[0], names[1]]
    elif names is not None and slot == "minutes":
        minutes_models = [names[0], names[1]]
    elif names is not None and slot == "involvement":
        involvement_models = [names[0], names[1]]
    if scoreline:
        extra_score.append(raw_de_vig_spec(coverage_read=spec.gates.coverage.read))

    if (
        not score_models
        and not minutes_models
        and not involvement_models
        and not extra_score
    ):
        logger.warning("skipping components suite: no slot models resolved")
        return None, None, None

    sims: list[int] | None = None if is_n_sims_lever(spec) else []
    results: list[ComponentEvalResult] = []
    for season in plan.seasons:
        result = run_component_eval(
            data_dir,
            season,
            from_gw=plan.from_gw,
            to_gw=plan.to_gw,
            score_models=score_models,
            minutes_model=minutes_models,
            involvement_model=involvement_models,
            layered_model=plan.treatment,
            sims=sims,
            output_dir=output_dir / season,
            write_per_match=scoreline,
            extra_score_specs=extra_score or None,
        )
        results.append(result)

    summary = pl.concat([part.summary for part in results], how="vertical_relaxed")
    per_match_parts = [
        part.per_match for part in results if not part.per_match.is_empty()
    ]
    per_match = (
        pl.concat(per_match_parts, how="vertical_relaxed") if per_match_parts else None
    )
    deltas = _component_deltas(summary, names) if names is not None else None
    combined = ComponentEvalResult(
        summary=summary,
        sims=results[0].sims,
        output_dir=output_dir,
        per_match=per_match if per_match is not None else results[0].per_match,
    )
    return combined, per_match, deltas


def _component_deltas(summary: pl.DataFrame, names: tuple[str, str]) -> dict[str, Any]:
    if summary.is_empty():
        return {}
    control_name, treatment_name = names
    deltas: dict[str, Any] = {}
    metrics = summary["metric"].unique().to_list()
    for metric in metrics:
        c = summary.filter(
            (pl.col("model_name") == control_name) & (pl.col("metric") == metric)
        )
        t = summary.filter(
            (pl.col("model_name") == treatment_name) & (pl.col("metric") == metric)
        )
        if c.is_empty() or t.is_empty():
            continue
        c_val = c["value"][0]
        t_val = t["value"][0]
        if not isinstance(c_val, int | float) or not isinstance(t_val, int | float):
            continue
        if isinstance(c_val, bool) or isinstance(t_val, bool):
            continue
        # Lower is better for RPS / logloss / Brier / MAE.
        deltas[f"{metric}_delta"] = float(c_val) - float(t_val)
    return deltas


def _pooled_arm_metric(per_match: pl.DataFrame, arm: str, metric: str) -> float | None:
    part = per_match.filter((pl.col("arm") == arm) & (pl.col("metric") == metric))
    if part.is_empty():
        return None
    value = part["value"].mean()
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, int | float):
        return float(value)
    return None


__all__ = [
    "ChangeSpec",
    "CoverageResult",
    "CoverageSpec",
    "EvalSpec",
    "GatesSpec",
    "PreflightResult",
    "PromoteEvalPlan",
    "PromoteEvalResult",
    "ReportSpec",
    "SuitesSpec",
    "WindowSpec",
    "compare_arm_names",
    "default_evals_dir",
    "effective_param",
    "execute_promote_eval",
    "format_plan",
    "git_is_dirty",
    "is_effectively_on",
    "is_equivalence_kind",
    "is_n_sims_lever",
    "is_scoreline_lever",
    "load_eval_spec",
    "plan_promote_eval",
    "plan_to_json",
    "preflight_promote_eval",
    "raw_de_vig_spec",
    "render_recommendation",
    "repo_root",
    "slot_component_names",
    "uv_lock_sha256",
]
