"""Standard A/B promote-eval: control vs treatment ModelSpec.

Any lever — new staged data, feature version, hyperparam fork, layered
component registry swap, or full predictor fork — is expressed as two YAML
specs plus an ``EvalSpec``. v1 of this module loads the EvalSpec and builds a
dry-run plan; later phases call ``compare`` / ``eval-components`` /
``decisions``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fplt.models.spec import load_model_specs
from fplt.optimise.plan import DEFAULT_PLAN_HORIZON
from fplt.optimise.realised import DEFAULT_MAX_MEAN_PPG

EvalKind = Literal["data", "feature", "config", "component", "model"]
Lever = Literal[
    "data_table",
    "feature_version",
    "hyperparam",
    "component_swap",
    "model_fork",
]
Tier = Literal["S0", "S1", "S2", "S3", "S4"]
CompareModeName = Literal["walk_forward", "fixed"]

_TIER_TO_GW: dict[Tier, int] = {
    "S0": 1,
    "S1": 10,
    "S2": 20,
    "S3": 10,
    "S4": 38,
}


class ChangeSpec(BaseModel):
    """What differs between control and treatment (one lever)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    lever: Lever
    slot: str | None = None
    control_kind: str | None = None
    treatment_kind: str | None = None
    tables: list[str] = Field(default_factory=list)
    features: list[str] = Field(default_factory=list)
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


class GatesSpec(BaseModel):
    """Coverage and keep/drop hints (enforced in later phases)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    min_coverage: float | None = Field(default=None, ge=0.0, le=1.0)
    allow_low_coverage: bool = False
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
    description: str = ""
    change: ChangeSpec
    control: str
    treatment: str
    window: WindowSpec = Field(default_factory=WindowSpec)
    suites: SuitesSpec = Field(default_factory=SuitesSpec)
    gates: GatesSpec = Field(default_factory=GatesSpec)
    report: ReportSpec = Field(default_factory=ReportSpec)

    @model_validator(mode="after")
    def _kind_lever_align(self) -> EvalSpec:
        expected: dict[EvalKind, set[Lever]] = {
            "data": {"data_table"},
            "feature": {"feature_version"},
            "config": {"hyperparam"},
            "component": {"component_swap"},
            "model": {"model_fork"},
        }
        allowed = expected[self.kind]
        if self.change.lever not in allowed:
            raise ValueError(
                f"kind {self.kind!r} expects change.lever in {sorted(allowed)}, "
                f"got {self.change.lever!r}"
            )
        if self.control == self.treatment:
            raise ValueError("control and treatment must differ")
        return self


class PromoteEvalPlan(BaseModel):
    """Resolved dry-run plan for one EvalSpec."""

    model_config = ConfigDict(frozen=True)

    eval_name: str
    kind: EvalKind
    lever: Lever
    control: str
    treatment: str
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
        dry_run: When true, plan only (v1 always plans; runner comes later).
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
    if start > end:
        raise ValueError(f"from_gw {start} > to_gw {end}")

    model_names = set(load_model_specs(models_dir))
    for arm in (spec.control, spec.treatment):
        if arm not in model_names:
            warnings.append(f"ModelSpec {arm!r} not found under configs/models")

    commands = _commands_for(
        spec,
        seasons=season_list,
        from_gw=start,
        to_gw=end,
        suites=suites,
        modes=list(window.modes),
        seasons_back=window.seasons_back,
        fit_gw=window.fit_gw,
    )
    if not dry_run:
        warnings.append(
            "execute path not implemented yet; use --dry-run or run commands manually"
        )

    return PromoteEvalPlan(
        eval_name=spec.name,
        kind=spec.kind,
        lever=spec.change.lever,
        control=spec.control,
        treatment=spec.treatment,
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
        f"  seasons={','.join(plan.seasons)}  gw={plan.from_gw}-{plan.to_gw}  "
        f"seasons_back={plan.seasons_back}  modes={','.join(plan.modes)}",
        (
            f"  suites: prediction={plan.suites.prediction} "
            f"components={plan.suites.components} "
            f"decisions={plan.suites.decisions}"
        ),
    ]
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
) -> list[str]:
    commands: list[str] = []
    arms = f"{spec.control} {spec.treatment}"
    for season in seasons:
        if suites.components:
            commands.append(
                "uv run fplt eval-components "
                f"--season {season} --from-gw {from_gw} --to-gw {to_gw}"
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


__all__ = [
    "ChangeSpec",
    "EvalSpec",
    "GatesSpec",
    "PromoteEvalPlan",
    "ReportSpec",
    "SuitesSpec",
    "WindowSpec",
    "default_evals_dir",
    "format_plan",
    "load_eval_spec",
    "plan_promote_eval",
    "plan_to_json",
]
