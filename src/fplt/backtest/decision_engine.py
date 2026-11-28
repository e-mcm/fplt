"""Experiment decision engine (toolkit §2.5 / experiments §9 step 5).

Superiority, non-inferiority and equivalence tests on paired per-GW
differences reuse ``fplt.backtest.intervals`` (HAC / DM + stationary
bootstrap). Gates freeze at submit; a clear component-stage failure
short-circuits downstream stages; the holdout season is re-tested only
for the selected arm.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal, cast

from pydantic import BaseModel, ConfigDict, Field, model_validator

from fplt.backtest.intervals import (
    PairedInterval,
    interval_below_threshold,
    interval_clears_threshold,
    interval_inside_band,
    interval_to_dict,
    paired_interval,
)
from fplt.schemas.manifests import ExperimentManifest, ExperimentVerdict
from fplt.store.experiments import frozen_gates

DecisionType = Literal["none", "superiority", "non_inferiority", "equivalence"]
WindowRole = Literal["primary", "confirm", "holdout"]
VerdictLabel = Literal["pass", "fail", "inconclusive", "none"]
StageStatus = Literal["pass", "fail_clear", "inconclusive", "skipped"]

# Standing holdout per fplt-experiments.md §2 rule 4 (rolls forward when a
# season completes; submitted experiments keep the holdout frozen on the
# manifest).
DEFAULT_HOLDOUT_SEASON = "2025-26"

_SELECTION_WINDOWS: tuple[WindowRole, ...] = ("primary", "confirm")


class PrimaryMetric(BaseModel):
    """Primary metric for the experiment verdict."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    window: WindowRole = "holdout"


class GateSpec(BaseModel):
    """One stage gate evaluated before (or instead of) the primary metric.

    ``rule`` is retained for reporting (e.g. ``diff <= 0`` on a loss metric).
    Clear-failure short-circuit uses the decision type and margin on the
    oriented interval (positive ⇒ treatment better).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: str = Field(min_length=1)
    metric: str = Field(min_length=1)
    rule: str | None = None
    short_circuit: bool = True
    window: WindowRole = "primary"


class DecisionSpec(BaseModel):
    """Decision block frozen into the experiment manifest at submit.

    Matches the YAML shape in ``fplt-experiments.md`` §6.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    type: DecisionType = "none"
    primary: PrimaryMetric | None = None
    gates: list[GateSpec] = Field(default_factory=list[GateSpec])
    alpha: float = Field(default=0.05, gt=0.0, lt=1.0)
    margin: float | None = Field(default=None, gt=0.0)
    exclude_from_winner: list[str] = Field(default_factory=list[str])
    short_circuit: bool = True

    @model_validator(mode="after")
    def _margin_required(self) -> DecisionSpec:
        if self.type in ("non_inferiority", "equivalence") and self.margin is None:
            raise ValueError(f"{self.type} requires a positive margin")
        if self.type != "none" and self.primary is None:
            raise ValueError(f"{self.type} requires primary metric")
        return self


@dataclass(frozen=True)
class TestResult:
    """Pass / fail / inconclusive for one paired series under a decision type."""

    decision_type: DecisionType
    verdict: VerdictLabel
    clear_failure: bool
    interval: PairedInterval
    threshold: float | None
    margin: float | None
    detail: dict[str, Any]


@dataclass(frozen=True)
class StageOutcome:
    """One measured stage after gate / short-circuit evaluation."""

    stage: str
    status: StageStatus
    reason: str | None
    gate: GateSpec | None
    test: TestResult | None


@dataclass(frozen=True)
class DecisionResult:
    """Full engine readout for one experiment analysis pass."""

    decision: DecisionSpec
    overall: VerdictLabel
    selected_arm: str | None
    holdout_season: str | None
    holdout_arms: tuple[str, ...]
    stages: tuple[StageOutcome, ...]
    primary: TestResult | None
    gates_frozen: dict[str, Any]


def freeze_decision(decision: DecisionSpec) -> dict[str, Any]:
    """Serialize a ``DecisionSpec`` for the manifest ``gates`` field."""
    return decision.model_dump(mode="json")


def load_frozen_decision(gates: Mapping[str, Any]) -> DecisionSpec:
    """Parse a frozen decision dict from the experiment manifest.

    Accepts either a bare ``DecisionSpec`` dump or a nested
    ``{"decision": {...}}`` wrapper so promote-eval gate dicts can coexist.
    """
    payload: Mapping[str, Any]
    if "type" in gates:
        payload = gates
    elif "decision" in gates and isinstance(gates["decision"], Mapping):
        payload = cast(Mapping[str, Any], gates["decision"])
    else:
        raise ValueError(
            "frozen gates must contain a DecisionSpec dump or a 'decision' key"
        )
    return DecisionSpec.model_validate(dict(payload))


def standing_holdout_season(
    *,
    complete_seasons: Sequence[str] | None = None,
) -> str:
    """Return the standing holdout season (latest complete season).

    Args:
        complete_seasons: Optional ordered or unordered complete-season labels.
            When provided, the lexicographically latest label wins (season
            labels sort chronologically as ``YYYY-YY``). When omitted, returns
            ``DEFAULT_HOLDOUT_SEASON``.

    Returns:
        Holdout season label.
    """
    if not complete_seasons:
        return DEFAULT_HOLDOUT_SEASON
    cleaned = [s.strip() for s in complete_seasons if s and s.strip()]
    if not cleaned:
        return DEFAULT_HOLDOUT_SEASON
    return max(cleaned)


def select_arm(
    arm_scores: Mapping[str, float],
    *,
    exclude: Sequence[str] = (),
    control_arm: str = "control",
) -> str:
    """Pick the best non-excluded, non-control arm on selection windows.

    Scores are oriented (higher ⇒ better). Ties break by arm name.

    Args:
        arm_scores: Point estimates (e.g. mean paired Δ) per treatment arm.
        exclude: Arms barred from winning (e.g. oracle).
        control_arm: Control label, never selected as the winner.

    Returns:
        Selected treatment arm name.

    Raises:
        ValueError: No eligible arm remains.
    """
    blocked = {control_arm, *exclude}
    eligible = [
        (score, name) for name, score in arm_scores.items() if name not in blocked
    ]
    if not eligible:
        raise ValueError("no eligible treatment arm to select")
    eligible.sort(key=lambda item: (-item[0], item[1]))
    return eligible[0][1]


def holdout_arms_to_test(
    selected_arm: str,
    *,
    control_arm: str = "control",
) -> tuple[str, ...]:
    """Return arms whose holdout series must be computed.

    Multiple-comparison guard: only the selected arm (plus control for
    pairing) is re-tested on the holdout.
    """
    return tuple(sorted({control_arm, selected_arm}))


def evaluate_paired_series(
    d_t: Sequence[float],
    *,
    decision_type: DecisionType,
    metric: str,
    alpha: float = 0.05,
    margin: float | None = None,
    bootstrap_reps: int = 500,
    bootstrap_seed: int = 0,
) -> TestResult:
    """Run one superiority / non-inferiority / equivalence test on ``d_t``.

    ``d_t`` is signed so a positive mean means the treatment is better.
    Intervals reuse HAC OLS + stationary bootstrap.

    Args:
        d_t: Paired per-GW differentials.
        decision_type: Test family (``none`` returns descriptive only).
        metric: Metric label on the interval.
        alpha: Two-sided type-I rate for CIs.
        margin: Required for non-inferiority / equivalence.
        bootstrap_reps: Bootstrap replications.
        bootstrap_seed: Bootstrap seed.

    Returns:
        ``TestResult`` with verdict and clear-failure flag.
    """
    interval = paired_interval(
        list(d_t),
        metric=metric,
        bootstrap_reps=bootstrap_reps,
        bootstrap_seed=bootstrap_seed,
        alpha=alpha,
    )
    if decision_type == "none":
        return TestResult(
            decision_type=decision_type,
            verdict="none",
            clear_failure=False,
            interval=interval,
            threshold=None,
            margin=None,
            detail=interval_to_dict(interval),
        )

    bounds = (
        interval.hac_ci_low,
        interval.hac_ci_high,
        interval.bootstrap_ci_low,
        interval.bootstrap_ci_high,
    )
    if decision_type == "superiority":
        passed = interval_clears_threshold(*bounds, threshold=0.0)
        clear_failure = interval_below_threshold(*bounds, threshold=0.0)
        threshold: float | None = 0.0
        used_margin: float | None = None
    elif decision_type == "non_inferiority":
        if margin is None:
            raise ValueError("non_inferiority requires margin")
        threshold = -margin
        passed = interval_clears_threshold(*bounds, threshold=threshold)
        clear_failure = interval_below_threshold(*bounds, threshold=threshold)
        used_margin = margin
    elif decision_type == "equivalence":
        if margin is None:
            raise ValueError("equivalence requires margin")
        threshold = None
        passed = interval_inside_band(*bounds, margin=margin)
        clear_failure = interval_below_threshold(
            *bounds, threshold=-margin
        ) or interval_clears_threshold(*bounds, threshold=margin)
        used_margin = margin
    else:
        raise ValueError(f"unknown decision type {decision_type!r}")

    if passed:
        verdict: VerdictLabel = "pass"
    elif clear_failure:
        verdict = "fail"
    else:
        verdict = "inconclusive"

    return TestResult(
        decision_type=decision_type,
        verdict=verdict,
        clear_failure=clear_failure,
        interval=interval,
        threshold=threshold,
        margin=used_margin,
        detail={
            **interval_to_dict(interval),
            "threshold": threshold,
            "margin": used_margin,
            "clear_failure": clear_failure,
        },
    )


def evaluate_gated_stages(
    stages: Sequence[str],
    intervals: Mapping[str, Sequence[float]],
    decision: DecisionSpec,
    *,
    bootstrap_reps: int = 500,
    bootstrap_seed: int = 0,
) -> tuple[StageOutcome, ...]:
    """Evaluate stages in order, short-circuiting on clear gate failure.

    Args:
        stages: Pipeline stages in measurement order (cheap first).
        intervals: Oriented per-GW ``d_t`` series keyed by stage (gate metric).
        decision: Frozen decision spec (gates + short-circuit flag).
        bootstrap_reps: Bootstrap replications for gate intervals.
        bootstrap_seed: Bootstrap seed.

    Returns:
        One ``StageOutcome`` per requested stage.
    """
    gate_by_stage = {gate.stage: gate for gate in decision.gates}
    outcomes: list[StageOutcome] = []
    skip_rest = False
    skip_reason: str | None = None

    for stage in stages:
        if skip_rest:
            outcomes.append(
                StageOutcome(
                    stage=stage,
                    status="skipped",
                    reason=skip_reason,
                    gate=gate_by_stage.get(stage),
                    test=None,
                )
            )
            continue

        gate = gate_by_stage.get(stage)
        series = intervals.get(stage)
        if series is None:
            outcomes.append(
                StageOutcome(
                    stage=stage,
                    status="inconclusive",
                    reason="missing paired series",
                    gate=gate,
                    test=None,
                )
            )
            continue

        # Gates inherit the decision type / margin so a component Brier gate
        # under a season-points superiority experiment still short-circuits
        # when the CI lies entirely on the wrong side of zero (or margin).
        test = evaluate_paired_series(
            series,
            decision_type=(decision.type if decision.type != "none" else "superiority"),
            metric=gate.metric if gate is not None else "diff",
            alpha=decision.alpha,
            margin=decision.margin,
            bootstrap_reps=bootstrap_reps,
            bootstrap_seed=bootstrap_seed,
        )
        if test.clear_failure:
            status: StageStatus = "fail_clear"
        elif test.verdict == "pass":
            status = "pass"
        else:
            status = "inconclusive"

        reason = None
        if status == "fail_clear":
            reason = (
                f"CI entirely on the wrong side of "
                f"{'margin' if decision.margin is not None else 'zero'}"
            )
            sc = decision.short_circuit and (gate is None or gate.short_circuit)
            if sc:
                skip_rest = True
                skip_reason = f"short-circuit: {stage} clear failure"

        outcomes.append(
            StageOutcome(
                stage=stage,
                status=status,
                reason=reason,
                gate=gate,
                test=test,
            )
        )

    return tuple(outcomes)


def evaluate_decision(
    decision: DecisionSpec,
    *,
    selection_scores: Mapping[str, float] | None = None,
    holdout_series: Mapping[str, Sequence[float]] | None = None,
    stage_series: Mapping[str, Sequence[float]] | None = None,
    stages: Sequence[str] | None = None,
    holdout_season: str | None = None,
    control_arm: str = "control",
    bootstrap_reps: int = 500,
    bootstrap_seed: int = 0,
    gates_frozen: Mapping[str, Any] | None = None,
) -> DecisionResult:
    """Compute the experiment verdict from frozen gates and paired series.

    Selection uses primary/confirm scores only. The holdout series map should
    contain only the arms returned by ``holdout_arms_to_test`` (control +
    selected); extra arms are ignored for the reported verdict.

    Args:
        decision: Decision rule (prefer the frozen copy from the manifest).
        selection_scores: Oriented scores per treatment arm on selection
            windows (required when ``type != none`` and multiple arms).
        holdout_series: Oriented per-GW ``d_t`` for ``control`` vs each arm
            on the holdout window, keyed by treatment arm.
        stage_series: Oriented per-GW series keyed by stage for gates /
            short-circuit.
        stages: Ordered stage names to walk (defaults to gate stages then
            primary stage).
        holdout_season: Holdout label recorded on the result.
        control_arm: Control arm name.
        bootstrap_reps: Bootstrap replications.
        bootstrap_seed: Bootstrap seed.
        gates_frozen: Snapshot recorded on the result (defaults to a fresh
            freeze of ``decision``).

    Returns:
        ``DecisionResult`` with overall verdict and stage outcomes.
    """
    if gates_frozen is not None:
        frozen = dict(gates_frozen)
    else:
        frozen = freeze_decision(decision)
    if decision.type == "none":
        return DecisionResult(
            decision=decision,
            overall="none",
            selected_arm=None,
            holdout_season=holdout_season,
            holdout_arms=(),
            stages=(),
            primary=None,
            gates_frozen=frozen,
        )

    selected: str | None = None
    holdout_arms: tuple[str, ...] = ()
    if selection_scores is not None:
        selected = select_arm(
            selection_scores,
            exclude=decision.exclude_from_winner,
            control_arm=control_arm,
        )
        holdout_arms = holdout_arms_to_test(selected, control_arm=control_arm)

    stage_order = list(stages) if stages is not None else _default_stages(decision)
    stage_outcomes = evaluate_gated_stages(
        stage_order,
        stage_series or {},
        decision,
        bootstrap_reps=bootstrap_reps,
        bootstrap_seed=bootstrap_seed,
    )
    if any(o.status == "skipped" for o in stage_outcomes):
        return DecisionResult(
            decision=decision,
            overall="fail",
            selected_arm=selected,
            holdout_season=holdout_season,
            holdout_arms=holdout_arms,
            stages=stage_outcomes,
            primary=None,
            gates_frozen=frozen,
        )

    primary_test: TestResult | None = None
    if (
        decision.primary is not None
        and decision.primary.window == "holdout"
        and selected is not None
        and holdout_series is not None
    ):
        series = holdout_series.get(selected)
        if series is None:
            primary_test = None
            overall: VerdictLabel = "inconclusive"
        else:
            primary_test = evaluate_paired_series(
                series,
                decision_type=decision.type,
                metric=decision.primary.metric,
                alpha=decision.alpha,
                margin=decision.margin,
                bootstrap_reps=bootstrap_reps,
                bootstrap_seed=bootstrap_seed,
            )
            overall = primary_test.verdict
    elif decision.primary is not None and stage_series is not None:
        series = stage_series.get(decision.primary.stage)
        if series is None:
            overall = "inconclusive"
        else:
            primary_test = evaluate_paired_series(
                series,
                decision_type=decision.type,
                metric=decision.primary.metric,
                alpha=decision.alpha,
                margin=decision.margin,
                bootstrap_reps=bootstrap_reps,
                bootstrap_seed=bootstrap_seed,
            )
            overall = primary_test.verdict
    else:
        overall = "inconclusive"

    return DecisionResult(
        decision=decision,
        overall=overall,
        selected_arm=selected,
        holdout_season=holdout_season,
        holdout_arms=holdout_arms,
        stages=stage_outcomes,
        primary=primary_test,
        gates_frozen=frozen,
    )


def verdict_from_manifest(
    manifest: ExperimentManifest,
    *,
    selection_scores: Mapping[str, float] | None = None,
    holdout_series: Mapping[str, Sequence[float]] | None = None,
    stage_series: Mapping[str, Sequence[float]] | None = None,
    stages: Sequence[str] | None = None,
    holdout_season: str | None = None,
    control_arm: str = "control",
    bootstrap_reps: int = 500,
    bootstrap_seed: int = 0,
    computed_at: datetime | None = None,
) -> tuple[DecisionResult, ExperimentVerdict]:
    """Evaluate using gates frozen on the manifest (ignore live YAML edits).

    Args:
        manifest: Submitted experiment (gates immutable).
        selection_scores: Selection-window scores per treatment arm.
        holdout_series: Holdout ``d_t`` keyed by treatment arm.
        stage_series: Stage gate series.
        stages: Ordered stages.
        holdout_season: Override holdout label (default from windows).
        control_arm: Control arm name.
        bootstrap_reps: Bootstrap replications.
        bootstrap_seed: Bootstrap seed.
        computed_at: Timestamp for ``ExperimentVerdict``.

    Returns:
        Engine result plus an ``ExperimentVerdict`` ready for
        ``results/verdict.json``.
    """
    gates = frozen_gates(manifest)
    decision = load_frozen_decision(gates)
    season = holdout_season
    if season is None:
        for window in manifest.windows:
            if window.role == "holdout" and window.seasons:
                season = window.seasons[-1]
                break
    result = evaluate_decision(
        decision,
        selection_scores=selection_scores,
        holdout_series=holdout_series,
        stage_series=stage_series,
        stages=stages,
        holdout_season=season,
        control_arm=control_arm,
        bootstrap_reps=bootstrap_reps,
        bootstrap_seed=bootstrap_seed,
        gates_frozen=gates,
    )
    when = computed_at if computed_at is not None else datetime.now(UTC)
    payload = ExperimentVerdict(
        overall=result.overall,
        gates=[
            {
                "stage": outcome.stage,
                "status": outcome.status,
                "reason": outcome.reason,
                "metric": outcome.gate.metric if outcome.gate else None,
                "test": None
                if outcome.test is None
                else {
                    "verdict": outcome.test.verdict,
                    "clear_failure": outcome.test.clear_failure,
                    "mean": outcome.test.interval.mean,
                    "hac_ci": [
                        outcome.test.interval.hac_ci_low,
                        outcome.test.interval.hac_ci_high,
                    ],
                    "bootstrap_ci": [
                        outcome.test.interval.bootstrap_ci_low,
                        outcome.test.interval.bootstrap_ci_high,
                    ],
                },
            }
            for outcome in result.stages
        ],
        computed_at=when,
        eval_spec_hash=manifest.eval_spec_hash,
    )
    return result, payload


def selection_window_roles() -> tuple[WindowRole, ...]:
    """Window roles used for arm selection (never the holdout)."""
    return _SELECTION_WINDOWS


def _default_stages(decision: DecisionSpec) -> list[str]:
    stages: list[str] = []
    for gate in decision.gates:
        if gate.stage not in stages:
            stages.append(gate.stage)
    if decision.primary is not None and decision.primary.stage not in stages:
        stages.append(decision.primary.stage)
    return stages


__all__ = [
    "DEFAULT_HOLDOUT_SEASON",
    "DecisionResult",
    "DecisionSpec",
    "DecisionType",
    "GateSpec",
    "PrimaryMetric",
    "StageOutcome",
    "TestResult",
    "VerdictLabel",
    "WindowRole",
    "evaluate_decision",
    "evaluate_gated_stages",
    "evaluate_paired_series",
    "freeze_decision",
    "holdout_arms_to_test",
    "load_frozen_decision",
    "select_arm",
    "selection_window_roles",
    "standing_holdout_season",
    "verdict_from_manifest",
]
