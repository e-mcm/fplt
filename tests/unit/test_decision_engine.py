"""Decision engine: superiority / non-inferiority / equivalence (toolkit §2.5)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from pydantic import ValidationError

from fplt.backtest.decision_engine import (
    DEFAULT_HOLDOUT_SEASON,
    DecisionSpec,
    GateSpec,
    PrimaryMetric,
    evaluate_decision,
    evaluate_gated_stages,
    evaluate_paired_series,
    freeze_decision,
    holdout_arms_to_test,
    load_frozen_decision,
    select_arm,
    selection_window_roles,
    standing_holdout_season,
    verdict_from_manifest,
)
from fplt.backtest.intervals import (
    interval_below_threshold,
    interval_clears_threshold,
    interval_inside_band,
    paired_interval,
)
from fplt.schemas.manifests import ExperimentArm, ExperimentWindow
from fplt.store.experiments import frozen_gates, submit_experiment

_TS = datetime(2026, 11, 29, 4, 0, 0, tzinfo=UTC)
_BOOT = {"bootstrap_reps": 300, "bootstrap_seed": 0}


def _effect_series(n: int = 24, mean: float = 0.4, noise: float = 0.05) -> list[float]:
    rng = np.random.default_rng(0)
    return (mean + noise * rng.standard_normal(n)).tolist()


def _null_series(n: int = 24, noise: float = 0.05) -> list[float]:
    return _effect_series(n=n, mean=0.0, noise=noise)


def test_interval_threshold_helpers() -> None:
    assert interval_clears_threshold(0.1, 0.5, 0.2, 0.6, threshold=0.0) is True
    assert interval_clears_threshold(-0.1, 0.5, 0.2, 0.6, threshold=0.0) is False
    assert interval_clears_threshold(None, None, None, None, threshold=0.0) is False
    assert interval_below_threshold(-0.5, -0.1, -0.4, -0.05, threshold=0.0) is True
    assert interval_inside_band(-0.2, 0.2, -0.1, 0.1, margin=0.5) is True
    assert interval_inside_band(-0.2, 0.2, -0.1, 0.6, margin=0.5) is False
    with pytest.raises(ValueError, match="margin"):
        interval_inside_band(-0.1, 0.1, -0.1, 0.1, margin=0.0)


def test_paired_interval_alpha_and_invalid() -> None:
    d = _effect_series(n=24, mean=0.2, noise=0.15)
    # Smaller alpha → wider CI (99% vs 80%).
    wide = paired_interval(d, metric="x", alpha=0.01, bootstrap_reps=200)
    tight = paired_interval(d, metric="x", alpha=0.20, bootstrap_reps=200)
    assert wide.hac_ci_low is not None
    assert tight.hac_ci_low is not None
    assert wide.hac_ci_high is not None
    assert tight.hac_ci_high is not None
    assert (wide.hac_ci_high - wide.hac_ci_low) > (tight.hac_ci_high - tight.hac_ci_low)
    with pytest.raises(ValueError, match="alpha"):
        paired_interval(d, metric="x", alpha=0.0)


def test_superiority_effect_and_null() -> None:
    effect = evaluate_paired_series(
        _effect_series(),
        decision_type="superiority",
        metric="season_points",
        **_BOOT,
    )
    null = evaluate_paired_series(
        _null_series(),
        decision_type="superiority",
        metric="season_points",
        **_BOOT,
    )
    assert effect.verdict == "pass"
    assert effect.clear_failure is False
    assert effect.interval.mean > 0
    assert null.verdict in {"inconclusive", "fail"}
    assert null.verdict != "pass"


def test_non_inferiority_effect_and_null() -> None:
    margin = 0.15
    # Slightly negative but within margin → non-inferior.
    ok = evaluate_paired_series(
        _effect_series(mean=-0.05, noise=0.02),
        decision_type="non_inferiority",
        metric="rmse",
        margin=margin,
        **_BOOT,
    )
    # Clearly worse than -margin → fail.
    bad = evaluate_paired_series(
        _effect_series(mean=-0.5, noise=0.02),
        decision_type="non_inferiority",
        metric="rmse",
        margin=margin,
        **_BOOT,
    )
    # Near-zero with a tiny margin and wide noise → not a clear pass.
    null = evaluate_paired_series(
        _null_series(noise=0.25),
        decision_type="non_inferiority",
        metric="rmse",
        margin=0.01,
        **_BOOT,
    )
    assert ok.verdict == "pass"
    assert bad.verdict == "fail"
    assert bad.clear_failure is True
    assert null.verdict != "pass"


def test_equivalence_effect_and_null() -> None:
    margin = 0.25
    inside = evaluate_paired_series(
        _effect_series(mean=0.02, noise=0.02),
        decision_type="equivalence",
        metric="delta_xpts",
        margin=margin,
        **_BOOT,
    )
    outside = evaluate_paired_series(
        _effect_series(mean=0.6, noise=0.02),
        decision_type="equivalence",
        metric="delta_xpts",
        margin=margin,
        **_BOOT,
    )
    null_wide = evaluate_paired_series(
        _null_series(noise=0.4),
        decision_type="equivalence",
        metric="delta_xpts",
        margin=0.05,
        **_BOOT,
    )
    assert inside.verdict == "pass"
    assert outside.verdict == "fail"
    assert outside.clear_failure is True
    assert null_wide.verdict in {"inconclusive", "fail"}


def test_decision_spec_requires_margin_and_primary() -> None:
    primary = PrimaryMetric(stage="s", metric="m")
    with pytest.raises(ValidationError):
        DecisionSpec(type="non_inferiority", primary=primary)
    with pytest.raises(ValidationError):
        DecisionSpec(type="superiority")
    with pytest.raises(ValidationError):
        DecisionSpec(type="equivalence", primary=primary, margin=0.0)


def test_short_circuit_skips_downstream() -> None:
    decision = DecisionSpec(
        type="superiority",
        primary=PrimaryMetric(stage="season", metric="season_points"),
        gates=[
            GateSpec(
                stage="component:minutes",
                metric="brier_60",
                rule="diff <= 0",
                short_circuit=True,
            )
        ],
        short_circuit=True,
    )
    stages = evaluate_gated_stages(
        ["component:minutes", "prediction", "season"],
        {
            "component:minutes": _effect_series(mean=-0.5, noise=0.02),
            "prediction": _effect_series(mean=0.4),
            "season": _effect_series(mean=0.4),
        },
        decision,
        **_BOOT,
    )
    assert stages[0].status == "fail_clear"
    assert stages[1].status == "skipped"
    assert stages[2].status == "skipped"
    assert stages[1].reason is not None
    assert "short-circuit" in stages[1].reason

    # --no-short-circuit forces every stage.
    no_sc = decision.model_copy(update={"short_circuit": False})
    forced = evaluate_gated_stages(
        ["component:minutes", "prediction", "season"],
        {
            "component:minutes": _effect_series(mean=-0.5, noise=0.02),
            "prediction": _effect_series(mean=0.4),
            "season": _effect_series(mean=0.4),
        },
        no_sc,
        **_BOOT,
    )
    assert forced[0].status == "fail_clear"
    assert forced[1].status == "pass"
    assert forced[2].status == "pass"


def test_holdout_only_for_selected_arm() -> None:
    assert selection_window_roles() == ("primary", "confirm")
    assert standing_holdout_season() == DEFAULT_HOLDOUT_SEASON
    assert standing_holdout_season(complete_seasons=["2023-24", "2024-25"]) == "2024-25"

    selected = select_arm(
        {"blend_07": 0.1, "blend_09": 0.3, "oracle": 0.9},
        exclude=["oracle"],
    )
    assert selected == "blend_09"
    arms = holdout_arms_to_test(selected)
    assert arms == ("blend_09", "control")
    assert "blend_07" not in arms
    assert "oracle" not in arms

    decision = DecisionSpec(
        type="superiority",
        primary=PrimaryMetric(stage="season", metric="season_points", window="holdout"),
        exclude_from_winner=["oracle"],
        alpha=0.05,
    )
    # Holdout series provided for every arm; engine must only use selected.
    result = evaluate_decision(
        decision,
        selection_scores={"blend_07": 0.1, "blend_09": 0.3, "oracle": 0.9},
        holdout_series={
            "blend_07": _effect_series(mean=-0.5),  # would fail if selected
            "blend_09": _effect_series(mean=0.4),
            "oracle": _effect_series(mean=0.8),
        },
        holdout_season=DEFAULT_HOLDOUT_SEASON,
        **_BOOT,
    )
    assert result.selected_arm == "blend_09"
    assert result.holdout_arms == ("blend_09", "control")
    assert result.overall == "pass"
    assert result.primary is not None
    assert result.primary.interval.mean == pytest.approx(
        float(np.mean(_effect_series(mean=0.4))),
        abs=1e-9,
    )


def test_post_submit_gate_edits_do_not_change_verdict(tmp_path: Path) -> None:
    live = DecisionSpec(
        type="superiority",
        primary=PrimaryMetric(stage="season", metric="season_points", window="holdout"),
        gates=[
            GateSpec(stage="component:minutes", metric="brier_60", short_circuit=True)
        ],
        alpha=0.05,
        short_circuit=True,
    )
    frozen = freeze_decision(live)
    assert load_frozen_decision(frozen).alpha == 0.05
    assert load_frozen_decision({"decision": frozen}).type == "superiority"

    manifest = submit_experiment(
        tmp_path,
        flavour="ab",
        question="Does the blend help?",
        arms=[
            ExperimentArm(
                arm="control",
                role="control",
                spec_hash="a" * 16,
                spec_label="control",
            ),
            ExperimentArm(
                arm="blend_09",
                role="treatment",
                spec_hash="b" * 16,
                spec_label="blend_09",
            ),
        ],
        windows=[
            ExperimentWindow(role="primary", seasons=["2023-24"], from_gw=1, to_gw=38),
            ExperimentWindow(role="confirm", seasons=["2024-25"], from_gw=1, to_gw=38),
            ExperimentWindow(
                role="holdout",
                seasons=[DEFAULT_HOLDOUT_SEASON],
                from_gw=1,
                to_gw=38,
            ),
        ],
        horizon=1,
        code_sha="abc",
        lockfile_hash="c" * 64,
        gates=frozen,
        eval_spec="demo",
        eval_spec_hash="d" * 64,
        submitted_by="test",
        host="test",
        experiment_id="exp_20261129T040000Z_dec01",
        submitted_at=_TS,
    )
    assert frozen_gates(manifest)["alpha"] == 0.05

    series_kwargs = {
        "selection_scores": {"blend_09": 0.3},
        "holdout_series": {"blend_09": _effect_series(mean=0.4)},
        "stage_series": {
            "component:minutes": _effect_series(mean=0.2),
            "season": _effect_series(mean=0.4),
        },
        "stages": ["component:minutes", "season"],
        "bootstrap_reps": 300,
        "bootstrap_seed": 0,
        "computed_at": _TS,
    }
    result_before, verdict_before = verdict_from_manifest(manifest, **series_kwargs)

    # Mutate the live DecisionSpec and a copy of the frozen dict the caller
    # still holds — the manifest on disk must drive the verdict.
    live_mutated = DecisionSpec(
        type="superiority",
        primary=PrimaryMetric(stage="season", metric="season_points", window="holdout"),
        gates=[
            GateSpec(stage="component:minutes", metric="brier_60", short_circuit=True)
        ],
        alpha=0.99,
        short_circuit=False,
    )
    frozen["alpha"] = 0.99
    assert live_mutated.alpha == 0.99
    assert frozen_gates(manifest)["alpha"] == 0.05

    result_after, verdict_after = verdict_from_manifest(manifest, **series_kwargs)
    assert result_before.overall == result_after.overall
    assert result_before.decision.alpha == 0.05
    assert result_after.decision.alpha == 0.05
    assert verdict_before.overall == verdict_after.overall
    assert result_before.gates_frozen["alpha"] == 0.05


def test_evaluate_decision_short_circuit_skips_primary() -> None:
    decision = DecisionSpec(
        type="superiority",
        primary=PrimaryMetric(stage="season", metric="season_points", window="holdout"),
        gates=[GateSpec(stage="component:minutes", metric="brier_60")],
    )
    result = evaluate_decision(
        decision,
        selection_scores={"blend_09": 0.3},
        holdout_series={"blend_09": _effect_series(mean=0.4)},
        stage_series={
            "component:minutes": _effect_series(mean=-0.5, noise=0.02),
            "season": _effect_series(mean=0.4),
        },
        stages=["component:minutes", "season"],
        **_BOOT,
    )
    assert result.overall == "fail"
    assert result.primary is None
    assert result.stages[1].status == "skipped"


def test_select_arm_errors_and_none_decision() -> None:
    with pytest.raises(ValueError, match="no eligible"):
        select_arm({"oracle": 1.0}, exclude=["oracle"])
    none = evaluate_decision(DecisionSpec(type="none"))
    assert none.overall == "none"
    assert none.primary is None


def test_load_frozen_decision_rejects_bad_payload() -> None:
    with pytest.raises(ValueError, match="DecisionSpec"):
        load_frozen_decision({"min_coverage": 0.8})


def test_standing_holdout_blank_seasons() -> None:
    assert (
        standing_holdout_season(complete_seasons=["", "  "]) == DEFAULT_HOLDOUT_SEASON
    )


def test_evaluate_paired_none_and_margin_errors() -> None:
    none = evaluate_paired_series(
        _null_series(),
        decision_type="none",
        metric="look",
        **_BOOT,
    )
    assert none.verdict == "none"
    with pytest.raises(ValueError, match="margin"):
        evaluate_paired_series(
            _null_series(),
            decision_type="non_inferiority",
            metric="m",
            margin=None,
            **_BOOT,
        )
    with pytest.raises(ValueError, match="margin"):
        evaluate_paired_series(
            _null_series(),
            decision_type="equivalence",
            metric="m",
            margin=None,
            **_BOOT,
        )
    with pytest.raises(ValueError, match="unknown"):
        evaluate_paired_series(
            _null_series(),
            decision_type=cast(Any, "not_a_type"),
            metric="m",
            **_BOOT,
        )


def test_gated_stages_missing_series_and_inconclusive() -> None:
    decision = DecisionSpec(
        type="superiority",
        primary=PrimaryMetric(stage="season", metric="season_points"),
        gates=[GateSpec(stage="component:minutes", metric="brier_60")],
    )
    outcomes = evaluate_gated_stages(
        ["component:minutes", "season"],
        {"season": _effect_series(mean=0.05, noise=0.2)},
        decision,
        **_BOOT,
    )
    assert outcomes[0].status == "inconclusive"
    assert outcomes[0].reason == "missing paired series"
    # Noisy near-zero on season → inconclusive (not clear fail / not pass).
    only = evaluate_gated_stages(
        ["season"],
        {"season": _null_series(noise=0.3)},
        decision,
        **_BOOT,
    )
    assert only[0].status == "inconclusive"


def test_gate_short_circuit_false_does_not_skip() -> None:
    decision = DecisionSpec(
        type="superiority",
        primary=PrimaryMetric(stage="season", metric="season_points"),
        gates=[
            GateSpec(
                stage="component:minutes",
                metric="brier_60",
                short_circuit=False,
            )
        ],
        short_circuit=True,
    )
    stages = evaluate_gated_stages(
        ["component:minutes", "season"],
        {
            "component:minutes": _effect_series(mean=-0.5, noise=0.02),
            "season": _effect_series(mean=0.4),
        },
        decision,
        **_BOOT,
    )
    assert stages[0].status == "fail_clear"
    assert stages[1].status == "pass"


def test_evaluate_decision_primary_on_stage_window() -> None:
    decision = DecisionSpec(
        type="superiority",
        primary=PrimaryMetric(stage="season", metric="season_points", window="primary"),
    )
    # Default stages from primary only; no selection / holdout.
    result = evaluate_decision(
        decision,
        stage_series={"season": _effect_series(mean=0.4)},
        **_BOOT,
    )
    assert result.selected_arm is None
    assert result.overall == "pass"
    assert result.primary is not None

    missing = evaluate_decision(
        decision,
        stage_series={"other": _effect_series(mean=0.4)},
        **_BOOT,
    )
    assert missing.overall == "inconclusive"

    no_series = evaluate_decision(decision, **_BOOT)
    assert no_series.overall == "inconclusive"


def test_holdout_missing_selected_series_inconclusive() -> None:
    decision = DecisionSpec(
        type="superiority",
        primary=PrimaryMetric(stage="season", metric="season_points", window="holdout"),
    )
    result = evaluate_decision(
        decision,
        selection_scores={"blend_09": 0.2},
        holdout_series={"other_arm": _effect_series(mean=0.4)},
        **_BOOT,
    )
    assert result.selected_arm == "blend_09"
    assert result.overall == "inconclusive"
    assert result.primary is None


def test_default_stages_dedupes_gate_and_primary() -> None:
    decision = DecisionSpec(
        type="superiority",
        primary=PrimaryMetric(stage="season", metric="season_points"),
        gates=[
            GateSpec(stage="season", metric="season_points"),
            GateSpec(stage="season", metric="season_points"),
        ],
    )
    result = evaluate_decision(
        decision,
        stage_series={"season": _effect_series(mean=0.4)},
        **_BOOT,
    )
    assert [s.stage for s in result.stages] == ["season"]


def test_verdict_from_manifest_without_holdout_window(tmp_path: Path) -> None:
    decision = DecisionSpec(
        type="superiority",
        primary=PrimaryMetric(stage="season", metric="season_points", window="primary"),
    )
    manifest = submit_experiment(
        tmp_path,
        flavour="ab",
        question="No holdout window?",
        arms=[
            ExperimentArm(
                arm="control",
                role="control",
                spec_hash="a" * 16,
                spec_label="control",
            ),
            ExperimentArm(
                arm="t",
                role="treatment",
                spec_hash="b" * 16,
                spec_label="t",
            ),
        ],
        windows=[
            ExperimentWindow(role="primary", seasons=["2023-24"], from_gw=1, to_gw=10)
        ],
        horizon=1,
        code_sha="abc",
        lockfile_hash="c" * 64,
        gates=freeze_decision(decision),
        eval_spec="demo",
        eval_spec_hash="d" * 64,
        submitted_by="test",
        host="test",
        experiment_id="exp_20261129T040000Z_dec02",
        submitted_at=_TS,
    )
    result, verdict = verdict_from_manifest(
        manifest,
        stage_series={"season": _effect_series(mean=0.4)},
        bootstrap_reps=300,
        bootstrap_seed=0,
        computed_at=_TS,
    )
    assert result.holdout_season is None
    assert result.overall == "pass"
    assert verdict.overall == "pass"
