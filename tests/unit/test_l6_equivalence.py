"""L6 native equivalence gate helpers."""

from __future__ import annotations

import polars as pl

from fplt.backtest.l6_equivalence import (
    MEAN_ABS_XPTS_MAX,
    assess_l6_gates,
    mean_abs_xpts_delta,
)
from fplt.backtest.promote_eval import load_eval_spec, plan_promote_eval


def _arm(model_id: str, *, xpts: float, shift: float = 0.0) -> pl.DataFrame:
    rows = []
    for gw in (10, 11, 12, 13):
        for i, pos in enumerate(("GK", "DEF", "MID", "FWD")):
            rows.append(
                {
                    "model_id": model_id,
                    "as_of_season": "2025-26",
                    "as_of_gw": gw,
                    "target_season": "2025-26",
                    "target_gw": gw,
                    "horizon": 1,
                    "player_key": 100 + i,
                    "fixture_id": 1000 + gw,
                    "position": pos,
                    "xpts": xpts + shift + 0.01 * i,
                    "p_haul": 0.1,
                    "actual_pts": 2.0,
                }
            )
    return pl.DataFrame(rows)


def test_dc_layered_v4_native_ab_eval_loads() -> None:
    spec = load_eval_spec("dc_layered_v4_native_ab")
    assert spec.control == "dc_layered_v4"
    assert spec.treatment == "dc_layered_v4_native"
    assert spec.change.lever == "hyperparam"
    assert spec.suites.steps == 8
    assert spec.window.from_gw == 10
    assert spec.window.to_gw == 17
    assert any("mc_backend" in item for item in spec.change.diff)
    assert spec.report.store_doc == "native-mc-l6-equivalence.md"
    plan = plan_promote_eval(spec, dry_run=True)
    assert plan.control == "dc_layered_v4"
    assert plan.treatment == "dc_layered_v4_native"


def test_mean_abs_xpts_within_and_beyond_bar() -> None:
    control = _arm("c", xpts=4.0)
    close = _arm("t", xpts=4.0, shift=0.005)
    mean_close, n = mean_abs_xpts_delta(control, close)
    assert n == control.height
    assert mean_close <= MEAN_ABS_XPTS_MAX

    far = _arm("t", xpts=4.0, shift=0.05)
    mean_far, _ = mean_abs_xpts_delta(control, far)
    assert mean_far > MEAN_ABS_XPTS_MAX


def test_assess_l6_gates_pass_on_identical_arms() -> None:
    control = _arm("c", xpts=4.0)
    treatment = _arm("t", xpts=4.0, shift=0.0)
    scored = pl.concat([control, treatment])
    board = assess_l6_gates(
        control,
        treatment,
        control_id="c",
        treatment_id="t",
        scored_ledger=scored,
    )
    assert board.passed
    assert board.mean_abs_xpts == 0.0
    names = {gate.name for gate in board.gates}
    assert "mean_abs_xpts" in names
    assert "signed_xpts_ci_overall" in names
    assert "haul_brier_ci_overall" in names


def test_assess_l6_gates_fail_on_systematic_bias() -> None:
    control = _arm("c", xpts=4.0)
    treatment = _arm("t", xpts=4.0, shift=0.05)
    board = assess_l6_gates(
        control,
        treatment,
        control_id="c",
        treatment_id="t",
        scored_ledger=None,
    )
    assert not board.passed
    assert board.mean_abs_xpts > MEAN_ABS_XPTS_MAX
    by_name = {gate.name: gate for gate in board.gates}
    assert by_name["mean_abs_xpts"].passed is False
    assert by_name["signed_xpts_ci_overall"].passed is False
