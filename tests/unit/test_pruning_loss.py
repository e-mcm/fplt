"""Pruning-loss harness: week-1 agreement, pool scaling, report I/O."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from fplt.optimise.candidate_prune import DEFAULT_PRUNE_POLICY, MIN_POS_POOL
from fplt.optimise.highs_options import HighsSolveOptions
from fplt.optimise.plan import plan_squad
from fplt.optimise.pruning_loss import (
    DEFAULT_POOL_TARGETS,
    DEFAULT_TARGET_POOL,
    PruneArm,
    Week1Actions,
    check_week1_equivalence,
    default_sweep_arms,
    format_pruning_loss_summary,
    policy_for_target_pool,
    run_pruning_loss_harness,
    sample_synthetic_scenarios,
    summarise_pruning_loss,
    week1_actions_agree,
    week1_actions_from_plan,
    write_pruning_loss_report,
)
from fplt.schemas.pruning_loss import PruningLossRowSchema

# Small pool keeps HiGHS solves reliable under the full `make check` suite.
_TEST_POOL = {"GK": 6, "DEF": 16, "MID": 22, "FWD": 10}


def _fast_options() -> HighsSolveOptions:
    # Do not set threads=1 after other tests have used parallel HiGHS —
    # highspy 1.16 can return kNotset immediately on the next solve.
    return HighsSolveOptions(
        time_limit=30.0,
        mip_rel_gap=0.0,
        accept_time_limit=True,
    )


def _scenarios(n: int, *, seed: int):
    return sample_synthetic_scenarios(n, seed=seed, horizon=5, pool_counts=_TEST_POOL)


def test_sample_synthetic_scenarios_covers_tags() -> None:
    """Twenty-four samples cycle through the four §3.1 scenario tags."""
    scenarios = sample_synthetic_scenarios(
        24, seed=1, horizon=5, pool_counts=_TEST_POOL
    )
    assert len(scenarios) == 24
    tags = {scenario.tag for scenario in scenarios}
    assert tags == {"early_season", "dgw_runup", "rebuild", "mid_banked_fts"}
    for scenario in scenarios:
        assert scenario.horizon == 5
        assert scenario.players.height >= sum(MIN_POS_POOL.values())
        if scenario.tag in {"early_season", "rebuild"}:
            assert scenario.initial
            assert scenario.state.owned == []
        else:
            assert not scenario.initial
            assert len(scenario.state.owned) == 15
        if scenario.tag == "mid_banked_fts":
            assert scenario.state.free_transfers == 2


def test_policy_for_target_pool_scales_quotas() -> None:
    """Smaller N shrinks top_xpts / top_value; cheapest fillers stay fixed."""
    tight = policy_for_target_pool(100, reference_n=DEFAULT_TARGET_POOL)
    assert tight.keep_cheapest == DEFAULT_PRUNE_POLICY.keep_cheapest
    assert tight.top_xpts is not None and DEFAULT_PRUNE_POLICY.top_xpts is not None
    assert tight.top_xpts["MID"] < DEFAULT_PRUNE_POLICY.top_xpts["MID"]
    assert tight.top_xpts["GK"] >= MIN_POS_POOL["GK"]
    wide = policy_for_target_pool(300)
    assert wide.top_xpts is not None
    assert wide.top_xpts["MID"] > DEFAULT_PRUNE_POLICY.top_xpts["MID"]


def test_policy_for_target_pool_rejects_below_minima() -> None:
    """Targets smaller than a legal fifteen are rejected."""
    with pytest.raises(ValueError, match="squad minima"):
        policy_for_target_pool(sum(MIN_POS_POOL.values()) - 1)


def test_default_sweep_arms_include_reference_targets_and_current() -> None:
    """Arms are reference, each N, then the shipped default policy."""
    arms = default_sweep_arms()
    ids = [arm.arm_id for arm in arms]
    assert ids[0] == "reference"
    assert arms[0].policy is None
    assert ids[-1] == "current"
    assert arms[-1].policy == DEFAULT_PRUNE_POLICY
    for target in DEFAULT_POOL_TARGETS:
        assert f"n={target}" in ids


def test_week1_actions_and_equivalence_on_identical_plans() -> None:
    """Same solve twice → identical week-1 actions (equivalence pass)."""
    scenario = _scenarios(1, seed=2)[0]
    plan_a = plan_squad(
        scenario.players,
        scenario.points,
        scenario.state,
        initial=scenario.initial,
        horizon=scenario.horizon,
        prune=False,
        highs_options=_fast_options(),
    )
    plan_b = plan_squad(
        scenario.players,
        scenario.points,
        scenario.state,
        initial=scenario.initial,
        horizon=scenario.horizon,
        prune=False,
        highs_options=_fast_options(),
    )
    assert plan_a.solve_elapsed is not None
    assert plan_a.mip_status is not None
    assert plan_a.pool_size == scenario.players.height
    left = week1_actions_from_plan(plan_a, initial=scenario.initial)
    right = week1_actions_from_plan(plan_b, initial=scenario.initial)
    assert week1_actions_agree(left, right)
    result = check_week1_equivalence(plan_a, plan_b, initial=scenario.initial)
    assert result.identical
    assert result.mismatches == ()
    assert len(left.xi) == 11
    assert left.captain is not None
    assert left.vice is not None
    if scenario.initial:
        assert left.transfers_in == ()
        assert left.transfers_out == ()


def test_week1_actions_detect_transfer_mismatch() -> None:
    """Mismatches list names the differing week-1 fields."""
    left = Week1Actions(
        transfers_in=(1,),
        transfers_out=(2,),
        xi=(1, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12),
        captain=1,
        vice=3,
    )
    right = Week1Actions(
        transfers_in=(9,),
        transfers_out=(2,),
        xi=left.xi,
        captain=1,
        vice=3,
    )
    assert not week1_actions_agree(left, right)
    assert left.mismatches(right) == ("transfers_in",)


def test_harness_records_reference_agreement_and_writes_report(
    tmp_path: Path,
) -> None:
    """Reference arm always agrees with itself; report artifacts validate."""
    scenarios = _scenarios(4, seed=3)
    # Scale against a small reference so n=40 actually prunes the test pool.
    tight = policy_for_target_pool(40, reference_n=60)
    arms = (
        PruneArm(arm_id="reference", policy=None),
        PruneArm(arm_id="n=40", policy=tight, target_pool=40),
        PruneArm(arm_id="current", policy=DEFAULT_PRUNE_POLICY),
    )
    report = run_pruning_loss_harness(
        scenarios,
        arms=arms,
        reference_options=_fast_options(),
        sweep_options=_fast_options(),
    )
    checked = PruningLossRowSchema.validate(report.rows)
    assert isinstance(checked, pl.DataFrame)
    assert checked.height == len(scenarios) * len(arms)
    ref_rows = checked.filter(pl.col("arm_id") == "reference")
    assert ref_rows["week1_agree"].all()
    assert (ref_rows["objective_gap"].abs() < 1e-9).all()
    assert set(ref_rows["mip_status"].drop_nulls().to_list()).issubset(
        {"kOptimal", "kTimeLimit"}
    )
    summary = summarise_pruning_loss(checked)
    assert "agree_rate" in summary.columns
    text = format_pruning_loss_summary(summary)
    assert "reference" in text
    paths = write_pruning_loss_report(report, tmp_path)
    assert paths["rows"].is_file()
    assert paths["summary"].is_file()
    assert paths["json"].is_file()
    assert "identical week-1 actions" in paths["json"].read_text()


def test_harness_rejects_non_reference_first_arm() -> None:
    """The first arm must be the full-pool reference."""
    scenarios = _scenarios(1, seed=4)
    with pytest.raises(ValueError, match="reference"):
        run_pruning_loss_harness(
            scenarios,
            arms=(PruneArm(arm_id="current", policy=DEFAULT_PRUNE_POLICY),),
            reference_options=_fast_options(),
            sweep_options=_fast_options(),
        )


def test_owned_players_survive_scaled_policy() -> None:
    """Scaled prune policies still keep every owned player."""
    scenario = next(item for item in _scenarios(8, seed=5) if not item.initial)
    policy = policy_for_target_pool(40, reference_n=60)
    pruned = policy.apply(
        scenario.players,
        scenario.points,
        owned_keys=[player.player_key for player in scenario.state.owned],
    )
    owned_keys = {player.player_key for player in scenario.state.owned}
    assert owned_keys.issubset(set(pruned["player_key"].to_list()))
    for position, minimum in MIN_POS_POOL.items():
        assert pruned.filter(pl.col("position") == position).height >= minimum


def test_cli_pruning_loss_smoke(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """CLI writes the report and prints the summary table."""
    from fplt.cli import main

    def _tiny_scenarios(
        n: int,
        *,
        seed: int = 0,
        horizon: int = 5,
        pool_counts: object = None,
    ):
        return sample_synthetic_scenarios(
            n, seed=seed, horizon=horizon, pool_counts=_TEST_POOL
        )

    monkeypatch.setattr(
        "fplt.optimise.pruning_loss.sample_synthetic_scenarios",
        _tiny_scenarios,
    )
    out = tmp_path / "out"
    main(
        [
            "pruning-loss",
            "--n-scenarios",
            "2",
            "--seed",
            "6",
            "--horizon",
            "5",
            "--targets",
            "100",
            "--reference-time-limit",
            "30",
            "--sweep-time-limit",
            "30",
            "--output-dir",
            str(out),
        ]
    )
    captured = capsys.readouterr()
    assert "reference" in captured.out
    assert (out / "summary.json").is_file()
