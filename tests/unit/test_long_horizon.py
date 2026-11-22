"""Long-horizon helpers: pick shift, MIP gap, warm-start week-1 agreement."""

from __future__ import annotations

import polars as pl
import pytest

from fplt.optimise.highs_options import HighsSolveOptions
from fplt.optimise.long_horizon import (
    DEFAULT_LONG_HORIZON_POLICY,
    LONG_HORIZON_MIN,
    LongHorizonPolicy,
    shift_horizon_picks,
)
from fplt.optimise.plan import plan_squad, shift_gameweek_plan
from fplt.optimise.pruning_loss import (
    check_week1_equivalence,
    sample_synthetic_scenarios,
)


def test_long_horizon_policy_defaults_off() -> None:
    """Shipped policy waits for H≥6 and leaves shrink heuristics off."""
    policy = DEFAULT_LONG_HORIZON_POLICY
    assert policy.auto_min_horizon == LONG_HORIZON_MIN
    assert policy.relax_and_fix is False
    assert policy.coarsen_from_step is None
    assert policy.should_auto_apply(5) is False
    assert policy.should_auto_apply(6) is True


def test_long_horizon_policy_rejects_negative_coarsen() -> None:
    """coarsen_from_step must be non-negative when set."""
    with pytest.raises(ValueError, match="coarsen_from_step"):
        LongHorizonPolicy(coarsen_from_step=-1)


def test_shift_horizon_picks_renumbers_from_zero() -> None:
    """Dropping step 0 leaves prior step 1 as the new step 0."""
    picks = pl.DataFrame(
        {
            "step": [0, 0, 1, 1, 2],
            "player_key": [1, 2, 1, 3, 1],
        }
    )
    shifted = shift_horizon_picks(picks, steps=1)
    assert shifted["step"].to_list() == [0, 0, 1]
    assert shifted["player_key"].to_list() == [1, 3, 1]
    assert shift_horizon_picks(picks, steps=0).equals(picks)
    with pytest.raises(ValueError, match="steps"):
        shift_horizon_picks(picks, steps=-1)


def test_shift_gameweek_plan_trims_horizon_metadata() -> None:
    """Hits / free transfers / chips shift with picks; points are cleared."""
    scenario = sample_synthetic_scenarios(
        1,
        seed=3,
        horizon=3,
        pool_counts={"GK": 6, "DEF": 14, "MID": 18, "FWD": 8},
    )[0]
    plan = plan_squad(
        scenario.players,
        scenario.points,
        scenario.state,
        initial=scenario.initial,
        horizon=scenario.horizon,
        prune=False,
        highs_options=_fast_options(),
    )
    shifted = shift_gameweek_plan(plan, steps=1)
    assert shifted.picks["step"].min() == 0
    assert shifted.hits == plan.hits[1:]
    assert shifted.free_transfers == plan.free_transfers[1:]
    assert shifted.points == 0.0
    assert shifted.solve_elapsed is None


def test_warm_start_from_own_solution_keeps_week1_actions() -> None:
    """Re-solving with the same plan as a MIP start (shift 0) agrees on week 1."""
    scenario = sample_synthetic_scenarios(
        1,
        seed=11,
        horizon=3,
        pool_counts={"GK": 6, "DEF": 14, "MID": 18, "FWD": 8},
    )[0]
    options = _fast_options()
    reference = plan_squad(
        scenario.players,
        scenario.points,
        scenario.state,
        initial=scenario.initial,
        horizon=scenario.horizon,
        prune=False,
        highs_options=options,
    )
    warmed = plan_squad(
        scenario.players,
        scenario.points,
        scenario.state,
        initial=scenario.initial,
        horizon=scenario.horizon,
        prune=False,
        highs_options=options,
        warm_start=reference,
        warm_start_shift=0,
    )
    result = check_week1_equivalence(reference, warmed, initial=scenario.initial)
    assert result.identical, result.mismatches


def test_small_mip_gap_keeps_week1_actions_vs_proven_optimal() -> None:
    """Default 1e-3 gap matches gap-0 week-1 actions on a small synthetic state."""
    scenario = sample_synthetic_scenarios(
        1,
        seed=17,
        horizon=3,
        pool_counts={"GK": 6, "DEF": 14, "MID": 18, "FWD": 8},
    )[0]
    proven = plan_squad(
        scenario.players,
        scenario.points,
        scenario.state,
        initial=scenario.initial,
        horizon=scenario.horizon,
        prune=False,
        highs_options=HighsSolveOptions(
            time_limit=30.0,
            mip_rel_gap=0.0,
            accept_time_limit=True,
        ),
    )
    gapped = plan_squad(
        scenario.players,
        scenario.points,
        scenario.state,
        initial=scenario.initial,
        horizon=scenario.horizon,
        prune=False,
        highs_options=HighsSolveOptions(
            time_limit=30.0,
            mip_rel_gap=1e-3,
            accept_time_limit=True,
        ),
    )
    result = check_week1_equivalence(proven, gapped, initial=scenario.initial)
    assert result.identical, result.mismatches


def test_warm_start_shift_rejects_negative() -> None:
    """Negative warm_start_shift is rejected before the solve."""
    scenario = sample_synthetic_scenarios(
        1,
        seed=5,
        horizon=1,
        pool_counts={"GK": 6, "DEF": 14, "MID": 18, "FWD": 8},
    )[0]
    seed = plan_squad(
        scenario.players,
        scenario.points,
        scenario.state,
        initial=scenario.initial,
        horizon=scenario.horizon,
        prune=False,
        highs_options=_fast_options(),
    )
    with pytest.raises(ValueError, match="warm_start_shift"):
        plan_squad(
            scenario.players,
            scenario.points,
            scenario.state,
            initial=scenario.initial,
            horizon=scenario.horizon,
            prune=False,
            highs_options=_fast_options(),
            warm_start=seed,
            warm_start_shift=-1,
        )


def _fast_options() -> HighsSolveOptions:
    return HighsSolveOptions(
        time_limit=30.0,
        mip_rel_gap=0.0,
        accept_time_limit=True,
    )
