"""Long-horizon helpers: pick shift, MIP gap, warm-start week-1 agreement."""

from __future__ import annotations

import polars as pl
import pytest

from fplt.optimise.chip_windows import ChipUseWindows
from fplt.optimise.highs_options import HighsSolveOptions
from fplt.optimise.long_horizon import (
    DEFAULT_LONG_HORIZON_POLICY,
    LONG_HORIZON_MIN,
    LongHorizonPolicy,
    coarsen_steps,
    shift_horizon_picks,
)
from fplt.optimise.plan import plan_squad, shift_gameweek_plan
from fplt.optimise.pruning_loss import (
    check_week1_equivalence,
    sample_synthetic_scenarios,
)
from fplt.optimise.state import ChipInventory


def test_long_horizon_policy_defaults_relax_and_coarsen() -> None:
    """Shipped policy enables relax-and-fix and coarsening from step 3 at H≥6."""
    policy = DEFAULT_LONG_HORIZON_POLICY
    assert policy.auto_min_horizon == LONG_HORIZON_MIN
    assert policy.relax_and_fix is True
    assert policy.binary_steps == 2
    assert policy.coarsen_from_step == 3
    assert policy.should_auto_apply(5) is False
    assert policy.should_auto_apply(6) is True
    assert policy.effective_coarsen_from(5) is None
    assert policy.effective_coarsen_from(6) == 3
    assert policy.effective_coarsen_from(3) is None


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


def test_coarsen_steps_skips_forced_chip_gameweeks() -> None:
    """Forced-chip GWs stay out of the coarsened step set."""
    windows = ChipUseWindows(
        allowed={"3xc": frozenset({17})},
        forced={"3xc": frozenset({17})},
    )
    gameweeks = [10, 11, 12, 13, 14, 15, 16, 17]
    assert coarsen_steps(8, 3) == frozenset({3, 4, 5, 6, 7})
    assert coarsen_steps(8, 3, gameweeks=gameweeks, chip_windows=windows) == frozenset(
        {3, 4, 5, 6}
    )
    assert coarsen_steps(8, None, gameweeks=gameweeks, chip_windows=windows) == (
        frozenset()
    )


def test_far_step_coarsening_keeps_week1_actions_at_h6() -> None:
    """H=6 with captain/vice dropped from step 3 still matches week-1 actions.

    Pool / time_limit match the H=8 integrity toy (FDB-419 T1): week-1
    equivalence and far-step coarsening must still hold; we do not need a
    large candidate universe.
    """
    scenario = sample_synthetic_scenarios(
        1,
        seed=31,
        horizon=6,
        pool_counts=_FAST_LONG_HORIZON_POOL,
    )[0]
    options = _h6_options()
    # Full armbands every step; still use relax-and-fix so both arms share it.
    # Disable lineup_reopt so we assert MILP coarsening, not the post-hoc
    # EV lineup pass that re-assigns captain/vice on every locked fifteen.
    detailed = plan_squad(
        scenario.players,
        scenario.points,
        scenario.state,
        initial=scenario.initial,
        horizon=scenario.horizon,
        prune=False,
        highs_options=options,
        lineup_reopt=False,
        long_horizon_policy=LongHorizonPolicy(
            relax_and_fix=True,
            coarsen_from_step=None,
            auto_min_horizon=6,
        ),
    )
    coarsened = plan_squad(
        scenario.players,
        scenario.points,
        scenario.state,
        initial=scenario.initial,
        horizon=scenario.horizon,
        prune=False,
        highs_options=options,
        lineup_reopt=False,
        long_horizon_policy=LongHorizonPolicy(
            relax_and_fix=True,
            coarsen_from_step=3,
            auto_min_horizon=6,
        ),
    )
    result = check_week1_equivalence(detailed, coarsened, initial=scenario.initial)
    assert result.identical, result.mismatches
    # Coarsened far steps have no captain/vice selected.
    far = coarsened.picks.filter(pl.col("step") >= 3)
    assert far.filter(pl.col("captain")).is_empty()
    assert far.filter(pl.col("vice")).is_empty()


def test_forced_tc_on_coarsened_step_stays_feasible() -> None:
    """Forced TC on a far step keeps armbands; week-1 matches full-detail TC.

    Same compact pool as the other H=6 cases (FDB-419 T1). Feasibility of
    forced TC on a coarsened step and week-1 agreement remain the asserts.
    """
    scenario = sample_synthetic_scenarios(
        1,
        seed=41,
        horizon=6,
        pool_counts=_FAST_LONG_HORIZON_POOL,
    )[0]
    options = _h6_options()
    gameweeks = [10, 11, 12, 13, 14, 15]
    windows = ChipUseWindows(
        allowed={"3xc": frozenset({15})},
        forced={"3xc": frozenset({15})},
    )
    chips = ChipInventory(
        remaining={
            "wildcard": 0,
            "freehit": 0,
            "bboost": 0,
            "3xc": 1,
        },
        played=[],
    )
    state = scenario.state.model_copy(update={"chips": chips})
    # Without the forced-chip skip this used to be kInfeasible (TC=1 vs pin 0).
    coarsened = plan_squad(
        scenario.players,
        scenario.points,
        state,
        initial=scenario.initial,
        horizon=scenario.horizon,
        season="2025-26",
        gameweeks=gameweeks,
        chip_windows=windows,
        prune=False,
        highs_options=options,
        lineup_reopt=False,
        long_horizon_policy=LongHorizonPolicy(
            relax_and_fix=True,
            coarsen_from_step=3,
            auto_min_horizon=6,
        ),
    )
    assert coarsened.chips[-1] == "3xc"
    forced_step = coarsened.picks.filter(pl.col("step") == 5)
    assert forced_step.filter(pl.col("captain")).height == 1
    # Free (non-forced) far steps still coarsen.
    free_far = coarsened.picks.filter(pl.col("step").is_in([3, 4]))
    assert free_far.filter(pl.col("captain")).is_empty()
    assert free_far.filter(pl.col("vice")).is_empty()
    detailed = plan_squad(
        scenario.players,
        scenario.points,
        state,
        initial=scenario.initial,
        horizon=scenario.horizon,
        season="2025-26",
        gameweeks=gameweeks,
        chip_windows=windows,
        prune=False,
        highs_options=options,
        lineup_reopt=False,
        long_horizon_policy=LongHorizonPolicy(
            relax_and_fix=True,
            coarsen_from_step=None,
            auto_min_horizon=6,
        ),
    )
    result = check_week1_equivalence(detailed, coarsened, initial=scenario.initial)
    assert result.identical, result.mismatches


def test_relax_and_fix_keeps_week1_actions_at_h6() -> None:
    """H=6 relax-and-fix matches a full-integer solve on week-1 actions.

    Compact pool (FDB-419 T1); week-1 identity vs full-integer remains required.
    Seed 47 (not 23): on the integrity-sized pool seed 23 diverges week-1 even
    at gap-0, so it no longer tests the RAF approximation we care about.
    """
    scenario = sample_synthetic_scenarios(
        1,
        seed=47,
        horizon=6,
        pool_counts=_FAST_LONG_HORIZON_POOL,
    )[0]
    options = _h6_options()
    full = plan_squad(
        scenario.players,
        scenario.points,
        scenario.state,
        initial=scenario.initial,
        horizon=scenario.horizon,
        prune=False,
        highs_options=options,
        long_horizon_policy=LongHorizonPolicy(
            relax_and_fix=False,
            coarsen_from_step=None,
        ),
    )
    relaxed = plan_squad(
        scenario.players,
        scenario.points,
        scenario.state,
        initial=scenario.initial,
        horizon=scenario.horizon,
        prune=False,
        highs_options=options,
        long_horizon_policy=LongHorizonPolicy(
            relax_and_fix=True,
            binary_steps=2,
            auto_min_horizon=6,
            coarsen_from_step=None,
        ),
    )
    result = check_week1_equivalence(full, relaxed, initial=scenario.initial)
    assert result.identical, result.mismatches


def test_fast_long_horizon_pool_stays_integrity_sized() -> None:
    """Guard against re-inflating the H=6 toy pool (FDB-419 T1).

    Prevention: these cases used 6/14/18/8 (~13.5s sum). Shrinking must stay
    at the H=8 integrity size; growing it back would restore the long pole
    without strengthening week-1 asserts.
    """
    assert _FAST_LONG_HORIZON_POOL == {"GK": 4, "DEF": 10, "MID": 12, "FWD": 6}
    assert _H6_TIME_LIMIT == 2.0
    assert _H6_MIP_REL_GAP == 1e-2


def test_relax_and_fix_skipped_below_auto_min_horizon() -> None:
    """Short horizons ignore relax-and-fix even when the flag is on."""
    scenario = sample_synthetic_scenarios(
        1,
        seed=29,
        horizon=3,
        pool_counts={"GK": 6, "DEF": 14, "MID": 18, "FWD": 8},
    )[0]
    options = _fast_options()
    plain = plan_squad(
        scenario.players,
        scenario.points,
        scenario.state,
        initial=scenario.initial,
        horizon=scenario.horizon,
        prune=False,
        highs_options=options,
        long_horizon_policy=LongHorizonPolicy(relax_and_fix=False),
    )
    flagged = plan_squad(
        scenario.players,
        scenario.points,
        scenario.state,
        initial=scenario.initial,
        horizon=scenario.horizon,
        prune=False,
        highs_options=options,
        long_horizon_policy=LongHorizonPolicy(
            relax_and_fix=True,
            auto_min_horizon=6,
        ),
    )
    result = check_week1_equivalence(plain, flagged, initial=scenario.initial)
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


# Match H=8 integrity toy size (#319 / FDB-419 T1): enough for transfers,
# small enough that dual H=6 solves stay off the xdist long pole.
_FAST_LONG_HORIZON_POOL = {"GK": 4, "DEF": 10, "MID": 12, "FWD": 6}
_H6_TIME_LIMIT = 2.0
_H6_MIP_REL_GAP = 1e-2


def _fast_options() -> HighsSolveOptions:
    """HiGHS options for short-horizon agreement checks."""
    return HighsSolveOptions(
        time_limit=30.0,
        mip_rel_gap=0.0,
        accept_time_limit=True,
    )


def _h6_options() -> HighsSolveOptions:
    """Compact H=6 toys: integrity-sized pool + H=8-style gap/time ceiling."""
    return HighsSolveOptions(
        time_limit=_H6_TIME_LIMIT,
        mip_rel_gap=_H6_MIP_REL_GAP,
        accept_time_limit=True,
    )
