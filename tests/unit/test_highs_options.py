"""HiGHS solve-option policy for the transfer MILP."""

from __future__ import annotations

import highspy
import polars as pl
import pytest

from fplt.optimise.highs_options import HighsSolveOptions
from fplt.optimise.plan import GameweekPlan, plan_squad
from fplt.optimise.state import SquadState


def test_open_fpl_defaults_accept_time_limit() -> None:
    """Default options use 600s, a small MIP gap, and accept incumbents."""
    options = HighsSolveOptions.open_fpl_defaults()
    assert options.time_limit == 600.0
    assert options.mip_rel_gap == pytest.approx(1e-3)
    assert options.accepts_status(highspy.HighsModelStatus.kOptimal)
    assert options.accepts_status(highspy.HighsModelStatus.kTimeLimit)


def test_unlimited_optimal_rejects_time_limit() -> None:
    """Pre-policy behaviour requires proven optimality."""
    options = HighsSolveOptions.unlimited_optimal()
    assert options.time_limit is None
    assert options.accepts_status(highspy.HighsModelStatus.kOptimal)
    assert not options.accepts_status(highspy.HighsModelStatus.kTimeLimit)


def test_board_deterministic_is_single_threaded_seeded() -> None:
    """Chip boards use no time limit, one thread, and a fixed random seed."""
    options = HighsSolveOptions.board_deterministic()
    assert options.time_limit is None
    assert options.threads == 1
    assert options.parallel == "off"
    assert options.random_seed == 0
    assert options.mip_rel_gap == pytest.approx(0.0)
    assert options.accept_time_limit is False
    assert not options.accepts_status(highspy.HighsModelStatus.kTimeLimit)
    assert options.accepts_status(highspy.HighsModelStatus.kOptimal)

    solver = highspy.Highs()
    options.apply(solver)

    def _opt(name: str) -> object:
        status, value = solver.getOptionValue(name)
        assert status == highspy.HighsStatus.kOk
        return value

    assert _opt("threads") == 1
    assert _opt("random_seed") == 0
    assert _opt("parallel") == "off"


def _synthetic_league_pool(*, n_teams: int = 20) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Build a mid-size candidate pool + H=3 xPts for prune / HiGHS tests."""
    rows: list[dict[str, int | str]] = []
    player_key = 1
    for team in range(1, n_teams + 1):
        for position, count, base_price in (
            ("GK", 4, 40),
            ("DEF", 8, 40),
            ("MID", 8, 45),
            ("FWD", 5, 45),
        ):
            for offset in range(count):
                rows.append(
                    {
                        "player_key": player_key,
                        "position": position,
                        "team_key": team,
                        "price": base_price + offset,
                    }
                )
                player_key += 1
    candidates = pl.DataFrame(rows)
    points = pl.DataFrame(
        {
            "player_key": [int(r["player_key"]) for r in rows for _ in range(3)],
            "step": [step for _ in rows for step in range(3)],
            "xpts": [
                float((int(r["player_key"]) % 50) / 10 + step)
                for r in rows
                for step in range(3)
            ],
        }
    )
    return candidates, points


def _step0_signature(plan: GameweekPlan) -> tuple[object, ...]:
    """Stable fingerprint of a plan's actionable week + objective."""
    picks = plan.picks.filter(pl.col("step") == 0).sort("player_key")
    squad = tuple(int(k) for k in picks["player_key"].to_list())
    xi = tuple(int(k) for k in picks.filter(pl.col("in_xi"))["player_key"].to_list())
    return (round(float(plan.points), 10), squad, xi)


def test_board_deterministic_solves_pruned_empty_squad_h3() -> None:
    """Regression: board_deterministic solves pruned GW1 drafts (FDB-269).

    Candidate prune initializes HiGHS' thread-local scheduler with the default
    ``threads=0`` (N cores). Pinning ``threads=1`` without
    ``Highs.resetThreadScheduler`` then returns ``kNotset``. ``apply`` must
    reset before the pin so empty-squad H=3 drafts stay solvable.
    """
    candidates, points = _synthetic_league_pool()
    state = SquadState(bank=1000, free_transfers=1, ft_bank_cap=5)
    plan = plan_squad(
        candidates,
        points,
        state,
        initial=True,
        horizon=3,
        prune=True,
        highs_options=HighsSolveOptions.board_deterministic(),
        lineup_reopt=True,
    )
    assert plan.mip_status == "kOptimal"
    assert plan.pool_size is not None and plan.pool_size < candidates.height


def test_board_deterministic_identical_across_reruns() -> None:
    """Identical inputs → identical step-0 squad / XI / objective (FDB-269)."""
    candidates, points = _synthetic_league_pool()
    state = SquadState(bank=1000, free_transfers=1, ft_bank_cap=5)
    options = HighsSolveOptions.board_deterministic()
    signatures = [
        _step0_signature(
            plan_squad(
                candidates,
                points,
                state,
                initial=True,
                horizon=3,
                prune=True,
                highs_options=options,
                lineup_reopt=True,
            )
        )
        for _ in range(3)
    ]
    assert len(set(signatures)) == 1


def test_apply_threads_pin_after_default_scheduler() -> None:
    """Pinning threads after a default-thread solve must not leave kNotset."""
    candidates, points = _synthetic_league_pool(n_teams=8)
    state = SquadState(bank=1000, free_transfers=1, ft_bank_cap=5)
    # Contaminate the thread-local scheduler (default threads → N cores).
    warm = HighsSolveOptions(
        time_limit=None,
        mip_rel_gap=0.0,
        parallel="off",
        threads=None,
        random_seed=0,
        accept_time_limit=False,
    )
    warm_plan = plan_squad(
        candidates,
        points,
        state,
        initial=True,
        horizon=3,
        prune=False,
        highs_options=warm,
        lineup_reopt=False,
    )
    assert warm_plan.mip_status == "kOptimal"
    # board_deterministic applies threads=1 after resetThreadScheduler.
    pinned = plan_squad(
        candidates,
        points,
        state,
        initial=True,
        horizon=3,
        prune=False,
        highs_options=HighsSolveOptions.board_deterministic(),
        lineup_reopt=False,
    )
    assert pinned.mip_status == "kOptimal"


def test_time_limit_must_be_positive() -> None:
    """Zero / negative time limits are rejected."""
    with pytest.raises(ValueError, match="time_limit"):
        HighsSolveOptions(time_limit=0.0)


def test_apply_writes_options_onto_solver() -> None:
    """apply() sets the HiGHS option bag the MILP will use."""
    solver = highspy.Highs()
    options = HighsSolveOptions(
        time_limit=12.5,
        mip_rel_gap=1e-3,
        output_flag=False,
        presolve="on",
        parallel="off",
        threads=1,
        random_seed=7,
    )
    options.apply(solver)

    def _opt(name: str) -> object:
        status, value = solver.getOptionValue(name)
        assert status == highspy.HighsStatus.kOk
        return value

    assert _opt("time_limit") == 12.5
    assert _opt("mip_rel_gap") == pytest.approx(1e-3)
    assert _opt("presolve") == "on"
    assert _opt("parallel") == "off"
    assert _opt("threads") == 1
    assert _opt("random_seed") == 7
