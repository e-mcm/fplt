"""HiGHS solve-option policy for the transfer MILP."""

from __future__ import annotations

import highspy
import polars as pl
import pytest

from fplt.optimise.highs_options import HighsSolveOptions
from fplt.optimise.plan import plan_squad
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


def test_board_deterministic_is_seeded_parallel_off() -> None:
    """Chip boards use no time limit, parallel off, and a fixed random seed.

    ``threads`` stays ``None``: pinning ``threads=1`` makes highspy 1.16
    return ``kNotset`` on pruned empty-squad H≥3 MILPs (FDB-258).
    """
    options = HighsSolveOptions.board_deterministic()
    assert options.time_limit is None
    assert options.threads is None
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


def test_board_deterministic_solves_pruned_empty_squad_h3() -> None:
    """Regression: board_deterministic must solve pruned GW1 drafts (FDB-258).

    highspy 1.16 returns ``kNotset`` (elapsed 0) when ``threads=1`` is pinned
    on auto-pruned empty-squad H=3 MILPs; ``threads=None`` + ``parallel=off``
    + ``random_seed=0`` stays feasible and deterministic.
    """
    candidates, points = _synthetic_league_pool()
    state = SquadState(bank=1000, free_transfers=1, ft_bank_cap=5)
    broken = HighsSolveOptions(
        time_limit=None,
        mip_rel_gap=0.0,
        output_flag=False,
        presolve="on",
        parallel="off",
        threads=1,
        random_seed=0,
        accept_time_limit=False,
    )
    with pytest.raises(ValueError, match="no plan satisfies"):
        plan_squad(
            candidates,
            points,
            state,
            initial=True,
            horizon=3,
            prune=True,
            highs_options=broken,
            lineup_reopt=True,
        )
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
