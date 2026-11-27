"""Locked-fifteen integrity helpers and lineup-reopt far-step contract."""

from __future__ import annotations

import polars as pl
import pytest

from fplt.optimise.chip_windows import ChipUseWindows
from fplt.optimise.highs_options import HighsSolveOptions
from fplt.optimise.lineup import reoptimize_plan_lineups
from fplt.optimise.long_horizon import LongHorizonPolicy
from fplt.optimise.plan import plan_squad
from fplt.optimise.pruning_loss import sample_synthetic_scenarios
from fplt.optimise.squad_integrity import (
    SQUAD_POSITION_QUOTA,
    format_step_integrity,
    inspect_horizon_picks,
    locked_fifteen_issue,
    require_locked_steps,
    squad_position_counts,
)
from fplt.optimise.state import ChipInventory


def test_locked_fifteen_issue_reports_gk1_mid6() -> None:
    """The post-#150 H=8 crash signature is a clear formation issue string."""
    positions = [
        "GK",
        "DEF",
        "DEF",
        "DEF",
        "DEF",
        "DEF",
        "MID",
        "MID",
        "MID",
        "MID",
        "MID",
        "MID",
        "FWD",
        "FWD",
        "FWD",
    ]
    squad = pl.DataFrame(
        {
            "player_key": list(range(1, 16)),
            "position": positions,
        }
    )
    issue = locked_fifteen_issue(squad)
    assert issue is not None
    assert "GK" in issue
    assert "1" in issue
    assert squad_position_counts(squad) == {
        "GK": 1,
        "DEF": 5,
        "MID": 6,
        "FWD": 3,
    }
    assert locked_fifteen_issue(_legal_fifteen()) is None


def test_reoptimize_skips_illegal_far_step_keeps_step0() -> None:
    """Far-step GK:1/MID:6 must not crash reopt; actionable step still re-lined.

    Regression for post-#150 H=8 boards: relax-and-fix far steps can
    0.5-threshold into an illegal fifteen; lineup_reopt used to raise in
    ``_validate_squad`` after ``kOptimal``.
    """
    step0 = _pick_rows(step=0, positions=_legal_positions(), xpts=3.0)
    # Illegal locked-looking fifteen on a far continuous step.
    bad_positions = [
        "GK",
        "DEF",
        "DEF",
        "DEF",
        "DEF",
        "DEF",
        "MID",
        "MID",
        "MID",
        "MID",
        "MID",
        "MID",
        "FWD",
        "FWD",
        "FWD",
    ]
    step3 = _pick_rows(step=3, positions=bad_positions, xpts=4.0, key_base=200)
    picks = pl.concat([step0, step3])
    assert locked_fifteen_issue(step3.filter(~pl.col("sold"))) is not None

    refreshed = reoptimize_plan_lineups(picks)
    # Step 0 reopted (still 11 in XI, one captain).
    s0 = refreshed.filter((pl.col("step") == 0) & ~pl.col("sold"))
    assert s0.filter(pl.col("in_xi")).height == 11
    assert int(s0["captain"].sum()) == 1
    # Step 3 left untouched (illegal formation kept as MILP shadow).
    s3 = refreshed.filter(pl.col("step") == 3)
    assert s3.select("in_xi", "captain", "vice").equals(
        step3.select("in_xi", "captain", "vice")
    )


def test_reoptimize_still_requires_legal_scenario_step() -> None:
    """Actionable step must remain a legal fifteen — that is a real bug."""
    bad = _pick_rows(
        step=0,
        positions=[
            "GK",
            "DEF",
            "DEF",
            "DEF",
            "DEF",
            "DEF",
            "MID",
            "MID",
            "MID",
            "MID",
            "MID",
            "MID",
            "FWD",
            "FWD",
            "FWD",
        ],
        xpts=2.0,
    )
    with pytest.raises(ValueError, match="scenario_step 0"):
        reoptimize_plan_lineups(bad)


def test_inspect_and_require_locked_steps() -> None:
    """Post-mortem helper surfaces the bad step; require_locked_steps fails it."""
    picks = pl.concat(
        [
            _pick_rows(step=0, positions=_legal_positions(), xpts=1.0),
            _pick_rows(
                step=2,
                positions=[
                    "GK",
                    "DEF",
                    "DEF",
                    "DEF",
                    "DEF",
                    "DEF",
                    "MID",
                    "MID",
                    "MID",
                    "MID",
                    "MID",
                    "MID",
                    "FWD",
                    "FWD",
                    "FWD",
                ],
                xpts=1.0,
                key_base=50,
            ),
        ]
    )
    integrity = inspect_horizon_picks(picks, chips=("bboost", None, "freehit"))
    assert integrity[0].ok
    assert not integrity[1].ok
    assert "MID" in format_step_integrity(integrity[1])
    require_locked_steps(integrity, (0,), context="test")
    with pytest.raises(ValueError, match="step=2"):
        require_locked_steps(integrity, (0, 2), context="test")


@pytest.mark.slow
def test_plan_squad_h8_raf_lineup_reopt_with_freehit_windows() -> None:
    """Default post-solve pipeline must tolerate RAF + FH allow-lists at H=8.

    Prevention: long-horizon tests used to set lineup_reopt=False, so the
    production composition (RAF + reopt + chips) was untested.

    Pool / time_limit are kept deliberately small (FDB-387): assertions only
    need a feasible step-0 locked fifteen under the production flags, not a
    large candidate universe.
    """
    scenario = sample_synthetic_scenarios(
        1,
        seed=11,
        horizon=8,
        pool_counts={"GK": 4, "DEF": 10, "MID": 12, "FWD": 6},
    )[0]
    gameweeks = list(range(5, 13))
    windows = ChipUseWindows(
        allowed={
            "freehit": frozenset(gameweeks),
            "wildcard": frozenset(gameweeks),
            "bboost": frozenset({9}),
            "3xc": frozenset({5, 10}),
        },
        forced={"bboost": frozenset({9})},
    )
    chips = ChipInventory(
        remaining={"wildcard": 1, "freehit": 1, "bboost": 1, "3xc": 1},
        played=[],
    )
    state = scenario.state.model_copy(update={"chips": chips})
    plan = plan_squad(
        scenario.players,
        scenario.points,
        state,
        initial=scenario.initial,
        horizon=8,
        season="2025-26",
        gameweeks=gameweeks,
        chip_windows=windows,
        prune=False,
        highs_options=HighsSolveOptions(time_limit=5.0, mip_rel_gap=1e-2),
        lineup_reopt=True,
        long_horizon_policy=LongHorizonPolicy(
            relax_and_fix=True,
            binary_steps=2,
            coarsen_from_step=3,
            auto_min_horizon=6,
        ),
    )
    step0 = plan.picks.filter((pl.col("step") == 0) & ~pl.col("sold"))
    assert locked_fifteen_issue(step0) is None
    assert step0.filter(pl.col("in_xi")).height == 11
    assert dict(SQUAD_POSITION_QUOTA) == {
        "GK": 2,
        "DEF": 5,
        "MID": 5,
        "FWD": 3,
    }


def _legal_positions() -> list[str]:
    return [
        "GK",
        "GK",
        "DEF",
        "DEF",
        "DEF",
        "DEF",
        "DEF",
        "MID",
        "MID",
        "MID",
        "MID",
        "MID",
        "FWD",
        "FWD",
        "FWD",
    ]


def _legal_fifteen() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "player_key": list(range(1, 16)),
            "position": _legal_positions(),
        }
    )


def _pick_rows(
    *,
    step: int,
    positions: list[str],
    xpts: float,
    key_base: int = 1,
) -> pl.DataFrame:
    n = len(positions)
    return pl.DataFrame(
        {
            "step": [step] * n,
            "player_key": list(range(key_base, key_base + n)),
            "position": positions,
            "team_key": [1 + (i % 3) for i in range(n)],
            "price": [50] * n,
            "xpts": [xpts] * n,
            "in_xi": [i < 11 for i in range(n)],
            "captain": [i == 2 for i in range(n)],
            "vice": [i == 3 for i in range(n)],
            "bought": [False] * n,
            "sold": [False] * n,
        }
    )
