"""Season-level chip scheduler: coarse scores and option-value windows."""

from __future__ import annotations

import polars as pl
import pytest

from fplt.optimise.chip_scheduler import (
    ChipOpportunity,
    schedule_chip_windows,
    score_chip_opportunities,
    windows_from_opportunities,
)
from fplt.optimise.plan import plan_squad
from fplt.optimise.state import ChipInventory, SquadState
from fplt.scoring.chips import ChipGrant, ChipRules


def test_option_value_holds_chip_when_better_double_is_later() -> None:
    """Near-term weak BB is excluded when a bigger double sits after H."""
    fixtures = _fixtures_early_weak_late_strong()
    rules = ChipRules.for_season("2025-26")
    inventory = ChipInventory(
        remaining={"wildcard": 0, "freehit": 0, "bboost": 1, "3xc": 0},
        played=[],
    )
    windows = schedule_chip_windows(
        fixtures,
        season="2024-25",
        rules=rules,
        as_of_gw=8,
        short_horizon=2,
        inventory=inventory,
        gameweeks=list(range(8, 15)),
    )
    # GW10 is the big double (outside H=2 from GW8 → near_end=9).
    assert 10 in windows.allowed.get("bboost", frozenset())
    # Early weak double (GW8) must not beat holding for GW10.
    assert 8 not in windows.allowed.get("bboost", frozenset())
    assert 8 in windows.no_chip


def test_first_half_week_allowed_despite_higher_second_half_scores() -> None:
    """GW5 positive first-half score stays allowed when second-half scores dominate.

    Regression: option-value must key ``deferred_best`` by ``ChipGrant``.
    Merging halves emptied the first-half allow-list unless force-play packed
    GW16–19.
    """
    fixtures = _fixtures_first_half_and_stronger_second_half()
    rules = ChipRules.for_season("2025-26")
    inventory = ChipInventory(
        remaining={"wildcard": 0, "freehit": 0, "bboost": 1, "3xc": 0},
        played=[],
    )
    windows = schedule_chip_windows(
        fixtures,
        season="2024-25",
        rules=rules,
        as_of_gw=5,
        short_horizon=3,
        inventory=inventory,
        force_play=False,
    )
    assert windows.allows("bboost", 5)
    assert 5 in windows.allowed.get("bboost", frozenset())
    # Second-half double is still recorded, but must not empty the first-half list.
    assert 25 in windows.allowed.get("bboost", frozenset())


def test_chip_opportunity_grant_is_typed_map_key() -> None:
    """Opportunity maps must key by ChipGrant, not bare chip name."""
    first = ChipOpportunity(chip="bboost", window="first", gameweek=5, score=1.0)
    second = ChipOpportunity(chip="bboost", window="second", gameweek=25, score=9.0)
    assert first.grant == ChipGrant(name="bboost", window="first")
    assert second.grant == ChipGrant(name="bboost", window="second")
    assert first.grant != second.grant
    by_grant: dict[ChipGrant, list[ChipOpportunity]] = {}
    by_grant.setdefault(first.grant, []).append(first)
    by_grant.setdefault(second.grant, []).append(second)
    assert set(by_grant) == {
        ChipGrant(name="bboost", window="first"),
        ChipGrant(name="bboost", window="second"),
    }
    # Cross-half scores stay separate when fed through windows_from_opportunities.
    windows = windows_from_opportunities(
        (first, second),
        as_of_gw=5,
        short_horizon=3,
        force_play=False,
    )
    assert windows.allows("bboost", 5)
    assert windows.allows("bboost", 25)


def test_force_best_chip_inside_short_horizon_when_it_beats_later() -> None:
    fixtures = _fixtures_best_now()
    rules = ChipRules.for_season("2025-26")
    inventory = ChipInventory(
        remaining={"wildcard": 0, "freehit": 0, "bboost": 1, "3xc": 0},
        played=[],
    )
    windows = schedule_chip_windows(
        fixtures,
        season="2024-25",
        rules=rules,
        as_of_gw=10,
        short_horizon=3,
        inventory=inventory,
        gameweeks=list(range(10, 16)),
    )
    assert windows.forced.get("bboost") == frozenset({10})
    assert windows.allows("bboost", 10)
    assert windows.requires("bboost", 10)


def test_squad_continuity_boosts_free_hit_on_owned_blank() -> None:
    fixtures = pl.DataFrame(
        {
            "season": ["2024-25"] * 3,
            "gw": [11, 11, 12],
            "team_h_key": [2, 3, 1],
            "team_a_key": [3, 4, 2],
        }
    )
    # Team 1 blanks GW11; owned players are all on team 1.
    player_teams = pl.DataFrame(
        {
            "player_key": list(range(1, 16)),
            "team_key": [1] * 15,
        }
    )
    predictions = pl.DataFrame(
        {
            "gw": [11] * 15 + [12] * 15,
            "player_key": list(range(1, 16)) * 2,
            "xpts": [6.0] * 15 + [4.0] * 15,
        }
    )
    rules = ChipRules.for_season("2025-26")
    inventory = ChipInventory(
        remaining={"wildcard": 0, "freehit": 1, "bboost": 0, "3xc": 0},
        played=[],
    )
    opps = score_chip_opportunities(
        fixtures,
        season="2024-25",
        rules=rules,
        as_of_gw=11,
        predictions=predictions,
        player_teams=player_teams,
        owned_player_keys=list(range(1, 16)),
        inventory=inventory,
        gameweeks=[11, 12],
    )
    fh = {row.gameweek: row.score for row in opps if row.chip == "freehit"}
    assert fh[11] > fh[12]


def test_scheduled_windows_constrain_short_horizon_milp() -> None:
    pool = _legal_pool()
    rows: list[dict[str, object]] = []
    for key, xpts in zip(
        pool["player_key"].to_list(), pool["xpts"].to_list(), strict=True
    ):
        for step in (0, 1):
            rows.append({"player_key": key, "step": step, "xpts": float(xpts)})
    # Scheduler output consumed by plan_squad (forced TC only on GW6).
    windows = schedule_chip_windows(
        pl.DataFrame(
            {
                "season": ["2024-25", "2024-25", "2024-25"],
                "gw": [5, 6, 6],
                "team_h_key": [1, 1, 1],
                "team_a_key": [2, 2, 3],
            }
        ),
        season="2024-25",
        rules=ChipRules.for_season("2025-26"),
        as_of_gw=5,
        short_horizon=2,
        predictions=pl.DataFrame(
            {
                "gw": [5, 6],
                "player_key": [1, 1],
                "xpts": [4.0, 20.0],
            }
        ),
        player_teams=pool.select("player_key", "team_key"),
        inventory=ChipInventory(
            remaining={"wildcard": 0, "freehit": 0, "bboost": 0, "3xc": 1},
            played=[],
        ),
        gameweeks=[5, 6],
    )
    assert windows.requires("3xc", 6)
    plan = plan_squad(
        pool.drop("xpts"),
        pl.DataFrame(rows),
        SquadState(
            bank=1000,
            free_transfers=1,
            ft_bank_cap=2,
            hit_cost=4,
            decay=1.0,
            bench_weight=0.0,
            owned=[],
            chips=ChipInventory(
                remaining={"wildcard": 0, "freehit": 0, "bboost": 0, "3xc": 1},
                played=[],
            ),
        ),
        initial=True,
        horizon=2,
        season="2025-26",
        gameweeks=[5, 6],
        chip_windows=windows,
    )
    assert plan.chips == (None, "3xc")


def test_scheduler_skips_chip_already_played_in_window() -> None:
    """Regression: second-half remaining must not re-force a spent first-half chip."""
    from fplt.optimise.state import ChipPlay

    fixtures = _fixtures_best_now()
    rules = ChipRules.for_season("2025-26")
    inventory = ChipInventory(
        remaining={"wildcard": 1, "freehit": 1, "bboost": 1, "3xc": 1},
        played=[ChipPlay(name="freehit", gw=10)],
    )
    opps = score_chip_opportunities(
        fixtures,
        season="2024-25",
        rules=rules,
        as_of_gw=11,
        inventory=inventory,
        gameweeks=list(range(11, 19)),
    )
    assert not any(row.chip == "freehit" for row in opps)


def test_mid_window_unused_chip_is_optional() -> None:
    """Force-play does not fire mid-half when stop_event is beyond H."""
    fixtures = _fixtures_ordinary_first_half()
    rules = ChipRules.for_season("2025-26")
    inventory = ChipInventory(
        remaining={"wildcard": 0, "freehit": 0, "bboost": 1, "3xc": 0},
        played=[],
    )
    windows = schedule_chip_windows(
        fixtures,
        season="2024-25",
        rules=rules,
        as_of_gw=8,
        short_horizon=3,
        inventory=inventory,
        gameweeks=list(range(8, 15)),
    )
    assert not windows.forced.get("bboost")
    assert not windows.requires("bboost", 8)


def test_final_gw_of_half_forces_unused_chip() -> None:
    """Unused first-half chip is forced on GW19 (window stop)."""
    fixtures = _fixtures_ordinary_through(19)
    rules = ChipRules.for_season("2025-26")
    inventory = ChipInventory(
        remaining={"wildcard": 0, "freehit": 0, "bboost": 1, "3xc": 0},
        played=[],
    )
    windows = schedule_chip_windows(
        fixtures,
        season="2024-25",
        rules=rules,
        as_of_gw=19,
        short_horizon=3,
        inventory=inventory,
        gameweeks=[19, 20, 21],
    )
    assert windows.forced.get("bboost") == frozenset({19})
    assert windows.requires("bboost", 19)
    assert windows.allows("bboost", 19)


def test_final_gw_of_half_does_not_force_after_play() -> None:
    """Spent first-half chip is not forced again on the window stop."""
    from fplt.optimise.state import ChipPlay

    fixtures = _fixtures_ordinary_through(19)
    rules = ChipRules.for_season("2025-26")
    inventory = ChipInventory(
        remaining={"wildcard": 0, "freehit": 0, "bboost": 0, "3xc": 0},
        played=[ChipPlay(name="bboost", gw=12)],
    )
    windows = schedule_chip_windows(
        fixtures,
        season="2024-25",
        rules=rules,
        as_of_gw=19,
        short_horizon=3,
        inventory=inventory,
        gameweeks=[19, 20, 21],
    )
    assert not windows.forced.get("bboost")
    assert not windows.requires("bboost", 19)


def test_force_play_packs_multiple_chips_before_window_end() -> None:
    """Several unused half-grants pack onto the last eligible GWs (one per GW)."""
    fixtures = _fixtures_ordinary_through(19)
    rules = ChipRules.for_season("2025-26")
    inventory = ChipInventory(
        remaining={"wildcard": 0, "freehit": 1, "bboost": 1, "3xc": 1},
        played=[],
    )
    windows = schedule_chip_windows(
        fixtures,
        season="2024-25",
        rules=rules,
        as_of_gw=17,
        short_horizon=3,
        inventory=inventory,
        gameweeks=[17, 18, 19],
    )
    forced_gws = sorted(gw for weeks in windows.forced.values() for gw in weeks)
    assert forced_gws == [17, 18, 19]
    assert len(windows.forced) == 3
    assert set(windows.forced) <= {"freehit", "bboost", "3xc"}


def test_no_force_play_skips_half_window_packing() -> None:
    """``force_play=False`` leaves remaining chips unforced at the half end."""
    fixtures = _fixtures_ordinary_through(19)
    rules = ChipRules.for_season("2025-26")
    inventory = ChipInventory(
        remaining={"wildcard": 0, "freehit": 1, "bboost": 1, "3xc": 1},
        played=[],
    )
    packed = schedule_chip_windows(
        fixtures,
        season="2024-25",
        rules=rules,
        as_of_gw=17,
        short_horizon=3,
        inventory=inventory,
        gameweeks=[17, 18, 19],
        force_play=True,
    )
    free = schedule_chip_windows(
        fixtures,
        season="2024-25",
        rules=rules,
        as_of_gw=17,
        short_horizon=3,
        inventory=inventory,
        gameweeks=[17, 18, 19],
        force_play=False,
    )
    assert len(packed.forced) == 3
    assert free.forced == {}
    # Option-value still allows chips inside the short horizon.
    assert any(free.allows(chip, 17) for chip in ("freehit", "bboost", "3xc"))


def test_force_play_windows_respected_by_plan_squad() -> None:
    """plan_squad plays a chip forced on the half-window stop."""
    pool = _legal_pool()
    points = pool.select("player_key", pl.lit(0).alias("step"), "xpts")
    fixtures = _fixtures_ordinary_through(19)
    windows = schedule_chip_windows(
        fixtures,
        season="2024-25",
        rules=ChipRules.for_season("2025-26"),
        as_of_gw=19,
        short_horizon=1,
        inventory=ChipInventory(
            remaining={"wildcard": 0, "freehit": 0, "bboost": 1, "3xc": 0},
            played=[],
        ),
        gameweeks=[19],
    )
    assert windows.requires("bboost", 19)
    plan = plan_squad(
        pool.drop("xpts"),
        points,
        SquadState(
            bank=1000,
            free_transfers=1,
            ft_bank_cap=2,
            hit_cost=4,
            decay=1.0,
            bench_weight=0.0,
            owned=[],
            chips=ChipInventory(
                remaining={"wildcard": 0, "freehit": 0, "bboost": 1, "3xc": 0},
                played=[],
            ),
        ),
        initial=True,
        horizon=1,
        season="2025-26",
        gameweeks=[19],
        chip_windows=windows,
    )
    assert plan.chips == ("bboost",)


def test_as_of_out_of_range_raises() -> None:
    with pytest.raises(ValueError, match="as_of_gw"):
        schedule_chip_windows(
            pl.DataFrame(
                {
                    "season": ["2024-25"],
                    "gw": [1],
                    "team_h_key": [1],
                    "team_a_key": [2],
                }
            ),
            season="2024-25",
            rules=ChipRules.for_season("2025-26"),
            as_of_gw=0,
        )


def _fixtures_early_weak_late_strong() -> pl.DataFrame:
    """GW8 small double, GW10 large double (first-half window)."""
    rows: list[dict[str, object]] = []
    # GW8: one team doubles (teams 1–3).
    rows.extend(
        [
            {"season": "2024-25", "gw": 8, "team_h_key": 1, "team_a_key": 2},
            {"season": "2024-25", "gw": 8, "team_h_key": 1, "team_a_key": 3},
        ]
    )
    # GW9: ordinary.
    rows.append({"season": "2024-25", "gw": 9, "team_h_key": 1, "team_a_key": 2})
    # GW10: bigger double (four teams doubled).
    rows.extend(
        [
            {"season": "2024-25", "gw": 10, "team_h_key": 1, "team_a_key": 2},
            {"season": "2024-25", "gw": 10, "team_h_key": 1, "team_a_key": 3},
            {"season": "2024-25", "gw": 10, "team_h_key": 4, "team_a_key": 5},
            {"season": "2024-25", "gw": 10, "team_h_key": 2, "team_a_key": 4},
        ]
    )
    for gw in (11, 12, 13, 14):
        rows.append({"season": "2024-25", "gw": gw, "team_h_key": 1, "team_a_key": 2})
    return pl.DataFrame(rows)


def _fixtures_best_now() -> pl.DataFrame:
    """Biggest double is GW10; later weeks are ordinary."""
    rows: list[dict[str, object]] = [
        {"season": "2024-25", "gw": 10, "team_h_key": 1, "team_a_key": 2},
        {"season": "2024-25", "gw": 10, "team_h_key": 1, "team_a_key": 3},
        {"season": "2024-25", "gw": 10, "team_h_key": 4, "team_a_key": 5},
        {"season": "2024-25", "gw": 10, "team_h_key": 2, "team_a_key": 4},
    ]
    for gw in (11, 12, 13, 14, 15):
        rows.append({"season": "2024-25", "gw": gw, "team_h_key": 1, "team_a_key": 2})
    return pl.DataFrame(rows)


def _fixtures_first_half_and_stronger_second_half() -> pl.DataFrame:
    """Modest GW5 double plus a much larger GW25 double (cross-half trap)."""
    rows: list[dict[str, object]] = [
        {"season": "2024-25", "gw": 5, "team_h_key": 1, "team_a_key": 2},
        {"season": "2024-25", "gw": 5, "team_h_key": 1, "team_a_key": 3},
    ]
    for gw in list(range(6, 20)) + list(range(20, 25)) + list(range(26, 39)):
        rows.append({"season": "2024-25", "gw": gw, "team_h_key": 1, "team_a_key": 2})
    # GW25: four teams doubled — beats any first-half league_term if halves merge.
    rows.extend(
        [
            {"season": "2024-25", "gw": 25, "team_h_key": 1, "team_a_key": 2},
            {"season": "2024-25", "gw": 25, "team_h_key": 1, "team_a_key": 3},
            {"season": "2024-25", "gw": 25, "team_h_key": 4, "team_a_key": 5},
            {"season": "2024-25", "gw": 25, "team_h_key": 2, "team_a_key": 4},
        ]
    )
    return pl.DataFrame(rows)


def _fixtures_ordinary_first_half() -> pl.DataFrame:
    """Ordinary single fixtures across early first-half weeks."""
    return _fixtures_ordinary_through(14)


def _fixtures_ordinary_through(last_gw: int) -> pl.DataFrame:
    """One fixture per GW from 2..``last_gw`` (first-half force-play fixtures)."""
    rows = [
        {
            "season": "2024-25",
            "gw": gw,
            "team_h_key": 1,
            "team_a_key": 2,
        }
        for gw in range(2, last_gw + 1)
    ]
    return pl.DataFrame(rows)


def _legal_pool() -> pl.DataFrame:
    """Thirty-player draft pool (six clubs) matching plan chip tests."""
    rows: list[dict[str, object]] = []
    player = 1
    for club in range(1, 7):
        for position in ("GK", "DEF", "DEF", "MID", "FWD"):
            rows.append(
                {
                    "player_key": player,
                    "position": position,
                    "team_key": club,
                    "price": 40 + club,
                    "xpts": float(club),
                }
            )
            player += 1
    return pl.DataFrame(rows)
