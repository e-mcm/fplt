"""Plan-path lineup re-opt and p_haul captaincy (Wave D1 PR3)."""

from __future__ import annotations

import numpy as np
import polars as pl

from fplt.optimise.lineup import reoptimize_plan_lineups, select_lineup_scenarios
from fplt.optimise.plan import plan_squad
from fplt.optimise.scenarios import ScenarioBundle
from fplt.optimise.squad import select_squad
from fplt.optimise.state import SquadState, load_squad_state


def test_default_squad_state_bench_weight_is_zero() -> None:
    state = load_squad_state()
    assert state.bench_weight == 0.0


def test_select_squad_p_haul_breaks_captain_ties() -> None:
    pool = _draft_pool().with_columns(
        pl.when(pl.col("player_key") == 9)
        .then(0.95)
        .when(pl.col("player_key") == 10)
        .then(0.05)
        .otherwise(0.0)
        .alias("p_haul"),
        pl.lit(5.0).alias("xpts"),
    )
    chosen = select_squad(pool, haul_weight=1.0)
    captain = int(chosen.players.filter(pl.col("captain"))["player_key"][0])
    assert captain == 9


def test_plan_squad_lineup_reopt_default_returns_valid_lineup() -> None:
    pool = _draft_pool()
    points = pl.DataFrame(
        {
            "player_key": pool["player_key"].to_list() * 2,
            "step": [0] * pool.height + [1] * pool.height,
            "xpts": pool["xpts"].to_list() * 2,
        }
    )
    plan = plan_squad(
        pool.drop("xpts"),
        points,
        _state(),
        initial=True,
        horizon=2,
    )
    step0 = plan.picks.filter((pl.col("step") == 0) & ~pl.col("sold"))
    assert step0.height == 15
    assert step0.filter(pl.col("in_xi")).height == 11
    assert int(step0["captain"].sum()) == 1
    assert int(step0["vice"].sum()) == 1
    off = plan_squad(
        pool.drop("xpts"),
        points,
        _state(),
        initial=True,
        horizon=2,
        lineup_reopt=False,
    )
    assert off.picks.filter((pl.col("step") == 0) & pl.col("in_xi")).height == 11


def test_reoptimize_plan_lineups_prefers_scenario_insurance() -> None:
    fifteen = _fifteen()
    picks = fifteen.with_columns(
        pl.lit(0).alias("step"),
        pl.lit(True).alias("bought"),
        pl.lit(False).alias("sold"),
        pl.col("player_key").is_in(list(range(1, 12))).alias("in_xi"),
        (pl.col("player_key") == 2).alias("captain"),
        (pl.col("player_key") == 3).alias("vice"),
    )
    keys = fifteen["player_key"].to_list()
    n = len(keys)
    points = np.full((n, 2), 2.0)
    minutes = np.full((n, 2), 90, dtype=np.int64)
    i11 = keys.index(11)
    points[i11, 0] = 10.0
    points[i11, 1] = 0.0
    minutes[i11, 1] = 0
    i13 = keys.index(13)
    points[i13] = 6.0
    bundle = ScenarioBundle(
        player_keys=np.asarray(keys, dtype=np.int64),
        points=points,
        minutes=minutes,
    )
    direct = select_lineup_scenarios(fifteen, bundle)
    refreshed = reoptimize_plan_lineups(picks, scenarios=bundle)
    assert refreshed.filter(pl.col("in_xi")).height == 11
    assert set(refreshed.filter(pl.col("in_xi"))["player_key"].to_list()) == set(
        direct.players.filter(pl.col("in_xi"))["player_key"].to_list()
    )


def test_reoptimize_plan_lineups_p_haul_captain() -> None:
    fifteen = _fifteen().with_columns(pl.lit(5.0).alias("xpts"))
    picks = fifteen.with_columns(
        pl.lit(0).alias("step"),
        pl.lit(True).alias("bought"),
        pl.lit(False).alias("sold"),
        pl.col("player_key").is_in(list(range(1, 12))).alias("in_xi"),
        (pl.col("player_key") == 1).alias("captain"),
        (pl.col("player_key") == 2).alias("vice"),
    )
    haul = pl.DataFrame(
        {
            "player_key": fifteen["player_key"].to_list(),
            "step": [0] * 15,
            "p_haul": [
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.9,
                0.1,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
            ],
        }
    )
    refreshed = reoptimize_plan_lineups(picks, p_haul=haul, haul_weight=1.0)
    assert int(refreshed.filter(pl.col("captain"))["player_key"][0]) == 9


def _state() -> SquadState:
    return SquadState(
        bank=1000,
        free_transfers=1,
        ft_bank_cap=5,
        hit_cost=4,
        decay=1.0,
        bench_weight=0.0,
        owned=[],
    )


def _draft_pool() -> pl.DataFrame:
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
                    "xpts": float(club + (1 if position == "DEF" else 0)),
                }
            )
            player += 1
    return pl.DataFrame(rows)


def _fifteen() -> pl.DataFrame:
    positions = [
        "GK",
        "DEF",
        "DEF",
        "DEF",
        "MID",
        "MID",
        "MID",
        "MID",
        "FWD",
        "FWD",
        "FWD",
        "GK",
        "MID",
        "DEF",
        "DEF",
    ]
    return pl.DataFrame(
        {
            "player_key": list(range(1, 16)),
            "position": positions,
            "team_key": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 1, 2, 3],
            "price": [45] * 15,
            "xpts": [2.0] * 15,
        }
    )
