"""Bench Boost and Triple Captain constraints in the multi-period MILP."""

from __future__ import annotations

import polars as pl
import pytest

from fplt.optimise.plan import plan_squad
from fplt.optimise.squad import select_squad
from fplt.optimise.state import ChipInventory, ChipPlay, OwnedPlayer, SquadState


def test_chips_stay_off_without_season_and_gameweeks() -> None:
    pool = _pool()
    points = pool.select("player_key", pl.lit(0).alias("step"), "xpts")
    plan = plan_squad(pool.drop("xpts"), points, _state(), initial=True, horizon=1)
    assert plan.chips == (None,)
    with_inventory = plan_squad(
        pool.drop("xpts"),
        points,
        _state(chips=_inventory(bboost=2, triple=2)),
        initial=True,
        horizon=1,
    )
    assert with_inventory.chips == (None,)
    assert with_inventory.points == pytest.approx(plan.points)


def test_season_and_gameweeks_must_match_horizon() -> None:
    pool = _pool().drop("xpts")
    points = _pool().select("player_key", pl.lit(0).alias("step"), "xpts")
    with pytest.raises(ValueError, match="together"):
        plan_squad(
            pool,
            points,
            _state(),
            initial=True,
            horizon=1,
            season="2025-26",
        )
    with pytest.raises(ValueError, match="one entry"):
        plan_squad(
            pool,
            points,
            _state(),
            initial=True,
            horizon=1,
            season="2025-26",
            gameweeks=[2, 3],
        )


def test_triple_captain_adds_a_third_captain_copy() -> None:
    pool = _pool()
    points = pool.select("player_key", pl.lit(0).alias("step"), "xpts")
    off = plan_squad(pool.drop("xpts"), points, _state(), initial=True, horizon=1)
    on = plan_squad(
        pool.drop("xpts"),
        points,
        _state(chips=_inventory(triple=1)),
        initial=True,
        horizon=1,
        season="2025-26",
        gameweeks=[5],
    )
    assert on.chips == ("3xc",)
    assert on.picks.filter(pl.col("captain")).height == 1
    assert on.points == pytest.approx(_xi_plus_captain(on) + _captain_xpts(on))
    assert on.points == pytest.approx(off.points + _captain_xpts(on))


def test_bench_boost_scores_the_bench() -> None:
    pool = _pool()
    points = pool.select("player_key", pl.lit(0).alias("step"), "xpts")
    on = plan_squad(
        pool.drop("xpts"),
        points,
        _state(chips=_inventory(bboost=1)),
        initial=True,
        horizon=1,
        season="2025-26",
        gameweeks=[5],
    )
    bench = on.picks.filter(~pl.col("sold") & ~pl.col("in_xi"))
    assert on.chips == ("bboost",)
    assert bench.height == 4
    assert on.points == pytest.approx(_xi_plus_captain(on) + float(bench["xpts"].sum()))


def test_one_chip_per_gameweek_even_when_both_are_valuable() -> None:
    pool = _pool()
    points = pool.select("player_key", pl.lit(0).alias("step"), "xpts")
    plan = plan_squad(
        pool.drop("xpts"),
        points,
        _state(chips=_inventory(bboost=1, triple=1)),
        initial=True,
        horizon=1,
        season="2025-26",
        gameweeks=[5],
    )
    assert plan.chips in {("bboost",), ("3xc",)}
    assert plan.chips.count("bboost") + plan.chips.count("3xc") == 1


def test_remaining_inventory_caps_chip_uses_across_the_horizon() -> None:
    pool = _pool()
    rows: list[dict[str, object]] = []
    keys = pool["player_key"].to_list()
    scores = pool["xpts"].to_list()
    for key, xpts in zip(keys, scores, strict=True):
        for step in (0, 1):
            rows.append({"player_key": key, "step": step, "xpts": float(xpts)})
    plan = plan_squad(
        pool.drop("xpts"),
        pl.DataFrame(rows),
        _state(chips=_inventory(triple=1)),
        initial=True,
        horizon=2,
        season="2025-26",
        gameweeks=[5, 6],
    )
    assert plan.chips.count("3xc") == 1
    assert plan.chips.count("bboost") == 0
    assert None in plan.chips


def test_half_season_window_blocks_chips_outside_grants() -> None:
    pool = _pool()
    points = pool.select("player_key", pl.lit(0).alias("step"), "xpts")
    # GW1 is before the 2025-26 first window (starts at GW2).
    early = plan_squad(
        pool.drop("xpts"),
        points,
        _state(chips=_inventory(bboost=1, triple=1)),
        initial=True,
        horizon=1,
        season="2025-26",
        gameweeks=[1],
    )
    assert early.chips == (None,)
    # First-half Triple Captain already played; second-half grant remains.
    mid = plan_squad(
        pool.drop("xpts"),
        points,
        _state(
            chips=ChipInventory(
                remaining={"wildcard": 0, "freehit": 0, "bboost": 0, "3xc": 1},
                played=[ChipPlay(name="3xc", gw=5)],
            )
        ),
        initial=True,
        horizon=1,
        season="2025-26",
        gameweeks=[10],
    )
    assert mid.chips == (None,)
    late = plan_squad(
        pool.drop("xpts"),
        points,
        _state(
            chips=ChipInventory(
                remaining={"wildcard": 0, "freehit": 0, "bboost": 0, "3xc": 1},
                played=[ChipPlay(name="3xc", gw=5)],
            )
        ),
        initial=True,
        horizon=1,
        season="2025-26",
        gameweeks=[20],
    )
    assert late.chips == ("3xc",)


def test_zero_remaining_keeps_chips_off_with_season() -> None:
    pool = _pool()
    points = pool.select("player_key", pl.lit(0).alias("step"), "xpts")
    plan = plan_squad(
        pool.drop("xpts"),
        points,
        _state(chips=_inventory()),
        initial=True,
        horizon=1,
        season="2025-26",
        gameweeks=[5],
    )
    assert plan.chips == (None,)


def test_transfer_plan_respects_one_chip_and_remaining() -> None:
    """Owned-squad path still satisfies chip caps (constraint check)."""
    pool = _pool()
    selected = select_squad(pool).players
    owned = [
        OwnedPlayer(player_key=int(key), selling_price=int(price))
        for key, price in selected.select("player_key", "price").iter_rows()
    ]
    bank = 1000 - int(selected["price"].sum())
    points = pool.select("player_key", pl.lit(0).alias("step"), "xpts")
    plan = plan_squad(
        pool.drop("xpts"),
        points,
        _state(
            bank=bank,
            owned=owned,
            chips=_inventory(bboost=1, triple=1),
        ),
        initial=False,
        horizon=1,
        season="2025-26",
        gameweeks=[8],
    )
    assert plan.chips in {("bboost",), ("3xc",)}
    assert sum(1 for chip in plan.chips if chip is not None) <= 1


def _captain_xpts(plan) -> float:
    return float(plan.picks.filter(pl.col("captain"))["xpts"][0])


def _xi_plus_captain(plan) -> float:
    xi = plan.picks.filter(~pl.col("sold") & pl.col("in_xi"))
    return float(xi["xpts"].sum()) + _captain_xpts(plan)


def _inventory(*, bboost: int = 0, triple: int = 0) -> ChipInventory:
    return ChipInventory(
        remaining={
            "wildcard": 0,
            "freehit": 0,
            "bboost": bboost,
            "3xc": triple,
        },
        played=[],
    )


def _state(**overrides: object) -> SquadState:
    values: dict[str, object] = {
        "bank": 1000,
        "free_transfers": 1,
        "ft_bank_cap": 2,
        "hit_cost": 4,
        "decay": 1.0,
        "bench_weight": 0.0,
        "owned": [],
        "chips": _inventory(),
    }
    values.update(overrides)
    return SquadState.model_validate(values)


def _pool() -> pl.DataFrame:
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
