"""Chip constraints in the multi-period MILP (BB/TC/WC/FH)."""

from __future__ import annotations

import polars as pl
import pytest

from fplt.optimise.chip_windows import ChipUseWindows
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


def test_wildcard_suppresses_hits_on_many_transfers() -> None:
    """Wildcard lets the solver take hit-sized moves at zero hit cost."""
    candidates, points, owned, bank = _mirror_swap(gain=2.0)
    off = plan_squad(
        candidates,
        points,
        _state(bank=bank, owned=owned, free_transfers=1, chips=_inventory()),
        initial=False,
        horizon=1,
        season="2025-26",
        gameweeks=[5],
    )
    assert off.picks.filter(pl.col("bought")).height == 1
    assert off.hits == (0.0,)
    on = plan_squad(
        candidates,
        points,
        _state(
            bank=bank,
            owned=owned,
            free_transfers=1,
            chips=_inventory(wildcard=1),
        ),
        initial=False,
        horizon=1,
        season="2025-26",
        gameweeks=[5],
    )
    assert on.chips == ("wildcard",)
    assert on.hits == (0.0,)
    assert on.picks.filter(pl.col("bought")).height >= 2


def test_wildcard_resets_free_transfers_next_week() -> None:
    pool, owned, bank = _owned_squad()
    # Make week-0 transfers valuable so WC is used; week 1 flat.
    plan = plan_squad(
        pool.drop("xpts"),
        _wildcard_horizon_points(pool),
        _state(
            bank=bank,
            owned=owned,
            free_transfers=2,
            ft_bank_cap=2,
            chips=_inventory(wildcard=1),
        ),
        initial=False,
        horizon=2,
        season="2025-26",
        gameweeks=[5, 6],
    )
    assert plan.chips[0] == "wildcard"
    assert plan.hits[0] == 0.0
    assert plan.free_transfers == (2, 1)


def test_free_hit_reverts_squad_next_week() -> None:
    pool, owned, bank = _owned_squad()
    owned_keys = {player.player_key for player in owned}
    plan = plan_squad(
        pool.drop("xpts"),
        _free_hit_horizon_points(pool, owned_keys),
        _state(
            bank=bank,
            owned=owned,
            free_transfers=1,
            chips=_inventory(freehit=1),
        ),
        initial=False,
        horizon=2,
        season="2025-26",
        gameweeks=[5, 6],
    )
    assert plan.chips[0] == "freehit"
    assert plan.hits == (0.0, 0.0)
    week0 = set(
        plan.picks.filter((pl.col("step") == 0) & ~pl.col("sold"))["player_key"]
    )
    week1 = set(
        plan.picks.filter((pl.col("step") == 1) & ~pl.col("sold"))["player_key"]
    )
    assert week0 != owned_keys
    assert week1 == owned_keys


def test_free_hit_banks_free_transfers_through() -> None:
    pool, owned, bank = _owned_squad()
    owned_keys = {player.player_key for player in owned}
    plan = plan_squad(
        pool.drop("xpts"),
        _free_hit_horizon_points(pool, owned_keys),
        _state(
            bank=bank,
            owned=owned,
            free_transfers=1,
            ft_bank_cap=2,
            chips=_inventory(freehit=1),
        ),
        initial=False,
        horizon=2,
        season="2025-26",
        gameweeks=[5, 6],
    )
    assert plan.chips[0] == "freehit"
    # Unused FT banks: 1 → 2 under the cap.
    assert plan.free_transfers == (1, 2)


def test_one_chip_per_gameweek_includes_wildcard_and_free_hit() -> None:
    pool, owned, bank = _owned_squad()
    owned_keys = {player.player_key for player in owned}
    points = _free_hit_horizon_points(pool, owned_keys).filter(pl.col("step") == 0)
    plan = plan_squad(
        pool.drop("xpts"),
        points,
        _state(
            bank=bank,
            owned=owned,
            free_transfers=1,
            chips=_inventory(wildcard=1, freehit=1, bboost=1, triple=1),
        ),
        initial=False,
        horizon=1,
        season="2025-26",
        gameweeks=[5],
    )
    assert sum(1 for chip in plan.chips if chip is not None) == 1


def test_no_chip_window_blocks_all_chips() -> None:
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
        chip_windows=ChipUseWindows(no_chip=frozenset({5})),
    )
    assert plan.chips == (None,)


def test_allowed_chip_gws_restrict_playable_weeks() -> None:
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
        chip_windows=ChipUseWindows(allowed={"3xc": frozenset({6})}),
    )
    assert plan.chips == (None, "3xc")


def test_forced_chip_gws_require_the_chip() -> None:
    pool = _pool()
    points = pool.select("player_key", pl.lit(0).alias("step"), "xpts")
    plan = plan_squad(
        pool.drop("xpts"),
        points,
        _state(chips=_inventory(bboost=1)),
        initial=True,
        horizon=1,
        season="2025-26",
        gameweeks=[5],
        chip_windows=ChipUseWindows(
            allowed={"bboost": frozenset({5})},
            forced={"bboost": frozenset({5})},
        ),
    )
    assert plan.chips == ("bboost",)


def test_chip_rules_season_can_differ_from_fixture_season() -> None:
    """Counterfactual: 2024-25 fixtures under 2025-26 two-set rules."""
    pool = _pool()
    points = pool.select("player_key", pl.lit(0).alias("step"), "xpts")
    plan = plan_squad(
        pool.drop("xpts"),
        points,
        _state(chips=_inventory(triple=1)),
        initial=True,
        horizon=1,
        season="2024-25",
        gameweeks=[5],
        chip_rules_season="2025-26",
    )
    assert plan.chips == ("3xc",)


def test_wildcard_remaining_inventory_is_respected() -> None:
    pool, owned, bank = _owned_squad()
    plan = plan_squad(
        pool.drop("xpts"),
        _wildcard_horizon_points(pool),
        _state(
            bank=bank,
            owned=owned,
            free_transfers=1,
            chips=_inventory(wildcard=0),
        ),
        initial=False,
        horizon=2,
        season="2025-26",
        gameweeks=[5, 6],
    )
    assert plan.chips.count("wildcard") == 0


def _captain_xpts(plan) -> float:
    return float(plan.picks.filter(pl.col("captain"))["xpts"][0])


def _xi_plus_captain(plan) -> float:
    xi = plan.picks.filter(~pl.col("sold") & pl.col("in_xi"))
    return float(xi["xpts"].sum()) + _captain_xpts(plan)


def _inventory(
    *,
    wildcard: int = 0,
    freehit: int = 0,
    bboost: int = 0,
    triple: int = 0,
) -> ChipInventory:
    return ChipInventory(
        remaining={
            "wildcard": wildcard,
            "freehit": freehit,
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


def _owned_squad() -> tuple[pl.DataFrame, list[OwnedPlayer], int]:
    pool = _pool()
    selected = select_squad(pool).players
    owned = [
        OwnedPlayer(player_key=int(key), selling_price=int(price))
        for key, price in selected.select("player_key", "price").iter_rows()
    ]
    bank = 1000 - int(selected["price"].sum())
    return pool, owned, bank


def _mirror_swap(
    *, gain: float
) -> tuple[pl.DataFrame, pl.DataFrame, list[OwnedPlayer], int]:
    """Two mirror replacements worth ``gain`` each (same pattern as test_plan)."""
    pool = _pool()
    selected = select_squad(pool).players
    templates = selected.head(2)
    mirrors = templates.with_columns((pl.col("player_key") + 1000).alias("player_key"))
    candidates = pl.concat(
        [
            pool.select("player_key", "position", "team_key", "price"),
            mirrors.select("player_key", "position", "team_key", "price"),
        ]
    )
    owned_keys = set(selected["player_key"].to_list())
    mirror_keys = set(mirrors["player_key"].to_list())
    points = pl.DataFrame(
        {
            "player_key": candidates["player_key"].to_list(),
            "step": [0] * candidates.height,
            "xpts": [
                gain if key in mirror_keys else 1.0 if key in owned_keys else 0.0
                for key in candidates["player_key"].to_list()
            ],
        }
    )
    owned = [
        OwnedPlayer(player_key=int(key), selling_price=int(price))
        for key, price in selected.select("player_key", "price").iter_rows()
    ]
    bank = 1000 - int(selected["price"].sum())
    return candidates, points, owned, bank


def _wildcard_horizon_points(pool: pl.DataFrame) -> pl.DataFrame:
    """Week 0: outsiders haul; week 1: flat — WC pays for a multi-buy redraft."""
    selected = set(select_squad(pool).players["player_key"].to_list())
    rows: list[dict[str, object]] = []
    for row in pool.iter_rows(named=True):
        key = int(row["player_key"])
        base = float(row["xpts"])
        week0 = base + (20.0 if key not in selected else 0.0)
        rows.append({"player_key": key, "step": 0, "xpts": week0})
        rows.append({"player_key": key, "step": 1, "xpts": base})
    return pl.DataFrame(rows)


def _free_hit_horizon_points(pool: pl.DataFrame, owned_keys: set[int]) -> pl.DataFrame:
    """Week 0: non-owned haul (FH); week 1: owned haul again (revert)."""
    rows: list[dict[str, object]] = []
    for row in pool.iter_rows(named=True):
        key = int(row["player_key"])
        base = float(row["xpts"])
        week0 = base + (25.0 if key not in owned_keys else 0.0)
        week1 = base + (25.0 if key in owned_keys else 0.0)
        rows.append({"player_key": key, "step": 0, "xpts": week0})
        rows.append({"player_key": key, "step": 1, "xpts": week1})
    return pl.DataFrame(rows)
