"""FPL 50% sell-on math and owned-price refresh."""

from __future__ import annotations

import polars as pl
import pytest

from fplt.optimise.prices import (
    apply_market_prices,
    fpl_selling_price,
    market_prices_from_candidates,
    refresh_owned_selling_prices,
)
from fplt.optimise.state import OwnedPlayer, SquadState


@pytest.mark.parametrize(
    ("purchase", "current", "expected"),
    [
        (50, 50, 50),
        (50, 51, 50),  # +0.1 → keep 0
        (50, 52, 51),  # +0.2 → keep 0.1
        (50, 60, 55),  # +1.0 → keep 0.5
        (50, 61, 55),  # +1.1 → keep 0.5 (floor)
        (50, 49, 49),  # fall: full loss
        (50, 40, 40),
        (100, 100, 100),
        (0, 0, 0),
        (0, 1, 0),
    ],
)
def test_fpl_selling_price_keep_half_rises_full_falls(
    purchase: int, current: int, expected: int
) -> None:
    assert fpl_selling_price(purchase, current) == expected


def test_fpl_selling_price_rejects_negatives() -> None:
    with pytest.raises(ValueError, match="prices must be >= 0"):
        fpl_selling_price(-1, 50)
    with pytest.raises(ValueError, match="prices must be >= 0"):
        fpl_selling_price(50, -1)


def test_refresh_owned_applies_sell_on() -> None:
    owned = [
        OwnedPlayer(player_key=1, purchase_price=50, selling_price=50),
        OwnedPlayer(player_key=2, purchase_price=80, selling_price=80),
        OwnedPlayer(player_key=3, purchase_price=60, selling_price=60),
    ]
    market = {1: 60, 2: 70, 3: 60}
    refreshed = refresh_owned_selling_prices(owned, market)
    assert refreshed[0].selling_price == 55
    assert refreshed[0].purchase_price == 50
    assert refreshed[1].selling_price == 70
    assert refreshed[2].selling_price == 60


def test_refresh_keeps_missing_market_rows() -> None:
    owned = [OwnedPlayer(player_key=9, purchase_price=40, selling_price=42)]
    assert refresh_owned_selling_prices(owned, {}) == owned


def test_apply_market_prices_updates_squad_state() -> None:
    state = SquadState(
        bank=10,
        free_transfers=1,
        ft_bank_cap=2,
        owned=[OwnedPlayer(player_key=1, purchase_price=50, selling_price=50)],
    )
    updated = apply_market_prices(state, {1: 54})
    assert updated.owned[0].selling_price == 52
    assert updated.owned[0].purchase_price == 50
    assert updated.bank == 10


def test_owned_player_defaults_purchase_to_selling() -> None:
    player = OwnedPlayer(player_key=1, selling_price=55)
    assert player.purchase_price == 55


def test_market_prices_from_candidates() -> None:
    frame = pl.DataFrame(
        {
            "player_key": [1, 2],
            "position": ["MID", "FWD"],
            "team_key": [1, 2],
            "price": [45, 90],
        }
    )
    assert market_prices_from_candidates(frame) == {1: 45, 2: 90}


def test_plan_squad_sell_on_funds_buy_after_rise() -> None:
    """Half of a list-price rise is spendable; frozen purchase would not be."""
    from fplt.optimise.plan import plan_squad
    from fplt.optimise.squad import select_squad

    base = _candidate_pool()
    selected = select_squad(base.with_columns(pl.lit(1.0).alias("xpts"))).players
    sell_key = int(selected["player_key"][0])
    sell_pos = str(selected["position"][0])
    sell_team = int(selected["team_key"][0])
    # Bought at 45; list rose to 55 → selling price 50. Replacement costs 50.
    # Bank 0 and every other owned sell credit still 45, so only sell-on works.
    purchase = 45
    market = 55
    sell_credit = fpl_selling_price(purchase, market)
    assert sell_credit == 50

    replacement_key = 9001
    candidates = pl.concat(
        [
            base.with_columns(
                pl.when(pl.col("player_key") == sell_key)
                .then(pl.lit(market))
                .otherwise(pl.col("price"))
                .alias("price")
            ),
            pl.DataFrame(
                {
                    "player_key": [replacement_key],
                    "position": [sell_pos],
                    "team_key": [sell_team + 50],
                    "price": [sell_credit],
                }
            ),
        ]
    )
    owned_keys = {int(key) for key in selected["player_key"].to_list()}
    owned = [
        OwnedPlayer(
            player_key=int(key),
            purchase_price=int(price),
            selling_price=int(price),
        )
        for key, price in selected.select("player_key", "price").iter_rows()
    ]
    state = SquadState(
        bank=0,
        free_transfers=1,
        ft_bank_cap=2,
        hit_cost=4,
        decay=1.0,
        bench_weight=0.0,
        owned=owned,
    )
    points = pl.DataFrame(
        {
            "player_key": candidates["player_key"].to_list(),
            "step": [0] * candidates.height,
            "xpts": [
                20.0
                if key == replacement_key
                else 0.0
                if key == sell_key
                else 1.0
                if key in owned_keys
                else 0.0
                for key in candidates["player_key"].to_list()
            ],
        }
    )
    plan = plan_squad(candidates, points, state, initial=False, horizon=1)
    bought = set(plan.picks.filter(pl.col("bought"))["player_key"].to_list())
    sold = set(plan.picks.filter(pl.col("sold"))["player_key"].to_list())
    assert sell_key in sold
    assert replacement_key in bought


def test_advance_keeps_purchase_and_applies_sell_on() -> None:
    """`_advance` sticks purchase_price and credits half of list rises."""
    from fplt.optimise.decisions import _advance
    from fplt.optimise.plan import GameweekPlan

    owned = [
        OwnedPlayer(player_key=key, purchase_price=45, selling_price=45)
        for key in range(1, 16)
    ]
    state = SquadState(
        bank=10,
        free_transfers=1,
        ft_bank_cap=2,
        hit_cost=4,
        decay=1.0,
        bench_weight=0.0,
        owned=owned,
    )
    # Keep 1..14 at risen list 55; sell 15 for sell-on 50; buy 99 at 50.
    rows: list[dict[str, object]] = []
    for key in range(1, 15):
        rows.append(
            {
                "step": 0,
                "player_key": key,
                "position": "MID",
                "team_key": 1,
                "price": 55,
                "xpts": 1.0,
                "in_xi": key <= 11,
                "captain": key == 1,
                "vice": key == 2,
                "bought": False,
                "sold": False,
            }
        )
    rows.append(
        {
            "step": 0,
            "player_key": 15,
            "position": "MID",
            "team_key": 1,
            "price": 55,
            "xpts": 0.0,
            "in_xi": False,
            "captain": False,
            "vice": False,
            "bought": False,
            "sold": True,
        }
    )
    rows.append(
        {
            "step": 0,
            "player_key": 99,
            "position": "MID",
            "team_key": 2,
            "price": 50,
            "xpts": 5.0,
            "in_xi": False,
            "captain": False,
            "vice": False,
            "bought": True,
            "sold": False,
        }
    )
    plan = GameweekPlan(
        picks=pl.DataFrame(rows),
        hits=(0.0,),
        free_transfers=(1,),
        points=10.0,
        chips=(None,),
    )
    pool = pl.DataFrame(
        {
            "player_key": list(range(1, 16)) + [99],
            "position": ["MID"] * 16,
            "team_key": [1] * 15 + [2],
            "price": [55] * 15 + [50],
        }
    )
    next_state = _advance(state, plan, pool, gameweek=2)
    by_key = {player.player_key: player for player in next_state.owned}
    assert 15 not in by_key
    assert by_key[1].purchase_price == 45
    assert by_key[1].selling_price == 50  # 45 + (55-45)//2
    assert by_key[99].purchase_price == 50
    assert by_key[99].selling_price == 50
    # bank 10 + sell 50 - buy 50
    assert next_state.bank == 10


def _candidate_pool() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    key = 1
    for position, count in (("GK", 4), ("DEF", 10), ("MID", 10), ("FWD", 6)):
        for index in range(count):
            rows.append(
                {
                    "player_key": key,
                    "position": position,
                    "team_key": (index % 5) + 1,
                    "price": 45,
                }
            )
            key += 1
    return pl.DataFrame(rows)
