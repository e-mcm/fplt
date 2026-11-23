"""Within-horizon list-price forecasts and plan wiring."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from fplt.data.pit import PITView
from fplt.optimise.price_forecast import (
    forecast_list_prices,
    pit_deadline_prices,
)
from fplt.optimise.prices import (
    fpl_selling_price,
    list_price_matrix,
    sell_price_matrix,
)
from fplt.optimise.state import OwnedPlayer
from fplt.types import GWKey


def test_flat_forecast_copies_as_of() -> None:
    frame = forecast_list_prices({1: 50, 2: 80}, horizon=3, method="flat")
    assert frame.height == 6
    assert set(frame.filter(pl.col("step") == 0)["list_price"].to_list()) == {50, 80}
    assert set(frame.filter(pl.col("step") == 2)["list_price"].to_list()) == {50, 80}


def test_mean_gw_delta_accumulates_rounded_steps() -> None:
    history = pl.DataFrame(
        {
            "player_key": [1, 1, 1, 2, 2, 2],
            "gw": [1, 2, 3, 1, 2, 3],
            "list_price": [50, 52, 54, 80, 80, 79],
        }
    )
    frame = forecast_list_prices(
        {1: 54, 2: 79},
        horizon=3,
        method="mean_gw_delta",
        deadline_prices=history,
        delta_lookback=2,
    )
    by = {
        (int(key), int(step)): int(price)
        for key, step, price in frame.select(
            "player_key", "step", "list_price"
        ).iter_rows()
    }
    # Player 1: last two deltas +2,+2 → mean 2; steps 0/1/2 → 54/56/58.
    assert by[(1, 0)] == 54
    assert by[(1, 1)] == 56
    assert by[(1, 2)] == 58
    # Player 2: last two deltas 0,-1 → mean -0.5; 79 / 79 / 78.
    assert by[(2, 0)] == 79
    assert by[(2, 1)] == 78  # round(79 - 0.5) = 78
    assert by[(2, 2)] == 78  # round(79 - 1.0) = 78


def test_mean_gw_delta_without_history_is_flat() -> None:
    frame = forecast_list_prices(
        {7: 45}, horizon=2, method="mean_gw_delta", deadline_prices=None
    )
    assert frame["list_price"].to_list() == [45, 45]


def test_forecast_rejects_bad_inputs() -> None:
    with pytest.raises(ValueError, match="horizon"):
        forecast_list_prices({1: 40}, horizon=0)
    with pytest.raises(ValueError, match="as_of_prices"):
        forecast_list_prices({}, horizon=1)
    with pytest.raises(ValueError, match="unknown price forecast method"):
        forecast_list_prices({1: 40}, horizon=1, method="magic")  # type: ignore[arg-type]


def test_list_price_matrix_none_matches_flat() -> None:
    keys = [1, 2]
    as_of = {1: 50, 2: 60}
    flat = forecast_list_prices(as_of, horizon=2, method="flat")
    left = list_price_matrix(keys, 2, as_of, price_forecast=None)
    right = list_price_matrix(keys, 2, as_of, price_forecast=flat)
    assert left.tolist() == right.tolist() == [[50.0, 60.0], [50.0, 60.0]]


def test_list_price_matrix_rejects_step0_mismatch() -> None:
    forecast = pl.DataFrame(
        {
            "player_key": [1, 1],
            "step": [0, 1],
            "list_price": [51, 52],
        }
    )
    with pytest.raises(ValueError, match="step 0"):
        list_price_matrix([1], 2, {1: 50}, price_forecast=forecast)


def test_sell_price_matrix_applies_sell_on_per_step() -> None:
    list_prices = list_price_matrix(
        [1, 2],
        2,
        {1: 50, 2: 80},
        price_forecast=pl.DataFrame(
            {
                "player_key": [1, 1, 2, 2],
                "step": [0, 1, 0, 1],
                "list_price": [50, 60, 80, 70],
            }
        ),
    )
    owned = [OwnedPlayer(player_key=1, purchase_price=50, selling_price=50)]
    sell = sell_price_matrix(list_prices, owned, {1: 0, 2: 1})
    assert sell[0, 0] == float(fpl_selling_price(50, 50))
    assert sell[1, 0] == float(fpl_selling_price(50, 60))
    assert sell[0, 1] == 80.0
    assert sell[1, 1] == 70.0


class _MemorySource:
    """Minimal table source for PIT deadline-price tests."""

    def __init__(self, tables: dict[str, pl.DataFrame], deadlines: dict[int, datetime]):
        self._tables = tables
        self._deadlines = deadlines

    def table(self, name: str) -> pl.LazyFrame:
        return self._tables[name].lazy()

    def deadline(self, gw: GWKey) -> datetime:
        return self._deadlines[gw.gw]


def test_pit_deadline_prices_hides_future_snapshots() -> None:
    t0 = datetime(2025, 8, 1, 12, 0, tzinfo=UTC)
    t1 = t0 + timedelta(days=7)
    t2 = t0 + timedelta(days=14)
    source = _MemorySource(
        {
            "gameweeks": pl.DataFrame(
                {
                    "season": ["2025-26", "2025-26", "2025-26"],
                    "gw": [1, 2, 3],
                    "deadline_time": [t0, t1, t2],
                    "available_at": [t0, t1, t2],
                }
            ),
            "player_status": pl.DataFrame(
                {
                    "season": ["2025-26"] * 4,
                    "player_key": [1, 1, 1, 1],
                    "price": [50, 51, 52, 99],
                    "available_at": [
                        t0 - timedelta(hours=1),
                        t1 - timedelta(hours=1),
                        t1 + timedelta(hours=1),  # after GW2 deadline
                        t2 - timedelta(hours=1),  # future vs as_of GW2
                    ],
                    "status": ["a"] * 4,
                    "news": [""] * 4,
                    "news_added": [None] * 4,
                    "ep_next": [None] * 4,
                    "chance_of_playing": [None] * 4,
                    "chance_of_playing_this_round": [None] * 4,
                    "penalties_order": [None] * 4,
                    "source": ["test"] * 4,
                }
            ),
        },
        {1: t0, 2: t1, 3: t2},
    )
    view = PITView(as_of=GWKey("2025-26", 2), cutoff=t1, _source=source)
    prices = pit_deadline_prices(view)
    # As-of GW2: only deadlines ≤ t1; GW2 price is 51 (pre-deadline), not 52/99.
    assert prices.sort("gw")["list_price"].to_list() == [50, 51]
    assert prices["gw"].to_list() == [1, 2]


def test_plan_squad_price_forecast_none_matches_flat() -> None:
    from fplt.optimise.plan import plan_squad
    from fplt.optimise.squad import select_squad
    from fplt.optimise.state import SquadState

    base = _candidate_pool()
    selected = select_squad(base.with_columns(pl.lit(1.0).alias("xpts"))).players
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
    keys = base["player_key"].to_list()
    owned_keys = {int(k) for k in selected["player_key"]}
    points = pl.DataFrame(
        {
            "player_key": keys * 2,
            "step": [0] * len(keys) + [1] * len(keys),
            "xpts": [1.0 if key in owned_keys else 0.0 for key in keys] * 2,
        }
    )
    flat = forecast_list_prices(
        {int(k): int(p) for k, p in base.select("player_key", "price").iter_rows()},
        horizon=2,
        method="flat",
    )
    control = plan_squad(
        base, points, state, initial=False, horizon=2, prune=False, lineup_reopt=False
    )
    treatment = plan_squad(
        base,
        points,
        state,
        initial=False,
        horizon=2,
        prune=False,
        lineup_reopt=False,
        price_forecast=flat,
    )
    assert control.picks.sort(["step", "player_key"]).equals(
        treatment.picks.sort(["step", "player_key"])
    )


def test_plan_squad_forecast_rise_funds_later_buy() -> None:
    """Predicted step-1 sell-on can fund a buy that step-0 cash cannot."""
    from fplt.optimise.plan import plan_squad
    from fplt.optimise.squad import select_squad
    from fplt.optimise.state import SquadState

    base = _candidate_pool()
    selected = select_squad(base.with_columns(pl.lit(1.0).alias("xpts"))).players
    sell_key = int(selected.filter(pl.col("position") == "MID")["player_key"][0])
    sell_team = int(selected.filter(pl.col("player_key") == sell_key)["team_key"][0])
    purchase = 45
    # Step 0 list 45 → sell 45; step 1 list 55 → sell-on 50.
    replacement_key = 9001
    candidates = pl.concat(
        [
            base,
            pl.DataFrame(
                {
                    "player_key": [replacement_key],
                    "position": ["MID"],
                    "team_key": [sell_team + 50],
                    "price": [50],
                }
            ),
        ]
    )
    owned_keys = {int(key) for key in selected["player_key"].to_list()}
    owned = [
        OwnedPlayer(
            player_key=int(key),
            purchase_price=purchase,
            selling_price=purchase,
        )
        for key in owned_keys
    ]
    state = SquadState(
        bank=0,
        free_transfers=2,
        ft_bank_cap=2,
        hit_cost=4,
        decay=1.0,
        bench_weight=0.0,
        owned=owned,
    )
    # Prefer keeping everyone at step 0; at step 1 the risen sell + replacement wins.
    rows: list[dict[str, object]] = []
    for step in (0, 1):
        for key in candidates["player_key"].to_list():
            if key == replacement_key:
                xpts = 0.0 if step == 0 else 30.0
            elif key == sell_key:
                xpts = 1.0 if step == 0 else 0.0
            elif key in owned_keys:
                xpts = 1.0
            else:
                xpts = 0.0
            rows.append({"player_key": key, "step": step, "xpts": xpts})
    points = pl.DataFrame(rows)
    as_of = {
        int(key): int(price)
        for key, price in candidates.select("player_key", "price").iter_rows()
    }
    forecast_rows: list[dict[str, int]] = []
    for key, price in as_of.items():
        forecast_rows.append({"player_key": key, "step": 0, "list_price": price})
        step1 = 55 if key == sell_key else price
        forecast_rows.append({"player_key": key, "step": 1, "list_price": step1})
    forecast = pl.DataFrame(forecast_rows)

    # Without forecast, selling at 45 cannot buy the 50 replacement.
    flat_plan = plan_squad(
        candidates,
        points,
        state,
        initial=False,
        horizon=2,
        prune=False,
        lineup_reopt=False,
    )
    flat_bought = set(
        flat_plan.picks.filter((pl.col("step") == 1) & pl.col("bought"))[
            "player_key"
        ].to_list()
    )
    assert replacement_key not in flat_bought

    risen = plan_squad(
        candidates,
        points,
        state,
        initial=False,
        horizon=2,
        prune=False,
        lineup_reopt=False,
        price_forecast=forecast,
    )
    step1 = risen.picks.filter(pl.col("step") == 1)
    bought = set(step1.filter(pl.col("bought"))["player_key"].to_list())
    sold = set(step1.filter(pl.col("sold"))["player_key"].to_list())
    assert sell_key in sold
    assert replacement_key in bought


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
