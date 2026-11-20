"""Opaque predictions → transfer decisions via the public MILP boundary."""

import polars as pl
import pytest

from fplt.optimise.engine import TransferPlan, horizon_points, optimize_transfers
from fplt.optimise.squad import select_squad
from fplt.optimise.state import OwnedPlayer, SquadState


def test_optimize_transfers_drafts_a_legal_squad() -> None:
    pool = _pool()
    points = pool.select(
        "player_key",
        pl.lit(1).alias("gw"),
        "xpts",
    )
    plan = optimize_transfers(
        pool.drop("xpts"),
        points,
        _state(),
        gameweeks=[1],
    )
    kept = plan.picks.filter(~pl.col("sold"))
    assert kept.height == 15
    assert len(plan.xi) == 11
    assert plan.captain is not None
    assert plan.vice is not None
    assert plan.captain in plan.xi
    assert plan.vice in plan.xi
    assert plan.captain != plan.vice
    assert plan.hit == 0.0
    assert plan.is_draft
    assert plan.transfers_in == ()
    assert plan.transfers_out == ()
    assert plan.picks.filter(pl.col("bought")).height == 15
    counts = kept.group_by("position").len().sort("position")
    by_pos = {row["position"]: row["len"] for row in counts.iter_rows(named=True)}
    assert by_pos == {"DEF": 5, "FWD": 3, "GK": 2, "MID": 5}
    assert kept["team_key"].value_counts()["count"].max() <= 3
    assert int(kept["price"].sum()) <= 1000


def test_optimize_transfers_takes_a_hit_only_when_worth_it() -> None:
    worth = _swap_plan(6.0)
    assert len(worth.transfers_in) == 2
    assert len(worth.transfers_out) == 2
    assert worth.hit == 4.0
    declined = _swap_plan(2.0)
    assert len(declined.transfers_in) == 1
    assert declined.hit == 0.0


def test_optimize_transfers_banks_a_free_transfer() -> None:
    plan = _bank_plan()
    buys = [
        plan.picks.filter((pl.col("step") == step) & pl.col("bought")).height
        for step in (0, 1)
    ]
    assert buys == [0, 2]
    assert plan.hits == (0.0, 0.0)
    assert plan.free_transfers == (1, 2)
    assert plan.gameweeks == (1, 2)
    assert plan.transfers_in == ()


def test_horizon_points_maps_gameweeks_to_steps() -> None:
    frame = pl.DataFrame(
        {
            "player_key": [1, 1, 2],
            "gw": [3, 4, 3],
            "xpts": [1.5, 2.0, 0.5],
        }
    )
    mapped = horizon_points(frame, [3, 4])
    assert mapped.sort(["player_key", "step"]).to_dicts() == [
        {"player_key": 1, "step": 0, "xpts": 1.5},
        {"player_key": 1, "step": 1, "xpts": 2.0},
        {"player_key": 2, "step": 0, "xpts": 0.5},
    ]
    assert horizon_points(frame.clear(), [3]).is_empty()


def test_optimize_transfers_rejects_bad_horizons() -> None:
    pool = _pool().drop("xpts")
    points = pl.DataFrame(
        {"player_key": [1], "gw": [1], "xpts": [1.0]},
    )
    with pytest.raises(ValueError, match="non-empty"):
        optimize_transfers(pool, points, _state(), gameweeks=[])
    with pytest.raises(ValueError, match="repeat"):
        optimize_transfers(pool, points, _state(), gameweeks=[1, 1])
    with pytest.raises(ValueError, match="outside"):
        optimize_transfers(pool, points, _state(), gameweeks=[0])
    with pytest.raises(ValueError, match="no plan"):
        optimize_transfers(pool, points, _state(bank=0), gameweeks=[1])


def test_transfer_plan_exposes_first_week_armbands() -> None:
    plan = TransferPlan(
        gameweeks=(1,),
        picks=pl.DataFrame(
            {
                "step": [0, 0, 0],
                "player_key": [3, 1, 2],
                "position": ["MID", "GK", "DEF"],
                "team_key": [1, 1, 2],
                "price": [50, 40, 45],
                "xpts": [5.0, 2.0, 3.0],
                "in_xi": [True, True, False],
                "captain": [True, False, False],
                "vice": [False, True, False],
                "bought": [True, False, False],
                "sold": [False, False, True],
            }
        ),
        hits=(4.0,),
        free_transfers=(1,),
        points=10.0,
    )
    assert plan.transfers_in == (3,)
    assert plan.transfers_out == (2,)
    assert plan.xi == (1, 3)
    assert plan.captain == 3
    assert plan.vice == 1
    assert plan.hit == 4.0


def _state(**overrides: object) -> SquadState:
    values: dict[str, object] = {
        "bank": 1000,
        "free_transfers": 1,
        "ft_bank_cap": 2,
        "hit_cost": 4,
        "decay": 1.0,
        "bench_weight": 0.0,
        "owned": [],
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


def _swap_plan(outsider: float) -> TransferPlan:
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
            "gw": [1] * candidates.height,
            "xpts": [
                outsider if key in mirror_keys else 1.0 if key in owned_keys else 0.0
                for key in candidates["player_key"].to_list()
            ],
        }
    )
    owned = [
        OwnedPlayer(player_key=int(key), selling_price=int(price))
        for key, price in selected.select("player_key", "price").iter_rows()
    ]
    bank = 1000 - int(selected["price"].sum())
    return optimize_transfers(
        candidates,
        points,
        _state(bank=bank, owned=owned),
        gameweeks=[1],
    )


def _bank_plan() -> TransferPlan:
    pool = _pool()
    selected = select_squad(pool).players
    templates = selected.head(3)
    keys = templates["player_key"].to_list()
    mirrors = pl.concat(
        [
            templates.head(1).with_columns(
                (pl.col("player_key") + 1000).alias("player_key")
            ),
            templates.tail(2).with_columns(
                (pl.col("player_key") + 2000).alias("player_key")
            ),
        ]
    )
    candidates = pl.concat(
        [
            pool.select("player_key", "position", "team_key", "price"),
            mirrors.select("player_key", "position", "team_key", "price"),
        ]
    )
    owned_keys = set(selected["player_key"].to_list())
    small = {int(keys[0]) + 1000}
    big = {int(key) + 2000 for key in keys[1:]}
    rows: list[dict[str, object]] = []
    for key in candidates["player_key"].to_list():
        for gw in (1, 2):
            if key in small:
                xpts = 1.5 if gw == 1 else 1.0
            elif key in big:
                xpts = 0.0 if gw == 1 else 3.0
            elif key in owned_keys:
                xpts = 1.0
            else:
                xpts = 0.0
            rows.append({"player_key": key, "gw": gw, "xpts": xpts})
    owned = [
        OwnedPlayer(player_key=int(key), selling_price=int(price))
        for key, price in selected.select("player_key", "price").iter_rows()
    ]
    return optimize_transfers(
        candidates,
        pl.DataFrame(rows),
        _state(
            bank=1000 - int(selected["price"].sum()),
            owned=owned,
            bench_weight=0.005,
        ),
        gameweeks=[1, 2],
    )
