"""Candidate pool pruning for long-horizon MILPs."""

from __future__ import annotations

import polars as pl
import pytest

from fplt.optimise.plan import (
    DEFAULT_CANDIDATE_TOP_K,
    PRUNE_AUTO_MIN_HORIZON,
    plan_squad,
    prune_candidates,
)
from fplt.optimise.state import OwnedPlayer, SquadState


def _players() -> pl.DataFrame:
    """Build a deep midfield-heavy pool with a few cheap fillers."""
    rows: list[dict[str, object]] = []
    key = 1
    # 5 GKs: one premium, rest cheap/low xPts
    for i in range(5):
        rows.append(
            {
                "player_key": key,
                "position": "GK",
                "team_key": 1 + (i % 3),
                "price": 45 if i == 0 else 40,
                "sum_xpts": 20.0 - i,
            }
        )
        key += 1
    # 12 DEFs across clubs
    for i in range(12):
        rows.append(
            {
                "player_key": key,
                "position": "DEF",
                "team_key": 1 + (i % 4),
                "price": 40 + (i % 5),
                "sum_xpts": 15.0 - 0.5 * i,
            }
        )
        key += 1
    # 20 MIDs — top-K should drop the tail
    for i in range(20):
        rows.append(
            {
                "player_key": key,
                "position": "MID",
                "team_key": 1 + (i % 5),
                "price": 45 + (i % 6),
                "sum_xpts": 12.0 - 0.4 * i,
            }
        )
        key += 1
    # 8 FWDs
    for i in range(8):
        rows.append(
            {
                "player_key": key,
                "position": "FWD",
                "team_key": 1 + (i % 4),
                "price": 45 + i,
                "sum_xpts": 14.0 - 0.5 * i,
            }
        )
        key += 1
    return pl.DataFrame(rows)


def _points(players: pl.DataFrame, *, horizon: int = 5) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for row in players.iter_rows(named=True):
        per_step = float(row["sum_xpts"]) / horizon
        for step in range(horizon):
            rows.append(
                {
                    "player_key": int(row["player_key"]),
                    "step": step,
                    "xpts": per_step,
                }
            )
    return pl.DataFrame(rows)


def test_prune_keeps_top_k_owned_and_cheapest() -> None:
    """Top-K drops low-xPts MIDs; owned zero-xPts and cheapest stay."""
    players = _players()
    # Mark the worst MID as owned with zero points in the horizon frame.
    owned_key = int(
        players.filter(pl.col("position") == "MID")
        .sort("sum_xpts")
        .head(1)["player_key"][0]
    )
    points = _points(players).with_columns(
        pl.when(pl.col("player_key") == owned_key)
        .then(0.0)
        .otherwise(pl.col("xpts"))
        .alias("xpts")
    )
    pruned = prune_candidates(
        players.drop("sum_xpts"),
        points,
        owned_keys=[owned_key],
        top_k={"GK": 3, "DEF": 6, "MID": 5, "FWD": 4},
        keep_cheapest=2,
    )
    assert owned_key in pruned["player_key"].to_list()
    mid = pruned.filter(pl.col("position") == "MID")
    # 5 top-K + owned (if outside top) + up to 2 cheapest — well under 20.
    assert mid.height < 20
    assert mid.height >= 5
    # Cheapest two GKs (price 40) must survive keep_cheapest.
    cheap_gk = (
        players.filter((pl.col("position") == "GK") & (pl.col("price") == 40))
        .sort("player_key")
        .head(2)["player_key"]
        .to_list()
    )
    kept = set(pruned["player_key"].to_list())
    assert set(cheap_gk).issubset(kept)


def test_prune_rejects_thin_top_k() -> None:
    """top_k below squad minima is rejected early."""
    players = _players().drop("sum_xpts")
    with pytest.raises(ValueError, match="squad minimum"):
        prune_candidates(
            players,
            _points(_players()),
            top_k={"GK": 1, "DEF": 5, "MID": 5, "FWD": 3},
        )


def _draft_state(**overrides: object) -> SquadState:
    base: dict[str, object] = {
        "bank": 1000,
        "free_transfers": 1,
        "ft_bank_cap": 2,
        "hit_cost": 4,
        "decay": 1.0,
        "bench_weight": 0.0,
        "owned": [],
    }
    base.update(overrides)
    return SquadState.model_validate(base)


def test_plan_squad_auto_prunes_from_horizon_five() -> None:
    """H≥5 auto-prunes and still returns a legal fifteen."""
    assert PRUNE_AUTO_MIN_HORIZON == 5
    assert sum(DEFAULT_CANDIDATE_TOP_K.values()) == 160
    players = _players()
    cand = players.drop("sum_xpts")
    plan = plan_squad(
        cand,
        _points(players, horizon=5),
        _draft_state(),
        initial=True,
        horizon=5,
        prune=True,
    )
    assert plan.picks.filter((pl.col("step") == 0) & ~pl.col("sold")).height == 15
    plan_auto = plan_squad(
        cand,
        _points(players, horizon=5),
        _draft_state(),
        initial=True,
        horizon=5,
    )
    assert plan_auto.picks.filter((pl.col("step") == 0) & ~pl.col("sold")).height == 15


def test_plan_squad_prune_false_keeps_full_pool_solvable() -> None:
    """Explicit prune=False still solves on the unfiltered pool."""
    players = _players()
    plan = plan_squad(
        players.drop("sum_xpts"),
        _points(players, horizon=5),
        _draft_state(),
        initial=True,
        horizon=5,
        prune=False,
    )
    assert plan.picks.filter((pl.col("step") == 0) & ~pl.col("sold")).height == 15


def test_plan_squad_prune_retains_owned_after_zero_xpts() -> None:
    """Forced prune keeps an owned zero-xPts player in the solved step-0 rows."""
    players = _players()
    cand = players.drop("sum_xpts")
    seed = plan_squad(
        cand,
        _points(players, horizon=1),
        _draft_state(),
        initial=True,
        horizon=1,
        prune=False,
    )
    seed_keys = seed.picks.filter((pl.col("step") == 0) & ~pl.col("sold"))[
        "player_key"
    ].to_list()
    owned_key = int(
        players.filter(pl.col("position") == "MID")
        .sort("sum_xpts")
        .head(1)["player_key"][0]
    )
    if owned_key not in seed_keys:
        mid_in_seed = int(
            seed.picks.filter(
                (pl.col("step") == 0) & ~pl.col("sold") & (pl.col("position") == "MID")
            )["player_key"][0]
        )
        seed_keys = [owned_key if key == mid_in_seed else int(key) for key in seed_keys]
    owned = [
        OwnedPlayer(
            player_key=int(key),
            purchase_price=int(cand.filter(pl.col("player_key") == key)["price"][0]),
            selling_price=int(cand.filter(pl.col("player_key") == key)["price"][0]),
        )
        for key in seed_keys
    ]
    points = _points(players, horizon=5).with_columns(
        pl.when(pl.col("player_key") == owned_key)
        .then(0.0)
        .otherwise(pl.col("xpts"))
        .alias("xpts")
    )
    bank = 1000 - sum(player.selling_price for player in owned)
    plan = plan_squad(
        cand,
        points,
        _draft_state(bank=bank, free_transfers=1, owned=owned),
        initial=False,
        horizon=5,
        prune=True,
    )
    assert owned_key in plan.picks.filter(pl.col("step") == 0)["player_key"].to_list()
