"""Candidate pool pruning for long-horizon MILPs."""

from __future__ import annotations

import math

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fplt.optimise.candidate_prune import (
    DEFAULT_KEEP_CHEAPEST,
    DEFAULT_MIN_P_60,
    DEFAULT_PRUNE_POLICY,
    DEFAULT_TOP_PER_STEP,
    DEFAULT_TOP_VALUE,
    DEFAULT_TOP_WINDOW,
    DEFAULT_TOP_XPTS,
    DEFAULT_WINDOW_LENGTHS,
    MAX_PER_CLUB,
    MIN_POS_POOL,
    CandidatePrunePolicy,
    _rank_players,
    drop_dominated_players,
)
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


def _tight_policy(**overrides: object) -> CandidatePrunePolicy:
    """Small keep quotas for unit tests on the synthetic pool.

    Step / window tops are off by default so older assertions stay
    focused; enable them in the dedicated per-step tests.
    """
    base: dict[str, object] = {
        "top_xpts": {"GK": 3, "DEF": 6, "MID": 5, "FWD": 4},
        "top_value": {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
        "top_per_step": None,
        "top_window": None,
        "window_lengths": (),
        "keep_cheapest": 2,
    }
    base.update(overrides)
    return CandidatePrunePolicy.model_validate(base)


def test_default_policy_knobs_target_about_130() -> None:
    """Defaults are the ~130-player union including step / window tops."""
    assert DEFAULT_KEEP_CHEAPEST == 4
    assert DEFAULT_MIN_P_60 == 0.5
    assert dict(DEFAULT_TOP_XPTS) == {"GK": 16, "DEF": 36, "MID": 46, "FWD": 23}
    assert dict(DEFAULT_TOP_VALUE) == {"GK": 10, "DEF": 23, "MID": 29, "FWD": 13}
    assert dict(DEFAULT_TOP_PER_STEP) == {"GK": 2, "DEF": 5, "MID": 6, "FWD": 3}
    assert dict(DEFAULT_TOP_WINDOW) == {"GK": 2, "DEF": 5, "MID": 6, "FWD": 3}
    assert DEFAULT_WINDOW_LENGTHS == (2, 3)
    assert DEFAULT_PRUNE_POLICY.top_xpts == dict(DEFAULT_TOP_XPTS)
    assert DEFAULT_PRUNE_POLICY.top_value == dict(DEFAULT_TOP_VALUE)
    assert DEFAULT_PRUNE_POLICY.top_per_step == dict(DEFAULT_TOP_PER_STEP)
    assert DEFAULT_PRUNE_POLICY.top_window == dict(DEFAULT_TOP_WINDOW)
    assert DEFAULT_PRUNE_POLICY.window_lengths == DEFAULT_WINDOW_LENGTHS
    assert DEFAULT_PRUNE_POLICY.min_p_60 == DEFAULT_MIN_P_60
    assert sum(DEFAULT_CANDIDATE_TOP_K.values()) == 160


def test_prune_keeps_top_xpts_value_owned_and_cheapest() -> None:
    """Union keeps top xPts, a high-PAR mid, owned, and cheapest fillers."""
    players = _players()
    mids = players.filter(pl.col("position") == "MID").sort("sum_xpts", descending=True)
    # Outside tight top_xpts=5; mid price premium over *r* → elite PAR.
    value_mid = int(mids.row(6, named=True)["player_key"])
    # Two lowest-xPts MIDs: playable replacement *r* (owned) + a filler so
    # keep_cheapest cannot be what saves value_mid.
    replacement = int(mids.row(-1, named=True)["player_key"])
    cheap_filler = int(mids.row(-2, named=True)["player_key"])
    owned_key = replacement
    cand = players.drop("sum_xpts").with_columns(
        pl.when(pl.col("player_key") == replacement)
        .then(40)
        .when(pl.col("player_key") == cheap_filler)
        .then(41)
        .when(pl.col("player_key") == value_mid)
        .then(55)
        .when(pl.col("position") == "MID")
        .then(70)
        .otherwise(pl.col("price"))
        .alias("price")
    )
    points = _points(players).with_columns(
        pl.when(pl.col("player_key") == owned_key)
        .then(0.0)
        .otherwise(pl.col("xpts"))
        .alias("xpts"),
        pl.lit(0.8).alias("p_60"),
    )
    policy = _tight_policy()
    pruned = policy.apply(cand, points, owned_keys=[owned_key])
    kept = set(pruned["player_key"].to_list())
    assert owned_key in kept
    assert value_mid in kept
    assert cheap_filler in kept  # keep_cheapest path
    # value_mid is outside top_xpts and outside keep_cheapest (price 55).
    xpts_and_cheap = CandidatePrunePolicy(
        top_xpts={"GK": 3, "DEF": 6, "MID": 5, "FWD": 4},
        top_value=None,
        top_per_step=None,
        top_window=None,
        window_lengths=(),
        keep_cheapest=2,
        min_p_60=0.5,
    )
    without_par = set(
        xpts_and_cheap.apply(cand, points, owned_keys=[owned_key])["player_key"]
    )
    assert value_mid not in without_par
    mid = pruned.filter(pl.col("position") == "MID")
    assert mid.height < 20
    assert mid.height >= 5
    cheap_gk = (
        players.filter((pl.col("position") == "GK") & (pl.col("price") == 40))
        .sort("player_key")
        .head(2)["player_key"]
        .to_list()
    )
    assert set(cheap_gk).issubset(kept)


def test_legacy_top_k_kwargs_still_work() -> None:
    """top_k= / keep_cheapest= build the #132-compatible policy."""
    players = _players()
    pruned = prune_candidates(
        players.drop("sum_xpts"),
        _points(players),
        top_k={"GK": 3, "DEF": 6, "MID": 5, "FWD": 4},
        keep_cheapest=2,
    )
    assert pruned.filter(pl.col("position") == "MID").height <= 7


def test_prune_rejects_thin_top_xpts() -> None:
    """top_xpts below squad minima is rejected early."""
    with pytest.raises(ValueError, match="squad minimum"):
        CandidatePrunePolicy(
            top_xpts={"GK": 1, "DEF": 5, "MID": 5, "FWD": 3},
            top_value=None,
            keep_cheapest=0,
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


def test_plan_squad_auto_prunes_from_horizon_three() -> None:
    """H≥3 (live default) auto-prunes and still returns a legal fifteen.

    Regression for forecast-combine H=3 continuous hangs: full-pool MILPs
    at the live horizon must shortlist via the default prune policy.
    """
    assert PRUNE_AUTO_MIN_HORIZON == 3
    policy = CandidatePrunePolicy.target_pool()
    assert policy.should_auto_apply(2) is False
    assert policy.should_auto_apply(3) is True
    players = _players()
    cand = players.drop("sum_xpts")
    plan = plan_squad(
        cand,
        _points(players, horizon=3),
        _draft_state(),
        initial=True,
        horizon=3,
        prune=True,
    )
    assert plan.picks.filter((pl.col("step") == 0) & ~pl.col("sold")).height == 15
    plan_auto = plan_squad(
        cand,
        _points(players, horizon=3),
        _draft_state(),
        initial=True,
        horizon=3,
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


def test_par_value_formula_uses_decayed_xpts_and_skips_at_or_below_r() -> None:
    """PAR = (xPts_i − xPts_r) / (price_i − price_r); null when price ≤ r."""
    players = pl.DataFrame(
        {
            "player_key": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15],
            "position": (["GK", "GK"] + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3),
            "team_key": [1, 2, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 1, 2, 3],
            # MID 8 = replacement (cheapest playable); 9–12 above *r*.
            "price": [45, 45, 45, 45, 45, 45, 45, 40, 50, 45, 60, 41, 45, 45, 45],
        }
    )
    # Flat per-step xPts so decayed sums are easy to check.
    # key 8 (r): 2/step; key 9: 5/step early-heavy; key 12: 4/step.
    rows: list[dict[str, object]] = []
    for key in players["player_key"].to_list():
        for step in range(3):
            if key == 8:
                xpts = 2.0
            elif key == 9:
                xpts = 5.0 if step == 0 else 1.0
            elif key == 12:
                xpts = 4.0
            else:
                xpts = 3.0
            rows.append(
                {
                    "player_key": int(key),
                    "step": step,
                    "xpts": xpts,
                    "p_60": 0.8,
                }
            )
    points = pl.DataFrame(rows)
    decay = 0.9
    ranked = _rank_players(players, points, decay=decay, min_p_60=0.5)
    by_key = {
        int(row["player_key"]): row
        for row in ranked.select(
            "player_key", "sum_xpts", "par_value", "price"
        ).iter_rows(named=True)
    }
    xpts_r = float(by_key[8]["sum_xpts"])
    assert xpts_r == pytest.approx(2.0 * (1.0 + decay + decay**2))
    assert by_key[8]["par_value"] is None  # priced at *r*
    # key 12 price 41: (sum_12 − sum_r) / (41 − 40)
    sum_12 = float(by_key[12]["sum_xpts"])
    assert by_key[12]["par_value"] == pytest.approx((sum_12 - xpts_r) / 1.0)
    sum_9 = float(by_key[9]["sum_xpts"])
    assert by_key[9]["par_value"] == pytest.approx((sum_9 - xpts_r) / 10.0)
    assert sum_9 == pytest.approx(5.0 + 1.0 * decay + 1.0 * (decay**2))


def test_par_replacement_is_cheapest_playable_not_bench() -> None:
    """Replacement *r* prefers p_60 ≥ min_p_60 over a cheaper non-player."""
    players = pl.DataFrame(
        {
            "player_key": list(range(1, 16)),
            "position": (["GK", "GK"] + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3),
            "team_key": [1 + (i % 5) for i in range(15)],
            # MID 8 cheapest but non-playing; 9 next-cheapest and playable = *r*.
            "price": [45, 45, 45, 45, 45, 45, 45, 40, 42, 55, 60, 70, 45, 45, 45],
        }
    )
    points = pl.DataFrame(
        [
            {
                "player_key": key,
                "step": step,
                "xpts": 1.0 if key == 8 else (2.0 if key == 9 else 5.0),
                "p_60": 0.1 if key == 8 else 0.9,
            }
            for key in players["player_key"].to_list()
            for step in range(2)
        ]
    )
    ranked = _rank_players(players, points, decay=1.0, min_p_60=0.5)
    # *r* = key 9 (price 42). key 8 is cheaper but not playable → not *r*,
    # so key 8 has null PAR only if priced ≤ *r*; 40 < 42 → still skipped.
    mid = {
        int(row["player_key"]): row["par_value"]
        for row in ranked.filter(pl.col("position") == "MID")
        .select("player_key", "par_value")
        .iter_rows(named=True)
    }
    assert mid[8] is None
    assert mid[9] is None  # *r* itself
    # key 10 at 55: (10 − 4) / (55 − 42) with flat sums over 2 steps.
    assert mid[10] == pytest.approx((10.0 - 4.0) / (55.0 - 42.0))


def test_par_value_cut_skips_players_at_or_below_replacement() -> None:
    """top_value never selects a player priced at or below replacement."""
    players = pl.DataFrame(
        {
            "player_key": list(range(1, 18)),
            "position": (["GK", "GK"] + ["DEF"] * 5 + ["MID"] * 7 + ["FWD"] * 3),
            "team_key": [1 + (i % 6) for i in range(17)],
            # MID 8 = *r* at 40; MID 9 also 40 (≤ *r*); 10–14 dearer.
            "price": [45] * 7 + [40, 40, 50, 55, 60, 65, 70] + [45, 45, 45],
        }
    )
    points = pl.DataFrame(
        [
            {
                "player_key": key,
                "step": 0,
                "xpts": 10.0 if key >= 10 else 1.0,
                "p_60": 0.9,
            }
            for key in players["player_key"].to_list()
        ]
    )
    policy = CandidatePrunePolicy(
        top_xpts=None,
        top_value={"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
        top_per_step=None,
        top_window=None,
        window_lengths=(),
        keep_cheapest=0,
        min_p_60=0.5,
    )
    pruned = policy.apply(players, points, decay=1.0)
    kept_mids = set(pruned.filter(pl.col("position") == "MID")["player_key"].to_list())
    # Five dearer MIDs fill top_value (= MID minima); price-40 pair stays out.
    assert kept_mids == {10, 11, 12, 13, 14}
    assert 8 not in kept_mids
    assert 9 not in kept_mids


def test_rank_uses_passed_decay_not_flat_sum() -> None:
    """Decayed ranking matches Σ decay^step · xPts and prefers early points."""
    # Eight MIDs so top_xpts MID=5 can drop the late-heavy one.
    players = pl.DataFrame(
        {
            "player_key": list(range(1, 19)),
            "position": (["GK", "GK"] + ["DEF"] * 5 + ["MID"] * 8 + ["FWD"] * 3),
            "team_key": [1 + (i % 5) for i in range(18)],
            "price": [45] * 18,
        }
    )
    # Player 10: all points in step 0. Player 11: same flat sum in step 4.
    early_key, late_key = 10, 11
    rows: list[dict[str, object]] = []
    for key in players["player_key"].to_list():
        for step in range(5):
            if key == early_key:
                xpts = 10.0 if step == 0 else 0.0
            elif key == late_key:
                xpts = 10.0 if step == 4 else 0.0
            else:
                # Steady ~8.2 decayed — beats late (6.56) but loses to early (10).
                xpts = 2.0
            rows.append({"player_key": int(key), "step": step, "xpts": xpts})
    points = pl.DataFrame(rows)
    decay = 0.9
    ranked = _rank_players(players, points, decay=decay)
    early = float(ranked.filter(pl.col("player_key") == early_key)["sum_xpts"][0])
    late = float(ranked.filter(pl.col("player_key") == late_key)["sum_xpts"][0])
    assert early == pytest.approx(10.0 * (decay**0))
    assert late == pytest.approx(10.0 * (decay**4))
    assert early > late
    mid_order = (
        ranked.filter(pl.col("position") == "MID")
        .sort(["sum_xpts", "player_key"], descending=[True, False])["player_key"]
        .to_list()
    )
    assert mid_order[0] == early_key
    assert mid_order.index(early_key) < mid_order.index(late_key)
    policy = CandidatePrunePolicy(
        top_xpts={"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
        top_value=None,
        top_per_step=None,
        top_window=None,
        window_lengths=(),
        keep_cheapest=0,
    )
    pruned = policy.apply(players, points, decay=decay)
    kept = set(pruned["player_key"].to_list())
    assert early_key in kept
    assert late_key not in kept


def test_rank_decay_matches_planner_weight() -> None:
    """Prune decay weights equal plan_squad's state.decay**step."""
    players = _players().drop("sum_xpts")
    points = _points(_players(), horizon=4)
    decay = 0.85
    ranked = _rank_players(players, points, decay=decay)
    by_key = {
        int(row["player_key"]): float(row["sum_xpts"])
        for row in ranked.select("player_key", "sum_xpts").iter_rows(named=True)
    }
    for key, got in by_key.items():
        expected = 0.0
        for row in points.filter(pl.col("player_key") == key).iter_rows(named=True):
            expected += float(row["xpts"]) * (decay ** int(row["step"]))
        assert got == pytest.approx(expected)


def test_playing_fillers_prefer_p_60_over_cheaper_bench() -> None:
    """keep_cheapest skips non-playing cheap players when enough play."""
    players = _players()
    cand = players.drop("sum_xpts")
    # Price-40 GKs by ascending xPts. Give the two mid ones a unique late
    # spike so exact dominance cannot drop them before keep_cheapest runs;
    # the lowest-xPts GK stays cheap + non-playing with a different spike.
    gk40 = players.filter((pl.col("position") == "GK") & (pl.col("price") == 40)).sort(
        "sum_xpts"
    )
    cheap_non_playing = int(gk40.row(0, named=True)["player_key"])
    playing_keys = [
        int(gk40.row(1, named=True)["player_key"]),
        int(gk40.row(2, named=True)["player_key"]),
    ]
    horizon = 5
    rows: list[dict[str, object]] = []
    for row in players.iter_rows(named=True):
        key = int(row["player_key"])
        base = float(row["sum_xpts"]) / horizon
        for step in range(horizon):
            xpts = base
            if key in playing_keys and step == horizon - 1:
                xpts = base + 20.0
            elif key == cheap_non_playing and step == horizon - 2:
                xpts = base + 20.0
            rows.append({"player_key": key, "step": step, "xpts": xpts})
    points = pl.DataFrame(rows).with_columns(
        pl.when(pl.col("player_key").is_in(playing_keys))
        .then(0.8)
        .otherwise(0.1)
        .alias("p_60")
    )
    policy = _tight_policy(
        top_xpts={"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
        top_value=None,
        keep_cheapest=2,
        min_p_60=0.5,
    )
    pruned = policy.apply(cand, points, decay=1.0)
    kept = set(pruned["player_key"].to_list())
    assert set(playing_keys).issubset(kept)
    assert cheap_non_playing not in kept


def test_playing_fillers_fall_back_when_too_few() -> None:
    """If fewer than keep_cheapest pass min_p_60, use plain cheapest."""
    players = _players()
    cand = players.drop("sum_xpts")
    points = _points(players).with_columns(pl.lit(0.0).alias("p_60"))
    policy = _tight_policy(
        top_xpts={"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
        top_value=None,
        keep_cheapest=2,
        min_p_60=0.5,
    )
    pruned = policy.apply(cand, points)
    kept = set(pruned["player_key"].to_list())
    cheap_gk = (
        cand.filter(pl.col("position") == "GK")
        .sort(["price", "player_key"])
        .head(2)["player_key"]
        .to_list()
    )
    assert set(cheap_gk).issubset(kept)


def test_owned_survive_every_rule() -> None:
    """Owned keys survive dominance, decayed ranking, and filler cuts."""
    players = _players()
    cand = players.drop("sum_xpts")
    owned_key = int(
        players.filter(pl.col("position") == "MID")
        .sort("sum_xpts")
        .head(1)["player_key"][0]
    )
    # Make owned strictly dominated on price and every step so only the
    # owned safeguard can keep them.
    points = _points(players).with_columns(
        pl.when(pl.col("player_key") == owned_key)
        .then(0.0)
        .otherwise(pl.col("xpts") + 1.0)
        .alias("xpts"),
        pl.when(pl.col("player_key") == owned_key)
        .then(0.0)
        .otherwise(0.9)
        .alias("p_60"),
    )
    cand = cand.with_columns(
        pl.when(pl.col("player_key") == owned_key)
        .then(pl.col("price") + 20)
        .otherwise(pl.col("price"))
        .alias("price")
    )
    policy = _tight_policy(keep_cheapest=2, min_p_60=0.5)
    pruned = policy.apply(cand, points, owned_keys=[owned_key], decay=0.9)
    assert owned_key in set(pruned["player_key"].to_list())
    # Without owned protection the same player is dropped by dominance.
    bare = drop_dominated_players(cand, points, owned_keys=())
    assert owned_key not in set(bare["player_key"].to_list())


def test_dominance_drops_clearly_dominated() -> None:
    """A dearer worse-at-every-step player is dropped before top-k."""
    players = _players()
    cand = players.drop("sum_xpts")
    # Inflate the worst MID's price and zero their xPts so many cheaper
    # better MIDs dominate them under the MID quota of 5.
    victim = int(
        players.filter(pl.col("position") == "MID")
        .sort("sum_xpts")
        .head(1)["player_key"][0]
    )
    cand = cand.with_columns(
        pl.when(pl.col("player_key") == victim)
        .then(90)
        .otherwise(pl.col("price"))
        .alias("price")
    )
    points = _points(players).with_columns(
        pl.when(pl.col("player_key") == victim)
        .then(0.0)
        .otherwise(pl.col("xpts"))
        .alias("xpts")
    )
    # Heuristic quotas large enough that only dominance can remove them.
    policy = CandidatePrunePolicy(
        top_xpts={"GK": 5, "DEF": 12, "MID": 20, "FWD": 8},
        top_value=None,
        top_per_step=None,
        top_window=None,
        window_lengths=(),
        keep_cheapest=0,
    )
    pruned = policy.apply(cand, points)
    assert victim not in set(pruned["player_key"].to_list())
    assert pruned.height < cand.height


def test_dominance_compares_per_step_not_sums() -> None:
    """Higher Σ xPts alone does not dominate when any step is worse."""
    # Two steps: front-loaded cheap player fails to dominate a flat rival.
    players = pl.DataFrame(
        {
            "player_key": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16],
            "position": (["GK", "GK"] + ["DEF"] * 5 + ["MID"] * 6 + ["FWD"] * 3),
            "team_key": [1, 2, 1, 2, 3, 4, 5, 1, 2, 3, 4, 5, 6, 1, 2, 3],
            "price": [45, 45, 45, 45, 45, 45, 45, 40, 45, 45, 45, 45, 45, 45, 45, 45],
        }
    )
    # key 8 (MID, price 40): [10, 0] — higher sum than key 9's [4, 4].
    # key 9 (MID, price 45): [4, 4] — not dominated by 8 on step 1.
    rows: list[dict[str, object]] = []
    for key in players["player_key"].to_list():
        for step in range(2):
            if key == 8:
                xpts = 10.0 if step == 0 else 0.0
            elif key == 9:
                xpts = 4.0
            else:
                xpts = 3.0
            rows.append({"player_key": int(key), "step": step, "xpts": xpts})
    points = pl.DataFrame(rows)
    kept = drop_dominated_players(players, points)
    assert 9 in set(kept["player_key"].to_list())


def test_dominance_club_cap_safeguard() -> None:
    """Dominators from one club cannot fill quota beyond MAX_PER_CLUB."""
    # Five club-1 DEFs dominate an expensive DEF, but club-feasible count
    # is only 3 < DEF quota 5, so the victim survives.
    rows: list[dict[str, object]] = []
    key = 1
    for _ in range(2):
        rows.append(
            {"player_key": key, "position": "GK", "team_key": 10 + key, "price": 45}
        )
        key += 1
    # Five cheap high-xPts DEFs, all club 1.
    club1_defs: list[int] = []
    for _ in range(5):
        club1_defs.append(key)
        rows.append({"player_key": key, "position": "DEF", "team_key": 1, "price": 40})
        key += 1
    victim = key
    rows.append({"player_key": key, "position": "DEF", "team_key": 9, "price": 70})
    key += 1
    # Two more DEFs that do *not* dominate (dearer, worse points).
    for _ in range(2):
        rows.append({"player_key": key, "position": "DEF", "team_key": 8, "price": 80})
        key += 1
    for _ in range(5):
        rows.append(
            {"player_key": key, "position": "MID", "team_key": 20 + key, "price": 50}
        )
        key += 1
    for _ in range(3):
        rows.append(
            {"player_key": key, "position": "FWD", "team_key": 30 + key, "price": 55}
        )
        key += 1
    players = pl.DataFrame(rows)
    point_rows: list[dict[str, object]] = []
    for prow in players.iter_rows(named=True):
        pk = int(prow["player_key"])
        for step in range(2):
            if pk in club1_defs:
                xpts = 8.0
            elif pk == victim:
                xpts = 1.0
            else:
                xpts = 0.5
            point_rows.append({"player_key": pk, "step": step, "xpts": xpts})
    points = pl.DataFrame(point_rows)
    assert MAX_PER_CLUB == 3
    kept = drop_dominated_players(players, points)
    assert victim in set(kept["player_key"].to_list())
    # Adding three more club-diverse dominators makes the drop legal.
    extra = pl.DataFrame(
        {
            "player_key": [key, key + 1, key + 2],
            "position": ["DEF", "DEF", "DEF"],
            "team_key": [2, 3, 4],
            "price": [40, 40, 40],
        }
    )
    wider = pl.concat([players, extra]).sort("player_key")
    extra_pts = pl.DataFrame(
        [
            {"player_key": k, "step": s, "xpts": 8.0}
            for k in (key, key + 1, key + 2)
            for s in range(2)
        ]
    )
    wider_pts = pl.concat([points, extra_pts])
    dropped = drop_dominated_players(wider, wider_pts)
    assert victim not in set(dropped["player_key"].to_list())


def _week1_actions(plan: object) -> tuple[object, ...]:
    """Squad / XI / C / V / buys / sells at step 0 (no quality metrics)."""
    picks = plan.picks  # type: ignore[attr-defined]
    assert isinstance(picks, pl.DataFrame)
    step0 = picks.filter(pl.col("step") == 0)
    squad = tuple(sorted(step0.filter(~pl.col("sold"))["player_key"].to_list()))
    xi = tuple(sorted(step0.filter(pl.col("in_xi"))["player_key"].to_list()))
    captains = step0.filter(pl.col("captain"))["player_key"].to_list()
    vices = step0.filter(pl.col("vice"))["player_key"].to_list()
    bought = tuple(sorted(step0.filter(pl.col("bought"))["player_key"].to_list()))
    sold = tuple(sorted(step0.filter(pl.col("sold"))["player_key"].to_list()))
    return (
        squad,
        xi,
        int(captains[0]) if captains else None,
        int(vices[0]) if vices else None,
        bought,
        sold,
    )


def test_dominance_lossless_week1_actions() -> None:
    """Dominance-only filter keeps optimal week-1 actions on a synthetic pool."""
    players = _players().drop("sum_xpts")
    # Inject three clearly dominated MIDs (dear, zero xPts) that a legal
    # XI never needs — full-pool and dominance-filtered solves must match.
    base_key = int(players["player_key"].max()) + 1
    extras = pl.DataFrame(
        {
            "player_key": [base_key, base_key + 1, base_key + 2],
            "position": ["MID", "MID", "MID"],
            "team_key": [91, 92, 93],
            "price": [95, 96, 97],
        }
    )
    full = pl.concat([players, extras]).sort("player_key")
    # Unique per-player step-0 scores so the optimal XI / C / V is unique;
    # dominated extras stay at 0.
    point_rows: list[dict[str, object]] = []
    for key in full["player_key"].to_list():
        for step in range(2):
            if key in (base_key, base_key + 1, base_key + 2):
                xpts = 0.0
            else:
                # Strictly decreasing in player_key so ties cannot reshuffle XI.
                xpts = 30.0 - 0.1 * int(key) if step == 0 else 1.0
            point_rows.append({"player_key": int(key), "step": step, "xpts": xpts})
    points = pl.DataFrame(point_rows)
    seed = plan_squad(
        players,
        pl.DataFrame(
            [
                {
                    "player_key": int(key),
                    "step": 0,
                    "xpts": 30.0 - 0.1 * int(key),
                }
                for key in players["player_key"].to_list()
            ]
        ),
        _draft_state(),
        initial=True,
        horizon=1,
        prune=False,
    )
    seed_keys = seed.picks.filter((pl.col("step") == 0) & ~pl.col("sold"))[
        "player_key"
    ].to_list()
    owned = [
        OwnedPlayer(
            player_key=int(key),
            purchase_price=int(players.filter(pl.col("player_key") == key)["price"][0]),
            selling_price=int(players.filter(pl.col("player_key") == key)["price"][0]),
        )
        for key in seed_keys
    ]
    bank = 1000 - sum(player.selling_price for player in owned)
    state = _draft_state(bank=bank, free_transfers=1, owned=owned)
    reference = plan_squad(full, points, state, initial=False, horizon=2, prune=False)
    filtered = drop_dominated_players(full, points, owned_keys=seed_keys)
    assert filtered.height < full.height
    for key in (base_key, base_key + 1, base_key + 2):
        assert key not in set(filtered["player_key"].to_list())
    dominated_plan = plan_squad(
        filtered, points, state, initial=False, horizon=2, prune=False
    )
    assert _week1_actions(reference) == _week1_actions(dominated_plan)


def test_prune_never_expands_pool_and_covers_minima() -> None:
    """Pruned pool is a subset and always meets position minima."""
    players = _players()
    cand = players.drop("sum_xpts")
    points = _points(players).with_columns(pl.lit(0.7).alias("p_60"))
    policy = _tight_policy()
    pruned = policy.apply(cand, points, decay=0.9)
    raw_keys = set(cand["player_key"].to_list())
    kept = set(pruned["player_key"].to_list())
    assert kept.issubset(raw_keys)
    assert len(kept) <= len(raw_keys)
    for position, minimum in MIN_POS_POOL.items():
        assert pruned.filter(pl.col("position") == position).height >= minimum


@given(decay=st.floats(min_value=0.5, max_value=1.0, allow_nan=False))
@settings(max_examples=20, deadline=None)
def test_prune_subset_property(decay: float) -> None:
    """Property: pruning never grows the pool and covers squad minima."""
    players = _players()
    cand = players.drop("sum_xpts")
    points = _points(players, horizon=3).with_columns(pl.lit(0.6).alias("p_60"))
    policy = _tight_policy(keep_cheapest=1)
    pruned = policy.apply(cand, points, decay=decay)
    kept = set(pruned["player_key"].to_list())
    assert kept.issubset(set(cand["player_key"].to_list()))
    assert pruned.height <= cand.height
    for position, minimum in MIN_POS_POOL.items():
        assert pruned.filter(pl.col("position") == position).height >= minimum


def test_plan_squad_passes_state_decay_into_prune(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """plan_squad wires SquadState.decay into prune_candidates."""
    captured: dict[str, object] = {}
    real = prune_candidates

    def _capture(*args: object, **kwargs: object) -> pl.DataFrame:
        captured["decay"] = kwargs.get("decay")
        points = args[1] if len(args) > 1 else kwargs.get("points")
        assert isinstance(points, pl.DataFrame)
        captured["has_p_60"] = "p_60" in points.columns
        return real(*args, **kwargs)  # type: ignore[arg-type]

    monkeypatch.setattr("fplt.optimise.plan.prune_candidates", _capture)
    players = _players()
    cand = players.drop("sum_xpts")
    points = _points(players, horizon=5).with_columns(pl.lit(0.7).alias("p_60"))
    decay = 0.87
    plan_squad(
        cand,
        points,
        _draft_state(decay=decay),
        initial=True,
        horizon=5,
        prune=True,
    )
    assert captured["decay"] == decay
    assert captured["has_p_60"] is True
    # Planner weight identity used by both MILP and ranking.
    assert math.isclose(decay**0, 1.0)


def test_per_step_tops_keep_one_week_punt() -> None:
    """A late one-week spike survives via top_per_step, not horizon sum."""
    # Legal minima + enough MIDs that a tight top_xpts drops the punt.
    players = pl.DataFrame(
        {
            "player_key": list(range(1, 19)),
            "position": (["GK", "GK"] + ["DEF"] * 5 + ["MID"] * 8 + ["FWD"] * 3),
            "team_key": [1 + (i % 5) for i in range(18)],
            "price": [45] * 18,
        }
    )
    # Keys 8–15 are MID. Punt = last MID: tiny horizon sum, best at step 3.
    punt_key = 15
    rows: list[dict[str, object]] = []
    for key in players["player_key"].to_list():
        for step in range(4):
            if key == punt_key:
                # Beats every rival at step 3; horizon sum 6 << steady ~20.
                xpts = 6.0 if step == 3 else 0.0
            else:
                # Steady points so horizon tops fill without the punt.
                xpts = 5.0
            rows.append({"player_key": int(key), "step": step, "xpts": xpts})
    points = pl.DataFrame(rows)
    without = CandidatePrunePolicy(
        top_xpts={"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
        top_value=None,
        top_per_step=None,
        top_window=None,
        window_lengths=(),
        keep_cheapest=0,
    )
    kept_without = set(without.apply(players, points, decay=0.9)["player_key"])
    assert punt_key not in kept_without
    with_step = CandidatePrunePolicy(
        top_xpts={"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
        top_value=None,
        top_per_step={"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
        top_window=None,
        window_lengths=(),
        keep_cheapest=0,
    )
    kept_with = set(with_step.apply(players, points, decay=0.9)["player_key"])
    assert punt_key in kept_with
    assert kept_without.issubset(kept_with)


def test_per_step_tops_use_raw_step_xpts_not_decay() -> None:
    """Per-step ranking ignores planner decay (step xPts for that step)."""
    players = pl.DataFrame(
        {
            "player_key": list(range(1, 17)),
            "position": (["GK", "GK"] + ["DEF"] * 5 + ["MID"] * 6 + ["FWD"] * 3),
            "team_key": [1 + (i % 5) for i in range(16)],
            "price": [45] * 16,
        }
    )
    # At step 4: late_key has 10 raw xPts; early_key has 9. With decay=0.5
    # the decayed step-4 scores are 0.625 vs 0.5625, but per-step tops must
    # still prefer late_key on raw 10 > 9.
    late_key, early_key = 10, 11
    rows: list[dict[str, object]] = []
    for key in players["player_key"].to_list():
        for step in range(5):
            if key == late_key:
                xpts = 10.0 if step == 4 else 0.1
            elif key == early_key:
                xpts = 9.0 if step == 4 else 0.1
            else:
                xpts = 1.0
            rows.append({"player_key": int(key), "step": step, "xpts": xpts})
    points = pl.DataFrame(rows)
    policy = CandidatePrunePolicy(
        top_xpts=None,
        top_value=None,
        top_per_step={"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
        top_window=None,
        window_lengths=(),
        keep_cheapest=0,
    )
    pruned = policy.apply(players, points, decay=0.5)
    kept = set(pruned["player_key"].to_list())
    assert late_key in kept
    # early_key is 2nd at step 4 among MIDs; quota MID=5 keeps both, so only
    # assert late survives — and that a zero flat MID without a spike does not
    # displace late when quotas are tight.
    assert early_key in kept


def test_window_tops_keep_short_stretch_specialist() -> None:
    """A player strong only on a 2-step stretch survives via top_window."""
    players = pl.DataFrame(
        {
            "player_key": list(range(1, 19)),
            "position": (["GK", "GK"] + ["DEF"] * 5 + ["MID"] * 8 + ["FWD"] * 3),
            "team_key": [1 + (i % 5) for i in range(18)],
            "price": [45] * 18,
        }
    )
    specialist = 15  # last MID: points only on steps 1–2
    mid_keys = set(players.filter(pl.col("position") == "MID")["player_key"].to_list())
    rows: list[dict[str, object]] = []
    for key in players["player_key"].to_list():
        for step in range(5):
            if key == specialist:
                # Window (1, 2) sum = 24; horizon sum = 24.
                xpts = 12.0 if step in (1, 2) else 0.0
            elif key in mid_keys:
                # Strong outside the window → horizon sum 25 > specialist.
                xpts = 0.5 if step in (1, 2) else 8.0
            else:
                xpts = 3.0
            rows.append({"player_key": int(key), "step": step, "xpts": xpts})
    points = pl.DataFrame(rows)
    no_window = CandidatePrunePolicy(
        top_xpts={"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
        top_value=None,
        top_per_step=None,
        top_window=None,
        window_lengths=(),
        keep_cheapest=0,
    )
    kept_without = set(no_window.apply(players, points, decay=1.0)["player_key"])
    assert specialist not in kept_without
    with_window = CandidatePrunePolicy(
        top_xpts={"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
        top_value=None,
        top_per_step=None,
        top_window={"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
        window_lengths=(2,),
        keep_cheapest=0,
    )
    kept_with = set(with_window.apply(players, points, decay=1.0)["player_key"])
    assert specialist in kept_with


def test_step_and_window_union_with_owned_and_minima() -> None:
    """Step / window tops union with owned and still form a legal fifteen."""
    players = _players()
    cand = players.drop("sum_xpts")
    owned_key = int(
        players.filter(pl.col("position") == "MID")
        .sort("sum_xpts")
        .head(1)["player_key"][0]
    )
    points = _points(players, horizon=4).with_columns(
        pl.when(pl.col("player_key") == owned_key)
        .then(0.0)
        .otherwise(pl.col("xpts"))
        .alias("xpts")
    )
    policy = _tight_policy(
        top_per_step={"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
        top_window={"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
        window_lengths=(2, 3),
    )
    pruned = policy.apply(cand, points, owned_keys=[owned_key], decay=0.9)
    kept = set(pruned["player_key"].to_list())
    assert owned_key in kept
    assert kept.issubset(set(cand["player_key"].to_list()))
    for position, minimum in MIN_POS_POOL.items():
        assert pruned.filter(pl.col("position") == position).height >= minimum


def test_window_lengths_reject_below_two() -> None:
    """window_lengths entries must be at least 2."""
    with pytest.raises(ValueError, match="window_lengths"):
        CandidatePrunePolicy(
            top_xpts={"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
            top_value=None,
            top_per_step=None,
            top_window={"GK": 2, "DEF": 5, "MID": 5, "FWD": 3},
            window_lengths=(1, 2),
            keep_cheapest=0,
        )


def test_validate_policy_edge_errors() -> None:
    """Pos-count / window / empty-keep validators raise clearly."""
    from fplt.optimise.candidate_prune import (
        _validate_pos_counts,
        _validate_window_lengths,
    )

    assert _validate_window_lengths(None) == ()
    with pytest.raises(ValueError, match="missing position"):
        _validate_pos_counts({"GK": 2, "DEF": 5, "MID": 5}, field_name="top_xpts")
    with pytest.raises(ValueError, match="must be >= 0"):
        _validate_pos_counts(
            {"GK": -1, "DEF": 5, "MID": 5, "FWD": 3}, field_name="top_xpts"
        )
    with pytest.raises(ValueError, match="at least one active keep rule"):
        CandidatePrunePolicy(
            top_xpts={"GK": 0, "DEF": 0, "MID": 0, "FWD": 0},
            top_value=None,
            top_per_step=None,
            top_window=None,
            window_lengths=(),
            keep_cheapest=0,
        )


def test_apply_rejects_bad_decay_missing_owned_and_empty() -> None:
    """apply() validates decay, owned keys, and returns empty pools unchanged."""
    players = _players().drop("sum_xpts")
    points = _points(_players())
    policy = _tight_policy()
    with pytest.raises(ValueError, match="decay"):
        policy.apply(players, points, decay=0.0)
    with pytest.raises(ValueError, match="owned players are not in the pool"):
        policy.apply(players, points, owned_keys=[999_999])
    empty = players.clear()
    assert policy.apply(empty, points).is_empty()


def test_apply_skips_zero_quotas_and_legacy_conflicts() -> None:
    """Zero quotas are skipped; policy+top_k conflict; default policy path."""
    players = _players().drop("sum_xpts")
    points = _points(_players())
    policy = CandidatePrunePolicy(
        top_xpts={"GK": 3, "DEF": 6, "MID": 0, "FWD": 4},
        top_value={"GK": 2, "DEF": 5, "MID": 0, "FWD": 3},
        top_per_step=None,
        top_window=None,
        window_lengths=(),
        keep_cheapest=2,
    )
    pruned = policy.apply(players, points)
    assert pruned.filter(pl.col("position") == "MID").height >= MIN_POS_POOL["MID"]

    with pytest.raises(ValueError, match="not both"):
        prune_candidates(players, points, policy=policy, top_k={"GK": 2})

    defaulted = prune_candidates(players, points)
    assert defaulted.height <= players.height
    assert defaulted.height >= 15


def test_rank_players_empty_points_and_errors() -> None:
    """_rank_players handles empty points, missing columns, and step_weights."""
    players = _players().drop("sum_xpts")
    ranked = _rank_players(players, pl.DataFrame())
    assert "sum_xpts" in ranked.columns
    assert ranked["sum_xpts"].to_list() == [0.0] * players.height

    with pytest.raises(ValueError, match="missing columns"):
        _rank_players(players, pl.DataFrame({"player_key": [1]}))
    with pytest.raises(ValueError, match="step"):
        _rank_players(
            players,
            pl.DataFrame({"player_key": [1], "xpts": [1.0]}),
            decay=0.9,
        )
    with pytest.raises(ValueError, match="positive"):
        _rank_players(
            players,
            pl.DataFrame({"player_key": [1], "step": [0], "xpts": [1.0]}),
            step_weights=[1.0, 0.0],
        )

    flat = _rank_players(
        players.head(2),
        pl.DataFrame({"player_key": [1, 1, 2], "xpts": [1.0, 2.0, 3.0]}),
    )
    by_key = {
        int(r["player_key"]): float(r["sum_xpts"]) for r in flat.iter_rows(named=True)
    }
    assert by_key[1] == pytest.approx(3.0)
    assert by_key[2] == pytest.approx(3.0)


def test_helper_early_exits_for_tops_and_cheapest() -> None:
    """Private helpers return empty sets on empty / zero-quota inputs."""
    from fplt.optimise.candidate_prune import (
        _cheapest_keys,
        _club_feasible_count,
        _points_with_position,
        _top_keys,
        _top_keys_at_each_step,
        _top_keys_at_windows,
        _weakly_dominates,
    )

    players = _players().drop("sum_xpts")
    ranked = _rank_players(players, _points(_players()))
    assert _cheapest_keys(ranked, "GK", 0, min_p_60=0.5) == set()
    assert _cheapest_keys(ranked.clear(), "GK", 2, min_p_60=0.5) == set()
    assert _top_keys(ranked, "GK", 0, "sum_xpts") == set()

    assert _points_with_position(players, pl.DataFrame()) is None
    assert (
        _points_with_position(players, pl.DataFrame({"player_key": [1], "xpts": [1.0]}))
        is None
    )
    with pytest.raises(ValueError, match="missing columns"):
        _points_with_position(players, pl.DataFrame({"player_key": [1], "step": [0]}))

    assert (
        _top_keys_at_each_step(
            players, pl.DataFrame(), quotas=dict(DEFAULT_TOP_PER_STEP)
        )
        == set()
    )
    zero_quotas = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    pts = _points(_players())
    assert _top_keys_at_each_step(players, pts, quotas=zero_quotas) == set()
    gk_only = players.filter(pl.col("position") == "GK")
    kept = _top_keys_at_each_step(
        gk_only, pts, quotas={"GK": 1, "DEF": 1, "MID": 1, "FWD": 1}
    )
    assert kept.issubset(set(gk_only["player_key"].to_list()))
    assert len(kept) == 1

    assert (
        _top_keys_at_windows(
            players,
            pl.DataFrame(),
            quotas=dict(DEFAULT_TOP_WINDOW),
            window_lengths=(2,),
        )
        == set()
    )
    assert (
        _top_keys_at_windows(
            players, pts, quotas=dict(DEFAULT_TOP_WINDOW), window_lengths=(99,)
        )
        == set()
    )
    assert (
        _top_keys_at_windows(players, pts, quotas=zero_quotas, window_lengths=(2,))
        == set()
    )

    assert _club_feasible_count([], 0) is True
    assert _club_feasible_count([1, 1, 1, 1], 4) is False

    assert (
        _weakly_dominates(price_j=40, xpts_j=(2.0, 2.0), price_i=40, xpts_i=(2.0, 2.0))
        is False
    )
    assert (
        _weakly_dominates(price_j=50, xpts_j=(2.0,), price_i=40, xpts_i=(1.0,)) is False
    )
    with pytest.raises(ValueError, match="same steps"):
        _weakly_dominates(price_j=40, xpts_j=(1.0,), price_i=50, xpts_i=(1.0, 2.0))


def test_drop_dominated_empty_and_per_step_xpts_edges() -> None:
    """drop_dominated on empty pool; _per_step_xpts missing columns / steps."""
    from fplt.optimise.candidate_prune import _per_step_xpts

    empty = _players().clear().drop("sum_xpts")
    assert drop_dominated_players(empty, pl.DataFrame()).is_empty()

    players = _players().drop("sum_xpts")
    assert all(v == () for v in _per_step_xpts(players, pl.DataFrame()).values())
    with pytest.raises(ValueError, match="missing columns"):
        _per_step_xpts(players, pl.DataFrame({"player_key": [1], "step": [0]}))
    assert all(
        v == ()
        for v in _per_step_xpts(
            players, pl.DataFrame({"player_key": [], "step": [], "xpts": []})
        ).values()
    )


def test_attach_par_value_empty_and_no_replacements() -> None:
    """_attach_par_value covers empty ranked frames."""
    from fplt.optimise.candidate_prune import _attach_par_value

    empty = _players().clear().drop("sum_xpts")
    out = _attach_par_value(
        empty.with_columns(pl.lit(0.0).alias("sum_xpts")), min_p_60=0.5
    )
    assert "par_value" in out.columns
    assert out.is_empty()
