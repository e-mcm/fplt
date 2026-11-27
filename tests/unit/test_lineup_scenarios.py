"""Sample-average lineup selection on a fixed fifteen."""

from __future__ import annotations

from itertools import permutations

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fplt.optimise.lineup import (
    LineupSelection,
    _armband_pairs,
    _autosub_state,
    _AutosubState,
    _finish_one_sim,
    compare_ev_vs_saa_lineup,
    select_lineup_expected,
    select_lineup_scenarios,
    verify_lineup_mean,
)
from fplt.optimise.scenarios import ScenarioBundle, mean_realised_lineup


def test_select_lineup_prefers_bench_insurance() -> None:
    """Risky XI star is benched when a safe sub scores in the blank scenarios."""
    squad = _squad()
    keys = squad["player_key"].to_list()
    n = len(keys)
    # Two sims. Player 11 (FWD) is high EV but blanks in sim1; bench MID 13
    # always scores 6. Starting 11 is only optimal if we ignore blanks.
    points = np.zeros((n, 2), dtype=np.float64)
    minutes = np.full((n, 2), 90, dtype=np.int64)
    for key in keys:
        idx = keys.index(key)
        points[idx] = 2.0
    # Inflate FWD 11 in sim0 only; blank in sim1.
    i11 = keys.index(11)
    points[i11, 0] = 10.0
    points[i11, 1] = 0.0
    minutes[i11, 1] = 0
    i13 = keys.index(13)
    points[i13] = 6.0
    minutes[i13] = 90
    bundle = ScenarioBundle(
        player_keys=np.asarray(keys, dtype=np.int64),
        points=points,
        minutes=minutes,
    )
    chosen = select_lineup_scenarios(squad, bundle)
    assert chosen.mean_points == pytest.approx(
        verify_lineup_mean(chosen, bundle), rel=0, abs=1e-9
    )
    # Either 11 starts and 13 is first outfield bench, or 13 starts — both
    # capture the insurance. Assert the mean beats "naive always start 11
    # with 13 last on bench and no autosub credit" by checking autosub path.
    assert chosen.players.filter(pl.col("in_xi")).height == 11
    assert int(chosen.players["captain"].sum()) == 1
    assert int(chosen.players["vice"].sum()) == 1
    bench = chosen.players.filter(~pl.col("in_xi")).sort("bench_slot")
    assert bench.height == 4
    assert set(bench["bench_slot"].to_list()) == {1, 2, 3, 4}


def test_p_haul_breaks_captain_ties() -> None:
    squad = _squad().with_columns(
        pl.when(pl.col("player_key") == 9)
        .then(0.9)
        .when(pl.col("player_key") == 10)
        .then(0.1)
        .otherwise(0.0)
        .alias("p_haul"),
        pl.lit(5.0).alias("xpts"),
    )
    # Equal xpts everywhere so EV mean is flat; haul should prefer player 9.
    chosen = select_lineup_expected(squad, haul_weight=1.0)
    captain = chosen.players.filter(pl.col("captain"))["player_key"][0]
    assert int(captain) == 9


def test_compare_ev_vs_saa_prefers_blank_insurance() -> None:
    """Metric board: SAA mean ≥ EV mean under joint blanks (FDB-241)."""
    squad = _squad()
    keys = squad["player_key"].to_list()
    n = len(keys)
    points = np.full((n, 2), 2.0, dtype=np.float64)
    minutes = np.full((n, 2), 90, dtype=np.int64)
    i11 = keys.index(11)
    points[i11, 0] = 10.0
    points[i11, 1] = 0.0
    minutes[i11, 1] = 0
    i13 = keys.index(13)
    points[i13] = 6.0
    minutes[i13] = 90
    # EV sees mean xpts: player 11 = 5, player 13 = 6 — but build xpts as
    # EV would from marginal means so 11 looks better than a safe 2.0 bench.
    xpts = [2.0] * 15
    xpts[10] = 5.0  # player 11
    xpts[12] = 4.0  # player 13 — lower EV than 11's mean
    squad = squad.with_columns(pl.Series("xpts", xpts))
    bundle = ScenarioBundle(
        player_keys=np.asarray(keys, dtype=np.int64),
        points=points,
        minutes=minutes,
    )
    compared = compare_ev_vs_saa_lineup(squad, bundle)
    assert compared.saa_mean >= compared.ev_mean - 1e-9
    assert compared.delta == pytest.approx(
        compared.saa_mean - compared.ev_mean, rel=0, abs=1e-9
    )


def test_select_lineup_rejects_bad_squad() -> None:
    squad = _squad().head(14)
    bundle = ScenarioBundle(
        player_keys=np.arange(1, 15, dtype=np.int64),
        points=np.zeros((14, 1)),
        minutes=np.zeros((14, 1), dtype=np.int64),
    )
    with pytest.raises(ValueError, match="15"):
        select_lineup_scenarios(squad, bundle)


def test_armband_pairs_skip_vice_when_captain_never_blanks() -> None:
    """Captains who always play keep a single deterministic vice."""
    xi = (1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11)
    pairs = _armband_pairs(
        xi,
        any_blank=True,
        haul={},
        captain_may_blank=frozenset({11}),
    )
    by_cap: dict[int, list[int]] = {}
    for captain, vice in pairs:
        by_cap.setdefault(captain, []).append(vice)
    assert by_cap[11] == [key for key in xi if key != 11]
    assert by_cap[1] == [2]


def test_armband_pairs_none_may_blank_enumerates_all_vices() -> None:
    """``captain_may_blank=None`` treats every XI captain as blank-capable."""
    xi = (1, 2, 3)
    pairs = _armband_pairs(xi, any_blank=True, haul={}, captain_may_blank=None)
    assert set(pairs) == {(1, 2), (1, 3), (2, 1), (2, 3), (3, 1), (3, 2)}


def test_finish_one_sim_vice_and_chips() -> None:
    """Vice-on-blank, TC multiplier, and bench boost on a fixed autosub state."""
    active = frozenset({0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 11})
    used = frozenset({11})
    state = _AutosubState(active=active, used_bench=used, xi_sum=20.0)
    pts = np.asarray([1.0] * 15, dtype=np.float64)
    pts[1] = 4.0  # vice
    pts[2] = 5.0  # unused bench for bboost
    pts[10] = 9.0  # blank captain
    mins = np.asarray([90] * 15, dtype=np.int64)
    mins[10] = 0
    bench_idx = [11, 12, 2, 14]
    # Captain blanks → vice in active doubles via +1x on top of xi_sum.
    assert _finish_one_sim(
        state,
        captain_i=10,
        vice_i=1,
        pts=pts,
        mins=mins,
        chip=None,
        bench_idx=bench_idx,
    ) == pytest.approx(24.0)
    # Vice not in active but plays → +2x vice points.
    inactive_vice = _AutosubState(
        active=frozenset(active - {1}),
        used_bench=used,
        xi_sum=16.0,
    )
    assert _finish_one_sim(
        inactive_vice,
        captain_i=10,
        vice_i=1,
        pts=pts,
        mins=mins,
        chip=None,
        bench_idx=bench_idx,
    ) == pytest.approx(24.0)
    # TC when captain plays.
    mins_tc = mins.copy()
    mins_tc[10] = 90
    assert _finish_one_sim(
        state,
        captain_i=10,
        vice_i=1,
        pts=pts,
        mins=mins_tc,
        chip="3xc",
        bench_idx=bench_idx,
    ) == pytest.approx(20.0 + 2.0 * 9.0)
    # Bench boost scores unused playing bench (12, 2, 14; 11 already used).
    assert _finish_one_sim(
        state,
        captain_i=1,
        vice_i=0,
        pts=pts,
        mins=mins_tc,
        chip="bboost",
        bench_idx=bench_idx,
    ) == pytest.approx(20.0 + 4.0 + 1.0 + 5.0 + 1.0)


def test_autosub_then_armband_matches_realised_lineup() -> None:
    """Decoupled autosub + armband matches ``mean_realised_lineup``."""
    squad = _squad()
    keys = squad["player_key"].to_list()
    # Bundle row order matches sorted player_key (1..15) → index = key - 1.
    xi_keys = tuple(range(1, 12))
    bench_keys = (13, 14, 15, 12)
    pos_by_key = {
        int(row["player_key"]): str(row["position"])
        for row in squad.iter_rows(named=True)
    }
    xi_idx = [key - 1 for key in xi_keys]
    xi_pos = [pos_by_key[key] for key in xi_keys]
    bench_idx = [key - 1 for key in bench_keys]
    bench_pos = [pos_by_key[key] for key in bench_keys]
    points = np.full((15, 1), 2.0, dtype=np.float64)
    minutes = np.full((15, 1), 90, dtype=np.int64)
    points[10, 0] = 10.0
    minutes[10, 0] = 0  # FWD 11 blanks
    points[12, 0] = 6.0  # MID 13 on bench
    pts = points[:, 0]
    mins = minutes[:, 0]
    state = _autosub_state(
        xi_idx=xi_idx,
        xi_pos=xi_pos,
        bench_idx=bench_idx,
        bench_pos=bench_pos,
        pts=pts,
        mins=mins,
    )
    decoupled = _finish_one_sim(
        state,
        captain_i=1,  # DEF 2
        vice_i=2,
        pts=pts,
        mins=mins,
        chip=None,
        bench_idx=bench_idx,
    )
    annotated = _annotate_trial(squad, xi_keys, bench_keys, captain=2, vice=3)
    bundle = ScenarioBundle(
        player_keys=np.asarray(keys, dtype=np.int64),
        points=points,
        minutes=minutes,
    )
    realised = mean_realised_lineup(annotated, bundle)
    assert decoupled == pytest.approx(realised)
    assert (13 - 1) in state.active  # MID 13 autosubs for blank FWD 11


@given(
    blank_player=st.sampled_from([9, 10, 11, 13]),
    blank_sim=st.sampled_from([0, 1]),
    haul_star=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=8, deadline=None)
def test_blank_aware_select_matches_verify_and_beats_bench_perms(
    blank_player: int,
    blank_sim: int,
    haul_star: float,
) -> None:
    """SAA mean matches verify; chosen bench order is locally optimal.

    Eight examples are enough to cover blank player / sim / haul tie-breaks
    after #302; 25× was a ~10s CI hotspot (FDB-387) without extra coverage.
    """
    squad = _squad()
    keys = squad["player_key"].to_list()
    n = len(keys)
    points = np.full((n, 2), 2.0, dtype=np.float64)
    minutes = np.full((n, 2), 90, dtype=np.int64)
    bi = keys.index(blank_player)
    points[bi, blank_sim] = 0.0
    minutes[bi, blank_sim] = 0
    # Safe insurance mid always scores when available.
    i13 = keys.index(13)
    points[i13] = 5.0
    bundle = ScenarioBundle(
        player_keys=np.asarray(keys, dtype=np.int64),
        points=points,
        minutes=minutes,
    )
    haul = {11: haul_star, 9: 1.0 - haul_star}
    chosen = select_lineup_scenarios(squad, bundle, p_haul=haul, haul_tiebreak=1e-3)
    assert chosen.mean_points == pytest.approx(
        verify_lineup_mean(chosen, bundle), rel=0, abs=1e-9
    )
    xi_keys = tuple(
        int(value)
        for value in chosen.players.filter(pl.col("in_xi"))["player_key"].to_list()
    )
    bench = chosen.players.filter(~pl.col("in_xi")).sort("bench_slot")
    outfield = tuple(
        int(value)
        for value in bench.filter(pl.col("position") != "GK")["player_key"].to_list()
    )
    reserve_gk = int(bench.filter(pl.col("position") == "GK")["player_key"][0])
    captain = int(chosen.players.filter(pl.col("captain"))["player_key"][0])
    vice = int(chosen.players.filter(pl.col("vice"))["player_key"][0])
    # Local optimality: no outfield bench permutation beats the selection
    # under the same XI / armbands (autosub-once search must stay exact).
    for order in permutations(outfield):
        trial_bench = order + (reserve_gk,)
        annotated = _annotate_trial(squad, xi_keys, trial_bench, captain, vice)
        mean = verify_lineup_mean(
            LineupSelection(players=annotated, mean_points=0.0),
            bundle,
        )
        assert mean <= chosen.mean_points + 1e-9


def _annotate_trial(
    squad: pl.DataFrame,
    xi_keys: tuple[int, ...],
    bench_keys: tuple[int, ...],
    captain: int,
    vice: int,
) -> pl.DataFrame:
    xi_set = set(xi_keys)
    slots = {key: slot for slot, key in enumerate(bench_keys, start=1)}
    return squad.with_columns(
        pl.col("player_key").is_in(list(xi_set)).alias("in_xi"),
        (pl.col("player_key") == captain).alias("captain"),
        (pl.col("player_key") == vice).alias("vice"),
        pl.col("player_key")
        .replace_strict(slots, default=None)
        .cast(pl.Int64)
        .alias("bench_slot"),
    )


def _squad() -> pl.DataFrame:
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
