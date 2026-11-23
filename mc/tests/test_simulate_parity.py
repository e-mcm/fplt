"""L1/L2/L3/L5 tests for native K4 match-delta + K5 simulate/aggregate."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

pytest.importorskip("fplt_mc")

from fplt.models.simulation import (
    MatchSimUniforms,
    SideAllocateUniforms,
    _points,
    simulate_match,
)
from fplt.scoring.rules import Rules
from fplt_mc import _testing as mc_testing

_RULES = Rules.for_season("2024-25")
_POS = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
_GOLDEN = (
    Path(__file__).resolve().parents[2] / "tests" / "fixtures" / "mc_golden" / "pr5_k45.json"
)


def _rules_kwargs(rules: Rules = _RULES) -> dict:
    """Flatten ``Rules`` into native keyword args."""
    order = ("GK", "DEF", "MID", "FWD")
    conceded = np.array(
        [
            float(rules.goals_conceded.points)
            if pos in rules.goals_conceded.positions
            else 0.0
            for pos in order
        ],
        dtype=np.float64,
    )
    return {
        "appearance_long_minutes": int(rules.appearance.long_minutes),
        "appearance_short": float(rules.appearance.short),
        "appearance_long": float(rules.appearance.long),
        "goals_by_pos": np.array([float(rules.goals[p]) for p in order], dtype=np.float64),
        "assists_pts": float(rules.assists),
        "cs_by_pos": np.array(
            [float(rules.clean_sheets[p]) for p in order], dtype=np.float64
        ),
        "conceded_every": int(rules.goals_conceded.every),
        "conceded_by_pos": conceded,
    }


def _encode_positions(positions: np.ndarray) -> np.ndarray:
    return np.array([_POS[str(p)] for p in positions], dtype=np.uint8)


def _encode_sides(sides: np.ndarray) -> np.ndarray:
    return np.array([0 if s == "home" else 1 for s in sides], dtype=np.uint8)


@dataclass
class FixedMatchOther:
    """Table-backed match offsets for L2 (no fit required)."""

    delta_goals: np.ndarray
    delta_cs: np.ndarray
    delta_scored: np.ndarray

    def match_delta(
        self,
        *,
        positions: np.ndarray,
        minutes: np.ndarray,
        team_goals: np.ndarray,
        clean_sheet: np.ndarray,
        player_goals: np.ndarray,
    ) -> np.ndarray:
        played = np.asarray(minutes, dtype=np.float64)
        players, sims = played.shape
        band = np.where(team_goals <= 0, 0, np.where(team_goals == 1, 1, 2))
        cs_idx = np.asarray(clean_sheet, dtype=np.int64)
        scored_idx = (np.asarray(player_goals) >= 1).astype(np.int64)
        delta = np.zeros((players, sims), dtype=np.float64)
        for i, pos in enumerate(positions):
            index = _POS[str(pos)]
            delta[i] = (
                self.delta_goals[index, band[i]]
                + self.delta_cs[index, cs_idx[i]]
                + self.delta_scored[index, scored_idx[i]]
            )
            delta[i] = np.where(played[i] > 0, delta[i], 0.0)
        return delta


def _sample_tables(seed: int) -> FixedMatchOther:
    rng = np.random.default_rng(seed)
    return FixedMatchOther(
        delta_goals=rng.normal(0, 0.2, size=(4, 3)),
        delta_cs=rng.normal(0, 0.1, size=(4, 2)),
        delta_scored=rng.normal(0, 0.15, size=(4, 2)),
    )


# ---------------------------------------------------------------------------
# L2 — match-delta inside points
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 5, 17])
def test_l2_points_match_other_matches_python(seed: int) -> None:
    """Native K4 tables match Python ``match_other.match_delta`` path."""
    rng = np.random.default_rng(seed)
    players, n = 6, 24
    positions = np.array(["GK", "DEF", "MID", "FWD", "MID", "DEF"], dtype=object)
    sides = np.array(["home", "home", "home", "away", "away", "away"], dtype=object)
    minutes = rng.choice([0.0, 45.0, 60.0, 90.0], size=(players, n))
    goals = rng.integers(0, 3, size=(players, n), dtype=np.int64)
    assists = rng.integers(0, 2, size=(players, n), dtype=np.int64)
    home = rng.integers(0, 4, size=n, dtype=np.int64)
    away = rng.integers(0, 4, size=n, dtype=np.int64)
    other = rng.random((players, 3)) * 0.4
    tables = _sample_tables(seed + 99)

    py_pts, py_cs = _points(
        _RULES,
        positions,
        sides,
        minutes,
        goals,
        assists,
        home,
        away,
        other,
        np.random.default_rng(0),
        cs_red_void_prob=0.0,
        conceded_mode="scaled",
        sample_other=False,
        match_other=tables,
    )
    native_pts, native_cs = mc_testing.points_with_uniforms(
        _encode_positions(positions),
        _encode_sides(sides),
        np.ascontiguousarray(minutes, dtype=np.float64),
        np.ascontiguousarray(goals, dtype=np.int64),
        np.ascontiguousarray(assists, dtype=np.int64),
        np.ascontiguousarray(home, dtype=np.int64),
        np.ascontiguousarray(away, dtype=np.int64),
        np.ascontiguousarray(other, dtype=np.float64),
        **_rules_kwargs(),
        cs_red_void_prob=0.0,
        conceded_scaled=True,
        sample_other=False,
        delta_goals=np.ascontiguousarray(tables.delta_goals, dtype=np.float64),
        delta_cs=np.ascontiguousarray(tables.delta_cs, dtype=np.float64),
        delta_scored=np.ascontiguousarray(tables.delta_scored, dtype=np.float64),
    )
    np.testing.assert_allclose(py_pts, native_pts, rtol=0, atol=1e-12)
    np.testing.assert_array_equal(py_cs, native_cs.astype(bool))


# ---------------------------------------------------------------------------
# L2 — aggregate + full simulate fuse
# ---------------------------------------------------------------------------


def test_l2_aggregate_matches_numpy() -> None:
    rng = np.random.default_rng(3)
    players, n = 5, 40
    minutes = rng.choice([0.0, 30.0, 60.0, 90.0], size=(players, n))
    points = rng.normal(4.0, 3.0, size=(players, n))
    goals = rng.integers(0, 3, size=(players, n), dtype=np.int64)
    assists = rng.integers(0, 2, size=(players, n), dtype=np.int64)
    cs = (rng.random((players, n)) < 0.3).astype(np.uint8)

    out = mc_testing.aggregate_fixture(
        np.ascontiguousarray(minutes, dtype=np.float64),
        np.ascontiguousarray(points, dtype=np.float64),
        np.ascontiguousarray(goals, dtype=np.int64),
        np.ascontiguousarray(assists, dtype=np.int64),
        np.ascontiguousarray(cs, dtype=np.uint8),
    )
    xpts, xpts_sd, p_played, p_60, p_goal, p_assist, p_cs, p_haul = out
    np.testing.assert_allclose(xpts, points.mean(axis=1), rtol=0, atol=1e-12)
    np.testing.assert_allclose(xpts_sd, points.std(axis=1), rtol=0, atol=1e-12)
    np.testing.assert_allclose(p_played, (minutes > 0).mean(axis=1), atol=1e-12)
    np.testing.assert_allclose(p_60, (minutes >= 60).mean(axis=1), atol=1e-12)
    np.testing.assert_allclose(p_goal, (goals >= 1).mean(axis=1), atol=1e-12)
    np.testing.assert_allclose(p_assist, (assists >= 1).mean(axis=1), atol=1e-12)
    np.testing.assert_allclose(p_cs, cs.mean(axis=1), atol=1e-12)
    np.testing.assert_allclose(p_haul, (points >= 10).mean(axis=1), atol=1e-12)


@pytest.mark.parametrize("seed", [2, 9, 23])
def test_l2_simulate_match_matches_python_inject(seed: int) -> None:
    """Full K5 fuse matches ``simulate_match`` with the same uniforms."""
    rng = np.random.default_rng(seed)
    players, n = 8, 16
    positions = np.array(
        ["GK", "DEF", "DEF", "MID", "MID", "FWD", "MID", "DEF"], dtype=object
    )
    sides = np.array(["home"] * 4 + ["away"] * 4, dtype=object)
    keys = np.arange(100, 100 + players)
    minutes = rng.choice([0.0, 45.0, 60.0, 90.0], size=(players, n)).astype(np.float64)
    goal_weight = rng.random(players) + 0.1
    assist_weight = rng.random(players) + 0.1
    other = rng.random((players, 3)) * 0.3
    home = rng.integers(0, 3, size=n, dtype=np.int64)
    away = rng.integers(0, 3, size=n, dtype=np.int64)
    scorelines = np.column_stack([home, away])
    tables = _sample_tables(seed + 7)

    home_max = int(home.max())
    away_max = int(away.max())
    home_u = SideAllocateUniforms(
        scorer=rng.random((max(home_max, 1), n)),
        assist_flag=rng.random((max(home_max, 1), n)),
        assister=rng.random((max(home_max, 1), n)),
    )
    away_u = SideAllocateUniforms(
        scorer=rng.random((max(away_max, 1), n)),
        assist_flag=rng.random((max(away_max, 1), n)),
        assister=rng.random((max(away_max, 1), n)),
    )
    uniforms = MatchSimUniforms(home=home_u, away=away_u)

    players_df = pl.DataFrame(
        {
            "player_key": keys.tolist(),
            "side": sides.tolist(),
            "position": positions.tolist(),
            "goal_weight": goal_weight.tolist(),
            "assist_weight": assist_weight.tolist(),
        }
    )
    py = simulate_match(
        scorelines,
        players_df,
        minutes,
        other,
        _RULES,
        np.random.default_rng(0),
        own_goal_weight=0.05,
        p_assist=0.75,
        cs_red_void_prob=0.0,
        conceded_mode="scaled",
        sample_other=False,
        match_other=tables,
        uniforms=uniforms,
    )

    native = mc_testing.simulate_match_with_uniforms(
        np.ascontiguousarray(home, dtype=np.int64),
        np.ascontiguousarray(away, dtype=np.int64),
        _encode_sides(sides),
        _encode_positions(positions),
        np.ascontiguousarray(minutes, dtype=np.float64),
        np.ascontiguousarray(goal_weight, dtype=np.float64),
        np.ascontiguousarray(assist_weight, dtype=np.float64),
        np.ascontiguousarray(other, dtype=np.float64),
        **_rules_kwargs(),
        own_goal_weight=0.05,
        p_assist=0.75,
        cs_red_void_prob=0.0,
        conceded_scaled=True,
        sample_other=False,
        home_scorer_u=np.ascontiguousarray(home_u.scorer, dtype=np.float64),
        home_assist_flag_u=np.ascontiguousarray(home_u.assist_flag, dtype=np.float64),
        home_assister_u=np.ascontiguousarray(home_u.assister, dtype=np.float64),
        away_scorer_u=np.ascontiguousarray(away_u.scorer, dtype=np.float64),
        away_assist_flag_u=np.ascontiguousarray(away_u.assist_flag, dtype=np.float64),
        away_assister_u=np.ascontiguousarray(away_u.assister, dtype=np.float64),
        delta_goals=np.ascontiguousarray(tables.delta_goals, dtype=np.float64),
        delta_cs=np.ascontiguousarray(tables.delta_cs, dtype=np.float64),
        delta_scored=np.ascontiguousarray(tables.delta_scored, dtype=np.float64),
        return_samples=True,
    )

    np.testing.assert_array_equal(py.goals, native["goals"])
    np.testing.assert_array_equal(py.assists, native["assists"])
    np.testing.assert_allclose(py.points, native["points"], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(py.clean_sheets, native["clean_sheets"].astype(bool))
    np.testing.assert_array_equal(py.unattributed_home, native["unattributed_home"])
    np.testing.assert_array_equal(py.unattributed_away, native["unattributed_away"])
    np.testing.assert_allclose(native["xpts"], py.points.mean(axis=1), atol=1e-12)
    np.testing.assert_allclose(native["xpts_sd"], py.points.std(axis=1), atol=1e-12)


# ---------------------------------------------------------------------------
# L1 behavioural
# ---------------------------------------------------------------------------


def test_l1_zero_minutes_zero_match_delta() -> None:
    tables = FixedMatchOther(
        delta_goals=np.ones((4, 3)),
        delta_cs=np.ones((4, 2)),
        delta_scored=np.ones((4, 2)),
    )
    pts, _ = mc_testing.points_with_uniforms(
        np.array([1], dtype=np.uint8),
        np.array([0], dtype=np.uint8),
        np.zeros((1, 1), dtype=np.float64),
        np.zeros((1, 1), dtype=np.int64),
        np.zeros((1, 1), dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.array([0], dtype=np.int64),
        np.zeros((1, 3), dtype=np.float64),
        **_rules_kwargs(),
        cs_red_void_prob=0.0,
        conceded_scaled=True,
        sample_other=False,
        delta_goals=tables.delta_goals,
        delta_cs=tables.delta_cs,
        delta_scored=tables.delta_scored,
    )
    assert pts[0, 0] == 0.0


# ---------------------------------------------------------------------------
# L3 properties
# ---------------------------------------------------------------------------


@given(seed=st.integers(0, 2000), n_sims=st.integers(1, 12))
@settings(max_examples=25, deadline=None)
def test_l3_simulate_conserves_goals(seed: int, n_sims: int) -> None:
    rng = np.random.default_rng(seed)
    players = 6
    positions = np.array(["GK", "DEF", "MID", "FWD", "MID", "DEF"], dtype=object)
    sides = np.array(["home", "home", "home", "away", "away", "away"], dtype=object)
    minutes = rng.choice([0.0, 60.0, 90.0], size=(players, n_sims)).astype(np.float64)
    gw = np.ones(players)
    aw = np.ones(players)
    other = np.zeros((players, 3))
    home = rng.integers(0, 3, size=n_sims, dtype=np.int64)
    away = rng.integers(0, 3, size=n_sims, dtype=np.int64)
    home_max = max(int(home.max()), 1)
    away_max = max(int(away.max()), 1)
    hs = np.empty((home_max, n_sims))
    hf = np.empty((home_max, n_sims))
    ha = np.empty((home_max, n_sims))
    a_s = np.empty((away_max, n_sims))
    af = np.empty((away_max, n_sims))
    aa = np.empty((away_max, n_sims))
    for g in range(home_max):
        for j in range(n_sims):
            base = g * n_sims + j
            hs[g, j] = mc_testing.philox_u01_batch(seed, 1, base, 1)[0]
            hf[g, j] = mc_testing.philox_u01_batch(seed, 2, base, 1)[0]
            ha[g, j] = mc_testing.philox_u01_batch(seed, 3, base, 1)[0]
    for g in range(away_max):
        for j in range(n_sims):
            base = g * n_sims + j
            a_s[g, j] = mc_testing.philox_u01_batch(seed, 4, base, 1)[0]
            af[g, j] = mc_testing.philox_u01_batch(seed, 5, base, 1)[0]
            aa[g, j] = mc_testing.philox_u01_batch(seed, 6, base, 1)[0]

    out = mc_testing.simulate_match_with_uniforms(
        home,
        away,
        _encode_sides(sides),
        _encode_positions(positions),
        minutes,
        gw,
        aw,
        other,
        **_rules_kwargs(),
        own_goal_weight=0.05,
        p_assist=0.8,
        cs_red_void_prob=0.0,
        conceded_scaled=True,
        sample_other=False,
        home_scorer_u=hs,
        home_assist_flag_u=hf,
        home_assister_u=ha,
        away_scorer_u=a_s,
        away_assist_flag_u=af,
        away_assister_u=aa,
        return_samples=True,
    )
    side_codes = _encode_sides(sides)
    for j in range(n_sims):
        home_total = int(out["unattributed_home"][j])
        away_total = int(out["unattributed_away"][j])
        for i in range(players):
            if side_codes[i] == 0:
                home_total += int(out["goals"][i, j])
            else:
                away_total += int(out["goals"][i, j])
        assert home_total == int(home[j])
        assert away_total == int(away[j])
    assert np.isfinite(out["xpts"]).all()
    assert ((out["p_played"] >= 0) & (out["p_played"] <= 1)).all()


# ---------------------------------------------------------------------------
# L5 cross-platform golden
# ---------------------------------------------------------------------------


def _sha256_f64(arr: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(arr, dtype=np.float64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def _sha256_i64(arr: np.ndarray) -> str:
    contiguous = np.ascontiguousarray(arr, dtype=np.int64)
    return hashlib.sha256(contiguous.tobytes()).hexdigest()


def test_l5_golden_hashes() -> None:
    """Fixed inputs → sha256 of summary + sample planes (cross-platform)."""
    seed = 20261123
    players, n = 6, 32
    positions = np.array(["GK", "DEF", "MID", "FWD", "MID", "DEF"], dtype=object)
    sides = np.array(["home", "home", "home", "away", "away", "away"], dtype=object)
    minutes = np.array(
        [[90.0, 60.0, 0.0, 90.0] * (n // 4) for _ in range(players)], dtype=np.float64
    )
    gw = np.array([1.0, 1.2, 0.8, 1.1, 0.9, 1.0], dtype=np.float64)
    aw = np.array([1.0, 0.9, 1.1, 0.8, 1.0, 1.2], dtype=np.float64)
    other = np.full((players, 3), 0.25, dtype=np.float64)
    home = np.array([i % 3 for i in range(n)], dtype=np.int64)
    away = np.array([(i + 1) % 3 for i in range(n)], dtype=np.int64)
    tables = FixedMatchOther(
        delta_goals=np.linspace(-0.2, 0.2, 12).reshape(4, 3),
        delta_cs=np.linspace(-0.1, 0.1, 8).reshape(4, 2),
        delta_scored=np.linspace(-0.05, 0.15, 8).reshape(4, 2),
    )
    home_max = max(int(home.max()), 1)
    away_max = max(int(away.max()), 1)
    hs = np.empty((home_max, n))
    hf = np.empty((home_max, n))
    ha = np.empty((home_max, n))
    a_s = np.empty((away_max, n))
    af = np.empty((away_max, n))
    aa = np.empty((away_max, n))
    for g in range(home_max):
        for j in range(n):
            base = g * n + j
            hs[g, j] = mc_testing.philox_u01_batch(seed, 1, base, 1)[0]
            hf[g, j] = mc_testing.philox_u01_batch(seed, 2, base, 1)[0]
            ha[g, j] = mc_testing.philox_u01_batch(seed, 3, base, 1)[0]
    for g in range(away_max):
        for j in range(n):
            base = g * n + j
            a_s[g, j] = mc_testing.philox_u01_batch(seed, 4, base, 1)[0]
            af[g, j] = mc_testing.philox_u01_batch(seed, 5, base, 1)[0]
            aa[g, j] = mc_testing.philox_u01_batch(seed, 6, base, 1)[0]
    cs_u = np.empty((players, n))
    other_u = np.empty((players, n))
    for i in range(players):
        for j in range(n):
            cs_u[i, j] = mc_testing.philox_u01_batch(seed, 10, i * n + j, 1)[0]
            other_u[i, j] = mc_testing.philox_u01_batch(seed, 11, i * n + j, 1)[0]

    out = mc_testing.simulate_match_with_uniforms(
        home,
        away,
        _encode_sides(sides),
        _encode_positions(positions),
        minutes,
        gw,
        aw,
        other,
        **_rules_kwargs(),
        own_goal_weight=0.05,
        p_assist=0.75,
        cs_red_void_prob=0.15,
        conceded_scaled=True,
        sample_other=True,
        home_scorer_u=hs,
        home_assist_flag_u=hf,
        home_assister_u=ha,
        away_scorer_u=a_s,
        away_assist_flag_u=af,
        away_assister_u=aa,
        cs_void_u=cs_u,
        other_u=other_u,
        delta_goals=tables.delta_goals,
        delta_cs=tables.delta_cs,
        delta_scored=tables.delta_scored,
        return_samples=True,
    )

    hashes = {
        "xpts": _sha256_f64(out["xpts"]),
        "xpts_sd": _sha256_f64(out["xpts_sd"]),
        "p_played": _sha256_f64(out["p_played"]),
        "p_60": _sha256_f64(out["p_60"]),
        "p_goal": _sha256_f64(out["p_goal"]),
        "p_assist": _sha256_f64(out["p_assist"]),
        "p_cs": _sha256_f64(out["p_cs"]),
        "p_haul": _sha256_f64(out["p_haul"]),
        "points": _sha256_f64(out["points"]),
        "goals": _sha256_i64(out["goals"]),
        "assists": _sha256_i64(out["assists"]),
        "unattributed_home": _sha256_i64(out["unattributed_home"]),
        "unattributed_away": _sha256_i64(out["unattributed_away"]),
    }
    assert _GOLDEN.is_file(), f"missing golden fixture {_GOLDEN}"
    expected = json.loads(_GOLDEN.read_text(encoding="utf-8"))
    assert expected["version"] == "0.4.0"
    assert expected["seed"] == seed
    assert expected["hashes"] == hashes
