"""Lean inject/RNG parity for the vectorised Python MC reference (PR2).

Keeps the L2 inject surface exercised against the production RNG path.
Frozen pre-PR0/PR2 slow-loop bodies are gone — behavioural coverage lives in
``test_simulation`` / ``test_hard_starters`` / ``test_distributions``.
"""

from __future__ import annotations

import hashlib

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fplt.models.distributions import MINUTE_BINS, enforce_hard_starters
from fplt.models.simulation import (
    MatchSimulation,
    MatchSimUniforms,
    SideAllocateUniforms,
    _allocate_side,
    _draw_other,
    _points,
    _poisson_sequential,
    simulate_match,
)
from fplt.scoring.rules import Rules

_RULES = Rules.for_season("2024-25")


def _ledger_digest(result: MatchSimulation) -> str:
    """Stable sha256 over the match-simulation ledger arrays."""
    hasher = hashlib.sha256()
    for array in (
        result.player_key,
        result.points,
        result.goals,
        result.assists,
        result.clean_sheets.astype(np.uint8),
        result.unattributed_home,
        result.unattributed_away,
    ):
        hasher.update(np.ascontiguousarray(array).tobytes())
    return hasher.hexdigest()


def _squad(
    weights: list[tuple[float, float]],
    sides: list[str] | None = None,
    positions: list[str] | None = None,
) -> pl.DataFrame:
    count = len(weights)
    chosen_sides = sides or (["home"] * (count // 2) + ["away"] * (count - count // 2))
    chosen_positions = (
        positions or (["GK", "DEF", "MID", "FWD"] * ((count + 3) // 4))[:count]
    )
    return pl.DataFrame(
        {
            "player_key": list(range(1, count + 1)),
            "side": chosen_sides,
            "position": chosen_positions,
            "goal_weight": [goal for goal, _ in weights],
            "assist_weight": [assist for _, assist in weights],
        }
    )


def _side_uniforms_from_seed(
    seed: int, max_goals: int, n_sims: int
) -> SideAllocateUniforms:
    """Build allocate uniforms in the same call order as the RNG path."""
    rng = np.random.default_rng(seed)
    scorer = np.empty((max_goals, n_sims), dtype=np.float64)
    assist_flag = np.empty((max_goals, n_sims), dtype=np.float64)
    assister = np.empty((max_goals, n_sims), dtype=np.float64)
    for goal_index in range(max_goals):
        scorer[goal_index] = rng.random(n_sims)
        assist_flag[goal_index] = rng.random(n_sims)
        assister[goal_index] = rng.random(n_sims)
    return SideAllocateUniforms(
        scorer=scorer, assist_flag=assist_flag, assister=assister
    )


@pytest.mark.parametrize("seed", [0, 3, 11])
def test_enforce_hard_starters_injected_noise_matches_rng(seed: int) -> None:
    rng_data = np.random.default_rng(seed)
    n_players, n_sims = 24, 64
    minutes = rng_data.choice(MINUTE_BINS, size=(n_players, n_sims)).astype(np.int64)
    team = np.array([0] * 12 + [1] * 12, dtype=np.int64)
    weight = rng_data.random(n_players)
    positions = np.array(
        ["GK"] + ["MID"] * 11 + ["GK"] + ["DEF"] * 11,
        dtype=object,
    )
    noise_seed = seed + 50
    noise = np.random.default_rng(noise_seed).random(n_players)
    via_inject = enforce_hard_starters(
        minutes,
        team,
        weight,
        np.random.default_rng(0),
        positions=positions,
        tiebreak_noise=noise,
    )
    via_rng = enforce_hard_starters(
        minutes,
        team,
        weight,
        np.random.default_rng(noise_seed),
        positions=positions,
    )
    np.testing.assert_array_equal(via_inject, via_rng)


@pytest.mark.parametrize("seed", [1, 5, 13])
def test_allocate_side_injected_uniforms_match_rng(seed: int) -> None:
    rng_data = np.random.default_rng(seed)
    players, n_sims = 9, 32
    team_goals = rng_data.integers(0, 4, size=n_sims)
    max_goals = int(team_goals.max())
    minutes = rng_data.choice([0.0, 60.0, 90.0], size=(players, n_sims))
    goal_weight = rng_data.random(players) + 0.01
    assist_weight = rng_data.random(players) + 0.01
    uniforms = _side_uniforms_from_seed(seed + 21, max_goals, n_sims)
    via_inject = _allocate_side(
        team_goals,
        minutes,
        goal_weight,
        assist_weight,
        0.05,
        0.8,
        np.random.default_rng(0),
        uniforms=uniforms,
    )
    via_rng = _allocate_side(
        team_goals,
        minutes,
        goal_weight,
        assist_weight,
        0.05,
        0.8,
        np.random.default_rng(seed + 21),
    )
    for left, right in zip(via_inject, via_rng, strict=True):
        np.testing.assert_array_equal(left, right)


@pytest.mark.parametrize("seed", [4, 9, 15])
def test_simulate_match_injected_allocate_and_cs_void(seed: int) -> None:
    """Injected allocate + CS-void uniforms reproduce the RNG ledger."""
    rng_data = np.random.default_rng(seed)
    n_home, n_away, n_sims = 6, 6, 24
    weights = [(float(x), float(y)) for x, y in rng_data.random((n_home + n_away, 2))]
    sides = ["home"] * n_home + ["away"] * n_away
    players = _squad(
        weights,
        sides=sides,
        positions=["GK", "DEF", "DEF", "MID", "MID", "FWD"] * 2,
    )
    scorelines = rng_data.integers(0, 3, size=(n_sims, 2))
    minutes = rng_data.choice([0.0, 45.0, 90.0], size=(n_home + n_away, n_sims))
    other = np.zeros((n_home + n_away, 3))
    home_goals = scorelines[:, 0]
    away_goals = scorelines[:, 1]
    stream_seed = seed + 30
    stream = np.random.default_rng(stream_seed)
    home_rows = int(home_goals.max())
    away_rows = int(away_goals.max())
    home_scorer = np.empty((max(home_rows, 1), n_sims), dtype=np.float64)
    home_flag = np.empty_like(home_scorer)
    home_ast = np.empty_like(home_scorer)
    for goal_index in range(home_rows):
        home_scorer[goal_index] = stream.random(n_sims)
        home_flag[goal_index] = stream.random(n_sims)
        home_ast[goal_index] = stream.random(n_sims)
    away_scorer = np.empty((max(away_rows, 1), n_sims), dtype=np.float64)
    away_flag = np.empty_like(away_scorer)
    away_ast = np.empty_like(away_scorer)
    for goal_index in range(away_rows):
        away_scorer[goal_index] = stream.random(n_sims)
        away_flag[goal_index] = stream.random(n_sims)
        away_ast[goal_index] = stream.random(n_sims)
    cs_void = stream.random((n_home + n_away, n_sims))
    uniforms = MatchSimUniforms(
        home=SideAllocateUniforms(home_scorer, home_flag, home_ast),
        away=SideAllocateUniforms(away_scorer, away_flag, away_ast),
        cs_void=cs_void,
    )
    via_inject = simulate_match(
        scorelines,
        players,
        minutes,
        other,
        _RULES,
        np.random.default_rng(0),
        own_goal_weight=0.05,
        p_assist=0.7,
        cs_red_void_prob=0.12,
        sample_other=False,
        uniforms=uniforms,
    )
    via_rng = simulate_match(
        scorelines,
        players,
        minutes,
        other,
        _RULES,
        np.random.default_rng(stream_seed),
        own_goal_weight=0.05,
        p_assist=0.7,
        cs_red_void_prob=0.12,
        sample_other=False,
    )
    assert _ledger_digest(via_inject) == _ledger_digest(via_rng)


def test_poisson_sequential_zero_rate_is_zero() -> None:
    rate = np.zeros((3, 4), dtype=np.float64)
    uniforms = np.random.default_rng(0).random(rate.shape)
    np.testing.assert_array_equal(_poisson_sequential(rate, uniforms), 0)


def test_poisson_sequential_is_deterministic() -> None:
    rate = np.array([[0.2, 1.5], [0.0, 0.8]], dtype=np.float64)
    uniforms = np.array([[0.1, 0.9], [0.5, 0.3]], dtype=np.float64)
    first = _poisson_sequential(rate, uniforms)
    second = _poisson_sequential(rate, uniforms)
    np.testing.assert_array_equal(first, second)
    drawn = _draw_other(rate, np.random.default_rng(0), uniforms=uniforms)
    np.testing.assert_array_equal(drawn, first.astype(np.float64))


def test_points_injected_cs_void_matches_rng() -> None:
    players, n_sims = 4, 20
    positions = np.array(["GK", "DEF", "MID", "FWD"], dtype=object)
    sides = np.array(["home", "home", "away", "away"], dtype=object)
    minutes = np.full((players, n_sims), 90.0)
    goals = np.zeros((players, n_sims), dtype=np.int64)
    assists = np.zeros_like(goals)
    home = np.zeros(n_sims, dtype=np.int64)
    away = np.zeros(n_sims, dtype=np.int64)
    other = np.zeros((players, 3))
    void_seed = 3
    void_u = np.random.default_rng(void_seed).random((players, n_sims))
    via_inject = _points(
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
        cs_red_void_prob=0.25,
        conceded_mode="scaled",
        sample_other=False,
        cs_void_uniforms=void_u,
    )
    via_rng = _points(
        _RULES,
        positions,
        sides,
        minutes,
        goals,
        assists,
        home,
        away,
        other,
        np.random.default_rng(void_seed),
        cs_red_void_prob=0.25,
        conceded_mode="scaled",
        sample_other=False,
    )
    np.testing.assert_array_equal(via_inject[0], via_rng[0])
    np.testing.assert_array_equal(via_inject[1], via_rng[1])


def test_bad_tiebreak_noise_shape_raises() -> None:
    minutes = np.full((4, 2), 90, dtype=np.int64)
    with pytest.raises(ValueError, match="tiebreak_noise"):
        enforce_hard_starters(
            minutes,
            np.zeros(4, dtype=np.int64),
            np.ones(4),
            np.random.default_rng(0),
            positions=np.array(["GK", "MID", "MID", "FWD"], dtype=object),
            tiebreak_noise=np.ones(3),
        )


def test_bad_allocate_uniform_shape_raises() -> None:
    with pytest.raises(ValueError, match="scorer uniforms"):
        _allocate_side(
            np.array([2, 1]),
            np.ones((3, 2)),
            np.ones(3),
            np.ones(3),
            0.05,
            0.5,
            np.random.default_rng(0),
            uniforms=SideAllocateUniforms(
                scorer=np.zeros((1, 2)),
                assist_flag=np.zeros((2, 2)),
                assister=np.zeros((2, 2)),
            ),
        )


@settings(max_examples=25)
@given(
    home=st.integers(0, 3),
    away=st.integers(0, 3),
    seed=st.integers(0, 10_000),
)
def test_injected_match_conserves_goals(home: int, away: int, seed: int) -> None:
    players = _squad(
        [(1.0, 1.0), (1.0, 0.5), (0.8, 0.8)],
        sides=["home", "home", "away"],
        positions=["MID", "FWD", "MID"],
    )
    scorelines = np.array([[home, away]])
    minutes = np.array([[90.0], [90.0], [90.0]])
    other = np.zeros((3, 3))
    home_u = _side_uniforms_from_seed(seed, max(home, 1), 1)
    away_u = _side_uniforms_from_seed(seed + 1, max(away, 1), 1)
    result = simulate_match(
        scorelines,
        players,
        minutes,
        other,
        _RULES,
        np.random.default_rng(0),
        own_goal_weight=0.1,
        p_assist=0.5,
        uniforms=MatchSimUniforms(home=home_u, away=away_u),
    )
    assert result.goals[:2].sum() + result.unattributed_home[0] == home
    assert result.goals[2:].sum() + result.unattributed_away[0] == away
