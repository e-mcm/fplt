"""Bit-identical equivalence for PR2 injected-uniform refactor.

Freezes the pre-PR2 RNG-only bodies and asserts the refactored reference
matches them when ``uniforms`` / ``tiebreak_noise`` are omitted. Also checks
that injecting the same U(0,1) stream the RNG path would have drawn yields
identical allocate / hard-XI / CS-void outputs (L2 lever for native).
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
    _pick,
    _points,
    _poisson_sequential,
    simulate_match,
)
from fplt.scoring.rules import Rules

_RULES = Rules.for_season("2024-25")


def _legacy_enforce_hard_starters(
    minutes: np.ndarray,
    team_keys: np.ndarray,
    start_weight: np.ndarray,
    rng: np.random.Generator,
    *,
    positions: np.ndarray,
    target: int = 11,
    starter_threshold: int = 60,
    start_minutes: int = 90,
    sub_minutes: int = 15,
    gk_position: str = "GK",
) -> np.ndarray:
    """Pre-PR2 enforce_hard_starters (always draws noise from ``rng``)."""
    from fplt.models.distributions import _adjust_starter_count

    drawn = np.asarray(minutes, dtype=np.int64)
    teams = np.asarray(team_keys)
    weights = np.asarray(start_weight, dtype=np.float64)
    pos = np.asarray(positions)
    out = drawn.copy()
    noise = np.asarray(rng.random(int(drawn.shape[0])), dtype=np.float64)
    order_key = np.asarray(weights + 1e-9 * noise, dtype=np.float64)
    for team in np.unique(teams):
        indexes = np.flatnonzero(teams == team)
        if indexes.size == 0:
            continue
        is_gk = np.asarray([str(pos[i]) == gk_position for i in indexes])
        gk_idx = indexes[is_gk]
        out_idx = indexes[~is_gk]
        eligible_gk = gk_idx[weights[gk_idx] > 0]
        eligible_out = out_idx[weights[out_idx] > 0]
        want = min(target, int(eligible_gk.size + eligible_out.size))
        gk_want = 1 if eligible_gk.size > 0 and want >= 1 else 0
        out_want = min(want - gk_want, int(eligible_out.size))
        _adjust_starter_count(
            out,
            gk_idx,
            weights,
            order_key,
            want=gk_want,
            starter_threshold=starter_threshold,
            start_minutes=start_minutes,
            sub_minutes=sub_minutes,
        )
        _adjust_starter_count(
            out,
            out_idx,
            weights,
            order_key,
            want=out_want,
            starter_threshold=starter_threshold,
            start_minutes=start_minutes,
            sub_minutes=sub_minutes,
        )
    return out


def _legacy_allocate_side(
    team_goals: np.ndarray,
    minutes: np.ndarray,
    goal_weight: np.ndarray,
    assist_weight: np.ndarray,
    own_goal_weight: float,
    p_assist: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Pre-PR2 ``_allocate_side`` (RNG draws inside the goal loop)."""
    players = int(minutes.shape[0])
    n = int(minutes.shape[1])
    scored = np.zeros((players, n), dtype=np.int64)
    assisted = np.zeros_like(scored)
    missed = np.zeros(n, dtype=np.int64)
    if n == 0 or int(team_goals.max(initial=0)) == 0:
        return scored, assisted, missed
    playing = minutes > 0
    goal_rates = np.where(playing, goal_weight[:, None] * minutes / 90.0, 0.0)
    assist_rates = np.where(playing, assist_weight[:, None] * minutes / 90.0, 0.0)
    if players == 0:
        return scored, assisted, team_goals.copy()
    for _ in range(int(team_goals.max())):
        active = team_goals > scored.sum(axis=0) + missed
        if not np.any(active):
            break
        draw = rng.random(n)
        scorer = _pick(goal_rates, draw, own_goal_weight)
        claimed = active & (scorer >= 0)
        missed += active & (scorer < 0)
        column = np.arange(n)
        np.add.at(scored, (np.maximum(scorer, 0), column), claimed)
        assist_draw = rng.random(n)
        want_assist = claimed & (assist_draw < p_assist)
        blocked = assist_rates.copy()
        scored_now = scorer >= 0
        blocked[np.maximum(scorer, 0), column] = np.where(
            scored_now, 0.0, blocked[np.maximum(scorer, 0), column]
        )
        second = rng.random(n)
        assister = _pick(blocked, second, 0.0)
        given = want_assist & (assister >= 0)
        np.add.at(assisted, (np.maximum(assister, 0), column), given)
    return scored, assisted, missed


def _legacy_draw_other(means: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Pre-PR2 ``_draw_other`` (``rng.poisson`` only)."""
    rate = np.abs(means)
    drawn = rng.poisson(rate).astype(np.float64)
    return np.where(means >= 0.0, drawn, -drawn)


def _legacy_points(
    rules: Rules,
    positions: np.ndarray,
    sides: np.ndarray,
    minutes: np.ndarray,
    goals: np.ndarray,
    assists: np.ndarray,
    home: np.ndarray,
    away: np.ndarray,
    other_points: np.ndarray,
    rng: np.random.Generator,
    *,
    cs_red_void_prob: float,
    conceded_mode: str,
    sample_other: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Pre-PR2 ``_points`` (RNG CS void + poisson other)."""
    bucket_edges = (0, 60)
    goal_value = np.array([rules.goals[pos] for pos in positions], dtype=np.float64)
    sheet_value = np.array(
        [rules.clean_sheets[pos] for pos in positions], dtype=np.float64
    )
    conceded_value = np.array(
        [
            rules.goals_conceded.points if pos in rules.goals_conceded.positions else 0
            for pos in positions
        ],
        dtype=np.float64,
    )
    opponent = np.where(sides[:, None] == "home", away[None, :], home[None, :])
    every = rules.goals_conceded.every
    long = rules.appearance.long_minutes
    appearance = np.where(
        minutes >= long,
        rules.appearance.long,
        np.where(minutes > 0, rules.appearance.short, 0),
    )
    clean_sheet = (opponent == 0) & (minutes >= long)
    if cs_red_void_prob > 0.0:
        voided = rng.random(minutes.shape) < cs_red_void_prob
        clean_sheet = clean_sheet & ~voided
    if conceded_mode == "played":
        exposure = np.where(minutes > 0, 1.0, 0.0)
    else:
        exposure = np.where(minutes > 0, minutes / 90.0, 0.0)
    conceded = conceded_value[:, None] * (opponent // every) * exposure
    bucket = np.where(minutes <= 0, 0, np.where(minutes < bucket_edges[1], 1, 2))
    other_mean = np.take_along_axis(other_points, bucket, axis=1)
    other = _legacy_draw_other(other_mean, rng) if sample_other else other_mean
    points = (
        appearance
        + goal_value[:, None] * goals
        + rules.assists * assists
        + sheet_value[:, None] * clean_sheet
        + conceded
        + other
    )
    return points, clean_sheet.astype(np.bool_)


def _legacy_simulate_match(
    scorelines: np.ndarray,
    players: pl.DataFrame,
    minutes: np.ndarray,
    other_points: np.ndarray,
    rules: Rules,
    rng: np.random.Generator,
    *,
    own_goal_weight: float = 0.05,
    p_assist: float = 0.75,
    cs_red_void_prob: float = 0.0,
    conceded_mode: str = "scaled",
    sample_other: bool = False,
) -> MatchSimulation:
    """Pre-PR2 ``simulate_match`` wiring (no uniforms kwarg)."""
    from fplt.models.simulation import _minutes, _other, _players, _scorelines

    home, away = _scorelines(scorelines)
    n = home.shape[0]
    keys, sides, positions, goal_weight, assist_weight = _players(players)
    played = _minutes(minutes, keys.shape[0], n)
    other = _other(other_points, keys.shape[0])
    goals = np.zeros((keys.shape[0], n), dtype=np.int64)
    assists = np.zeros_like(goals)
    unattributed = np.zeros((2, n), dtype=np.int64)
    for side_index, side_goals in enumerate((home, away)):
        mask = sides == ("home" if side_index == 0 else "away")
        side_goals_out, side_assists, missed = _legacy_allocate_side(
            side_goals,
            played[mask],
            goal_weight[mask],
            assist_weight[mask],
            own_goal_weight,
            p_assist,
            rng,
        )
        goals[mask] = side_goals_out
        assists[mask] = side_assists
        unattributed[side_index] = missed
    points, clean_sheets = _legacy_points(
        rules,
        positions,
        sides,
        played,
        goals,
        assists,
        home,
        away,
        other,
        rng,
        cs_red_void_prob=cs_red_void_prob,
        conceded_mode=conceded_mode,
        sample_other=sample_other,
    )
    return MatchSimulation(
        player_key=keys,
        points=points,
        goals=goals,
        assists=assists,
        clean_sheets=clean_sheets,
        unattributed_home=unattributed[0],
        unattributed_away=unattributed[1],
    )


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


@pytest.mark.parametrize("seed", [0, 1, 7, 42, 99])
def test_enforce_hard_starters_rng_matches_legacy(seed: int) -> None:
    rng_data = np.random.default_rng(seed)
    n_home, n_away, n_sims = 18, 18, 128
    n_players = n_home + n_away
    minutes = rng_data.choice(MINUTE_BINS, size=(n_players, n_sims)).astype(np.int64)
    team = np.array([0] * n_home + [1] * n_away, dtype=np.int64)
    weight = rng_data.random(n_players)
    weight[0] = max(weight[0], 0.1)
    weight[n_home] = max(weight[n_home], 0.1)
    positions = np.array(
        ["GK"] + ["MID"] * (n_home - 1) + ["GK"] + ["DEF"] * (n_away - 1),
        dtype=object,
    )
    new = enforce_hard_starters(
        minutes,
        team,
        weight,
        np.random.default_rng(seed + 1000),
        positions=positions,
    )
    old = _legacy_enforce_hard_starters(
        minutes,
        team,
        weight,
        np.random.default_rng(seed + 1000),
        positions=positions,
    )
    np.testing.assert_array_equal(new, old)


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


@pytest.mark.parametrize("seed", [0, 2, 8, 17])
def test_allocate_side_rng_matches_legacy(seed: int) -> None:
    rng_data = np.random.default_rng(seed)
    players, n_sims = 11, 64
    team_goals = rng_data.integers(0, 5, size=n_sims)
    minutes = rng_data.choice([0.0, 45.0, 90.0], size=(players, n_sims))
    goal_weight = rng_data.random(players)
    assist_weight = rng_data.random(players)
    new = _allocate_side(
        team_goals,
        minutes,
        goal_weight,
        assist_weight,
        0.05,
        0.75,
        np.random.default_rng(seed + 9),
    )
    old = _legacy_allocate_side(
        team_goals,
        minutes,
        goal_weight,
        assist_weight,
        0.05,
        0.75,
        np.random.default_rng(seed + 9),
    )
    for left, right in zip(new, old, strict=True):
        np.testing.assert_array_equal(left, right)


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


@pytest.mark.parametrize(
    ("seed", "sample_other", "cs_void"),
    [
        (0, False, 0.0),
        (1, False, 0.15),
        (2, True, 0.0),
        (3, True, 0.2),
        (7, False, 0.05),
    ],
)
def test_simulate_match_rng_ledger_matches_legacy(
    seed: int, sample_other: bool, cs_void: float
) -> None:
    rng_data = np.random.default_rng(seed)
    n_home, n_away, n_sims = 8, 8, 40
    weights = [(float(x), float(y)) for x, y in rng_data.random((n_home + n_away, 2))]
    players = _squad(
        weights,
        sides=["home"] * n_home + ["away"] * n_away,
        positions=(
            ["GK"]
            + ["DEF"] * 3
            + ["MID"] * 3
            + ["FWD"]
            + ["GK"]
            + ["DEF"] * 3
            + ["MID"] * 3
            + ["FWD"]
        ),
    )
    scorelines = rng_data.integers(0, 4, size=(n_sims, 2))
    minutes = rng_data.choice([0.0, 15.0, 60.0, 90.0], size=(n_home + n_away, n_sims))
    other = rng_data.normal(0.0, 0.4, size=(n_home + n_away, 3))
    new = simulate_match(
        scorelines,
        players,
        minutes,
        other,
        _RULES,
        np.random.default_rng(seed + 100),
        own_goal_weight=0.05,
        p_assist=0.75,
        cs_red_void_prob=cs_void,
        sample_other=sample_other,
    )
    old = _legacy_simulate_match(
        scorelines,
        players,
        minutes,
        other,
        _RULES,
        np.random.default_rng(seed + 100),
        own_goal_weight=0.05,
        p_assist=0.75,
        cs_red_void_prob=cs_void,
        sample_other=sample_other,
    )
    assert _ledger_digest(new) == _ledger_digest(old)
    np.testing.assert_array_equal(new.points, old.points)
    np.testing.assert_array_equal(new.goals, old.goals)
    np.testing.assert_array_equal(new.assists, old.assists)
    np.testing.assert_array_equal(new.clean_sheets, old.clean_sheets)


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


def test_draw_other_rng_matches_legacy() -> None:
    means = np.array([[0.0, 0.5, -0.3], [1.2, -2.0, 0.0]], dtype=np.float64)
    new = _draw_other(means, np.random.default_rng(11))
    old = _legacy_draw_other(means, np.random.default_rng(11))
    np.testing.assert_array_equal(new, old)


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
