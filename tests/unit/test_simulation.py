"""Simulated goals, clean sheets, and allocations follow the drawn scoreline."""

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fplt.models.distributions import MAX_GOALS, ScorelineDist
from fplt.models.simulation import simulate_match
from fplt.scoring.rules import Rules

_RULES = Rules.for_season("2024-25")


def test_allocated_goals_sum_to_the_scoreline() -> None:
    result = _play(
        scorelines=np.array([[2, 1]]),
        minutes=np.array([[90.0], [0.0], [90.0]]),
        weights=[(1.0, 0.0), (1.0, 1.0), (1.0, 1.0)],
        own_goal_weight=0.0,
        p_assist=1.0,
        seed=1,
    )
    assert result.goals[:, 0].tolist() == [2, 0, 1]
    assert int(result.assists[2, 0]) == 0
    assert result.goals[:2].sum() + result.unattributed_home[0] == 2
    assert result.goals[2].sum() + result.unattributed_away[0] == 1


def test_a_benched_player_is_not_given_a_goal() -> None:
    result = _play(
        scorelines=np.array([[3, 0]]),
        minutes=np.array([[0.0]]),
        weights=[(1.0, 1.0)],
        own_goal_weight=0.0,
        p_assist=1.0,
        seed=2,
    )
    assert result.goals[0, 0] == 0
    assert result.assists[0, 0] == 0
    assert result.unattributed_home[0] == 3


def test_an_assist_goes_to_someone_else() -> None:
    result = _play(
        scorelines=np.array([[1, 0]]),
        minutes=np.array([[90.0], [90.0]]),
        weights=[(1.0, 1.0), (1.0, 1.0)],
        own_goal_weight=0.0,
        p_assist=1.0,
        seed=3,
    )
    assert result.goals[:, 0].sum() == 1
    assert result.assists[:, 0].sum() == 1
    scorer = int(result.goals[:, 0].argmax())
    assert result.assists[scorer, 0] == 0


def test_the_only_player_on_the_pitch_has_no_assister() -> None:
    result = _play(
        scorelines=np.array([[1, 0]]),
        minutes=np.array([[90.0]]),
        weights=[(1.0, 1.0)],
        own_goal_weight=0.0,
        p_assist=1.0,
        seed=4,
    )
    assert result.assists[0, 0] == 0


def test_clean_sheet_needs_sixty_minutes_and_a_scoreless_opponent() -> None:
    result = _play(
        scorelines=np.array([[0, 0], [2, 0], [0, 1]]),
        minutes=np.array([[90.0, 45.0, 90.0]]),
        weights=[(0.0, 0.0)],
        sides=["home"],
        positions=["DEF"],
        own_goal_weight=1.0,
        p_assist=0.0,
        seed=5,
    )
    assert result.points[0].tolist() == pytest.approx([6.0, 1.0, 2.0])


def test_goals_conceded_scale_with_minutes() -> None:
    full, half = _play(
        scorelines=np.array([[2, 0], [2, 0]]),
        minutes=np.array([[90.0, 45.0]]),
        weights=[(0.0, 0.0)],
        sides=["away"],
        positions=["GK"],
        own_goal_weight=1.0,
        p_assist=0.0,
        seed=6,
    ).points[0]
    assert full == pytest.approx(1.0)
    assert half == pytest.approx(0.5)


def test_goals_conceded_played_mode_ignores_minutes_scale() -> None:
    full, half = _play(
        scorelines=np.array([[2, 0], [2, 0]]),
        minutes=np.array([[90.0, 45.0]]),
        weights=[(0.0, 0.0)],
        sides=["away"],
        positions=["GK"],
        own_goal_weight=1.0,
        p_assist=0.0,
        seed=6,
        conceded_mode="played",
    ).points[0]
    assert full == pytest.approx(1.0)
    assert half == pytest.approx(0.0)


def test_a_red_void_probability_removes_clean_sheets() -> None:
    result = _play(
        scorelines=np.array([[0, 0]] * 2000),
        minutes=np.full((1, 2000), 90.0),
        weights=[(0.0, 0.0)],
        sides=["home"],
        positions=["DEF"],
        own_goal_weight=1.0,
        p_assist=0.0,
        seed=12,
        cs_red_void_prob=1.0,
    )
    # Appearance 2 only; the 4-point clean sheet is always voided.
    assert result.points[0].tolist() == pytest.approx([2.0] * 2000)


def test_sampled_other_points_keep_the_mean_and_add_noise() -> None:
    mean = _play(
        scorelines=np.array([[0, 0]] * 4000),
        minutes=np.full((1, 4000), 90.0),
        weights=[(0.0, 0.0)],
        other=np.array([[0.0, 0.0, 2.0]]),
        own_goal_weight=0.0,
        p_assist=0.0,
        seed=13,
        sample_other=False,
    ).points[0]
    sampled = _play(
        scorelines=np.array([[0, 0]] * 4000),
        minutes=np.full((1, 4000), 90.0),
        weights=[(0.0, 0.0)],
        other=np.array([[0.0, 0.0, 2.0]]),
        own_goal_weight=0.0,
        p_assist=0.0,
        seed=13,
        sample_other=True,
    ).points[0]
    assert mean.mean() == pytest.approx(4.0)
    assert sampled.mean() == pytest.approx(4.0, abs=0.1)
    assert sampled.std() > mean.std()
    assert len(np.unique(sampled)) > 1


def test_other_points_follow_the_minutes_bucket() -> None:
    result = _play(
        scorelines=np.array([[0, 0], [0, 0], [0, 0]]),
        minutes=np.array([[0.0, 30.0, 90.0]]),
        weights=[(0.0, 0.0)],
        other=np.array([[1.0, 2.0, 4.0]]),
        own_goal_weight=0.0,
        p_assist=0.0,
        seed=7,
    )
    assert result.points[0].tolist() == pytest.approx([1.0, 3.0, 6.0])


def test_mean_goals_and_clean_sheets_match_the_distribution() -> None:
    pmf = np.zeros((1, MAX_GOALS + 2, MAX_GOALS + 2))
    pmf[0, 0, 0] = 0.5
    pmf[0, 2, 0] = 0.5
    dist = ScorelineDist(np.array([7]), pmf)
    rng = np.random.default_rng(8)
    drawn = dist.sample(rng, 4000)[0]
    result = _play(
        scorelines=drawn,
        minutes=np.full((2, 4000), 90.0),
        weights=[(0.0, 0.0), (0.0, 0.0)],
        sides=["home", "away"],
        positions=["DEF", "GK"],
        own_goal_weight=1.0,
        p_assist=0.0,
        seed=9,
    )
    home_goals = result.goals[0] + result.unattributed_home
    assert home_goals.mean() == pytest.approx(1.0, abs=0.05)
    assert (result.points[0] == 6).mean() == pytest.approx(
        dist.p_clean_sheet("home")[0], abs=0.02
    )
    assert (result.points[1] == 6).mean() == pytest.approx(
        dist.p_clean_sheet("away")[0], abs=0.05
    )


def test_the_same_seed_repeats_the_allocation() -> None:
    kwargs = {
        "scorelines": np.array([[2, 1]]),
        "minutes": np.array([[90.0], [90.0], [90.0]]),
        "weights": [(1.0, 1.0), (1.0, 1.0), (3.0, 1.0)],
        "own_goal_weight": 0.2,
        "p_assist": 0.75,
        "seed": 10,
    }
    first = _play(**kwargs)
    second = _play(**kwargs)
    assert np.array_equal(first.goals, second.goals)
    assert np.array_equal(first.assists, second.assists)


def test_a_side_with_nobody_leaves_the_goals_unattributed() -> None:
    result = _play(
        scorelines=np.array([[0, 2]]),
        minutes=np.array([[90.0]]),
        weights=[(1.0, 0.0)],
        own_goal_weight=0.0,
        p_assist=0.0,
        seed=11,
    )
    assert result.unattributed_away[0] == 2
    assert result.goals.sum() == 0


def test_bad_inputs_raise() -> None:
    players = _squad([(1.0, 0.0)])
    minutes = np.array([[90.0]])
    other = np.zeros((1, 3))
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="scorelines"):
        simulate_match(np.array([1, 0]), players, minutes, other, _RULES, rng)
    with pytest.raises(ValueError, match="side"):
        simulate_match(
            np.array([[1, 0]]),
            players.with_columns(pl.lit("bench").alias("side")),
            minutes,
            other,
            _RULES,
            rng,
        )
    with pytest.raises(ValueError, match="weights"):
        simulate_match(
            np.array([[1, 0]]),
            players.with_columns(pl.lit(-1.0).alias("goal_weight")),
            minutes,
            other,
            _RULES,
            rng,
        )
    with pytest.raises(ValueError, match="minutes"):
        simulate_match(
            np.array([[1, 0]]), players, np.array([[-1.0]]), other, _RULES, rng
        )
    with pytest.raises(ValueError, match="p_assist"):
        simulate_match(
            np.array([[1, 0]]),
            players,
            minutes,
            other,
            _RULES,
            rng,
            p_assist=1.5,
        )
    with pytest.raises(ValueError, match="own_goal_weight"):
        simulate_match(
            np.array([[1, 0]]),
            players,
            minutes,
            other,
            _RULES,
            rng,
            own_goal_weight=-0.1,
        )
    with pytest.raises(ValueError, match="cs_red_void_prob"):
        simulate_match(
            np.array([[1, 0]]),
            players,
            minutes,
            other,
            _RULES,
            rng,
            cs_red_void_prob=1.5,
        )
    with pytest.raises(ValueError, match="conceded_mode"):
        simulate_match(
            np.array([[1, 0]]),
            players,
            minutes,
            other,
            _RULES,
            rng,
            conceded_mode="half",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="player_key"):
        simulate_match(
            np.array([[1, 0]]),
            players.drop("player_key"),
            minutes,
            other,
            _RULES,
            rng,
        )


@settings(max_examples=40)
@given(
    home=st.integers(0, 3),
    away=st.integers(0, 3),
    left=st.sampled_from([0, 45, 90]),
    right=st.sampled_from([0, 90]),
)
def test_every_draw_conserves_goals(
    home: int, away: int, left: int, right: int
) -> None:
    result = _play(
        scorelines=np.array([[home, away]]),
        minutes=np.array([[float(left)], [float(right)], [90.0]]),
        weights=[(0.2, 0.2), (0.8, 0.0), (1.0, 1.0)],
        own_goal_weight=0.1,
        p_assist=0.5,
        seed=home * 10 + away + left + right,
    )
    home_players = result.goals[:2].sum()
    assert home_players + result.unattributed_home[0] == home
    assert result.goals[2].sum() + result.unattributed_away[0] == away
    assert result.goals[0, 0] == 0 or left > 0
    assert result.assists[0, 0] == 0 or left > 0


def _play(
    *,
    scorelines: np.ndarray,
    minutes: np.ndarray,
    weights: list[tuple[float, float]],
    own_goal_weight: float,
    p_assist: float,
    seed: int,
    sides: list[str] | None = None,
    positions: list[str] | None = None,
    other: np.ndarray | None = None,
    cs_red_void_prob: float = 0.0,
    conceded_mode: str = "scaled",
    sample_other: bool = False,
):
    players = _squad(weights, sides=sides, positions=positions)
    if other is None:
        other = np.zeros((len(weights), 3))
    return simulate_match(
        scorelines,
        players,
        minutes,
        other,
        _RULES,
        np.random.default_rng(seed),
        own_goal_weight=own_goal_weight,
        p_assist=p_assist,
        cs_red_void_prob=cs_red_void_prob,
        conceded_mode=conceded_mode,  # type: ignore[arg-type]
        sample_other=sample_other,
    )


def _squad(
    weights: list[tuple[float, float]],
    sides: list[str] | None = None,
    positions: list[str] | None = None,
) -> pl.DataFrame:
    count = len(weights)
    chosen_sides = sides or (["home", "home", "away"][:count])
    chosen_positions = positions or (["FWD", "MID", "GK"][:count])
    return pl.DataFrame(
        {
            "player_key": list(range(1, count + 1)),
            "side": chosen_sides,
            "position": chosen_positions,
            "goal_weight": [goal for goal, _ in weights],
            "assist_weight": [assist for _, assist in weights],
        }
    )
