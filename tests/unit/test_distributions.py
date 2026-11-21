"""Scoreline grids stay probability distributions after truncation."""

import numpy as np
import polars as pl
import pytest
from hypothesis import given
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from fplt.models.distributions import (
    MAX_GOALS,
    MINUTE_BINS,
    MinutesDist,
    ScorelineDist,
)

_GOALS = MAX_GOALS + 1


def _ids(count: int) -> np.ndarray:
    return np.arange(count, dtype=np.int64)


def _dist(pmf: np.ndarray) -> ScorelineDist:
    return ScorelineDist(_ids(pmf.shape[0]), pmf)


def test_clean_sheet_and_marginal_match_a_known_grid() -> None:
    pmf = np.zeros((2, _GOALS, _GOALS))
    pmf[0, 1, 0] = 1.0
    pmf[1, 0, 2] = 1.0
    dist = _dist(pmf)
    np.testing.assert_allclose(dist.p_clean_sheet("home"), [1.0, 0.0])
    np.testing.assert_allclose(dist.p_clean_sheet("away"), [0.0, 1.0])
    assert dist.marginal("home")[0, 1] == pytest.approx(1.0)
    assert dist.marginal("away")[1, 2] == pytest.approx(1.0)


def test_tiny_tail_is_renormalised_onto_the_kept_grid() -> None:
    wide = np.zeros((1, _GOALS + 4, _GOALS + 4))
    wide[0, 0, 0] = 1.0 - 1e-8
    wide[0, -1, -1] = 1e-8
    dist = _dist(wide)
    assert dist.pmf.shape == (1, _GOALS, _GOALS)
    np.testing.assert_allclose(dist.pmf.sum(), 1.0)
    assert dist.pmf[0, 0, 0] == pytest.approx(1.0)


def test_the_rating_bound_keeps_its_tail() -> None:
    from scipy.stats import poisson

    # 3 * 3 * 1.3, the largest rate the Dixon-Coles bounds can emit.
    goals = np.arange(MAX_GOALS + 8)
    home = poisson.pmf(goals, 11.7)
    away = poisson.pmf(goals, 11.7)
    joint = home[:, None] * away[None, :]
    dist = ScorelineDist(np.array([1]), joint[None, :, :])
    assert dist.pmf.shape == (1, MAX_GOALS + 1, MAX_GOALS + 1)
    np.testing.assert_allclose(dist.pmf.sum(), 1.0)


def test_a_heavy_tail_is_refused() -> None:
    wide = np.zeros((1, _GOALS + 2, _GOALS + 2))
    wide[0, 0, 0] = 1.0 - 1e-4
    wide[0, -1, -1] = 1e-4
    with pytest.raises(ValueError, match="truncation"):
        _dist(wide)


def test_invalid_grids_are_refused() -> None:
    with pytest.raises(ValueError, match="shape"):
        _dist(np.ones((1, 11, 12)))
    with pytest.raises(ValueError, match="cover"):
        _dist(np.ones((1, 5, 5)))
    with pytest.raises(ValueError, match="non-negative"):
        _dist(np.full((1, _GOALS, _GOALS), -0.1))
    with pytest.raises(ValueError, match="positive"):
        _dist(np.zeros((1, _GOALS, _GOALS)))
    with pytest.raises(ValueError, match="lengths"):
        ScorelineDist(np.array([1, 2]), np.ones((1, _GOALS, _GOALS)))
    with pytest.raises(ValueError, match="side"):
        _dist(np.ones((1, _GOALS, _GOALS))).p_clean_sheet("both")  # type: ignore[arg-type]


def test_sample_follows_a_point_mass_and_a_seed() -> None:
    pmf = np.zeros((2, _GOALS, _GOALS))
    pmf[0, 2, 3] = 1.0
    pmf[1, 0, 0] = 1.0
    dist = _dist(pmf)
    drawn = dist.sample(np.random.default_rng(7), 4)
    assert drawn.shape == (2, 4, 2)
    assert np.all(drawn[0] == np.array([2, 3]))
    assert np.all(drawn[1] == np.array([0, 0]))
    again = dist.sample(np.random.default_rng(7), 4)
    np.testing.assert_array_equal(drawn, again)
    assert dist.sample(np.random.default_rng(1), 0).shape == (2, 0, 2)
    with pytest.raises(ValueError, match="non-negative"):
        dist.sample(np.random.default_rng(1), -1)


@given(
    weights=arrays(
        np.float64,
        (2, _GOALS, _GOALS),
        elements=st.floats(0, 3, allow_nan=False, allow_infinity=False),
    )
)
def test_renormalised_grid_is_a_distribution(weights: np.ndarray) -> None:
    if np.any(weights.sum(axis=(1, 2)) <= 0):
        return
    dist = _dist(weights)
    totals = dist.pmf.sum(axis=(1, 2))
    np.testing.assert_allclose(totals, np.ones(2))
    for side in ("home", "away"):
        clean = dist.p_clean_sheet(side)
        assert np.all((clean >= -1e-12) & (clean <= 1 + 1e-12))
        marginal = dist.marginal(side)
        np.testing.assert_allclose(marginal.sum(axis=1), np.ones(2))


def _minutes(pmf: np.ndarray) -> MinutesDist:
    keys = pl.DataFrame(
        {
            "player_key": list(range(pmf.shape[0])),
            "fixture_id": [1] * pmf.shape[0],
        }
    )
    return MinutesDist(keys, MINUTE_BINS, pmf)


def test_minutes_sample_stays_on_the_bins_and_is_independent() -> None:
    pmf = np.zeros((2, len(MINUTE_BINS)))
    pmf[0, 0] = 1.0
    pmf[1, -1] = 1.0
    dist = _minutes(pmf)
    drawn = dist.sample(np.random.default_rng(3), 5)
    assert drawn.shape == (2, 5)
    assert np.all(drawn[0] == 0)
    assert np.all(drawn[1] == 90)
    assert dist.sample(np.random.default_rng(1), 0).shape == (2, 0)
    empty = _minutes(np.zeros((0, len(MINUTE_BINS))))
    assert empty.sample(np.random.default_rng(1), 4).shape == (0, 4)


def test_invalid_minutes_are_refused() -> None:
    pmf = np.ones((1, len(MINUTE_BINS)))
    with pytest.raises(ValueError, match="player_key"):
        MinutesDist(pl.DataFrame({"fixture_id": [1]}), MINUTE_BINS, pmf)
    with pytest.raises(ValueError, match="shape"):
        _minutes(np.ones((1, 3)))
    with pytest.raises(ValueError, match="non-negative"):
        bad = pmf.copy()
        bad[0, 0] = -1
        _minutes(bad)
    with pytest.raises(ValueError, match="positive"):
        _minutes(np.zeros((1, len(MINUTE_BINS))))
    with pytest.raises(ValueError, match="lengths"):
        MinutesDist(
            pl.DataFrame({"player_key": [1, 2], "fixture_id": [1, 1]}),
            MINUTE_BINS,
            pmf,
        )
    with pytest.raises(ValueError, match="non-negative"):
        _minutes(pmf).sample(np.random.default_rng(1), -1)


@given(
    weights=arrays(
        np.float64,
        (2, len(MINUTE_BINS)),
        elements=st.floats(0.1, 3, allow_nan=False, allow_infinity=False),
    )
)
def test_minutes_rows_sum_to_one_and_samples_are_bins(weights: np.ndarray) -> None:
    dist = _minutes(weights)
    np.testing.assert_allclose(dist.pmf.sum(axis=1), np.ones(2))
    drawn = dist.sample(np.random.default_rng(0), 6)
    assert set(np.unique(drawn)).issubset(set(MINUTE_BINS.tolist()))
