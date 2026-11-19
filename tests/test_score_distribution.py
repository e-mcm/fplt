"""Tests for ScoreDistribution."""

import numpy as np
import pytest

from fplt.models.score_distribution import ScoreDistribution


def _uniform_pmf(n_fixtures: int = 3, max_goals: int = 5) -> np.ndarray:
    g = max_goals + 1
    pmf = np.ones((n_fixtures, g, g)) / (g * g)
    return pmf


def test_pmf_sums_to_one() -> None:
    pmf = _uniform_pmf()
    dist = ScoreDistribution(pmf, list(range(3)))
    for i in range(len(dist)):
        assert abs(dist.pmf[i].sum() - 1.0) < 1e-9


def test_p_clean_sheet_home() -> None:
    # PMF where away always scores 0 → home CS should be 1.0
    pmf = np.zeros((2, 4, 4))
    pmf[:, :, 0] = 0.25  # all mass on away_goals=0
    dist = ScoreDistribution(pmf, [1, 2])
    cs = dist.p_clean_sheet("home")
    np.testing.assert_allclose(cs, [1.0, 1.0], atol=1e-9)


def test_p_clean_sheet_away() -> None:
    pmf = np.zeros((2, 4, 4))
    pmf[:, 0, :] = 0.25  # all mass on home_goals=0 → away CS
    dist = ScoreDistribution(pmf, [10, 20])
    cs = dist.p_clean_sheet("away")
    np.testing.assert_allclose(cs, [1.0, 1.0], atol=1e-9)


def test_p_team_scored_k_marginal() -> None:
    pmf = _uniform_pmf(n_fixtures=1, max_goals=3)
    dist = ScoreDistribution(pmf, [0])
    # Marginal for home side, k=0: sum over all away goals of pmf[0, 0, :]
    expected = pmf[0, 0, :].sum()
    result = dist.p_team_scored_k("home", 0)[0]
    assert abs(result - expected) < 1e-9


def test_p_team_scored_k_out_of_range_returns_zeros() -> None:
    pmf = _uniform_pmf(n_fixtures=2, max_goals=3)
    dist = ScoreDistribution(pmf, [0, 1])
    result = dist.p_team_scored_k("home", 100)
    np.testing.assert_array_equal(result, [0.0, 0.0])


def test_expected_goals_positive() -> None:
    pmf = _uniform_pmf(n_fixtures=3, max_goals=5)
    dist = ScoreDistribution(pmf, list(range(3)))
    eg = dist.expected_goals("home")
    assert (eg > 0).all()


def test_fixture_index() -> None:
    dist = ScoreDistribution(_uniform_pmf(2), ["fix_1", "fix_2"])
    assert dist.fixture_index("fix_2") == 1


def test_invalid_pmf_shape_raises() -> None:
    with pytest.raises(ValueError):
        ScoreDistribution(np.ones((3, 4, 5)), [0, 1, 2])  # non-square


def test_repr() -> None:
    dist = ScoreDistribution(_uniform_pmf(5, max_goals=10), list(range(5)))
    assert "5" in repr(dist)
    assert "10" in repr(dist)
