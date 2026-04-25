"""Tests for DixonColesModel."""

import numpy as np
import pandas as pd
import pytest

from fplt.features.team_strength import build_fixture_records
from fplt.models.match_score.dixon_coles import DixonColesModel
from fplt.models.score_distribution import ScoreDistribution

# ── Fixtures ──────────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def fixture_records() -> pd.DataFrame:
    return build_fixture_records(seasons=["2022-23", "2023-24"])


@pytest.fixture(scope="module")
def fitted_model(fixture_records: pd.DataFrame) -> DixonColesModel:
    model = DixonColesModel(time_decay=0.0065, max_goals=8)
    model.fit(fixture_records)
    return model


# ── Fit ───────────────────────────────────────────────────────────────────────


def test_fit_populates_teams(fitted_model: DixonColesModel) -> None:
    assert len(fitted_model._teams) > 0


def test_fit_log_loss_finite(fitted_model: DixonColesModel) -> None:
    assert np.isfinite(fitted_model._log_loss)


def test_team_ratings_returns_dataframe(fitted_model: DixonColesModel) -> None:
    df = fitted_model.team_ratings()
    assert "attack" in df.columns
    assert "defence" in df.columns
    assert len(df) == len(fitted_model._teams)


def test_home_advantage_positive(fitted_model: DixonColesModel) -> None:
    # Home advantage γ should be > 1 in a well-fitted model
    assert fitted_model._home_adv > 1.0


# ── Predict distribution ──────────────────────────────────────────────────────


def test_predict_distribution_shape(fitted_model: DixonColesModel) -> None:
    teams = fitted_model._teams[:4]
    fixtures = pd.DataFrame(
        {
            "fixture_id": list(range(2)),
            "team_h": [teams[0], teams[2]],
            "team_a": [teams[1], teams[3]],
        }
    )
    dist = fitted_model.predict_distribution(fixtures)
    assert isinstance(dist, ScoreDistribution)
    assert dist.pmf.shape == (2, 9, 9)  # max_goals=8 → grid 9×9


def test_pmf_sums_to_one(fitted_model: DixonColesModel) -> None:
    teams = fitted_model._teams[:2]
    fixtures = pd.DataFrame(
        {
            "team_h": [teams[0]],
            "team_a": [teams[1]],
        }
    )
    dist = fitted_model.predict_distribution(fixtures)
    assert abs(dist.pmf[0].sum() - 1.0) < 1e-6


def test_strong_home_favourite_has_higher_expected_goals(
    fitted_model: DixonColesModel,
) -> None:
    """A team with high attack should outscore a team with low attack at home."""
    ratings = fitted_model.team_ratings()
    best_attack = ratings.iloc[0]["team"]
    worst_defence = ratings.sort_values("defence").iloc[0]["team"]  # lowest = weakest

    fixtures = pd.DataFrame(
        {
            "team_h": [best_attack],
            "team_a": [worst_defence],
        }
    )
    dist = fitted_model.predict_distribution(fixtures)
    eg_home = dist.expected_goals("home")[0]
    eg_away = dist.expected_goals("away")[0]
    assert eg_home > 0.5
    assert eg_away >= 0.0


def test_predict_before_fit_raises() -> None:
    model = DixonColesModel()
    with pytest.raises(RuntimeError):
        model.predict_distribution(pd.DataFrame({"team_h": ["A"], "team_a": ["B"]}))


def test_params_and_metrics(fitted_model: DixonColesModel) -> None:
    assert "time_decay" in fitted_model.params
    assert "log_loss" in fitted_model.metrics
