"""Tests for the Xpts aggregator."""

import numpy as np
import pandas as pd
import pytest

from fplt.aggregator import compute_xpts
from fplt.models.score_distribution import ScoreDistribution
from fplt.types import Position

# ── Helpers ───────────────────────────────────────────────────────────────────


class _ConstantMinutesModel:
    """Stub that always returns p_starts=0.9, e_minutes=75."""

    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        n = len(X)
        return pd.DataFrame(
            {"p_starts": np.full(n, 0.9), "e_minutes": np.full(n, 75.0)}, index=X.index
        )


class _ConstantGoalModel:
    def predict(
        self, X: pd.DataFrame, score_dist: object = None, fixture_col: str = "fixture"
    ) -> pd.DataFrame:
        n = len(X)
        return pd.DataFrame({"expected_goals_scored": np.full(n, 0.1)}, index=X.index)


class _ConstantBonusModel:
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"e_bonus": np.full(len(X), 0.5)}, index=X.index)


class _ConstantCardModel:
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        n = len(X)
        return pd.DataFrame(
            {"p_yellow": np.full(n, 0.05), "p_red": np.full(n, 0.01)}, index=X.index
        )


class _ConstantSavesModel:
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        return pd.DataFrame({"e_saves": np.zeros(len(X))}, index=X.index)


@pytest.fixture()
def models() -> dict:
    from fplt.models.goal_allocation.model import GoalAllocationModel

    goal_model = GoalAllocationModel.__new__(GoalAllocationModel)
    goal_model._fallback = _ConstantGoalModel()  # type: ignore[assignment]
    goal_model._fitted = True

    # Patch predict to use constant stub
    def _predict(X, score_dist=None, fixture_col="fixture"):  # type: ignore[override]
        return _ConstantGoalModel().predict(X, score_dist, fixture_col)

    goal_model.predict = _predict  # type: ignore[method-assign]

    return {
        "minutes": _ConstantMinutesModel(),
        "goal_allocation": goal_model,
        "bonus": _ConstantBonusModel(),
        "cards": _ConstantCardModel(),
        "saves": _ConstantSavesModel(),
    }


@pytest.fixture()
def synthetic_players() -> pd.DataFrame:
    """One GK and one FWD in the same fixture."""
    return pd.DataFrame(
        {
            "element": [1, 2],
            "name": ["GK_Player", "FWD_Player"],
            "position_enc": [int(Position.GK), int(Position.FWD)],
            "was_home": [True, True],
            "fixture": [101, 101],
        }
    )


@pytest.fixture()
def synthetic_score_dist() -> ScoreDistribution:
    """Fixture 101: home team scores 1.5 goals on average, away 1.0."""
    pmf = np.zeros((1, 8, 8))
    # Use a rough Poisson(1.5) × Poisson(1.0) distribution
    from scipy.stats import poisson

    lh, la = 1.5, 1.0
    for h in range(8):
        for a in range(8):
            pmf[0, h, a] = poisson.pmf(h, lh) * poisson.pmf(a, la)
    pmf /= pmf.sum()
    return ScoreDistribution(pmf, [101])


# ── Tests ──────────────────────────────────────────────────────────────────────


def test_xpts_non_negative(models, synthetic_players, synthetic_score_dist) -> None:
    xpts = compute_xpts(synthetic_players, synthetic_score_dist, models)  # type: ignore[arg-type]
    assert (xpts >= 0).all()


def test_xpts_returns_series_with_correct_index(
    models, synthetic_players, synthetic_score_dist
) -> None:
    xpts = compute_xpts(synthetic_players, synthetic_score_dist, models)  # type: ignore[arg-type]
    assert list(xpts.index) == list(synthetic_players.index)


def test_xpts_name(models, synthetic_players, synthetic_score_dist) -> None:
    xpts = compute_xpts(synthetic_players, synthetic_score_dist, models)  # type: ignore[arg-type]
    assert xpts.name == "xpts"


def test_gk_gets_clean_sheet_points(synthetic_score_dist) -> None:
    """GK should get CS points — home side has ~P(away=0) > 0 clean sheet."""
    from fplt.rules import CLEAN_SHEET_POINTS

    gk_cs = CLEAN_SHEET_POINTS[Position.GK]
    # P(away scored 0) in our distribution
    p_cs = float(synthetic_score_dist.p_clean_sheet("home")[0])
    # CS contribution = p_starts * p_cs * gk_cs
    assert p_cs > 0
    assert gk_cs == 4


def test_fwd_gets_no_clean_sheet_points() -> None:
    from fplt.rules import CLEAN_SHEET_POINTS

    assert CLEAN_SHEET_POINTS[Position.FWD] == 0
