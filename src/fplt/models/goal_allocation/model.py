"""Goal allocation model.

Given a ScoreDistribution (from Dixon-Coles) and per-player xG share features,
computes each player's expected goals scored by integrating over the full team
goal distribution:

    E[goals_i] = sum_{k=1}^{K}  P(team scored k)  *  k  *  xg_share_i

This is purely analytical (no learned parameters) when xG data is available.
For seasons without xG, it falls back to a lightweight LightGBM model trained
on position and historical goals rolling averages.

The fallback LightGBM model is what fit() trains.  predict() always prefers
the analytical path when xg_share is non-null.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame

from fplt.models.base import PointModel
from fplt.models.score_distribution import ScoreDistribution
from fplt.predict.lgbm import LGBMPredictor
from fplt.schemas import FeatureSchema, GoalAllocPredSchema

_MAX_GOALS_ITER = 10  # cap the k summation


class GoalAllocationModel(PointModel):
    """Allocates team goals to individual players.

    Two prediction paths:
    1. Analytical (preferred): uses xg_share from player_share.py + ScoreDistribution.
    2. Fallback LightGBM: used when xg_share is missing (pre-xG seasons).

    The ScoreDistribution must be provided at predict time via the
    `score_dist` keyword argument.

    Example:
        >>> model = GoalAllocationModel()
        >>> model.fit(X_train, y_train)
        >>> dist = dixon_coles_model.predict_distribution(fixtures)
        >>> preds = model.predict(X_test, score_dist=dist, fixture_col="fixture")
    """

    def __init__(self) -> None:
        self._fallback = LGBMPredictor()
        self._fitted = False
        self._train_log_loss: float = float("nan")

    # ------------------------------------------------------------------
    # FPLModel interface
    # ------------------------------------------------------------------

    @property
    def params(self) -> dict[str, object]:
        return {"fallback_model": "LGBMPredictor"}

    @property
    def metrics(self) -> dict[str, float]:
        return {"train_log_loss": self._train_log_loss}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: DataFrame[FeatureSchema],
        y: pd.DataFrame | pd.Series | None = None,
        **kwargs: object,
    ) -> GoalAllocationModel:
        """Train the fallback LightGBM model on historical goals data.

        Only rows where xg_share is NaN (missing xG) use the fallback at
        predict time; this model trains on those rows.

        Args:
            X: Player feature DataFrame including goals_scored column when y
               is None.
            y: Series of goals_scored targets.  If None, taken from X.

        Returns:
            self
        """
        if y is None:
            if "goals_scored" not in X.columns:
                raise ValueError("y or X['goals_scored'] required for fit()")
            target = X["goals_scored"]
            features = X.drop(columns=["goals_scored"])
        else:
            target = y if isinstance(y, pd.Series) else y.iloc[:, 0]
            features = X

        self._fallback.fit(features, target)
        self._fitted = True
        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    @pa.check_types
    def predict(
        self,
        X: DataFrame[FeatureSchema],
        score_dist: ScoreDistribution | None = None,
        fixture_col: str = "fixture",
    ) -> DataFrame[GoalAllocPredSchema]:
        """Predict expected goals per player.

        Args:
            X: Player feature DataFrame.  Must contain `xg_share` and
               `was_home` for the analytical path.  Must contain `fixture`
               (or whatever `fixture_col` names) to look up the distribution.
            score_dist: ScoreDistribution from DixonColesModel.  Required for
                        the analytical path; if None, uses fallback only.
            fixture_col: Column name in X containing fixture IDs that match
                         ScoreDistribution.fixture_ids.

        Returns:
            DataFrame with column `expected_goals_scored`, same index as X.
        """
        result = pd.Series(
            np.zeros(len(X)), index=X.index, name="expected_goals_scored"
        )

        xg_available = (
            "xg_share" in X.columns
            and score_dist is not None
            and fixture_col in X.columns
        )

        if xg_available:
            result = self._predict_analytical(X, score_dist, fixture_col)  # type: ignore[arg-type]
        elif self._fitted:
            raw = self._fallback.predict(X)
            result = pd.Series(
                np.clip(raw, 0.0, 5.0), index=X.index, name="expected_goals_scored"
            )
        else:
            # No xG and no fitted fallback — return zeros with a warning
            pass

        return result.to_frame()  # type: ignore[return-value]

    def _predict_analytical(
        self,
        X: pd.DataFrame,
        score_dist: ScoreDistribution,
        fixture_col: str,
    ) -> pd.Series:
        """Integrate over the team goal distribution using xg_share."""
        out = np.zeros(len(X))
        xg_shares = X["xg_share"].fillna(0.0).to_numpy()
        was_home = (
            X["was_home"].to_numpy()
            if "was_home" in X.columns
            else np.ones(len(X), dtype=bool)
        )
        fixture_ids = X[fixture_col].to_numpy()

        for i in range(len(X)):
            fid = fixture_ids[i]
            try:
                fi = score_dist.fixture_index(fid)
            except ValueError:
                continue  # unknown fixture — leave as 0

            side = "home" if was_home[i] else "away"
            e_goals = 0.0
            for k in range(1, _MAX_GOALS_ITER + 1):
                p_k = score_dist.p_team_scored_k(side, k)[fi]
                e_goals += p_k * k * xg_shares[i]
            out[i] = e_goals

        return pd.Series(out, index=X.index, name="expected_goals_scored")
