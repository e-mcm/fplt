"""LightGBM-based FPL points predictor."""

from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd

from fplt.predict.base import BasePredictor

# Always present after feature engineering (all seasons)
BASE_FEATURE_COLS: list[str] = [
    "position_enc",
    "was_home",
    "value",
    "opponent_team",
    "opp_strength_overall",
    "opp_strength_attack",
    "opp_strength_defence",
    "total_points_rolling_3",
    "total_points_rolling_5",
    "total_points_rolling_10",
    "minutes_rolling_3",
    "minutes_rolling_5",
    "minutes_rolling_10",
    "minutes_std_5",
    "goals_scored_rolling_5",
    "assists_rolling_5",
    "clean_sheets_rolling_5",
    "goals_conceded_rolling_5",
    "saves_rolling_5",
    "bonus_rolling_5",
    "bps_rolling_5",
    "ict_index_rolling_5",
    "threat_rolling_5",
    "creativity_rolling_5",
    "influence_rolling_5",
]

# Present from ~2019-20 onwards — included when available
OPTIONAL_FEATURE_COLS: list[str] = [
    "xP_rolling_3",
    "xP_rolling_5",
    "xP_rolling_10",
    "starts_rolling_5",
    "expected_goals_rolling_5",
    "expected_assists_rolling_5",
    "expected_goal_involvements_rolling_5",
    "expected_goals_conceded_rolling_5",
]

DEFAULT_PARAMS: dict[str, Any] = {
    "objective": "regression",
    "metric": "mae",
    "learning_rate": 0.05,
    "num_leaves": 63,
    "min_child_samples": 20,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
}

DEFAULT_NUM_ROUNDS = 500


class LGBMPredictor(BasePredictor):
    """LightGBM regression model for predicting FPL gameweek points.

    Train on historical seasons, predict on future GWs. GW-1 rows (which have
    all-NaN rolling features) should be excluded from training data — LightGBM
    handles NaN natively for mid-season arrivals.

    Example:
        >>> from fplt.predict.lgbm import LGBMPredictor
        >>> model = LGBMPredictor()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
    """

    def __init__(
        self,
        params: dict[str, Any] | None = None,
        num_rounds: int = DEFAULT_NUM_ROUNDS,
    ) -> None:
        self.params = params or DEFAULT_PARAMS
        self.num_rounds = num_rounds
        self._model: lgb.Booster | None = None
        self._feature_cols: list[str] = []

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        X_val: pd.DataFrame | None = None,
        y_val: pd.Series | None = None,
    ) -> "LGBMPredictor":
        """Train the model.

        Args:
            X: Feature DataFrame.
            y: Target series (total_points).
            X_val: Optional validation features for early stopping.
            y_val: Optional validation targets for early stopping.

        Returns:
            self
        """
        self._feature_cols = [
            c for c in BASE_FEATURE_COLS + OPTIONAL_FEATURE_COLS
            if c in X.columns
        ]

        train_data = lgb.Dataset(X[self._feature_cols], label=y)

        callbacks: list[Any] = []
        valid_sets: list[lgb.Dataset] = [train_data]
        valid_names: list[str] = ["train"]

        if X_val is not None and y_val is not None:
            val_data = lgb.Dataset(
                X_val[self._feature_cols], label=y_val, reference=train_data
            )
            valid_sets.append(val_data)
            valid_names.append("val")
            callbacks.append(lgb.early_stopping(50, verbose=False))
            callbacks.append(lgb.log_evaluation(period=-1))
        else:
            callbacks.append(lgb.log_evaluation(period=-1))

        self._model = lgb.train(
            self.params,
            train_data,
            num_boost_round=self.num_rounds,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict points for each row in X."""
        if self._model is None:
            raise RuntimeError("Call fit() before predict().")
        return self._model.predict(X[self._feature_cols])  # type: ignore[return-value]

    @property
    def feature_cols(self) -> list[str]:
        return list(self._feature_cols)

    @property
    def is_fitted(self) -> bool:
        return self._model is not None

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances sorted descending by gain.

        Returns:
            DataFrame with columns ['feature', 'importance'].
        """
        if self._model is None:
            raise RuntimeError("Call fit() before feature_importance().")
        return (
            pd.DataFrame(
                {
                    "feature": self._feature_cols,
                    "importance": self._model.feature_importance(
                        importance_type="gain"
                    ),
                }
            )
            .sort_values("importance", ascending=False)
            .reset_index(drop=True)
        )

    def save(self, path: str | Path) -> None:
        """Save the trained booster to disk."""
        if self._model is None:
            raise RuntimeError("Call fit() before save().")
        self._model.save_model(str(path))

    @classmethod
    def load(cls, path: str | Path) -> "LGBMPredictor":
        """Load a previously saved booster from disk."""
        instance = cls.__new__(cls)
        instance._model = lgb.Booster(model_file=str(path))
        instance._feature_cols = instance._model.feature_name()
        instance.params = DEFAULT_PARAMS
        instance.num_rounds = DEFAULT_NUM_ROUNDS
        return instance
