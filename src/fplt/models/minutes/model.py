"""Minutes model — two-stage: P(starts) then E[minutes | starts].

Stage 1: LightGBM classifier → P(player starts) ∈ [0, 1].
Stage 2: LightGBM regressor on starter rows → E[minutes | starts].

Output: DataFrame with columns `p_starts` and `e_minutes`.
`e_minutes` is the unconditional expectation: p_starts * E[min | starts].
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame

from fplt.models.base import PointModel
from fplt.predict.lgbm import DEFAULT_PARAMS, LGBMPredictor
from fplt.schemas import FeatureSchema, MinutesPredSchema

_STARTS_PARAMS = {**DEFAULT_PARAMS, "objective": "binary", "metric": "binary_logloss"}
_MINUTES_PARAMS = {**DEFAULT_PARAMS, "objective": "regression", "metric": "mae"}


class MinutesModel(PointModel):
    """Predicts P(starts) and E[minutes | starts] independently.

    Key features used: minutes_rolling_3/5/10, starts_rolling_5, minutes_std_5.

    Example:
        >>> model = MinutesModel()
        >>> model.fit(X_train, y_train)  # y_train has 'starts' and 'minutes'
        >>> preds = model.predict(X_test)
        >>> preds[["p_starts", "e_minutes"]]
    """

    def __init__(self) -> None:
        self._starts_model = LGBMPredictor(params=_STARTS_PARAMS)
        self._minutes_model = LGBMPredictor(params=_MINUTES_PARAMS)
        self._mae: float = float("nan")

    @property
    def params(self) -> dict[str, object]:
        return {"stage1": "binary_lgbm", "stage2": "regression_lgbm"}

    @property
    def metrics(self) -> dict[str, float]:
        return {"minutes_mae": self._mae}

    def fit(
        self,
        X: DataFrame[FeatureSchema],
        y: pd.DataFrame | pd.Series | None = None,
        **kwargs: object,
    ) -> MinutesModel:
        """Train both stages.

        Args:
            X: Feature DataFrame.
            y: DataFrame with columns 'starts' (binary) and 'minutes'.
               If y is None, both are taken from X.

        Returns:
            self
        """
        if y is None:
            targets = X[["starts", "minutes"]].copy()
            X = X.drop(columns=["starts", "minutes"], errors="ignore")
        else:
            targets = y if isinstance(y, pd.DataFrame) else y.to_frame()

        y_starts = (
            (targets["starts"] > 0).astype(float)
            if "starts" in targets.columns
            else (targets["minutes"] >= 60).astype(float)
        )
        y_minutes = targets["minutes"]

        # Stage 1: starts classifier
        self._starts_model.fit(X, y_starts)

        # Stage 2: minutes regressor — train only on rows where player started
        started_mask = y_starts > 0
        if started_mask.sum() > 10:
            self._minutes_model.fit(X[started_mask], y_minutes[started_mask])
        else:
            self._minutes_model.fit(X, y_minutes)

        # Compute training MAE for e_minutes
        p_s = self._starts_model.predict(X)
        e_min_given_starts = self._minutes_model.predict(X)
        e_min = p_s * e_min_given_starts
        self._mae = float(np.mean(np.abs(e_min - y_minutes.to_numpy())))

        return self

    @pa.check_types
    def predict(self, X: DataFrame[FeatureSchema]) -> DataFrame[MinutesPredSchema]:
        """Return DataFrame with p_starts and e_minutes columns."""
        if not self._starts_model.is_fitted:
            raise RuntimeError("Call fit() before predict().")

        p_starts = np.clip(self._starts_model.predict(X), 0.0, 1.0)
        e_min_given_starts = np.clip(self._minutes_model.predict(X), 0.0, 90.0)
        e_minutes = p_starts * e_min_given_starts

        return pd.DataFrame(  # type: ignore[return-value]
            {"p_starts": p_starts, "e_minutes": e_minutes},
            index=X.index,
        )
