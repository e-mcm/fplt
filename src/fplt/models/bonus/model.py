"""Bonus points model — position-stratified LightGBM.

Wraps ByPositionPredictor to predict expected bonus points (0–3) per player.
Key features: bps_rolling_5, ict_index_rolling_5.

Output: DataFrame with column `e_bonus`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame

from fplt.models.base import PointModel
from fplt.predict.by_position import ByPositionPredictor
from fplt.schemas import BonusPredSchema, FeatureSchema


class BonusModel(PointModel):
    """Predicts expected bonus points using position-stratified LightGBM."""

    def __init__(self) -> None:
        self._model = ByPositionPredictor()
        self._mae: float = float("nan")

    @property
    def params(self) -> dict[str, object]:
        return {"model": "ByPositionPredictor"}

    @property
    def metrics(self) -> dict[str, float]:
        return {"bonus_mae": self._mae}

    def fit(
        self,
        X: DataFrame[FeatureSchema],
        y: pd.DataFrame | pd.Series | None = None,
        **kwargs: object,
    ) -> "BonusModel":
        """Train position-stratified bonus predictor.

        Args:
            X: Feature DataFrame.
            y: Series of bonus targets.  If None, taken from X['bonus'].

        Returns:
            self
        """
        if y is None:
            target = X["bonus"]
            X = X.drop(columns=["bonus"], errors="ignore")
        else:
            target = y if isinstance(y, pd.Series) else y.iloc[:, 0]

        self._model.fit(X, target)
        preds = np.clip(self._model.predict(X), 0.0, 3.0)
        self._mae = float(np.mean(np.abs(preds - target.to_numpy())))
        return self

    @pa.check_types
    def predict(self, X: DataFrame[FeatureSchema]) -> DataFrame[BonusPredSchema]:
        """Return DataFrame with e_bonus column."""
        if not self._model.is_fitted:
            raise RuntimeError("Call fit() before predict().")
        raw = self._model.predict(X)
        return pd.DataFrame(  # type: ignore[return-value]
            {"e_bonus": np.clip(raw, 0.0, 3.0)},
            index=X.index,
        )
