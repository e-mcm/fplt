"""GK saves model — E[saves] from opponent shot volume features.

Trained only on GK rows.  At predict time, non-GK rows receive 0 saves.

Key features: opp_strength_attack, expected_goals_conceded_rolling_5,
saves_rolling_5, goals_conceded_rolling_5.

Output: DataFrame with column `e_saves`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame

from fplt.models.base import PointModel
from fplt.predict.lgbm import LGBMPredictor
from fplt.schemas import FeatureSchema, SavesPredSchema
from fplt.types import Position


class SavesModel(PointModel):
    """Predicts expected saves for goalkeepers."""

    def __init__(self) -> None:
        self._model = LGBMPredictor()
        self._mae: float = float("nan")

    @property
    def params(self) -> dict[str, object]:
        return {"model": "LGBMPredictor", "position_filter": "GK"}

    @property
    def metrics(self) -> dict[str, float]:
        return {"saves_mae": self._mae}

    def fit(
        self,
        X: DataFrame[FeatureSchema],
        y: pd.DataFrame | pd.Series | None = None,
        **kwargs: object,
    ) -> "SavesModel":
        """Train on GK rows only.

        Args:
            X: Feature DataFrame including `position_enc`.
            y: Series of saves targets.  If None, taken from X['saves'].

        Returns:
            self
        """
        if y is None:
            target = X["saves"]
            X = X.drop(columns=["saves"], errors="ignore")
        else:
            target = y if isinstance(y, pd.Series) else y.iloc[:, 0]

        gk_mask = X["position_enc"] == int(Position.GK)
        if gk_mask.sum() < 10:
            # Fallback: train on all positions if insufficient GK data
            self._model.fit(X, target)
        else:
            self._model.fit(X[gk_mask], target[gk_mask])

        preds = np.clip(self._model.predict(X[gk_mask]), 0.0, 15.0)
        self._mae = float(np.mean(np.abs(preds - target[gk_mask].to_numpy())))
        return self

    @pa.check_types
    def predict(self, X: DataFrame[FeatureSchema]) -> DataFrame[SavesPredSchema]:
        """Return DataFrame with e_saves column (0 for non-GKs)."""
        if not self._model.is_fitted:
            raise RuntimeError("Call fit() before predict().")

        e_saves = np.zeros(len(X), dtype=np.float64)
        gk_mask = (X["position_enc"] == int(Position.GK)).to_numpy()

        if gk_mask.any():
            raw = self._model.predict(X[gk_mask])
            e_saves[gk_mask] = np.clip(raw, 0.0, 15.0)

        return pd.DataFrame({"e_saves": e_saves}, index=X.index)  # type: ignore[return-value]
