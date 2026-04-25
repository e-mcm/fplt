"""Position-stratified LightGBM predictor.

Trains one LGBMPredictor per position (GK, DEF, MID, FWD) so each model
learns position-specific scoring patterns without being distorted by the
others.
"""

import numpy as np
import pandas as pd
from pandera.typing import DataFrame

from fplt.predict.base import BasePredictor
from fplt.predict.lgbm import LGBMPredictor
from fplt.schemas import FeatureImportanceSchema, FeatureSchema
from fplt.types import Position


class ByPositionPredictor(BasePredictor):
    """Trains a separate LGBMPredictor for each outfield position.

    Rows with unknown position (position_enc == -1 or 2 for AM) are routed
    to the MID model as a sensible fallback.

    Example:
        >>> model = ByPositionPredictor()
        >>> model.fit(X_train, y_train)
        >>> preds = model.predict(X_test)
    """

    def __init__(self, **lgbm_kwargs: object) -> None:
        """
        Args:
            **lgbm_kwargs: Forwarded to each LGBMPredictor (e.g. num_rounds).
        """
        self._lgbm_kwargs = lgbm_kwargs
        self._models: dict[int, LGBMPredictor] = {}

    def fit(
        self,
        X: DataFrame[FeatureSchema],
        y: pd.Series,
        X_val: DataFrame[FeatureSchema] | None = None,
        y_val: pd.Series | None = None,
    ) -> "ByPositionPredictor":
        """Train one model per position.

        Args:
            X: Feature DataFrame. Must contain ``position_enc``.
            y: Target series (total_points).
            X_val: Optional validation features for early stopping.
            y_val: Optional validation targets for early stopping.

        Returns:
            self
        """
        self._models = {}

        for pos in Position:
            mask = X["position_enc"] == pos
            if mask.sum() == 0:
                continue

            X_pos = X[mask]
            y_pos = y[mask]

            X_val_pos: pd.DataFrame | None = None
            y_val_pos: pd.Series | None = None
            if X_val is not None and y_val is not None:
                val_mask = X_val["position_enc"] == pos
                if val_mask.sum() > 0:
                    X_val_pos = X_val[val_mask]
                    y_val_pos = y_val[val_mask]

            model = LGBMPredictor(**self._lgbm_kwargs)  # type: ignore[arg-type]
            model.fit(X_pos, y_pos, X_val=X_val_pos, y_val=y_val_pos)
            self._models[pos] = model
            print(f"  [{pos.name}] trained on {mask.sum():,} rows")

        return self

    def predict(self, X: DataFrame[FeatureSchema]) -> np.ndarray:
        """Predict points, routing each row to its position model.

        Rows with unrecognised position_enc fall back to the MID model.
        """
        if not self._models:
            raise RuntimeError("Call fit() before predict().")

        preds = np.zeros(len(X), dtype=np.float64)

        for pos_enc, model in self._models.items():
            mask = (X["position_enc"] == pos_enc).to_numpy()
            if mask.any():
                preds[mask] = model.predict(X[mask])

        # Fallback: rows not matched by any position → MID model
        fallback_model = self._models.get(Position.MID)
        if fallback_model is not None:
            unmatched = ~np.isin(X["position_enc"].to_numpy(), list(self._models))
            if unmatched.any():
                preds[unmatched] = fallback_model.predict(X[unmatched])

        return preds

    @property
    def feature_cols(self) -> list[str]:
        if not self._models:
            raise RuntimeError("Call fit() before accessing feature_cols.")
        # All position models share the same feature set
        return next(iter(self._models.values())).feature_cols

    @property
    def is_fitted(self) -> bool:
        return len(self._models) > 0

    def feature_importance(self) -> dict[str, DataFrame[FeatureImportanceSchema]]:
        """Return feature importances per position model.

        Returns:
            Dict mapping position label → importance DataFrame.
        """
        if not self._models:
            raise RuntimeError("Call fit() before feature_importance().")
        return {
            pos.name: model.feature_importance() for pos, model in self._models.items()
        }
