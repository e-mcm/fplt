"""Card model — P(yellow card) and P(red card) per player.

Two independent LightGBM classifiers.  Red cards are rare; the model
predicts a probability which feeds into the aggregator as a penalty.

Output: DataFrame with columns `p_yellow` and `p_red`.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame

from fplt.models.base import PointModel
from fplt.predict.lgbm import DEFAULT_PARAMS, LGBMPredictor
from fplt.schemas import CardPredSchema, FeatureSchema

_CARD_PARAMS = {**DEFAULT_PARAMS, "objective": "binary", "metric": "binary_logloss"}


class CardModel(PointModel):
    """Predicts P(yellow card) and P(red card) as independent classifiers."""

    def __init__(self) -> None:
        self._yellow_model = LGBMPredictor(params=_CARD_PARAMS)
        self._red_model = LGBMPredictor(params=_CARD_PARAMS)
        self._yellow_brier: float = float("nan")
        self._red_brier: float = float("nan")

    @property
    def params(self) -> dict[str, object]:
        return {
            "yellow_model": "LGBMPredictor_binary",
            "red_model": "LGBMPredictor_binary",
        }

    @property
    def metrics(self) -> dict[str, float]:
        return {"yellow_brier": self._yellow_brier, "red_brier": self._red_brier}

    def fit(
        self,
        X: DataFrame[FeatureSchema],
        y: pd.DataFrame | pd.Series | None = None,
        **kwargs: object,
    ) -> CardModel:
        """Train yellow and red card classifiers.

        Args:
            X: Feature DataFrame.
            y: DataFrame with columns 'yellow_cards' and 'red_cards'.
               If None, taken from X.

        Returns:
            self
        """
        if y is None:
            targets = X[["yellow_cards", "red_cards"]].copy()
            X = X.drop(columns=["yellow_cards", "red_cards"], errors="ignore")
        else:
            targets = y if isinstance(y, pd.DataFrame) else y.to_frame()

        y_yellow = (targets["yellow_cards"] > 0).astype(float)
        y_red = (
            (targets["red_cards"] > 0).astype(float)
            if "red_cards" in targets.columns
            else pd.Series(np.zeros(len(X)), index=X.index)
        )

        self._yellow_model.fit(X, y_yellow)
        self._red_model.fit(X, y_red)

        p_y = np.clip(self._yellow_model.predict(X), 0.0, 1.0)
        p_r = np.clip(self._red_model.predict(X), 0.0, 1.0)
        self._yellow_brier = float(np.mean((p_y - y_yellow.to_numpy()) ** 2))
        self._red_brier = float(np.mean((p_r - y_red.to_numpy()) ** 2))

        return self

    @pa.check_types
    def predict(self, X: DataFrame[FeatureSchema]) -> DataFrame[CardPredSchema]:
        """Return DataFrame with p_yellow and p_red columns."""
        if not self._yellow_model.is_fitted:
            raise RuntimeError("Call fit() before predict().")
        p_yellow = np.clip(self._yellow_model.predict(X), 0.0, 1.0)
        p_red = np.clip(self._red_model.predict(X), 0.0, 1.0)
        return pd.DataFrame(  # type: ignore[return-value]
            {"p_yellow": p_yellow, "p_red": p_red},
            index=X.index,
        )
