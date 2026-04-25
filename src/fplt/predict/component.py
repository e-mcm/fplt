"""Component-based FPL predictor.

Trains a separate LGBMPredictor for each scoring component (minutes, goals,
assists, clean_sheets, saves, bonus), then combines predictions using the
official FPL scoring rules.

This avoids the mean-regression problem of direct point prediction: the model
is free to predict a high goal probability independently of a high assist
probability, and the rules engine combines them correctly.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame

from fplt.predict.base import BasePredictor
from fplt.predict.lgbm import LGBMPredictor
from fplt.rules import score_components
from fplt.schemas import ComponentPredSchema, FeatureImportanceSchema, FeatureSchema


@dataclass(frozen=True)
class ComponentConfig:
    """Configuration for a single scoring component model.

    Attributes:
        name: Identifier used as the model key and output column name.
        column: Column name in the training DataFrame.
        clip_min: Lower bound applied to predictions (physically plausible).
        clip_max: Upper bound applied to predictions (physically plausible).
    """

    name: str
    column: str
    clip_min: float
    clip_max: float


COMPONENTS: list[ComponentConfig] = [
    ComponentConfig("minutes", "minutes", 0.0, 90.0),
    ComponentConfig("goals_scored", "goals_scored", 0.0, 5.0),
    ComponentConfig("assists", "assists", 0.0, 3.0),
    ComponentConfig("clean_sheets", "clean_sheets", 0.0, 1.0),
    ComponentConfig("saves", "saves", 0.0, 15.0),
    ComponentConfig("bonus", "bonus", 0.0, 3.0),
    ComponentConfig("yellow_cards", "yellow_cards", 0.0, 1.0),
]


class ComponentPredictor(BasePredictor):
    """Predicts each FPL scoring component separately, then combines via rules.

    Each component gets its own ``BasePredictor`` instance, injected via the
    ``predictors`` argument. This allows different model types per component
    (e.g. a classifier for clean_sheets, a regressor for minutes).

    Any component not supplied in ``predictors`` falls back to a default
    ``LGBMPredictor()``.

    Example:
        >>> # All components use LGBMPredictor with default params
        >>> model = ComponentPredictor()

        >>> # Override the goals model; rest stay as LGBMPredictor
        >>> model = ComponentPredictor(predictors={
        ...     "goals_scored": ByPositionPredictor(num_rounds=300),
        ... })

        >>> model.fit(X_train, y_train_df)
        >>> preds = model.predict(X_test)
    """

    def __init__(
        self,
        predictors: dict[str, BasePredictor] | None = None,
    ) -> None:
        """
        Args:
            predictors: Mapping of component name → unfitted BasePredictor.
                        Components absent from this dict use LGBMPredictor().
        """
        provided = predictors or {}
        self._predictors: dict[str, BasePredictor] = {
            cfg.name: provided.get(cfg.name, LGBMPredictor()) for cfg in COMPONENTS
        }

    def fit(
        self,
        X: DataFrame[FeatureSchema],
        y: pd.DataFrame,
        X_val: DataFrame[FeatureSchema] | None = None,
        y_val: pd.DataFrame | None = None,
    ) -> "ComponentPredictor":
        """Train each injected predictor on its component target.

        Args:
            X: Feature DataFrame.
            y: DataFrame with one column per component target.
            X_val: Optional validation features.
            y_val: Optional validation targets (same columns as y).

        Returns:
            self
        """
        for cfg in COMPONENTS:
            if cfg.column not in y.columns:
                continue

            y_val_series: pd.Series | None = None
            if y_val is not None and cfg.column in y_val.columns:
                y_val_series = y_val[cfg.column]

            self._predictors[cfg.name].fit(
                X,
                y[cfg.column],
                X_val=X_val,  # type: ignore[call-arg]
                y_val=y_val_series,  # type: ignore[call-arg]
            )
            print(f"  [{cfg.name}] trained")

        return self

    @pa.check_types
    def predict_components(
        self, X: DataFrame[FeatureSchema]
    ) -> DataFrame[ComponentPredSchema]:
        """Return predicted value for each component as a DataFrame.

        Returns:
            DataFrame with one column per component, clipped to valid range.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before predict_components().")

        out: dict[str, np.ndarray] = {}
        for cfg in COMPONENTS:
            predictor = self._predictors.get(cfg.name)
            if predictor is not None and predictor.is_fitted:
                raw = predictor.predict(X)
                out[cfg.name] = np.clip(raw, cfg.clip_min, cfg.clip_max)

        return pd.DataFrame(out, index=X.index)  # type: ignore[return-value]

    def predict(self, X: DataFrame[FeatureSchema]) -> np.ndarray:
        """Predict total points by combining component predictions via rules.

        Args:
            X: Feature DataFrame. Must contain ``position_enc``.

        Returns:
            1-D array of predicted total points.
        """
        components = self.predict_components(X)
        preds = np.zeros(len(X), dtype=np.float64)

        for i, (idx, row) in enumerate(components.iterrows()):
            pos = int(X.loc[idx, "position_enc"])
            preds[i] = score_components(
                position_enc=pos,
                minutes=row.get("minutes", 0.0),
                goals_scored=row.get("goals_scored", 0.0),
                assists=row.get("assists", 0.0),
                clean_sheets=row.get("clean_sheets", 0.0),
                saves=row.get("saves", 0.0),
                yellow_cards=row.get("yellow_cards", 0.0),
                bonus=row.get("bonus", 0.0),
            )

        return preds

    @property
    def feature_cols(self) -> list[str]:
        fitted = [p for p in self._predictors.values() if p.is_fitted]
        if not fitted:
            raise RuntimeError("Call fit() before accessing feature_cols.")
        return fitted[0].feature_cols

    @property
    def is_fitted(self) -> bool:
        return any(p.is_fitted for p in self._predictors.values())

    def feature_importance(self) -> dict[str, DataFrame[FeatureImportanceSchema]]:
        """Return feature importances per component model.

        Returns:
            Dict mapping component name → importance DataFrame.
        """
        if not self.is_fitted:
            raise RuntimeError("Call fit() before feature_importance().")
        return {
            name: predictor.feature_importance()  # type: ignore[attr-defined]
            for name, predictor in self._predictors.items()
            if predictor.is_fitted
        }
