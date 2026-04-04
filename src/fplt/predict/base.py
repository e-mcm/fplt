"""Abstract base class for FPL point predictors."""

from abc import ABC, abstractmethod

import numpy as np
import pandas as pd


class BasePredictor(ABC):
    """Interface that all FPL predictors must implement."""

    @abstractmethod
    def fit(self, X: pd.DataFrame, y: pd.Series) -> "BasePredictor":
        """Train the model.

        Args:
            X: Feature DataFrame. Must contain all columns in ``feature_cols``.
            y: Target series (total_points).

        Returns:
            self, to allow chaining.
        """
        ...

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict points for each row in X.

        Args:
            X: Feature DataFrame. Must contain all columns in ``feature_cols``.

        Returns:
            1-D array of predicted points, same length as X.
        """
        ...

    @property
    @abstractmethod
    def feature_cols(self) -> list[str]:
        """Ordered list of feature column names the model was trained on."""
        ...

    @property
    @abstractmethod
    def is_fitted(self) -> bool:
        """True after fit() has been called successfully."""
        ...
