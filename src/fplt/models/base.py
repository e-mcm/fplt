"""Abstract base classes for all FPL pipeline models.

Two model types:
- PointModel: outputs point estimates or probabilities per input row.
- DistributionalModel: outputs a full probability distribution per input row.

Serialization is handled by MLflow (pickle of __dict__).  The prediction
methods exist for the pipeline, not for MLflow.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

import pandas as pd

if TYPE_CHECKING:
    from fplt.models.score_distribution import ScoreDistribution


class FPLModel(ABC):
    """Abstract base for all FPL pipeline models."""

    @abstractmethod
    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series | None = None,
        **kwargs: object,
    ) -> FPLModel:
        """Fit the model. Returns self for chaining."""
        ...

    @property
    @abstractmethod
    def params(self) -> dict[str, object]:
        """Hyperparameters logged to MLflow on each training run."""
        ...

    @property
    @abstractmethod
    def metrics(self) -> dict[str, float]:
        """Evaluation metrics populated after fit(), logged to MLflow."""
        ...


class PointModel(FPLModel):
    """Models that output point estimates or probabilities — one row per input row.

    Examples: MinutesModel, GoalAllocationModel, BonusModel.
    """

    @abstractmethod
    def predict(self, X: pd.DataFrame) -> pd.DataFrame:
        """Return a DataFrame of predictions, one column per output quantity."""
        ...


class DistributionalModel(FPLModel):
    """Models that output a full probability distribution per input row.

    Examples: DixonColesModel.
    """

    @abstractmethod
    def predict_distribution(self, X: pd.DataFrame) -> ScoreDistribution:
        """Return a ScoreDistribution for the given fixture rows."""
        ...
