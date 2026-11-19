"""Top-level predictor contract and the component protocols beneath it.

``ScorelineDist`` and ``MinutesDist`` stay as placeholders until the score
and minutes models replace them with the concrete classes.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol, Self, runtime_checkable

import polars as pl

from fplt.data.pit import PITView
from fplt.models.spec import ModelSpec
from fplt.types import GWKey


class ScorelineDist(Protocol):
    """Joint distribution over home and away goals."""


class MinutesDist(Protocol):
    """Binned distribution over minutes played."""


@runtime_checkable
class PointsPredictor(Protocol):
    """What the backtester and the optimiser see."""

    spec: ModelSpec

    def fit(self, view: PITView) -> None:
        """Fit on data visible at ``view.as_of``. No-op for external models."""
        ...

    def predict(self, view: PITView, horizon: list[GWKey]) -> pl.DataFrame:
        """Predictions for every player-fixture in the horizon."""
        ...

    def save(self, path: Path) -> None:
        """Write a fitted artifact. Must refuse to overwrite."""
        ...

    @classmethod
    def load(cls, path: Path, spec: ModelSpec) -> Self:
        """Load an artifact written by ``save``."""
        ...


class TeamScoreModel(Protocol):
    """Joint scoreline distribution for a set of fixtures."""

    def fit(self, view: PITView) -> None: ...

    def predict(self, view: PITView, fixtures: pl.DataFrame) -> ScorelineDist: ...


class MinutesModel(Protocol):
    """Minutes distribution for player-fixtures."""

    def fit(self, view: PITView) -> None: ...

    def predict(self, view: PITView, player_fixtures: pl.DataFrame) -> MinutesDist: ...


class InvolvementModel(Protocol):
    """Per-90 goal and assist weights within a team."""

    def fit(self, view: PITView) -> None: ...

    def predict(self, view: PITView, player_fixtures: pl.DataFrame) -> pl.DataFrame: ...


class OtherPointsModel(Protocol):
    """Expected non-goal points conditional on a minutes bucket."""

    def fit(self, view: PITView) -> None: ...

    def predict(self, view: PITView, player_fixtures: pl.DataFrame) -> pl.DataFrame: ...
