"""Direct LightGBM points model. Hyperparameters live in the YAML spec."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import lightgbm as lgb
import numpy as np
import polars as pl

from fplt.data.pit import PITView
from fplt.features.form import FEATURE_COLUMNS, training_frame
from fplt.features.registry import build_feature_matrix
from fplt.models.predictions import prediction_frame
from fplt.models.spec import ModelSpec, register_model, write_identity
from fplt.models.time_decay import calendar_sample_weights
from fplt.models.universe import player_fixtures
from fplt.types import GWKey

_DEFAULT_FEATURES = ["context@1", "rolling_form@1", "opponent_strength@1"]
_PARAM_KEYS = (
    "learning_rate",
    "num_leaves",
    "min_child_samples",
    "feature_fraction",
    "bagging_fraction",
    "bagging_freq",
)


@register_model("lgbm_direct")
class LGBMDirectPredictor:
    """Regression on PIT features for total points per player-fixture.

    ``fit`` trains only on appearances visible in the view. ``predict`` joins
    the features named in the spec through ``build_feature_matrix`` so the
    walk-forward slices match every other ``PointsPredictor``.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._booster: lgb.Booster | None = None
        self._features = list(spec.features) or list(_DEFAULT_FEATURES)

    def fit(self, view: PITView) -> None:
        """Train on visible appearances. A tiny sample leaves the model unfitted.

        When ``spec.train_window.time_decay`` is set, rows are weighted by
        ``exp(-λ × days_before_cutoff)`` via ``calendar_sample_weights``.
        """
        train = training_frame(view)
        if train.height < 2:
            self._booster = None
            return
        matrix = train.select(FEATURE_COLUMNS).to_numpy()
        label = train["total_points"].to_numpy().astype(float)
        weights = calendar_sample_weights(
            train["kickoff_time"].to_numpy(),
            view.cutoff,
            self.spec.train_window.time_decay,
        )
        dataset = lgb.Dataset(
            matrix,
            label=label,
            weight=weights,
            feature_name=list(FEATURE_COLUMNS),
        )
        booster = lgb.train(  # pyright: ignore[reportUnknownMemberType]
            _params(self.spec),
            dataset,
            num_boost_round=int(self.spec.params.get("num_boost_round", 100)),
        )
        self._booster = booster

    def predict(self, view: PITView, horizon: list[GWKey]) -> pl.DataFrame:
        """Score every player-fixture in ``horizon``. Returns ``PredictionSchema``."""
        rows = player_fixtures(view, horizon)
        if rows.is_empty():
            zero = pl.lit(0.0)
            return prediction_frame(self.spec, view.as_of, rows, zero, zero, zero, zero)
        matrix = build_feature_matrix(view, self._features, horizon)
        values = (
            matrix.select(FEATURE_COLUMNS)
            .with_columns([pl.col(column).fill_null(0.0) for column in FEATURE_COLUMNS])
            .to_numpy()
        )
        if self._booster is None:
            predicted = np.zeros(matrix.height, dtype=np.float64)
        else:
            raw = np.asarray(
                self._booster.predict(values),  # pyright: ignore[reportUnknownMemberType]
                dtype=np.float64,
            )
            predicted = np.clip(raw, -10.0, 40.0)
        scored = matrix.select("player_key", "fixture_id").with_columns(
            pl.Series("xpts", predicted)
        )
        joined = rows.join(scored, on=["player_key", "fixture_id"], how="left")
        return prediction_frame(
            self.spec,
            view.as_of,
            joined,
            pl.col("xpts").fill_null(0.0),
            pl.lit(None),
            pl.lit(None),
            pl.lit(None),
        )

    def save(self, path: Path) -> None:
        """Write the booster and feature list. Refuses to overwrite."""
        write_identity(path, self.spec)
        (path / "features.json").write_text(
            json.dumps(self._features) + "\n", encoding="utf-8"
        )
        if self._booster is None:
            (path / "unfitted").write_text("1\n", encoding="utf-8")
            return
        self._booster.save_model(str(path / "model.txt"))

    @classmethod
    def load(cls, path: Path, spec: ModelSpec) -> LGBMDirectPredictor:
        """Load an artifact written by ``save``."""
        model = cls(spec)
        raw = (path / "features.json").read_text(encoding="utf-8")
        model._features = json.loads(raw)
        if (path / "unfitted").exists():
            model._booster = None
        else:
            model._booster = lgb.Booster(model_file=str(path / "model.txt"))
        return model


def _params(spec: ModelSpec) -> dict[str, Any]:
    params: dict[str, Any] = {
        "objective": "regression",
        "metric": "mae",
        "learning_rate": 0.05,
        "num_leaves": 31,
        "min_child_samples": 1,
        "feature_fraction": 1.0,
        "bagging_fraction": 1.0,
        "bagging_freq": 0,
        "verbosity": -1,
        "seed": spec.seed,
        "deterministic": True,
        "num_threads": 1,
        "force_col_wise": True,
    }
    for key in _PARAM_KEYS:
        if key in spec.params:
            params[key] = spec.params[key]
    return params
