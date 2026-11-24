"""Direct LightGBM points model. Hyperparameters live in the YAML spec."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

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
_PROB_HEADS = ("p_haul", "p_60", "p_played")
_HEAD_FILES = {
    "p_haul": "haul.txt",
    "p_60": "minutes60.txt",
    "p_played": "played.txt",
}


@register_model("lgbm_direct")
class LGBMDirectPredictor:
    """Regression on PIT features for total points per player-fixture.

    ``fit`` trains only on appearances visible in the view. ``predict`` joins
    the features named in the spec through ``build_feature_matrix`` so the
    walk-forward slices match every other ``PointsPredictor``.

    When ``params.emit_probs`` is true, also trains binary heads for
    ``p_haul`` (points ≥ 10), ``p_60`` (minutes ≥ 60), and ``p_played``
    (minutes > 0). Single-class training labels fall back to a constant prior
    so tiny PIT windows stay well-defined.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._booster: lgb.Booster | None = None
        self._features = list(spec.features) or list(_DEFAULT_FEATURES)
        self._emit_probs = bool(spec.params.get("emit_probs", False))
        self._prob_boosters: dict[str, lgb.Booster | None] = {
            name: None for name in _PROB_HEADS
        }
        self._prob_priors: dict[str, float] = {name: 0.0 for name in _PROB_HEADS}

    def fit(self, view: PITView) -> None:
        """Train on visible appearances. A tiny sample leaves the model unfitted.

        When ``spec.train_window.time_decay`` is set, rows are weighted by
        ``exp(-λ × days_before_cutoff)`` via ``calendar_sample_weights``.
        """
        train = training_frame(view)
        if train.height < 2:
            self._booster = None
            for name in _PROB_HEADS:
                self._prob_boosters[name] = None
                self._prob_priors[name] = 0.0
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
        if not self._emit_probs:
            for name in _PROB_HEADS:
                self._prob_boosters[name] = None
                self._prob_priors[name] = 0.0
            return
        minutes = train["minutes"].to_numpy().astype(float)
        heads = {
            "p_haul": (label >= 10.0).astype(float),
            "p_60": (minutes >= 60.0).astype(float),
            "p_played": (minutes > 0.0).astype(float),
        }
        for name, y in heads.items():
            fitted, prior = _fit_binary_head(
                matrix,
                y,
                weights,
                self.spec,
            )
            self._prob_boosters[name] = fitted
            self._prob_priors[name] = prior

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
        if self._emit_probs:
            for name in _PROB_HEADS:
                scored = scored.with_columns(
                    pl.Series(
                        name,
                        _predict_binary(
                            self._prob_boosters[name],
                            self._prob_priors[name],
                            values,
                            matrix.height,
                        ),
                    )
                )
        else:
            for name in _PROB_HEADS:
                scored = scored.with_columns(pl.lit(None).cast(pl.Float64).alias(name))
        joined = rows.join(scored, on=["player_key", "fixture_id"], how="left")
        return prediction_frame(
            self.spec,
            view.as_of,
            joined,
            pl.col("xpts").fill_null(0.0),
            pl.lit(None),
            pl.col("p_haul"),
            pl.col("p_60"),
            p_played=pl.col("p_played"),
        )

    def save(self, path: Path) -> None:
        """Write the booster and feature list. Refuses to overwrite."""
        write_identity(path, self.spec)
        (path / "features.json").write_text(
            json.dumps(self._features) + "\n", encoding="utf-8"
        )
        meta = {
            "emit_probs": self._emit_probs,
            "prob_priors": self._prob_priors,
        }
        (path / "lgbm_meta.json").write_text(
            json.dumps(meta, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )
        if self._booster is None:
            (path / "unfitted").write_text("1\n", encoding="utf-8")
        else:
            self._booster.save_model(str(path / "model.txt"))
        if self._emit_probs:
            for name, filename in _HEAD_FILES.items():
                booster = self._prob_boosters[name]
                if booster is None:
                    (path / f"{name}.prior_only").write_text("1\n", encoding="utf-8")
                else:
                    booster.save_model(str(path / filename))

    @classmethod
    def load(cls, path: Path, spec: ModelSpec) -> LGBMDirectPredictor:
        """Load an artifact written by ``save``."""
        model = cls(spec)
        raw = (path / "features.json").read_text(encoding="utf-8")
        model._features = json.loads(raw)
        meta_path = path / "lgbm_meta.json"
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            model._emit_probs = bool(meta.get("emit_probs", model._emit_probs))
            priors_raw = meta.get("prob_priors", {})
            if isinstance(priors_raw, dict):
                priors = cast(dict[str, Any], priors_raw)
                for name in _PROB_HEADS:
                    if name in priors:
                        model._prob_priors[name] = float(priors[name])
        if (path / "unfitted").exists():
            model._booster = None
        else:
            model._booster = lgb.Booster(model_file=str(path / "model.txt"))
        if model._emit_probs:
            for name, filename in _HEAD_FILES.items():
                if (path / f"{name}.prior_only").exists():
                    model._prob_boosters[name] = None
                elif (path / filename).exists():
                    model._prob_boosters[name] = lgb.Booster(
                        model_file=str(path / filename)
                    )
                else:
                    model._prob_boosters[name] = None
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


def _binary_params(spec: ModelSpec) -> dict[str, Any]:
    params = _params(spec)
    params["objective"] = "binary"
    params["metric"] = "binary_logloss"
    return params


def _fit_binary_head(
    matrix: np.ndarray,
    labels: np.ndarray,
    weights: np.ndarray,
    spec: ModelSpec,
) -> tuple[lgb.Booster | None, float]:
    """Train one binary head, or fall back to the empirical prior.

    Args:
        matrix: Feature matrix for the training rows.
        labels: Binary 0/1 labels.
        weights: Sample weights aligned with ``matrix``.
        spec: ModelSpec supplying shared LightGBM hyperparameters.

    Returns:
        ``(booster_or_none, prior)``. When both classes are present the booster
        is fitted; otherwise the booster is ``None`` and ``prior`` is the
        weighted mean of ``labels``.
    """
    prior = float(np.average(labels, weights=weights)) if labels.size else 0.0
    if labels.size < 2 or len(np.unique(labels)) < 2:
        return None, prior
    dataset = lgb.Dataset(
        matrix,
        label=labels,
        weight=weights,
        feature_name=list(FEATURE_COLUMNS),
    )
    booster = lgb.train(  # pyright: ignore[reportUnknownMemberType]
        _binary_params(spec),
        dataset,
        num_boost_round=int(spec.params.get("num_boost_round", 100)),
    )
    return booster, prior


def _predict_binary(
    booster: lgb.Booster | None,
    prior: float,
    values: np.ndarray,
    n_rows: int,
) -> np.ndarray:
    """Score a binary head; clip to [0, 1]."""
    if booster is None:
        return np.full(n_rows, prior, dtype=np.float64)
    raw = np.asarray(
        booster.predict(values),  # pyright: ignore[reportUnknownMemberType]
        dtype=np.float64,
    )
    return np.clip(raw, 0.0, 1.0)
