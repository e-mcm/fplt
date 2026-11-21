"""Calendar sample weights for TrainWindow.time_decay."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

import fplt.models.direct  # noqa: F401 — register lgbm_direct
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.features.form import FEATURE_COLUMNS, training_frame
from fplt.models.direct import LGBMDirectPredictor
from fplt.models.spec import ModelSpec, TrainWindow, load_model_specs
from fplt.models.time_decay import calendar_sample_weights
from fplt.types import GWKey
from tests.builders import build_league

_AS_OF = datetime(2024, 8, 23, 17, tzinfo=UTC)
_OLD = datetime(2024, 5, 1, 14, tzinfo=UTC)
_RECENT = datetime(2024, 8, 20, 14, tzinfo=UTC)


def test_null_decay_returns_equal_weights() -> None:
    times = np.array([_naive(_OLD), _naive(_RECENT)], dtype="datetime64[us]")
    weights = calendar_sample_weights(times, _AS_OF, None)
    assert weights.tolist() == pytest.approx([1.0, 1.0])


def test_older_rows_get_lower_weight() -> None:
    times = np.array(
        [_naive(_OLD), _naive(_RECENT)],
        dtype="datetime64[us]",
    )
    weights = calendar_sample_weights(times, _AS_OF, 0.0065)
    assert weights[0] < weights[1]
    assert weights[1] == pytest.approx(1.0, abs=0.05)
    days_old = (_AS_OF.replace(tzinfo=None) - _OLD.replace(tzinfo=None)).total_seconds()
    days_old /= 86400.0
    assert weights[0] == pytest.approx(np.exp(-0.0065 * days_old))


def test_decay_is_relative_to_as_of_cutoff_not_latest_event() -> None:
    """PIT-safe: an older as_of must down-weight the same kickoff more."""
    times = np.array([_naive(_RECENT)], dtype="datetime64[us]")
    early = _RECENT + timedelta(days=1)
    late = _RECENT + timedelta(days=100)
    early_w = calendar_sample_weights(times, early, 0.01)
    late_w = calendar_sample_weights(times, late, 0.01)
    assert early_w[0] > late_w[0]


def test_negative_decay_raises() -> None:
    with pytest.raises(ValueError, match="time_decay"):
        calendar_sample_weights(
            np.array([_naive(_RECENT)], dtype="datetime64[us]"),
            _AS_OF,
            -0.1,
        )


def test_training_frame_exposes_kickoff_for_weights(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    view = PITData(store).view(GWKey("2024-25", 2))
    train = training_frame(view)
    assert "kickoff_time" in train.columns
    assert set(FEATURE_COLUMNS).issubset(train.columns)
    weights = calendar_sample_weights(
        train["kickoff_time"].to_numpy(),
        view.cutoff,
        0.0065,
    )
    assert weights.shape == (train.height,)
    assert np.all(weights > 0)


def test_lgbm_fit_accepts_time_decay(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path / "league"))
    base = load_model_specs()["lgbm_direct_v1"]
    spec = base.model_copy(
        update={"train_window": TrainWindow(seasons_back=4, time_decay=0.0065)}
    )
    view = PITData(store).view(GWKey("2024-25", 2))
    model = LGBMDirectPredictor(spec)
    model.fit(view)
    assert model._booster is not None
    predicted = model.predict(view, [GWKey("2024-25", 2)])
    assert predicted.height == 2
    assert predicted["xpts"].null_count() == 0


def test_default_lgbm_spec_leaves_time_decay_off() -> None:
    spec = load_model_specs()["lgbm_direct_v1"]
    assert spec.train_window.time_decay is None
    assert ModelSpec(name="x", kind="lgbm_direct").train_window.time_decay is None


def _naive(value: datetime) -> datetime:
    return value.replace(tzinfo=None)
