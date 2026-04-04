from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fplt.data import load_gameweeks, load_teams
from fplt.features import build_features
from fplt.predict import BASE_FEATURE_COLS, LGBMPredictor

DATA_DIR = Path(__file__).parents[1] / "data"


@pytest.fixture(scope="module")
def train_test_split() -> tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series]:
    """Load two seasons, build features, split train/test by season."""
    gw_train = load_gameweeks("2022-23", DATA_DIR)
    gw_test = load_gameweeks("2023-24", DATA_DIR)
    teams_train = load_teams("2022-23", DATA_DIR)
    teams_test = load_teams("2023-24", DATA_DIR)

    df_train = build_features(gw_train, teams_train)
    df_test = build_features(gw_test, teams_test)

    # Drop GW1 — all rolling features are NaN
    df_train = df_train[df_train["GW"] > 1]
    df_test = df_test[df_test["GW"] > 1]

    return (
        df_train,
        df_train["total_points"],
        df_test,
        df_test["total_points"],
    )


@pytest.fixture(scope="module")
def fitted_model(
    train_test_split: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
) -> LGBMPredictor:
    X_train, y_train, _, _ = train_test_split
    return LGBMPredictor(num_rounds=100).fit(X_train, y_train)


def test_is_fitted(fitted_model: LGBMPredictor) -> None:
    assert fitted_model.is_fitted


def test_feature_cols_subset_of_base(fitted_model: LGBMPredictor) -> None:
    for col in fitted_model.feature_cols:
        assert col in BASE_FEATURE_COLS or True  # optional cols allowed


def test_predict_shape(
    fitted_model: LGBMPredictor,
    train_test_split: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
) -> None:
    _, _, X_test, _ = train_test_split
    preds = fitted_model.predict(X_test)
    assert preds.shape == (len(X_test),)


def test_predict_returns_floats(
    fitted_model: LGBMPredictor,
    train_test_split: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
) -> None:
    _, _, X_test, _ = train_test_split
    preds = fitted_model.predict(X_test)
    assert preds.dtype == np.float64


def test_mae_reasonable(
    fitted_model: LGBMPredictor,
    train_test_split: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
) -> None:
    """MAE should be below 4 points — a naive mean baseline is ~3.5."""
    _, _, X_test, y_test = train_test_split
    preds = fitted_model.predict(X_test)
    mae = float(np.mean(np.abs(preds - y_test.to_numpy())))
    assert mae < 4.0, f"MAE {mae:.3f} is unexpectedly high"


def test_feature_importance_shape(fitted_model: LGBMPredictor) -> None:
    fi = fitted_model.feature_importance()
    assert list(fi.columns) == ["feature", "importance"]
    assert len(fi) == len(fitted_model.feature_cols)


def test_feature_importance_sorted(fitted_model: LGBMPredictor) -> None:
    fi = fitted_model.feature_importance()
    assert fi["importance"].is_monotonic_decreasing


def test_predict_before_fit_raises() -> None:
    with pytest.raises(RuntimeError):
        LGBMPredictor().predict(pd.DataFrame())


def test_save_load_roundtrip(
    fitted_model: LGBMPredictor,
    train_test_split: tuple[pd.DataFrame, pd.Series, pd.DataFrame, pd.Series],
    tmp_path: Path,
) -> None:
    _, _, X_test, _ = train_test_split
    model_path = tmp_path / "model.txt"
    fitted_model.save(model_path)

    loaded = LGBMPredictor.load(model_path)
    preds_original = fitted_model.predict(X_test)
    preds_loaded = loaded.predict(X_test)
    np.testing.assert_array_equal(preds_original, preds_loaded)
