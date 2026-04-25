"""Smoke tests for ByPositionPredictor and ComponentPredictor."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fplt.data import load_gameweeks, load_teams
from fplt.features import build_features
from fplt.predict import ByPositionPredictor, ComponentPredictor, LGBMPredictor
from fplt.predict.component import COMPONENTS

DATA_DIR = Path(__file__).parents[1] / "data"
COMPONENT_COLS = [cfg.column for cfg in COMPONENTS]


@pytest.fixture(scope="module")
def split() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    gw_train = load_gameweeks("2022-23", DATA_DIR)
    gw_test = load_gameweeks("2023-24", DATA_DIR)
    df_train = build_features(gw_train, load_teams("2022-23", DATA_DIR))
    df_test = build_features(gw_test, load_teams("2023-24", DATA_DIR))
    df_train = df_train[df_train["GW"] > 1].copy()
    df_test = df_test[df_test["GW"] > 1].copy()
    return df_train, df_test


# --- ByPositionPredictor ---


@pytest.fixture(scope="module")
def by_pos_model(split: tuple) -> ByPositionPredictor:
    df_train, df_test = split
    return ByPositionPredictor(num_rounds=50).fit(
        df_train,
        df_train["total_points"],
        X_val=df_test,
        y_val=df_test["total_points"],
    )


def test_by_position_is_fitted(by_pos_model: ByPositionPredictor) -> None:
    assert by_pos_model.is_fitted


def test_by_position_predict_shape(
    by_pos_model: ByPositionPredictor, split: tuple
) -> None:
    _, df_test = split
    preds = by_pos_model.predict(df_test)
    assert preds.shape == (len(df_test),)


def test_by_position_mae(by_pos_model: ByPositionPredictor, split: tuple) -> None:
    _, df_test = split
    preds = by_pos_model.predict(df_test)
    mae = float(np.mean(np.abs(preds - df_test["total_points"].to_numpy())))
    assert mae < 4.0


def test_by_position_importance_keys(
    by_pos_model: ByPositionPredictor,
) -> None:
    fi = by_pos_model.feature_importance()
    assert set(fi.keys()) == {"GK", "DEF", "MID", "FWD"}


# --- ComponentPredictor ---


@pytest.fixture(scope="module")
def component_model(split: tuple) -> ComponentPredictor:
    df_train, df_test = split
    predictors = {cfg.name: LGBMPredictor(num_rounds=50) for cfg in COMPONENTS}
    return ComponentPredictor(predictors=predictors).fit(
        df_train,
        df_train[COMPONENT_COLS],
        X_val=df_test,
        y_val=df_test[COMPONENT_COLS],
    )


def test_component_is_fitted(component_model: ComponentPredictor) -> None:
    assert component_model.is_fitted


def test_component_predict_shape(
    component_model: ComponentPredictor, split: tuple
) -> None:
    _, df_test = split
    preds = component_model.predict(df_test)
    assert preds.shape == (len(df_test),)


def test_component_predict_components_cols(
    component_model: ComponentPredictor, split: tuple
) -> None:
    _, df_test = split
    comp = component_model.predict_components(df_test)
    for cfg in COMPONENTS:
        name = cfg.name
        assert name in comp.columns


def test_component_predictions_clipped(
    component_model: ComponentPredictor, split: tuple
) -> None:
    _, df_test = split
    comp = component_model.predict_components(df_test)
    assert (comp["minutes"] >= 0).all() and (comp["minutes"] <= 90).all()
    assert (comp["clean_sheets"] >= 0).all()
    assert (comp["clean_sheets"] <= 1).all()


def test_component_mae(component_model: ComponentPredictor, split: tuple) -> None:
    _, df_test = split
    preds = component_model.predict(df_test)
    mae = float(np.mean(np.abs(preds - df_test["total_points"].to_numpy())))
    assert mae < 5.0  # slightly looser — rules conversion adds noise


def test_component_importance_keys(
    component_model: ComponentPredictor,
) -> None:
    fi = component_model.feature_importance()
    assert "minutes" in fi
    assert "goals_scored" in fi
