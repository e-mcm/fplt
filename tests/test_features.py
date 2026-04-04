from pathlib import Path

import pandas as pd
import pytest

from fplt.data import load_gameweeks, load_teams
from fplt.features import WINDOWS, build_features

DATA_DIR = Path(__file__).parents[1] / "data"
SEASON = "2024-25"


@pytest.fixture(scope="module")
def gw() -> pd.DataFrame:
    return load_gameweeks(SEASON, DATA_DIR)


@pytest.fixture(scope="module")
def teams() -> pd.DataFrame:
    return load_teams(SEASON, DATA_DIR)


@pytest.fixture(scope="module")
def features(gw: pd.DataFrame, teams: pd.DataFrame) -> pd.DataFrame:
    return build_features(gw, teams)


def test_rolling_points_cols_present(features: pd.DataFrame) -> None:
    for w in WINDOWS:
        assert f"total_points_rolling_{w}" in features.columns


def test_rolling_minutes_cols_present(features: pd.DataFrame) -> None:
    for w in WINDOWS:
        assert f"minutes_rolling_{w}" in features.columns


def test_minutes_std_present(features: pd.DataFrame) -> None:
    assert "minutes_std_5" in features.columns


def test_opponent_strength_cols_present(features: pd.DataFrame) -> None:
    assert "opp_strength_overall" in features.columns
    assert "opp_strength_attack" in features.columns
    assert "opp_strength_defence" in features.columns


def test_position_enc_values(features: pd.DataFrame) -> None:
    assert set(features["position_enc"].unique()).issubset({0, 1, 2, 3, -1})


def test_no_leakage_gw1(features: pd.DataFrame) -> None:
    """Rolling stats for GW1 must be NaN — no prior data exists."""
    gw1 = features[features["GW"] == 1]
    assert gw1["total_points_rolling_3"].isna().all()


def test_rolling_fills_after_gw1(features: pd.DataFrame) -> None:
    gw2_plus = features[features["GW"] > 1]
    assert gw2_plus["total_points_rolling_3"].notna().any()


def test_total_points_retained(features: pd.DataFrame) -> None:
    assert "total_points" in features.columns


def test_row_count_unchanged(
    gw: pd.DataFrame, features: pd.DataFrame
) -> None:
    assert len(features) == len(gw)
