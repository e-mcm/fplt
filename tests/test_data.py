from pathlib import Path

import pandas as pd
import pytest

from fplt.data import (
    ALL_SEASONS,
    CORE_GW_COLS,
    load_all_seasons,
    load_fixtures,
    load_gameweeks,
    load_teams,
)

DATA_DIR = Path(__file__).parents[1] / "data"


@pytest.mark.parametrize("season", ["2016-17", "2020-21", "2024-25"])
def test_load_gameweeks_core_cols_present(season: str) -> None:
    df = load_gameweeks(season, DATA_DIR)
    for col in CORE_GW_COLS:
        assert col in df.columns, f"Missing core column '{col}' in {season}"


@pytest.mark.parametrize("season", ["2016-17", "2020-21", "2024-25"])
def test_load_gameweeks_no_empty(season: str) -> None:
    df = load_gameweeks(season, DATA_DIR)
    assert len(df) > 0


def test_load_gameweeks_season_col() -> None:
    df = load_gameweeks("2024-25", DATA_DIR)
    assert (df["season"] == "2024-25").all()


def test_load_gameweeks_kickoff_datetime() -> None:
    df = load_gameweeks("2024-25", DATA_DIR)
    assert pd.api.types.is_datetime64_any_dtype(df["kickoff_time"])


def test_load_fixtures() -> None:
    df = load_fixtures("2024-25", DATA_DIR)
    assert len(df) > 0
    assert "team_h" in df.columns
    assert "team_a_difficulty" in df.columns


def test_load_teams() -> None:
    df = load_teams("2024-25", DATA_DIR)
    assert len(df) == 20
    assert "short_name" in df.columns
    assert "strength_attack_home" in df.columns


def test_load_all_seasons() -> None:
    df = load_all_seasons(["2023-24", "2024-25"], DATA_DIR)
    assert df["season"].nunique() == 2
    assert df["season"].isin(["2023-24", "2024-25"]).all()


def test_load_all_seasons_sorted() -> None:
    df = load_all_seasons(["2023-24", "2024-25"], DATA_DIR)
    assert list(df["season"].unique()) == ["2023-24", "2024-25"]


def test_load_all_seasons_skips_missing(tmp_path: Path) -> None:
    # Should not raise even if a season folder is absent
    df = load_all_seasons(["2099-00"], tmp_path)
    assert len(df) == 0
