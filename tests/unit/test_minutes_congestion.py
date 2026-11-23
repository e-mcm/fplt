"""All-comp rest / congestion helpers for minutes v2."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.components.minutes.congestion import (
    competition_team_kickoffs,
    team_schedule_features,
)
from fplt.types import GWKey

_SEASON = "2024-25"
_START = datetime(2024, 8, 1, tzinfo=UTC)


def test_team_schedule_features_pl_only_matches_week_rest() -> None:
    fixtures = pl.DataFrame(
        {
            "season": [_SEASON, _SEASON],
            "fixture_id": [1, 2],
            "team_h_key": [10, 10],
            "team_a_key": [20, 30],
            "kickoff_time": [
                _START,
                _START + timedelta(days=7),
            ],
        }
    )
    out = team_schedule_features(fixtures)
    row = out.filter(
        (pl.col("fixture_id") == 2) & (pl.col("team_key") == 10)
    ).to_dicts()[0]
    assert row["rest_days"] == pytest.approx(1.0)
    assert row["games_last_7d"] == pytest.approx(0.0)


def test_midweek_europe_shortens_rest_and_counts_congestion() -> None:
    pl_fixtures = pl.DataFrame(
        {
            "season": [_SEASON, _SEASON],
            "fixture_id": [1, 2],
            "team_h_key": [10, 10],
            "team_a_key": [20, 30],
            "kickoff_time": [
                _START,
                _START + timedelta(days=7),
            ],
        }
    )
    europe = pl.DataFrame(
        {
            "season": [_SEASON],
            "team_key": [10],
            "kickoff_time": [_START + timedelta(days=3)],
        }
    )
    out = team_schedule_features(pl_fixtures, europe)
    row = out.filter(
        (pl.col("fixture_id") == 2) & (pl.col("team_key") == 10)
    ).to_dicts()[0]
    assert row["rest_days"] == pytest.approx(4.0 / 7.0)
    assert row["games_last_7d"] == pytest.approx(1.0)


def test_competition_kickoffs_expand_mapped_sides(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": [_SEASON],
                "gw": [2],
                "deadline_time": [_START + timedelta(days=10)],
                "finished": [False],
                "data_checked": [False],
                "available_at": [_START],
                "source": ["test"],
            }
        ),
    )
    store.write(
        "competition_fixtures",
        pl.DataFrame(
            {
                "season": [_SEASON],
                "api_fixture_id": [9001],
                "competition": ["ucl"],
                "league_id": [2],
                "team_h_key": [14],
                "team_a_key": [None],
                "api_team_h_id": [40],
                "api_team_a_id": [541],
                "kickoff_time": [_START + timedelta(days=3)],
                "available_at": [_START],
                "source": ["api_football"],
            }
        ),
    )
    view = PITData(store).view(GWKey(_SEASON, 2))
    kickoffs = competition_team_kickoffs(view)
    assert kickoffs.height == 1
    assert kickoffs["team_key"].to_list() == [14]
