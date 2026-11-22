"""Registered availability@1 feature from player_status."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.features.registry import get_feature, materialise_feature
from fplt.types import GWKey

_SEASON = "2024-25"
_START = datetime(2024, 8, 1, tzinfo=UTC)


def test_availability_feature_parses_news_and_gate(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    store.write(
        "players",
        pl.DataFrame(
            {
                "season": [_SEASON, _SEASON],
                "player_key": [1, 2],
                "element_id": [1, 2],
                "team_key": [10, 20],
                "position": ["MID", "FWD"],
                "web_name": ["A", "B"],
                "available_at": [_START, _START],
                "source": ["fpl_api", "fpl_api"],
            }
        ),
    )
    store.write(
        "teams",
        pl.DataFrame(
            {
                "season": [_SEASON, _SEASON],
                "team_key": [10, 20],
                "team_id": [1, 2],
                "name": ["Home", "Away"],
                "short_name": ["HOM", "AWY"],
                "available_at": [_START, _START],
                "source": ["fpl_api", "fpl_api"],
            }
        ),
    )
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": [_SEASON],
                "gw": [1],
                "deadline_time": [_START + timedelta(days=1)],
                "finished": [False],
                "data_checked": [False],
                "available_at": [_START],
                "source": ["fpl_api"],
            }
        ),
    )
    store.write(
        "fixtures",
        pl.DataFrame(
            {
                "season": [_SEASON],
                "fixture_id": [1],
                "gw": [1],
                "team_h_key": [10],
                "team_a_key": [20],
                "kickoff_time": [_START + timedelta(days=2)],
                "team_h_score": [None],
                "team_a_score": [None],
                "finished": [False],
                "available_at": [_START],
                "source": ["fpl_api"],
            }
        ),
    )
    store.write(
        "player_status",
        pl.DataFrame(
            {
                "season": [_SEASON, _SEASON],
                "player_key": [1, 2],
                "status": ["a", "a"],
                "news": ["Knock - 75% chance of playing", ""],
                "news_added": [datetime(2024, 8, 10, tzinfo=UTC), None],
                "price": [70, 60],
                "ep_next": [4.0, 3.0],
                "chance_of_playing": [None, 100.0],
                "chance_of_playing_this_round": [None, None],
                "penalties_order": [None, None],
                "available_at": [_START, _START],
                "source": ["fpl_api", "fpl_api"],
            }
        ),
    )
    view = PITData(store).view(GWKey(_SEASON, 1))
    feature = get_feature("availability", 1)
    frame = feature.fn(view, [GWKey(_SEASON, 1)]).sort("player_key")
    feature.schema.validate(frame)
    assert frame["has_news"].to_list() == [1.0, 0.0]
    assert frame["has_news_added"].to_list() == [1.0, 0.0]
    assert frame["news_pct"].to_list() == [75.0, None]
    assert frame["chance_gate"].to_list() == [0.5, 1.0]
    path = materialise_feature(view, "availability", 1, tmp_path)
    assert path.is_file()
