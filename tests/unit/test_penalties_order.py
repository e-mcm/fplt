"""Stage nullable ``penalties_order`` into ``player_status``."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from pandera.errors import SchemaError

from fplt.data.pit import PITData
from fplt.data.staging import stage_bootstrap
from fplt.data.store import DataStore
from fplt.schemas.staged import PlayerStatusSchema
from fplt.types import GWKey

_CAPTURED = datetime(2024, 8, 10, tzinfo=UTC)
_BOOTSTRAP = {
    "teams": [
        {"id": 1, "code": 3, "name": "Arsenal", "short_name": "ARS"},
        {"id": 2, "code": 7, "name": "Spurs", "short_name": "TOT"},
    ],
    "elements": [
        {
            "id": 10,
            "code": 100,
            "team": 1,
            "element_type": 3,
            "web_name": "Saka",
            "status": "a",
            "news": "",
            "now_cost": 100,
            "ep_next": "6.0",
            "chance_of_playing_next_round": 100,
            "penalties_order": 1,
        },
        {
            "id": 11,
            "code": 200,
            "team": 2,
            "element_type": 4,
            "web_name": "Son",
            "status": "a",
            "news": "",
            "now_cost": 95,
            "ep_next": "5.0",
            "chance_of_playing_next_round": None,
            "penalties_order": None,
        },
        {
            "id": 12,
            "code": 300,
            "team": 1,
            "element_type": 3,
            "web_name": "Odegaard",
            "status": "a",
            "news": "",
            "now_cost": 85,
            "ep_next": "4.0",
            "chance_of_playing_next_round": 100,
        },
    ],
    "events": [
        {
            "id": 1,
            "deadline_time": "2024-08-16T17:30:00Z",
            "finished": False,
            "data_checked": False,
        }
    ],
}
_FIXTURES = [
    {
        "id": 50,
        "event": 1,
        "team_h": 1,
        "team_a": 2,
        "kickoff_time": "2024-08-17T14:00:00Z",
        "finished": False,
        "team_h_score": None,
        "team_a_score": None,
    }
]


def test_stage_bootstrap_reads_penalties_order() -> None:
    status = stage_bootstrap(_BOOTSTRAP, _FIXTURES, _CAPTURED, "2024-25")[
        "player_status"
    ].sort("player_key")
    PlayerStatusSchema.validate(status)
    assert status["penalties_order"].to_list() == [1, None, None]


def test_player_status_schema_rejects_zero_order() -> None:
    when = datetime(2024, 8, 10, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "season": ["2024-25"],
            "player_key": [1],
            "status": ["a"],
            "news": [""],
            "price": [50],
            "ep_next": [None],
            "chance_of_playing": [None],
            "penalties_order": [0],
            "available_at": [when],
            "source": ["fpl_api"],
        }
    )
    with pytest.raises(SchemaError):
        PlayerStatusSchema.validate(frame)


def test_pit_latest_keeps_penalties_order(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    early = stage_bootstrap(_BOOTSTRAP, _FIXTURES, _CAPTURED, "2024-25")
    later = {
        **_BOOTSTRAP,
        "elements": [
            {**_BOOTSTRAP["elements"][0], "penalties_order": 2},
            _BOOTSTRAP["elements"][1],
            _BOOTSTRAP["elements"][2],
        ],
    }
    late = stage_bootstrap(
        later, _FIXTURES, datetime(2024, 8, 18, tzinfo=UTC), "2024-25"
    )
    for frames in (early, late):
        for name, frame in frames.items():
            store.write(name, frame)
    view = PITData(store).view(GWKey("2024-25", 1))
    status = (
        view.latest("player_status", by=["player_key"]).collect().sort("player_key")
    )
    assert status["penalties_order"].to_list() == [1, None, None]


def test_store_widen_fills_null_penalties_order(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    when = datetime(2024, 8, 10, tzinfo=UTC)
    store.write(
        "player_status",
        pl.DataFrame(
            {
                "season": ["2024-25"],
                "player_key": [1],
                "status": ["a"],
                "news": [""],
                "price": [50],
                "ep_next": [None],
                "chance_of_playing": [None],
                "available_at": [when],
                "source": ["fpl_api"],
            }
        ),
    )
    got = store.read("player_status")
    assert got["penalties_order"].to_list() == [None]
