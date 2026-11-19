"""Staged snapshots hide rows that were not available at the deadline."""

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

from fplt.data.pit import PITData
from fplt.data.raw import ingest_raw
from fplt.data.staging import stage_bootstrap
from fplt.data.store import DataStore
from fplt.types import GWKey
from tests.builders import build_league

_EARLY = datetime(2024, 8, 10, tzinfo=UTC)
_LATE = datetime(2024, 8, 18, tzinfo=UTC)
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
        }
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


def test_view_hides_a_later_price_change(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    early = stage_bootstrap(_BOOTSTRAP, _FIXTURES, _EARLY, "2024-25")
    later_payload = {
        **_BOOTSTRAP,
        "elements": [{**_BOOTSTRAP["elements"][0], "now_cost": 105, "ep_next": "9.0"}],
    }
    late = stage_bootstrap(later_payload, _FIXTURES, _LATE, "2024-25")
    for frames in (early, late):
        for name, frame in frames.items():
            store.write(name, frame)
    view = PITData(store).view(GWKey("2024-25", 1))
    status = view.latest("player_status", by=["player_key"]).collect()
    assert status["price"].to_list() == [100]
    assert status["player_key"].to_list() == [100]
    assert view.cutoff < _LATE


def test_future_stats_stay_hidden(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    view = PITData(store).view(GWKey("2024-25", 2))
    visible = view.table("player_fixture_stats").collect()
    assert visible["total_points"].sort().to_list() == [2, 8]
    assert 40 not in visible["total_points"].to_list()


def test_ingest_raw_writes_staged_parquet(tmp_path: Path) -> None:
    captured = (
        tmp_path
        / "raw"
        / "fpl_api"
        / "season=2024-25"
        / "captured_at=2024-08-10T00-00-00Z"
    )
    captured.mkdir(parents=True)
    for name, payload in (("bootstrap_static", _BOOTSTRAP), ("fixtures", _FIXTURES)):
        with gzip.open(captured / f"{name}.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
    store = DataStore(tmp_path / "store")
    count = ingest_raw(store, tmp_path / "raw")
    assert count == 1
    players = store.read("players")
    assert players["player_key"].to_list() == [100]
    assert players["source"].to_list() == ["fpl_api"]
    assert store.read("fixtures")["team_h_score"].to_list() == [None]
