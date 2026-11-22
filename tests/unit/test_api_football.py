"""API-Football injuries stage with PIT stamps and id-map gaps."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from fplt.data.api_football import (
    available_at_from_path,
    categorize_injury,
    ingest_api_football,
    injuries_raw_path,
    load_id_maps,
    rows_from_injuries_payload,
    season_label_from_api_year,
    stage_player_injuries,
)
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.types import GWKey

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "api_football"
_PAYLOAD = _FIXTURES / "injuries.json"
_MAPS = _FIXTURES / "id_maps.yaml"
_CAPTURE = datetime(2024, 8, 15, 12, 0, tzinfo=UTC)
_DEADLINE = datetime(2024, 8, 16, 10, 0, tzinfo=UTC)


def test_categorize_injury_marks_suspensions_out() -> None:
    assert categorize_injury("Suspension", "Red Card") == "out"
    assert categorize_injury("Missing Fixture", "Knee Injury") == "out"
    assert categorize_injury("Injury", "Knee Injury") == "doubtful"


def test_load_id_maps_reads_yaml() -> None:
    players, teams = load_id_maps(_MAPS)
    assert players == {276: 100, 277: 200}
    assert teams == {34: 4, 40: 14, 42: 3}


def test_season_label_helpers() -> None:
    assert season_label_from_api_year(2024) == "2024-25"


def test_rows_leave_unmapped_player_key_null() -> None:
    payload = json.loads(_PAYLOAD.read_text(encoding="utf-8"))
    players, teams = load_id_maps(_MAPS)
    rows, unmapped_players, unmapped_teams = rows_from_injuries_payload(
        payload,
        season="2024-25",
        available_at=_CAPTURE,
        player_keys=players,
        team_keys=teams,
    )
    assert len(rows) == 3
    assert unmapped_players == {999001}
    assert unmapped_teams == set()
    by_api = {int(row["api_player_id"]): row for row in rows}
    assert by_api[276]["player_key"] == 100
    assert by_api[276]["category"] == "doubtful"
    assert by_api[999001]["player_key"] is None
    assert by_api[277]["category"] == "out"


def test_stage_and_pit_hides_later_captures(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    _write_gameweek(store)
    payload = json.loads(_PAYLOAD.read_text(encoding="utf-8"))
    players, teams = load_id_maps(_MAPS)
    stage_player_injuries(store, payload, "2024-25", _CAPTURE, players, teams)
    late = datetime(2024, 8, 17, 12, 0, tzinfo=UTC)
    late_payload = {
        "response": [
            {
                "player": {"id": 276, "name": "N. Pope"},
                "team": {"id": 34, "name": "Newcastle"},
                "fixture": {"id": 1208099},
                "type": "Injury",
                "reason": "Hamstring",
            }
        ]
    }
    stage_player_injuries(store, late_payload, "2024-25", late, players, teams)
    view = PITData(store).view(GWKey("2024-25", 1))
    visible = view.table("player_injuries").collect()
    assert visible.height == 3
    assert visible["available_at"].max() == _CAPTURE
    assert store.read("player_injuries").height == 4


def test_ingest_api_football_stages_raw_files(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    path = injuries_raw_path(raw, "2024-25", _CAPTURE)
    path.parent.mkdir(parents=True)
    payload = json.loads(_PAYLOAD.read_text(encoding="utf-8"))
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    players, teams = load_id_maps(_MAPS)
    store = DataStore(tmp_path / "data")
    assert ingest_api_football(store, raw, players, teams) == 1
    staged = store.read("player_injuries")
    assert staged.height == 3
    assert staged.filter(pl.col("player_key").is_null()).height == 1
    assert staged["available_at"][0] == _CAPTURE
    assert available_at_from_path(path) == _CAPTURE
    assert ingest_api_football(store, raw, players, teams) == 1
    assert store.read("player_injuries").height == 3


def test_ingest_skips_other_seasons(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    path = injuries_raw_path(raw, "2023-24", _CAPTURE)
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump({"response": []}, handle)
    store = DataStore(tmp_path / "data")
    players, teams = load_id_maps(_MAPS)
    assert ingest_api_football(store, raw, players, teams, seasons=["2024-25"]) == 0


def _write_gameweek(store: DataStore) -> None:
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": ["2024-25"],
                "gw": [1],
                "deadline_time": [_DEADLINE],
                "finished": [False],
                "data_checked": [False],
                "available_at": [_CAPTURE],
                "source": ["fpl_api"],
            }
        ).with_columns(
            pl.col("deadline_time").cast(pl.Datetime("us", "UTC")),
            pl.col("available_at").cast(pl.Datetime("us", "UTC")),
        ),
    )
