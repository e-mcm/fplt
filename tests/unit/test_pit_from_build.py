"""Stage B6 PR2: PITView / PITData over a curated build."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fplt.data.pit import PITData, PITView
from fplt.data.staging import stage_bootstrap
from fplt.data.store import DataStore
from fplt.store.builds import (
    CuratedBuildSource,
    build_from_staged,
    latest_build_id,
    list_build_ids,
    load_build_manifest,
    table_sha_path,
)
from fplt.types import GWKey

_TS = datetime(2024, 8, 10, tzinfo=UTC)
_DEADLINE = datetime(2024, 8, 16, 17, 30, tzinfo=UTC)
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


def _seed_build(data_dir: Path) -> str:
    raw = (
        data_dir
        / "raw"
        / "fpl_api"
        / "season=2024-25"
        / "captured_at=2024-08-10T12-00-00Z"
        / "bootstrap_static.json.gz"
    )
    raw.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw, "wt", encoding="utf-8") as handle:
        json.dump(_BOOTSTRAP, handle)
    store = DataStore(data_dir)
    early = stage_bootstrap(_BOOTSTRAP, _FIXTURES, _TS, "2024-25")
    later_payload = {
        **_BOOTSTRAP,
        "elements": [{**_BOOTSTRAP["elements"][0], "now_cost": 105, "ep_next": "9.0"}],
    }
    late = stage_bootstrap(later_payload, _FIXTURES, _LATE, "2024-25")
    for frames in (early, late):
        for name, frame in frames.items():
            if not frame.is_empty():
                store.write(name, frame)
    return build_from_staged(data_dir, code_sha="abc", created_at=_TS).build_id


def test_pit_data_from_build_hides_post_deadline_rows(tmp_path: Path) -> None:
    build_id = _seed_build(tmp_path)
    pit = PITData.from_build(tmp_path, build_id)
    assert pit.build_id == build_id
    assert pit.deadline(GWKey("2024-25", 1)) == _DEADLINE
    view = pit.view(GWKey("2024-25", 1))
    assert view.build_id == build_id
    status = view.latest("player_status", by=["player_key"]).collect()
    assert status["price"].to_list() == [100]


def test_pit_view_from_build_factory(tmp_path: Path) -> None:
    build_id = _seed_build(tmp_path)
    view = PITView.from_build(
        tmp_path,
        build_id,
        deadline=_DEADLINE,
        as_of=GWKey("2024-25", 1),
    )
    assert view.build_id == build_id
    assert view.cutoff == _DEADLINE
    assert view.table("players").collect().height == 1
    # Tables absent from the build are empty typed frames.
    assert view.table("match_odds").collect().is_empty()


def test_within_seasons_preserves_build_id(tmp_path: Path) -> None:
    build_id = _seed_build(tmp_path)
    view = PITData.from_build(tmp_path, build_id).view(GWKey("2024-25", 1))
    narrowed = view.within_seasons({"2024-25"})
    assert narrowed.build_id == build_id


def test_curated_source_missing_table_file_raises(tmp_path: Path) -> None:
    build_id = _seed_build(tmp_path)
    manifest = load_build_manifest(tmp_path, build_id)
    name, entry = next(iter(manifest.tables.items()))
    table_sha_path(tmp_path, name, entry.sha).unlink()
    source = CuratedBuildSource.load(tmp_path, build_id)
    with pytest.raises(FileNotFoundError):
        source.table(name).collect()


def test_curated_source_unknown_gameweek(tmp_path: Path) -> None:
    build_id = _seed_build(tmp_path)
    pit = PITData.from_build(tmp_path, build_id)
    with pytest.raises(KeyError, match="gw38"):
        pit.deadline(GWKey("2024-25", 38))


def test_latest_build_id_picks_newest(tmp_path: Path) -> None:
    assert latest_build_id(tmp_path) is None
    assert list_build_ids(tmp_path) == []
    first = _seed_build(tmp_path)
    # Second build: change raw so build_id differs.
    raw = (
        tmp_path
        / "raw"
        / "fpl_api"
        / "season=2024-25"
        / "captured_at=2024-08-11T12-00-00Z"
        / "bootstrap_static.json.gz"
    )
    raw.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw, "wt", encoding="utf-8") as handle:
        json.dump(_BOOTSTRAP, handle)
    second = build_from_staged(
        tmp_path,
        code_sha="abc",
        created_at=datetime(2024, 8, 11, tzinfo=UTC),
    ).build_id
    assert first != second
    assert set(list_build_ids(tmp_path)) == {first, second}
    assert latest_build_id(tmp_path) == second
