"""Understat xG is staged with a clock, and later matches stay hidden."""

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.data.understat import (
    available_at_from_match,
    ingest_understat,
    load_id_maps,
    match_raw_path,
    normalize_match,
    season_label_from_understat_year,
    stage_understat,
    understat_year_from_season,
)
from fplt.features.registry import get_feature, materialise_feature
from fplt.types import GWKey

_WHEN = datetime(2024, 8, 1, tzinfo=UTC)
_EARLY = datetime(2024, 8, 10, tzinfo=UTC)
_LATE = datetime(2024, 8, 20, tzinfo=UTC)
_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures"
_FIXTURE = _FIXTURES / "understat_match.json"
_AJAX = _FIXTURES / "understat" / "match_ajax.json"
_MAPS = _FIXTURES / "understat" / "id_maps.yaml"
_PLAYERS = {501: 100, 502: 200}
_TEAMS = {10: 3, 20: 7}


def test_visible_matches_set_the_shares(tmp_path: Path) -> None:
    store = _store(tmp_path)
    matches = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    staged = stage_understat(store, matches, "2024-25", _EARLY, _PLAYERS, _TEAMS)
    assert staged.filter(pl.col("player_key").is_null()).height == 1
    stage_understat(store, [_later()], "2024-25", _LATE, _PLAYERS, _TEAMS)
    early = _shares(store, 1)
    assert early["player_key"].sort().to_list() == [100, 200]
    first = early.filter(pl.col("player_key") == 100).row(0, named=True)
    assert first["xg_per_90"] == pytest.approx(0.5)
    assert first["xa_per_90"] == pytest.approx(0.1)
    assert first["xg_share"] == pytest.approx(1.0)
    assert first["xa_share"] == pytest.approx(0.1 / 0.3)
    second = early.filter(pl.col("player_key") == 200).row(0, named=True)
    assert second["xa_per_90"] == pytest.approx(0.4)
    assert second["xg_share"] == pytest.approx(0.0)
    later = _shares(store, 2).filter(pl.col("player_key") == 100).row(0, named=True)
    assert later["xg_share"] == pytest.approx(0.5)
    assert later["xg_per_90"] == pytest.approx(0.25)


def test_an_unknown_team_is_rejected(tmp_path: Path) -> None:
    matches = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    with pytest.raises(KeyError, match="team 10"):
        stage_understat(DataStore(tmp_path), matches, "2024-25", _EARLY, _PLAYERS, {})


def test_materialise_keeps_the_first_file(tmp_path: Path) -> None:
    store = _store(tmp_path)
    matches = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    stage_understat(store, matches, "2024-25", _EARLY, _PLAYERS, _TEAMS)
    view = PITData(store).view(GWKey("2024-25", 1))
    path = materialise_feature(view, "understat_share", 1, tmp_path)
    assert path.name == "part.parquet"
    assert "v=1" in path.parts
    with pytest.raises(FileExistsError):
        materialise_feature(view, "understat_share", 1, tmp_path)
    with pytest.raises(KeyError, match="not registered"):
        get_feature("understat_share", 2)


def test_normalize_match_reads_get_match_data() -> None:
    payload = json.loads(_AJAX.read_text(encoding="utf-8"))
    match = normalize_match(42, payload)
    assert match["id"] == 42
    assert match["team_h"] == 83
    assert match["team_a"] == 87
    assert match["h"][0] == {"id": 501, "xg": 0.5, "xa": 0.1, "time": 90}
    assert match["a"][0]["id"] == 999


def test_available_at_is_kickoff_plus_three_hours() -> None:
    payload = json.loads(_AJAX.read_text(encoding="utf-8"))
    assert available_at_from_match(payload) == datetime(2024, 8, 10, 18, 0, tzinfo=UTC)


def test_season_label_helpers() -> None:
    assert season_label_from_understat_year(2024) == "2024-25"
    assert understat_year_from_season("2024-25") == "2024"


def test_load_id_maps_reads_yaml() -> None:
    players, teams = load_id_maps(_MAPS)
    assert players == {501: 100, 502: 200}
    assert teams == {83: 3, 87: 14}


def test_ingest_understat_stages_raw_match_files(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    path = match_raw_path(raw, "2024-25", 42)
    path.parent.mkdir(parents=True)
    payload = json.loads(_AJAX.read_text(encoding="utf-8"))
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle)
    players, teams = load_id_maps(_MAPS)
    store = DataStore(tmp_path / "data")
    assert ingest_understat(store, raw, players, teams) == 1
    staged = store.read("understat_player_matches")
    assert staged.height == 3
    assert staged.filter(pl.col("player_key").is_null()).height == 1
    assert staged["available_at"][0] == datetime(2024, 8, 10, 18, 0, tzinfo=UTC)
    assert ingest_understat(store, raw, players, teams) == 1
    assert store.read("understat_player_matches").height == 3


def test_ingest_skips_other_seasons(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    path = match_raw_path(raw, "2023-24", 42)
    path.parent.mkdir(parents=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(json.loads(_AJAX.read_text(encoding="utf-8")), handle)
    players, teams = load_id_maps(_MAPS)
    count = ingest_understat(
        DataStore(tmp_path), raw, players, teams, seasons=["2024-25"]
    )
    assert count == 0


def _shares(store: DataStore, gameweek: int):
    from fplt.features.understat import understat_share

    view = PITData(store).view(GWKey("2024-25", gameweek))
    return understat_share(view)


def _later() -> dict[str, object]:
    return {
        "id": 2,
        "team_h": 10,
        "team_a": 20,
        "h": [
            {"id": 501, "xg": 0.0, "xa": 0.0, "time": 90},
            {"id": 502, "xg": 0.5, "xa": 0.0, "time": 90},
        ],
        "a": [],
    }


def _store(root: Path) -> DataStore:
    store = DataStore(root)
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": ["2024-25", "2024-25"],
                "gw": [1, 2],
                "deadline_time": [
                    datetime(2024, 8, 16, 18, tzinfo=UTC),
                    datetime(2024, 8, 23, 18, tzinfo=UTC),
                ],
                "finished": [True, False],
                "data_checked": [True, False],
                "available_at": [_WHEN, _WHEN],
                "source": ["test", "test"],
            }
        ),
    )
    return store
