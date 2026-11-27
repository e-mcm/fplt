"""Easy leftover coverage for fplt.data.understat helpers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from fplt.data.store import DataStore
from fplt.data.understat import (
    _ppda_pair,
    available_at_from_match,
    dates_raw_path,
    default_id_maps_path,
    iter_raw_dates,
    iter_raw_matches,
    iter_raw_teams,
    load_id_maps,
    normalize_match,
    parse_embedded_json,
    stage_understat,
    stage_understat_team_matches,
)

_EARLY = datetime(2024, 8, 10, tzinfo=UTC)
_PLAYERS = {501: 100, 502: 200}
_TEAMS = {10: 3, 20: 7}


def test_default_id_maps_path_and_empty_yaml(tmp_path: Path) -> None:
    assert default_id_maps_path().name == "id_maps.yaml"
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    players, teams = load_id_maps(empty)
    assert players == {}
    assert teams == {}
    bad = tmp_path / "bad.yaml"
    bad.write_text("- not a map\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_id_maps(bad)
    stamp = datetime(2024, 8, 1, 12, 0, tzinfo=UTC)
    path = dates_raw_path(tmp_path, "2024-25", stamp)
    assert "captured_at=2024-08-01T12-00-00Z" in str(path)
    html = "datesData = JSON.parse('{\"ok\": true}')"
    assert parse_embedded_json(html, "datesData") == {"ok": True}
    with pytest.raises(KeyError):
        parse_embedded_json(html, "missing")


def test_available_at_variants() -> None:
    assert available_at_from_match(
        {"available_at": datetime(2024, 8, 1, 12, tzinfo=UTC)}
    ) == datetime(2024, 8, 1, 12, tzinfo=UTC)
    assert available_at_from_match({"available_at": "2024-08-01 12:00:00"}) == datetime(
        2024, 8, 1, 12, tzinfo=UTC
    )
    assert available_at_from_match({"datetime": "2024-08-01 12:00:00"}) == datetime(
        2024, 8, 1, 15, tzinfo=UTC
    )
    with pytest.raises(KeyError):
        available_at_from_match({})


def test_iter_raw_empty_and_filters(tmp_path: Path) -> None:
    assert iter_raw_matches(tmp_path) == []
    assert iter_raw_teams(tmp_path) == []
    assert iter_raw_dates(tmp_path) == []
    base = tmp_path / "understat" / "season=2024-25"
    match = base / "match_id=9" / "match.json.gz"
    match.parent.mkdir(parents=True)
    match.write_bytes(b"")
    teams = base / "teams" / "captured_at=2024-08-01T00-00-00Z" / "teams.json.gz"
    teams.parent.mkdir(parents=True)
    teams.write_bytes(b"")
    dates = base / "dates" / "captured_at=2024-08-01T00-00-00Z" / "dates.json.gz"
    dates.parent.mkdir(parents=True)
    dates.write_bytes(b"")
    assert len(iter_raw_matches(tmp_path, seasons=["2023-24"])) == 0
    assert len(iter_raw_matches(tmp_path)) == 1
    assert len(iter_raw_teams(tmp_path, seasons=["2023-24"])) == 0
    assert len(iter_raw_teams(tmp_path)) == 1
    assert len(iter_raw_dates(tmp_path, seasons=["2023-24"])) == 0
    assert len(iter_raw_dates(tmp_path)) == 1


def test_normalize_already_normalised_and_side_list() -> None:
    payload = {
        "team_h": 1,
        "team_a": 2,
        "h": [{"id": 1, "xg": 0.1, "xa": 0.0, "time": 90}],
        "a": [],
    }
    out = normalize_match(99, payload)
    assert out["id"] == 99
    assert out["h"][0]["id"] == 1

    ajax = {
        "tmpl": {"h": 10, "a": 20, "date": "2024-08-01 15:00:00"},
        "rosters": {
            "h": [{"player_id": 1, "xG": 0.2, "xA": 0.1, "time": 45, "shots": ""}],
            "a": "bad",
        },
    }
    with pytest.raises(ValueError, match="dict or list"):
        normalize_match(1, ajax)


def test_stage_rejects_non_list_players(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="must be a list"):
        stage_understat(
            DataStore(tmp_path),
            [{"id": 1, "team_h": 10, "team_a": 20, "h": "nope", "a": []}],
            "2024-25",
            _EARLY,
            _PLAYERS,
            _TEAMS,
        )


def test_ppda_and_team_match_skip(tmp_path: Path) -> None:
    assert _ppda_pair(None) == (0.0, 0.0)
    assert _ppda_pair({"att": 1, "def": 2}) == (1.0, 2.0)
    teams = {
        "10": {
            "history": [
                {
                    "date": "2024-08-01 12:00:00",
                    "h_a": "h",
                    "deep": 1,
                    "deep_allowed": 0,
                    "xG": 0.5,
                    "xGA": 0.2,
                }
            ]
        }
    }
    staged = stage_understat_team_matches(
        DataStore(tmp_path), teams, [], "2024-25", {10: 3}
    )
    assert staged.height == 0
    with pytest.raises(KeyError, match="no team_key"):
        stage_understat_team_matches(
            DataStore(tmp_path / "b"), teams, [], "2024-25", {}
        )
