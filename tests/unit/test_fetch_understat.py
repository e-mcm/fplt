"""Understat fetch writes finished matches once and skips unfinished ones."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

_PATH = Path(__file__).parents[2] / "scripts" / "fetch_understat.py"
_SPEC = importlib.util.spec_from_file_location("fplt_fetch_understat", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
fetch = importlib.util.module_from_spec(_SPEC)
sys.modules["fplt_fetch_understat"] = fetch
_SPEC.loader.exec_module(fetch)

_FIXTURES = Path(__file__).parents[1] / "fixtures" / "understat"
_NOW = datetime(2024, 8, 16, 10, 0, tzinfo=UTC)
_LEAGUE_HTML = (_FIXTURES / "league.html").read_text(encoding="utf-8")
_MAINTENANCE_HTML = (_FIXTURES / "maintenance.html").read_text(encoding="utf-8")
_LEAGUE_DATA_AJAX = (_FIXTURES / "league_data_ajax.json").read_text(encoding="utf-8")


def _load(name: str) -> object:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def test_finished_match_ids_skip_unplayed() -> None:
    dates = _load("dates.json")
    assert isinstance(dates, list)
    assert fetch.finished_match_ids(dates) == [42]


def test_season_helpers() -> None:
    assert fetch.season_label(_NOW) == "2024-25"
    assert fetch.understat_year("2024-25") == "2024"


def test_parse_embedded_teams_data() -> None:
    teams = fetch.parse_embedded_json(_LEAGUE_HTML, "teamsData")
    assert teams["83"]["history"][0]["deep"] == 8


def test_dates_from_league_data_ajax_fixture() -> None:
    """Recorded getLeagueData JSON is the post-embedding dates source."""
    payload = json.loads(_LEAGUE_DATA_AJAX)
    dates = fetch.dates_from_league_data(payload)
    assert len(dates) >= 2
    assert fetch.finished_match_ids(dates) == [28778, 28779]
    teams = fetch.teams_from_league_data(payload)
    assert "87" in teams and "73" in teams


def test_old_html_parser_fails_on_ajax_league_data_fixture() -> None:
    """The pre-AJAX datesData HTML parser must reject the new JSON shape."""
    with pytest.raises(KeyError, match="datesData not found"):
        fetch.parse_embedded_json(_LEAGUE_DATA_AJAX, "datesData")


def test_capture_writes_dates_teams_and_finished_matches_once(tmp_path: Path) -> None:
    dates = _load("dates.json")
    match = _load("match_ajax.json")
    calls = {"dates": 0, "match": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/league/EPL/2024":
            return httpx.Response(200, text=_LEAGUE_HTML)
        if path == "/getLeagueData/EPL/2024":
            calls["dates"] += 1
            body = {"dates": dates, "teams": _load("teams.json"), "players": []}
            return httpx.Response(200, json=body)
        if path == "/getMatchData/42":
            calls["match"] += 1
            return httpx.Response(200, json=match)
        raise AssertionError(path)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    root = tmp_path / "raw"
    first = fetch.capture(client, root, "2024-25", now=_NOW, delay=0)
    second = fetch.capture(client, root, "2024-25", now=_NOW, delay=0)
    assert first.wrote_dates
    assert first.wrote_teams
    assert first.match_ids == [42]
    assert not second.wrote_dates
    assert not second.wrote_teams
    assert second.match_ids == []
    assert calls["match"] == 1
    path = fetch.match_path(root, "2024-25", 42)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        assert json.load(handle)["tmpl"]["h"] == "83"
    assert not fetch.match_path(root, "2024-25", 43).exists()
    teams_file = fetch.teams_path(root, "2024-25", _NOW)
    with gzip.open(teams_file, "rt", encoding="utf-8") as handle:
        teams = json.load(handle)
    assert teams["83"]["history"][0]["ppda"]["att"] == 240


def test_capture_uses_ajax_league_data_without_html_embed(tmp_path: Path) -> None:
    """New page shape: league HTML has no datesData; AJAX carries dates/teams."""
    payload = json.loads(_LEAGUE_DATA_AJAX)
    bare_html = "<html><head><title>EPL</title></head><body></body></html>"
    match = _load("match_ajax.json")

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/league/EPL/2025":
            return httpx.Response(200, text=bare_html)
        if path == "/getLeagueData/EPL/2025":
            return httpx.Response(200, json=payload)
        if path.startswith("/getMatchData/"):
            return httpx.Response(200, json=match)
        raise AssertionError(path)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    now = datetime(2025, 8, 16, 10, 0, tzinfo=UTC)
    result = fetch.capture(client, tmp_path / "raw", "2025-26", now=now, delay=0)
    assert result.wrote_dates
    assert result.wrote_teams
    assert result.match_ids == [28778, 28779]


def test_capture_falls_back_to_html_dates_when_ajax_html(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/league/EPL/2024":
            return httpx.Response(200, text=_LEAGUE_HTML)
        if path == "/getLeagueData/EPL/2024":
            return httpx.Response(200, text="<html>down</html>")
        if path == "/getMatchData/42":
            return httpx.Response(200, text="<html>down</html>")
        raise AssertionError(path)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    result = fetch.capture(client, tmp_path / "raw", "2024-25", now=_NOW, delay=0)
    assert result.wrote_dates
    assert result.wrote_teams
    assert result.match_ids == []


def test_capture_raises_on_maintenance_html(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_MAINTENANCE_HTML)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    with pytest.raises(fetch.UnderstatUnavailable, match="maintenance"):
        fetch.capture(client, tmp_path / "raw", "2025-26", now=_NOW, delay=0)


def test_main_exits_zero_when_understat_unavailable(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, text=_MAINTENANCE_HTML)

    real_client = httpx.Client

    def fake_client(*args: object, **kwargs: object) -> httpx.Client:
        kwargs = dict(kwargs)
        kwargs["transport"] = httpx.MockTransport(handler)
        return real_client(*args, **kwargs)

    monkeypatch.setattr(fetch.httpx, "Client", fake_client)
    assert fetch.main(["--out", str(tmp_path / "raw"), "--season", "2025-26"]) == 0
