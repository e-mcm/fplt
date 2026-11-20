"""Understat fetch writes finished matches once and skips unfinished ones."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import httpx

_PATH = Path(__file__).parents[2] / "scripts" / "fetch_understat.py"
_SPEC = importlib.util.spec_from_file_location("fplt_fetch_understat", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
fetch = importlib.util.module_from_spec(_SPEC)
sys.modules["fplt_fetch_understat"] = fetch
_SPEC.loader.exec_module(fetch)

_FIXTURES = Path(__file__).parents[1] / "fixtures" / "understat"
_NOW = datetime(2024, 8, 16, 10, 0, tzinfo=UTC)


def _load(name: str) -> object:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def test_finished_match_ids_skip_unplayed() -> None:
    dates = _load("dates.json")
    assert isinstance(dates, list)
    assert fetch.finished_match_ids(dates) == [42]


def test_season_helpers() -> None:
    assert fetch.season_label(_NOW) == "2024-25"
    assert fetch.understat_year("2024-25") == "2024"


def test_capture_writes_dates_and_finished_matches_once(tmp_path: Path) -> None:
    dates = _load("dates.json")
    match = _load("match_ajax.json")
    calls = {"dates": 0, "match": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/league/EPL/2024":
            return httpx.Response(200, text="<html/>")
        if path == "/getLeagueData/EPL/2024":
            calls["dates"] += 1
            body = {"dates": dates, "teams": {}, "players": []}
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
    assert first.match_ids == [42]
    assert not second.wrote_dates
    assert second.match_ids == []
    assert calls["match"] == 1
    path = fetch.match_path(root, "2024-25", 42)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        assert json.load(handle)["tmpl"]["h"] == "83"
    assert not fetch.match_path(root, "2024-25", 43).exists()
