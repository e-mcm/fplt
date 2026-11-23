"""API-Football fetch script: write-once, key env, rate-limit skip."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx
import pytest

from scripts.fetch_api_football import (
    capture,
    capture_fixtures,
    fixtures_path,
    injuries_path,
    load_competitions,
    resolve_api_key,
    should_fetch,
    write_once,
)


def test_resolve_api_key_prefers_fplt_name() -> None:
    assert (
        resolve_api_key(
            {
                "FPLT_API_FOOTBALL_KEY": "a",
                "API_FOOTBALL_KEY": "b",
                "APISPORTS_KEY": "c",
            }
        )
        == "a"
    )
    assert resolve_api_key({"API_FOOTBALL_KEY": "b"}) == "b"
    assert resolve_api_key({"APISPORTS_KEY": "c"}) == "c"
    assert resolve_api_key({}) is None


def test_write_once_never_overwrites(tmp_path: Path) -> None:
    path = tmp_path / "injuries.json.gz"
    assert write_once(path, {"ok": 1}) is True
    assert write_once(path, {"ok": 2}) is False
    # Original payload retained.
    import gzip

    with gzip.open(path, "rt", encoding="utf-8") as handle:
        assert json.load(handle) == {"ok": 1}


def test_should_fetch_respects_min_interval(tmp_path: Path) -> None:
    season = "2024-25"
    first = datetime(2024, 8, 15, 12, 0, tzinfo=UTC)
    path = injuries_path(tmp_path, season, first)
    write_once(path, {"response": []})
    now = first + timedelta(hours=2)
    assert should_fetch(tmp_path, season, now, min_interval=timedelta(hours=4)) is False
    later = first + timedelta(hours=5)
    assert (
        should_fetch(tmp_path, season, later, min_interval=timedelta(hours=4)) is True
    )


def test_capture_skips_without_http_when_recent(tmp_path: Path) -> None:
    season = "2024-25"
    first = datetime(2024, 8, 15, 12, 0, tzinfo=UTC)
    write_once(injuries_path(tmp_path, season, first), {"response": []})

    def handler(request: httpx.Request) -> httpx.Response:
        raise AssertionError(f"unexpected request {request.url}")

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        result = capture(
            client,
            tmp_path,
            season,
            now=first + timedelta(hours=1),
            min_interval_hours=4.0,
        )
    assert result.skipped is True
    assert result.wrote is False


def test_capture_writes_from_mock_http(tmp_path: Path) -> None:
    season = "2024-25"
    moment = datetime(2024, 8, 15, 12, 0, tzinfo=UTC)
    payload = {"response": [{"player": {"id": 1}}]}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/injuries")
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        result = capture(
            client,
            tmp_path,
            season,
            now=moment,
            min_interval_hours=0.0,
        )
    assert result.wrote is True
    assert result.path is not None
    assert result.path.exists()


def test_main_exits_without_key(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from scripts import fetch_api_football as mod

    for name in ("FPLT_API_FOOTBALL_KEY", "API_FOOTBALL_KEY", "APISPORTS_KEY"):
        monkeypatch.delenv(name, raising=False)
    assert mod.main(["--out", str(tmp_path), "--season", "2024-25"]) == 2


def test_load_competitions_includes_cups_not_epl() -> None:
    pairs = load_competitions()
    assert (2, "ucl") in pairs
    assert all(league_id != 39 for league_id, _ in pairs)


def test_capture_fixtures_writes_from_mock_http(tmp_path: Path) -> None:
    season = "2024-25"
    moment = datetime(2024, 8, 15, 12, 0, tzinfo=UTC)
    payload = {"response": [{"fixture": {"id": 1}}]}

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/fixtures")
        assert "league=2" in str(request.url)
        return httpx.Response(200, json=payload)

    transport = httpx.MockTransport(handler)
    with httpx.Client(transport=transport) as client:
        result = capture_fixtures(
            client,
            tmp_path,
            season,
            2,
            now=moment,
            min_interval_hours=0.0,
        )
    assert result.wrote is True
    assert result.kind == "fixtures"
    assert result.path == fixtures_path(tmp_path, season, 2, moment)
    assert result.path is not None and result.path.exists()
