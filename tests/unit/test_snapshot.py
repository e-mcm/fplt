"""Snapshot decisions and a dry run against a recorded payload."""

from __future__ import annotations

import gzip
import importlib.util
import json
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

import httpx

_PATH = Path(__file__).parents[2] / "scripts" / "snapshot.py"
_SPEC = importlib.util.spec_from_file_location("fplt_snapshot", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
snapshot = importlib.util.module_from_spec(_SPEC)
sys.modules["fplt_snapshot"] = snapshot
_SPEC.loader.exec_module(snapshot)

_FIXTURES = Path(__file__).parents[1] / "fixtures" / "snapshot"
_NOW = datetime(2024, 8, 16, 10, 0, tzinfo=UTC)


def _load(name: str) -> object:
    return json.loads((_FIXTURES / name).read_text())


def test_should_snapshot_when_deadline_is_close() -> None:
    deadline = _NOW + timedelta(hours=5)
    last = _NOW - timedelta(hours=1)
    assert snapshot.should_snapshot(_NOW, deadline, last)


def test_should_snapshot_when_stale() -> None:
    deadline = _NOW + timedelta(days=3)
    last = _NOW - timedelta(hours=13)
    assert snapshot.should_snapshot(_NOW, deadline, last)


def test_skips_when_fresh_and_deadline_is_far() -> None:
    deadline = _NOW + timedelta(days=3)
    last = _NOW - timedelta(hours=2)
    assert not snapshot.should_snapshot(_NOW, deadline, last)


def test_dry_run_writes_gzipped_json_once(tmp_path: Path) -> None:
    bootstrap = _load("bootstrap_static.json")
    fixtures = _load("fixtures.json")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/bootstrap-static/"):
            return httpx.Response(200, json=bootstrap)
        if request.url.path.endswith("/fixtures/"):
            return httpx.Response(200, json=fixtures)
        raise AssertionError(request.url.path)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    root = tmp_path / "raw"
    first = snapshot.capture(client, root, now=_NOW)
    second = snapshot.capture(client, root, now=_NOW)
    assert first.wrote_static
    assert not second.wrote_static
    path = snapshot.snapshot_path(root, "2024-25", _NOW, "bootstrap_static")
    assert path.exists()
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        assert json.load(handle)["events"][0]["id"] == 1
    fixtures_path = snapshot.snapshot_path(root, "2024-25", _NOW, "fixtures")
    assert fixtures_path.exists()


def test_checked_gameweek_is_captured_once(tmp_path: Path) -> None:
    bootstrap = _load("bootstrap_checked.json")
    calls = {"live": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/bootstrap-static/"):
            return httpx.Response(200, json=bootstrap)
        if request.url.path.endswith("/fixtures/"):
            return httpx.Response(200, json=[])
        if request.url.path.endswith("/event/1/live/"):
            calls["live"] += 1
            return httpx.Response(200, json=_load("event_live.json"))
        raise AssertionError(request.url.path)

    client = httpx.Client(transport=httpx.MockTransport(handler))
    snapshot.capture(client, tmp_path, now=_NOW)
    snapshot.capture(client, tmp_path, now=_NOW + timedelta(minutes=5))
    assert calls["live"] == 1
