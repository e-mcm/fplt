"""Shared ops feed refresh used by ops-tick and ops-catchup."""

from pathlib import Path

import pytest

from fplt.automation.ops_feeds import FeedRefreshResult, refresh_ops_feeds
from fplt.settings import get_settings


def test_refresh_ops_feeds_calls_fetch_and_ingests(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    calls: list[str] = []

    def fetch(
        dest: Path,
        settings: object = None,
        *,
        source: str = "auto",
        remote: str = "origin",
        branch: str = "snapshots",
    ) -> list[Path]:
        calls.append(f"fetch:{source}:{remote}:{branch}")
        return [dest / "a.json.gz"]

    monkeypatch.setattr("fplt.automation.ops_feeds.fetch_raw", fetch)
    monkeypatch.setattr(
        "fplt.automation.ops_feeds.ingest_raw",
        lambda store, raw: calls.append("ingest_raw") or 2,
    )
    monkeypatch.setattr(
        "fplt.automation.ops_feeds.import_available",
        lambda store, root: calls.append("vaastav") or [],
    )
    monkeypatch.setattr("fplt.automation.ops_feeds.load_id_maps", lambda _p: ({}, {}))
    monkeypatch.setattr(
        "fplt.automation.ops_feeds.default_id_maps_path",
        lambda: tmp_path / "u.yaml",
    )
    monkeypatch.setattr(
        "fplt.automation.ops_feeds.ingest_understat",
        lambda *a, **k: calls.append("understat") or 3,
    )
    monkeypatch.setattr(
        "fplt.automation.ops_feeds.load_api_football_id_maps",
        lambda _p: ({}, {}),
    )
    monkeypatch.setattr(
        "fplt.automation.ops_feeds.api_football_id_maps_path",
        lambda: tmp_path / "af.yaml",
    )
    monkeypatch.setattr(
        "fplt.automation.ops_feeds.ingest_api_football",
        lambda *a, **k: calls.append("api_football") or 4,
    )

    result = refresh_ops_feeds(source="branch", remote="/tmp/r", branch="snapshots")
    assert result == FeedRefreshResult(1, 2, 3, 4)
    assert calls == [
        "fetch:branch:/tmp/r:snapshots",
        "ingest_raw",
        "vaastav",
        "understat",
        "api_football",
    ]
    get_settings.cache_clear()
