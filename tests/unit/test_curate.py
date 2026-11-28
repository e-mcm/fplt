"""#345: curate job library — build + status/curated/latest.json pointer."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fplt.automation.curate import (
    curated_latest_path,
    load_curated_latest,
    run_curate,
)
from fplt.cli import main
from fplt.settings import get_settings
from fplt.store.builds import build_manifest_path, load_build_manifest

_TS = datetime(2026, 11, 24, 12, 0, 0, tzinfo=UTC)

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


def _write_capture(data_dir: Path, *, stamp: str) -> None:
    dest = data_dir / "raw" / "fpl_api" / "season=2024-25" / f"captured_at={stamp}"
    dest.mkdir(parents=True, exist_ok=True)
    with gzip.open(dest / "bootstrap_static.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(_BOOTSTRAP, handle)
    with gzip.open(dest / "fixtures.json.gz", "wt", encoding="utf-8") as handle:
        json.dump(_FIXTURES, handle)


def _patch_feeds_local_only(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Skip network / side feeds; keep real ingest_raw + build path."""
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    monkeypatch.setattr(
        "fplt.automation.ops_feeds.fetch_raw",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "fplt.automation.ops_feeds.pull_quarantine",
        lambda *a, **k: False,
    )
    monkeypatch.setattr(
        "fplt.automation.ops_feeds.import_available",
        lambda *a, **k: [],
    )
    monkeypatch.setattr("fplt.automation.ops_feeds.load_id_maps", lambda _p: ({}, {}))
    monkeypatch.setattr(
        "fplt.automation.ops_feeds.default_id_maps_path",
        lambda: tmp_path / "u.yaml",
    )
    monkeypatch.setattr(
        "fplt.automation.ops_feeds.ingest_understat",
        lambda *a, **k: 0,
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
        lambda *a, **k: 0,
    )


def test_run_curate_writes_build_and_pointer_idempotent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_feeds_local_only(monkeypatch, tmp_path)
    _write_capture(tmp_path, stamp="2026-11-24T12-00-00Z")

    first = run_curate(tmp_path, source="branch", code_sha="sha-one")
    assert first.pointer_updated is True
    assert first.pointer is not None
    assert first.pointer.build_id == first.manifest.build_id
    assert first.pointer.code_sha == "sha-one"
    assert curated_latest_path(tmp_path).is_file()
    assert build_manifest_path(tmp_path, first.manifest.build_id).is_file()
    pointer_text = curated_latest_path(tmp_path).read_text(encoding="utf-8")

    second = run_curate(tmp_path, source="branch", code_sha="sha-two")
    assert second.pointer_updated is False
    assert second.manifest.build_id == first.manifest.build_id
    assert curated_latest_path(tmp_path).read_text(encoding="utf-8") == pointer_text
    loaded = load_curated_latest(tmp_path)
    assert loaded is not None
    assert loaded.build_id == first.manifest.build_id
    assert loaded.code_sha == "sha-one"
    get_settings.cache_clear()


def test_run_curate_new_raw_updates_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _patch_feeds_local_only(monkeypatch, tmp_path)
    _write_capture(tmp_path, stamp="2026-11-24T12-00-00Z")
    first = run_curate(tmp_path, source="branch", code_sha="sha-a")
    first_id = first.manifest.build_id

    # New capture path changes raw_manifest_hash → new build_id.
    _write_capture(tmp_path, stamp="2026-11-24T13-00-00Z")
    second = run_curate(tmp_path, source="branch", code_sha="sha-b")
    assert second.pointer_updated is True
    assert second.manifest.build_id != first_id
    assert second.pointer is not None
    assert second.pointer.build_id == second.manifest.build_id
    assert second.pointer.code_sha == "sha-b"
    assert load_curated_latest(tmp_path) == second.pointer
    assert load_build_manifest(tmp_path, first_id).build_id == first_id
    get_settings.cache_clear()


def test_cli_data_curate_prints_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _patch_feeds_local_only(monkeypatch, tmp_path)

    def fake_run(*_a: object, **_k: object) -> object:
        from fplt.automation.curate import CurateResult, pointer_from_manifest
        from fplt.automation.ops_feeds import FeedRefreshResult
        from fplt.schemas.manifests import BuildManifest, BuildTableEntry

        manifest = BuildManifest(
            build_id="build_cli",
            created_at=_TS,
            staging_version=1,
            code_sha="cli",
            raw_manifest_hash="a" * 64,
            raw_cutoff=_TS,
            tables={
                "players": BuildTableEntry(sha="b" * 64, rows=1, schema_hash="c" * 64),
            },
        )
        pointer = pointer_from_manifest(manifest)
        return CurateResult(
            feeds=FeedRefreshResult(0, 1, 0, 0),
            manifest=manifest,
            pointer=pointer,
            pointer_updated=True,
        )

    monkeypatch.setattr("fplt.automation.curate.run_curate", fake_run)
    main(["data", "curate", "--data-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert "build_id=build_cli" in out
    assert "pointer_updated=yes" in out
    get_settings.cache_clear()
