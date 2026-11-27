"""ops-06 staging skip: quarantined captures never reach staged/ or PIT."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from fplt.data.gates import GATE_NULL_SHARE
from fplt.data.pit import PITData
from fplt.data.quarantine import (
    append_quarantine,
    capture_relpath,
    load_quarantine,
    pull_quarantine,
    push_quarantine,
    quarantined_capture_paths,
)
from fplt.data.raw import ingest_raw
from fplt.data.remote import Remote, RemoteError
from fplt.data.store import DataStore
from fplt.schemas.l2 import SCHEMA_VERSION
from fplt.schemas.quarantine import QuarantineRecordSchema
from fplt.settings import Settings, get_settings
from fplt.types import GWKey

_GOOD_AT = datetime(2024, 8, 10, 0, 0, 0, tzinfo=UTC)
_BAD_AT = datetime(2024, 8, 18, 0, 0, 0, tzinfo=UTC)
_BOOTSTRAP: dict[str, Any] = {
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


def _write_capture(raw_root: Path, stamp: datetime, bootstrap: dict[str, Any]) -> Path:
    label = stamp.strftime("%Y-%m-%dT%H-%M-%SZ")
    captured = raw_root / "fpl_api" / "season=2024-25" / f"captured_at={label}"
    captured.mkdir(parents=True)
    for name, payload in (("bootstrap_static", bootstrap), ("fixtures", _FIXTURES)):
        with gzip.open(captured / f"{name}.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
    return captured


def _quarantine_row(capture_path: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": ["2024-25"],
            "source": ["fpl_api"],
            "capture_path": [capture_path],
            "captured_at": [_BAD_AT],
            "gate": [GATE_NULL_SHARE],
            "detail": ["synthetic quarantine for staging-skip test"],
            "previous_capture_path": [None],
            "available_at": [_BAD_AT],
            "schema_version": [SCHEMA_VERSION],
        }
    )


def test_append_and_load_quarantine(tmp_path: Path) -> None:
    empty = load_quarantine(tmp_path)
    assert empty.is_empty()
    assert quarantined_capture_paths(tmp_path) == frozenset()
    path = "fpl_api/season=2024-25/captured_at=2024-08-18T00-00-00Z"
    merged = append_quarantine(tmp_path, _quarantine_row(path))
    QuarantineRecordSchema.validate(merged)
    assert GATE_NULL_SHARE in merged["gate"].to_list()
    assert path in quarantined_capture_paths(tmp_path)


def test_append_empty_frame_returns_existing(tmp_path: Path) -> None:
    path = "fpl_api/season=2024-25/captured_at=2024-08-18T00-00-00Z"
    append_quarantine(tmp_path, _quarantine_row(path))
    empty_schema = load_quarantine(tmp_path).clear()
    assert empty_schema.is_empty()
    out = append_quarantine(tmp_path, empty_schema)
    assert out.height == 1
    assert path in quarantined_capture_paths(tmp_path)


def test_append_merges_with_existing_rows(tmp_path: Path) -> None:
    first = "fpl_api/season=2024-25/captured_at=2024-08-18T00-00-00Z"
    second = "fpl_api/season=2024-25/captured_at=2024-08-19T00-00-00Z"
    append_quarantine(tmp_path, _quarantine_row(first))
    second_row = _quarantine_row(second).with_columns(
        pl.lit(datetime(2024, 8, 19, 0, 0, 0, tzinfo=UTC)).alias("captured_at"),
        pl.lit(datetime(2024, 8, 19, 0, 0, 0, tzinfo=UTC)).alias("available_at"),
    )
    merged = append_quarantine(tmp_path, second_row)
    assert merged.height == 2
    assert quarantined_capture_paths(tmp_path) == frozenset({first, second})


def test_pull_quarantine_noop_without_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "fplt.data.quarantine.Remote.from_settings", lambda *_a, **_k: None
    )
    assert pull_quarantine(tmp_path, Settings()) is False


def test_pull_quarantine_tolerates_missing_remote_prefix(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = Remote(target="fplt:fplt-data", kind="data", env={})
    monkeypatch.setattr(
        "fplt.data.quarantine.Remote.from_settings", lambda *_a, **_k: remote
    )

    def _fail_copy(*_a: object, **_k: object) -> None:
        raise RemoteError("Directory not found")

    monkeypatch.setattr("fplt.data.quarantine.copy_mutable", _fail_copy)
    assert pull_quarantine(tmp_path, Settings()) is True


def test_pull_quarantine_reraises_other_remote_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = Remote(target="fplt:fplt-data", kind="data", env={})
    monkeypatch.setattr(
        "fplt.data.quarantine.Remote.from_settings", lambda *_a, **_k: remote
    )

    def _fail_copy(*_a: object, **_k: object) -> None:
        raise RemoteError("rclone exploded")

    monkeypatch.setattr("fplt.data.quarantine.copy_mutable", _fail_copy)
    with pytest.raises(RemoteError, match="rclone exploded"):
        pull_quarantine(tmp_path, Settings())


def test_pull_quarantine_with_empty_local_keeps_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    remote = Remote(target="fplt:fplt-data", kind="data", env={})
    monkeypatch.setattr(
        "fplt.data.quarantine.Remote.from_settings", lambda *_a, **_k: remote
    )
    remote_path = "fpl_api/season=2024-25/captured_at=2024-08-19T00-00-00Z"
    remote_row = _quarantine_row(remote_path)

    def _copy_remote(_remote: Remote, _src: str, dest: str) -> None:
        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)
        remote_row.write_parquet(dest_path / "part.parquet")

    monkeypatch.setattr("fplt.data.quarantine.copy_mutable", _copy_remote)
    assert pull_quarantine(data_dir, Settings()) is True
    assert quarantined_capture_paths(data_dir) == frozenset({remote_path})


def test_pull_quarantine_merges_local_and_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    local_path = "fpl_api/season=2024-25/captured_at=2024-08-18T00-00-00Z"
    append_quarantine(data_dir, _quarantine_row(local_path))

    remote_root = tmp_path / "remote_bucket"
    remote = Remote(target=str(remote_root), kind="data", env={})
    monkeypatch.setattr(
        "fplt.data.quarantine.Remote.from_settings", lambda *_a, **_k: remote
    )

    remote_path = "fpl_api/season=2024-25/captured_at=2024-08-19T00-00-00Z"
    remote_row = _quarantine_row(remote_path).with_columns(
        pl.lit(datetime(2024, 8, 19, 0, 0, 0, tzinfo=UTC)).alias("captured_at"),
        pl.lit(datetime(2024, 8, 19, 0, 0, 0, tzinfo=UTC)).alias("available_at"),
    )

    def _copy_remote(_remote: Remote, _src: str, dest: str) -> None:
        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)
        # Simulate remote overwrite of local parquet.
        remote_row.write_parquet(dest_path / "part.parquet")

    monkeypatch.setattr("fplt.data.quarantine.copy_mutable", _copy_remote)
    assert pull_quarantine(data_dir, Settings()) is True
    paths = quarantined_capture_paths(data_dir)
    assert local_path in paths
    assert remote_path in paths


def test_pull_quarantine_restores_local_when_remote_empty(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    local_path = "fpl_api/season=2024-25/captured_at=2024-08-18T00-00-00Z"
    append_quarantine(data_dir, _quarantine_row(local_path))

    remote = Remote(target="fplt:fplt-data", kind="data", env={})
    monkeypatch.setattr(
        "fplt.data.quarantine.Remote.from_settings", lambda *_a, **_k: remote
    )

    def _copy_clears(_remote: Remote, _src: str, dest: str) -> None:
        dest_path = Path(dest)
        dest_path.mkdir(parents=True, exist_ok=True)
        # Remote copy leaves no parquet → load_quarantine returns empty.
        for child in dest_path.iterdir():
            child.unlink()

    monkeypatch.setattr("fplt.data.quarantine.copy_mutable", _copy_clears)
    assert pull_quarantine(data_dir, Settings()) is True
    assert local_path in quarantined_capture_paths(data_dir)


def test_pull_quarantine_uses_get_settings_when_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    get_settings.cache_clear()
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    monkeypatch.setattr(
        "fplt.data.quarantine.Remote.from_settings", lambda *_a, **_k: None
    )
    try:
        assert pull_quarantine(tmp_path) is False
    finally:
        get_settings.cache_clear()


def test_push_quarantine_noop_without_remote_or_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "fplt.data.quarantine.Remote.from_settings", lambda *_a, **_k: None
    )
    assert push_quarantine(tmp_path, Settings()) is False

    remote = Remote(target="fplt:fplt-data", kind="data", env={})
    monkeypatch.setattr(
        "fplt.data.quarantine.Remote.from_settings", lambda *_a, **_k: remote
    )
    assert push_quarantine(tmp_path, Settings()) is False
    empty = tmp_path / "quarantine"
    empty.mkdir()
    assert push_quarantine(tmp_path, Settings()) is False


def test_push_quarantine_copies_when_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    data_dir = tmp_path / "data"
    append_quarantine(
        data_dir,
        _quarantine_row("fpl_api/season=2024-25/captured_at=2024-08-18T00-00-00Z"),
    )
    remote = Remote(target="fplt:fplt-data", kind="data", env={})
    monkeypatch.setattr(
        "fplt.data.quarantine.Remote.from_settings", lambda *_a, **_k: remote
    )
    seen: dict[str, str] = {}

    def _copy(_remote: Remote, src: str, dest: str) -> None:
        seen["src"] = src
        seen["dest"] = dest

    monkeypatch.setattr("fplt.data.quarantine.copy_mutable", _copy)
    assert push_quarantine(data_dir, Settings()) is True
    assert seen["src"] == str(data_dir / "quarantine")
    assert seen["dest"] == "fplt:fplt-data/status/quarantine"


def test_ingest_skips_quarantined_capture_pit_excludes(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    data_dir = tmp_path / "data"
    _write_capture(raw_root, _GOOD_AT, _BOOTSTRAP)
    bad_bootstrap = {
        **_BOOTSTRAP,
        "elements": [{**_BOOTSTRAP["elements"][0], "now_cost": 999}],
    }
    bad_dir = _write_capture(raw_root, _BAD_AT, bad_bootstrap)
    bad_rel = capture_relpath(raw_root, bad_dir)
    append_quarantine(data_dir, _quarantine_row(bad_rel))

    store = DataStore(data_dir)
    count = ingest_raw(store, raw_root)
    assert count == 1
    status = store.read("player_status")
    assert status["price"].to_list() == [100]
    assert _BAD_AT not in status["available_at"].to_list()

    view = PITData(store).view(GWKey("2024-25", 1))
    pit_status = view.latest("player_status", by=["player_key"]).collect()
    assert pit_status["price"].to_list() == [100]
    assert pit_status["available_at"].to_list() == [_GOOD_AT]
