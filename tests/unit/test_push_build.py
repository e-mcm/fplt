"""Stage B6 PR2: push curated builds to the data bucket."""

from __future__ import annotations

import gzip
import json
import shutil
from datetime import UTC, datetime
from pathlib import Path

import pytest

from fplt.data import remote as remote_mod
from fplt.data.remote import Remote
from fplt.data.staging import stage_bootstrap
from fplt.data.store import DataStore
from fplt.settings import Settings
from fplt.store.builds import build_from_staged, build_manifest_path, table_sha_path
from fplt.store.push import maybe_push_build, push_build

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


def _make_build(data_dir: Path) -> str:
    raw = (
        data_dir
        / "raw"
        / "fpl_api"
        / "season=2024-25"
        / "captured_at=2026-11-24T12-00-00Z"
        / "bootstrap_static.json.gz"
    )
    raw.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(raw, "wt", encoding="utf-8") as handle:
        json.dump(_BOOTSTRAP, handle)
    store = DataStore(data_dir)
    frames = stage_bootstrap(_BOOTSTRAP, _FIXTURES, _TS, "2024-25")
    for name, frame in frames.items():
        if not frame.is_empty():
            store.write(name, frame)
    return build_from_staged(data_dir, code_sha="abc", created_at=_TS).build_id


def _fake_rclone_copy(args: list[str], *, env: dict[str, str]) -> None:
    """Minimal rclone stand-in: ``copy src dest`` into a local path."""
    del env
    assert args[0] == "copy"
    src = Path(args[1])
    dest = Path(args[2])
    dest.mkdir(parents=True, exist_ok=True)
    if src.is_dir():
        for path in src.rglob("*"):
            if path.is_file():
                target = dest / path.relative_to(src)
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(path, target)
    elif src.is_file():
        shutil.copy2(src, dest / src.name)


def test_push_build_to_data_local_dir(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(remote_mod, "_rclone", _fake_rclone_copy)
    build_id = _make_build(tmp_path)
    bucket = tmp_path / "data-bucket"
    remote = Remote(target=str(bucket), kind="data", env={})
    pushed = push_build(tmp_path, build_id, remote=remote)
    assert pushed.kind == "data"
    assert (bucket / "curated" / "builds" / f"build={build_id}.json").is_file()
    local_manifest = build_manifest_path(tmp_path, build_id)
    # At least one table sha landed under curated/tables/.
    assert any((bucket / "curated" / "tables").rglob("sha=*.parquet"))
    assert local_manifest.is_file()


def test_push_build_rejects_lab_remote(tmp_path: Path) -> None:
    build_id = _make_build(tmp_path)
    remote = Remote(target=str(tmp_path / "lab"), kind="lab", env={})
    with pytest.raises(ValueError, match="data bucket"):
        push_build(tmp_path, build_id, remote=remote)


def test_push_build_requires_configured_bucket(tmp_path: Path) -> None:
    build_id = _make_build(tmp_path)
    with pytest.raises(ValueError, match="data bucket is not configured"):
        push_build(tmp_path, build_id, settings=Settings())


def test_maybe_push_build_skips_when_bucket_unset(tmp_path: Path) -> None:
    build_id = _make_build(tmp_path)
    assert maybe_push_build(tmp_path, build_id, settings=Settings()) is None


def test_push_build_missing_table_file(tmp_path: Path) -> None:
    build_id = _make_build(tmp_path)
    # Delete one content-addressed table file after the build.
    from fplt.store.builds import load_build_manifest

    manifest = load_build_manifest(tmp_path, build_id)
    name, entry = next(iter(manifest.tables.items()))
    table_sha_path(tmp_path, name, entry.sha).unlink()
    remote = Remote(target=str(tmp_path / "bucket"), kind="data", env={})
    with pytest.raises(FileNotFoundError):
        push_build(tmp_path, build_id, remote=remote)
