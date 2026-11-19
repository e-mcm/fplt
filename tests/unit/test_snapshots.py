"""Snapshot fetches copy new gzip files and leave existing ones alone."""

import subprocess
from pathlib import Path

import pytest

from fplt.data.snapshots import SnapshotFetchError, fetch_snapshots


def _commit(repo: Path, branch: str) -> None:
    subprocess.run(
        ["git", "init", "-b", branch, str(repo)],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        ["git", "-C", str(repo), "add", "."],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test",
            "commit",
            "-m",
            "snapshot",
        ],
        check=True,
        capture_output=True,
    )


def test_fetch_copies_new_files_once(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    raw = remote / "data" / "raw" / "fpl_api" / "season=2024-25"
    raw.mkdir(parents=True)
    (raw / "bootstrap_static.json.gz").write_bytes(b"abc")
    _commit(remote, "snapshots")
    dest = tmp_path / "raw"
    first = fetch_snapshots(dest, remote=str(remote))
    assert [path.name for path in first] == ["bootstrap_static.json.gz"]
    assert first[0].read_bytes() == b"abc"
    assert fetch_snapshots(dest, remote=str(remote)) == []


def test_fetch_without_a_raw_directory_returns_nothing(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "README.md").write_text("no snapshots\n", encoding="utf-8")
    _commit(remote, "snapshots")
    assert fetch_snapshots(tmp_path / "raw", remote=str(remote)) == []


def test_missing_branch_raises(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    remote.mkdir()
    (remote / "README.md").write_text("main\n", encoding="utf-8")
    _commit(remote, "main")
    with pytest.raises(SnapshotFetchError):
        fetch_snapshots(tmp_path / "raw", remote=str(remote))
