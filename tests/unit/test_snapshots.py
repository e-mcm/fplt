"""Snapshot fetches copy new gzip files and leave existing ones alone."""

import subprocess
from pathlib import Path
from typing import cast

import pytest

from fplt.data.remote import Remote, RemoteError
from fplt.data.snapshots import (
    RawSource,
    SnapshotFetchError,
    fetch_raw,
    fetch_snapshots,
)
from fplt.settings import Settings


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


def test_fetch_copies_football_data_csv_once(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    odds = (
        remote
        / "data"
        / "raw"
        / "football_data"
        / "season=2024-25"
        / "captured_at=2025-11-01"
    )
    odds.mkdir(parents=True)
    (odds / "E0.csv").write_text("Div,Date\nE0,01/01/2024\n", encoding="utf-8")
    _commit(remote, "snapshots")
    dest = tmp_path / "raw"
    first = fetch_snapshots(dest, remote=str(remote))
    assert len(first) == 1
    assert first[0].name == "E0.csv"
    assert "football_data" in first[0].parts
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


def test_fetch_raw_auto_uses_branch_when_bucket_unset(tmp_path: Path) -> None:
    remote = tmp_path / "remote"
    raw = remote / "data" / "raw" / "fpl_api" / "season=2024-25"
    raw.mkdir(parents=True)
    (raw / "bootstrap_static.json.gz").write_bytes(b"branch")
    _commit(remote, "snapshots")
    dest = tmp_path / "raw"
    written = fetch_raw(
        dest,
        Settings(),
        source="auto",
        remote=str(remote),
    )
    assert [path.name for path in written] == ["bootstrap_static.json.gz"]
    assert written[0].read_bytes() == b"branch"


def test_fetch_raw_auto_pulls_bucket_when_configured(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    dest = tmp_path / "raw"
    expected = [dest / "fpl_api" / "a.json.gz"]
    settings = Settings(data_bucket="fplt-data")
    remote = Remote(target=str(tmp_path / "bucket"), kind="data", env={})
    pull_calls: list[tuple[Remote, str, Path]] = []

    def fake_from_settings(
        _cls: type[Remote], s: Settings, bucket: str | None = None
    ) -> Remote | None:
        assert s is settings
        assert bucket is None
        return remote

    def fake_pull(r: Remote, prefix: str, d: Path) -> list[Path]:
        pull_calls.append((r, prefix, d))
        return expected

    monkeypatch.setattr(
        Remote,
        "from_settings",
        classmethod(fake_from_settings),
    )
    monkeypatch.setattr("fplt.data.snapshots.pull", fake_pull)
    assert fetch_raw(dest, settings, source="auto") == expected
    assert pull_calls == [(remote, "raw", dest)]


def test_fetch_raw_source_branch_ignores_configured_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    remote = tmp_path / "remote"
    raw = remote / "data" / "raw" / "fpl_api"
    raw.mkdir(parents=True)
    (raw / "bootstrap_static.json.gz").write_bytes(b"ok")
    _commit(remote, "snapshots")

    def boom(*_a: object, **_k: object) -> list[Path]:
        raise AssertionError("bucket path must not run for source=branch")

    monkeypatch.setattr("fplt.data.snapshots.pull", boom)
    written = fetch_raw(
        tmp_path / "raw",
        Settings(data_bucket="fplt-data"),
        source="branch",
        remote=str(remote),
    )
    assert len(written) == 1


def test_fetch_raw_source_bucket_requires_data_bucket(tmp_path: Path) -> None:
    with pytest.raises(SnapshotFetchError, match="FPLT_DATA_BUCKET"):
        fetch_raw(tmp_path / "raw", Settings(), source="bucket")


def test_fetch_raw_auto_falls_back_to_branch_when_credentials_incomplete(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """DATA_BUCKET set without R2 secrets must not fail auto fetch."""
    # Soft-skip path: clear host GITHUB_REF so push-to-main CI does not raise.
    monkeypatch.delenv("GITHUB_REF", raising=False)
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    remote = tmp_path / "remote"
    raw = remote / "data" / "raw" / "fpl_api"
    raw.mkdir(parents=True)
    (raw / "bootstrap_static.json.gz").write_bytes(b"branch")
    _commit(remote, "snapshots")

    def boom(*_a: object, **_k: object) -> list[Path]:
        raise AssertionError("bucket pull must not run without credentials")

    monkeypatch.setattr("fplt.data.snapshots.pull", boom)
    written = fetch_raw(
        tmp_path / "raw",
        Settings(data_bucket="fplt-data"),
        source="auto",
        remote=str(remote),
    )
    assert len(written) == 1
    assert written[0].read_bytes() == b"branch"


def test_fetch_raw_source_bucket_requires_complete_credentials(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Soft-skip → SnapshotFetchError; isolate from push-to-main GITHUB_REF.
    monkeypatch.delenv("GITHUB_REF", raising=False)
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    with pytest.raises(SnapshotFetchError, match="complete R2 credentials"):
        fetch_raw(
            tmp_path / "raw",
            Settings(data_bucket="fplt-data"),
            source="bucket",
        )


def test_fetch_raw_source_bucket_propagates_remote_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    settings = Settings(data_bucket="fplt-data")

    def fake_from_settings(
        _cls: type[Remote], _s: Settings, bucket: str | None = None
    ) -> Remote:
        return Remote(target="fplt:fplt-data", kind="data", env={})

    monkeypatch.setattr(Remote, "from_settings", classmethod(fake_from_settings))

    def boom(*_a: object, **_k: object) -> list[Path]:
        raise RemoteError("rclone failed")

    monkeypatch.setattr("fplt.data.snapshots.pull", boom)
    with pytest.raises(RemoteError, match="rclone failed"):
        fetch_raw(tmp_path / "raw", settings, source="bucket")


def test_fetch_raw_rejects_unknown_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="source must be"):
        fetch_raw(
            tmp_path / "raw",
            Settings(),
            source=cast(RawSource, "other"),
        )
