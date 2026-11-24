"""Branch tip vs bucket raw parity (dual-write A6)."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest

from fplt.cli import main
from fplt.data import remote as remote_mod
from fplt.data.parity import ParityError, ParityResult, check_raw_parity
from fplt.data.remote import Remote, RemoteError
from fplt.settings import Settings

_HAS_RCLONE = shutil.which("rclone") is not None


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


def _point_at_local_bucket(monkeypatch: pytest.MonkeyPatch, bucket: Path) -> None:
    def _remote(s: Settings, bucket_name: str | None = None) -> Remote:
        return Remote(target=str(bucket), kind="data", env={})

    monkeypatch.setattr("fplt.data.parity.Remote.from_settings", _remote)


def test_check_one_way_argv(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_rclone(args: list[str], *, env: dict[str, str]) -> None:
        seen["args"] = args
        seen["env"] = env

    monkeypatch.setattr(remote_mod, "_rclone", fake_rclone)
    src = tmp_path / "src"
    src.mkdir()
    remote = Remote(target="fplt:fplt-data", kind="data", env={"K": "V"})
    remote_mod.check_one_way(src, remote, "raw")
    assert seen["args"] == [
        "check",
        "--one-way",
        "--fast-list",
        str(src),
        "fplt:fplt-data/raw",
    ]
    assert seen["env"] == {"K": "V"}


def test_check_one_way_propagates_remote_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail(args: list[str], *, env: dict[str, str]) -> None:
        raise RemoteError("1 differences found")

    monkeypatch.setattr(remote_mod, "_rclone", fail)
    with pytest.raises(RemoteError, match="differences"):
        remote_mod.check_one_way(
            tmp_path, Remote(target="fplt:fplt-data", kind="data", env={}), "raw"
        )


def test_check_raw_parity_requires_data_bucket(tmp_path: Path) -> None:
    with pytest.raises(ParityError, match="data bucket is not configured"):
        check_raw_parity(Settings(), remote=str(tmp_path))


def test_check_raw_parity_clone_failure(tmp_path: Path) -> None:
    settings = Settings(data_bucket="fplt-data")
    missing = tmp_path / "no-such-repo"
    with pytest.raises(ParityError, match="(?i)(could not|fatal|not)"):
        check_raw_parity(settings, remote=str(missing), branch="snapshots")


@pytest.mark.skipif(not _HAS_RCLONE, reason="rclone binary not installed")
def test_local_parity_ok_extra_bucket_ok_missing_fails(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    branch_repo = tmp_path / "branch"
    raw = branch_repo / "data" / "raw" / "fpl_api"
    raw.mkdir(parents=True)
    (raw / "a.json.gz").write_bytes(b"abc")
    _commit(branch_repo, "snapshots")

    bucket = tmp_path / "bucket"
    (bucket / "raw" / "fpl_api").mkdir(parents=True)
    (bucket / "raw" / "fpl_api" / "a.json.gz").write_bytes(b"abc")
    (bucket / "raw" / "extra.txt").write_text("only-in-bucket", encoding="utf-8")

    _point_at_local_bucket(monkeypatch, bucket)
    settings = Settings(data_bucket="fplt-data")
    result = check_raw_parity(settings, remote=str(branch_repo), branch="snapshots")
    assert result.source_files == 1
    assert result.branch == "snapshots"
    assert result.remote_target == str(bucket)

    (raw / "b.json.gz").write_bytes(b"missing-from-bucket")
    subprocess.run(
        ["git", "-C", str(branch_repo), "add", "."],
        check=True,
        capture_output=True,
    )
    subprocess.run(
        [
            "git",
            "-C",
            str(branch_repo),
            "-c",
            "user.email=test@example.com",
            "-c",
            "user.name=Test",
            "commit",
            "-m",
            "add b",
        ],
        check=True,
        capture_output=True,
    )
    with pytest.raises(ParityError, match="(?i)(missing|differ|differences)"):
        check_raw_parity(settings, remote=str(branch_repo), branch="snapshots")


@pytest.mark.skipif(not _HAS_RCLONE, reason="rclone binary not installed")
def test_local_parity_size_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    branch_repo = tmp_path / "branch"
    raw = branch_repo / "data" / "raw"
    raw.mkdir(parents=True)
    (raw / "a.txt").write_text("branch", encoding="utf-8")
    _commit(branch_repo, "snapshots")

    bucket = tmp_path / "bucket"
    (bucket / "raw").mkdir(parents=True)
    (bucket / "raw" / "a.txt").write_text("bucket-different-size!!", encoding="utf-8")

    _point_at_local_bucket(monkeypatch, bucket)
    with pytest.raises(ParityError):
        check_raw_parity(
            Settings(data_bucket="fplt-data"),
            remote=str(branch_repo),
            branch="snapshots",
        )


@pytest.mark.skipif(not _HAS_RCLONE, reason="rclone binary not installed")
def test_local_parity_empty_branch_ok(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    branch_repo = tmp_path / "branch"
    branch_repo.mkdir()
    (branch_repo / "README.md").write_text("no raw yet\n", encoding="utf-8")
    _commit(branch_repo, "snapshots")
    bucket = tmp_path / "bucket"
    (bucket / "raw").mkdir(parents=True)
    _point_at_local_bucket(monkeypatch, bucket)
    result = check_raw_parity(
        Settings(data_bucket="fplt-data"),
        remote=str(branch_repo),
        branch="snapshots",
    )
    assert result.source_files == 0


def test_cli_parity_ok(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def fake_check(**kwargs: object) -> ParityResult:
        return ParityResult(
            source_files=3, remote_target="fplt:fplt-data", branch="snapshots"
        )

    monkeypatch.setattr("fplt.data.parity.check_raw_parity", fake_check)
    main(["data", "parity", "--remote", "/tmp/repo"])
    out = capsys.readouterr().out
    assert "raw parity ok: 3 files" in out
    assert "fplt:fplt-data/raw" in out


def test_cli_parity_mismatch_exits_1(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(**kwargs: object) -> None:
        raise ParityError("1 files missing")

    monkeypatch.setattr("fplt.data.parity.check_raw_parity", boom)
    with pytest.raises(SystemExit) as exc:
        main(["data", "parity"])
    assert exc.value.code == 1
