"""Unit and local-dir integration tests for ``fplt.data.remote``."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import SecretStr

from fplt.data import remote as remote_mod
from fplt.data.remote import PUSHABLE, Remote, RemoteError, pull, push
from fplt.settings import Settings

_HAS_RCLONE = shutil.which("rclone") is not None


def test_rclone_env_maps_settings_without_putting_secret_in_argv() -> None:
    settings = Settings(
        r2_account_id="acct",
        r2_access_key_id="AKIA",
        r2_secret_access_key=SecretStr("sekrit"),
        data_bucket="fplt-data",
    )
    env = remote_mod._rclone_env(settings)
    assert env["RCLONE_CONFIG_FPLT_TYPE"] == "s3"
    assert env["RCLONE_CONFIG_FPLT_PROVIDER"] == "Cloudflare"
    assert env["RCLONE_CONFIG_FPLT_ENDPOINT"] == (
        "https://acct.r2.cloudflarestorage.com"
    )
    assert env["RCLONE_CONFIG_FPLT_ACCESS_KEY_ID"] == "AKIA"
    assert env["RCLONE_CONFIG_FPLT_SECRET_ACCESS_KEY"] == "sekrit"
    assert env["RCLONE_CONFIG_FPLT_NO_CHECK_BUCKET"] == "true"


def test_from_settings_returns_none_when_bucket_unset() -> None:
    assert Remote.from_settings(Settings()) is None
    assert Remote.from_settings(Settings(lab_bucket="fplt-lab")) is None


def test_from_settings_builds_data_and_lab_remotes() -> None:
    settings = Settings(
        r2_account_id="acct",
        r2_access_key_id="key",
        r2_secret_access_key=SecretStr("secret"),
        data_bucket="fplt-data",
        lab_bucket="fplt-lab",
    )
    data = Remote.from_settings(settings)
    assert data is not None
    assert data.target == "fplt:fplt-data"
    assert data.kind == "data"
    assert "RCLONE_CONFIG_FPLT_ACCESS_KEY_ID" in data.env

    lab = Remote.from_settings(settings, bucket="fplt-lab")
    assert lab is not None
    assert lab.target == "fplt:fplt-lab"
    assert lab.kind == "lab"


def test_from_settings_rejects_unknown_bucket_name() -> None:
    settings = Settings(data_bucket="fplt-data", lab_bucket="fplt-lab")
    with pytest.raises(ValueError, match="unknown bucket"):
        Remote.from_settings(settings, bucket="other")


def test_pull_argv_shape_and_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}

    def fake_rclone(args: list[str], *, env: dict[str, str]) -> None:
        seen["args"] = args
        seen["env"] = env
        (tmp_path / "dest" / "new.txt").parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / "dest" / "new.txt").write_text("x", encoding="utf-8")

    monkeypatch.setattr(remote_mod, "_rclone", fake_rclone)
    remote = Remote(target="fplt:fplt-data", kind="data", env={"K": "V"})
    dest = tmp_path / "dest"
    written = pull(remote, "raw", dest)
    assert seen["args"] == [
        "copy",
        "fplt:fplt-data/raw",
        str(dest),
        "--immutable",
        "--fast-list",
        "--transfers",
        "16",
    ]
    assert seen["env"] == {"K": "V"}
    assert [path.name for path in written] == ["new.txt"]
    assert "sekrit" not in " ".join(seen["args"])  # type: ignore[arg-type]


def test_push_argv_shape(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    seen: dict[str, object] = {}

    def fake_rclone(args: list[str], *, env: dict[str, str]) -> None:
        seen["args"] = args
        seen["env"] = env

    monkeypatch.setattr(remote_mod, "_rclone", fake_rclone)
    src = tmp_path / "src"
    src.mkdir()
    remote = Remote(target="fplt:fplt-lab", kind="lab", env={"E": "1"})
    push(remote, src, "runs/adhoc")
    assert seen["args"] == [
        "copy",
        str(src),
        "fplt:fplt-lab/runs/adhoc",
        "--immutable",
        "--fast-list",
    ]


def test_push_refuses_non_whitelisted_prefix() -> None:
    remote = Remote(target="fplt:fplt-data", kind="data", env={})
    with pytest.raises(ValueError, match="not pushable"):
        push(remote, Path("/tmp"), "experiments")
    lab = Remote(target="fplt:fplt-lab", kind="lab", env={})
    with pytest.raises(ValueError, match="not pushable"):
        push(lab, Path("/tmp"), "raw")
    assert "raw" in PUSHABLE["data"]
    assert "raw" not in PUSHABLE["lab"]


def test_rclone_nonzero_exit_raises_remote_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise subprocess.CalledProcessError(
            6, ["rclone", "copy"], stderr="immutable file modified"
        )

    monkeypatch.setattr(subprocess, "run", boom)
    with pytest.raises(RemoteError, match="immutable"):
        remote_mod._rclone(["copy", "a", "b"], env={})


def test_missing_rclone_binary_raises_with_install_hint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def missing(*args: object, **kwargs: object) -> None:
        raise FileNotFoundError("rclone")

    monkeypatch.setattr(subprocess, "run", missing)
    with pytest.raises(RemoteError, match="rclone is not installed"):
        remote_mod._rclone(["version"], env={})


def test_pull_propagates_rclone_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fail(args: list[str], *, env: dict[str, str]) -> None:
        raise RemoteError("failed")

    monkeypatch.setattr(remote_mod, "_rclone", fail)
    remote = Remote(target="fplt:fplt-data", kind="data", env={})
    with pytest.raises(RemoteError, match="failed"):
        pull(remote, "raw", tmp_path / "dest")


@pytest.mark.skipif(not _HAS_RCLONE, reason="rclone binary not installed")
def test_local_rclone_pull_push_immutable(tmp_path: Path) -> None:
    bucket = tmp_path / "bucket"
    raw = bucket / "raw"
    raw.mkdir(parents=True)
    (raw / "a.txt").write_text("abc", encoding="utf-8")
    remote = Remote(target=str(bucket), kind="data", env={})
    dest = tmp_path / "dest"

    first = pull(remote, "raw", dest)
    assert [path.name for path in first] == ["a.txt"]
    assert (dest / "a.txt").read_text(encoding="utf-8") == "abc"
    assert pull(remote, "raw", dest) == []

    (dest / "a.txt").write_text("changed", encoding="utf-8")
    with pytest.raises(RemoteError, match="immutable"):
        pull(remote, "raw", dest)

    src = tmp_path / "upload"
    src.mkdir()
    (src / "b.txt").write_text("new", encoding="utf-8")
    push(remote, src, "raw")
    assert (raw / "b.txt").read_text(encoding="utf-8") == "new"
    (src / "b.txt").write_text("other", encoding="utf-8")
    with pytest.raises(RemoteError, match="immutable"):
        push(remote, src, "raw")
