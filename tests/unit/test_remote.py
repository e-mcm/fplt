"""Unit and local-dir integration tests for ``fplt.data.remote``."""

from __future__ import annotations

import logging
import shutil
import subprocess
from pathlib import Path

import pytest
from pydantic import SecretStr

from fplt.data import remote as remote_mod
from fplt.data.remote import PUSHABLE, Remote, RemoteError, pull, push, push_replace
from fplt.settings import Settings, get_settings
from fplt.utils.timing import (
    TimingSession,
    classify_phase,
    load_timing_records,
    timed_phase,
    timing_artifacts,
    timing_root,
    timing_session,
)

_HAS_RCLONE = shutil.which("rclone") is not None


@pytest.fixture(autouse=True)
def _isolate_timing_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep remote timing artifacts out of the repo ``./data`` tree."""
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path / "timing-root"))
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


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


def test_from_settings_returns_none_when_credentials_incomplete(
    caplog: pytest.LogCaptureFixture,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Bucket name alone must soft-skip off main; empty account yields ``.r2…``."""
    # Unit tests must not inherit GITHUB_REF=main from a push-to-main GHA job.
    monkeypatch.delenv("GITHUB_REF", raising=False)
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    settings = Settings(data_bucket="fplt-data", lab_bucket="fplt-lab")
    with caplog.at_level(logging.INFO, logger="fplt.data.remote"):
        assert Remote.from_settings(settings) is None
        assert Remote.from_settings(settings, bucket="fplt-lab") is None
    assert "soft skip" in caplog.text
    assert "credentials incomplete" in caplog.text

    partial = Settings(
        data_bucket="fplt-data",
        r2_account_id="acct",
        r2_access_key_id="key",
        # secret missing
    )
    assert Remote.from_settings(partial) is None
    assert not remote_mod.r2_credentials_complete(partial)


def test_from_settings_raises_on_main_when_credentials_incomplete(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """GITHUB_REF=main with a configured bucket must not soft-skip (#278)."""
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    settings = Settings(data_bucket="fplt-data")
    with pytest.raises(RemoteError, match="credentials incomplete on main"):
        Remote.from_settings(settings)


def test_from_settings_soft_skips_off_main_and_on_pull_request(
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """Non-main refs and pull_request keep soft-skip when secrets are empty."""
    settings = Settings(data_bucket="fplt-data")
    monkeypatch.setenv("GITHUB_REF", "refs/heads/cursor/feature")
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    with caplog.at_level(logging.INFO, logger="fplt.data.remote"):
        assert Remote.from_settings(settings) is None
    assert "soft skip" in caplog.text

    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_EVENT_NAME", "pull_request")
    assert Remote.from_settings(settings) is None
    assert remote_mod.github_requires_r2() is False


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


def test_push_replace_status_omits_immutable(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    seen: dict[str, object] = {}

    def fake_rclone(args: list[str], *, env: dict[str, str]) -> None:
        seen["args"] = args

    monkeypatch.setattr(remote_mod, "_rclone", fake_rclone)
    src = tmp_path / "status.json"
    src.write_text("{}", encoding="utf-8")
    remote = Remote(target="fplt:fplt-data", kind="data", env={})
    push_replace(remote, src, "status/status.json")
    assert seen["args"] == [
        "copyto",
        str(src),
        "fplt:fplt-data/status/status.json",
        "--fast-list",
    ]
    assert "--immutable" not in seen["args"]  # type: ignore[operator]


def test_push_replace_refuses_immutable_prefixes(tmp_path: Path) -> None:
    src = tmp_path / "x.json"
    src.write_text("{}", encoding="utf-8")
    remote = Remote(target="fplt:fplt-data", kind="data", env={})
    with pytest.raises(ValueError, match="mutable rewrite"):
        push_replace(remote, src, "raw/foo.json")
    lab = Remote(target="fplt:fplt-lab", kind="lab", env={})
    with pytest.raises(ValueError, match="mutable rewrite"):
        push_replace(lab, src, "status/status.json")


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


def test_pull_emits_remote_timing_artifact(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    def fake_rclone(args: list[str], *, env: dict[str, str]) -> None:
        dest = Path(args[2])
        dest.mkdir(parents=True, exist_ok=True)
        (dest / "a.bin").write_bytes(b"hello")

    monkeypatch.setattr(remote_mod, "_rclone", fake_rclone)
    remote = Remote(target="fplt:fplt-data", kind="data", env={})
    dest = tmp_path / "dest"
    with caplog.at_level(logging.INFO, logger="fplt.data.remote"):
        written = pull(remote, "raw", dest)
    assert [path.name for path in written] == ["a.bin"]
    records = load_timing_records(get_settings().data_dir, kind="remote")
    assert len(records) == 1
    record = records[0]
    assert record["kind"] == "remote"
    assert "remote/pull" in record["phases"]
    assert record["meta"]["op"] == "pull"
    assert record["meta"]["prefix"] == "raw"
    assert record["meta"]["n_objects"] == 1
    assert record["meta"]["bytes"] == 5
    assert "remote pull prefix=raw" in caplog.text
    assert classify_phase("remote/pull") == "io"


def test_pull_records_into_active_timing_session(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    def fake_rclone(args: list[str], *, env: dict[str, str]) -> None:
        return None

    monkeypatch.setattr(remote_mod, "_rclone", fake_rclone)
    remote = Remote(target="fplt:fplt-data", kind="data", env={})
    bag = TimingSession()
    with timing_artifacts(tmp_path / "run-root"):
        with timing_session(bag):
            with timed_phase("setup"):
                pull(remote, "raw", tmp_path / "empty-dest")
    assert bag.get("remote/pull") > 0
    assert bag.get("setup") >= bag.get("remote/pull")
    nested = list((timing_root(tmp_path / "run-root")).glob("remote_*.json"))
    assert len(nested) == 1


def test_push_emits_remote_timing_with_source_bytes(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(remote_mod, "_rclone", lambda args, *, env: None)
    src = tmp_path / "src"
    src.mkdir()
    (src / "x.txt").write_text("abcd", encoding="utf-8")
    remote = Remote(target="fplt:fplt-lab", kind="lab", env={})
    push(remote, src, "runs/adhoc")
    records = load_timing_records(get_settings().data_dir, kind="remote")
    assert len(records) == 1
    assert records[0]["meta"]["op"] == "push"
    assert records[0]["meta"]["bytes"] == 4
    assert records[0]["meta"]["n_objects"] == 1
    assert "remote/push" in records[0]["phases"]


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
