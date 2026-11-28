"""Vaastav capture writes write-once raw trees against a local git fixture."""

from __future__ import annotations

import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pytest

from fplt.schemas.vaastav_source import VaastavSourceManifest
from scripts import bootstrap_vaastav

capture_vaastav = bootstrap_vaastav.capture_vaastav
default_vaastav_cache_dir = bootstrap_vaastav.default_vaastav_cache_dir
fetch_vaastav_repo = bootstrap_vaastav.fetch_vaastav_repo
is_season_dir = bootstrap_vaastav.is_season_dir
list_season_dirs = bootstrap_vaastav.list_season_dirs
load_prior_file_hashes = bootstrap_vaastav.load_prior_file_hashes
sha256_file = bootstrap_vaastav.sha256_file


def _git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", *args],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )


def _init_repo(root: Path) -> Path:
    repo = root / "vaastav"
    repo.mkdir()
    _git(repo, "init", "-b", "main")
    _git(repo, "config", "user.email", "test@example.com")
    _git(repo, "config", "user.name", "Test")
    return repo


def _season(data: Path, name: str, body: str = "element,GW\n1,1\n") -> Path:
    season = data / name
    gws = season / "gws"
    gws.mkdir(parents=True)
    (gws / "merged_gw.csv").write_text(body, encoding="utf-8")
    (season / "players_raw.csv").write_text("id,code\n1,100\n", encoding="utf-8")
    return season


def _commit_all(repo: Path, message: str) -> str:
    _git(repo, "add", "-A")
    _git(repo, "commit", "-m", message)
    return (
        subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=repo,
            check=True,
            capture_output=True,
            text=True,
        )
        .stdout.strip()
        .lower()
    )


def _fake_vaastav(tmp_path: Path) -> Path:
    repo = _init_repo(tmp_path)
    data = repo / "data"
    data.mkdir()
    (data / "DATA_DICTIONARY.md").write_text("# dict\n", encoding="utf-8")
    _season(data, "2021-22", "element,GW\n1,1\n")
    _season(data, "2024-25", "element,GW\n2,1\n")
    _season(data, "2025-26", "element,GW\n3,1\n")
    _commit_all(repo, "seed")
    return repo


def test_is_season_dir_and_list(tmp_path: Path) -> None:
    data = tmp_path / "data"
    data.mkdir()
    _season(data, "2020-21")
    _season(data, "2024-25")
    (data / "notes").mkdir()
    assert is_season_dir(data / "2024-25")
    assert not is_season_dir(data / "notes")
    assert [p.name for p in list_season_dirs(data)] == ["2020-21", "2024-25"]


def test_full_capture_writes_layout_and_manifest(tmp_path: Path) -> None:
    repo = _fake_vaastav(tmp_path)
    raw = tmp_path / "raw"
    stamp = datetime(2026, 11, 27, 12, 0, tzinfo=UTC)
    written = capture_vaastav(
        raw,
        repo_path=repo,
        full=True,
        captured_at=stamp,
    )
    assert written is not None
    assert written.name == "captured_at=2026-11-27T12-00-00Z"
    assert (written / "data" / "2021-22" / "gws" / "merged_gw.csv").is_file()
    assert (written / "data" / "DATA_DICTIONARY.md").is_file()
    manifest = VaastavSourceManifest.model_validate(
        json.loads((written / "_source.json").read_text(encoding="utf-8"))
    )
    assert manifest.seasons == ["2021-22", "2024-25", "2025-26"]
    assert len(manifest.commit_sha) == 40
    assert any(f.path.endswith("merged_gw.csv") for f in manifest.files)
    assert load_prior_file_hashes(raw)


def test_unchanged_rerun_writes_nothing(tmp_path: Path) -> None:
    repo = _fake_vaastav(tmp_path)
    raw = tmp_path / "raw"
    first = capture_vaastav(
        raw,
        repo_path=repo,
        full=True,
        captured_at=datetime(2026, 11, 27, 12, 0, tzinfo=UTC),
    )
    assert first is not None
    second = capture_vaastav(
        raw,
        repo_path=repo,
        full=True,
        captured_at=datetime(2026, 11, 27, 13, 0, tzinfo=UTC),
    )
    assert second is None
    assert list((raw / "vaastav").iterdir()) == [first]


def test_changed_current_season_only(tmp_path: Path) -> None:
    repo = _fake_vaastav(tmp_path)
    raw = tmp_path / "raw"
    capture_vaastav(
        raw,
        repo_path=repo,
        full=True,
        captured_at=datetime(2026, 11, 27, 12, 0, tzinfo=UTC),
    )
    current = repo / "data" / "2025-26" / "gws" / "merged_gw.csv"
    current.write_text("element,GW\n3,2\n", encoding="utf-8")
    _commit_all(repo, "bump current")
    written = capture_vaastav(
        raw,
        repo_path=repo,
        full=False,
        captured_at=datetime(2026, 11, 28, 12, 0, tzinfo=UTC),
    )
    assert written is not None
    assert (written / "data" / "2025-26" / "gws" / "merged_gw.csv").is_file()
    assert not (written / "data" / "2021-22").exists()
    assert not (written / "data" / "2024-25").exists()
    manifest = VaastavSourceManifest.model_validate(
        json.loads((written / "_source.json").read_text(encoding="utf-8"))
    )
    assert manifest.seasons == ["2025-26"]
    # Digests for older seasons still come from the first capture.
    prior = load_prior_file_hashes(raw)
    old = (
        raw
        / "vaastav"
        / "captured_at=2026-11-27T12-00-00Z"
        / "data"
        / "2021-22"
        / "gws"
        / "merged_gw.csv"
    )
    assert prior["data/2021-22/gws/merged_gw.csv"] == sha256_file(old)


def test_fetch_vaastav_repo_pulls_and_clones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / ".cache"
    (cache / "data").mkdir(parents=True)
    (cache / ".git").mkdir()
    calls: list[tuple[list[str], Path]] = []

    def fake_run(args: list[str], cwd: Path) -> str:
        calls.append((args, cwd))
        if args[:2] == ["rev-parse", "HEAD"]:
            return "abc1234"
        return ""

    monkeypatch.setattr(bootstrap_vaastav, "_run_git", fake_run)
    assert fetch_vaastav_repo(cache) == cache
    assert calls == [(["pull", "--ff-only", "origin", "main"], cache)]

    cache2 = tmp_path / ".cache2"

    def fake_clone(args: list[str], cwd: Path) -> str:
        calls.append((args, cwd))
        if args[0] == "clone":
            staging = cwd / args[-1]
            (staging / "data").mkdir(parents=True)
            (staging / ".git").mkdir()
        return ""

    calls.clear()
    monkeypatch.setattr(bootstrap_vaastav, "_run_git", fake_clone)
    got = fetch_vaastav_repo(cache2, repo_url="https://example.test/repo.git")
    assert got == cache2
    assert calls[0][0][:4] == ["clone", "--depth=1", "--filter=blob:none", "--sparse"]


def test_main_with_local_repo(tmp_path: Path, capsys: Any) -> None:
    repo = _fake_vaastav(tmp_path)
    out = tmp_path / "raw"
    code = bootstrap_vaastav.main(
        [
            "--out",
            str(out),
            "--repo",
            str(repo),
            "--full",
            "--captured-at",
            "2026-11-27T15-00-00Z",
        ]
    )
    assert code == 0
    assert "wrote" in capsys.readouterr().out
    assert (
        out / "vaastav" / "captured_at=2026-11-27T15-00-00Z" / "_source.json"
    ).is_file()


def test_main_nothing_new(tmp_path: Path, capsys: Any) -> None:
    repo = _fake_vaastav(tmp_path)
    out = tmp_path / "raw"
    assert (
        bootstrap_vaastav.main(
            [
                "--out",
                str(out),
                "--repo",
                str(repo),
                "--full",
                "--captured-at",
                "2026-11-27T15-00-00Z",
            ]
        )
        == 0
    )
    code = bootstrap_vaastav.main(
        [
            "--out",
            str(out),
            "--repo",
            str(repo),
            "--full",
            "--captured-at",
            "2026-11-27T16-00-00Z",
        ]
    )
    assert code == 0
    assert "nothing new" in capsys.readouterr().out


def test_main_invalid_captured_at(tmp_path: Path) -> None:
    code = bootstrap_vaastav.main(
        ["--out", str(tmp_path), "--captured-at", "not-a-stamp"]
    )
    assert code == 1


def test_default_raw_dir(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.delenv("FPLT_DATA_DIR", raising=False)
    assert bootstrap_vaastav.default_raw_dir() == Path("./data/raw")
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path / "custom"))
    assert bootstrap_vaastav.default_raw_dir() == tmp_path / "custom" / "raw"


def test_default_vaastav_cache_dir_is_sibling_not_child(tmp_path: Path) -> None:
    """Library default keeps the clone outside the raw root (#339)."""
    raw = tmp_path / "raw"
    cache = default_vaastav_cache_dir(raw)
    assert cache == (tmp_path / "vaastav_cache").resolve()
    assert raw.resolve() not in cache.parents
    assert cache != raw.resolve()


def test_capture_default_cache_leaves_no_dot_entries_under_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Capture with ``--out X`` must not leave any '.' entries under ``X``."""
    upstream = tmp_path / "upstream"
    upstream.mkdir()
    repo = _fake_vaastav(upstream)
    out = tmp_path / "out" / "raw"

    def fake_fetch(cache_dir: Path, *, repo_url: str = "") -> Path:
        assert out.resolve() not in cache_dir.resolve().parents
        assert cache_dir.resolve() != out.resolve()
        if cache_dir.exists():
            shutil.rmtree(cache_dir)
        shutil.copytree(repo, cache_dir)
        return cache_dir

    monkeypatch.setattr(bootstrap_vaastav, "fetch_vaastav_repo", fake_fetch)
    written = capture_vaastav(
        out,
        full=True,
        captured_at=datetime(2026, 11, 28, 12, 0, tzinfo=UTC),
    )
    assert written is not None
    assert written.parent == out / "vaastav"
    # Only vaastav/captured_at=*/… under out; no '.'-prefixed names anywhere.
    names = {p.relative_to(out).as_posix() for p in out.rglob("*")}
    assert all(not any(part.startswith(".") for part in Path(n).parts) for n in names)
    top = sorted(p.name for p in out.iterdir())
    assert top == ["vaastav"]
    assert (tmp_path / "out" / "vaastav_cache").is_dir()


def test_cache_dir_under_raw_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Explicit cache under the raw root is rejected."""
    out = tmp_path / "raw"
    monkeypatch.setattr(
        bootstrap_vaastav,
        "fetch_vaastav_repo",
        lambda cache_dir, **kwargs: cache_dir,
    )
    with pytest.raises(ValueError, match="must not be under raw root"):
        capture_vaastav(out, cache_dir=out / ".vaastav_cache")


def test_main_upstream_unreachable_counts_then_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> Path | None:
        raise RuntimeError("git failed")

    monkeypatch.setattr(bootstrap_vaastav, "capture_vaastav", boom)
    status_dir = tmp_path / "status"
    args = [
        "--out",
        str(tmp_path / "raw"),
        "--status-dir",
        str(status_dir),
        "--unavailable-threshold",
        "3",
    ]
    assert bootstrap_vaastav.main(args) == 0
    assert bootstrap_vaastav.main(args) == 0
    assert bootstrap_vaastav.main(args) == 1
    state = json.loads(
        (status_dir / "vaastav_unavailable.json").read_text(encoding="utf-8")
    )
    assert state["consecutive_unavailable"] == 3


def test_load_prior_file_hashes_skips_incomplete_capture(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Partial capture with ``_source.json`` but a missing file is ignored (#353)."""
    import logging

    repo = _fake_vaastav(tmp_path)
    raw = tmp_path / "raw"
    written = capture_vaastav(
        raw,
        repo_path=repo,
        full=True,
        captured_at=datetime(2026, 11, 27, 12, 0, tzinfo=UTC),
    )
    assert written is not None
    # Drop a listed payload file so the capture looks like a timed-out upload.
    victim = next(written.joinpath("data").rglob("merged_gw.csv"))
    victim.unlink()
    with caplog.at_level(logging.WARNING):
        prior = load_prior_file_hashes(raw)
    assert prior == {}
    assert "missing" in caplog.text
    # Next full capture must rewrite seasons rather than treat them as known.
    again = capture_vaastav(
        raw,
        repo_path=repo,
        full=True,
        captured_at=datetime(2026, 11, 27, 13, 0, tzinfo=UTC),
    )
    assert again is not None
    assert load_prior_file_hashes(raw)
