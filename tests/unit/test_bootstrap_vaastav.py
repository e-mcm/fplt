"""Bootstrap places vaastav season dirs where import_available expects them."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path
from typing import Any

import pytest

_PATH = Path(__file__).parents[2] / "scripts" / "bootstrap_vaastav.py"
_SPEC = importlib.util.spec_from_file_location("fplt_bootstrap_vaastav", _PATH)
assert _SPEC is not None and _SPEC.loader is not None
bootstrap_vaastav = importlib.util.module_from_spec(_SPEC)
sys.modules["fplt_bootstrap_vaastav"] = bootstrap_vaastav
_SPEC.loader.exec_module(bootstrap_vaastav)


def _season(root: Path, name: str, *, with_gw: bool = True) -> Path:
    season = root / name
    gws = season / "gws"
    gws.mkdir(parents=True)
    if with_gw:
        (gws / "merged_gw.csv").write_text("element,GW\n", encoding="utf-8")
    (season / "players_raw.csv").write_text("id,code\n", encoding="utf-8")
    return season


def test_is_season_dir_requires_merged_gw(tmp_path: Path) -> None:
    good = _season(tmp_path, "2024-25")
    bad = _season(tmp_path, "2023-24", with_gw=False)
    early = _season(tmp_path, "2020-21")
    assert bootstrap_vaastav.is_season_dir(good)
    assert not bootstrap_vaastav.is_season_dir(bad)
    assert not bootstrap_vaastav.is_season_dir(early)
    assert not bootstrap_vaastav.is_season_dir(tmp_path / "missing")


def test_list_seasons_filters_and_sorts(tmp_path: Path) -> None:
    _season(tmp_path, "2020-21")
    _season(tmp_path, "2023-24")
    _season(tmp_path, "2021-22")
    _season(tmp_path, "2022-23", with_gw=False)
    names = [path.name for path in bootstrap_vaastav.list_seasons(tmp_path)]
    assert names == ["2021-22", "2023-24"]
    assert bootstrap_vaastav.list_seasons(tmp_path / "absent") == []


def test_place_seasons_copies_and_skips_existing(tmp_path: Path) -> None:
    source = tmp_path / "source"
    dest = tmp_path / "data"
    _season(source, "2021-22")
    _season(source, "2024-25")
    first = bootstrap_vaastav.place_seasons(source, dest)
    assert first == ["2021-22", "2024-25"]
    assert (dest / "2021-22" / "gws" / "merged_gw.csv").is_file()
    (dest / "2021-22" / "marker.txt").write_text("keep", encoding="utf-8")
    second = bootstrap_vaastav.place_seasons(source, dest)
    assert second == []
    assert (dest / "2021-22" / "marker.txt").read_text(encoding="utf-8") == "keep"
    refreshed = bootstrap_vaastav.place_seasons(source, dest, force=True)
    assert refreshed == ["2021-22", "2024-25"]
    assert not (dest / "2021-22" / "marker.txt").exists()


def test_default_data_dir_reads_env(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.delenv("FPLT_DATA_DIR", raising=False)
    assert bootstrap_vaastav.default_data_dir() == Path("./data")
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path / "custom"))
    assert bootstrap_vaastav.default_data_dir() == tmp_path / "custom"


def test_bootstrap_with_source_skips_git(tmp_path: Path) -> None:
    source = tmp_path / "vaastav_data"
    dest = tmp_path / "data"
    _season(source, "2022-23")
    placed = bootstrap_vaastav.bootstrap(dest, source=source)
    assert placed == ["2022-23"]
    assert (dest / "2022-23" / "gws" / "merged_gw.csv").is_file()
    assert not (dest / ".vaastav").exists()


def test_fetch_vaastav_pulls_existing_cache(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / ".vaastav"
    (cache / "data").mkdir(parents=True)
    (cache / ".git").mkdir()
    calls: list[tuple[list[str], Path]] = []

    def fake_run(args: list[str], cwd: Path) -> None:
        calls.append((args, cwd))

    monkeypatch.setattr(bootstrap_vaastav, "_run_git", fake_run)
    result = bootstrap_vaastav.fetch_vaastav(cache)
    assert result == cache / "data"
    assert calls == [(["pull", "--ff-only", "origin", "main"], cache)]


def test_fetch_vaastav_clones_when_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    cache = tmp_path / ".vaastav"
    calls: list[tuple[list[str], Path]] = []

    def fake_run(args: list[str], cwd: Path) -> None:
        calls.append((args, cwd))
        if args[0] == "clone":
            staging = cwd / args[-1]
            (staging / "data" / "2024-25" / "gws").mkdir(parents=True)
            (staging / ".git").mkdir()

    monkeypatch.setattr(bootstrap_vaastav, "_run_git", fake_run)
    result = bootstrap_vaastav.fetch_vaastav(
        cache, repo_url="https://example.test/repo.git"
    )
    assert result == cache / "data"
    assert (cache / "data" / "2024-25").is_dir()
    assert calls[0][0][:4] == ["clone", "--depth=1", "--filter=blob:none", "--sparse"]
    assert calls[1][0] == ["sparse-checkout", "set", "data"]


def test_run_git_raises_on_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    class Result:
        returncode = 1
        stderr = "boom"
        stdout = ""

    monkeypatch.setattr(
        bootstrap_vaastav.subprocess,
        "run",
        lambda *args, **kwargs: Result(),
    )
    with pytest.raises(RuntimeError, match="boom"):
        bootstrap_vaastav._run_git(["status"], cwd=tmp_path)


def test_main_with_source_exits_zero(tmp_path: Path, capsys: Any) -> None:
    source = tmp_path / "source"
    dest = tmp_path / "data"
    _season(source, "2021-22")
    code = bootstrap_vaastav.main(["--data-dir", str(dest), "--source", str(source)])
    assert code == 0
    out = capsys.readouterr().out
    assert "2021-22" in out
    assert (dest / "2021-22" / "gws" / "merged_gw.csv").is_file()


def test_main_reports_nothing_copied(tmp_path: Path, capsys: Any) -> None:
    source = tmp_path / "source"
    dest = tmp_path / "data"
    source.mkdir()
    code = bootstrap_vaastav.main(["--data-dir", str(dest), "--source", str(source)])
    assert code == 0
    assert "no seasons copied" in capsys.readouterr().out


def test_main_returns_one_on_runtime_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*args: object, **kwargs: object) -> list[str]:
        raise RuntimeError("git failed")

    monkeypatch.setattr(bootstrap_vaastav, "bootstrap", boom)
    code = bootstrap_vaastav.main(["--data-dir", str(tmp_path)])
    assert code == 1


def test_main_uses_env_data_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = tmp_path / "source"
    dest = tmp_path / "from_env"
    _season(source, "2023-24")
    monkeypatch.setenv("FPLT_DATA_DIR", str(dest))
    code = bootstrap_vaastav.main(["--source", str(source)])
    assert code == 0
    assert (dest / "2023-24" / "gws" / "merged_gw.csv").is_file()
    # Avoid leaking into other tests if the fixture order changes.
    assert os.environ["FPLT_DATA_DIR"] == str(dest)
