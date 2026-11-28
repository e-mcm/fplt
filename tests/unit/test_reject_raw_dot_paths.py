"""Regression tests for raw-tree '.'-component rejection (#339)."""

from __future__ import annotations

from pathlib import Path
from typing import Any

from scripts.reject_raw_dot_paths import (
    find_dot_component_paths,
    main,
    reject_raw_dot_paths,
)


def test_clean_tree_is_accepted(tmp_path: Path) -> None:
    """A normal capture layout has no '.'-prefixed components."""
    dest = tmp_path / "raw" / "vaastav" / "captured_at=2026-11-28T00-00-00Z"
    dest.mkdir(parents=True)
    (dest / "_source.json").write_text("{}\n", encoding="utf-8")
    (dest / "data").mkdir()
    (dest / "data" / "merged_gw.csv").write_text("x\n", encoding="utf-8")
    assert reject_raw_dot_paths(tmp_path / "raw") == []
    assert main([str(tmp_path / "raw")]) == 0


def test_dot_dir_is_rejected(tmp_path: Path, capsys: Any) -> None:
    """A source tree containing ``.foo/bar`` exits non-zero with ``::error::``."""
    bad = tmp_path / "raw" / ".foo"
    bad.mkdir(parents=True)
    (bad / "bar").write_text("nope\n", encoding="utf-8")
    (tmp_path / "raw" / "ok.txt").write_text("ok\n", encoding="utf-8")
    found = find_dot_component_paths(tmp_path / "raw")
    assert ".foo" in found
    assert ".foo/bar" in found
    assert reject_raw_dot_paths(tmp_path / "raw") == found
    assert main([str(tmp_path / "raw")]) == 1
    err = capsys.readouterr().err
    assert "::error::" in err
    assert ".foo" in err


def test_nested_dotfile_is_rejected(tmp_path: Path) -> None:
    """Hidden files under an otherwise clean tree are also rejected."""
    base = tmp_path / "raw" / "vaastav" / "captured_at=x"
    base.mkdir(parents=True)
    (base / ".hidden").write_text("x\n", encoding="utf-8")
    assert any(p.endswith(".hidden") for p in reject_raw_dot_paths(tmp_path / "raw"))


def test_missing_root_is_clean(tmp_path: Path) -> None:
    """A missing root is treated as empty (nothing to reject)."""
    assert reject_raw_dot_paths(tmp_path / "absent") == []
