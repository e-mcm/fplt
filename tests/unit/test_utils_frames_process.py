"""Tests for assert→exception and resolved-subprocess helpers (#311)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from fplt.utils.frames import require_dataframe, require_not_none
from fplt.utils.process import resolve_executable, run_command


def test_require_dataframe_accepts_frame() -> None:
    frame = pl.DataFrame({"a": [1]})
    assert require_dataframe(frame) is frame


def test_require_dataframe_rejects_other() -> None:
    with pytest.raises(TypeError, match="expected DataFrame"):
        require_dataframe([1, 2, 3], what="demo.validate")


def test_require_not_none_accepts_value() -> None:
    assert require_not_none(7, what="x") == 7


def test_require_not_none_rejects_none() -> None:
    with pytest.raises(ValueError, match="x is required"):
        require_not_none(None, what="x")


def test_resolve_executable_finds_python() -> None:
    path = resolve_executable("python3")
    assert Path(path).is_file()


def test_resolve_executable_missing() -> None:
    with pytest.raises(FileNotFoundError, match="not-a-real-bin-xyz"):
        resolve_executable("not-a-real-bin-xyz")


def test_run_command_resolves_and_runs() -> None:
    completed = run_command(
        ["python3", "-c", "print('ok')"], capture_output=True, text=True
    )
    assert completed.returncode == 0
    assert "ok" in (completed.stdout or "")


def test_run_command_empty_argv() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        run_command([])


def test_run_command_missing_executable() -> None:
    with (
        patch("fplt.utils.process.shutil.which", return_value=None),
        pytest.raises(FileNotFoundError, match="missing-tool"),
    ):
        run_command(["missing-tool"])
