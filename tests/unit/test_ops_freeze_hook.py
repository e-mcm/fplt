"""ops-04 bridge: fplt.ops.freeze.in_prod_freeze for Renovate / ops-12."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from fplt.ops.freeze import in_prod_freeze

_SEASON = "2025-26"
_DEADLINE = datetime(2025, 11, 29, 11, 0, tzinfo=UTC)
_KICKOFF = datetime(2025, 11, 29, 12, 30, tzinfo=UTC)


def _stage(tmp_path: Path) -> None:
    # Minimal columns for freeze_windows; DataStore.write validates full schema,
    # so write parquet directly under staged/ (read widens missing columns).
    gw = pl.DataFrame(
        {
            "season": [_SEASON],
            "gw": [13],
            "deadline_time": [_DEADLINE],
        }
    )
    fx = pl.DataFrame(
        {
            "season": [_SEASON],
            "gw": [13],
            "kickoff_time": [_KICKOFF],
        }
    )
    (tmp_path / "staged" / "gameweeks").mkdir(parents=True)
    (tmp_path / "staged" / "fixtures").mkdir(parents=True)
    gw.write_parquet(tmp_path / "staged" / "gameweeks" / "part.parquet")
    fx.write_parquet(tmp_path / "staged" / "fixtures" / "part.parquet")


def test_in_prod_freeze_false_without_staged(tmp_path: Path) -> None:
    frozen, detail = in_prod_freeze(data_dir=tmp_path)
    assert frozen is False
    assert detail == "outside freeze"


def test_in_prod_freeze_true_inside_window(tmp_path: Path) -> None:
    _stage(tmp_path)
    frozen, detail = in_prod_freeze(
        now=_DEADLINE - timedelta(hours=1),
        data_dir=tmp_path,
    )
    assert frozen is True
    assert "inside freeze" in detail or "GW13" in detail


def test_in_prod_freeze_false_after_kickoff(tmp_path: Path) -> None:
    _stage(tmp_path)
    frozen, detail = in_prod_freeze(now=_KICKOFF, data_dir=tmp_path)
    assert frozen is False
    assert detail == "outside freeze"
