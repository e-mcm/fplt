"""Tests for ``fplt.store.plan_runs`` (data-model cut-over readers)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fplt.models.spec import load_model_specs
from fplt.store.plan_runs import (
    latest_plan_run_id,
    plan_reports,
    plan_runs,
    read_latest_plan,
)
from fplt.store.runs import run_dir, write_plan_run
from fplt.types import GWKey

_MODEL = "rolling_mean_v1"
_KEY = GWKey("2026-27", 13)


def _plan(model_id: str, *, xpts: float = 1.0, steps: int = 1) -> pl.DataFrame:
    return pl.DataFrame(
        [
            {
                "model_id": model_id,
                "as_of_season": _KEY.season,
                "as_of_gw": _KEY.gw,
                "target_gw": _KEY.gw + step,
                "step": step,
                "player_key": 1,
                "position": "MID",
                "team_key": 1,
                "in_xi": True,
                "captain": True,
                "vice": False,
                "bought": False,
                "sold": False,
                "xpts": xpts,
                "hit": 0.0,
            }
            for step in range(steps)
        ]
    )


def _write(data_dir: Path, *, xpts: float, report: str | None, steps: int = 1) -> str:
    spec = load_model_specs()[_MODEL]
    manifest = write_plan_run(
        data_dir,
        spec=spec,
        as_of=_KEY,
        deadline_utc=datetime(2026, 12, 1, 18, 30, tzinfo=UTC),
        horizon=steps,
        frame=_plan(spec.model_id, xpts=xpts, steps=steps),
        report_md=report,
    )
    return manifest.run_id


def test_latest_plan_is_newest_run_not_legacy_path(tmp_path: Path) -> None:
    """Two plan runs for one key: readers get the newest (never a stale file)."""
    first = _write(tmp_path, xpts=1.0, report="# old")
    second = _write(tmp_path, xpts=9.0, report="# new")
    rows = plan_runs(tmp_path, spec_label=_MODEL, key=_KEY, horizon=1)
    assert rows.get_column("run_id").to_list() == [first, second]
    assert (
        latest_plan_run_id(tmp_path, spec_label=_MODEL, key=_KEY, horizon=1) == second
    )
    plan = read_latest_plan(tmp_path, spec_label=_MODEL, key=_KEY, horizon=1)
    assert plan["xpts"].to_list() == [9.0]
    assert (run_dir(tmp_path, second) / "report.md").read_text(encoding="utf-8") == (
        "# new"
    )
    assert plan_reports(tmp_path) == "# old\n# new"
    # No legacy plans/ folder is involved.
    assert not (tmp_path / "plans").exists()


def test_plan_runs_filters_horizon_spec_and_key(tmp_path: Path) -> None:
    _write(tmp_path, xpts=1.0, report=None, steps=3)
    assert plan_runs(tmp_path, spec_label=_MODEL, key=_KEY, horizon=1).is_empty()
    assert plan_runs(tmp_path, spec_label=_MODEL, key=_KEY, horizon=3).height == 1
    assert plan_runs(tmp_path, spec_label="other", key=_KEY, horizon=3).is_empty()
    assert plan_runs(
        tmp_path, spec_label=_MODEL, key=GWKey("2026-27", 12), horizon=3
    ).is_empty()
    assert plan_runs(tmp_path, spec_label=_MODEL, key=_KEY, horizon=0).is_empty()
    # A run without report.md contributes nothing to plan_reports.
    assert plan_reports(tmp_path) == ""


def test_missing_plan_run_raises_and_empty_dir(tmp_path: Path) -> None:
    assert latest_plan_run_id(tmp_path, spec_label=_MODEL, key=_KEY, horizon=1) is None
    assert plan_reports(tmp_path) == ""
    with pytest.raises(FileNotFoundError, match="no plan run"):
        read_latest_plan(tmp_path, spec_label=_MODEL, key=_KEY, horizon=1)
