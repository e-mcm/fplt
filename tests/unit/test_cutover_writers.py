"""Data cut-over 2/2: no legacy plans/ or predictions/ writes (#387)."""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from fplt.automation.cycle import plan_deadline
from fplt.backtest.predict_live import run_predict
from fplt.data.store import DataStore
from fplt.models.spec import load_model_specs
from fplt.optimise.decisions import run_plan
from fplt.store.catalog import scan_succeeded_runs
from fplt.store.plan_runs import latest_plan_run_id, read_latest_plan
from fplt.types import GWKey
from tests.builders import build_league, write_test_plan_run
from tests.unit.test_plan import _score, _stage, _state

_SRC = Path(__file__).resolve().parents[2] / "src" / "fplt"
_SEASON = "2024-25"


def _stub_plan_frame(*, gw: int) -> pl.DataFrame:
    spec = load_model_specs()["rolling_mean_v1"]
    return pl.DataFrame(
        {
            "model_id": [spec.model_id],
            "as_of_season": [_SEASON],
            "as_of_gw": [gw],
            "target_gw": [gw],
            "step": [0],
            "player_key": [100],
            "position": ["MID"],
            "team_key": [3],
            "in_xi": [True],
            "captain": [True],
            "vice": [False],
            "bought": [False],
            "sold": [False],
            "xpts": [5.0],
            "hit": [0.0],
        }
    )


def test_run_plan_and_predict_write_runs_only(tmp_path: Path) -> None:
    """Live predict and plan runs land under runs/ only — no top-level archives."""
    build_league(DataStore(tmp_path))
    pred_path = run_predict(tmp_path, "rolling_mean_v1", 1, season=_SEASON, gw=2)
    assert "runs" in pred_path.parts
    assert pred_path.exists()
    assert not (tmp_path / "predictions").exists()

    plan_root = tmp_path / "plan"
    plan_root.mkdir()
    _stage(plan_root)
    _score(plan_root)
    _solved, report_path, _report, plan_id = run_plan(
        plan_root,
        "rolling_mean_v1",
        _SEASON,
        1,
        1,
        squad=_state(),
        chip_arm="off",
    )
    assert report_path.exists()
    assert "runs" in report_path.parts
    assert report_path.name == "report.md"
    assert not (plan_root / "plans").exists()

    catalog = scan_succeeded_runs(plan_root)
    assert catalog.filter(pl.col("run_type") == "plan").height == 1
    assert (
        latest_plan_run_id(
            plan_root, spec_label="rolling_mean_v1", key=GWKey(_SEASON, 1), horizon=1
        )
        == plan_id
    )


def test_replan_creates_two_plan_runs_newest_wins(tmp_path: Path) -> None:
    """Re-plan the same key twice → two plan runs; readers return the newer."""
    frame = _stub_plan_frame(gw=2)
    first_id = write_test_plan_run(tmp_path, frame, report="# one")
    second_id = write_test_plan_run(tmp_path, frame, report="# two")
    assert first_id != second_id
    assert (
        latest_plan_run_id(
            tmp_path, spec_label="rolling_mean_v1", key=GWKey(_SEASON, 2), horizon=1
        )
        == second_id
    )
    read_frame = read_latest_plan(
        tmp_path, spec_label="rolling_mean_v1", key=GWKey(_SEASON, 2), horizon=1
    )
    assert not read_frame.is_empty()
    assert not (tmp_path / "plans").exists()


def test_plan_deadline_stale_writes_new_run_fresh_skips(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T−30m refresh: stale → new plan run; fresh → none (no retire_plans)."""
    build_league(DataStore(tmp_path))
    key = GWKey(_SEASON, 2)
    now = datetime(2024, 8, 22, 16, 0, tzinfo=UTC)
    frame = _stub_plan_frame(gw=2)
    plan_writes: list[str] = []

    def _fake_play(data_dir: Path, *_a: object, **_k: object) -> None:
        plan_writes.append(
            write_test_plan_run(data_dir, frame, report=f"# {len(plan_writes)}")
        )

    monkeypatch.setattr(
        "fplt.automation.cycle.archive_if_missing", lambda *a, **k: None
    )
    monkeypatch.setattr("fplt.automation.cycle.run_backtest", lambda *a, **k: None)
    monkeypatch.setattr("fplt.automation.cycle._play", _fake_play)

    assert plan_deadline(tmp_path, key, now, ["rolling_mean_v1"], 1) is True
    first = latest_plan_run_id(
        tmp_path, spec_label="rolling_mean_v1", key=key, horizon=1
    )
    assert first is not None
    assert plan_deadline(tmp_path, key, now, ["rolling_mean_v1"], 1) is False

    monkeypatch.setattr(
        "fplt.automation.late_refresh.latest_plan_finished_at",
        lambda *_a, **_k: now - timedelta(hours=2),
    )
    store = DataStore(tmp_path)
    store.write(
        "player_status",
        pl.DataFrame(
            {
                "season": [_SEASON],
                "player_key": [100],
                "status": ["d"],
                "news": ["late"],
                "price": [100],
                "ep_next": [2.0],
                "chance_of_playing": [25.0],
                "penalties_order": [None],
                "available_at": [now - timedelta(minutes=10)],
                "source": ["fpl_api"],
            }
        ),
    )
    assert (
        plan_deadline(tmp_path, key, now, ["rolling_mean_v1"], 1, refresh_if_stale=True)
        is True
    )
    second = latest_plan_run_id(
        tmp_path, spec_label="rolling_mean_v1", key=key, horizon=1
    )
    assert second is not None
    assert second != first
    assert not (tmp_path / "plans").exists()


_ALLOWLIST = frozenset(
    {"store/legacy.py", "store/runs.py", "backtest/ledger.py"},
)
_LEGACY_PATH = re.compile(
    r"""(?:data_dir|root)\s*/\s*["'](?:plans|predictions)["']""",
)


def test_src_fplt_has_no_legacy_plans_or_predictions_writers() -> None:
    """No src/fplt module joins data_dir/root to top-level plans/ or predictions/."""
    offenders: list[str] = []
    for path in sorted(_SRC.rglob("*.py")):
        rel = path.relative_to(_SRC).as_posix()
        if rel in _ALLOWLIST:
            continue
        text = path.read_text(encoding="utf-8")
        for match in _LEGACY_PATH.finditer(text):
            line_no = text.count("\n", 0, match.start()) + 1
            offenders.append(f"{rel}:{line_no}: {match.group(0)!r}")
    assert not offenders, "legacy top-level writers remain:\n" + "\n".join(offenders)


def test_write_test_plan_run_helper_writes_run_only(tmp_path: Path) -> None:
    run_id = write_test_plan_run(tmp_path, _stub_plan_frame(gw=1), report="# hi\n")
    assert run_id
    assert not (tmp_path / "plans").exists()
    assert (
        latest_plan_run_id(
            tmp_path, spec_label="rolling_mean_v1", key=GWKey(_SEASON, 1), horizon=1
        )
        == run_id
    )
