"""Phase timing aggregation for walk-forward wall-clock logs."""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path

import pytest

from fplt.backtest.walk_forward import walk_forward
from fplt.data.pit import PITData, PITView
from fplt.data.store import DataStore
from fplt.models.baselines import RollingMeanPredictor
from fplt.models.spec import ModelSpec
from fplt.types import GWKey
from fplt.utils.timing import (
    TimingSession,
    active_timing_dir,
    binds_timing_artifacts,
    build_timing_record,
    classify_phase,
    format_gw_breakdown,
    format_hotspot_report,
    format_kind_summary,
    format_run_summary,
    load_timing_records,
    rank_hotspots,
    timed_phase,
    timing_artifacts,
    timing_root,
    timing_session,
    write_timing_record,
)
from tests.builders import build_league


def test_timing_session_merges_and_shares() -> None:
    first = TimingSession()
    first.add("fit", 2.0)
    first.add("predict", 8.0)
    first.add("fit/score", 1.5)
    second = TimingSession()
    second.add("fit", 1.0)
    second.add("predict", 4.0)
    second.add("write", 0.5)
    second.add("fit/score", 0.5)
    first.merge(second)
    assert first.get("fit") == 3.0
    assert first.get("predict") == 12.0
    assert first.get("write") == 0.5
    assert first.total() == 15.5
    assert first.nested() == {"fit/score": 2.0}
    summary = format_run_summary("demo", first, n_gw=2)
    assert "walk_forward timing model=demo gws=2 total=15.5s" in summary
    assert "fit=3.0s (19%)" in summary
    assert "predict=12.0s (77%)" in summary
    assert "write=0.5s (3%)" in summary
    assert "| fit/score=2.0s" in summary
    assert format_gw_breakdown(first) == "fit=3.0s predict=12.0s write=0.5s"


def test_timed_phase_records_only_with_active_session() -> None:
    orphan = TimingSession()
    with timed_phase("fit"):
        time.sleep(0.01)
    assert orphan.seconds == {}
    with timing_session(orphan):
        with timed_phase("fit"):
            time.sleep(0.01)
        with timed_phase("predict"):
            time.sleep(0.01)
        with timed_phase("fit/score"):
            time.sleep(0.01)
    assert orphan.get("fit") > 0
    assert orphan.get("predict") > 0
    assert orphan.get("fit/score") > 0


def test_timing_artifacts_context_and_decorator(tmp_path: Path) -> None:
    session = TimingSession()
    session.add("predict", 1.0)
    assert write_timing_record("walk_forward", session) is None
    assert active_timing_dir() is None
    with timing_artifacts(tmp_path):
        assert active_timing_dir() == tmp_path
        path = write_timing_record("walk_forward", session, meta={"model": "ctx"})
    assert active_timing_dir() is None
    assert path is not None
    assert path.parent == timing_root(tmp_path)

    @binds_timing_artifacts()
    def _entry(data_dir: Path) -> Path | None:
        bag = TimingSession()
        bag.add("fit", 2.0)
        return write_timing_record("plan", bag, meta={"model": "dec"})

    written = _entry(tmp_path)
    assert written is not None
    assert written.parent == timing_root(tmp_path)


def test_build_and_write_timing_record(tmp_path: Path) -> None:
    session = TimingSession()
    session.add("fit", 2.0)
    session.add("predict", 8.0)
    session.add("predict/fixtures", 7.5)
    record = build_timing_record(
        "walk_forward",
        session,
        meta={"model": "demo", "n_gw": 2},
    )
    assert record["schema_version"] == 1
    assert record["kind"] == "walk_forward"
    assert record["total_s"] == 10.0
    assert record["phases"]["predict/fixtures"] == 7.5
    assert "recorded_at" in record
    empty = TimingSession()
    assert write_timing_record("walk_forward", empty, data_dir=tmp_path) is None
    path = write_timing_record(
        "walk_forward", session, meta={"model": "demo"}, data_dir=tmp_path
    )
    assert path is not None
    assert path.parent == timing_root(tmp_path)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["kind"] == "walk_forward"
    assert loaded["meta"]["model"] == "demo"


def test_rank_hotspots_and_classify() -> None:
    records = [
        {
            "phases": {
                "predict/fixtures": 90.0,
                "fit": 5.0,
                "optimise/milp": 10.0,
                "ingest": 2.0,
            }
        },
        {"phases": {"predict/fixtures": 10.0, "fit": 1.0}},
    ]
    ranked = rank_hotspots(records)
    assert ranked[0].phase == "predict/fixtures"
    assert ranked[0].seconds == 100.0
    assert ranked[0].category == "python_loop"
    assert classify_phase("optimise/milp") == "solver"
    assert classify_phase("fit/minutes") == "fit"
    assert classify_phase("plan/write") == "io"
    assert classify_phase("remote/pull") == "io"
    assert classify_phase("remote/push") == "io"
    report = format_hotspot_report(records)
    assert "predict/fixtures=" in report
    assert "[python_loop]" in report
    assert format_hotspot_report([]) == "No timing artifacts found."
    remote_report = format_hotspot_report(
        [
            {
                "kind": "remote",
                "total_s": 1.5,
                "meta": {
                    "op": "pull",
                    "prefix": "raw",
                    "bucket_kind": "data",
                    "n_objects": 3,
                    "bytes": 1200,
                },
                "phases": {"remote/pull": 1.5},
            }
        ]
    )
    assert "kind=remote" in remote_report
    assert "op=pull" in remote_report
    assert "objects=3" in remote_report
    assert "bytes=1200" in remote_report
    assert "[io]" in remote_report


def test_perf_report_cli_json(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    from fplt.cli import main
    from fplt.settings import get_settings

    session = TimingSession()
    session.add("predict/fixtures", 12.0)
    session.add("fit", 3.0)
    write_timing_record(
        "walk_forward", session, meta={"model": "cli"}, data_dir=tmp_path
    )
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        main(["perf-report", "--json", "--limit", "5"])
    finally:
        get_settings.cache_clear()
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_records"] == 1
    assert payload["hotspots"][0]["phase"] == "predict/fixtures"
    assert payload["hotspots"][0]["category"] == "python_loop"


def test_load_timing_records_filters_kind(tmp_path: Path) -> None:
    session = TimingSession()
    session.add("optimise/milp", 3.0)
    session.add("plan/chips", 1.0)
    write_timing_record("plan", session, meta={"model": "x"}, data_dir=tmp_path)
    walk = TimingSession()
    walk.add("predict", 9.0)
    write_timing_record("walk_forward", walk, meta={"model": "y"}, data_dir=tmp_path)
    plans = load_timing_records(tmp_path, kind="plan")
    assert len(plans) == 1
    assert plans[0]["kind"] == "plan"
    limited = load_timing_records(tmp_path, limit=1)
    assert len(limited) == 1
    summary = format_kind_summary("plan", session, meta={"model": "x", "horizon": 3})
    assert "timing kind=plan" in summary
    assert "optimise/milp=" in summary


class _SlowMean(RollingMeanPredictor):
    """Baseline that sleeps in fit/predict so wall times are measurable."""

    def fit(self, view: PITView) -> None:
        time.sleep(0.02)
        super().fit(view)

    def predict(self, view: PITView, horizon: list[GWKey]):
        time.sleep(0.03)
        return super().predict(view, horizon)


def test_walk_forward_logs_fit_predict_split_and_summary(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = build_league(DataStore(tmp_path))
    spec = ModelSpec(
        name="rolling_mean",
        kind="rolling_mean",
        params={"window": 5},
        refit="never",
    )
    as_ofs = [GWKey("2024-25", 1), GWKey("2024-25", 2)]
    with caplog.at_level(logging.INFO, logger="fplt.backtest.walk_forward"):
        with timing_artifacts(tmp_path):
            walk_forward(
                _SlowMean(spec),
                PITData(store),
                as_ofs,
                steps=1,
            )
    assert "gw 1 done in" in caplog.text
    assert "fit=" in caplog.text
    assert "predict=" in caplog.text
    assert "write=" in caplog.text
    assert "rolling_mean 2024-25 gw 2 (2/2) predicting" in caplog.text
    summary_lines = [
        record.getMessage()
        for record in caplog.records
        if record.getMessage().startswith("walk_forward timing model=rolling_mean")
    ]
    assert len(summary_lines) == 1
    summary = summary_lines[0]
    assert "gws=2" in summary
    assert "fit=" in summary and "predict=" in summary
    assert "%" in summary
    artifacts = list(timing_root(tmp_path).glob("walk_forward_*.json"))
    assert len(artifacts) == 1
    payload = json.loads(artifacts[0].read_text(encoding="utf-8"))
    assert payload["kind"] == "walk_forward"
    assert payload["meta"]["model"] == "rolling_mean"
    assert "predict" in payload["phases"]
