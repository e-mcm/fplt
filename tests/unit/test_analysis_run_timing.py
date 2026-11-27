"""Unit tests for ``fplt.analysis.run_timing`` panel shaping."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fplt.analysis.run_timing import (
    discover_ops_timing,
    load_ops_timing_file,
    parse_timing_record,
    run_timing_panel,
    sub_phases_table,
    timing_trend,
    top_level_phases,
    transfer_table,
)
from fplt.schemas.timing import TimingRecord
from fplt.utils.timing import SHARED_SUBPHASES, ops_phase_total, ops_timing_path

_TS = datetime(2026, 11, 27, 12, 0, tzinfo=UTC)


def _ops_record(**phases: float) -> TimingRecord:
    payload = dict(phases)
    return TimingRecord(
        kind="ops_tick",
        recorded_at=_TS,
        meta={"models": ["demo"]},
        phases=payload,
        total_s=ops_phase_total(payload),
    )


def test_shared_subphases_constant_used() -> None:
    assert "fit/score" in SHARED_SUBPHASES
    assert "optimise/milp" in SHARED_SUBPHASES


def test_parse_timing_record_rejects_invalid() -> None:
    assert parse_timing_record(None) is None
    assert parse_timing_record({"kind": ""}) is None


def test_top_and_sub_phases_split() -> None:
    record = _ops_record(
        setup=2.0,
        ingest=10.0,
        **{
            "fit/demo": 20.0,
            "fit/score": 5.0,
            "optimise/milp": 8.0,
            "remote/push": 3.0,
        },
    )
    top = top_level_phases(record)
    assert set(top["phase"].to_list()) == {"setup", "ingest", "fit/demo", "remote/push"}
    assert top["seconds"].sum() == pytest.approx(35.0)
    sub = sub_phases_table(record)
    assert set(sub["phase"].to_list()) == {"fit/score", "optimise/milp"}
    assert sub.sort("phase")["seconds"].to_list() == pytest.approx([5.0, 8.0])


def test_transfer_table_from_remote_records_and_fallback() -> None:
    remote = [
        {
            "kind": "remote",
            "phases": {"remote/pull": 4.0},
            "meta": {
                "op": "pull",
                "prefix": "raw",
                "n_objects": 10,
                "bytes": 8 * 1024 * 1024,
            },
        }
    ]
    frame = transfer_table(remote)
    assert frame.height == 1
    assert frame["op"][0] == "pull"
    assert frame["n_objects"][0] == 10
    assert frame["throughput_mib_s"][0] == pytest.approx(2.0)
    fallback = transfer_table([], phase_seconds={"remote/push": 1.5})
    assert fallback.height == 1
    assert fallback["op"][0] == "push"
    assert fallback["bytes"][0] is None


def test_timing_trend_and_panel_absent() -> None:
    r1 = _ops_record(ingest=5.0, setup=1.0)
    r2 = _ops_record(ingest=7.0, setup=2.0)
    trend = timing_trend(
        [
            ("2026-27", 12, "T-90m", r1),
            ("2026-27", 13, "T-90m", r2),
        ]
    )
    assert trend.height == 2
    assert trend["total_s"].to_list() == pytest.approx([6.0, 9.0])
    assert "ingest" in trend.columns
    panel = run_timing_panel(
        ops_record=None,
        trend_records=[("2026-27", 12, "T-90m", r1)],
    )
    assert panel.present is False
    assert panel.top_phases.is_empty()
    assert panel.trend.height == 1


def test_run_timing_panel_present() -> None:
    record = _ops_record(
        setup=1.0,
        ingest=4.0,
        **{"fit/score": 2.0, "remote/pull": 3.0},
    )
    remote = [
        {
            "kind": "remote",
            "phases": {"remote/pull": 3.0},
            "meta": {"op": "pull", "prefix": "raw", "n_objects": 2, "bytes": 100},
        }
    ]
    panel = run_timing_panel(ops_record=record, remote_records=remote)
    assert panel.present is True
    assert panel.kind == "ops_tick"
    assert panel.total_s == pytest.approx(8.0)
    assert not panel.top_phases.is_empty()
    assert "fit/score" in panel.sub_phases["phase"].to_list()
    assert panel.transfers.height == 1


def test_discover_and_load_ops_timing(tmp_path: Path) -> None:
    checkpoint = tmp_path / "official" / "season=2026-27" / "gw=13" / "checkpoint=T-90m"
    path = ops_timing_path(checkpoint)
    path.parent.mkdir(parents=True)
    record = _ops_record(ingest=2.0, setup=1.0)
    path.write_text(json.dumps(record.model_dump_json_doc()), encoding="utf-8")
    loaded = load_ops_timing_file(path)
    assert loaded is not None
    assert loaded.total_s == pytest.approx(3.0)
    found = discover_ops_timing(tmp_path, season="2026-27", checkpoint="T-90m")
    assert len(found) == 1
    assert found[0][1] == 13
    assert load_ops_timing_file(tmp_path / "missing.json") is None
    assert discover_ops_timing(tmp_path, season="2099-00") == []


def test_empty_frames_have_stable_schema() -> None:
    empty_panel = run_timing_panel(ops_record=None)
    assert empty_panel.top_phases.schema == {
        "phase": pl.Utf8(),
        "seconds": pl.Float64(),
        "share_pct": pl.Float64(),
    }
    assert transfer_table([]).height == 0
    assert timing_trend([]).height == 0
