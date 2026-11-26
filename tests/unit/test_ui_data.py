"""Tests for the read-only ``DashboardStore`` adapter and local fallbacks."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from fplt.data.quarantine import append_quarantine
from fplt.schemas.l2 import SCHEMA_VERSION
from fplt.schemas.manifests import AsOfDeadline, OfficialPointer
from fplt.store.runs import RunWriter
from fplt.ui.data import DashboardStore

_TS = datetime(2026, 11, 26, 12, 0, 0, tzinfo=UTC)


def _plan_rows() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model_id": ["deadbeef", "deadbeef"],
            "as_of_season": ["2025-26", "2025-26"],
            "as_of_gw": [12, 12],
            "target_gw": [12, 12],
            "step": [0, 0],
            "player_key": [101, 102],
            "position": ["MID", "FWD"],
            "team_key": [3, 7],
            "in_xi": [True, True],
            "captain": [True, False],
            "vice": [False, True],
            "bought": [True, False],
            "sold": [False, True],
            "xpts": [6.0, 5.0],
            "hit": [4.0, 4.0],
        }
    )


def test_dashboard_store_empty(tmp_path: Path) -> None:
    store = DashboardStore(tmp_path)
    assert store.runs_catalog().is_empty()
    assert store.official_lines().is_empty()
    assert store.latest_plan_summary().is_empty()
    assert store.quarantine_summary().is_empty()
    assert store.status_payload() is None


def test_dashboard_store_reads_runs_plan_official_quarantine_status(
    tmp_path: Path,
) -> None:
    deadline = AsOfDeadline(season="2025-26", gw=12, deadline_utc=_TS)
    with RunWriter(
        tmp_path,
        kind="official",
        mode="live",
        run_type="plan",
        spec_hash="a" * 64,
        spec_label="demo",
        model_id="deadbeef",
        as_of=[deadline],
        horizon=3,
        plan_horizon=3,
        code_sha="abc1234",
        lockfile_hash="b" * 64,
        host="test",
        started_at=_TS,
        run_id="r_20261126T120000Z_plan01",
    ) as writer:
        writer.write_parquet("plans/plan.parquet", _plan_rows())

    pointer = OfficialPointer(
        predict_run_id="r_pred",
        plan_run_id="r_20261126T120000Z_plan01",
        role="primary",
        squad_mode="real",
        written_at=_TS,
    )
    official = (
        tmp_path
        / "official"
        / "season=2025-26"
        / "gw=12"
        / "checkpoint=T-90m"
        / "line=v4_h3.json"
    )
    official.parent.mkdir(parents=True)
    official.write_text(pointer.model_dump_json(), encoding="utf-8")

    append_quarantine(
        tmp_path,
        pl.DataFrame(
            {
                "season": ["2025-26"],
                "source": ["fpl_api"],
                "capture_path": ["fpl_api/season=2025-26/captured_at=x"],
                "captured_at": [_TS],
                "gate": ["null_share"],
                "detail": ["fixture"],
                "previous_capture_path": [None],
                "available_at": [_TS],
                "schema_version": [SCHEMA_VERSION],
            }
        ),
    )

    status_path = tmp_path / "status" / "status.json"
    status_path.parent.mkdir(parents=True)
    status_path.write_text(
        json.dumps({"last_error": None, "quarantines": 1}),
        encoding="utf-8",
    )

    store = DashboardStore(tmp_path)
    runs = store.runs_catalog()
    assert runs.height == 1
    assert runs["run_id"].to_list()[0] == "r_20261126T120000Z_plan01"
    assert runs["kind"].to_list()[0] == "official"

    official_frame = store.official_lines()
    assert official_frame.height == 1
    assert official_frame["line"].to_list()[0] == "v4_h3"

    plan = store.latest_plan_summary()
    assert plan.height == 1
    assert int(plan["transfers_in"][0]) == 1
    assert int(plan["captain_key"][0]) == 101

    quarantine = store.quarantine_summary()
    assert quarantine.height == 1
    assert quarantine["source"].to_list()[0] == "fpl_api"

    assert store.status_payload() == {"last_error": None, "quarantines": 1}


def test_dashboard_store_falls_back_to_legacy_plans(tmp_path: Path) -> None:
    path = tmp_path / "plans" / "deadbeef" / "as_of=2025-26_gw12_h03" / "part.parquet"
    path.parent.mkdir(parents=True)
    _plan_rows().write_parquet(path)
    store = DashboardStore(tmp_path)
    plan = store.latest_plan_summary()
    assert plan.height == 1
    assert plan["as_of_gw"].to_list()[0] == 12
