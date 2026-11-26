"""Unit tests for ``fplt.analysis.overview`` shaping helpers."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from fplt.analysis.overview import (
    official_pointers_frame,
    plan_summary_frame,
    quarantine_summary_frame,
    runs_catalog_frame,
)
from fplt.schemas.manifests import AsOfDeadline, OfficialPointer, RunManifest
from fplt.schemas.quarantine import QuarantineRecordSchema

_TS = datetime(2026, 11, 26, 12, 0, 0, tzinfo=UTC)


def _manifest(**overrides: object) -> RunManifest:
    base: dict[str, object] = {
        "run_id": "r_20261126T120000Z_aaaaaa",
        "kind": "adhoc",
        "mode": "live",
        "run_type": "predict",
        "spec_hash": "a" * 64,
        "spec_label": "demo",
        "model_id": "deadbeef",
        "code_sha": "abc1234",
        "lockfile_hash": "b" * 64,
        "as_of": [AsOfDeadline(season="2025-26", gw=12, deadline_utc=_TS)],
        "horizon": 3,
        "host": "test",
        "started_at": _TS,
        "finished_at": _TS,
        "status": "succeeded",
    }
    base.update(overrides)
    return RunManifest.model_validate(base)


def test_runs_catalog_empty() -> None:
    frame = runs_catalog_frame([])
    assert frame.is_empty()
    assert "run_id" in frame.columns


def test_runs_catalog_newest_first() -> None:
    older = _manifest(run_id="r_20261126T110000Z_aaaaaa")
    newer = _manifest(run_id="r_20261126T120000Z_bbbbbb", line="v4_h3")
    frame = runs_catalog_frame([older, newer])
    assert frame["run_id"].to_list() == [
        "r_20261126T120000Z_bbbbbb",
        "r_20261126T110000Z_aaaaaa",
    ]
    assert frame["line"].to_list()[0] == "v4_h3"
    assert frame["season"].to_list()[0] == "2025-26"
    assert frame["gw"].to_list()[0] == 12


def test_plan_summary_step0() -> None:
    plan = pl.DataFrame(
        {
            "model_id": ["m1", "m1", "m1", "m1"],
            "as_of_season": ["2025-26"] * 4,
            "as_of_gw": [12] * 4,
            "target_gw": [12] * 4,
            "step": [0, 0, 0, 1],
            "player_key": [1, 2, 3, 4],
            "position": ["MID", "FWD", "DEF", "GK"],
            "team_key": [10, 11, 12, 13],
            "in_xi": [True, True, False, True],
            "captain": [True, False, False, False],
            "vice": [False, True, False, False],
            "bought": [True, False, False, False],
            "sold": [False, False, True, False],
            "xpts": [5.0, 4.0, 0.0, 3.0],
            "hit": [4.0, 4.0, 4.0, 0.0],
        }
    )
    summary = plan_summary_frame(plan)
    assert summary.height == 1
    row = summary.to_dicts()[0]
    assert row["transfers_in"] == 1
    assert row["transfers_out"] == 1
    assert row["hit"] == 4.0
    assert row["captain_key"] == 1
    assert row["vice_key"] == 2
    assert row["xi_count"] == 2


def test_plan_summary_empty_when_no_step0() -> None:
    plan = pl.DataFrame(
        {
            "model_id": ["m1"],
            "as_of_season": ["2025-26"],
            "as_of_gw": [12],
            "target_gw": [13],
            "step": [1],
            "player_key": [1],
            "position": ["MID"],
            "team_key": [10],
            "in_xi": [True],
            "captain": [True],
            "vice": [False],
            "bought": [False],
            "sold": [False],
            "xpts": [1.0],
            "hit": [0.0],
        }
    )
    assert plan_summary_frame(plan).is_empty()
    assert plan_summary_frame(pl.DataFrame()).is_empty()


def test_quarantine_summary_groups() -> None:
    assert quarantine_summary_frame(pl.DataFrame()).is_empty()
    frame = QuarantineRecordSchema.validate(
        pl.DataFrame(
            {
                "season": ["2025-26", "2025-26"],
                "source": ["fpl_api", "fpl_api"],
                "capture_path": ["a/path", "b/path"],
                "captured_at": [_TS, _TS],
                "gate": ["null_share", "null_share"],
                "detail": ["x", "y"],
                "previous_capture_path": [None, None],
                "available_at": [_TS, _TS],
                "schema_version": [1, 1],
            }
        )
    )
    assert isinstance(frame, pl.DataFrame)
    summary = quarantine_summary_frame(frame)
    assert summary.height == 1
    assert int(summary["n_rows"][0]) == 2
    assert summary["source"].to_list()[0] == "fpl_api"


def test_official_pointers_frame() -> None:
    pointer = OfficialPointer(
        predict_run_id="r_pred",
        plan_run_id="r_plan",
        role="primary",
        squad_mode="real",
        written_at=_TS,
    )
    frame = official_pointers_frame([("2025-26", 12, "T-90m", "v4_h3", pointer)])
    assert frame.height == 1
    assert frame["line"].to_list()[0] == "v4_h3"
    assert official_pointers_frame([]).is_empty()
