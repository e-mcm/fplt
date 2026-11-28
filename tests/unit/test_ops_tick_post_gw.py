"""ops-tick wires post_gw and skips idle when a checked GW is pending (#393)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from fplt.auth import AuthCheckResult
from fplt.automation.ops_tick import (
    OpsTickResult,
    format_ops_tick_status,
    run_ops_tick,
)
from fplt.automation.production import OfficialLine, ProductionOps
from fplt.data.store import DataStore
from fplt.models.pipeline_spec import live_pipeline_defaults
from fplt.schemas.manifests import BuildManifest, BuildTableEntry, CuratedLatestPointer
from fplt.settings import Settings
from fplt.types import GWKey
from tests.unit.test_decision_scoring import _decision_gw

_TS = datetime(2024, 9, 1, 12, 0, tzinfo=UTC)
_SEASON = "2024-25"


def _ops() -> ProductionOps:
    pipeline = live_pipeline_defaults(model="rolling_mean_v1", horizon=3)
    return ProductionOps(
        models=["rolling_mean_v1"],
        plan_horizon=3,
        pipeline=pipeline,
        lines=[
            OfficialLine(
                line="primary",
                spec="rolling_mean_v1",
                horizon=3,
                role="primary",
                squad_mode="real",
            )
        ],
        min_prior_seasons=1,
        background_quiet_minutes=45,
    )


def _ctx(build: BuildManifest) -> Any:
    pointer = CuratedLatestPointer(
        build_id=build.build_id,
        created_at=_TS,
        raw_cutoff=_TS,
        code_sha="abc1234",
    )
    return type(
        "Ctx",
        (),
        {"pointer": pointer, "build": build, "remote_by_fp": frozenset()},
    )()


def _build() -> BuildManifest:
    return BuildManifest(
        build_id="b" * 64,
        created_at=_TS,
        staging_version=1,
        code_sha="abc1234",
        raw_manifest_hash="r" * 64,
        raw_cutoff=_TS,
        tables={
            "gameweeks": BuildTableEntry(sha="g" * 64, rows=1, schema_hash="s" * 64),
            "player_fixture_outcomes": BuildTableEntry(
                sha="o" * 64, rows=1, schema_hash="s" * 64
            ),
            "fixture_outcomes": BuildTableEntry(
                sha="f" * 64, rows=1, schema_hash="s" * 64
            ),
        },
    )


def _auth_ok() -> AuthCheckResult:
    return AuthCheckResult(
        level="ok",
        reason="ok",
        message="ok",
        expires_at=None,
        next_deadline=None,
        manual_apply=False,
    )


def test_format_includes_post_gw_runs() -> None:
    text = format_ops_tick_status(
        OpsTickResult(
            snapshots_fetched=0,
            staged_snapshots=0,
            understat_matches=0,
            api_football_captures=0,
            catch_up_planned=[],
            pre_deadline_planned=None,
            still_pending=[],
            models=["rolling_mean_v1"],
            horizon=3,
            post_gw_runs=[GWKey(_SEASON, 1)],
        )
    )
    assert "post_gw runs:      2024-25 gw1" in text


def test_ops_tick_runs_post_gw_and_skips_second_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data"
    root.mkdir()
    store = DataStore(root)
    deadline = datetime.now(UTC) + timedelta(days=3)
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": [_SEASON, _SEASON],
                "gw": [1, 2],
                "deadline_time": [_TS, deadline],
                "finished": [True, False],
                "data_checked": [True, False],
                "available_at": [_TS, _TS],
                "source": ["fpl_api", "fpl_api"],
            }
        ),
    )
    decisions = _decision_gw(
        xi_pts=[2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 8],
        captain_index=10,
    ).with_columns(pl.lit(_SEASON).alias("season"))

    build = _build()
    monkeypatch.setattr(
        "fplt.automation.ops_tick.load_ops_build_context",
        lambda *a, **k: _ctx(build),
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.pull_required_tables", lambda *a, **k: []
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.decisions_for_slate",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.tables_for_slate",
        lambda *a, **k: {"gameweeks"},
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.load_production_ops",
        lambda: _ops(),
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.check_token",
        lambda **k: _auth_ok(),
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.enforce_training_history",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.catch_up",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.pre_deadline",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.pending_checked",
        lambda *a, **k: [],
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.publish_ops_status",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "fplt.automation.post_gw.read_decisions",
        lambda _root: decisions,
    )
    monkeypatch.setattr(
        "fplt.automation.post_gw.load_latest_overall_rank",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "fplt.automation.post_gw.maybe_push_run",
        lambda *a, **k: None,
    )
    # Force non-idle via pending post_gw even when fingerprint decisions empty.
    settings = Settings(data_dir=root)

    first = run_ops_tick(
        root,
        now=deadline - timedelta(days=2),
        settings=settings,
        notify=False,
    )
    assert first.post_gw_runs == [GWKey(_SEASON, 1)]

    second = run_ops_tick(
        root,
        now=deadline - timedelta(days=2),
        settings=settings,
        notify=False,
    )
    assert second.post_gw_runs == []


def test_ops_tick_post_gw_failure_is_non_fatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    root = tmp_path / "data"
    root.mkdir()
    store = DataStore(root)
    deadline = datetime.now(UTC) + timedelta(days=3)
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": [_SEASON, _SEASON],
                "gw": [1, 2],
                "deadline_time": [_TS, deadline],
                "finished": [True, False],
                "data_checked": [True, False],
                "available_at": [_TS, _TS],
                "source": ["fpl_api", "fpl_api"],
            }
        ),
    )
    build = _build()
    monkeypatch.setattr(
        "fplt.automation.ops_tick.load_ops_build_context",
        lambda *a, **k: _ctx(build),
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.pull_required_tables", lambda *a, **k: []
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.decisions_for_slate", lambda *a, **k: []
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.tables_for_slate", lambda *a, **k: {"gameweeks"}
    )
    monkeypatch.setattr("fplt.automation.ops_tick.load_production_ops", lambda: _ops())
    monkeypatch.setattr("fplt.automation.ops_tick.check_token", lambda **k: _auth_ok())
    monkeypatch.setattr(
        "fplt.automation.ops_tick.enforce_training_history", lambda *a, **k: None
    )
    monkeypatch.setattr("fplt.automation.ops_tick.catch_up", lambda *a, **k: [])
    monkeypatch.setattr("fplt.automation.ops_tick.pre_deadline", lambda *a, **k: None)
    monkeypatch.setattr("fplt.automation.ops_tick.pending_checked", lambda *a, **k: [])
    monkeypatch.setattr(
        "fplt.automation.ops_tick.publish_ops_status", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.run_pending_post_gw",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("post_gw boom")),
    )
    result = run_ops_tick(
        root,
        now=deadline - timedelta(days=2),
        settings=Settings(data_dir=root),
        notify=False,
    )
    assert result.post_gw_runs == []


def test_ops_tick_result_defaults_empty_post_gw() -> None:
    result = OpsTickResult(
        snapshots_fetched=0,
        staged_snapshots=0,
        understat_matches=0,
        api_football_captures=0,
        catch_up_planned=[],
        pre_deadline_planned=None,
        still_pending=[],
        models=["rolling_mean_v1"],
        horizon=3,
    )
    assert result.post_gw_runs == []
