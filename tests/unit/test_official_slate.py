"""Unit tests for T−90m official slate wiring (#307 product gaps)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from fplt.automation.official_slate import (
    OfficialCheckpointResult,
    run_official_checkpoint,
)
from fplt.automation.ops_feeds import FeedRefreshResult
from fplt.automation.ops_tick import run_ops_tick
from fplt.automation.production import OfficialLine, ProductionOps, load_production_ops
from fplt.data.store import DataStore
from fplt.models.pipeline_spec import live_pipeline_defaults
from fplt.settings import Settings, get_settings
from fplt.store.official import official_checkpoint_dir
from fplt.store.push import maybe_push_official, push_official
from fplt.types import GWKey
from fplt.utils.timing import ops_timing_path as timing_ops_path

_DEADLINE = datetime(2024, 9, 1, 17, 0, tzinfo=UTC)


def _tiny_ops(*, model: str = "rolling_mean_v1", horizon: int = 1) -> ProductionOps:
    pipeline = live_pipeline_defaults(model=model, horizon=horizon)
    return ProductionOps(
        models=[model],
        plan_horizon=horizon,
        pipeline=pipeline,
        lines=[
            OfficialLine(
                line="e2e_primary",
                spec=model,
                horizon=horizon,
                role="primary",
                squad_mode="real",
            )
        ],
    )


def test_ops_tick_t90m_calls_official_slate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("JOB_STARTED_AT", "1700000000")
    get_settings.cache_clear()
    store = DataStore(tmp_path)
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": ["2024-25"],
                "gw": [3],
                "deadline_time": [_DEADLINE],
                "finished": [False],
                "data_checked": [False],
                "available_at": [_DEADLINE - timedelta(days=20)],
            }
        ),
    )
    calls: list[str] = []
    cp_dir = tmp_path / "official" / "season=2024-25" / "gw=3" / "checkpoint=T-90m"
    cp_dir.mkdir(parents=True)

    def fake_official(
        data_dir: Path,
        ops: ProductionOps,
        key: GWKey,
        *,
        settings: Settings | None = None,
        chip_arm: str = "optimised",
    ) -> OfficialCheckpointResult:
        calls.append(f"{key.season} gw{key.gw}:{ops.primary_line().line}:{chip_arm}")
        return OfficialCheckpointResult(
            season=key.season,
            gw=key.gw,
            checkpoint="T-90m",
            pairs=(),
            checkpoint_dir=cp_dir,
            build_id="abc",
        )

    monkeypatch.setattr(
        "fplt.automation.ops_tick.refresh_ops_feeds",
        lambda *a, **k: FeedRefreshResult(0, 0, 0, 0),
    )
    monkeypatch.setattr("fplt.automation.ops_tick.catch_up", lambda *a, **k: [])
    monkeypatch.setattr("fplt.automation.ops_tick.pre_deadline", lambda *a, **k: None)
    monkeypatch.setattr("fplt.automation.ops_tick.pending_checked", lambda *a, **k: [])
    monkeypatch.setattr("fplt.automation.ops_tick.plans_exist", lambda *a, **k: True)
    monkeypatch.setattr(
        "fplt.automation.ops_tick.run_official_checkpoint", fake_official
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.load_production_ops",
        _tiny_ops,
    )
    from fplt.automation.checkpoints import CheckpointState, save_checkpoint_state

    save_checkpoint_state(
        tmp_path,
        CheckpointState(
            key=GWKey("2024-25", 3),
            deadline=_DEADLINE,
            done=frozenset({"T-24h", "T-6h"}),
            alarms_done=frozenset(),
        ),
    )

    result = run_ops_tick(
        tmp_path,
        now=_DEADLINE - timedelta(minutes=80),
        models=["rolling_mean_v1"],
        horizon=1,
        notify=False,
    )
    assert "T-90m" in result.checkpoints_run
    assert calls == ["2024-25 gw3:e2e_primary:optimised"]
    # Official ops_timing.json written when checkpoint_dir is passed through.
    assert timing_ops_path(cp_dir).is_file()
    get_settings.cache_clear()


def test_maybe_push_official_skips_without_bucket(tmp_path: Path) -> None:
    cp = official_checkpoint_dir(tmp_path, season="2024-25", gw=3, checkpoint="T-90m")
    cp.mkdir(parents=True)
    (cp / "applied.json").write_text("{}", encoding="utf-8")
    assert (
        maybe_push_official(tmp_path, season="2024-25", gw=3, settings=Settings())
        is None
    )


def test_push_official_to_local_remote(tmp_path: Path) -> None:
    from fplt.data.remote import Remote

    data = tmp_path / "data"
    bucket = tmp_path / "bucket"
    cp = official_checkpoint_dir(data, season="2024-25", gw=3, checkpoint="T-90m")
    cp.mkdir(parents=True)
    (cp / "line=e2e_primary.json").write_text('{"ok": true}\n', encoding="utf-8")
    remote = Remote(target=str(bucket), kind="data", env={})
    push_official(data, season="2024-25", gw=3, checkpoint="T-90m", remote=remote)
    dest = (
        bucket
        / "official"
        / "season=2024-25"
        / "gw=3"
        / "checkpoint=T-90m"
        / "line=e2e_primary.json"
    )
    assert dest.is_file()


def test_run_official_checkpoint_requires_sampled_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opaque models cannot write points_pmf; fail with a clear message."""
    from tests.builders import build_league

    build_league(DataStore(tmp_path))
    # Seed minimal staged content so build_from_staged can run.
    monkeypatch.setattr(
        "fplt.automation.official_slate.build_from_staged",
        lambda *a, **k: type("B", (), {"build_id": "deadbeef"})(),
    )

    def boom_predict(*_a: Any, **_k: Any) -> str:
        raise ValueError(
            "official predict requires SampledPointsPredictor.sample() "
            "for points PMFs; got RollingMeanPredictor"
        )

    monkeypatch.setattr(
        "fplt.automation.official_slate._official_predict", boom_predict
    )
    ops = _tiny_ops()
    with pytest.raises(ValueError, match="SampledPointsPredictor"):
        run_official_checkpoint(tmp_path, ops, GWKey("2024-25", 2))


def test_committed_production_slate_still_loads() -> None:
    ops = load_production_ops()
    horizons, lines, checkpoint = ops.official_slate()
    assert checkpoint == "T-90m"
    assert "dc_layered_v4_native" in horizons
    assert any(line.role == "primary" for line in lines)


def test_write_predict_run_persists_points_pmf(tmp_path: Path) -> None:
    from fplt.models.predictions import empty_predictions
    from fplt.models.spec import load_model_specs
    from fplt.store.api import Store
    from fplt.store.points_pmf import points_pmf_from_draws
    from fplt.store.runs import write_predict_run
    from tests.builders import build_league

    store = build_league(DataStore(tmp_path))
    as_of = GWKey("2024-25", 2)
    deadline = store.deadline(as_of)
    spec = load_model_specs()["rolling_mean_v1"]
    draws = pl.DataFrame(
        {
            "player_key": [100, 100],
            "team_key": [3, 3],
            "fixture_id": [2, 2],
            "target_season": ["2024-25", "2024-25"],
            "target_gw": [2, 2],
            "horizon": [0, 0],
            "draw_id": [0, 1],
            "points": [4.0, 6.0],
        }
    )

    def _pmf(run_id: str) -> tuple[pl.DataFrame, pl.DataFrame]:
        fx = points_pmf_from_draws(
            draws,
            run_id=run_id,
            as_of_season=as_of.season,
            as_of_gw=as_of.gw,
            deadline_utc=deadline,
            grain="fixture",
        )
        gw = points_pmf_from_draws(
            draws,
            run_id=run_id,
            as_of_season=as_of.season,
            as_of_gw=as_of.gw,
            deadline_utc=deadline,
            grain="gameweek",
        )
        return fx, gw

    manifest = write_predict_run(
        tmp_path,
        spec=spec,
        as_of=as_of,
        deadline_utc=deadline,
        horizon=1,
        frame=empty_predictions(),
        kind="official",
        checkpoint="T-90m",
        points_pmf_for_run_id=_pmf,
    )
    analytic = Store(tmp_path)
    assert not analytic.load("points_pmf", run=manifest).collect().is_empty()
    assert not analytic.load("points_pmf_gw", run=manifest).collect().is_empty()
