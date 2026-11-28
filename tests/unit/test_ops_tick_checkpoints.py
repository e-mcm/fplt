"""ops-tick deadline checkpoints and no-plan alarm integration."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.request import Request

import polars as pl
import pytest

from fplt.automation.checkpoints import (
    notified_path,
    write_notified_marker,
    write_plan_checkpoint_marker,
)
from fplt.automation.ops_tick import run_ops_tick
from fplt.automation.production import OfficialLine, ProductionOps
from fplt.data.remote import Remote
from fplt.data.store import DataStore
from fplt.models.pipeline_spec import live_pipeline_defaults
from fplt.settings import Settings, get_settings
from fplt.types import GWKey
from tests.unit._ops_tick_curated import patch_curated_ops_path

_DEADLINE = datetime(2024, 9, 1, 17, 0, tzinfo=UTC)
_PRIMARY = "v4_h3"
_HAS_RCLONE = shutil.which("rclone") is not None


def _tiny_ops() -> ProductionOps:
    pipeline = live_pipeline_defaults(model="rolling_mean_v1", horizon=1)
    return ProductionOps(
        models=["rolling_mean_v1"],
        plan_horizon=1,
        pipeline=pipeline,
        lines=[
            OfficialLine(
                line=_PRIMARY,
                spec="rolling_mean_v1",
                horizon=1,
                role="primary",
                squad_mode="real",
            )
        ],
        min_prior_seasons=3,
    )


def _league(root: Path, *, deadline: datetime = _DEADLINE) -> None:
    store = DataStore(root)
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": ["2024-25"],
                "gw": [3],
                "deadline_time": [deadline],
                "finished": [False],
                "data_checked": [False],
                "available_at": [deadline - timedelta(days=20)],
            }
        ),
    )


def _seed_done(
    root: Path,
    *,
    checkpoints: frozenset[str],
    alarms: frozenset[str] = frozenset(),
    primary_line: str = _PRIMARY,
) -> None:
    key = GWKey("2024-25", 3)
    for cp_id in checkpoints:
        write_plan_checkpoint_marker(root, key, cp_id, primary_line=primary_line)
        if cp_id == "T-6h":
            write_notified_marker(root, key)
        if cp_id == "T-90m":
            # T-90m done needs every slate line; tiny ops has only primary.
            pass
    for alarm_id in alarms:
        from fplt.automation.checkpoints import write_alarm_marker

        write_alarm_marker(root, key, alarm_id)


def _patch_tick_deps(
    monkeypatch: pytest.MonkeyPatch,
    *,
    plans_exist_impl: object | None = None,
    plan_calls: list[str] | None = None,
) -> list[Request]:
    patch_curated_ops_path(monkeypatch)
    monkeypatch.setattr("fplt.automation.ops_tick.catch_up", lambda *a, **k: [])
    monkeypatch.setattr("fplt.automation.ops_tick.pre_deadline", lambda *a, **k: None)
    monkeypatch.setattr("fplt.automation.ops_tick.pending_checked", lambda *a, **k: [])
    monkeypatch.setattr(
        "fplt.automation.ops_tick.reports",
        lambda _d: "plan report",
    )
    monkeypatch.setattr("fplt.automation.ops_tick.load_production_ops", _tiny_ops)
    monkeypatch.setattr(
        "fplt.automation.ops_tick.enforce_training_history",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "fplt.automation.status.push_status",
        lambda *a, **k: None,
    )

    calls = plan_calls if plan_calls is not None else []

    def plan_deadline(
        data_dir: Path,
        key: GWKey,
        now: datetime,
        models: list[str] | None = None,
        horizon: int | None = None,
        *,
        refresh_if_stale: bool = True,
    ) -> bool:
        calls.append(f"{key}:{now.isoformat()}:refresh={refresh_if_stale}")
        return False

    monkeypatch.setattr("fplt.automation.ops_tick.plan_deadline", plan_deadline)

    def fake_official(
        data_dir: Path,
        ops: object,
        key: GWKey,
        *,
        settings: object = None,
        chip_arm: str = "optimised",
        **_kwargs: object,
    ) -> object:
        from fplt.automation.official_slate import OfficialCheckpointResult

        calls.append(f"official:{key}")
        # Write T-90m primary pointer so subsequent ticks see it as done.
        write_plan_checkpoint_marker(data_dir, key, "T-90m", primary_line=_PRIMARY)
        return OfficialCheckpointResult(
            season=key.season,
            gw=key.gw,
            checkpoint="T-90m",
            pairs=(),
            checkpoint_dir=data_dir
            / "official"
            / f"season={key.season}"
            / f"gw={key.gw}"
            / "checkpoint=T-90m",
            build_id="testbuild",
        )

    monkeypatch.setattr(
        "fplt.automation.ops_tick.run_official_checkpoint", fake_official
    )

    if plans_exist_impl is None:

        def plans_exist(*_a: object, **_k: object) -> bool:
            return False

        monkeypatch.setattr("fplt.automation.ops_tick.plans_exist", plans_exist)
    else:
        monkeypatch.setattr("fplt.automation.ops_tick.plans_exist", plans_exist_impl)

    sent: list[Request] = []

    def post(request: Request) -> object:
        sent.append(request)
        return None

    def notify_recommendation(
        settings: Settings,
        title: str,
        message: str,
        **kwargs: object,
    ) -> bool:
        from fplt.automation.notify import notify
        from fplt.settings import secret_value

        raw = kwargs.get("priority")
        priority = raw if isinstance(raw, str) else None
        return notify(
            secret_value(settings.ntfy_url),
            title,
            message,
            post=post,
            priority=priority,
        )

    monkeypatch.setattr(
        "fplt.automation.ops_tick.notify_recommendation",
        notify_recommendation,
    )
    return sent


def _patch_local_data_remote(monkeypatch: pytest.MonkeyPatch, remote: Remote) -> None:
    """Point Remote.from_settings at a local-dir remote."""

    def _from_settings(
        _cls: type,
        _settings: Settings,
        bucket: str | None = None,
    ) -> Remote:
        del bucket
        return remote

    monkeypatch.setattr(
        "fplt.data.remote.Remote.from_settings",
        classmethod(_from_settings),
    )
    monkeypatch.setattr(
        "fplt.automation.status.push_status",
        lambda *a, **k: None,
    )


def test_ops_tick_runs_missed_checkpoint_on_late_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    _league(tmp_path)
    plan_calls: list[str] = []
    _patch_tick_deps(monkeypatch, plan_calls=plan_calls)
    moment = _DEADLINE - timedelta(hours=5)
    settings = Settings(data_dir=tmp_path, ntfy_url="https://ntfy.sh/fplt-test")
    result = run_ops_tick(
        tmp_path,
        now=moment,
        models=["rolling_mean_v1"],
        horizon=1,
        settings=settings,
        notify=True,
    )
    assert result.checkpoints_run == ["T-24h", "T-6h"]
    assert result.upcoming == GWKey("2024-25", 3)
    assert len(plan_calls) == 2
    assert notified_path(tmp_path, GWKey("2024-25", 3)).is_file()
    # Second tick is idempotent via markers.
    again = run_ops_tick(
        tmp_path,
        now=moment,
        models=["rolling_mean_v1"],
        horizon=1,
        settings=settings,
        notify=True,
    )
    assert again.checkpoints_run == []
    get_settings.cache_clear()


def test_ops_tick_missing_official_slate_alarm_at_t60m(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T−60m urgent alarm when T−90m slate pointers are still missing (#373)."""
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    _league(tmp_path)
    sent = _patch_tick_deps(
        monkeypatch,
        plans_exist_impl=lambda *a, **k: True,
    )

    def stale(*_a: object, **_k: object) -> None:
        from fplt.automation.ops_data import StaleBuildError

        raise StaleBuildError("curated build too old for test")

    monkeypatch.setattr("fplt.automation.ops_tick.enforce_build_freshness", stale)
    _seed_done(tmp_path, checkpoints=frozenset({"T-24h", "T-6h"}))
    settings = Settings(data_dir=tmp_path, ntfy_url="https://ntfy.sh/fplt-test")

    # Before T−60m: T−90m is due but skipped (stale); slate alarm not yet due.
    before = run_ops_tick(
        tmp_path,
        now=_DEADLINE - timedelta(minutes=61),
        models=["rolling_mean_v1"],
        horizon=1,
        settings=settings,
        notify=True,
    )
    assert "missing_official_slate" not in before.no_plan_alarms
    assert len(sent) == 0

    at_t60 = run_ops_tick(
        tmp_path,
        now=_DEADLINE - timedelta(minutes=60),
        models=["rolling_mean_v1"],
        horizon=1,
        settings=settings,
        notify=True,
    )
    assert at_t60.no_plan_alarms == ["missing_official_slate"]
    assert len(sent) == 1
    assert sent[0].get_header("Priority") == "5"
    marker = (
        tmp_path
        / "official"
        / "season=2024-25"
        / "gw=3"
        / "alarms"
        / "missing_official_slate.json"
    )
    assert marker.is_file()

    sent.clear()
    again = run_ops_tick(
        tmp_path,
        now=_DEADLINE - timedelta(minutes=50),
        models=["rolling_mean_v1"],
        horizon=1,
        settings=settings,
        notify=True,
    )
    assert again.no_plan_alarms == []
    assert len(sent) == 0
    get_settings.cache_clear()


def test_ops_tick_no_missing_slate_alarm_when_t90m_done(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pointers present → T−60m missing-slate alarm stays quiet."""
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    _league(tmp_path)
    sent = _patch_tick_deps(
        monkeypatch,
        plans_exist_impl=lambda *a, **k: True,
    )
    _seed_done(tmp_path, checkpoints=frozenset({"T-24h", "T-6h", "T-90m", "T-30m"}))
    settings = Settings(data_dir=tmp_path, ntfy_url="https://ntfy.sh/fplt-test")
    result = run_ops_tick(
        tmp_path,
        now=_DEADLINE - timedelta(minutes=45),
        models=["rolling_mean_v1"],
        horizon=1,
        settings=settings,
        notify=True,
    )
    assert result.no_plan_alarms == []
    assert len(sent) == 0
    get_settings.cache_clear()


def test_ops_tick_no_plan_retry_then_urgent_alarm(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Synthetic deadline + missing plan → retry at T−3h, urgent at T−2h."""
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    _league(tmp_path)
    plan_calls: list[str] = []
    sent = _patch_tick_deps(monkeypatch, plan_calls=plan_calls)
    settings = Settings(data_dir=tmp_path, ntfy_url="https://ntfy.sh/fplt-test")

    _seed_done(tmp_path, checkpoints=frozenset({"T-24h", "T-6h", "T-90m", "T-30m"}))

    at_retry = _DEADLINE - timedelta(hours=3)
    retry_result = run_ops_tick(
        tmp_path,
        now=at_retry,
        models=["rolling_mean_v1"],
        horizon=1,
        settings=settings,
        notify=True,
    )
    assert retry_result.no_plan_alarms == ["retry"]
    assert any("refresh=True" in call for call in plan_calls)
    assert (
        tmp_path / "official" / "season=2024-25" / "gw=3" / "alarms" / "retry.json"
    ).is_file()

    plan_calls.clear()
    sent.clear()
    at_urgent = _DEADLINE - timedelta(hours=2)
    urgent_result = run_ops_tick(
        tmp_path,
        now=at_urgent,
        models=["rolling_mean_v1"],
        horizon=1,
        settings=settings,
        notify=True,
    )
    assert urgent_result.no_plan_alarms == ["urgent"]
    assert len(sent) == 1
    assert sent[0].get_header("Priority") == "5"
    get_settings.cache_clear()


def test_ops_tick_skips_urgent_when_retry_creates_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    _league(tmp_path)

    has_plan = {"value": False}

    def plans_exist(*_a: object, **_k: object) -> bool:
        return has_plan["value"]

    plan_calls: list[str] = []

    def plan_deadline(*_a: object, **_k: object) -> bool:
        plan_calls.append("planned")
        has_plan["value"] = True
        return True

    patch_curated_ops_path(monkeypatch)
    monkeypatch.setattr("fplt.automation.ops_tick.catch_up", lambda *a, **k: [])
    monkeypatch.setattr("fplt.automation.ops_tick.pre_deadline", lambda *a, **k: None)
    monkeypatch.setattr("fplt.automation.ops_tick.pending_checked", lambda *a, **k: [])
    monkeypatch.setattr("fplt.automation.ops_tick.plans_exist", plans_exist)
    monkeypatch.setattr("fplt.automation.ops_tick.plan_deadline", plan_deadline)
    monkeypatch.setattr("fplt.automation.ops_tick.load_production_ops", _tiny_ops)
    monkeypatch.setattr(
        "fplt.automation.ops_tick.enforce_training_history",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "fplt.automation.status.push_status",
        lambda *a, **k: None,
    )
    notify_calls: list[str] = []
    monkeypatch.setattr(
        "fplt.automation.ops_tick.notify_recommendation",
        lambda *a, **k: notify_calls.append(str(k.get("priority"))) or True,
    )

    _seed_done(tmp_path, checkpoints=frozenset({"T-24h", "T-6h", "T-90m", "T-30m"}))

    result = run_ops_tick(
        tmp_path,
        now=_DEADLINE - timedelta(hours=1),
        models=["rolling_mean_v1"],
        horizon=1,
        notify=True,
    )
    assert result.no_plan_alarms == ["retry"]
    assert plan_calls == ["planned"]
    assert notify_calls == []
    get_settings.cache_clear()


def test_ops_tick_t90m_skips_apply(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    _league(tmp_path)
    _patch_tick_deps(
        monkeypatch,
        plans_exist_impl=lambda *a, **k: True,
    )
    _seed_done(tmp_path, checkpoints=frozenset({"T-24h", "T-6h"}))
    import logging

    with caplog.at_level(logging.INFO, logger="fplt.automation.ops_tick"):
        result = run_ops_tick(
            tmp_path,
            now=_DEADLINE - timedelta(minutes=80),
            models=["rolling_mean_v1"],
            horizon=1,
            notify=False,
        )
    assert "T-90m" in result.checkpoints_run
    assert any("skipping apply" in rec.message for rec in caplog.records)
    get_settings.cache_clear()


@pytest.mark.skipif(not _HAS_RCLONE, reason="rclone binary not installed")
def test_ops_tick_markers_survive_fresh_runner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two ticks on fresh data dirs sharing only the remote: second is a no-op."""
    bucket = tmp_path / "bucket"
    runner_a = tmp_path / "runner-a"
    runner_b = tmp_path / "runner-b"
    runner_a.mkdir()
    runner_b.mkdir()
    remote = Remote(target=str(bucket), kind="data", env={})
    _patch_local_data_remote(monkeypatch, remote)

    monkeypatch.setenv("FPLT_DATA_DIR", str(runner_a))
    monkeypatch.setenv("FPLT_DATA_BUCKET", "fplt-data")
    get_settings.cache_clear()
    _league(runner_a)
    plan_calls: list[str] = []
    sent = _patch_tick_deps(monkeypatch, plan_calls=plan_calls)
    settings = Settings(
        data_dir=runner_a,
        data_bucket="fplt-data",
        ntfy_url="https://ntfy.sh/fplt-test",
    )
    moment = _DEADLINE - timedelta(hours=5)
    first = run_ops_tick(
        runner_a,
        now=moment,
        models=["rolling_mean_v1"],
        horizon=1,
        settings=settings,
        notify=True,
    )
    assert first.checkpoints_run == ["T-24h", "T-6h"]
    assert len(sent) == 1
    assert (
        bucket
        / "official"
        / "season=2024-25"
        / "gw=3"
        / "checkpoint=T-6h"
        / "notified.json"
    ).is_file()

    plan_calls.clear()
    sent.clear()
    monkeypatch.setenv("FPLT_DATA_DIR", str(runner_b))
    get_settings.cache_clear()
    _league(runner_b)
    settings_b = Settings(
        data_dir=runner_b,
        data_bucket="fplt-data",
        ntfy_url="https://ntfy.sh/fplt-test",
    )
    second = run_ops_tick(
        runner_b,
        now=moment,
        models=["rolling_mean_v1"],
        horizon=1,
        settings=settings_b,
        notify=True,
    )
    assert second.checkpoints_run == []
    assert plan_calls == []
    assert sent == []
    get_settings.cache_clear()


@pytest.mark.skipif(not _HAS_RCLONE, reason="rclone binary not installed")
def test_ops_tick_t6h_notify_fires_once_across_ticks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """T−6h notify fires exactly once across N ticks between T−6h and T−90m."""
    bucket = tmp_path / "bucket"
    remote = Remote(target=str(bucket), kind="data", env={})
    _patch_local_data_remote(monkeypatch, remote)

    notify_titles: list[str] = []
    n_ticks = 4
    for i in range(n_ticks):
        runner = tmp_path / f"runner-{i}"
        runner.mkdir()
        monkeypatch.setenv("FPLT_DATA_DIR", str(runner))
        get_settings.cache_clear()
        _league(runner)
        plan_calls: list[str] = []
        _patch_tick_deps(monkeypatch, plan_calls=plan_calls)

        def _notify(
            settings: Settings,
            title: str,
            message: str,
            **kwargs: object,
        ) -> bool:
            del settings, message, kwargs
            notify_titles.append(title)
            return True

        monkeypatch.setattr("fplt.automation.ops_tick.notify_recommendation", _notify)
        settings = Settings(
            data_dir=runner,
            data_bucket="fplt-data",
            ntfy_url="https://ntfy.sh/fplt-test",
        )
        moment = _DEADLINE - timedelta(hours=4)
        result = run_ops_tick(
            runner,
            now=moment,
            models=["rolling_mean_v1"],
            horizon=1,
            settings=settings,
            notify=True,
        )
        if i == 0:
            assert "T-6h" in result.checkpoints_run
        else:
            assert result.checkpoints_run == []

    t6h_notifies = [title for title in notify_titles if "T-6h" in title]
    assert len(t6h_notifies) == 1
    get_settings.cache_clear()


def test_ops_tick_skips_t90m_when_official_pointers_exist(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing T−90m pointers → slate not re-run."""
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    _league(tmp_path)
    plan_calls: list[str] = []
    _patch_tick_deps(
        monkeypatch,
        plan_calls=plan_calls,
        plans_exist_impl=lambda *a, **k: True,
    )
    key = GWKey("2024-25", 3)
    write_plan_checkpoint_marker(tmp_path, key, "T-90m", primary_line=_PRIMARY)

    result = run_ops_tick(
        tmp_path,
        now=_DEADLINE - timedelta(minutes=80),
        models=["rolling_mean_v1"],
        horizon=1,
        notify=False,
    )
    assert "T-90m" not in result.checkpoints_run
    assert not any(call.startswith("official:") for call in plan_calls)
    get_settings.cache_clear()


def test_ops_tick_t6h_retries_notify_when_notified_marker_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Notify fails → no notified.json → retried next tick."""
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    _league(tmp_path)
    plan_calls: list[str] = []
    _patch_tick_deps(monkeypatch, plan_calls=plan_calls)

    outcomes = iter([False, True])
    titles: list[str] = []

    def flaky_notify(
        settings: Settings,
        title: str,
        message: str,
        **kwargs: object,
    ) -> bool:
        del settings, message, kwargs
        titles.append(title)
        return next(outcomes)

    monkeypatch.setattr("fplt.automation.ops_tick.notify_recommendation", flaky_notify)
    settings = Settings(data_dir=tmp_path, ntfy_url="https://ntfy.sh/fplt-test")
    moment = _DEADLINE - timedelta(hours=4)

    first = run_ops_tick(
        tmp_path,
        now=moment,
        models=["rolling_mean_v1"],
        horizon=1,
        settings=settings,
        notify=True,
    )
    assert "T-6h" in first.checkpoints_run
    assert not notified_path(tmp_path, GWKey("2024-25", 3)).is_file()

    second = run_ops_tick(
        tmp_path,
        now=moment,
        models=["rolling_mean_v1"],
        horizon=1,
        settings=settings,
        notify=True,
    )
    assert "T-6h" in second.checkpoints_run
    assert notified_path(tmp_path, GWKey("2024-25", 3)).is_file()
    assert len([t for t in titles if "T-6h" in t]) == 2

    third = run_ops_tick(
        tmp_path,
        now=moment,
        models=["rolling_mean_v1"],
        horizon=1,
        settings=settings,
        notify=True,
    )
    assert third.checkpoints_run == []
    get_settings.cache_clear()
