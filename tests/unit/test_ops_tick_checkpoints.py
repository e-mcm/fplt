"""ops-tick deadline checkpoints and no-plan alarm integration."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.request import Request

import polars as pl
import pytest

from fplt.automation.checkpoints import load_checkpoint_state
from fplt.automation.ops_feeds import FeedRefreshResult
from fplt.automation.ops_tick import run_ops_tick
from fplt.data.store import DataStore
from fplt.settings import Settings, get_settings
from fplt.types import GWKey

_DEADLINE = datetime(2024, 9, 1, 17, 0, tzinfo=UTC)


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


def _patch_tick_deps(
    monkeypatch: pytest.MonkeyPatch,
    *,
    plans_exist_impl: object | None = None,
    plan_calls: list[str] | None = None,
) -> list[Request]:
    monkeypatch.setattr(
        "fplt.automation.ops_tick.refresh_ops_feeds",
        lambda *a, **k: FeedRefreshResult(0, 0, 0, 0),
    )
    monkeypatch.setattr("fplt.automation.ops_tick.catch_up", lambda *a, **k: [])
    monkeypatch.setattr("fplt.automation.ops_tick.pre_deadline", lambda *a, **k: None)
    monkeypatch.setattr("fplt.automation.ops_tick.pending_checked", lambda *a, **k: [])
    monkeypatch.setattr(
        "fplt.automation.ops_tick.reports",
        lambda _d: "plan report",
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.load_production_ops",
        lambda: type(
            "Ops",
            (),
            {"models": ["rolling_mean_v1"], "plan_horizon": 1},
        )(),
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

        raw = kwargs.get("priority")
        priority = raw if isinstance(raw, str) else None
        from fplt.settings import secret_value

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


def test_ops_tick_runs_missed_checkpoint_on_late_tick(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    _league(tmp_path)
    plan_calls: list[str] = []
    _patch_tick_deps(monkeypatch, plan_calls=plan_calls)
    # Arrive late for T−6h: both T−24h and T−6h must run.
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
    state = load_checkpoint_state(tmp_path)
    assert state is not None
    assert state.done == frozenset({"T-24h", "T-6h"})
    # Second tick is idempotent.
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

    # Mark earlier checkpoints done so the tick focuses on the alarm window.
    from fplt.automation.checkpoints import (
        CheckpointState,
        save_checkpoint_state,
    )

    save_checkpoint_state(
        tmp_path,
        CheckpointState(
            key=GWKey("2024-25", 3),
            deadline=_DEADLINE,
            done=frozenset({"T-24h", "T-6h", "T-90m", "T-30m"}),
            alarms_done=frozenset(),
        ),
    )

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
    state = load_checkpoint_state(tmp_path)
    assert state is not None
    assert "retry" in state.alarms_done
    assert "urgent" not in state.alarms_done

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
    assert (
        b"NO PLAN" in (sent[0].data or b"")
        or b"no production plan" in (sent[0].data or b"").lower()
    )
    state = load_checkpoint_state(tmp_path)
    assert state is not None
    assert state.alarms_done == frozenset({"retry", "urgent"})
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

    monkeypatch.setattr(
        "fplt.automation.ops_tick.refresh_ops_feeds",
        lambda *a, **k: FeedRefreshResult(0, 0, 0, 0),
    )
    monkeypatch.setattr("fplt.automation.ops_tick.catch_up", lambda *a, **k: [])
    monkeypatch.setattr("fplt.automation.ops_tick.pre_deadline", lambda *a, **k: None)
    monkeypatch.setattr("fplt.automation.ops_tick.pending_checked", lambda *a, **k: [])
    monkeypatch.setattr("fplt.automation.ops_tick.plans_exist", plans_exist)
    monkeypatch.setattr("fplt.automation.ops_tick.plan_deadline", plan_deadline)
    monkeypatch.setattr(
        "fplt.automation.ops_tick.load_production_ops",
        lambda: type(
            "Ops",
            (),
            {"models": ["rolling_mean_v1"], "plan_horizon": 1},
        )(),
    )
    notify_calls: list[str] = []
    monkeypatch.setattr(
        "fplt.automation.ops_tick.notify_recommendation",
        lambda *a, **k: notify_calls.append(str(k.get("priority"))) or True,
    )

    from fplt.automation.checkpoints import CheckpointState, save_checkpoint_state

    save_checkpoint_state(
        tmp_path,
        CheckpointState(
            key=GWKey("2024-25", 3),
            deadline=_DEADLINE,
            done=frozenset({"T-24h", "T-6h", "T-90m", "T-30m"}),
            alarms_done=frozenset(),
        ),
    )

    # Both retry and urgent are due; retry creates the plan → no urgent.
    result = run_ops_tick(
        tmp_path,
        now=_DEADLINE - timedelta(hours=1),
        models=["rolling_mean_v1"],
        horizon=1,
        notify=True,
    )
    assert result.no_plan_alarms == ["retry"]
    assert plan_calls == ["planned"]
    assert notify_calls == []  # urgent suppressed; retry itself does not notify
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
