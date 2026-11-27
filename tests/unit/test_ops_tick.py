"""ops-tick wraps feed refresh → catch_up → pre_deadline."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from fplt.automation.ops_feeds import FeedRefreshResult
from fplt.automation.ops_tick import (
    OpsTickResult,
    format_ops_tick_status,
    run_ops_tick,
)
from fplt.cli import main
from fplt.data.snapshots import SnapshotFetchError
from fplt.settings import get_settings
from fplt.types import GWKey

_ROOT = Path(__file__).resolve().parents[2]


def test_format_ops_tick_status_lists_catch_up_and_pre_deadline() -> None:
    text = format_ops_tick_status(
        OpsTickResult(
            snapshots_fetched=1,
            staged_snapshots=2,
            understat_matches=3,
            api_football_captures=4,
            catch_up_planned=[GWKey("2024-25", 1)],
            pre_deadline_planned=GWKey("2024-25", 2),
            still_pending=[],
            models=["dc_layered_v4_native"],
            horizon=3,
            checkpoints_run=["T-6h"],
            no_plan_alarms=[],
            upcoming=GWKey("2024-25", 2),
        )
    )
    assert "ops-tick status" in text
    assert "auth check:        ok" in text
    assert "catch_up planned:  2024-25 gw1" in text
    assert "pre_deadline:      2024-25 gw2" in text
    assert "upcoming deadline: 2024-25 gw2" in text
    assert "checkpoints run:   T-6h" in text
    assert "no-plan alarms:    none" in text
    assert "dc_layered_v4_native / H=3" in text
    assert text.endswith("\n")


def test_format_ops_tick_status_uses_none_when_empty() -> None:
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
            horizon=1,
        )
    )
    assert "catch_up planned:  none" in text
    assert "pre_deadline:      none" in text
    assert "still pending:     none" in text


def test_run_ops_tick_calls_catch_up_and_pre_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("JOB_STARTED_AT", "1700000000")
    get_settings.cache_clear()
    calls: list[str] = []
    moment = datetime(2024, 8, 16, 9, 0, tzinfo=UTC)

    def refresh(
        data_dir: Path | None = None,
        *,
        source: str = "auto",
        remote: str = "origin",
        branch: str = "snapshots",
        settings: object = None,
    ) -> FeedRefreshResult:
        calls.append(f"refresh:{source}:{remote}:{branch}")
        return FeedRefreshResult(1, 2, 3, 4)

    def catch_up(
        data_dir: Path,
        models: list[str] | None = None,
        horizon: int | None = None,
    ) -> list[GWKey]:
        calls.append(f"catch_up:{models}:{horizon}")
        return [GWKey("2024-25", 1)]

    def pre(
        data_dir: Path,
        now: datetime,
        models: list[str] | None = None,
        horizon: int | None = None,
        *,
        refresh_if_stale: bool = True,
    ) -> GWKey | None:
        calls.append(f"pre_deadline:{now.isoformat()}:{models}:{horizon}")
        return GWKey("2024-25", 2)

    notify_calls: list[str] = []

    monkeypatch.setattr("fplt.automation.ops_tick.refresh_ops_feeds", refresh)
    monkeypatch.setattr("fplt.automation.ops_tick.catch_up", catch_up)
    monkeypatch.setattr("fplt.automation.ops_tick.pre_deadline", pre)
    monkeypatch.setattr("fplt.automation.ops_tick.pending_checked", lambda *a, **k: [])
    monkeypatch.setattr("fplt.automation.ops_tick.reports", lambda _d: "report")
    monkeypatch.setattr(
        "fplt.automation.ops_tick.notify_recommendation",
        lambda *a, **k: notify_calls.append("sent") or True,
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.load_production_ops",
        lambda: type(
            "Ops",
            (),
            {
                "models": ["rolling_mean_v1"],
                "plan_horizon": 3,
                "min_prior_seasons": 3,
            },
        )(),
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.enforce_training_history",
        lambda *a, **k: None,
    )

    result = run_ops_tick(
        now=moment,
        source="branch",
        remote="/tmp/remote",
        branch="snapshots",
    )
    assert result.catch_up_planned == [GWKey("2024-25", 1)]
    assert result.pre_deadline_planned == GWKey("2024-25", 2)
    assert result.snapshots_fetched == 1
    assert result.models == ["rolling_mean_v1"]
    assert result.horizon == 3
    assert calls == [
        "refresh:branch:/tmp/remote:snapshots",
        "catch_up:['rolling_mean_v1']:3",
        f"pre_deadline:{moment.isoformat()}:['rolling_mean_v1']:3",
    ]
    assert notify_calls == ["sent"]
    timing_files = list((tmp_path / "timing").glob("ops_tick_*.json"))
    assert len(timing_files) == 1
    get_settings.cache_clear()


def test_run_ops_tick_skips_notify_when_idle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    notify_calls: list[str] = []

    monkeypatch.setattr(
        "fplt.automation.ops_tick.refresh_ops_feeds",
        lambda *a, **k: FeedRefreshResult(0, 0, 0, 0),
    )
    monkeypatch.setattr("fplt.automation.ops_tick.catch_up", lambda *a, **k: [])
    monkeypatch.setattr("fplt.automation.ops_tick.pre_deadline", lambda *a, **k: None)
    monkeypatch.setattr("fplt.automation.ops_tick.pending_checked", lambda *a, **k: [])
    monkeypatch.setattr(
        "fplt.automation.ops_tick.notify_recommendation",
        lambda *a, **k: notify_calls.append("sent") or True,
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.load_production_ops",
        lambda: type(
            "Ops",
            (),
            {
                "models": ["rolling_mean_v1"],
                "plan_horizon": 1,
                "min_prior_seasons": 3,
            },
        )(),
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.enforce_training_history",
        lambda *a, **k: None,
    )

    idle = run_ops_tick(models=["rolling_mean_v1"], horizon=1)
    assert idle.catch_up_planned == []
    assert idle.pre_deadline_planned is None
    assert notify_calls == []
    get_settings.cache_clear()


def test_run_ops_tick_propagates_fetch_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("JOB_STARTED_AT", "1000")
    get_settings.cache_clear()

    def boom(*_a: object, **_k: object) -> FeedRefreshResult:
        raise SnapshotFetchError("missing")

    monkeypatch.setattr("fplt.automation.ops_tick.refresh_ops_feeds", boom)
    monkeypatch.setattr(
        "fplt.automation.ops_tick.load_production_ops",
        lambda: type(
            "Ops",
            (),
            {
                "models": ["rolling_mean_v1"],
                "plan_horizon": 1,
                "min_prior_seasons": 3,
            },
        )(),
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.enforce_training_history",
        lambda *a, **k: None,
    )
    with pytest.raises(SnapshotFetchError, match="missing"):
        run_ops_tick()
    # Failed tick still writes a local timing artifact; never ops_timing.json.
    timing_files = list((tmp_path / "timing").glob("ops_tick_*.json"))
    assert len(timing_files) == 1
    assert not list(tmp_path.glob("official/**/ops_timing.json"))
    get_settings.cache_clear()


def test_cli_ops_tick_prints_status(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("FPLT_ENV_FILE", str(tmp_path / "env"))
    get_settings.cache_clear()
    seen: dict[str, object] = {}

    def fake_run(**kwargs: object) -> OpsTickResult:
        seen.update(kwargs)
        return OpsTickResult(
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

    monkeypatch.setattr("fplt.automation.ops_tick.run_ops_tick", fake_run)
    main(["ops-tick", "--no-notify", "--source", "bucket"])
    out = capsys.readouterr().out
    assert "ops-tick status" in out
    assert "pre_deadline:      none" in out
    assert seen.get("source") == "bucket"
    assert seen.get("notify") is False
    get_settings.cache_clear()


def test_cli_ops_tick_exits_on_fetch_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()

    def boom(**_k: object) -> OpsTickResult:
        raise SnapshotFetchError("boom")

    monkeypatch.setattr("fplt.automation.ops_tick.run_ops_tick", boom)
    with pytest.raises(SystemExit) as exc:
        main(["ops-tick"])
    assert exc.value.code == 2
    get_settings.cache_clear()


def test_makefile_documents_ops_tick() -> None:
    makefile = (_ROOT / "Makefile").read_text(encoding="utf-8")
    assert "ops-tick:" in makefile
    assert "fplt ops-tick" in makefile


def test_run_ops_tick_expired_token_keeps_planning_with_banner(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """ops-01: expired token still plans; recommendation carries manual-apply banner."""
    import base64
    import json

    from fplt.auth import MANUAL_APPLY_BANNER

    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    exp = int((datetime(2020, 1, 1, tzinfo=UTC)).timestamp())
    header = base64.urlsafe_b64encode(b'{"alg":"none"}').decode("ascii").rstrip("=")
    payload = (
        base64.urlsafe_b64encode(json.dumps({"exp": exp}).encode("utf-8"))
        .decode("ascii")
        .rstrip("=")
    )
    monkeypatch.setenv("FPLT_FPL_API_TOKEN", f"{header}.{payload}.sig")
    get_settings.cache_clear()

    bodies: list[str] = []

    monkeypatch.setattr(
        "fplt.automation.ops_tick.refresh_ops_feeds",
        lambda *a, **k: FeedRefreshResult(0, 0, 0, 0),
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.catch_up",
        lambda *a, **k: [GWKey("2024-25", 1)],
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
        "fplt.automation.ops_tick.reports",
        lambda _d: "plan report body",
    )

    def capture_notify(
        _settings: object,
        title: str,
        message: str,
        **kwargs: object,
    ) -> bool:
        bodies.append(f"{title}\n{message}")
        return True

    monkeypatch.setattr(
        "fplt.automation.ops_tick.notify_recommendation",
        capture_notify,
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.load_production_ops",
        lambda: type(
            "Ops",
            (),
            {
                "models": ["rolling_mean_v1"],
                "plan_horizon": 1,
                "min_prior_seasons": 3,
            },
        )(),
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.enforce_training_history",
        lambda *a, **k: None,
    )

    result = run_ops_tick(
        now=datetime(2026, 11, 26, 12, 0, tzinfo=UTC),
        models=["rolling_mean_v1"],
        horizon=1,
        notify=True,
    )
    assert result.catch_up_planned == [GWKey("2024-25", 1)]
    assert result.auth_level == "urgent"
    assert result.auth_manual_apply is True
    assert any(MANUAL_APPLY_BANNER in body for body in bodies)
    assert any("plan report body" in body for body in bodies)
    # Auth alert + plan recommendation (at least two notify calls).
    assert len(bodies) >= 2
    get_settings.cache_clear()


def test_run_checkpoint_writes_ops_log_jsonl(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each ops checkpoint persists write-once ``ops_log.jsonl`` (#310)."""
    from datetime import timedelta

    from fplt.automation.checkpoints import Checkpoint
    from fplt.automation.ops_tick import _run_checkpoint
    from fplt.automation.production import load_production_ops
    from fplt.settings import Settings
    from fplt.utils.logging import (
        configure_logging,
        ops_log_path,
        reset_logging_for_tests,
    )

    reset_logging_for_tests()
    configure_logging(secrets=())
    monkeypatch.setattr(
        "fplt.automation.ops_tick.plan_deadline",
        lambda *a, **k: True,
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.reports",
        lambda _d: "report",
    )
    key = GWKey("2026-27", 12)
    # Non-T-90m path exercises plan_deadline + ops_log without official slate.
    checkpoint = Checkpoint(
        id="T-6h",
        offset=timedelta(hours=6),
        plan=True,
        refresh=False,
        notify=False,
        apply=False,
    )
    planned, notified, official = _run_checkpoint(
        tmp_path,
        key,
        checkpoint,
        datetime(2026, 11, 28, 3, 0, tzinfo=UTC),
        ["rolling_mean_v1"],
        3,
        Settings(data_dir=tmp_path),
        ops=load_production_ops(),
        notify=False,
    )
    assert planned is True
    assert notified is False
    assert official is None
    path = ops_log_path(
        tmp_path / "official" / "season=2026-27" / "gw=12" / "checkpoint=T-6h"
    )
    assert path.is_file()
    body = path.read_text(encoding="utf-8")
    assert "checkpoint" in body
    assert "T-6h" in body
    reset_logging_for_tests()

def test_run_ops_tick_history_guard_fails_and_alerts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#285: thin history aborts before catch_up and alerts ops-12."""
    from fplt.automation.history_guard import HistoryGuardError

    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    catch_calls: list[str] = []

    monkeypatch.setattr(
        "fplt.automation.ops_tick.refresh_ops_feeds",
        lambda *a, **k: FeedRefreshResult(0, 0, 0, 0),
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.catch_up",
        lambda *a, **k: catch_calls.append("catch_up") or [],
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.load_production_ops",
        lambda: type(
            "Ops",
            (),
            {
                "models": ["rolling_mean_v1"],
                "plan_horizon": 1,
                "min_prior_seasons": 3,
            },
        )(),
    )

    def boom(*_a: object, **_k: object) -> None:
        raise HistoryGuardError("need 3 prior seasons, found 0")

    monkeypatch.setattr("fplt.automation.ops_tick.enforce_training_history", boom)

    with pytest.raises(HistoryGuardError, match="prior seasons"):
        run_ops_tick(now=datetime(2024, 8, 16, 9, 0, tzinfo=UTC), notify=True)
    assert catch_calls == []
    get_settings.cache_clear()
