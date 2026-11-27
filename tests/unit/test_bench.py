"""Nightly performance bench — regression rule, baseline, budget (#312)."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import SecretStr

from fplt.automation.bench import (
    BenchConfig,
    BenchExistsError,
    build_bench_results,
    consolidate_fixture_phases,
    evaluate_regression,
    format_regression_alerts,
    format_trend_report,
    list_bench_results,
    load_bench_config,
    notify_regression_alerts,
    ops_budget_status,
    rolling_baseline,
    run_regression_check,
    write_bench_results,
)
from fplt.automation.ops_timing import format_ops_step_summary
from fplt.settings import Settings


def _cfg(**overrides: object) -> BenchConfig:
    base: dict[str, object] = {
        "baseline_window": 5,
        "regression_pct": 50.0,
        "consecutive_nights": 3,
        "ops_budget_pct": 50.0,
        "ops_timeout_minutes": 90,
        "tracked_phases": ("fit", "predict", "plan"),
    }
    base.update(overrides)
    return BenchConfig.model_validate(base)


def _night(
    day: str,
    *,
    fit: float,
    predict: float = 10.0,
    plan: float = 5.0,
):
    from fplt.schemas.bench import BenchResults

    result: BenchResults = build_bench_results(
        day=day,
        phases={"fit": fit, "predict": predict, "plan": plan},
        recorded_at=datetime(2026, 11, 1, tzinfo=UTC),
    )
    return result


def _history_stable(*, n: int = 8, fit: float = 10.0):
    """Newest-first stable history."""
    from fplt.schemas.bench import BenchResults

    start = datetime(2026, 11, 20, tzinfo=UTC)
    nights: list[BenchResults] = []
    for i in range(n):
        day = (start - timedelta(days=i)).date().isoformat()
        nights.append(_night(day, fit=fit))
    return nights


def test_load_default_bench_config() -> None:
    cfg = load_bench_config()
    assert cfg.baseline_window >= 1
    assert cfg.consecutive_nights >= 1
    assert "fit" in cfg.tracked_phases


def test_write_once_and_list(tmp_path: Path) -> None:
    results = build_bench_results(
        day="2026-11-27",
        phases={"fit": 12.0, "predict": 40.0},
        mc_bench={"simulate_match": 100.0},
        pytest_top25=[{"nodeid": "tests/unit/test_x.py::t", "seconds": 1.5}],
    )
    path = write_bench_results(tmp_path, results)
    assert path.is_file()
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["date"] == "2026-11-27"
    assert payload["phases"]["fit"] == pytest.approx(12.0)
    with pytest.raises(BenchExistsError):
        write_bench_results(tmp_path, results)
    write_bench_results(tmp_path, results, exist_ok=True)
    listed = list_bench_results(tmp_path)
    assert len(listed) == 1
    assert listed[0].date == "2026-11-27"


def test_rolling_baseline_excludes_latest() -> None:
    history = _history_stable(n=5, fit=10.0)
    # Newest is slower; baseline should still be ~10 from priors.
    history[0] = _night(history[0].date, fit=50.0)
    baseline = rolling_baseline(history, window=4, exclude_latest=True)
    assert baseline["fit"] == pytest.approx(10.0)


def test_stable_history_no_alert() -> None:
    history = _history_stable(n=8, fit=10.0)
    report = evaluate_regression(history, config=_cfg())
    assert not report.alerted
    assert report.baseline["fit"] == pytest.approx(10.0)


def test_step_change_alerts_after_k_nights() -> None:
    """Stable baseline then K consecutive 2× nights → alert; K-1 → no alert."""
    cfg = _cfg(consecutive_nights=3, regression_pct=50.0, baseline_window=5)
    # 5 stable priors (newest-first index 3..) + will prepend slow nights.
    priors = _history_stable(n=5, fit=10.0)
    # Two slow nights — not enough.
    two_slow = [
        _night("2026-11-22", fit=25.0),
        _night("2026-11-21", fit=25.0),
        *priors,
    ]
    assert not evaluate_regression(two_slow, config=cfg).alerted
    # Three consecutive slow nights.
    three_slow = [
        _night("2026-11-23", fit=25.0),
        _night("2026-11-22", fit=25.0),
        _night("2026-11-21", fit=25.0),
        *priors,
    ]
    report = evaluate_regression(three_slow, config=cfg)
    assert report.alerted
    assert len(report.alerts) == 1
    assert report.alerts[0].phase == "fit"
    assert report.alerts[0].consecutive == 3
    assert "fit:" in format_regression_alerts(report)


def test_notify_regression_uses_injected_post(tmp_path: Path) -> None:
    cfg = _cfg(consecutive_nights=2, baseline_window=3)
    priors = _history_stable(n=3, fit=10.0)
    history = [
        _night("2026-11-24", fit=30.0),
        _night("2026-11-23", fit=30.0),
        *priors,
    ]
    for night in reversed(history):
        write_bench_results(tmp_path, night)
    seen: list[str] = []

    def fake_post(request: object) -> object:
        seen.append(str(getattr(request, "full_url", request)))
        return None

    settings = Settings(ntfy_url=SecretStr("https://ntfy.sh/fplt-bench-test"))
    report = run_regression_check(
        tmp_path,
        settings=settings,
        config=cfg,
        notify_alerts=True,
        post=fake_post,
    )
    assert report.alerted
    assert seen
    assert (
        notify_regression_alerts(Settings(ntfy_url=None), report, post=fake_post)
        is False
    )


def test_ops_budget_warning_in_step_summary() -> None:
    """Over-budget total_s appends WARNING; under-budget stays clean."""
    cfg = _cfg(ops_budget_pct=50.0, ops_timeout_minutes=10)
    # Budget = 10 min * 60 * 0.5 = 300s.
    under = ops_budget_status(100.0, config=cfg)
    assert not under.over_budget
    assert under.warning_line is None
    over = ops_budget_status(301.0, config=cfg)
    assert over.over_budget
    assert over.warning_line is not None
    assert "WARNING" in over.warning_line

    record = {"phases": {"ingest": 200.0, "setup": 150.0}, "total_s": 350.0}
    text = format_ops_step_summary(record, timeout_minutes=10)
    assert "### Ops timing" in text
    assert "**WARNING:**" in text
    assert "350.0s" in text

    ok = format_ops_step_summary(
        {"phases": {"ingest": 10.0}, "total_s": 10.0},
        timeout_minutes=10,
    )
    assert "WARNING" not in ok


def test_consolidate_fixture_phases_skips_nested_subphases() -> None:
    raw = {
        "fit/dc_layered_v4_e2e": 12.0,
        "fit/score": 3.0,
        "predict/dc_layered_v4_e2e": 40.0,
        "predict/setup": 1.0,
        "plan/e2e_primary": 8.0,
        "plan/setup": 0.5,
        "optimise/milp": 5.0,
        "ingest": 2.0,
    }
    out = consolidate_fixture_phases(raw)
    assert out["fit"] == pytest.approx(12.0)
    assert out["predict"] == pytest.approx(40.0)
    assert out["plan"] == pytest.approx(8.0)
    assert out["optimise/milp"] == pytest.approx(5.0)
    assert "ingest" not in out


def test_trend_report_mentions_baseline(tmp_path: Path) -> None:
    for night in _history_stable(n=4, fit=11.0):
        write_bench_results(tmp_path, night)
    history = list_bench_results(tmp_path)
    text = format_trend_report(history, config=_cfg(), limit=4)
    assert "perf trend:" in text
    assert "fit" in text
    assert "rolling median baseline" in text
