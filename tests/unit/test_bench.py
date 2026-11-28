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
    bench_date_dir,
    bench_results_path,
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
    assert "exp_component" in cfg.tracked_phases


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


def test_bench_config_rss_warn_kb() -> None:
    cfg = _cfg(runner_memory_gb=16.0, rss_warn_pct=70.0)
    assert cfg.rss_warn_kb() == int(16.0 * 0.70 * 1024 * 1024)
    loaded = load_bench_config()
    assert "points_pmf" in loaded.tracked_phases
    assert loaded.rss_warn_pct == pytest.approx(70.0)


def test_consolidate_fixture_phases_skips_nested_subphases() -> None:
    raw = {
        "fit/dc_layered_v4_e2e": 12.0,
        "fit/score": 3.0,
        "predict/dc_layered_v4_e2e": 40.0,
        "predict/setup": 1.0,
        "points_pmf": 6.0,
        "plan/e2e_primary": 8.0,
        "plan/setup": 0.5,
        "optimise/milp": 5.0,
        "ingest": 2.0,
    }
    out = consolidate_fixture_phases(raw)
    assert out["fit"] == pytest.approx(12.0)
    assert out["predict"] == pytest.approx(40.0)
    assert out["plan"] == pytest.approx(8.0)
    assert out["points_pmf"] == pytest.approx(6.0)
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


def test_empty_config_yaml_and_date_helpers(tmp_path: Path) -> None:
    empty = tmp_path / "empty.yaml"
    empty.write_text("", encoding="utf-8")
    assert load_bench_config(empty) == BenchConfig()
    from datetime import date as date_cls

    assert bench_date_dir(tmp_path, date_cls(2026, 11, 27)).name == "date=2026-11-27"
    assert bench_results_path(tmp_path, "2026-11-27").name == "results.json"


def test_list_skips_unreadable_and_respects_limit(tmp_path: Path) -> None:
    assert list_bench_results(tmp_path) == []
    good = build_bench_results(day="2026-11-27", phases={"fit": 1.0})
    write_bench_results(tmp_path, good)
    bad_dir = tmp_path / "bench" / "date=2026-11-26"
    bad_dir.mkdir(parents=True)
    (bad_dir / "results.json").write_text("{not-json", encoding="utf-8")
    listed = list_bench_results(tmp_path, limit=1)
    assert len(listed) == 1
    assert listed[0].date == "2026-11-27"
    assert rolling_baseline([], window=0) == {}


def test_empty_history_and_missing_phase_paths() -> None:
    empty = evaluate_regression([], config=_cfg())
    assert not empty.alerted
    assert format_regression_alerts(empty) == "No performance regressions."
    assert format_trend_report([], config=_cfg()).startswith("perf trend: no")
    # Prior nights without the tracked phase → baseline missing → skip.
    priors = [
        build_bench_results(day="2026-11-20", phases={"predict": 5.0}),
        build_bench_results(day="2026-11-19", phases={"predict": 5.0}),
    ]
    recent = [
        build_bench_results(day="2026-11-22", phases={"fit": 99.0}),
        build_bench_results(day="2026-11-21", phases={"predict": 5.0}),
        *priors,
    ]
    report = evaluate_regression(recent, config=_cfg(tracked_phases=("fit",)))
    assert not report.alerted
    # Newest nights missing the phase break the streak.
    mixed = [
        build_bench_results(day="2026-11-23", phases={"predict": 1.0}),
        build_bench_results(day="2026-11-22", phases={"fit": 50.0}),
        build_bench_results(day="2026-11-21", phases={"fit": 50.0}),
        *_history_stable(n=3, fit=10.0),
    ]
    assert not evaluate_regression(
        mixed, config=_cfg(consecutive_nights=3, tracked_phases=("fit",))
    ).alerted


def test_notify_skips_when_not_alerted() -> None:
    report = evaluate_regression([], config=_cfg())
    assert (
        notify_regression_alerts(Settings(ntfy_url=None), report, post=lambda r: None)
        is False
    )


def test_trend_skips_missing_phase_and_n_a(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    night = build_bench_results(day="2026-11-27", phases={"fit": 10.0})
    # Tracked phase absent on the night → continue (no row).
    text = format_trend_report(
        [night],
        config=_cfg(tracked_phases=("fit", "plan")),
        limit=1,
    )
    assert "fit" in text
    assert "plan" not in text.split("rolling median")[0] or "plan:" not in text

    def _empty_baseline(*_a: object, **_k: object) -> dict[str, float]:
        return {}

    monkeypatch.setattr(
        "fplt.automation.bench.rolling_baseline",
        _empty_baseline,
    )
    text_na = format_trend_report(
        [night],
        config=_cfg(tracked_phases=("fit",)),
        limit=1,
    )
    assert "n/a" in text_na


def test_consolidate_skips_non_positive() -> None:
    assert consolidate_fixture_phases({"fit": 0.0, "predict": -1.0}) == {}


def test_schema_validators_and_z_suffix() -> None:
    from datetime import UTC, datetime

    from pydantic import ValidationError

    from fplt.schemas.bench import BenchResults

    with pytest.raises(ValidationError):
        BenchResults(
            date="2026-11-27",
            recorded_at=datetime(2026, 11, 27, tzinfo=UTC),
            phases={"": 1.0},
        )
    with pytest.raises(ValidationError):
        BenchResults(
            date="2026-11-27",
            recorded_at=datetime(2026, 11, 27, tzinfo=UTC),
            phases={"fit": True},  # type: ignore[dict-item]
        )
    with pytest.raises(ValidationError):
        BenchResults.model_validate(
            {
                "date": "2026-11-27",
                "recorded_at": "2026-11-27T00:00:00Z",
                "phases": ["not", "a", "dict"],
            }
        )
    none_phases = BenchResults.model_validate(
        {
            "date": "2026-11-27",
            "recorded_at": "2026-11-27T00:00:00Z",
            "phases": None,
        }
    )
    assert none_phases.phases == {}

    ok = BenchResults(
        date="2026-11-27",
        recorded_at=datetime(2026, 11, 27, 12, 0, tzinfo=UTC),
        phases={"fit": 1.0, "skip": 0.0},
        mc_bench={"a": -1.0, "b": 2.0},
    )
    assert "skip" not in ok.phases
    assert "a" not in ok.mc_bench
    assert ok.mc_bench["b"] == pytest.approx(2.0)
    doc = ok.model_dump_json_doc()
    assert str(doc["recorded_at"]).endswith("Z")


def test_ops_timeout_env_invalid_and_valid(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("OPS_TIMEOUT_MINUTES", "not-an-int")
    text = format_ops_step_summary(
        {"phases": {"ingest": 10.0}, "total_s": 10.0},
        timeout_minutes=None,
    )
    assert "WARNING" not in text
    monkeypatch.setenv("OPS_TIMEOUT_MINUTES", "10")
    over = format_ops_step_summary(
        {"phases": {"ingest": 400.0}, "total_s": 400.0},
        timeout_minutes=None,
    )
    assert "**WARNING:**" in over


def test_perf_report_trend_cli(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    from fplt.cli import main

    for night in _history_stable(n=3, fit=9.0):
        write_bench_results(tmp_path, night)
    monkeypatch.delenv("FPLT_LAB_DIR", raising=False)
    main(
        [
            "perf-report",
            "--trend",
            "--lab-dir",
            str(tmp_path),
            "--limit",
            "3",
        ]
    )
    out = capsys.readouterr().out
    assert "perf trend:" in out
    main(
        [
            "perf-report",
            "--trend",
            "--lab-dir",
            str(tmp_path),
            "--json",
            "--limit",
            "2",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert payload["n_nights"] == 2
    monkeypatch.setenv("FPLT_LAB_DIR", str(tmp_path))
    main(["perf-report", "--trend", "--limit", "1"])
    assert "perf trend:" in capsys.readouterr().out
