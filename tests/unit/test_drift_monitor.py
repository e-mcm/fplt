"""ops-09 live drift monitor — alerting logic on synthetic ledgers only."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from fplt.automation.drift import (
    DriftMonitorConfig,
    MetricThreshold,
    backtest_distributions,
    beyond_threshold,
    evaluate_drift,
    format_drift_alerts,
    load_drift_monitor_config,
    per_gw_metrics,
    rolling_live_metrics,
)

_SEASON = "2024-25"
_MODEL = "prod_model"


def _cfg(**overrides: object) -> DriftMonitorConfig:
    base: dict[str, object] = {
        "horizon": 1,
        "rolling_window": 3,
        "consecutive_beyond": 3,
        "metrics": {
            "rmse": MetricThreshold(z_score=2.0),
            "brier_60": MetricThreshold(z_score=2.0),
            "brier_goal": MetricThreshold(z_score=2.0),
            "brier_cs": MetricThreshold(z_score=2.0),
        },
    }
    base.update(overrides)
    return DriftMonitorConfig.model_validate(base)


def _row(
    *,
    run_kind: str,
    gw: int,
    xpts: float,
    actual_pts: int,
    p_60: float,
    minutes: int,
    p_goal: float = 0.2,
    goals: int = 0,
    p_cs: float = 0.3,
    cs: int = 0,
    player: int = 1,
) -> dict[str, object]:
    return {
        "model_id": _MODEL,
        "run_kind": run_kind,
        "as_of_season": _SEASON,
        "as_of_gw": gw,
        "target_season": _SEASON,
        "target_gw": gw,
        "horizon": 1,
        "player_key": player,
        "position": "MID",
        "fixture_id": 1000 + gw,
        "xpts": xpts,
        "xpts_sd": 1.0,
        "p_played": 0.9,
        "p_60": p_60,
        "p_goal": p_goal,
        "p_assist": 0.1,
        "p_cs": p_cs,
        "p_haul": 0.05,
        "actual_pts": actual_pts,
        "actual_minutes": minutes,
        "actual_goals": goals,
        "actual_assists": 0,
        "actual_cs": cs,
    }


def _stable_ledger(*, n_gw: int = 8, players: int = 4) -> pl.DataFrame:
    """Backtest + live rows with the same small residual each GW."""
    rows: list[dict[str, object]] = []
    for gw in range(1, n_gw + 1):
        for kind in ("backtest", "live"):
            for p in range(players):
                # xpts close to actual → low RMSE; probs close to outcomes.
                actual = 4 + (p % 2)
                rows.append(
                    _row(
                        run_kind=kind,
                        gw=gw,
                        xpts=float(actual) + 0.1,
                        actual_pts=actual,
                        p_60=0.8,
                        minutes=70,
                        p_goal=0.25,
                        goals=0,
                        p_cs=0.2,
                        cs=0,
                        player=10 + p,
                    )
                )
    return pl.DataFrame(rows)


def _step_change_ledger(*, n_gw: int = 10, players: int = 4) -> pl.DataFrame:
    """Live RMSE steps up sharply from GW 6 onward; backtest stays stable."""
    rows: list[dict[str, object]] = []
    for gw in range(1, n_gw + 1):
        for p in range(players):
            actual = 4 + (p % 2)
            rows.append(
                _row(
                    run_kind="backtest",
                    gw=gw,
                    xpts=float(actual) + 0.1,
                    actual_pts=actual,
                    p_60=0.8,
                    minutes=70,
                    player=10 + p,
                )
            )
            live_xpts = float(actual) + 0.1 if gw < 6 else float(actual) + 8.0
            live_p60 = 0.8 if gw < 6 else 0.1
            rows.append(
                _row(
                    run_kind="live",
                    gw=gw,
                    xpts=live_xpts,
                    actual_pts=actual,
                    p_60=live_p60,
                    minutes=70,
                    player=10 + p,
                )
            )
    return pl.DataFrame(rows)


def test_load_committed_drift_config() -> None:
    cfg = load_drift_monitor_config()
    assert cfg.horizon == 1
    assert cfg.rolling_window >= 1
    assert cfg.consecutive_beyond >= 1
    assert "rmse" in cfg.metrics


def test_beyond_threshold_with_std() -> None:
    assert beyond_threshold(5.0, mean=1.0, std=1.0, z_score=2.0)
    assert not beyond_threshold(2.5, mean=1.0, std=1.0, z_score=2.0)


def test_beyond_threshold_zero_std() -> None:
    assert beyond_threshold(1.1, mean=1.0, std=0.0, z_score=2.0)
    assert not beyond_threshold(1.0, mean=1.0, std=0.0, z_score=2.0)


def test_stable_series_never_alerts() -> None:
    report = evaluate_drift(_stable_ledger(), model_id=_MODEL, config=_cfg())
    assert not report.alerted
    assert report.alerts == ()


def test_step_change_alerts_after_k_gameweeks() -> None:
    # window=3, k=3 → first full rolling beyond at GW 8 (6,7,8 all high);
    # need 3 consecutive beyond flags ending at the latest GW.
    cfg = _cfg(rolling_window=3, consecutive_beyond=3)
    report = evaluate_drift(_step_change_ledger(), model_id=_MODEL, config=cfg)
    assert report.alerted
    metrics = {alert.metric for alert in report.alerts}
    assert "rmse" in metrics
    rmse_alert = next(a for a in report.alerts if a.metric == "rmse")
    assert rmse_alert.consecutive >= 3
    body = format_drift_alerts(report)
    assert "Alert only" in body
    assert "rmse" in body


def test_step_change_no_alert_before_k() -> None:
    # Truncate live history so the beyond streak is shorter than k.
    # With window=3, degradation from GW6 gives at most 2 consecutive
    # beyond flags by GW7 (rolling at GW6–7 only).
    ledger = _step_change_ledger(n_gw=7)
    cfg = _cfg(rolling_window=3, consecutive_beyond=3)
    report = evaluate_drift(ledger, model_id=_MODEL, config=cfg)
    assert not report.alerted


def test_per_gw_and_rolling_shapes() -> None:
    ledger = _stable_ledger(n_gw=5)
    live = per_gw_metrics(ledger, run_kind="live", horizon=1, model_id=_MODEL)
    assert live.height == 5
    assert set(live["rmse"].to_list())  # non-empty numeric
    rolled = rolling_live_metrics(live, window=3, metrics=("rmse",))
    assert "rolling_rmse" in rolled.columns
    assert rolled["rolling_rmse"].null_count() == 2


def test_backtest_distributions_std() -> None:
    ledger = _stable_ledger(n_gw=6)
    backtest = per_gw_metrics(ledger, run_kind="backtest", horizon=1, model_id=_MODEL)
    dists = {d.metric: d for d in backtest_distributions(backtest, metrics=("rmse",))}
    assert "rmse" in dists
    assert dists["rmse"].n == 6
    assert dists["rmse"].std >= 0.0


def test_load_drift_config_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_drift_monitor_config(tmp_path / "missing.yaml")
