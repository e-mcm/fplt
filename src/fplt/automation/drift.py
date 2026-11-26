"""Live forecast drift monitor vs backtest distribution (ops-09).

After each checked gameweek, compare rolling live RMSE and event Brier
(minutes ≥ 60, goal, clean sheet) for the production model against the
backtest per-GW distribution at the same horizon. Alert when the rolling
window sits beyond a configured z-threshold for ``k`` consecutive
gameweeks. Alert only — never switch models.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import polars as pl
import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

from fplt.automation.notify import notify_recommendation
from fplt.backtest.metrics import overall_summary
from fplt.settings import Settings

logger = logging.getLogger(__name__)

_DEFAULT_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "ops" / "drift_monitor.yaml"
)
DRIFT_METRICS: tuple[str, ...] = ("rmse", "brier_60", "brier_goal", "brier_cs")


class MetricThreshold(BaseModel):
    """Z-score threshold for one drift metric."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    z_score: float = Field(gt=0.0)


class DriftMonitorConfig(BaseModel):
    """Thresholds for the live vs backtest drift monitor."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    horizon: int = Field(ge=1)
    rolling_window: int = Field(ge=1)
    consecutive_beyond: int = Field(ge=1)
    metrics: dict[str, MetricThreshold]

    @field_validator("metrics")
    @classmethod
    def _known_metrics(
        cls, value: dict[str, MetricThreshold]
    ) -> dict[str, MetricThreshold]:
        if not value:
            raise ValueError("metrics must be non-empty")
        unknown = sorted(set(value) - set(DRIFT_METRICS))
        if unknown:
            raise ValueError(f"unknown drift metrics: {unknown}")
        return value


@dataclass(frozen=True)
class BacktestDistribution:
    """Mean and std of a backtest per-GW metric series."""

    metric: str
    mean: float
    std: float
    n: int


@dataclass(frozen=True)
class DriftAlert:
    """One metric that breached the consecutive-beyond rule."""

    metric: str
    model_id: str
    horizon: int
    consecutive: int
    threshold: float
    rolling_values: tuple[float, ...]
    gameweeks: tuple[int, ...]
    detail: str


@dataclass(frozen=True)
class DriftReport:
    """Full drift evaluation for one production model."""

    model_id: str
    horizon: int
    alerts: tuple[DriftAlert, ...]
    rolling: pl.DataFrame
    distributions: tuple[BacktestDistribution, ...]

    @property
    def alerted(self) -> bool:
        """True when at least one metric should notify."""
        return bool(self.alerts)


def default_drift_monitor_path() -> Path:
    """Return the repo-root path to ``configs/ops/drift_monitor.yaml``."""
    return _DEFAULT_PATH


def load_drift_monitor_config(path: Path | None = None) -> DriftMonitorConfig:
    """Load drift thresholds from YAML (default ``configs/ops/drift_monitor.yaml``).

    Args:
        path: Config file. Defaults to the committed ops path.

    Returns:
        Validated ``DriftMonitorConfig``.

    Raises:
        FileNotFoundError: The YAML path does not exist.
        ValueError: The file is empty, not a mapping, or fails validation.
    """
    source = default_drift_monitor_path() if path is None else path
    if not source.is_file():
        raise FileNotFoundError(f"drift monitor config not found: {source}")
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{source.name} must contain a mapping")
    return DriftMonitorConfig.model_validate(cast(dict[str, Any], loaded))


def per_gw_metrics(
    ledger: pl.DataFrame,
    *,
    run_kind: str,
    horizon: int,
    model_id: str | None = None,
) -> pl.DataFrame:
    """Compute per-target-GW RMSE / event Brier for one run kind.

    Args:
        ledger: Scored prediction ledger (``LedgerSchema`` columns).
        run_kind: ``live`` or ``backtest``.
        horizon: Prediction horizon to keep.
        model_id: Optional model filter.

    Returns:
        One row per ``target_gw`` with ``rmse``, ``brier_60``, ``brier_goal``,
        ``brier_cs`` (null when a metric cannot be scored). Empty when no
        scored rows match.
    """
    filtered = ledger.filter(
        (pl.col("run_kind") == run_kind) & (pl.col("horizon") == horizon)
    )
    if model_id is not None:
        filtered = filtered.filter(pl.col("model_id") == model_id)
    scored = filtered.filter(pl.col("actual_pts").is_not_null())
    if scored.is_empty():
        return _empty_per_gw()
    parts = scored.partition_by(["target_season", "target_gw"], as_dict=True)
    rows: list[dict[str, object]] = []
    for key, group in parts.items():
        season, gw = _season_gw_key(key)
        summary = overall_summary(group)
        if summary.is_empty():
            continue
        row = summary.row(0, named=True)
        rows.append(
            {
                "model_id": row["model_id"],
                "target_season": season,
                "target_gw": gw,
                "horizon": horizon,
                "run_kind": run_kind,
                "rmse": row.get("rmse"),
                "brier_60": row.get("brier_60"),
                "brier_goal": row.get("brier_goal"),
                "brier_cs": row.get("brier_cs"),
                "n": row.get("n"),
            }
        )
    if not rows:
        return _empty_per_gw()
    return pl.DataFrame(rows).sort(["target_season", "target_gw"])


def backtest_distributions(
    backtest_per_gw: pl.DataFrame,
    metrics: Sequence[str] = DRIFT_METRICS,
) -> list[BacktestDistribution]:
    """Estimate mean/std of each metric from backtest per-GW rows.

    Args:
        backtest_per_gw: Output of ``per_gw_metrics`` for ``run_kind=backtest``.
        metrics: Metric column names.

    Returns:
        One ``BacktestDistribution`` per metric with at least one non-null
        value. Std is 0 when only a single observation exists.
    """
    out: list[BacktestDistribution] = []
    for metric in metrics:
        if metric not in backtest_per_gw.columns:
            continue
        values = [float(v) for v in backtest_per_gw[metric].to_list() if v is not None]
        if not values:
            continue
        mean = sum(values) / len(values)
        if len(values) == 1:
            std = 0.0
        else:
            var = sum((v - mean) ** 2 for v in values) / (len(values) - 1)
            std = var**0.5
        out.append(
            BacktestDistribution(metric=metric, mean=mean, std=std, n=len(values))
        )
    return out


def rolling_live_metrics(
    live_per_gw: pl.DataFrame,
    *,
    window: int,
    metrics: Sequence[str] = DRIFT_METRICS,
) -> pl.DataFrame:
    """Rolling mean of live per-GW metrics over ``window`` consecutive GWs.

    Args:
        live_per_gw: Output of ``per_gw_metrics`` for ``run_kind=live``.
        window: Rolling window length (≥ 1).
        metrics: Metric columns to roll.

    Returns:
        Live rows with ``rolling_<metric>`` columns (null until the window
        is full). Sorted by season then gameweek.
    """
    if window < 1:
        raise ValueError("window must be >= 1")
    if live_per_gw.is_empty():
        return live_per_gw
    ordered = live_per_gw.sort(["target_season", "target_gw"])
    exprs = [
        pl.col(metric)
        .cast(pl.Float64)
        .rolling_mean(window_size=window, min_samples=window)
        .alias(f"rolling_{metric}")
        for metric in metrics
        if metric in ordered.columns
    ]
    if not exprs:
        return ordered
    return ordered.with_columns(exprs)


def evaluate_drift(
    ledger: pl.DataFrame,
    *,
    model_id: str,
    config: DriftMonitorConfig | None = None,
) -> DriftReport:
    """Compare rolling live metrics to the backtest distribution.

    Args:
        ledger: Combined live + backtest scored ledger rows.
        model_id: Production model id to monitor.
        config: Thresholds. Defaults to ``configs/ops/drift_monitor.yaml``.

    Returns:
        ``DriftReport`` with alerts (possibly empty). Never mutates models.
    """
    cfg = load_drift_monitor_config() if config is None else config
    metric_names = tuple(cfg.metrics.keys())
    live = per_gw_metrics(
        ledger, run_kind="live", horizon=cfg.horizon, model_id=model_id
    )
    backtest = per_gw_metrics(
        ledger, run_kind="backtest", horizon=cfg.horizon, model_id=model_id
    )
    dists = backtest_distributions(backtest, metrics=metric_names)
    dist_by_metric = {item.metric: item for item in dists}
    rolled = rolling_live_metrics(live, window=cfg.rolling_window, metrics=metric_names)
    alerts = _alerts_from_rolling(
        rolled,
        model_id=model_id,
        horizon=cfg.horizon,
        config=cfg,
        dist_by_metric=dist_by_metric,
    )
    return DriftReport(
        model_id=model_id,
        horizon=cfg.horizon,
        alerts=tuple(alerts),
        rolling=rolled,
        distributions=tuple(dists),
    )


def format_drift_alerts(report: DriftReport) -> str:
    """Render a short alert body for ntfy / SMTP."""
    if not report.alerts:
        return f"No drift alerts for model_id={report.model_id}."
    lines = [
        f"Live drift alert for model_id={report.model_id} "
        f"(horizon={report.horizon}). Alert only — do not auto-switch.",
    ]
    for alert in report.alerts:
        lines.append(f"- {alert.detail}")
    return "\n".join(lines)


def notify_drift_alerts(
    settings: Settings,
    report: DriftReport,
) -> bool:
    """Push drift alerts via the shared recommendation notify path.

    Args:
        settings: App settings (ntfy / SMTP).
        report: Drift evaluation result.

    Returns:
        True when a notification was sent.
    """
    if not report.alerted:
        return False
    return notify_recommendation(
        settings,
        "fplt drift alert",
        format_drift_alerts(report),
    )


def run_live_drift_monitor(
    data_dir: Path,
    *,
    model_names: Sequence[str] | None = None,
    settings: Settings | None = None,
    notify: bool = True,
    config: DriftMonitorConfig | None = None,
) -> list[DriftReport]:
    """Evaluate drift for production models against the local ledger view.

    Alert only — never retargets ``configs/ops/production.yaml``. Missing or
    empty ledgers yield empty reports without error.

    Args:
        data_dir: ``$FPLT_DATA_DIR`` root.
        model_names: ModelSpec names. Defaults to production ops models.
        settings: Optional settings for notify.
        notify: When True, push alerts via ntfy/SMTP.
        config: Thresholds. Defaults to committed YAML.

    Returns:
        One ``DriftReport`` per resolved model id.
    """
    from fplt.automation.production import load_production_ops
    from fplt.backtest.ledger import read_predictions
    from fplt.models.spec import load_model_specs
    from fplt.settings import get_settings

    cfg = get_settings() if settings is None else settings
    names = (
        list(load_production_ops().models) if model_names is None else list(model_names)
    )
    specs = load_model_specs()
    ledger = read_predictions(data_dir)
    reports: list[DriftReport] = []
    for name in names:
        if name not in specs:
            logger.warning("drift monitor: ModelSpec %r missing; skip", name)
            continue
        model_id = specs[name].model_id
        report = evaluate_drift(ledger, model_id=model_id, config=config)
        reports.append(report)
        if notify and report.alerted:
            notify_drift_alerts(cfg, report)
    return reports


_BEYOND_EPS = 1.0e-9


def beyond_threshold(
    value: float,
    *,
    mean: float,
    std: float,
    z_score: float,
) -> bool:
    """Return True when ``value`` exceeds ``mean + z_score * std``.

    A small absolute epsilon avoids false alerts when a stable live series
    matches the backtest mean within float noise (especially ``std == 0``).
    """
    limit = mean + z_score * std
    return value > limit + _BEYOND_EPS


def _alerts_from_rolling(
    rolled: pl.DataFrame,
    *,
    model_id: str,
    horizon: int,
    config: DriftMonitorConfig,
    dist_by_metric: Mapping[str, BacktestDistribution],
) -> list[DriftAlert]:
    if rolled.is_empty():
        return []
    alerts: list[DriftAlert] = []
    for metric, threshold in config.metrics.items():
        dist = dist_by_metric.get(metric)
        if dist is None:
            continue
        col = f"rolling_{metric}"
        if col not in rolled.columns:
            continue
        beyond_flags: list[bool] = []
        values: list[float | None] = []
        gws: list[int] = []
        for row in rolled.iter_rows(named=True):
            gws.append(int(row["target_gw"]))
            raw = row.get(col)
            if raw is None:
                beyond_flags.append(False)
                values.append(None)
                continue
            value = float(raw)
            values.append(value)
            beyond_flags.append(
                beyond_threshold(
                    value,
                    mean=dist.mean,
                    std=dist.std,
                    z_score=threshold.z_score,
                )
            )
        streak = _terminal_streak(beyond_flags)
        if streak < config.consecutive_beyond:
            continue
        streak_values = tuple(float(v) for v in values[-streak:] if v is not None)
        streak_gws = tuple(gws[-streak:])
        limit = (
            dist.mean if dist.std == 0.0 else dist.mean + threshold.z_score * dist.std
        )
        alerts.append(
            DriftAlert(
                metric=metric,
                model_id=model_id,
                horizon=horizon,
                consecutive=streak,
                threshold=limit,
                rolling_values=streak_values,
                gameweeks=streak_gws,
                detail=(
                    f"{metric}: rolling beyond "
                    f"{limit:.4f} for {streak} consecutive GWs "
                    f"{list(streak_gws)} (backtest mean={dist.mean:.4f}, "
                    f"std={dist.std:.4f}, z={threshold.z_score})"
                ),
            )
        )
    return alerts


def _terminal_streak(flags: Sequence[bool]) -> int:
    streak = 0
    for flag in reversed(flags):
        if not flag:
            break
        streak += 1
    return streak


def _season_gw_key(key: object) -> tuple[str, int]:
    if isinstance(key, tuple):
        items = cast(tuple[object, ...], key)
        if len(items) == 2:
            return str(items[0]), int(cast(Any, items[1]))
    raise TypeError(f"unexpected partition key: {key!r}")


def _empty_per_gw() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "model_id": pl.Utf8(),
            "target_season": pl.Utf8(),
            "target_gw": pl.Int64(),
            "horizon": pl.Int64(),
            "run_kind": pl.Utf8(),
            "rmse": pl.Float64(),
            "brier_60": pl.Float64(),
            "brier_goal": pl.Float64(),
            "brier_cs": pl.Float64(),
            "n": pl.Int64(),
        }
    )


__all__ = [
    "DRIFT_METRICS",
    "BacktestDistribution",
    "DriftAlert",
    "DriftMonitorConfig",
    "DriftReport",
    "MetricThreshold",
    "backtest_distributions",
    "beyond_threshold",
    "default_drift_monitor_path",
    "evaluate_drift",
    "format_drift_alerts",
    "load_drift_monitor_config",
    "notify_drift_alerts",
    "per_gw_metrics",
    "rolling_live_metrics",
    "run_live_drift_monitor",
]
