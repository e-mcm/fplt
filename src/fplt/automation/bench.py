"""Nightly performance bench: results, rolling baseline, regression alerts.

Writes write-once ``bench/date=<UTC>/results.json`` under a lab root (R2
``fplt-lab`` in production; a local directory in tests). The rolling median
baseline is computed at read time from prior nights. Alert (ntfy) when a
tracked phase is more than ``regression_pct`` slower than baseline for
``consecutive_nights`` nights — never fail PRs on timing (#312).
"""

from __future__ import annotations

import json
import logging
import statistics
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field

from fplt.automation.notify import notify
from fplt.automation.runbook import OPS_13_PERF_REGRESSION
from fplt.schemas.bench import (
    BENCH_RESULTS_SCHEMA_VERSION,
    BenchCaseDuration,
    BenchResults,
)
from fplt.settings import Settings, get_settings, secret_value
from fplt.utils.timing import SHARED_SUBPHASES

logger = logging.getLogger(__name__)

_DEFAULT_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "ops" / "bench.yaml"

BENCH_PREFIX = "bench"
RESULTS_FILENAME = "results.json"


class BenchExistsError(FileExistsError):
    """Raised when a write-once bench results path already exists."""


class BenchConfig(BaseModel):
    """Thresholds for nightly bench regression and ops runtime budget."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    baseline_window: int = Field(default=14, ge=1)
    regression_pct: float = Field(default=50.0, ge=0.0)
    consecutive_nights: int = Field(default=3, ge=1)
    ops_budget_pct: float = Field(default=50.0, gt=0.0, le=100.0)
    ops_timeout_minutes: int = Field(default=90, ge=1)
    runner_memory_gb: float = Field(default=16.0, gt=0.0)
    rss_warn_pct: float = Field(default=70.0, gt=0.0, le=100.0)
    tracked_phases: tuple[str, ...] = (
        "fit",
        "predict",
        "plan",
        "points_pmf",
        "optimise/milp",
        "mc_bench_total",
        "pytest_suite",
    )

    def rss_warn_kb(self) -> int:
        """Absolute KiB ceiling for peak-RSS WARN (#372)."""
        return int(self.runner_memory_gb * self.rss_warn_pct / 100.0 * 1024 * 1024)


@dataclass(frozen=True, slots=True)
class PhaseRegression:
    """One tracked phase that breached the consecutive-night rule."""

    phase: str
    consecutive: int
    baseline: float
    threshold: float
    values: tuple[float, ...]
    dates: tuple[str, ...]
    detail: str


@dataclass(frozen=True, slots=True)
class RegressionReport:
    """Result of evaluating the latest nights against the rolling baseline."""

    alerts: tuple[PhaseRegression, ...]
    baseline: Mapping[str, float]
    history_dates: tuple[str, ...]

    @property
    def alerted(self) -> bool:
        """True when at least one phase should notify."""
        return bool(self.alerts)


@dataclass(frozen=True, slots=True)
class OpsBudgetStatus:
    """Ops tick total vs configured fraction of the workflow timeout."""

    total_s: float
    budget_s: float
    timeout_minutes: int
    budget_pct: float
    over_budget: bool

    @property
    def warning_line(self) -> str | None:
        """Markdown WARNING line when over budget, else ``None``."""
        if not self.over_budget:
            return None
        return (
            f"**WARNING:** ops total_s={self.total_s:.1f}s exceeds "
            f"{self.budget_pct:.0f}% of workflow timeout "
            f"({self.timeout_minutes} min → budget {self.budget_s:.0f}s)."
        )


def default_bench_config_path() -> Path:
    """Return the repo-root path to ``configs/ops/bench.yaml``."""
    return _DEFAULT_CONFIG


def load_bench_config(path: Path | None = None) -> BenchConfig:
    """Load bench thresholds from YAML (default ``configs/ops/bench.yaml``).

    Args:
        path: Optional override path.

    Returns:
        Validated ``BenchConfig``.
    """
    source = default_bench_config_path() if path is None else path
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if loaded is None:
        return BenchConfig()
    return BenchConfig.model_validate(cast(dict[str, Any], loaded))


def bench_date_dir(lab_root: Path, day: date | str) -> Path:
    """Return ``lab_root/bench/date=<YYYY-MM-DD>``."""
    label = day if isinstance(day, str) else day.isoformat()
    return lab_root / BENCH_PREFIX / f"date={label}"


def bench_results_path(lab_root: Path, day: date | str) -> Path:
    """Return the write-once ``results.json`` path for ``day``."""
    return bench_date_dir(lab_root, day) / RESULTS_FILENAME


def write_bench_results(
    lab_root: Path,
    results: BenchResults,
    *,
    exist_ok: bool = False,
) -> Path:
    """Write ``results.json`` write-once under ``bench/date=<UTC>/``.

    Args:
        lab_root: Local directory standing in for ``fplt-lab``.
        results: Validated nightly snapshot.
        exist_ok: When True, skip if the path already exists (idempotent).

    Returns:
        Path written (or the existing path when ``exist_ok``).

    Raises:
        BenchExistsError: Path exists and ``exist_ok`` is False.
    """
    path = bench_results_path(lab_root, results.date)
    if path.exists():
        if exist_ok:
            return path
        raise BenchExistsError(f"bench results already exist: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = results.model_dump_json_doc()
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def load_bench_results(path: Path) -> BenchResults:
    """Load and validate one ``results.json`` file.

    Args:
        path: Path to a nightly results document.

    Returns:
        Validated ``BenchResults``.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    return BenchResults.model_validate(payload)


def list_bench_results(
    lab_root: Path,
    *,
    limit: int | None = None,
) -> list[BenchResults]:
    """Load nightly results newest-first under ``lab_root/bench/``.

    Args:
        lab_root: Local lab root.
        limit: Max nights to return (newest first). ``None`` = all.

    Returns:
        Validated results sorted by ``date`` descending.
    """
    root = lab_root / BENCH_PREFIX
    if not root.is_dir():
        return []
    found: list[BenchResults] = []
    for child in sorted(root.glob("date=*/" + RESULTS_FILENAME), reverse=True):
        try:
            found.append(load_bench_results(child))
        except (OSError, json.JSONDecodeError, ValueError) as exc:
            logger.warning("skipping unreadable bench results %s: %s", child, exc)
    if limit is not None:
        return found[: max(0, int(limit))]
    return found


def rolling_baseline(
    history: Sequence[BenchResults],
    *,
    window: int,
    exclude_latest: bool = True,
) -> dict[str, float]:
    """Compute per-phase median over the last ``window`` prior nights.

    Args:
        history: Results newest-first (as from ``list_bench_results``).
        window: Max nights included in the median.
        exclude_latest: When True, skip the newest night (the one under test).

    Returns:
        Phase → median seconds. Phases with no samples are omitted.
    """
    if window < 1:
        return {}
    start = 1 if exclude_latest and history else 0
    sample = list(history[start : start + window])
    by_phase: dict[str, list[float]] = {}
    for night in sample:
        for name, seconds in night.phases.items():
            by_phase.setdefault(name, []).append(float(seconds))
    return {
        name: float(statistics.median(values))
        for name, values in by_phase.items()
        if values
    }


def _terminal_streak(flags: Sequence[bool]) -> int:
    streak = 0
    for flag in reversed(flags):
        if not flag:
            break
        streak += 1
    return streak


def evaluate_regression(
    history: Sequence[BenchResults],
    *,
    config: BenchConfig | None = None,
) -> RegressionReport:
    """Alert when tracked phases stay slower than baseline for K nights.

    Shared runners are noisy: the baseline is the rolling median of the prior
    ``baseline_window`` nights (excluding the newest night). The newest
    ``consecutive_nights`` results are each compared to that baseline; alert
    only when the terminal streak of breaches reaches K. Never raises on a
    regression.

    Args:
        history: Results newest-first. Needs prior nights for a baseline and
            ``consecutive_nights`` recent samples for a possible alert.
        config: Thresholds; defaults to ``configs/ops/bench.yaml``.

    Returns:
        ``RegressionReport`` (possibly with empty alerts).
    """
    cfg = load_bench_config() if config is None else config
    if not history:
        return RegressionReport(alerts=(), baseline={}, history_dates=())
    dates = tuple(item.date for item in history)
    baseline = rolling_baseline(
        history, window=cfg.baseline_window, exclude_latest=True
    )
    recent = list(history[: cfg.consecutive_nights])
    alerts: list[PhaseRegression] = []
    for phase in cfg.tracked_phases:
        base = baseline.get(phase)
        if base is None or base <= 0:
            continue
        threshold = base * (1.0 + cfg.regression_pct / 100.0)
        # Newest-first flags; terminal streak = consecutive from tonight back.
        flags: list[bool] = []
        values: list[float | None] = []
        flag_dates: list[str] = []
        for night in recent:
            value = night.phases.get(phase)
            if value is None:
                flags.append(False)
                values.append(None)
                flag_dates.append(night.date)
                continue
            flags.append(float(value) > threshold)
            values.append(float(value))
            flag_dates.append(night.date)
        streak = _terminal_streak(flags)
        if streak < cfg.consecutive_nights:
            continue
        # values/dates were newest-first; reverse for chronological detail.
        streak_values = tuple(
            reversed([float(v) for v in values[:streak] if isinstance(v, float)])
        )
        streak_dates = tuple(reversed(flag_dates[:streak]))
        alerts.append(
            PhaseRegression(
                phase=phase,
                consecutive=streak,
                baseline=base,
                threshold=threshold,
                values=streak_values,
                dates=streak_dates,
                detail=(
                    f"{phase}: {streak} consecutive nights above "
                    f"{cfg.regression_pct:.0f}% over baseline "
                    f"(baseline={base:.2f}s, threshold={threshold:.2f}s, "
                    f"values={[round(v, 2) for v in streak_values]}, "
                    f"dates={list(streak_dates)})"
                ),
            )
        )
    return RegressionReport(
        alerts=tuple(alerts),
        baseline=baseline,
        history_dates=dates,
    )


def format_regression_alerts(report: RegressionReport) -> str:
    """Render a short ntfy body for performance regressions."""
    if not report.alerts:
        return "No performance regressions."
    lines = [
        "fplt nightly bench regression (alert only — do not fail PRs).",
    ]
    for alert in report.alerts:
        lines.append(f"- {alert.detail}")
    return "\n".join(lines)


def notify_regression_alerts(
    settings: Settings,
    report: RegressionReport,
    *,
    post: Any = None,
) -> bool:
    """POST regression alerts via ntfy when configured.

    Args:
        settings: Process settings (``ntfy_url``).
        report: Evaluation result.
        post: Optional fake HTTP sender for tests.

    Returns:
        True when a notification was sent.
    """
    if not report.alerted:
        return False
    return notify(
        secret_value(settings.ntfy_url),
        "fplt bench regression",
        format_regression_alerts(report),
        post=post,
        runbook_anchor=OPS_13_PERF_REGRESSION,
    )


def format_trend_report(
    history: Sequence[BenchResults],
    *,
    config: BenchConfig | None = None,
    limit: int = 14,
) -> str:
    """Render a plain-text trend of recent nights vs rolling median.

    Args:
        history: Results newest-first.
        config: Thresholds for tracked phases / window.
        limit: Max nights to show.

    Returns:
        Multi-line report for ``fplt perf-report --trend``.
    """
    cfg = load_bench_config() if config is None else config
    nights = list(history[: max(0, int(limit))])
    if not nights:
        return "perf trend: no bench/date=*/results.json found"
    baseline = rolling_baseline(
        nights, window=cfg.baseline_window, exclude_latest=False
    )
    lines = [
        f"perf trend: {len(nights)} night(s) (baseline_window={cfg.baseline_window})",
        "",
        f"{'date':<12} {'phase':<18} {'seconds':>10} {'vs_med%':>8}",
        f"{'-' * 12} {'-' * 18} {'-' * 10} {'-' * 8}",
    ]
    for night in nights:
        for phase in cfg.tracked_phases:
            value = night.phases.get(phase)
            if value is None:
                continue
            med = baseline.get(phase)
            if med and med > 0:
                delta = 100.0 * (float(value) - med) / med
                delta_s = f"{delta:+.1f}"
            else:
                delta_s = "n/a"
            lines.append(
                f"{night.date:<12} {phase:<18} {float(value):10.2f} {delta_s:>8}"
            )
    if baseline:
        lines.append("")
        lines.append("rolling median baseline:")
        for phase in cfg.tracked_phases:
            if phase in baseline:
                lines.append(f"  {phase}: {baseline[phase]:.2f}s")
    return "\n".join(lines)


def ops_budget_status(
    total_s: float,
    *,
    config: BenchConfig | None = None,
    timeout_minutes: int | None = None,
) -> OpsBudgetStatus:
    """Compare an ops ``total_s`` to the configured runtime budget.

    Args:
        total_s: Ops tick / catch-up wall-clock total seconds.
        config: Thresholds; defaults to committed YAML.
        timeout_minutes: Override for the workflow job timeout.

    Returns:
        ``OpsBudgetStatus`` with ``over_budget`` when total exceeds the share.
    """
    cfg = load_bench_config() if config is None else config
    minutes = (
        int(timeout_minutes)
        if timeout_minutes is not None
        else int(cfg.ops_timeout_minutes)
    )
    budget_s = minutes * 60.0 * (cfg.ops_budget_pct / 100.0)
    return OpsBudgetStatus(
        total_s=float(total_s),
        budget_s=budget_s,
        timeout_minutes=minutes,
        budget_pct=cfg.ops_budget_pct,
        over_budget=float(total_s) > budget_s,
    )


def build_bench_results(
    *,
    day: date | str,
    phases: Mapping[str, float],
    recorded_at: datetime | None = None,
    git_sha: str | None = None,
    mc_bench: Mapping[str, float] | None = None,
    pytest_top25: Sequence[Mapping[str, Any]] | None = None,
    meta: Mapping[str, Any] | None = None,
) -> BenchResults:
    """Build a validated nightly results document.

    Args:
        day: UTC calendar date (``date`` or ``YYYY-MM-DD``).
        phases: Named wall-clock seconds (fit / predict / plan / …).
        recorded_at: Timestamp; defaults to now UTC.
        git_sha: Optional commit sha.
        mc_bench: Optional per-step mc-bench milliseconds.
        pytest_top25: Optional ``{nodeid, seconds}`` rows.
        meta: Optional free-form identity.

    Returns:
        Validated ``BenchResults``.
    """
    date_s = day if isinstance(day, str) else day.isoformat()
    cases: list[BenchCaseDuration] = []
    for row in pytest_top25 or ():
        cases.append(
            BenchCaseDuration(
                nodeid=str(row["nodeid"]),
                seconds=float(row["seconds"]),
            )
        )
    return BenchResults(
        schema_version=BENCH_RESULTS_SCHEMA_VERSION,
        date=date_s,
        recorded_at=recorded_at or datetime.now(UTC),
        git_sha=git_sha,
        phases={str(k): float(v) for k, v in phases.items()},
        mc_bench={str(k): float(v) for k, v in (mc_bench or {}).items()},
        pytest_top25=cases,
        meta=dict(meta or {}),
    )


def consolidate_fixture_phases(
    timing_phases: Mapping[str, float],
) -> dict[str, float]:
    """Roll nested ops / walk-forward phase names into tracked bench keys.

    Sums ``fit`` / ``fit/…``, ``predict`` / ``predict/…``, and ``plan`` /
    ``plan/…`` into bucket totals, and keeps ``optimise/milp`` when present.

    Args:
        timing_phases: Raw phase → seconds from a timing session / record.

    Returns:
        Consolidated phases suitable for ``BenchResults.phases``.
    """
    out: dict[str, float] = {}
    for name, seconds in timing_phases.items():
        value = float(seconds)
        if value <= 0:
            continue
        # Keep MILP even though it is a shared nested name elsewhere.
        if name == "optimise/milp":
            out["optimise/milp"] = out.get("optimise/milp", 0.0) + value
            continue
        if name == "points_pmf":
            out["points_pmf"] = out.get("points_pmf", 0.0) + value
            continue
        if name in SHARED_SUBPHASES:
            continue
        if name == "fit" or name.startswith("fit/"):
            out["fit"] = out.get("fit", 0.0) + value
        elif name == "predict" or name.startswith("predict/"):
            out["predict"] = out.get("predict", 0.0) + value
        elif name == "plan" or name.startswith("plan/"):
            out["plan"] = out.get("plan", 0.0) + value
    return {k: v for k, v in out.items() if v > 0}


def run_regression_check(
    lab_root: Path,
    *,
    settings: Settings | None = None,
    config: BenchConfig | None = None,
    notify_alerts: bool = True,
    post: Any = None,
) -> RegressionReport:
    """Load history, evaluate regressions, optionally notify.

    Args:
        lab_root: Local lab root with ``bench/date=*/results.json``.
        settings: Optional settings for ntfy.
        config: Optional thresholds.
        notify_alerts: When True, POST ntfy on alert.
        post: Optional fake HTTP sender for tests.

    Returns:
        ``RegressionReport`` (never raises on a regression).
    """
    cfg = load_bench_config() if config is None else config
    history = list_bench_results(lab_root)
    report = evaluate_regression(history, config=cfg)
    if notify_alerts and report.alerted:
        cfg_settings = get_settings() if settings is None else settings
        notify_regression_alerts(cfg_settings, report, post=post)
    return report


__all__ = [
    "BENCH_PREFIX",
    "RESULTS_FILENAME",
    "BenchConfig",
    "BenchExistsError",
    "OpsBudgetStatus",
    "PhaseRegression",
    "RegressionReport",
    "bench_date_dir",
    "bench_results_path",
    "build_bench_results",
    "consolidate_fixture_phases",
    "default_bench_config_path",
    "evaluate_regression",
    "format_regression_alerts",
    "format_trend_report",
    "list_bench_results",
    "load_bench_config",
    "load_bench_results",
    "notify_regression_alerts",
    "ops_budget_status",
    "rolling_baseline",
    "run_regression_check",
    "write_bench_results",
]
