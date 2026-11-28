"""Late pre-deadline re-predict hooks when availability inputs move (FDB-253).

Plans are write-once. When ``player_status``, ``predicted_lineups``, or
``player_injuries`` gain rows newer than an existing plan's mtime inside the
pre-deadline window, retire the plan directory so ``pre_deadline`` can rebuild.
No LLM press parsing; no odds rotation model — those stay deferred.
"""

from __future__ import annotations

import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from fplt.data.store import DataStore
from fplt.models.spec import load_model_specs
from fplt.optimise.plan import covered_horizon, plan_parquet
from fplt.store.catalog import scan_succeeded_runs
from fplt.store.plan_runs import plan_runs
from fplt.types import GWKey

logger = logging.getLogger(__name__)

# Staged tables whose ``available_at`` can invalidate a near-deadline plan.
_LATE_SIGNAL_TABLES: tuple[str, ...] = (
    "player_status",
    "predicted_lineups",
    "player_injuries",
)


def latest_late_signal_at(
    store: DataStore,
    key: GWKey,
    *,
    now: datetime | None = None,
) -> datetime | None:
    """Newest availability-signal ``available_at`` at or before the deadline.

    Scans ``player_status``, ``predicted_lineups``, and ``player_injuries`` for
    ``season == key.season`` with ``available_at <= min(deadline, now)``.

    Args:
        store: Staged tables.
        key: Upcoming deadline.
        now: Optional clock; defaults to current UTC. Caps the signal so
            post-deadline rows never count.

    Returns:
        The max ``available_at``, or ``None`` when no signal rows exist.
    """
    deadline = store.deadline(key)
    moment = _aware(now if now is not None else datetime.now(UTC))
    cutoff = min(deadline, moment)
    peaks: list[datetime] = []
    for name in _LATE_SIGNAL_TABLES:
        if not store.path(name).exists():
            continue
        frame = store.read(name).filter(pl.col("season") == key.season)
        if frame.is_empty() or "available_at" not in frame.columns:
            continue
        visible = frame.filter(pl.col("available_at") <= cutoff)
        if visible.is_empty():
            continue
        peak = visible.select(pl.col("available_at").max()).item()
        if peak is not None:
            peaks.append(_aware(peak))
    if not peaks:
        return None
    return max(peaks)


def artifact_mtime(path: Path) -> datetime | None:
    """UTC mtime of ``path``, or ``None`` when missing."""
    if not path.exists():
        return None
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)


def plans_stale_vs_signals(
    root: Path,
    store: DataStore,
    models: list[str],
    key: GWKey,
    horizon: int,
    *,
    now: datetime | None = None,
) -> bool:
    """Whether any model plan is older than the latest late availability signal.

    Args:
        root: Data directory containing ``plans/``.
        store: Staged tables.
        models: YAML spec names that must each have a plan.
        key: Deadline.
        horizon: Weeks the plan covers.
        now: Optional clock for signal cutoff.

    Returns:
        True when every model has a plan run and at least one model's newest
        plan run finished strictly before ``latest_late_signal_at``. False
        when plans are missing (caller should create them) or signals are not
        newer. Plan times come from ``runs/`` manifests (``finished_at``), not
        ``plans/`` file mtimes (data-model cut-over).
    """
    weeks = covered_horizon(key.gw, horizon)
    if weeks < 1:
        return False
    signal = latest_late_signal_at(store, key, now=now)
    if signal is None:
        return False
    catalog = scan_succeeded_runs(root)
    finished: list[datetime] = []
    for name in models:
        stamp = latest_plan_finished_at(root, name, key, weeks, catalog=catalog)
        if stamp is None:
            return False
        finished.append(stamp)
    return any(stamp < signal for stamp in finished)


def latest_plan_finished_at(
    root: Path,
    model: str,
    key: GWKey,
    weeks: int,
    *,
    catalog: pl.DataFrame | None = None,
) -> datetime | None:
    """UTC ``finished_at`` of the newest succeeded plan run, or ``None``.

    Args:
        root: Data directory containing ``runs/``.
        model: YAML spec name.
        key: Deadline.
        weeks: Covered plan weeks.
        catalog: Optional preloaded run catalog.

    Returns:
        The newest matching plan run's finish time, or ``None`` when absent.
    """
    rows = plan_runs(root, spec_label=model, key=key, horizon=weeks, catalog=catalog)
    if rows.is_empty():
        return None
    return _aware(rows.get_column("finished_at")[-1])


def retire_plans(
    root: Path,
    models: list[str],
    key: GWKey,
    horizon: int,
    *,
    retired_at: datetime | None = None,
) -> list[Path]:
    """Move existing plan directories aside so a fresh write-once plan can land.

    Renames ``plans/<model_id>/as_of=…/`` to
    ``plans/<model_id>/superseded/as_of=…_at_<UTC stamp>/`` (including
    sibling ``report.md`` when present).

    Args:
        root: Data directory containing ``plans/``.
        models: YAML spec names.
        key: Deadline.
        horizon: Weeks the plan covers.
        retired_at: Stamp embedded in the superseded directory name.

    Returns:
        Paths of directories that were moved.
    """
    weeks = covered_horizon(key.gw, horizon)
    if weeks < 1:
        return []
    when = _aware(retired_at if retired_at is not None else datetime.now(UTC))
    stamp = when.strftime("%Y%m%dT%H%M%SZ")
    specs = load_model_specs()
    moved: list[Path] = []
    for name in models:
        path = plan_parquet(root, specs[name].model_id, key.season, key.gw, weeks)
        folder = path.parent
        if not folder.exists():
            continue
        dest_root = folder.parent / "superseded"
        dest_root.mkdir(parents=True, exist_ok=True)
        dest = dest_root / f"{folder.name}_at_{stamp}"
        if dest.exists():
            dest = dest_root / f"{folder.name}_at_{stamp}_{len(moved)}"
        shutil.move(str(folder), str(dest))
        logger.info("retired stale plan %s → %s", folder, dest)
        moved.append(dest)
    return moved


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = [
    "latest_plan_finished_at",
    "artifact_mtime",
    "latest_late_signal_at",
    "plans_stale_vs_signals",
    "retire_plans",
]
