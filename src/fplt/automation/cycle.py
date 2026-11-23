"""Catch up checked gameweeks, and refresh the one deadline inside three hours.

A gameweek counts as done when every requested model has a plan parquet for
that deadline and horizon. Catch-up walks the missing checked gameweeks in
order, so a week that was skipped while the process was down is picked up on
the next run. The pre-deadline refresh does the same for a single upcoming deadline
and does not walk earlier weeks.

When availability signals (``player_status``, ``predicted_lineups``,
``player_injuries``) move after a plan was written, ``pre_deadline`` retires
the stale plan and re-predicts / re-optimises (FDB-253 late-refresh hook).

When ``FPLT_ENTRY_ID`` is set (and ``FPLT_FPL_API_TOKEN`` for the live squad),
each plan starts from ``resolve_squad_state`` / ``load_current_team`` rather
than an empty YAML draft. Without an entry id the YAML behaviour is unchanged.

Plans use ``run_plan``'s live default chip policy
(``DEFAULT_LIVE_CHIP_ARM`` = optimised: season chip scheduler → MILP).
Horizon stays ``DEFAULT_PLAN_HORIZON`` (3).
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from fplt.automation.late_refresh import plans_stale_vs_signals, retire_plans
from fplt.automation.production import load_production_ops
from fplt.backtest.run import run_backtest
from fplt.data.benchmark_projections import archive_if_missing
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.spec import load_model_specs
from fplt.optimise.decisions import resolve_squad_state, run_plan
from fplt.optimise.plan import covered_horizon, plan_parquet
from fplt.types import GWKey

# Live list / horizon come from ``configs/ops/production.yaml`` (or env).
# Prefer ``load_production_ops()`` so overrides apply; this name stays for
# docs and callers that want the committed-default snapshot at import time.
PRODUCTION_MODELS = list(load_production_ops().models)
_WINDOW = timedelta(hours=3)
logger = logging.getLogger(__name__)


def pending_checked(
    store: DataStore,
    root: Path,
    models: list[str] | None = None,
    horizon: int | None = None,
) -> list[GWKey]:
    """Checked gameweeks that are still missing a plan, earliest first.

    Args:
        store: Staged tables, including ``gameweeks``.
        root: Data directory that contains ``plans/``.
        models: YAML spec names that must each have a plan. Defaults to
            ``load_production_ops().models``.
        horizon: Weeks the missing plan would cover. Defaults to
            ``load_production_ops().plan_horizon``.

    Returns:
        Deadlines whose ``data_checked`` flag is set and whose plan is absent.
    """
    ops = load_production_ops()
    chosen = ops.models if models is None else models
    steps = ops.plan_horizon if horizon is None else horizon
    frame = store.read("gameweeks").filter(pl.col("data_checked"))
    waiting: list[GWKey] = []
    seen: set[GWKey] = set()
    for season, gameweek in frame.select("season", "gw").iter_rows():
        key = GWKey(str(season), int(gameweek))
        if key in seen or plans_exist(root, chosen, key, steps):
            continue
        seen.add(key)
        waiting.append(key)
    return sorted(waiting)


def due_gameweek(store: DataStore, now: datetime) -> GWKey | None:
    """The earliest unchecked deadline inside the next three hours.

    Args:
        store: Staged tables, including ``gameweeks``.
        now: Clock reading, timezone-aware.

    Returns:
        The deadline to refresh, or ``None`` when nothing is due.
    """
    frame = store.read("gameweeks").filter(~pl.col("data_checked"))
    moment = _aware(now)
    due: list[tuple[datetime, GWKey]] = []
    for row in frame.iter_rows(named=True):
        deadline = _aware(row["deadline_time"])
        if moment < deadline <= moment + _WINDOW:
            due.append((deadline, GWKey(str(row["season"]), int(row["gw"]))))
    if not due:
        return None
    due.sort(key=lambda item: item[0])
    return due[0][1]


def catch_up(
    data_dir: Path,
    models: list[str] | None = None,
    horizon: int | None = None,
) -> list[GWKey]:
    """Fit, score, and plan every checked gameweek that has no plan yet.

    Args:
        data_dir: Root that contains ``staged/`` and receives ``ledger/`` and
            ``plans/``.
        models: YAML spec names. Defaults to ``load_production_ops().models``.
        horizon: Weeks each plan looks ahead. Defaults to
            ``load_production_ops().plan_horizon``.

    Returns:
        The gameweeks processed by this call, in play order.
    """
    ops = load_production_ops()
    chosen = list(ops.models) if models is None else models
    steps = ops.plan_horizon if horizon is None else horizon
    waiting = pending_checked(DataStore(data_dir), data_dir, chosen, steps)
    logger.info(
        "catch_up models=%s pending=%s horizon=%s",
        chosen,
        [str(key) for key in waiting],
        steps,
    )
    done: list[GWKey] = []
    for key in waiting:
        _play(data_dir, chosen, key, steps)
        done.append(key)
    return done


def pre_deadline(
    data_dir: Path,
    now: datetime,
    models: list[str] | None = None,
    horizon: int | None = None,
    *,
    refresh_if_stale: bool = True,
) -> GWKey | None:
    """Plan the deadline inside the next three hours when missing or stale.

    Args:
        data_dir: Root that contains ``staged/`` and receives the plan.
        now: Clock reading, timezone-aware.
        models: YAML spec names. Defaults to ``load_production_ops().models``.
        horizon: Weeks the plan looks ahead. Defaults to
            ``load_production_ops().plan_horizon``.
        refresh_if_stale: When true (default), retire write-once plans whose
            mtime is older than the latest ``player_status`` /
            ``predicted_lineups`` / ``player_injuries`` signal and rebuild.

    Returns:
        The deadline that was planned, or ``None`` when nothing was due.
    """
    ops = load_production_ops()
    chosen = list(ops.models) if models is None else models
    steps = ops.plan_horizon if horizon is None else horizon
    store = DataStore(data_dir)
    key = due_gameweek(store, now)
    if key is None:
        logger.info("pre_deadline: nothing due within %s", _WINDOW)
        return None
    # North-star N1: archive ep_next + OpenFPL before the deadline (write-once).
    archive_if_missing(data_dir, key, now=now)
    if plans_exist(data_dir, chosen, key, steps):
        if refresh_if_stale and plans_stale_vs_signals(
            data_dir, store, chosen, key, steps, now=now
        ):
            retired = retire_plans(data_dir, chosen, key, steps, retired_at=now)
            logger.info(
                "pre_deadline: retired %s stale plan(s) for %s; re-predicting",
                len(retired),
                key,
            )
        else:
            logger.info("pre_deadline: plan already exists for %s", key)
            return None
    logger.info("pre_deadline: planning %s models=%s", key, chosen)
    _play(data_dir, chosen, key, steps)
    return key


def reports(root: Path) -> str:
    """Concatenate every saved plan report.

    Args:
        root: Data directory that contains ``plans/``.

    Returns:
        The markdown, or an empty string when no report has been written.
    """
    folder = root / "plans"
    if not folder.exists():
        return ""
    parts = [
        path.read_text(encoding="utf-8") for path in sorted(folder.rglob("report.md"))
    ]
    return "\n".join(parts)


def _play(data_dir: Path, models: list[str], key: GWKey, horizon: int) -> None:
    logger.info("automation play %s models=%s horizon=%s", key, models, horizon)
    run_backtest(data_dir, key.season, key.gw, key.gw, models, steps=horizon)
    # Live squad when FPLT_ENTRY_ID is set; otherwise YAML draft / empty owned.
    view = PITData(DataStore(data_dir)).view(key)
    squad = resolve_squad_state(view, event=key.gw)
    for name in models:
        try:
            run_plan(data_dir, name, key.season, key.gw, horizon, squad=squad)
        except FileExistsError:
            logger.info("plan already exists for %s / %s; skipping", name, key)
            continue


def plans_exist(
    root: Path,
    models: list[str] | None,
    key: GWKey,
    horizon: int | None = None,
) -> bool:
    """Whether every model already has a plan for ``key`` at ``horizon``.

    Args:
        root: Data directory that contains ``plans/``.
        models: YAML spec names. ``None`` uses ``load_production_ops().models``.
        key: Deadline to look up.
        horizon: Weeks the plan covers. Weeks past 38 are not part of the file.
            Defaults to ``load_production_ops().plan_horizon``.

    Returns:
        True when each model's parquet is already on disk.
    """
    ops = load_production_ops()
    chosen = ops.models if models is None else models
    steps = ops.plan_horizon if horizon is None else horizon
    weeks = covered_horizon(key.gw, steps)
    if weeks < 1:
        return False
    specs = load_model_specs()
    return all(
        plan_parquet(root, specs[name].model_id, key.season, key.gw, weeks).exists()
        for name in chosen
    )


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
