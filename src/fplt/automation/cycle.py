"""Catch up checked gameweeks, and refresh the one deadline inside three hours.

A gameweek counts as done when every requested model has a plan parquet for
that deadline and horizon. Catch-up walks the missing checked gameweeks in
order, so a week that was skipped while the process was down is picked up on
the next run. The pre-deadline refresh does the same for a single upcoming deadline
and does not walk earlier weeks.

When ``FPLT_ENTRY_ID`` is set (and ``FPLT_FPL_API_TOKEN`` for the live squad),
each plan starts from ``resolve_squad_state`` / ``load_current_team`` rather
than an empty YAML draft. Without an entry id the YAML behaviour is unchanged.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from fplt.backtest.run import run_backtest
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.spec import load_model_specs
from fplt.optimise.decisions import resolve_squad_state, run_plan
from fplt.optimise.plan import covered_horizon, plan_parquet
from fplt.types import GWKey

PRODUCTION_MODELS = ["rolling_mean_v1"]
_WINDOW = timedelta(hours=3)
logger = logging.getLogger(__name__)


def pending_checked(
    store: DataStore, root: Path, models: list[str], horizon: int = 1
) -> list[GWKey]:
    """Checked gameweeks that are still missing a plan, earliest first.

    Args:
        store: Staged tables, including ``gameweeks``.
        root: Data directory that contains ``plans/``.
        models: YAML spec names that must each have a plan.
        horizon: Weeks the missing plan would cover.

    Returns:
        Deadlines whose ``data_checked`` flag is set and whose plan is absent.
    """
    frame = store.read("gameweeks").filter(pl.col("data_checked"))
    waiting: list[GWKey] = []
    seen: set[GWKey] = set()
    for season, gameweek in frame.select("season", "gw").iter_rows():
        key = GWKey(str(season), int(gameweek))
        if key in seen or plans_exist(root, models, key, horizon):
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
    horizon: int = 1,
) -> list[GWKey]:
    """Fit, score, and plan every checked gameweek that has no plan yet.

    Args:
        data_dir: Root that contains ``staged/`` and receives ``ledger/`` and
            ``plans/``.
        models: YAML spec names. Defaults to ``rolling_mean_v1``.
        horizon: Weeks each plan looks ahead.

    Returns:
        The gameweeks processed by this call, in play order.
    """
    chosen = PRODUCTION_MODELS if models is None else models
    waiting = pending_checked(DataStore(data_dir), data_dir, chosen, horizon)
    logger.info(
        "catch_up models=%s pending=%s horizon=%s",
        chosen,
        [str(key) for key in waiting],
        horizon,
    )
    done: list[GWKey] = []
    for key in waiting:
        _play(data_dir, chosen, key, horizon)
        done.append(key)
    return done


def pre_deadline(
    data_dir: Path,
    now: datetime,
    models: list[str] | None = None,
    horizon: int = 1,
) -> GWKey | None:
    """Plan the deadline inside the next three hours, when it has no plan yet.

    Args:
        data_dir: Root that contains ``staged/`` and receives the plan.
        now: Clock reading, timezone-aware.
        models: YAML spec names. Defaults to ``rolling_mean_v1``.
        horizon: Weeks the plan looks ahead.

    Returns:
        The deadline that was planned, or ``None`` when nothing was due.
    """
    chosen = PRODUCTION_MODELS if models is None else models
    key = due_gameweek(DataStore(data_dir), now)
    if key is None:
        logger.info("pre_deadline: nothing due within %s", _WINDOW)
        return None
    if plans_exist(data_dir, chosen, key, horizon):
        logger.info("pre_deadline: plan already exists for %s", key)
        return None
    logger.info("pre_deadline: planning %s models=%s", key, chosen)
    _play(data_dir, chosen, key, horizon)
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


def plans_exist(root: Path, models: list[str], key: GWKey, horizon: int = 1) -> bool:
    """Whether every model already has a plan for ``key`` at ``horizon``.

    Args:
        root: Data directory that contains ``plans/``.
        models: YAML spec names.
        key: Deadline to look up.
        horizon: Weeks the plan covers. Weeks past 38 are not part of the file.

    Returns:
        True when each model's parquet is already on disk.
    """
    weeks = covered_horizon(key.gw, horizon)
    if weeks < 1:
        return False
    specs = load_model_specs()
    return all(
        plan_parquet(root, specs[name].model_id, key.season, key.gw, weeks).exists()
        for name in models
    )


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
