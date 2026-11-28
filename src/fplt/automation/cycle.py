"""Catch up checked gameweeks (predict-only), and plan the upcoming deadline.

A checked gameweek counts as done for catch-up when every requested model has
a **predict** run for that deadline and horizon (#381). Catch-up walks the
missing checked gameweeks in order and writes ledger / outcome coverage for
drift — it does **not** plan against the live squad (that would reuse the
current fifteen, bank, FTs, and chips for a past GW). Plans are written only
for the upcoming deadline via ``plan_deadline`` / ``pre_deadline`` /
checkpoints / ``run_now``.

The pre-deadline refresh plans a single upcoming deadline and does not walk
earlier weeks. When availability signals (``player_status``,
``predicted_lineups``, ``player_injuries``) move after a plan was written,
``pre_deadline`` retires the stale plan and re-predicts / re-optimises
(FDB-253 late-refresh hook).

When ``FPLT_ENTRY_ID`` is set (and ``FPLT_FPL_API_TOKEN`` for the live squad),
upcoming-deadline plans start from ``resolve_squad_state`` /
``load_current_team`` rather than an empty YAML draft. Without an entry id the
YAML behaviour is unchanged. ``resolve_squad_state(..., as_of_gw=…)`` refuses
to resolve a finished GW from the live squad (#381).

Plans use ``run_plan``'s live default chip policy
(``DEFAULT_LIVE_CHIP_ARM`` = optimised: season chip scheduler → MILP).
Horizon stays ``DEFAULT_PLAN_HORIZON`` (3).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from fplt.automation.late_refresh import plans_stale_vs_signals, retire_plans
from fplt.automation.plan_index import (
    catalog_has_plans,
    catalog_has_predicts,
    resolve_plan_catalog,
)
from fplt.automation.production import load_production_ops
from fplt.backtest.run import run_backtest
from fplt.data.benchmark_projections import archive_if_missing
from fplt.data.pit import PITData
from fplt.data.remote import Remote
from fplt.data.store import DataStore
from fplt.optimise.decisions import resolve_squad_state, run_plan
from fplt.optimise.plan import covered_horizon
from fplt.settings import Settings
from fplt.store.catalog import scan_succeeded_runs
from fplt.store.plan_runs import plan_reports
from fplt.types import GWKey

# Live list / horizon come from ``configs/ops/production.yaml`` (or env).
# Prefer ``load_production_ops()`` so overrides apply. ``PRODUCTION_MODELS`` is
# resolved lazily via ``__getattr__`` so a bad/empty override cannot crash
# every ``fplt.automation`` import (#362).
_WINDOW = timedelta(hours=3)
logger = logging.getLogger(__name__)


def __getattr__(name: str) -> object:
    """Lazy ``PRODUCTION_MODELS`` mirror of ``load_production_ops().models``."""
    if name == "PRODUCTION_MODELS":
        return list(load_production_ops().models)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class CatchUpSeasonError(ValueError):
    """A live automation play was asked for a season other than the current one.

    The live squad (``FPLT_ENTRY_ID``) only exists for the current FPL season,
    so planning a past season with it would price current-season element ids
    against a past-season view (#369).
    """


def current_season(store: DataStore) -> str | None:
    """Return the latest season in staged ``gameweeks``, or ``None`` when empty.

    Season keys are ``YYYY-YY`` strings, so the lexicographic maximum is the
    most recent season. Staged history (e.g. vaastav seasons) sits alongside
    the live season in the same table; this picks the live one.
    """
    frame = store.read("gameweeks")
    if frame.is_empty():
        return None
    return str(frame.get_column("season").max())


def pending_checked(
    store: DataStore,
    root: Path,
    models: list[str] | None = None,
    horizon: int | None = None,
    *,
    settings: Settings | None = None,
    remotes: Sequence[Remote] | None = None,
    remote_catalog: pl.DataFrame | None = None,
) -> list[GWKey]:
    """Current-season checked gameweeks that are still missing a predict run.

    Only the current season (``current_season``) from
    ``ProductionOps.catch_up_from_gw`` onwards is considered: catch-up fills
    weeks live ops missed, it does not backtest staged history (#369). Done
    means every model has a ``run_type=predict`` row for the deadline /
    horizon — not a plan (#381).

    Local ``runs/`` is the fast path. When a data bucket is configured,
    gameweeks that look pending locally are checked against the durable R2
    catalog so a cold cache does not re-play finished weeks (#371).

    Args:
        store: Staged tables, including ``gameweeks``.
        root: Data directory that contains ``runs/``.
        models: YAML spec names that must each have a predict. Defaults to
            ``load_production_ops().models``.
        horizon: Weeks the missing predict would cover. Defaults to
            ``load_production_ops().plan_horizon``.
        settings: Process settings for remote consultation; defaults to
            ``get_settings()``.
        remotes: Optional pre-built remotes (tests).
        remote_catalog: Optional preloaded ``CatalogRunSchema`` frame so a
            batch of lookups shares one pull (tests / callers).

    Returns:
        Deadlines whose ``data_checked`` flag is set and whose predict is
        absent both locally and in the remote index, earliest first.
    """
    ops = load_production_ops()
    chosen = ops.models if models is None else models
    steps = ops.plan_horizon if horizon is None else horizon
    season = current_season(store)
    if season is None:
        return []
    frame = store.read("gameweeks").filter(
        pl.col("data_checked")
        & (pl.col("season") == season)
        & (pl.col("gw") >= ops.catch_up_from_gw)
    )
    local_pending: list[GWKey] = []
    seen: set[GWKey] = set()
    for season_key, gameweek in frame.select("season", "gw").iter_rows():
        key = GWKey(str(season_key), int(gameweek))
        if key in seen:
            continue
        seen.add(key)
        if _local_predicts_exist(root, chosen, key, steps):
            continue
        local_pending.append(key)
    if not local_pending:
        return []
    catalog = resolve_plan_catalog(
        settings=settings, remotes=remotes, remote_catalog=remote_catalog
    )
    waiting = [
        key
        for key in local_pending
        if not catalog_has_predicts(catalog, chosen, key, steps)
    ]
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
    """Fit and score missed current-season gameweeks (predict-only), oldest first.

    Does **not** plan: past-GW plans against the live squad are meaningless
    (#381). Plans for the upcoming deadline come from checkpoints /
    ``pre_deadline`` / ``run_now``.

    At most ``ProductionOps.max_catch_up_per_tick`` gameweeks are played per
    call; the rest stay pending for the next tick, so one tick cannot burn
    the job timeout (#369).

    Args:
        data_dir: Root that contains ``staged/`` and receives ``ledger/`` /
            ``runs/``.
        models: YAML spec names. Defaults to ``load_production_ops().models``.
        horizon: Weeks each predict looks ahead. Defaults to
            ``load_production_ops().plan_horizon``.

    Returns:
        The gameweeks predicted by this call, in play order.
    """
    ops = load_production_ops()
    chosen = list(ops.models) if models is None else models
    steps = ops.plan_horizon if horizon is None else horizon
    waiting = pending_checked(DataStore(data_dir), data_dir, chosen, steps)
    batch = waiting[: ops.max_catch_up_per_tick]
    logger.info(
        "catch_up models=%s pending=%s playing=%s deferred=%s horizon=%s "
        "(predict-only)",
        chosen,
        len(waiting),
        [str(key) for key in batch],
        len(waiting) - len(batch),
        steps,
    )
    done: list[GWKey] = []
    for key in batch:
        _predict_checked(data_dir, chosen, key, steps)
        done.append(key)
    return done


def plan_deadline(
    data_dir: Path,
    key: GWKey,
    now: datetime,
    models: list[str] | None = None,
    horizon: int | None = None,
    *,
    refresh_if_stale: bool = True,
) -> bool:
    """Plan a specific deadline when missing, or refresh when stale.

    Used by deadline checkpoints (ops-02) outside the three-hour
    ``pre_deadline`` window. Archives benchmark projections first (N1).

    Args:
        data_dir: Root that contains ``staged/`` and receives the plan.
        key: Deadline to plan.
        now: Clock reading, timezone-aware.
        models: YAML spec names. Defaults to ``load_production_ops().models``.
        horizon: Weeks the plan looks ahead. Defaults to
            ``load_production_ops().plan_horizon``.
        refresh_if_stale: When true (default), retire write-once plans whose
            mtime is older than the latest ``player_status`` /
            ``predicted_lineups`` / ``player_injuries`` signal and rebuild.

    Returns:
        True when a plan was written (or rebuilt) this call.
    """
    ops = load_production_ops()
    chosen = list(ops.models) if models is None else models
    steps = ops.plan_horizon if horizon is None else horizon
    store = DataStore(data_dir)
    # North-star N1: archive ep_next + OpenFPL before the deadline (write-once).
    archive_if_missing(data_dir, key, now=now)
    if plans_exist(data_dir, chosen, key, steps):
        if refresh_if_stale and plans_stale_vs_signals(
            data_dir, store, chosen, key, steps, now=now
        ):
            retired = retire_plans(data_dir, chosen, key, steps, retired_at=now)
            logger.info(
                "plan_deadline: retired %s stale plan(s) for %s; re-predicting",
                len(retired),
                key,
            )
        else:
            logger.info("plan_deadline: plan already exists for %s", key)
            return False
    logger.info("plan_deadline: planning %s models=%s", key, chosen)
    _play(data_dir, chosen, key, steps)
    return True


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
    wrote = plan_deadline(
        data_dir,
        key,
        now,
        chosen,
        steps,
        refresh_if_stale=refresh_if_stale,
    )
    return key if wrote else None


def reports(root: Path) -> str:
    """Concatenate every saved plan report (``report.md`` in plan runs).

    Args:
        root: Data directory that contains ``runs/``.

    Returns:
        The markdown, or an empty string when no report has been written.
    """
    return plan_reports(root)


def _predict_checked(
    data_dir: Path, models: list[str], key: GWKey, horizon: int
) -> None:
    """Walk-forward predict one checked GW for catch-up (no squad / plan)."""
    logger.info("automation predict-only %s models=%s horizon=%s", key, models, horizon)
    live = current_season(DataStore(data_dir))
    if key.season != live:
        raise CatchUpSeasonError(
            f"refusing to play {key}: live automation only plans the current "
            f"season ({live}); the live squad cannot be priced against "
            f"{key.season}"
        )
    run_backtest(data_dir, key.season, key.gw, key.gw, models, steps=horizon)


def _play(data_dir: Path, models: list[str], key: GWKey, horizon: int) -> None:
    """Predict then plan one upcoming deadline from the live / YAML squad."""
    logger.info("automation play %s models=%s horizon=%s", key, models, horizon)
    live = current_season(DataStore(data_dir))
    if key.season != live:
        raise CatchUpSeasonError(
            f"refusing to play {key}: live automation only plans the current "
            f"season ({live}); the live squad cannot be priced against "
            f"{key.season}"
        )
    run_backtest(data_dir, key.season, key.gw, key.gw, models, steps=horizon)
    # Live squad when FPLT_ENTRY_ID is set; otherwise YAML draft / empty owned.
    # as_of_gw refuses a finished GW from the live squad (#381).
    view = PITData(DataStore(data_dir)).view(key)
    resolved = resolve_squad_state(view, event=key.gw, as_of_gw=key.gw)
    for name in models:
        try:
            run_plan(
                data_dir,
                name,
                key.season,
                key.gw,
                horizon,
                squad=resolved.state,
                squad_source=resolved.source,
            )
        except FileExistsError:
            logger.info("plan already exists for %s / %s; skipping", name, key)
            continue


def predicts_exist(
    root: Path,
    models: list[str] | None,
    key: GWKey,
    horizon: int | None = None,
    *,
    settings: Settings | None = None,
    remotes: Sequence[Remote] | None = None,
    remote_catalog: pl.DataFrame | None = None,
) -> bool:
    """Whether every model already has a predict run for ``key`` at ``horizon``.

    Local ``runs/`` is checked first. When a data bucket is configured and
    the local run is missing, the durable R2 catalog is consulted (#371).

    Args:
        root: Data directory that contains ``runs/``.
        models: YAML spec names. ``None`` uses ``load_production_ops().models``.
        key: Deadline to look up.
        horizon: Weeks the predict covers. Defaults to
            ``load_production_ops().plan_horizon``.
        settings: Process settings for remote consultation; defaults to
            ``get_settings()``.
        remotes: Optional pre-built remotes (tests).
        remote_catalog: Optional preloaded catalog frame (tests / batch).

    Returns:
        True when each model's predict is already on disk, or when the remote
        index records a matching succeeded predict run for every model.
    """
    ops = load_production_ops()
    chosen = ops.models if models is None else models
    steps = ops.plan_horizon if horizon is None else horizon
    if _local_predicts_exist(root, chosen, key, steps):
        return True
    catalog = resolve_plan_catalog(
        settings=settings, remotes=remotes, remote_catalog=remote_catalog
    )
    return catalog_has_predicts(catalog, chosen, key, steps)


def plans_exist(
    root: Path,
    models: list[str] | None,
    key: GWKey,
    horizon: int | None = None,
    *,
    settings: Settings | None = None,
    remotes: Sequence[Remote] | None = None,
    remote_catalog: pl.DataFrame | None = None,
) -> bool:
    """Whether every model already has a plan for ``key`` at ``horizon``.

    Local ``runs/`` is checked first. When a data bucket is configured and
    the local parquet is missing, the durable R2 plan catalog is consulted
    (#371). Used by upcoming-deadline paths (``plan_deadline`` /
    ``pre_deadline``); catch-up uses ``predicts_exist`` instead (#381).

    Args:
        root: Data directory that contains ``plans/``.
        models: YAML spec names. ``None`` uses ``load_production_ops().models``.
        key: Deadline to look up.
        horizon: Weeks the plan covers. Weeks past 38 are not part of the file.
            Defaults to ``load_production_ops().plan_horizon``.
        settings: Process settings for remote consultation; defaults to
            ``get_settings()``.
        remotes: Optional pre-built remotes (tests).
        remote_catalog: Optional preloaded catalog frame (tests / batch).

    Returns:
        True when each model's parquet is already on disk, or when the remote
        plan index records a matching succeeded plan run for every model.
    """
    ops = load_production_ops()
    chosen = ops.models if models is None else models
    steps = ops.plan_horizon if horizon is None else horizon
    if _local_plans_exist(root, chosen, key, steps):
        return True
    catalog = resolve_plan_catalog(
        settings=settings, remotes=remotes, remote_catalog=remote_catalog
    )
    return catalog_has_plans(catalog, chosen, key, steps)


def _local_predicts_exist(
    root: Path, models: list[str], key: GWKey, horizon: int
) -> bool:
    """Whether every model has a succeeded local predict run for ``key``."""
    if horizon < 1:
        return False
    return catalog_has_predicts(scan_succeeded_runs(root), models, key, horizon)


def _local_plans_exist(root: Path, models: list[str], key: GWKey, horizon: int) -> bool:
    """Whether every model already has a succeeded local plan *run* for ``key``.

    Reads ``runs/`` (data-model cut-over), never ``plans/<model_id>/as_of=…``:
    that path ignores code, purpose and squad, so a stale file could pass.
    """
    weeks = covered_horizon(key.gw, horizon)
    if weeks < 1:
        return False
    return catalog_has_plans(scan_succeeded_runs(root), models, key, weeks)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
