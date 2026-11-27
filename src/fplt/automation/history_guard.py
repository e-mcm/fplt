"""Fail loud when staged training history is too thin for a live refit.

Ops must not write official pointers after a refit on almost-empty history
(vaastav missing from the runner). See issue #285.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime

import polars as pl

from fplt.automation.checkpoints import next_upcoming_deadline
from fplt.automation.notify import notify_recommendation
from fplt.automation.runbook import OPS_12_HISTORY_GUARD
from fplt.data.pit import PITView
from fplt.data.store import DataStore
from fplt.settings import Settings
from fplt.types import GWKey

logger = logging.getLogger(__name__)


class HistoryGuardError(RuntimeError):
    """Staged history is thinner than configured or missing finished GW results."""


@dataclass(frozen=True)
class HistoryGuardResult:
    """Outcome of ``check_training_history``.

    Attributes:
        ok: True when prior-season and finished-GW checks both pass.
        prior_seasons: Completed seasons before ``season`` visible at cutoff.
        missing_finished_gws: Current-season finished GWs lacking stats rows.
        reason: Failure summary when ``ok`` is False; otherwise ``None``.
    """

    ok: bool
    prior_seasons: tuple[str, ...]
    missing_finished_gws: tuple[int, ...]
    reason: str | None = None


def resolve_guard_target(
    store: DataStore,
    now: datetime,
) -> tuple[str, GWKey, datetime]:
    """Pick season / as_of / cutoff for the training-history guard.

    Prefers the next upcoming unchecked deadline; otherwise latest staged GW
    with ``cutoff=now``.

    Raises:
        HistoryGuardError: No staged gameweeks at all.
    """
    upcoming = next_upcoming_deadline(store, now)
    if upcoming is not None:
        key, deadline = upcoming
        return key.season, key, deadline

    frame = store.read("gameweeks")
    if frame.is_empty():
        raise HistoryGuardError("no staged gameweeks; training history empty")
    latest = frame.sort(["season", "gw"]).tail(1)
    season = str(latest["season"][0])
    gw = int(latest["gw"][0])
    return season, GWKey(season, gw), now


def check_training_history(
    store: DataStore,
    *,
    season: str,
    as_of: GWKey,
    cutoff: datetime,
    min_prior_seasons: int,
) -> HistoryGuardResult:
    """Check staged history visible at ``cutoff`` for a live refit.

    Returns:
        ``HistoryGuardResult`` describing pass/fail. Does not raise.
    """
    view = PITView(as_of=as_of, cutoff=cutoff, _source=store)
    gws = view.latest("gameweeks", by=["season", "gw"]).collect()
    stats = view.table("player_fixture_stats").collect()

    prior = (
        gws.filter((pl.col("season") < season) & pl.col("data_checked"))
        .select("season")
        .unique()
        .sort("season")
    )
    prior_list = tuple(str(value) for value in prior["season"].to_list())

    finished = gws.filter(
        (pl.col("season") == season)
        & (pl.col("finished") | pl.col("data_checked"))
        & (pl.col("deadline_time") < cutoff)
    ).sort("gw")
    missing = [
        int(gw)
        for gw in finished["gw"].to_list()
        if stats.filter((pl.col("season") == season) & (pl.col("gw") == gw)).is_empty()
    ]

    reasons: list[str] = []
    if len(prior_list) < min_prior_seasons:
        shown = f" ({', '.join(prior_list)})" if prior_list else ""
        reasons.append(
            f"need {min_prior_seasons} prior seasons, found {len(prior_list)}{shown}"
        )
    if missing:
        reasons.append(f"missing player_fixture_stats for finished GWs: {missing}")
    return HistoryGuardResult(
        ok=not reasons,
        prior_seasons=prior_list,
        missing_finished_gws=tuple(missing),
        reason="; ".join(reasons) if reasons else None,
    )


def require_training_history(
    store: DataStore,
    *,
    season: str,
    as_of: GWKey,
    cutoff: datetime,
    min_prior_seasons: int,
) -> HistoryGuardResult:
    """Raise ``HistoryGuardError`` when ``check_training_history`` fails."""
    result = check_training_history(
        store,
        season=season,
        as_of=as_of,
        cutoff=cutoff,
        min_prior_seasons=min_prior_seasons,
    )
    if not result.ok:
        raise HistoryGuardError(result.reason or "training history guard failed")
    return result


def enforce_training_history(
    store: DataStore,
    *,
    now: datetime,
    min_prior_seasons: int,
    settings: Settings,
    notify: bool,
    source: str,
) -> HistoryGuardResult:
    """Resolve the guard target, require history, and alert on failure.

    Args:
        store: Staged tables.
        now: Clock for target resolution.
        min_prior_seasons: From ``ProductionOps``.
        settings: Notify channels.
        notify: When True, push an ops-12 alert before re-raising.
        source: Log label (``ops-tick`` / ``ops-catchup``).

    Returns:
        The passing ``HistoryGuardResult``.

    Raises:
        HistoryGuardError: Thin history or missing finished GW results.
    """
    try:
        season, as_of, cutoff = resolve_guard_target(store, now)
        return require_training_history(
            store,
            season=season,
            as_of=as_of,
            cutoff=cutoff,
            min_prior_seasons=min_prior_seasons,
        )
    except HistoryGuardError as exc:
        logger.error("%s: history guard failed: %s", source, exc)
        if notify:
            notify_recommendation(
                settings,
                "fplt history guard",
                str(exc),
                priority="5",
                runbook_anchor=OPS_12_HISTORY_GUARD,
            )
        raise
