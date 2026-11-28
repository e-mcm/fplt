"""One-shot post-downtime recovery via the ``fplt`` CLI / GHA.

Runs ``fetch_raw`` → restage FPL (+ vaastav) → Understat / API-Football ingest
→ ``catch_up`` (predict-only for past checked GWs; #381) → status summary.
Prefer this after travel or long sleep when scheduled ops ticks have not
drained the backlog yet. Plans for the upcoming deadline come from
``ops-tick`` checkpoints / ``pre_deadline`` / ``run_now``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fplt.automation.bench import load_bench_config
from fplt.automation.cycle import (
    catch_up,
    pending_checked,
)
from fplt.automation.history_guard import enforce_training_history
from fplt.automation.ops_feeds import refresh_ops_feeds
from fplt.automation.ops_timing import finalize_ops_timing, record_job_setup
from fplt.automation.production import load_production_ops
from fplt.data.snapshots import RawSource
from fplt.data.store import DataStore
from fplt.optimise.plan import DEFAULT_PLAN_HORIZON
from fplt.settings import Settings, get_settings
from fplt.types import GWKey
from fplt.utils.timing import rss_warn_threshold, timing_session

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpsCatchupResult:
    """Counts and gameweeks from one ``run_ops_catchup`` call.

    Attributes:
        snapshots_fetched: New raw files copied by ``fetch_raw`` (bucket or
            snapshots branch).
        staged_snapshots: Bootstrap captures ingested by ``ingest_raw``.
        understat_matches: Understat match files staged.
        api_football_captures: API-Football injury captures staged.
        planned: Checked gameweeks predicted by ``catch_up`` this run
            (predict-only; #381).
        still_pending: Checked gameweeks still missing a predict after the run.
        models: Spec names used for catch-up.
        horizon: Predict / plan horizon used for catch-up.
    """

    snapshots_fetched: int
    staged_snapshots: int
    understat_matches: int
    api_football_captures: int
    planned: list[GWKey]
    still_pending: list[GWKey]
    models: list[str]
    horizon: int


def run_ops_catchup(
    data_dir: Path | None = None,
    *,
    models: list[str] | None = None,
    horizon: int = DEFAULT_PLAN_HORIZON,
    source: RawSource = "auto",
    remote: str = "origin",
    branch: str = "snapshots",
    settings: Settings | None = None,
    notify: bool = True,
) -> OpsCatchupResult:
    """Fetch raw captures, restage feeds, catch up predicts, and return status.

    Catch-up is predict-only for past checked GWs (#381); it does not write
    plans or push recommendation ntfy. Upcoming-deadline plans and notifies
    stay on ``ops-tick`` checkpoints / ``pre_deadline``.

    Args:
        data_dir: Root with ``raw/``, ``staged/``, and ``runs/``. Defaults to
            ``Settings.data_dir``.
        models: YAML spec names for catch-up. Defaults to
            ``load_production_ops().models``.
        horizon: Weeks each predict looks ahead. Defaults to
            ``DEFAULT_PLAN_HORIZON``.
        source: Raw fetch source: ``auto`` (bucket when configured, else branch),
            ``bucket``, or ``branch``.
        remote: Git remote (or local path in tests) for the snapshots branch.
        branch: Branch that holds write-once captures (default ``snapshots``).
        settings: Optional settings for bucket credentials. Defaults to
            ``get_settings()``.
        notify: Kept for CLI compatibility; catch-up no longer pushes plan
            recommendations (predict-only). Training-history alerts still
            honour this flag.

    Returns:
        Counts and pending/caught-up gameweeks for the operator status print.

    Raises:
        SnapshotFetchError: The snapshots branch could not be cloned, or
            ``source=bucket`` with no data bucket configured.
        RemoteError: An rclone pull against the bucket failed.
    """
    cfg = get_settings() if settings is None else settings
    root = cfg.data_dir if data_dir is None else data_dir
    ops = load_production_ops()
    chosen = list(ops.models) if models is None else models

    with (
        timing_session() as timings,
        rss_warn_threshold(load_bench_config().rss_warn_kb()),
    ):
        record_job_setup(timings)
        try:
            feeds = refresh_ops_feeds(
                root,
                source=source,
                remote=remote,
                branch=branch,
                settings=cfg,
            )

            store = DataStore(root)
            enforce_training_history(
                store,
                now=datetime.now(UTC),
                min_prior_seasons=ops.min_prior_seasons,
                settings=cfg,
                notify=notify,
                source="ops-catchup",
            )

            logger.info(
                "ops_catchup: catch_up (predict-only) models=%s horizon=%s",
                chosen,
                horizon,
            )
            done = catch_up(root, chosen, horizon=horizon)

            waiting = pending_checked(store, root, chosen, horizon)
            result = OpsCatchupResult(
                snapshots_fetched=feeds.snapshots_fetched,
                staged_snapshots=feeds.staged_snapshots,
                understat_matches=feeds.understat_matches,
                api_football_captures=feeds.api_football_captures,
                planned=done,
                still_pending=waiting,
                models=list(chosen),
                horizon=horizon,
            )
            logger.info(
                "ops_catchup done fetched=%s staged=%s understat=%s api_football=%s "
                "planned=%s pending=%s",
                result.snapshots_fetched,
                result.staged_snapshots,
                result.understat_matches,
                result.api_football_captures,
                [str(key) for key in result.planned],
                [str(key) for key in result.still_pending],
            )
            finalize_ops_timing(
                "ops_catchup",
                timings,
                root,
                meta={"models": list(chosen), "horizon": horizon},
                succeeded=True,
            )
            return result
        except Exception:
            try:
                finalize_ops_timing(
                    "ops_catchup",
                    timings,
                    root,
                    meta={"models": list(chosen), "horizon": horizon},
                    succeeded=False,
                )
            except Exception:
                logger.exception("failed to write partial ops-catchup timing")
            raise


def format_ops_catchup_status(result: OpsCatchupResult) -> str:
    """Render a human-readable status block for ``run_ops_catchup``.

    Args:
        result: Outcome from ``run_ops_catchup``.

    Returns:
        Multi-line text ending with a newline.
    """
    planned = ", ".join(f"{key.season} gw{key.gw}" for key in result.planned) or "none"
    pending = (
        ", ".join(f"{key.season} gw{key.gw}" for key in result.still_pending) or "none"
    )
    models = ", ".join(result.models)
    lines = [
        "ops-catchup status",
        f"  snapshots fetched: {result.snapshots_fetched}",
        f"  staged snapshots:  {result.staged_snapshots}",
        f"  understat matches: {result.understat_matches}",
        f"  api_football:      {result.api_football_captures}",
        f"  models / horizon:  {models} / H={result.horizon}",
        f"  predicted this run: {planned}",
        f"  still pending:      {pending}",
    ]
    return "\n".join(lines) + "\n"
