"""Cheap idempotent ops tick: refresh feeds, catch up, and pre-deadline plan.

GitHub Actions runs this on an offset ``*/15`` cron (see ``ops-tick.yml``).
Extra ticks are safe: ``catch_up`` and ``pre_deadline`` no-op when nothing is
due. Deadline checkpoints (ops-02) build on this later; this module only wires
the existing cycle helpers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fplt.automation.cycle import (
    catch_up,
    pending_checked,
    pre_deadline,
    reports,
)
from fplt.automation.notify import notify_recommendation
from fplt.automation.ops_feeds import refresh_ops_feeds
from fplt.automation.production import load_production_ops
from fplt.data.snapshots import RawSource
from fplt.data.store import DataStore
from fplt.settings import Settings, get_settings
from fplt.types import GWKey

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OpsTickResult:
    """Counts and gameweeks from one ``run_ops_tick`` call.

    Attributes:
        snapshots_fetched: New raw files copied by ``fetch_raw``.
        staged_snapshots: Bootstrap captures ingested by ``ingest_raw``.
        understat_matches: Understat match files staged.
        api_football_captures: API-Football injury captures staged.
        catch_up_planned: Checked gameweeks planned by ``catch_up`` this run.
        pre_deadline_planned: Deadline planned by ``pre_deadline``, if any.
        still_pending: Checked gameweeks still missing a plan after the run.
        models: Spec names used for planning.
        horizon: Plan horizon used.
    """

    snapshots_fetched: int
    staged_snapshots: int
    understat_matches: int
    api_football_captures: int
    catch_up_planned: list[GWKey]
    pre_deadline_planned: GWKey | None
    still_pending: list[GWKey]
    models: list[str]
    horizon: int


def run_ops_tick(
    data_dir: Path | None = None,
    *,
    now: datetime | None = None,
    models: list[str] | None = None,
    horizon: int | None = None,
    source: RawSource = "auto",
    remote: str = "origin",
    branch: str = "snapshots",
    settings: Settings | None = None,
    notify: bool = True,
) -> OpsTickResult:
    """Refresh feeds, catch up checked GWs, and plan a due pre-deadline window.

    Args:
        data_dir: Root with ``raw/``, ``staged/``, and ``plans/``. Defaults to
            ``Settings.data_dir``.
        now: Clock for the pre-deadline window. Defaults to ``datetime.now(UTC)``.
        models: YAML spec names. Defaults to ``load_production_ops().models``.
        horizon: Weeks each plan looks ahead. Defaults to
            ``load_production_ops().plan_horizon``.
        source: Raw fetch source: ``auto``, ``bucket``, or ``branch``.
        remote: Git remote (or local path in tests) for the snapshots branch.
        branch: Branch that holds write-once captures (default ``snapshots``).
        settings: Optional settings for notify and bucket credentials.
        notify: When True and any plan was written this tick, push the
            concatenated ``report.md`` via ntfy/SMTP.

    Returns:
        Counts and planned / pending gameweeks for the operator status print.

    Raises:
        SnapshotFetchError: Raw fetch failed (branch clone or missing bucket).
        RemoteError: An rclone pull against the bucket failed.
    """
    cfg = get_settings() if settings is None else settings
    root = cfg.data_dir if data_dir is None else data_dir
    ops = load_production_ops()
    chosen = list(ops.models) if models is None else models
    steps = ops.plan_horizon if horizon is None else horizon
    moment = datetime.now(UTC) if now is None else now

    feeds = refresh_ops_feeds(
        root,
        source=source,
        remote=remote,
        branch=branch,
        settings=cfg,
    )

    logger.info("ops_tick: catch_up models=%s horizon=%s", chosen, steps)
    caught = catch_up(root, chosen, horizon=steps)

    logger.info("ops_tick: pre_deadline now=%s", moment.isoformat())
    due = pre_deadline(root, moment, chosen, horizon=steps)

    if notify and (caught or due is not None):
        text = reports(root) or "No plan yet."
        notify_recommendation(cfg, "fplt recommendation", text)

    waiting = pending_checked(DataStore(root), root, chosen, steps)
    result = OpsTickResult(
        snapshots_fetched=feeds.snapshots_fetched,
        staged_snapshots=feeds.staged_snapshots,
        understat_matches=feeds.understat_matches,
        api_football_captures=feeds.api_football_captures,
        catch_up_planned=caught,
        pre_deadline_planned=due,
        still_pending=waiting,
        models=list(chosen),
        horizon=steps,
    )
    due_key = result.pre_deadline_planned
    logger.info(
        "ops_tick done fetched=%s staged=%s catch_up=%s pre_deadline=%s pending=%s",
        result.snapshots_fetched,
        result.staged_snapshots,
        [str(key) for key in result.catch_up_planned],
        None if due_key is None else str(due_key),
        [str(key) for key in result.still_pending],
    )
    return result


def format_ops_tick_status(result: OpsTickResult) -> str:
    """Render a human-readable status block for ``run_ops_tick``.

    Args:
        result: Outcome from ``run_ops_tick``.

    Returns:
        Multi-line text ending with a newline.
    """
    caught = (
        ", ".join(f"{key.season} gw{key.gw}" for key in result.catch_up_planned)
        or "none"
    )
    due = (
        "none"
        if result.pre_deadline_planned is None
        else (
            f"{result.pre_deadline_planned.season} gw{result.pre_deadline_planned.gw}"
        )
    )
    pending = (
        ", ".join(f"{key.season} gw{key.gw}" for key in result.still_pending) or "none"
    )
    models = ", ".join(result.models)
    lines = [
        "ops-tick status",
        f"  snapshots fetched: {result.snapshots_fetched}",
        f"  staged snapshots:  {result.staged_snapshots}",
        f"  understat matches: {result.understat_matches}",
        f"  api_football:      {result.api_football_captures}",
        f"  models / horizon:  {models} / H={result.horizon}",
        f"  catch_up planned:  {caught}",
        f"  pre_deadline:      {due}",
        f"  still pending:     {pending}",
    ]
    return "\n".join(lines) + "\n"
