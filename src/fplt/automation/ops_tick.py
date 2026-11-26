"""Cheap idempotent ops tick: refresh feeds, catch up, checkpoints, alarm.

GitHub Actions runs this on an offset cron ``7,22,37,52`` (see ``ops-tick.yml``).
Extra ticks are safe: ``catch_up``, deadline checkpoints, and the no-plan
alarm no-op when nothing is due. Missed checkpoint minutes are caught up on
the next tick (ops-02).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from fplt.automation.checkpoints import (
    Checkpoint,
    NoPlanAlarm,
    due_checkpoints,
    due_no_plan_alarms,
    mark_alarm_done,
    mark_checkpoint_done,
    next_upcoming_deadline,
    resolve_checkpoint_state,
    save_checkpoint_state,
)
from fplt.automation.cycle import (
    catch_up,
    pending_checked,
    plan_deadline,
    plans_exist,
    pre_deadline,
    reports,
)
from fplt.automation.notify import notify_recommendation
from fplt.automation.ops_feeds import refresh_ops_feeds
from fplt.automation.production import load_production_ops
from fplt.automation.runbook import OPS_02_NO_PLAN
from fplt.data.snapshots import RawSource
from fplt.data.store import DataStore
from fplt.settings import Settings, get_settings
from fplt.types import GWKey

logger = logging.getLogger(__name__)


def _empty_ids() -> list[str]:
    """Empty mutable list for dataclass defaults."""
    return []


@dataclass(frozen=True)
class OpsTickResult:
    """Counts and gameweeks from one ``run_ops_tick`` call.

    Attributes:
        snapshots_fetched: New raw files copied by ``fetch_raw``.
        staged_snapshots: Bootstrap captures ingested by ``ingest_raw``.
        understat_matches: Understat match files staged.
        api_football_captures: API-Football injury captures staged.
        catch_up_planned: Checked gameweeks planned by ``catch_up`` this run.
        pre_deadline_planned: Deadline planned by the 3h ``pre_deadline`` path.
        checkpoints_run: Checkpoint ids completed this tick.
        no_plan_alarms: No-plan alarm ids fired this tick (``retry`` / ``urgent``).
        still_pending: Checked gameweeks still missing a plan after the run.
        models: Spec names used for planning.
        horizon: Plan horizon used.
        upcoming: Upcoming deadline key when checkpoints were evaluated.
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
    checkpoints_run: list[str] = field(default_factory=_empty_ids)
    no_plan_alarms: list[str] = field(default_factory=_empty_ids)
    upcoming: GWKey | None = None


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
    """Refresh feeds, catch up, run due checkpoints, and raise no-plan alarms.

    Args:
        data_dir: Root with ``raw/``, ``staged/``, and ``plans/``. Defaults to
            ``Settings.data_dir``.
        now: Clock for checkpoints and the pre-deadline window. Defaults to
            ``datetime.now(UTC)``.
        models: YAML spec names. Defaults to ``load_production_ops().models``.
        horizon: Weeks each plan looks ahead. Defaults to
            ``load_production_ops().plan_horizon``.
        source: Raw fetch source: ``auto``, ``bucket``, or ``branch``.
        remote: Git remote (or local path in tests) for the snapshots branch.
        branch: Branch that holds write-once captures (default ``snapshots``).
        settings: Optional settings for notify and bucket credentials.
        notify: When True, push recommendation / alarm messages via ntfy/SMTP.

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

    store = DataStore(root)
    checkpoint_ids: list[str] = []
    alarm_ids: list[str] = []
    upcoming_key: GWKey | None = None
    notified_plan = False

    upcoming = next_upcoming_deadline(store, moment)
    if upcoming is not None:
        upcoming_key, deadline = upcoming
        state = resolve_checkpoint_state(root, upcoming_key, deadline)
        due = due_checkpoints(moment, deadline, state.done)
        logger.info(
            "ops_tick: checkpoints key=%s due=%s",
            upcoming_key,
            [cp.id for cp in due],
        )
        for checkpoint in due:
            _, sent = _run_checkpoint(
                root,
                upcoming_key,
                checkpoint,
                moment,
                chosen,
                steps,
                cfg,
                notify=notify,
            )
            state = mark_checkpoint_done(state, checkpoint.id)
            checkpoint_ids.append(checkpoint.id)
            if sent:
                notified_plan = True
            # Apply is ops-05; never call apply here even when checkpoint.apply.
            if checkpoint.apply:
                logger.info(
                    "ops_tick: skipping apply at %s (ops-05 not landed)",
                    checkpoint.id,
                )

        has_plan = plans_exist(root, chosen, upcoming_key, steps)
        alarms = due_no_plan_alarms(
            moment,
            deadline,
            has_plan=has_plan,
            alarms_done=state.alarms_done,
        )
        for alarm in alarms:
            # A successful T−3h retry can create the plan; skip later alarms.
            if plans_exist(root, chosen, upcoming_key, steps):
                break
            _run_no_plan_alarm(
                root,
                upcoming_key,
                alarm,
                moment,
                chosen,
                steps,
                cfg,
                notify=notify,
            )
            state = mark_alarm_done(state, alarm.id)
            alarm_ids.append(alarm.id)

        save_checkpoint_state(root, state)

    logger.info("ops_tick: pre_deadline now=%s", moment.isoformat())
    due_gw = pre_deadline(root, moment, chosen, horizon=steps)

    if notify and (caught or due_gw is not None) and not notified_plan:
        text = reports(root) or "No plan yet."
        notify_recommendation(cfg, "fplt recommendation", text)

    waiting = pending_checked(store, root, chosen, steps)
    result = OpsTickResult(
        snapshots_fetched=feeds.snapshots_fetched,
        staged_snapshots=feeds.staged_snapshots,
        understat_matches=feeds.understat_matches,
        api_football_captures=feeds.api_football_captures,
        catch_up_planned=caught,
        pre_deadline_planned=due_gw,
        still_pending=waiting,
        models=list(chosen),
        horizon=steps,
        checkpoints_run=checkpoint_ids,
        no_plan_alarms=alarm_ids,
        upcoming=upcoming_key,
    )
    due_key = result.pre_deadline_planned
    logger.info(
        "ops_tick done fetched=%s staged=%s catch_up=%s checkpoints=%s "
        "alarms=%s pre_deadline=%s pending=%s",
        result.snapshots_fetched,
        result.staged_snapshots,
        [str(key) for key in result.catch_up_planned],
        result.checkpoints_run,
        result.no_plan_alarms,
        None if due_key is None else str(due_key),
        [str(key) for key in result.still_pending],
    )
    return result


def _run_checkpoint(
    root: Path,
    key: GWKey,
    checkpoint: Checkpoint,
    now: datetime,
    models: list[str],
    horizon: int,
    settings: Settings,
    *,
    notify: bool,
) -> tuple[bool, bool]:
    """Execute one due checkpoint. Returns ``(planned, notified)``."""
    planned = False
    if checkpoint.plan or checkpoint.refresh:
        planned = plan_deadline(
            root,
            key,
            now,
            models,
            horizon,
            refresh_if_stale=checkpoint.refresh,
        )
    notified = False
    if notify and checkpoint.notify:
        text = reports(root) or f"No plan yet for {key.season} gw{key.gw}."
        notify_recommendation(
            settings,
            f"fplt plan {checkpoint.id} {key.season} gw{key.gw}",
            text,
        )
        notified = True
    logger.info(
        "ops_tick: checkpoint %s key=%s planned=%s notified=%s",
        checkpoint.id,
        key,
        planned,
        notified,
    )
    return planned, notified


def _run_no_plan_alarm(
    root: Path,
    key: GWKey,
    alarm: NoPlanAlarm,
    now: datetime,
    models: list[str],
    horizon: int,
    settings: Settings,
    *,
    notify: bool,
) -> None:
    """Retry planning at T−3h; urgent-notify at T−2h if still missing."""
    if alarm.id == "retry":
        logger.warning(
            "ops_tick: no-plan retry at T−3h for %s; forcing plan_deadline",
            key,
        )
        plan_deadline(
            root,
            key,
            now,
            models,
            horizon,
            refresh_if_stale=True,
        )
        return
    # urgent
    message = (
        f"URGENT: no production plan for {key.season} gw{key.gw} at T−2h. "
        "Run `gh workflow run ops-tick.yml -f force_live=true` or "
        "`fplt ops-catchup` / `make ops-catchup`."
    )
    logger.error("ops_tick: %s", message)
    if notify:
        notify_recommendation(
            settings,
            f"fplt NO PLAN {key.season} gw{key.gw}",
            message,
            priority="5",
            runbook_anchor=OPS_02_NO_PLAN,
        )


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
    checkpoints = ", ".join(result.checkpoints_run) or "none"
    alarms = ", ".join(result.no_plan_alarms) or "none"
    upcoming = (
        "none"
        if result.upcoming is None
        else f"{result.upcoming.season} gw{result.upcoming.gw}"
    )
    lines = [
        "ops-tick status",
        f"  snapshots fetched: {result.snapshots_fetched}",
        f"  staged snapshots:  {result.staged_snapshots}",
        f"  understat matches: {result.understat_matches}",
        f"  api_football:      {result.api_football_captures}",
        f"  models / horizon:  {models} / H={result.horizon}",
        f"  catch_up planned:  {caught}",
        f"  upcoming deadline: {upcoming}",
        f"  checkpoints run:   {checkpoints}",
        f"  no-plan alarms:    {alarms}",
        f"  pre_deadline:      {due}",
        f"  still pending:     {pending}",
    ]
    return "\n".join(lines) + "\n"
