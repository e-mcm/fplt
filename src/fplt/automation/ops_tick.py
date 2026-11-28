"""Cheap idempotent ops tick: refresh feeds, catch up, checkpoints, alarm.

GitHub Actions runs this on an offset cron ``7,22,37,52`` (see ``ops-tick.yml``).
Extra ticks are safe: ``catch_up``, deadline checkpoints, and the no-plan
alarm no-op when nothing is due. Missed checkpoint minutes are caught up on
the next tick (ops-02).

Official T−90m runs ``ProductionOps.official_slate`` via
``fplt.automation.official_slate.run_official_checkpoint`` (curated build,
predict + points PMF, plan per line, finalize pointers, push).

TODO(ops-04): pass the checkout ``prod`` tag into ``RunWriter(prod_tag=…)`` for
official runs so manifests record the frozen release.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from fplt.auth import AuthCheckResult, check_token, with_manual_apply_banner
from fplt.automation.checkpoints import (
    Checkpoint,
    NoPlanAlarm,
    due_checkpoints,
    due_no_plan_alarms,
    next_upcoming_deadline,
    push_alarm_markers,
    push_checkpoint_markers,
    resolve_done_sets,
    write_alarm_marker,
    write_notified_marker,
    write_plan_checkpoint_marker,
)
from fplt.automation.cycle import (
    catch_up,
    pending_checked,
    plan_deadline,
    plans_exist,
    pre_deadline,
    reports,
)
from fplt.automation.history_guard import enforce_training_history
from fplt.automation.notify import notify_recommendation
from fplt.automation.official_slate import (
    OfficialCheckpointResult,
    run_official_checkpoint,
)
from fplt.automation.ops_feeds import refresh_ops_feeds
from fplt.automation.ops_timing import finalize_ops_timing, record_job_setup
from fplt.automation.production import ProductionOps, load_production_ops
from fplt.automation.runbook import OPS_01_TOKEN_EXPIRY, OPS_02_NO_PLAN
from fplt.automation.status import publish_ops_status, record_last_error
from fplt.data.snapshots import RawSource
from fplt.data.store import DataStore
from fplt.settings import Settings, get_settings
from fplt.store.official import official_checkpoint_dir
from fplt.types import GWKey
from fplt.utils.frames import require_not_none
from fplt.utils.logging import (
    capture_ops_log,
    log_context,
    write_ops_log_once,
)
from fplt.utils.timing import timing_session

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
        auth_level: ops-01 token check level (``ok`` / ``soft_skip`` / …).
        auth_manual_apply: True when plan notifications carry the manual-apply
            banner because the token is already dead.
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
    auth_level: str = "ok"
    auth_manual_apply: bool = False


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

    with timing_session() as timings:
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
            checkpoint_ids: list[str] = []
            alarm_ids: list[str] = []
            upcoming_key: GWKey | None = None
            notified_plan = False
            upcoming_deadline: datetime | None = None
            official_cp_dir: Path | None = None

            upcoming = next_upcoming_deadline(store, moment)
            if upcoming is not None:
                upcoming_key, upcoming_deadline = upcoming

            # ops-01: watch token expiry before planning; never abort the tick.
            auth = check_token(
                now=moment,
                next_deadline=upcoming_deadline,
                deadline_aware=True,
                allow_missing=True,
            )
            _notify_auth_check(cfg, auth, notify=notify)

            # #285: abort before catch-up / official pointers when history is thin.
            enforce_training_history(
                store,
                now=moment,
                min_prior_seasons=ops.min_prior_seasons,
                settings=cfg,
                notify=notify,
                source="ops-tick",
            )

            logger.info("ops_tick: catch_up models=%s horizon=%s", chosen, steps)
            caught = catch_up(root, chosen, horizon=steps)

            if upcoming is not None:
                upcoming_key = require_not_none(upcoming_key, what="upcoming_key")
                upcoming_deadline = require_not_none(
                    upcoming_deadline, what="upcoming_deadline"
                )
                deadline = upcoming_deadline
                primary_line = ops.primary_line().line
                slate_lines = tuple(line.line for line in ops.lines)
                # Done = write-once official markers in R2 (lsf); no mutable state.
                done, alarms_done = resolve_done_sets(
                    root,
                    upcoming_key,
                    primary_line=primary_line,
                    slate_lines=slate_lines,
                    settings=cfg,
                )
                due = due_checkpoints(moment, deadline, done)
                logger.info(
                    "ops_tick: checkpoints key=%s due=%s",
                    upcoming_key,
                    [cp.id for cp in due],
                )
                for checkpoint in due:
                    _, sent, official = _run_checkpoint(
                        root,
                        upcoming_key,
                        checkpoint,
                        moment,
                        chosen,
                        steps,
                        cfg,
                        ops=ops,
                        notify=notify,
                        auth=auth,
                    )
                    _record_checkpoint_done(
                        root,
                        upcoming_key,
                        checkpoint,
                        primary_line=primary_line,
                        notified=sent,
                        notify_enabled=notify,
                        settings=cfg,
                    )
                    checkpoint_ids.append(checkpoint.id)
                    if sent:
                        notified_plan = True
                    if official is not None:
                        official_cp_dir = official.checkpoint_dir
                    # Apply is ops-05; never call apply here even when checkpoint.apply.
                    if checkpoint.apply:
                        logger.info(
                            "ops_tick: skipping apply at %s (ops-05 not landed)",
                            checkpoint.id,
                        )

                has_plan = plans_exist(root, chosen, upcoming_key, steps)
                # Refresh alarm done set after checkpoints may have written markers.
                _, alarms_done = resolve_done_sets(
                    root,
                    upcoming_key,
                    primary_line=primary_line,
                    slate_lines=slate_lines,
                    settings=cfg,
                )
                alarms = due_no_plan_alarms(
                    moment,
                    deadline,
                    has_plan=has_plan,
                    alarms_done=alarms_done,
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
                    write_alarm_marker(root, upcoming_key, alarm.id)
                    push_alarm_markers(root, upcoming_key, settings=cfg)
                    alarm_ids.append(alarm.id)

            logger.info("ops_tick: pre_deadline now=%s", moment.isoformat())
            due_gw = pre_deadline(root, moment, chosen, horizon=steps)

            if notify and (caught or due_gw is not None) and not notified_plan:
                text = with_manual_apply_banner(
                    reports(root) or "No plan yet.",
                    auth,
                )
                notify_recommendation(cfg, "fplt recommendation", text)

            # ops-09: alert-only live drift vs backtest (never auto-switch models).
            if caught:
                try:
                    from fplt.automation.drift import run_live_drift_monitor

                    drift_reports = run_live_drift_monitor(
                        root,
                        model_names=chosen,
                        settings=cfg,
                        notify=notify,
                    )
                    for drift in drift_reports:
                        if drift.alerted:
                            logger.warning(
                                "ops_tick drift alert model_id=%s metrics=%s",
                                drift.model_id,
                                [alert.metric for alert in drift.alerts],
                            )
                except (OSError, ValueError, KeyError, TypeError) as exc:
                    logger.exception(
                        "ops_tick: drift monitor failed (non-fatal): %s", exc
                    )

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
                auth_level=auth.level,
                auth_manual_apply=auth.manual_apply,
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
            publish_ops_status(
                root,
                now=moment,
                models=list(chosen),
                horizon=steps,
                settings=cfg,
                clear_error=True,
            )
            finalize_ops_timing(
                "ops_tick",
                timings,
                root,
                meta={"models": list(chosen), "horizon": steps},
                checkpoint_dir=official_cp_dir,
                succeeded=True,
            )
            return result
        except Exception as exc:
            try:
                finalize_ops_timing(
                    "ops_tick",
                    timings,
                    root,
                    meta={"models": list(chosen), "horizon": steps},
                    succeeded=False,
                )
            except Exception:
                logger.exception("failed to write partial ops-tick timing")
            try:
                record_last_error(root, exc, source="ops-tick", now=moment)
                publish_ops_status(
                    root,
                    now=moment,
                    models=list(chosen),
                    horizon=steps,
                    settings=cfg,
                    clear_error=False,
                )
            except Exception:
                logger.exception("failed to publish status after ops-tick error")
            raise


def _notify_auth_check(
    settings: Settings,
    auth: AuthCheckResult,
    *,
    notify: bool,
) -> None:
    """Push an ops-01 alert when the token is warning/urgent; soft-skip is quiet."""
    logger.info(
        "ops_tick: auth check level=%s reason=%s manual_apply=%s",
        auth.level,
        auth.reason,
        auth.manual_apply,
    )
    if auth.level == "soft_skip":
        logger.info("ops_tick: %s", auth.message)
        return
    if not notify or auth.level not in ("warning", "urgent"):
        return
    priority = "5" if auth.level == "urgent" else None
    notify_recommendation(
        settings,
        f"fplt auth {auth.level}",
        auth.message,
        priority=priority,
        runbook_anchor=OPS_01_TOKEN_EXPIRY,
    )


def _record_checkpoint_done(
    root: Path,
    key: GWKey,
    checkpoint: Checkpoint,
    *,
    primary_line: str,
    notified: bool,
    notify_enabled: bool,
    settings: Settings,
) -> None:
    """Write write-once official markers for a completed checkpoint and push.

    T−90m pointers are written by ``run_official_checkpoint``. Plan checkpoints
    write ``line=<primary>.json``; T−6h also writes ``notified.json`` when the
    ntfy send succeeded, or when notify is disabled for the tick. A failed
    send leaves ``notified.json`` absent so the next tick retries.
    """
    if checkpoint.id == "T-90m":
        # Official slate finalize + push already recorded every line pointer.
        return
    write_plan_checkpoint_marker(
        root,
        key,
        checkpoint.id,
        primary_line=primary_line,
    )
    if checkpoint.id == "T-6h":
        if notify_enabled and not notified:
            logger.info(
                "ops_tick: T-6h plan marker written without notified.json "
                "(notify failed); will retry notify next tick"
            )
            push_checkpoint_markers(root, key, checkpoint.id, settings=settings)
            return
        write_notified_marker(root, key)
    push_checkpoint_markers(root, key, checkpoint.id, settings=settings)


def _run_checkpoint(
    root: Path,
    key: GWKey,
    checkpoint: Checkpoint,
    now: datetime,
    models: list[str],
    horizon: int,
    settings: Settings,
    *,
    ops: ProductionOps,
    notify: bool,
    auth: AuthCheckResult | None = None,
) -> tuple[bool, bool, OfficialCheckpointResult | None]:
    """Execute one due checkpoint.

    Returns:
        ``(planned, notified, official)`` where ``official`` is set only for
        the T−90m official slate.
    """
    with (
        log_context(
            checkpoint=checkpoint.id,
            season=key.season,
            gw=key.gw,
        ),
        capture_ops_log() as ops_log,
    ):
        planned = False
        official: OfficialCheckpointResult | None = None
        if checkpoint.id == "T-90m":
            logger.info("ops_tick: official slate at %s for %s", checkpoint.id, key)
            official = run_official_checkpoint(root, ops, key, settings=settings)
            planned = True
        elif checkpoint.plan or checkpoint.refresh:
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
            if auth is not None:
                text = with_manual_apply_banner(text, auth)
            notified = bool(
                notify_recommendation(
                    settings,
                    f"fplt plan {checkpoint.id} {key.season} gw{key.gw}",
                    text,
                )
            )
        logger.info(
            "ops_tick: checkpoint %s key=%s planned=%s notified=%s",
            checkpoint.id,
            key,
            planned,
            notified,
        )
        checkpoint_dir = (
            official.checkpoint_dir
            if official is not None
            else official_checkpoint_dir(
                root,
                season=key.season,
                gw=key.gw,
                checkpoint=checkpoint.id,
            )
        )
        try:
            write_ops_log_once(checkpoint_dir, ops_log.lines)
        except FileExistsError:
            logger.info(
                "ops_tick: ops_log already present at %s (write-once)",
                checkpoint_dir,
            )
    return planned, notified, official


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
        f"  auth check:        {result.auth_level}"
        + (" (manual apply)" if result.auth_manual_apply else ""),
        f"  catch_up planned:  {caught}",
        f"  upcoming deadline: {upcoming}",
        f"  checkpoints run:   {checkpoints}",
        f"  no-plan alarms:    {alarms}",
        f"  pre_deadline:      {due}",
        f"  still pending:     {pending}",
    ]
    return "\n".join(lines) + "\n"
