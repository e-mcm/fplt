"""Deadline-anchored ops checkpoints and write-once alarms (ops-02).

GitHub Actions cron is best-effort: ticks can arrive late or be dropped.
``due_checkpoints`` lists every checkpoint whose fire time has passed and
that is not yet recorded in ``done``, so a missed T−24h still runs on the
next tick. After the deadline nothing is due.

**Done = write-once output exists in R2** (no mutable ops state). ``done`` /
``alarms_done`` are derived from an ``rclone lsf official/season=/gw=/ -R``
listing (or the local ``official/`` tree). Markers:

* T−24h / T−30m — ``checkpoint=<id>/line=<primary>.json``
* T−6h — that pointer **and** ``checkpoint=T-6h/notified.json``
* T−90m — ``checkpoint=T-90m/line=<line>.json`` for every official slate line
* alarms — ``alarms/<alarm_id>.json`` (no-plan + missing official slate)

Checkpoints (offsets before the deadline):

* ``T-24h`` — full plan
* ``T-6h`` — plan + notify
* ``T-90m`` — final refresh; apply is stubbed until ops-05
* ``T-30m`` — last-chance refresh only

Alarms (separate from the plan checkpoints):

* ``T-3h`` — no-plan retry / catch-up when any production model lacks a plan
* ``T-2h`` — no-plan urgent notification if still missing
* ``T-60m`` — urgent when the T−90m official slate pointers are still missing
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from fplt.data.store import DataStore
from fplt.store.official import official_checkpoint_dir, pointer_path
from fplt.types import GWKey

if TYPE_CHECKING:
    from collections.abc import Iterable

    from fplt.data.remote import Remote
    from fplt.settings import Settings

logger = logging.getLogger(__name__)

CheckpointId = Literal["T-24h", "T-6h", "T-90m", "T-30m"]
NoPlanAlarmId = Literal["retry", "urgent"]
MissingOfficialSlateAlarmId = Literal["missing_official_slate"]
AlarmId = NoPlanAlarmId | MissingOfficialSlateAlarmId

_NOTIFIED_NAME = "notified.json"
MISSING_OFFICIAL_SLATE_ALARM_ID: MissingOfficialSlateAlarmId = "missing_official_slate"
_KNOWN_ALARM_IDS: tuple[AlarmId, ...] = (
    "retry",
    "urgent",
    MISSING_OFFICIAL_SLATE_ALARM_ID,
)
_ALARM_KIND: dict[str, str] = {
    "retry": "no_plan_alarm",
    "urgent": "no_plan_alarm",
    MISSING_OFFICIAL_SLATE_ALARM_ID: "missing_official_slate_alarm",
}


@dataclass(frozen=True)
class Checkpoint:
    """One deadline-relative ops action window.

    Attributes:
        id: Stable identifier recorded via write-once official markers.
        offset: How long before the deadline this checkpoint becomes due.
        plan: Create a plan when missing.
        refresh: Retire and rebuild when availability signals are newer.
        notify: Push the recommendation report after planning.
        apply: Transfer apply (ops-05). Always False until that lands; callers
            must stub/skip apply even when this flag is True on ``T-90m``.
    """

    id: CheckpointId
    offset: timedelta
    plan: bool
    refresh: bool
    notify: bool
    apply: bool


# Earliest first (largest offset). Order is the catch-up order for missed ticks.
CHECKPOINTS: tuple[Checkpoint, ...] = (
    Checkpoint(
        id="T-24h",
        offset=timedelta(hours=24),
        plan=True,
        refresh=False,
        notify=False,
        apply=False,
    ),
    Checkpoint(
        id="T-6h",
        offset=timedelta(hours=6),
        plan=True,
        refresh=False,
        notify=True,
        apply=False,
    ),
    Checkpoint(
        id="T-90m",
        offset=timedelta(minutes=90),
        plan=True,
        refresh=True,
        notify=False,
        apply=True,  # stubbed in runners until ops-05
    ),
    Checkpoint(
        id="T-30m",
        offset=timedelta(minutes=30),
        plan=True,
        refresh=True,
        notify=False,
        apply=False,
    ),
)

_CHECKPOINT_BY_ID: dict[str, Checkpoint] = {cp.id: cp for cp in CHECKPOINTS}

NO_PLAN_RETRY_OFFSET = timedelta(hours=3)
NO_PLAN_URGENT_OFFSET = timedelta(hours=2)
MISSING_OFFICIAL_SLATE_OFFSET = timedelta(minutes=60)


@dataclass(frozen=True)
class NoPlanAlarm:
    """A no-plan alarm action that became due.

    Attributes:
        id: ``retry`` at T−3h or ``urgent`` at T−2h.
        offset: Offset before the deadline when this alarm fires.
    """

    id: NoPlanAlarmId
    offset: timedelta


@dataclass(frozen=True)
class MissingOfficialSlateAlarm:
    """Urgent alarm when the T−90m official slate is still missing at T−60m.

    Attributes:
        id: Always ``missing_official_slate``.
        offset: Offset before the deadline when this alarm fires (60m).
    """

    id: MissingOfficialSlateAlarmId
    offset: timedelta


def due_checkpoints(
    now: datetime,
    deadline: datetime | None,
    done: frozenset[str] | set[str],
) -> list[Checkpoint]:
    """Return checkpoints whose fire time has passed and are not yet done.

    A checkpoint is due when ``now >= deadline - offset`` and ``now < deadline``
    and its id is absent from ``done``. Missed exact minutes are never skipped:
    arriving late still returns every unfinished checkpoint in order.

    Args:
        now: Clock reading (timezone-aware or naive UTC).
        deadline: Upcoming deadline, or ``None`` for a blank / missing GW.
        done: Checkpoint ids already completed for this deadline.

    Returns:
        Due checkpoints earliest-first. Empty before T−24h, after the
        deadline, when ``deadline`` is ``None``, or when everything in range
        is already in ``done``.
    """
    if deadline is None:
        return []
    moment = _aware(now)
    cut = _aware(deadline)
    if moment >= cut:
        return []
    finished = set(done)
    due: list[Checkpoint] = []
    for checkpoint in CHECKPOINTS:
        if checkpoint.id in finished:
            continue
        if moment >= cut - checkpoint.offset:
            due.append(checkpoint)
    return due


def in_background_quiet_window(
    now: datetime,
    deadline: datetime | None,
    done: frozenset[str] | set[str],
    quiet: timedelta,
) -> bool:
    """Return whether mid-week fingerprint runs should stay quiet (#374).

    True when a not-yet-done checkpoint's fire time is within ``quiet`` of
    ``now`` (including already-due checkpoints). Background adhoc fingerprint
    work must not collide with T−24h / T−6h / T−90m / T−30m.

    Args:
        now: Clock reading.
        deadline: Upcoming deadline, or ``None`` when there is none.
        done: Checkpoint ids already completed for this deadline.
        quiet: Quiet window length (``ops.background_quiet_minutes``).

    Returns:
        True when background fingerprint runs should be skipped.
    """
    if deadline is None:
        return False
    moment = _aware(now)
    cut = _aware(deadline)
    if moment >= cut:
        return False
    finished = set(done)
    for checkpoint in CHECKPOINTS:
        if checkpoint.id in finished:
            continue
        fire_at = cut - checkpoint.offset
        if fire_at <= moment + quiet:
            return True
    return False


def due_no_plan_alarms(
    now: datetime,
    deadline: datetime | None,
    *,
    has_plan: bool,
    alarms_done: frozenset[str] | set[str],
) -> list[NoPlanAlarm]:
    """Return no-plan alarm actions that are due and not yet recorded.

    At T−3h, ``retry`` is due when any production model still lacks a plan.
    At T−2h, ``urgent`` is due when the plan is still missing. Both catch up
    on a late tick the same way as plan checkpoints.

    Args:
        now: Clock reading.
        deadline: Upcoming deadline, or ``None`` when there is none.
        has_plan: True when every production model already has a plan.
        alarms_done: Alarm ids already fired for this deadline.

    Returns:
        Due alarms earliest-first (``retry`` before ``urgent``). Empty when
        a plan exists, after the deadline, or when ``deadline`` is ``None``.
    """
    if deadline is None or has_plan:
        return []
    moment = _aware(now)
    cut = _aware(deadline)
    if moment >= cut:
        return []
    finished = set(alarms_done)
    due: list[NoPlanAlarm] = []
    if moment >= cut - NO_PLAN_RETRY_OFFSET and "retry" not in finished:
        due.append(NoPlanAlarm(id="retry", offset=NO_PLAN_RETRY_OFFSET))
    if moment >= cut - NO_PLAN_URGENT_OFFSET and "urgent" not in finished:
        due.append(NoPlanAlarm(id="urgent", offset=NO_PLAN_URGENT_OFFSET))
    return due


def due_missing_official_slate_alarm(
    now: datetime,
    deadline: datetime | None,
    *,
    has_official_slate: bool,
    alarms_done: frozenset[str] | set[str],
) -> MissingOfficialSlateAlarm | None:
    """Return the T−60m missing-official-slate alarm when due and not recorded.

    Fires once at/after ``deadline - 60m`` when the T−90m official slate
    pointers are still missing. Never before T−60m, never when the slate
    exists, and never after the deadline. Late ticks catch up the same way
    as other alarms.

    Args:
        now: Clock reading.
        deadline: Upcoming deadline, or ``None`` when there is none.
        has_official_slate: True when every official slate line has a
            ``checkpoint=T-90m/line=<line>.json`` pointer.
        alarms_done: Alarm ids already fired for this deadline.

    Returns:
        The alarm when due, else ``None``.
    """
    if deadline is None or has_official_slate:
        return None
    if MISSING_OFFICIAL_SLATE_ALARM_ID in alarms_done:
        return None
    moment = _aware(now)
    cut = _aware(deadline)
    if moment >= cut:
        return None
    if moment < cut - MISSING_OFFICIAL_SLATE_OFFSET:
        return None
    return MissingOfficialSlateAlarm(
        id=MISSING_OFFICIAL_SLATE_ALARM_ID,
        offset=MISSING_OFFICIAL_SLATE_OFFSET,
    )


def next_upcoming_deadline(
    store: DataStore,
    now: datetime,
) -> tuple[GWKey, datetime] | None:
    """Earliest unchecked gameweek whose deadline is still in the future.

    A blank stretch (no future unchecked rows) returns ``None``. Double
    gameweeks share one FPL deadline, so they yield a single pair.

    Args:
        store: Staged tables, including ``gameweeks``.
        now: Clock reading.

    Returns:
        ``(key, deadline_time)`` or ``None``.
    """
    path = store.path("gameweeks")
    if not path.exists():
        return None
    frame = store.read("gameweeks")
    if frame.is_empty():
        return None
    latest = frame.sort("available_at").unique(subset=["season", "gw"], keep="last")
    moment = _aware(now)
    upcoming: list[tuple[datetime, GWKey]] = []
    for row in latest.iter_rows(named=True):
        if bool(row["data_checked"]):
            continue
        deadline = _aware(row["deadline_time"])
        if deadline > moment:
            upcoming.append((deadline, GWKey(str(row["season"]), int(row["gw"]))))
    if not upcoming:
        return None
    upcoming.sort(key=lambda item: (item[0], item[1]))
    deadline, key = upcoming[0]
    return key, deadline


def official_gw_prefix(key: GWKey) -> str:
    """Return the R2 / local prefix for one season/GW under ``official/``."""
    return f"official/season={key.season}/gw={key.gw}"


def list_official_gw_outputs(
    root: Path,
    key: GWKey,
    *,
    settings: Settings | None = None,
    remote: Remote | None = None,
) -> frozenset[str]:
    """List relative paths under ``official/season=/gw=/`` (remote preferred).

    Prefers ``rclone lsf -R`` against the data bucket when configured so a
    fresh runner sees write-once markers without a full pull. Falls back to
    the local tree when no remote is available.

    Args:
        root: Local data directory (fallback listing root).
        key: Season / gameweek.
        settings: Process settings for ``Remote.from_settings``.
        remote: Optional pre-built remote (tests).

    Returns:
        Relative paths such as ``checkpoint=T-6h/notified.json``.
    """
    from fplt.data.remote import Remote, RemoteError, lsf
    from fplt.settings import get_settings

    cfg = get_settings() if settings is None else settings
    target = remote if remote is not None else Remote.from_settings(cfg)
    prefix = official_gw_prefix(key)
    if target is not None:
        if target.kind != "data":
            raise ValueError(
                f"official listing requires the data bucket; got kind={target.kind!r}"
            )
        try:
            return frozenset(lsf(target, prefix, recursive=True))
        except RemoteError:
            logger.exception(
                "official lsf failed for %s; falling back to local",
                prefix,
            )
    return _list_local_official_gw(root, key)


def done_from_official_listing(
    listing: Iterable[str],
    *,
    primary_line: str,
    slate_lines: Iterable[str],
) -> tuple[frozenset[str], frozenset[str]]:
    """Derive checkpoint / alarm completion sets from an official GW listing.

    Args:
        listing: Relative paths under ``official/season=/gw=/``.
        primary_line: Primary official line name (plan-checkpoint marker).
        slate_lines: Every line in ``ProductionOps.official_slate`` (T−90m).

    Returns:
        ``(done_checkpoints, alarms_done)``.
    """
    paths = frozenset(listing)
    slate = tuple(slate_lines)
    done: set[str] = set()
    for checkpoint in CHECKPOINTS:
        if checkpoint.id == "T-90m":
            if slate and all(
                f"checkpoint=T-90m/line={line}.json" in paths for line in slate
            ):
                done.add("T-90m")
            continue
        pointer = f"checkpoint={checkpoint.id}/line={primary_line}.json"
        if pointer not in paths:
            continue
        if checkpoint.id == "T-6h":
            if f"checkpoint=T-6h/{_NOTIFIED_NAME}" in paths:
                done.add("T-6h")
        else:
            done.add(checkpoint.id)
    alarms = frozenset(
        alarm for alarm in _KNOWN_ALARM_IDS if f"alarms/{alarm}.json" in paths
    )
    return frozenset(done), alarms


def resolve_done_sets(
    root: Path,
    key: GWKey,
    *,
    primary_line: str,
    slate_lines: Iterable[str],
    settings: Settings | None = None,
    remote: Remote | None = None,
) -> tuple[frozenset[str], frozenset[str]]:
    """List official outputs and return ``(done, alarms_done)``.

    Args:
        root: Local data directory.
        key: Upcoming gameweek.
        primary_line: Primary line name for plan-checkpoint markers.
        slate_lines: Official slate line names for T−90m completeness.
        settings: Process settings.
        remote: Optional pre-built remote (tests).

    Returns:
        Checkpoint ids and alarm ids already completed for ``key``.
    """
    listing = list_official_gw_outputs(root, key, settings=settings, remote=remote)
    return done_from_official_listing(
        listing,
        primary_line=primary_line,
        slate_lines=slate_lines,
    )


def notified_path(root: Path, key: GWKey, *, checkpoint: str = "T-6h") -> Path:
    """Path for the write-once T−6h notify marker."""
    return (
        official_checkpoint_dir(
            root, season=key.season, gw=key.gw, checkpoint=checkpoint
        )
        / _NOTIFIED_NAME
    )


def alarm_marker_path(root: Path, key: GWKey, alarm_id: str) -> Path:
    """Path for a write-once alarm marker under ``alarms/<alarm_id>.json``."""
    return (
        root
        / "official"
        / f"season={key.season}"
        / f"gw={key.gw}"
        / "alarms"
        / f"{alarm_id}.json"
    )


def write_plan_checkpoint_marker(
    root: Path,
    key: GWKey,
    checkpoint_id: str,
    *,
    primary_line: str,
    written_at: datetime | None = None,
) -> Path:
    """Write ``line=<primary>.json`` once for a plan checkpoint (T−24h/T−6h/T−30m).

    Idempotent: if the marker already exists, return it without error.

    Args:
        root: Data directory.
        key: Season / gameweek.
        checkpoint_id: Checkpoint id (directory name).
        primary_line: Primary official line name.
        written_at: Marker timestamp (default now UTC).

    Returns:
        Path of the marker file.
    """
    dest = pointer_path(
        root,
        season=key.season,
        gw=key.gw,
        line=primary_line,
        checkpoint=checkpoint_id,
    )
    moment = _aware(written_at or datetime.now(UTC))
    payload = {
        "checkpoint": checkpoint_id,
        "line": primary_line,
        "kind": "plan_checkpoint",
        "written_at": moment.isoformat(),
    }
    return _write_json_once(dest, payload)


def write_notified_marker(
    root: Path,
    key: GWKey,
    *,
    written_at: datetime | None = None,
) -> Path:
    """Write ``checkpoint=T-6h/notified.json`` once after a successful ntfy send.

    Idempotent: existing file is left untouched.

    Args:
        root: Data directory.
        key: Season / gameweek.
        written_at: Marker timestamp (default now UTC).

    Returns:
        Path of the marker file.
    """
    dest = notified_path(root, key)
    moment = _aware(written_at or datetime.now(UTC))
    payload = {
        "checkpoint": "T-6h",
        "kind": "notified",
        "written_at": moment.isoformat(),
    }
    return _write_json_once(dest, payload)


def write_alarm_marker(
    root: Path,
    key: GWKey,
    alarm_id: str,
    *,
    written_at: datetime | None = None,
    kind: str | None = None,
) -> Path:
    """Write ``alarms/<alarm_id>.json`` once after an alarm fires.

    Idempotent: existing file is left untouched.

    Args:
        root: Data directory.
        key: Season / gameweek.
        alarm_id: ``retry``, ``urgent``, or ``missing_official_slate``.
        written_at: Marker timestamp (default now UTC).
        kind: Marker kind override. Defaults from ``alarm_id``.

    Returns:
        Path of the marker file.
    """
    dest = alarm_marker_path(root, key, alarm_id)
    moment = _aware(written_at or datetime.now(UTC))
    payload = {
        "alarm": alarm_id,
        "kind": kind or _ALARM_KIND.get(alarm_id, "alarm"),
        "written_at": moment.isoformat(),
    }
    return _write_json_once(dest, payload)


def push_checkpoint_markers(
    root: Path,
    key: GWKey,
    checkpoint_id: str,
    *,
    settings: Settings | None = None,
    remote: Remote | None = None,
) -> bool:
    """Push write-once markers under ``official/.../checkpoint=<id>/``.

    Soft-skips when no data bucket is configured. Existing remote objects are
    left alone (``push`` uses ``--immutable``; identical first write wins).

    Args:
        root: Local data directory.
        key: Season / gameweek.
        checkpoint_id: Checkpoint directory to push.
        settings: Process settings.
        remote: Optional pre-built remote (tests).

    Returns:
        True when a push was attempted.
    """
    from fplt.data.remote import Remote, RemoteError
    from fplt.settings import get_settings
    from fplt.store.push import push_official

    cfg = get_settings() if settings is None else settings
    target = remote if remote is not None else Remote.from_settings(cfg)
    if target is None:
        return False
    try:
        push_official(
            root,
            season=key.season,
            gw=key.gw,
            checkpoint=checkpoint_id,
            settings=cfg,
            remote=target,
        )
    except (RemoteError, FileNotFoundError, ValueError) as exc:
        # Immutable conflict when the marker is already on R2 is success.
        detail = str(exc).lower()
        if "immutable" in detail or "already" in detail:
            logger.info(
                "official checkpoint markers already on remote key=%s cp=%s",
                key,
                checkpoint_id,
            )
            return True
        logger.exception(
            "failed to push checkpoint markers key=%s cp=%s",
            key,
            checkpoint_id,
        )
        return False
    return True


def push_alarm_markers(
    root: Path,
    key: GWKey,
    *,
    settings: Settings | None = None,
    remote: Remote | None = None,
) -> bool:
    """Push write-once ``alarms/`` markers for one GW.

    Args:
        root: Local data directory.
        key: Season / gameweek.
        settings: Process settings.
        remote: Optional pre-built remote (tests).

    Returns:
        True when a push was attempted.
    """
    from fplt.data.remote import Remote, RemoteError, push
    from fplt.settings import get_settings

    cfg = get_settings() if settings is None else settings
    target = remote if remote is not None else Remote.from_settings(cfg)
    if target is None:
        return False
    local = root / "official" / f"season={key.season}" / f"gw={key.gw}" / "alarms"
    if not local.is_dir() or not any(local.iterdir()):
        return False
    prefix = f"{official_gw_prefix(key)}/alarms"
    try:
        push(target, local, prefix)
    except (RemoteError, ValueError) as exc:
        detail = str(exc).lower()
        if "immutable" in detail or "already" in detail:
            logger.info("alarm markers already on remote key=%s", key)
            return True
        logger.exception("failed to push alarm markers key=%s", key)
        return False
    return True


def checkpoint_by_id(checkpoint_id: str) -> Checkpoint:
    """Look up a defined checkpoint by id.

    Raises:
        KeyError: Unknown id.
    """
    try:
        return _CHECKPOINT_BY_ID[checkpoint_id]
    except KeyError as exc:
        raise KeyError(f"unknown checkpoint id: {checkpoint_id!r}") from exc


def _list_local_official_gw(root: Path, key: GWKey) -> frozenset[str]:
    base = root / "official" / f"season={key.season}" / f"gw={key.gw}"
    if not base.is_dir():
        return frozenset()
    out: set[str] = set()
    for path in base.rglob("*"):
        if path.is_file():
            out.add(path.relative_to(base).as_posix())
    return frozenset(out)


def _write_json_once(dest: Path, payload: dict[str, Any]) -> Path:
    if dest.is_file():
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return dest


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = [
    "CHECKPOINTS",
    "MISSING_OFFICIAL_SLATE_ALARM_ID",
    "MISSING_OFFICIAL_SLATE_OFFSET",
    "AlarmId",
    "Checkpoint",
    "CheckpointId",
    "MissingOfficialSlateAlarm",
    "MissingOfficialSlateAlarmId",
    "NO_PLAN_RETRY_OFFSET",
    "NO_PLAN_URGENT_OFFSET",
    "NoPlanAlarm",
    "NoPlanAlarmId",
    "alarm_marker_path",
    "checkpoint_by_id",
    "done_from_official_listing",
    "due_checkpoints",
    "due_missing_official_slate_alarm",
    "due_no_plan_alarms",
    "list_official_gw_outputs",
    "next_upcoming_deadline",
    "notified_path",
    "official_gw_prefix",
    "push_alarm_markers",
    "push_checkpoint_markers",
    "resolve_done_sets",
    "write_alarm_marker",
    "write_notified_marker",
    "write_plan_checkpoint_marker",
]
