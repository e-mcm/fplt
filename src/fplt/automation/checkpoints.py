"""Deadline-anchored ops checkpoints and the no-plan alarm (ops-02).

GitHub Actions cron is best-effort: ticks can arrive late or be dropped.
``due_checkpoints`` lists every checkpoint whose fire time has passed and
that is not yet recorded in ``done``, so a missed T−24h still runs on the
next tick. After the deadline nothing is due.

Checkpoints (offsets before the deadline):

* ``T-24h`` — full plan
* ``T-6h`` — plan + notify
* ``T-90m`` — final refresh; apply is stubbed until ops-05
* ``T-30m`` — last-chance refresh only

No-plan alarm (separate from the plan checkpoints):

* ``T-3h`` — retry / catch-up when any production model lacks a plan
* ``T-2h`` — urgent notification if still missing
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Literal, cast

from fplt.data.store import DataStore
from fplt.types import GWKey

logger = logging.getLogger(__name__)

CheckpointId = Literal["T-24h", "T-6h", "T-90m", "T-30m"]
AlarmId = Literal["retry", "urgent"]

_STATE_REL = Path("ops") / "checkpoint_state.json"


@dataclass(frozen=True)
class Checkpoint:
    """One deadline-relative ops action window.

    Attributes:
        id: Stable identifier recorded in checkpoint state when completed.
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


@dataclass(frozen=True)
class NoPlanAlarm:
    """A no-plan alarm action that became due.

    Attributes:
        id: ``retry`` at T−3h or ``urgent`` at T−2h.
        offset: Offset before the deadline when this alarm fires.
    """

    id: AlarmId
    offset: timedelta


@dataclass(frozen=True)
class CheckpointState:
    """Persisted completion set for one upcoming deadline.

    Attributes:
        key: Gameweek the state belongs to.
        deadline: That gameweek's ``deadline_time`` (UTC).
        done: Checkpoint ids already completed for this deadline.
        alarms_done: No-plan alarm ids already fired for this deadline.
    """

    key: GWKey
    deadline: datetime
    done: frozenset[str]
    alarms_done: frozenset[str]


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


def checkpoint_state_path(root: Path) -> Path:
    """Return the on-disk path for checkpoint completion state."""
    return root / _STATE_REL


def load_checkpoint_state(root: Path) -> CheckpointState | None:
    """Load persisted checkpoint state, or ``None`` when missing/corrupt.

    Args:
        root: Data directory (contains ``ops/``).

    Returns:
        Parsed state, or ``None`` when the file is absent or unreadable.
    """
    path = checkpoint_state_path(root)
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        logger.warning("checkpoint state unreadable at %s: %s", path, exc)
        return None
    if not isinstance(raw, dict):
        return None
    payload = cast(dict[str, Any], raw)
    try:
        season = str(payload["season"])
        gw = int(payload["gw"])
        deadline = _aware(datetime.fromisoformat(str(payload["deadline"])))
        done_raw = payload.get("done", [])
        alarms_raw = payload.get("alarms_done", [])
        if not isinstance(done_raw, list) or not isinstance(alarms_raw, list):
            raise TypeError("done and alarms_done must be lists")
        done = frozenset(str(item) for item in cast(list[object], done_raw))
        alarms = frozenset(str(item) for item in cast(list[object], alarms_raw))
    except (KeyError, TypeError, ValueError) as exc:
        logger.warning("checkpoint state invalid at %s: %s", path, exc)
        return None
    return CheckpointState(
        key=GWKey(season, gw),
        deadline=deadline,
        done=done,
        alarms_done=alarms,
    )


def save_checkpoint_state(root: Path, state: CheckpointState) -> Path:
    """Write checkpoint completion state (overwrites the single status file).

    Args:
        root: Data directory.
        state: State to persist for the current upcoming deadline.

    Returns:
        Path written.
    """
    path = checkpoint_state_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "season": state.key.season,
        "gw": state.key.gw,
        "deadline": _aware(state.deadline).isoformat(),
        "done": sorted(state.done),
        "alarms_done": sorted(state.alarms_done),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def resolve_checkpoint_state(
    root: Path,
    key: GWKey,
    deadline: datetime,
) -> CheckpointState:
    """Return state for ``key``/``deadline``, resetting when the deadline moves.

    Args:
        root: Data directory.
        key: Upcoming gameweek.
        deadline: That gameweek's deadline.

    Returns:
        Existing state when it matches ``key`` and ``deadline``, otherwise a
        fresh empty state.
    """
    cut = _aware(deadline)
    existing = load_checkpoint_state(root)
    if (
        existing is not None
        and existing.key == key
        and _aware(existing.deadline) == cut
    ):
        return existing
    return CheckpointState(
        key=key,
        deadline=cut,
        done=frozenset(),
        alarms_done=frozenset(),
    )


def mark_checkpoint_done(state: CheckpointState, checkpoint_id: str) -> CheckpointState:
    """Return a copy of ``state`` with ``checkpoint_id`` added to ``done``."""
    return CheckpointState(
        key=state.key,
        deadline=state.deadline,
        done=frozenset(set(state.done) | {checkpoint_id}),
        alarms_done=state.alarms_done,
    )


def mark_alarm_done(state: CheckpointState, alarm_id: str) -> CheckpointState:
    """Return a copy of ``state`` with ``alarm_id`` added to ``alarms_done``."""
    return CheckpointState(
        key=state.key,
        deadline=state.deadline,
        done=state.done,
        alarms_done=frozenset(set(state.alarms_done) | {alarm_id}),
    )


def checkpoint_by_id(checkpoint_id: str) -> Checkpoint:
    """Look up a defined checkpoint by id.

    Raises:
        KeyError: Unknown id.
    """
    try:
        return _CHECKPOINT_BY_ID[checkpoint_id]
    except KeyError as exc:
        raise KeyError(f"unknown checkpoint id: {checkpoint_id!r}") from exc


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = [
    "CHECKPOINTS",
    "AlarmId",
    "Checkpoint",
    "CheckpointId",
    "CheckpointState",
    "NO_PLAN_RETRY_OFFSET",
    "NO_PLAN_URGENT_OFFSET",
    "NoPlanAlarm",
    "checkpoint_by_id",
    "checkpoint_state_path",
    "due_checkpoints",
    "due_no_plan_alarms",
    "load_checkpoint_state",
    "mark_alarm_done",
    "mark_checkpoint_done",
    "next_upcoming_deadline",
    "resolve_checkpoint_state",
    "save_checkpoint_state",
]
