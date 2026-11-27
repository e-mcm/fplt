"""Table-driven tests for deadline checkpoints and the no-plan alarm."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from fplt.automation.checkpoints import (
    CHECKPOINTS,
    CheckpointState,
    due_checkpoints,
    due_no_plan_alarms,
    load_checkpoint_state,
    mark_alarm_done,
    mark_checkpoint_done,
    next_upcoming_deadline,
    resolve_checkpoint_state,
    save_checkpoint_state,
)
from fplt.data.store import DataStore
from fplt.types import GWKey

_DEADLINE = datetime(2024, 9, 1, 17, 0, tzinfo=UTC)


def _ids(now: datetime, done: set[str]) -> list[str]:
    return [cp.id for cp in due_checkpoints(now, _DEADLINE, done)]


@pytest.mark.parametrize(
    ("offset_before", "done", "expected"),
    [
        # before T−24h
        (timedelta(hours=25), set(), []),
        # exactly at T−24h
        (timedelta(hours=24), set(), ["T-24h"]),
        # between T−24h and T−6h
        (timedelta(hours=12), set(), ["T-24h"]),
        (timedelta(hours=12), {"T-24h"}, []),
        # between T−6h and T−90m
        (timedelta(hours=4), {"T-24h"}, ["T-6h"]),
        (timedelta(hours=4), set(), ["T-24h", "T-6h"]),
        # between T−90m and T−30m
        (timedelta(minutes=60), {"T-24h", "T-6h"}, ["T-90m"]),
        # between T−30m and deadline
        (timedelta(minutes=15), {"T-24h", "T-6h", "T-90m"}, ["T-30m"]),
        # missed checkpoint caught up late (exact minute passed)
        (timedelta(hours=5, minutes=50), set(), ["T-24h", "T-6h"]),
        (
            timedelta(minutes=20),
            {"T-24h"},
            ["T-6h", "T-90m", "T-30m"],
        ),
        # all done inside the window
        (
            timedelta(minutes=10),
            {"T-24h", "T-6h", "T-90m", "T-30m"},
            [],
        ),
    ],
)
def test_due_checkpoints_table(
    offset_before: timedelta,
    done: set[str],
    expected: list[str],
) -> None:
    now = _DEADLINE - offset_before
    assert _ids(now, done) == expected


def test_due_checkpoints_empty_after_deadline() -> None:
    assert due_checkpoints(_DEADLINE, _DEADLINE, set()) == []
    assert due_checkpoints(_DEADLINE + timedelta(minutes=1), _DEADLINE, set()) == []
    # Missed work is not run after the deadline has passed.
    assert due_checkpoints(_DEADLINE + timedelta(hours=1), _DEADLINE, set()) == []


def test_due_checkpoints_blank_gameweek_has_no_deadline() -> None:
    assert due_checkpoints(_DEADLINE - timedelta(hours=1), None, set()) == []


def test_due_checkpoints_double_gameweek_one_deadline() -> None:
    """A DGW shares one FPL deadline → one checkpoint series."""
    now = _DEADLINE - timedelta(hours=5)
    assert [cp.id for cp in due_checkpoints(now, _DEADLINE, set())] == [
        "T-24h",
        "T-6h",
    ]
    # Completing against that single deadline clears the series.
    assert due_checkpoints(now, _DEADLINE, {"T-24h", "T-6h"}) == []


def test_checkpoint_actions_match_spec() -> None:
    by_id = {cp.id: cp for cp in CHECKPOINTS}
    assert by_id["T-24h"].plan
    assert not by_id["T-24h"].notify
    assert by_id["T-6h"].plan
    assert by_id["T-6h"].notify
    assert by_id["T-90m"].refresh
    assert by_id["T-90m"].apply
    assert by_id["T-30m"].refresh
    assert not by_id["T-30m"].apply
    assert by_id["T-90m"].offset == timedelta(minutes=90)
    assert by_id["T-30m"].offset == timedelta(minutes=30)


@pytest.mark.parametrize(
    ("offset_before", "has_plan", "alarms_done", "expected"),
    [
        (timedelta(hours=4), False, set(), []),
        (timedelta(hours=3), False, set(), ["retry"]),
        (timedelta(hours=3), True, set(), []),
        (timedelta(hours=2, minutes=30), False, {"retry"}, []),
        (timedelta(hours=2), False, {"retry"}, ["urgent"]),
        (timedelta(hours=2), False, set(), ["retry", "urgent"]),
        (timedelta(hours=1), True, set(), []),
        (timedelta(minutes=30), False, {"retry", "urgent"}, []),
    ],
)
def test_due_no_plan_alarms_table(
    offset_before: timedelta,
    has_plan: bool,
    alarms_done: set[str],
    expected: list[str],
) -> None:
    now = _DEADLINE - offset_before
    got = due_no_plan_alarms(now, _DEADLINE, has_plan=has_plan, alarms_done=alarms_done)
    assert [item.id for item in got] == expected


def test_due_no_plan_alarms_blank_and_after_deadline() -> None:
    now = _DEADLINE - timedelta(hours=1)
    assert due_no_plan_alarms(now, None, has_plan=False, alarms_done=set()) == []
    assert (
        due_no_plan_alarms(
            _DEADLINE + timedelta(minutes=1),
            _DEADLINE,
            has_plan=False,
            alarms_done=set(),
        )
        == []
    )


def test_next_upcoming_deadline_blank_and_double(
    tmp_path: Path,
) -> None:
    store = DataStore(tmp_path)
    assert next_upcoming_deadline(store, _DEADLINE - timedelta(hours=1)) is None

    # Blank stretch: only checked GWs remain.
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": ["2024-25", "2024-25"],
                "gw": [1, 2],
                "deadline_time": [
                    _DEADLINE - timedelta(days=14),
                    _DEADLINE - timedelta(days=7),
                ],
                "finished": [True, True],
                "data_checked": [True, True],
                "available_at": [
                    _DEADLINE - timedelta(days=20),
                    _DEADLINE - timedelta(days=20),
                ],
            }
        ),
    )
    assert next_upcoming_deadline(store, _DEADLINE - timedelta(hours=1)) is None

    # Double GW: one shared deadline for gw 3 (fixtures elsewhere); one pair.
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": ["2024-25"],
                "gw": [3],
                "deadline_time": [_DEADLINE],
                "finished": [False],
                "data_checked": [False],
                "available_at": [_DEADLINE - timedelta(days=20)],
            }
        ),
    )
    got = next_upcoming_deadline(store, _DEADLINE - timedelta(hours=1))
    assert got == (GWKey("2024-25", 3), _DEADLINE)


def test_checkpoint_state_roundtrip_and_reset(tmp_path: Path) -> None:
    key = GWKey("2024-25", 3)
    state = CheckpointState(
        key=key,
        deadline=_DEADLINE,
        done=frozenset({"T-24h"}),
        alarms_done=frozenset({"retry"}),
    )
    save_checkpoint_state(tmp_path, state)
    loaded = load_checkpoint_state(tmp_path)
    assert loaded == state

    same = resolve_checkpoint_state(tmp_path, key, _DEADLINE)
    assert same.done == frozenset({"T-24h"})

    # New deadline → fresh state.
    fresh = resolve_checkpoint_state(tmp_path, key, _DEADLINE + timedelta(days=7))
    assert fresh.done == frozenset()
    assert fresh.alarms_done == frozenset()

    advanced = mark_checkpoint_done(state, "T-6h")
    advanced = mark_alarm_done(advanced, "urgent")
    assert advanced.done == frozenset({"T-24h", "T-6h"})
    assert advanced.alarms_done == frozenset({"retry", "urgent"})
