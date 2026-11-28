"""Table-driven tests for deadline checkpoints and the no-plan alarm."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from fplt.automation.checkpoints import (
    CHECKPOINTS,
    MISSING_OFFICIAL_SLATE_ALARM_ID,
    done_from_official_listing,
    due_checkpoints,
    due_missing_official_slate_alarm,
    due_no_plan_alarms,
    list_official_gw_outputs,
    next_upcoming_deadline,
    push_alarm_markers,
    push_checkpoint_markers,
    resolve_done_sets,
    write_alarm_marker,
    write_notified_marker,
    write_plan_checkpoint_marker,
)
from fplt.data.remote import Remote, RemoteError
from fplt.data.store import DataStore
from fplt.settings import Settings
from fplt.types import GWKey

_DEADLINE = datetime(2024, 9, 1, 17, 0, tzinfo=UTC)
_PRIMARY = "v4_h3"
_SLATE = ("v4_h3", "v4_h5")


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


def test_due_checkpoints_none_deadline() -> None:
    assert due_checkpoints(_DEADLINE - timedelta(hours=1), None, set()) == []


def test_due_no_plan_alarms_table() -> None:
    assert (
        due_no_plan_alarms(
            _DEADLINE - timedelta(hours=4),
            _DEADLINE,
            has_plan=False,
            alarms_done=set(),
        )
        == []
    )
    retry = due_no_plan_alarms(
        _DEADLINE - timedelta(hours=3),
        _DEADLINE,
        has_plan=False,
        alarms_done=set(),
    )
    assert [a.id for a in retry] == ["retry"]
    both = due_no_plan_alarms(
        _DEADLINE - timedelta(hours=2),
        _DEADLINE,
        has_plan=False,
        alarms_done=set(),
    )
    assert [a.id for a in both] == ["retry", "urgent"]
    assert (
        due_no_plan_alarms(
            _DEADLINE - timedelta(hours=1),
            _DEADLINE,
            has_plan=True,
            alarms_done=set(),
        )
        == []
    )
    assert (
        due_no_plan_alarms(
            _DEADLINE - timedelta(hours=1),
            _DEADLINE,
            has_plan=False,
            alarms_done={"retry", "urgent"},
        )
        == []
    )


def test_due_missing_official_slate_alarm_table() -> None:
    """T−60m missing-slate alarm fires once when T−90m pointers are absent (#373)."""
    # Before T−60m: never.
    assert (
        due_missing_official_slate_alarm(
            _DEADLINE - timedelta(minutes=61),
            _DEADLINE,
            has_official_slate=False,
            alarms_done=set(),
        )
        is None
    )
    # Exactly at T−60m with no slate: fires.
    due = due_missing_official_slate_alarm(
        _DEADLINE - timedelta(minutes=60),
        _DEADLINE,
        has_official_slate=False,
        alarms_done=set(),
    )
    assert due is not None
    assert due.id == MISSING_OFFICIAL_SLATE_ALARM_ID
    # After T−60m, still missing: still due until recorded.
    late = due_missing_official_slate_alarm(
        _DEADLINE - timedelta(minutes=45),
        _DEADLINE,
        has_official_slate=False,
        alarms_done=set(),
    )
    assert late is not None
    assert late.id == MISSING_OFFICIAL_SLATE_ALARM_ID
    # Slate present: never.
    assert (
        due_missing_official_slate_alarm(
            _DEADLINE - timedelta(minutes=45),
            _DEADLINE,
            has_official_slate=True,
            alarms_done=set(),
        )
        is None
    )
    # Already recorded: never again.
    assert (
        due_missing_official_slate_alarm(
            _DEADLINE - timedelta(minutes=45),
            _DEADLINE,
            has_official_slate=False,
            alarms_done={MISSING_OFFICIAL_SLATE_ALARM_ID},
        )
        is None
    )
    # After deadline: never.
    assert (
        due_missing_official_slate_alarm(
            _DEADLINE + timedelta(minutes=1),
            _DEADLINE,
            has_official_slate=False,
            alarms_done=set(),
        )
        is None
    )
    assert (
        due_missing_official_slate_alarm(
            _DEADLINE - timedelta(minutes=30),
            None,
            has_official_slate=False,
            alarms_done=set(),
        )
        is None
    )


def test_checkpoint_offsets_match_spec() -> None:
    by_id = {cp.id: cp for cp in CHECKPOINTS}
    assert by_id["T-24h"].offset == timedelta(hours=24)
    assert by_id["T-6h"].notify is True
    assert by_id["T-90m"].apply is True
    assert by_id["T-30m"].refresh is True


def test_next_upcoming_deadline(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
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


def test_done_from_official_listing_markers() -> None:
    listing = {
        "checkpoint=T-24h/line=v4_h3.json",
        "checkpoint=T-6h/line=v4_h3.json",
        # missing notified.json → T-6h not done
        "checkpoint=T-90m/line=v4_h3.json",
        "alarms/retry.json",
    }
    done, alarms = done_from_official_listing(
        listing, primary_line=_PRIMARY, slate_lines=_SLATE
    )
    assert done == frozenset({"T-24h"})
    assert alarms == frozenset({"retry"})

    listing.add("checkpoint=T-6h/notified.json")
    listing.add("checkpoint=T-90m/line=v4_h5.json")
    listing.add("checkpoint=T-30m/line=v4_h3.json")
    listing.add("alarms/urgent.json")
    listing.add("alarms/missing_official_slate.json")
    done, alarms = done_from_official_listing(
        listing, primary_line=_PRIMARY, slate_lines=_SLATE
    )
    assert done == frozenset({"T-24h", "T-6h", "T-90m", "T-30m"})
    assert alarms == frozenset({"retry", "urgent", "missing_official_slate"})


def test_write_markers_idempotent(tmp_path: Path) -> None:
    key = GWKey("2024-25", 3)
    first = write_plan_checkpoint_marker(tmp_path, key, "T-24h", primary_line=_PRIMARY)
    second = write_plan_checkpoint_marker(tmp_path, key, "T-24h", primary_line=_PRIMARY)
    assert first == second
    assert first.is_file()
    write_notified_marker(tmp_path, key)
    write_notified_marker(tmp_path, key)
    write_alarm_marker(tmp_path, key, "retry")
    write_alarm_marker(tmp_path, key, "retry")
    done, alarms = resolve_done_sets(
        tmp_path,
        key,
        primary_line=_PRIMARY,
        slate_lines=_SLATE,
    )
    assert "T-24h" in done
    assert "T-6h" not in done  # notified without T-6h pointer
    assert alarms == frozenset({"retry"})


@pytest.mark.skipif(
    shutil.which("rclone") is None, reason="rclone binary not installed"
)
def test_list_official_gw_outputs_via_local_remote(tmp_path: Path) -> None:
    bucket = tmp_path / "bucket"
    data = tmp_path / "data"
    data.mkdir()
    key = GWKey("2024-25", 3)
    local_marker = write_plan_checkpoint_marker(
        data, key, "T-24h", primary_line=_PRIMARY
    )
    # Mirror into the local-dir remote as R2 would hold it.
    remote_cp = bucket / "official" / "season=2024-25" / "gw=3" / "checkpoint=T-24h"
    remote_cp.mkdir(parents=True)
    (remote_cp / f"line={_PRIMARY}.json").write_text(
        local_marker.read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    remote = Remote(target=str(bucket), kind="data", env={})
    fresh = tmp_path / "fresh"
    fresh.mkdir()
    listing = list_official_gw_outputs(
        fresh,
        key,
        settings=Settings(data_dir=fresh, data_bucket="fplt-data"),
        remote=remote,
    )
    assert f"checkpoint=T-24h/line={_PRIMARY}.json" in listing


def test_list_official_rejects_lab_remote(tmp_path: Path) -> None:
    key = GWKey("2024-25", 3)
    lab = Remote(target="fplt:fplt-lab", kind="lab", env={})
    with pytest.raises(ValueError, match="data bucket"):
        list_official_gw_outputs(tmp_path, key, remote=lab)


def test_list_official_falls_back_locally_on_lsf_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = GWKey("2024-25", 3)
    write_plan_checkpoint_marker(tmp_path, key, "T-24h", primary_line=_PRIMARY)
    remote = Remote(target="fplt:fplt-data", kind="data", env={})

    def boom(*_a: object, **_k: object) -> list[str]:
        raise RemoteError("rclone exploded")

    monkeypatch.setattr("fplt.data.remote.lsf", boom)
    listing = list_official_gw_outputs(tmp_path, key, remote=remote)
    assert f"checkpoint=T-24h/line={_PRIMARY}.json" in listing


def test_push_checkpoint_markers_immutable_and_other_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = GWKey("2024-25", 3)
    write_plan_checkpoint_marker(tmp_path, key, "T-24h", primary_line=_PRIMARY)
    remote = Remote(target="fplt:fplt-data", kind="data", env={})

    def immutable(*_a: object, **_k: object) -> None:
        raise RemoteError("immutable file modified")

    monkeypatch.setattr("fplt.store.push.push_official", immutable)
    assert push_checkpoint_markers(tmp_path, key, "T-24h", remote=remote) is True

    def other(*_a: object, **_k: object) -> None:
        raise RemoteError("network down")

    monkeypatch.setattr("fplt.store.push.push_official", other)
    assert push_checkpoint_markers(tmp_path, key, "T-24h", remote=remote) is False


def test_push_alarm_markers_empty_immutable_and_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    key = GWKey("2024-25", 3)
    remote = Remote(target="fplt:fplt-data", kind="data", env={})
    assert push_alarm_markers(tmp_path, key, remote=remote) is False

    write_alarm_marker(tmp_path, key, "retry")

    monkeypatch.setattr("fplt.data.remote.push", lambda *_a, **_k: None)
    assert push_alarm_markers(tmp_path, key, remote=remote) is True

    def immutable(*_a: object, **_k: object) -> None:
        raise RemoteError("Already exists / immutable")

    monkeypatch.setattr("fplt.data.remote.push", immutable)
    assert push_alarm_markers(tmp_path, key, remote=remote) is True

    def other(*_a: object, **_k: object) -> None:
        raise ValueError("bad prefix")

    monkeypatch.setattr("fplt.data.remote.push", other)
    assert push_alarm_markers(tmp_path, key, remote=remote) is False
