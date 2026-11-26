"""ops-04 freeze window: inside / outside / boundary / DGW / blank GW."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl
import pytest

from fplt.automation.freeze import (
    FREEZE_LEAD,
    FreezeWindow,
    active_freeze,
    format_freeze_block,
    freeze_windows,
    in_freeze_window,
)

_SEASON = "2025-26"
_DEADLINE = datetime(2025, 11, 29, 11, 0, tzinfo=UTC)
_KICKOFF = datetime(2025, 11, 29, 12, 30, tzinfo=UTC)


def _gameweeks(*rows: tuple[str, int, datetime]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [r[0] for r in rows],
            "gw": [r[1] for r in rows],
            "deadline_time": [r[2] for r in rows],
        }
    )


def _fixtures(*rows: tuple[str, int, datetime]) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [r[0] for r in rows],
            "gw": [r[1] for r in rows],
            "kickoff_time": [r[2] for r in rows],
        }
    )


def test_freeze_windows_uses_earliest_kickoff() -> None:
    later = _KICKOFF + timedelta(hours=5)
    windows = freeze_windows(
        _gameweeks((_SEASON, 13, _DEADLINE)),
        _fixtures((_SEASON, 13, later), (_SEASON, 13, _KICKOFF)),
    )
    assert len(windows) == 1
    assert windows[0].starts_at == _DEADLINE - FREEZE_LEAD
    assert windows[0].ends_at == _KICKOFF
    assert windows[0].deadline_utc == _DEADLINE


def test_outside_before_and_after_freeze() -> None:
    windows = freeze_windows(
        _gameweeks((_SEASON, 13, _DEADLINE)),
        _fixtures((_SEASON, 13, _KICKOFF)),
    )
    before = _DEADLINE - FREEZE_LEAD - timedelta(seconds=1)
    after = _KICKOFF
    assert active_freeze(before, windows) is None
    assert active_freeze(after, windows) is None


def test_inside_freeze_and_boundaries() -> None:
    windows = freeze_windows(
        _gameweeks((_SEASON, 13, _DEADLINE)),
        _fixtures((_SEASON, 13, _KICKOFF)),
    )
    start = _DEADLINE - FREEZE_LEAD
    mid = _DEADLINE - timedelta(hours=1)
    just_before_end = _KICKOFF - timedelta(seconds=1)
    assert active_freeze(start, windows) is not None
    assert active_freeze(mid, windows) is not None
    assert active_freeze(just_before_end, windows) is not None
    # Open at the right: at kickoff the freeze has ended.
    assert active_freeze(_KICKOFF, windows) is None


def test_double_gameweek_separate_windows() -> None:
    d1 = datetime(2025, 12, 6, 11, 0, tzinfo=UTC)
    k1 = datetime(2025, 12, 6, 12, 30, tzinfo=UTC)
    d2 = datetime(2025, 12, 7, 14, 0, tzinfo=UTC)
    k2 = datetime(2025, 12, 7, 15, 0, tzinfo=UTC)
    windows = freeze_windows(
        _gameweeks((_SEASON, 15, d1), (_SEASON, 16, d2)),
        _fixtures((_SEASON, 15, k1), (_SEASON, 16, k2)),
    )
    assert len(windows) == 2
    # Between the two kickoffs / deadlines: inside GW16 freeze only.
    between = d2 - timedelta(hours=2)
    active = active_freeze(between, windows)
    assert active is not None
    assert active.gw == 16
    # After both kickoffs: outside.
    assert active_freeze(k2, windows) is None


def test_blank_gameweek_ends_at_deadline() -> None:
    windows = freeze_windows(
        _gameweeks((_SEASON, 20, _DEADLINE)),
        _fixtures(),  # no fixtures → blank
    )
    assert len(windows) == 1
    assert windows[0].ends_at == _DEADLINE
    assert active_freeze(_DEADLINE - timedelta(hours=1), windows) is not None
    assert active_freeze(_DEADLINE, windows) is None


def test_empty_gameweeks_means_never_frozen() -> None:
    assert freeze_windows(_gameweeks(), _fixtures()) == []
    assert (
        in_freeze_window(
            datetime(2025, 1, 1, tzinfo=UTC),
            _gameweeks(),
            _fixtures(),
        )
        is None
    )


def test_naive_datetimes_treated_as_utc() -> None:
    naive_deadline = datetime(2025, 11, 29, 11, 0)
    naive_kickoff = datetime(2025, 11, 29, 12, 30)
    windows = freeze_windows(
        _gameweeks((_SEASON, 13, naive_deadline)),
        _fixtures((_SEASON, 13, naive_kickoff)),
    )
    assert windows[0].deadline_utc.tzinfo == UTC
    assert active_freeze(naive_deadline - timedelta(hours=1), windows) is not None


def test_format_freeze_block_names_gw() -> None:
    window = FreezeWindow(
        season=_SEASON,
        gw=13,
        deadline_utc=_DEADLINE,
        starts_at=_DEADLINE - FREEZE_LEAD,
        ends_at=_KICKOFF,
    )
    text = format_freeze_block(window, now=_DEADLINE - timedelta(hours=1))
    assert "GW13" in text
    assert "refusing release" in text


@pytest.mark.parametrize(
    ("offset", "expected"),
    [
        (timedelta(hours=-24, seconds=-1), False),
        (timedelta(hours=-24), True),
        (timedelta(0), True),
        (timedelta(hours=1, minutes=29, seconds=59), True),
        (timedelta(hours=1, minutes=30), False),
    ],
)
def test_freeze_offsets_from_deadline(offset: timedelta, expected: bool) -> None:
    """Table-driven: offsets relative to deadline; kickoff is +90m."""
    windows = freeze_windows(
        _gameweeks((_SEASON, 13, _DEADLINE)),
        _fixtures((_SEASON, 13, _KICKOFF)),
    )
    moment = _DEADLINE + offset
    assert (active_freeze(moment, windows) is not None) is expected
