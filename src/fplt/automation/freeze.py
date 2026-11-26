"""Pre-deadline freeze window for moving the ``prod`` tag (ops-04).

Ops workflows run from the moving ``prod`` tag. ``make release`` / ``release.yml``
may move that tag only outside the freeze: from 24h before a deadline until the
earliest kickoff of that gameweek (or the deadline itself when the GW is blank).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

import polars as pl

FREEZE_LEAD = timedelta(hours=24)


@dataclass(frozen=True)
class FreezeWindow:
    """One gameweek's no-release window.

    Attributes:
        season: Season label (for example ``2025-26``).
        gw: Gameweek number.
        deadline_utc: FPL deadline for this gameweek (UTC).
        starts_at: ``deadline_utc - FREEZE_LEAD``.
        ends_at: Earliest fixture kickoff for the GW, or ``deadline_utc`` when
            the GW has no fixtures (blank).
    """

    season: str
    gw: int
    deadline_utc: datetime
    starts_at: datetime
    ends_at: datetime


def freeze_windows(
    gameweeks: pl.DataFrame,
    fixtures: pl.DataFrame,
) -> list[FreezeWindow]:
    """Build freeze windows from staged ``gameweeks`` and ``fixtures``.

    Each distinct ``(season, gw)`` deadline gets one window. Double gameweeks
    are two rows with different ``gw`` values (and usually two deadlines). A
    blank gameweek (deadline present, zero fixtures) ends at the deadline.

    Args:
        gameweeks: Staged ``GameweeksSchema`` rows (at least ``season``, ``gw``,
            ``deadline_time``).
        fixtures: Staged ``FixturesSchema`` rows (at least ``season``, ``gw``,
            ``kickoff_time``). May be empty.

    Returns:
        Windows sorted by ``starts_at``, then ``season`` / ``gw``.
    """
    if gameweeks.is_empty():
        return []

    deadlines = (
        gameweeks.select(
            pl.col("season").cast(pl.Utf8),
            pl.col("gw").cast(pl.Int64),
            pl.col("deadline_time").alias("deadline_utc"),
        )
        .unique(subset=["season", "gw"], keep="first")
        .sort(["season", "gw"])
    )

    if fixtures.is_empty():
        earliest = pl.DataFrame(
            schema={
                "season": pl.Utf8,
                "gw": pl.Int64,
                "kickoff_utc": pl.Datetime(time_zone="UTC"),
            }
        )
    else:
        earliest = (
            fixtures.select(
                pl.col("season").cast(pl.Utf8),
                pl.col("gw").cast(pl.Int64),
                pl.col("kickoff_time").alias("kickoff_utc"),
            )
            .group_by(["season", "gw"])
            .agg(pl.col("kickoff_utc").min().alias("kickoff_utc"))
        )

    joined = deadlines.join(earliest, on=["season", "gw"], how="left")
    windows: list[FreezeWindow] = []
    for row in joined.iter_rows(named=True):
        deadline = _aware(row["deadline_utc"])
        kickoff_raw = row["kickoff_utc"]
        ends = _aware(kickoff_raw) if kickoff_raw is not None else deadline
        # Kickoff before the deadline (bad data) still ends at deadline so the
        # window never has ends_at < starts_at for a normal lead.
        if ends < deadline - FREEZE_LEAD:
            ends = deadline
        windows.append(
            FreezeWindow(
                season=str(row["season"]),
                gw=int(row["gw"]),
                deadline_utc=deadline,
                starts_at=deadline - FREEZE_LEAD,
                ends_at=ends,
            )
        )
    windows.sort(key=lambda w: (w.starts_at, w.season, w.gw))
    return windows


def active_freeze(
    now: datetime,
    windows: Sequence[FreezeWindow],
) -> FreezeWindow | None:
    """Return the freeze window that contains ``now``, if any.

    A moment is inside when ``starts_at <= now < ends_at``. At the earliest
    kickoff the freeze has ended ("ending after kickoff" = open at the right).

    Args:
        now: Clock reading (timezone-aware UTC preferred).
        windows: Windows from ``freeze_windows``.

    Returns:
        The matching ``FreezeWindow``, or ``None`` when outside every window.
        When several overlap (rare / bad data), the earliest ``starts_at`` wins.
    """
    moment = _aware(now)
    for window in windows:
        if window.starts_at <= moment < window.ends_at:
            return window
    return None


def in_freeze_window(
    now: datetime,
    gameweeks: pl.DataFrame,
    fixtures: pl.DataFrame,
) -> FreezeWindow | None:
    """Convenience: build windows then return the active one for ``now``.

    Args:
        now: Clock reading.
        gameweeks: Staged gameweeks table.
        fixtures: Staged fixtures table.

    Returns:
        Active ``FreezeWindow`` or ``None``.
    """
    return active_freeze(now, freeze_windows(gameweeks, fixtures))


def format_freeze_block(window: FreezeWindow, *, now: datetime) -> str:
    """Human message explaining why a release is refused.

    Args:
        window: Active freeze window.
        now: Clock reading used for the check.

    Returns:
        One-line operator message (no secrets).
    """
    moment = _aware(now)
    return (
        f"refusing release: inside freeze for {window.season} GW{window.gw} "
        f"(deadline {window.deadline_utc.isoformat()}, "
        f"freeze {window.starts_at.isoformat()} .. {window.ends_at.isoformat()}, "
        f"now {moment.isoformat()})"
    )


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
