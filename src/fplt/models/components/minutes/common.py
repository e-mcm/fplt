"""Shared helpers for empirical minutes components."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import numpy as np

from fplt.data.pit import PITView
from fplt.models.distributions import MINUTE_BINS

UNAVAILABLE = frozenset({"i", "s", "u", "n"})
WIDTH = len(MINUTE_BINS)
START_FALLBACK = np.zeros(WIDTH, dtype=np.float64)
START_FALLBACK[int(np.flatnonzero(MINUTE_BINS == 90)[0])] = 1.0
SUB_FALLBACK = np.zeros(WIDTH, dtype=np.float64)
SUB_FALLBACK[int(np.flatnonzero(MINUTE_BINS == 15)[0])] = 1.0


@dataclass(frozen=True)
class PlayerAvailability:
    """Latest FPL availability tags for one player."""

    status: str | None
    chance_next: float | None
    chance_this: float | None
    news: str
    news_added: datetime | None


def snap(minutes: np.ndarray) -> np.ndarray:
    """Bin index for each minute count.

    Zero stays in the zero bin. Any positive count snaps to the nearest of
    15, 30, 45, 60, 75 and 90, so a one-minute appearance is not stored as
    zero.
    """
    values = np.asarray(minutes, dtype=np.int64)
    positive = MINUTE_BINS[1:]
    nearest = np.abs(positive - values[:, None]).argmin(axis=1) + 1
    return np.where(values <= 0, 0, nearest).astype(np.int64)


def counts(minutes: np.ndarray) -> np.ndarray:
    """Histogram of snapped minute counts."""
    bins = np.zeros(WIDTH, dtype=np.float64)
    if len(minutes) == 0:
        return bins
    np.add.at(bins, snap(minutes), 1.0)
    return bins


def positions(view: PITView) -> dict[int, str]:
    """Latest position string per player on ``view``."""
    players = (
        view.latest("players", ["season", "player_key"])
        .collect()
        .sort("available_at")
        .unique(subset=["player_key"], keep="last")
    )
    return {
        int(player): str(position)
        for player, position in zip(
            players["player_key"].to_list(),
            players["position"].to_list(),
            strict=True,
        )
    }


def availability(view: PITView) -> dict[int, PlayerAvailability]:
    """Latest status, chance, and news tags per player."""
    status = (
        view.latest("player_status", ["season", "player_key"])
        .collect()
        .sort("available_at")
        .unique(subset=["player_key"], keep="last")
    )
    columns = [
        "player_key",
        "status",
        "chance_of_playing",
        "chance_of_playing_this_round",
        "news",
        "news_added",
    ]
    present = [name for name in columns if name in status.columns]
    found: dict[int, PlayerAvailability] = {}
    for row in status.select(present).iter_rows(named=True):
        chance_next = row.get("chance_of_playing")
        chance_this = row.get("chance_of_playing_this_round")
        news_added = row.get("news_added")
        found[int(row["player_key"])] = PlayerAvailability(
            status=None if row.get("status") is None else str(row["status"]),
            chance_next=None if chance_next is None else float(chance_next),
            chance_this=None if chance_this is None else float(chance_this),
            news=str(row.get("news") or ""),
            news_added=news_added if isinstance(news_added, datetime) else None,
        )
    return found


def chance(
    status: str | None,
    chance_of_playing: float | None,
    doubtful: float,
    *,
    chance_this: float | None = None,
    news: str = "",
    news_added: datetime | None = None,
) -> float:
    """Availability multiplier in ``[0, 1]``.

    Prefers ``chance_this`` when set, else ``chance_of_playing`` (next-round).
    A null chance is not treated as 100% when status is doubtful/unavailable or
    when FPL has stamped non-empty news / ``news_added``.
    """
    playing = chance_this if chance_this is not None else chance_of_playing
    if playing is not None:
        return playing / 100.0
    if status in UNAVAILABLE:
        return 0.0
    if status == "d":
        return doubtful
    if (news and news.strip()) or news_added is not None:
        # Null chance with news is a known FPL between-GW trap — do not
        # silently treat as fully available.
        return doubtful
    return 1.0


def gate_for(row: PlayerAvailability, doubtful: float) -> float:
    """Availability multiplier from a ``PlayerAvailability`` row."""
    return chance(
        row.status,
        row.chance_next,
        doubtful,
        chance_this=row.chance_this,
        news=row.news,
        news_added=row.news_added,
    )


def normalise(counts_arr: np.ndarray | None, fallback: np.ndarray) -> np.ndarray:
    """Turn counts into a PMF, or return ``fallback`` when empty."""
    if counts_arr is None or float(counts_arr.sum()) <= 0:
        return fallback
    return counts_arr / counts_arr.sum()


def shrunk(
    observed: np.ndarray | None,
    prior: np.ndarray | None,
    fallback: np.ndarray,
    shrinkage: float,
) -> np.ndarray:
    """Mix observed counts with a prior using ``shrinkage`` pseudo-counts."""
    base = prior if prior is not None else fallback
    raw = observed if observed is not None else np.zeros(WIDTH)
    mixed = raw + shrinkage * base
    total = float(mixed.sum())
    if total <= 0:
        return base
    return mixed / total
