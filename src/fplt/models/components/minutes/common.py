"""Shared helpers for empirical minutes components."""

from __future__ import annotations

import numpy as np

from fplt.data.pit import PITView
from fplt.models.distributions import MINUTE_BINS

UNAVAILABLE = frozenset({"i", "s", "u", "n"})
WIDTH = len(MINUTE_BINS)
START_FALLBACK = np.zeros(WIDTH, dtype=np.float64)
START_FALLBACK[int(np.flatnonzero(MINUTE_BINS == 90)[0])] = 1.0
SUB_FALLBACK = np.zeros(WIDTH, dtype=np.float64)
SUB_FALLBACK[int(np.flatnonzero(MINUTE_BINS == 15)[0])] = 1.0


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


def availability(view: PITView) -> dict[int, tuple[str | None, float | None]]:
    """Latest status and chance of playing per player."""
    status = (
        view.latest("player_status", ["season", "player_key"])
        .collect()
        .sort("available_at")
        .unique(subset=["player_key"], keep="last")
    )
    found: dict[int, tuple[str | None, float | None]] = {}
    for row in status.select("player_key", "status", "chance_of_playing").iter_rows(
        named=True
    ):
        chance = row["chance_of_playing"]
        found[int(row["player_key"])] = (
            None if row["status"] is None else str(row["status"]),
            None if chance is None else float(chance),
        )
    return found


def chance(
    status: str | None, chance_of_playing: float | None, doubtful: float
) -> float:
    """Availability multiplier in ``[0, 1]``."""
    if chance_of_playing is not None:
        return chance_of_playing / 100.0
    if status in UNAVAILABLE:
        return 0.0
    if status == "d":
        return doubtful
    return 1.0


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
