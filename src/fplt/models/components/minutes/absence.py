"""Absence-aware rolling shares and return-ramp helpers for minutes v2.

P4: injury / suspension zeros must not pollute last-N start shares. Tag each
appearance with a PIT status-at-deadline flag, compute shares only over
non-absence rows, and expose ``games_since_return`` / ``absence_length`` for
the logistic and a short post-return ramp on P(start).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import polars as pl

from fplt.data.pit import PITView
from fplt.models.components.minutes.common import UNAVAILABLE

_ABSENCE_CAP = 10.0


def annotate_absences(view: PITView, stats: pl.DataFrame) -> pl.DataFrame:
    """Add ``is_absence`` using the latest status visible at each GW deadline.

    An appearance is an absence when FPL marked the player unavailable
    (``i`` / ``s`` / ``u`` / ``n``), or when they recorded zero minutes while
    doubtful / chance-of-playing 0. Missing status (vaastav) leaves the row
    available so rotation zeros stay in the share.

    Args:
        view: PIT view supplying ``gameweeks`` and ``player_status``.
        stats: ``player_fixture_stats`` rows (must include ``season``, ``gw``,
            ``player_key``, ``minutes``).

    Returns:
        ``stats`` plus boolean ``is_absence``.
    """
    if stats.is_empty():
        return stats.with_columns(pl.lit(False).alias("is_absence"))
    deadlines = (
        view.latest("gameweeks", ["season", "gw"])
        .select("season", "gw", "deadline_time")
        .collect()
    )
    status_lf = view.table("player_status")
    status_cols = [
        "season",
        "player_key",
        "status",
        "chance_of_playing",
        "available_at",
    ]
    # ``chance_of_playing_this_round`` is optional on older fixtures / tests.
    schema_names = status_lf.collect_schema().names()
    if "chance_of_playing_this_round" in schema_names:
        status_cols.append("chance_of_playing_this_round")
    status = (
        status_lf.select(status_cols)
        .collect()
        .sort(["season", "player_key", "available_at"])
    )
    with_deadline = stats.join(deadlines, on=["season", "gw"], how="left")
    if status.is_empty() or "deadline_time" not in with_deadline.columns:
        return stats.with_columns(pl.lit(False).alias("is_absence"))
    ordered = with_deadline.sort(["season", "player_key", "deadline_time"])
    # Pre-sorted by (season, player_key, time); skip sortedness check (by-groups).
    joined = ordered.join_asof(
        status,
        left_on="deadline_time",
        right_on="available_at",
        by=["season", "player_key"],
        strategy="backward",
        check_sortedness=False,
    )
    if "chance_of_playing_this_round" in joined.columns:
        chance = pl.coalesce(
            pl.col("chance_of_playing_this_round"),
            pl.col("chance_of_playing"),
        )
    else:
        chance = pl.col("chance_of_playing")
    zero_mins = pl.col("minutes") <= 0
    flagged = (
        pl.col("status").is_in(list(UNAVAILABLE)).fill_null(False)
        | ((pl.col("status") == "d").fill_null(False) & zero_mins)
        | ((chance == 0.0).fill_null(False) & zero_mins)
    )
    tagged = joined.with_columns(flagged.alias("is_absence"))
    # Preserve original row order / columns; drop join helpers.
    keep = [name for name in stats.columns if name in tagged.columns] + ["is_absence"]
    return tagged.select(keep)


def return_metrics(is_absence: Sequence[bool]) -> tuple[float, float]:
    """Games since return and length of the most recent absence streak.

    Chronological ``is_absence`` flags. When the latest appearance is still an
    absence, ``games_since_return`` is 0 and ``absence_length`` is the current
    streak. Otherwise ``games_since_return`` counts trailing available
    appearances and ``absence_length`` is the preceding absence streak (0 if
    the player has never been absent in view).

    Args:
        is_absence: Oldest → newest absence flags.

    Returns:
        ``(games_since_return, absence_length)``.
    """
    n = len(is_absence)
    if n == 0:
        return 0.0, 0.0
    index = n - 1
    if is_absence[index]:
        length = 0
        while index >= 0 and is_absence[index]:
            length += 1
            index -= 1
        return 0.0, float(length)
    games = 0
    while index >= 0 and not is_absence[index]:
        games += 1
        index -= 1
    length = 0
    while index >= 0 and is_absence[index]:
        length += 1
        index -= 1
    return float(games), float(length)


def scale_return_feature(value: float) -> float:
    """Cap and unit-scale a return / absence count for the logistic."""
    return min(max(float(value), 0.0), _ABSENCE_CAP) / _ABSENCE_CAP


def return_ramp_factor(
    games_since_return: float,
    absence_length: float,
    ramp_games: float,
) -> float:
    """Multiplicative P(start) dampener for the first games back from absence.

    Args:
        games_since_return: Available appearances since the last absence.
        absence_length: Length of that absence (0 → no ramp).
        ramp_games: Games to full strength; ``<= 0`` disables the ramp.

    Returns:
        Factor in ``(0, 1]``.
    """
    if ramp_games <= 0 or absence_length <= 0:
        return 1.0
    if games_since_return >= ramp_games:
        return 1.0
    return float((games_since_return + 1.0) / (ramp_games + 1.0))


def available_rolling_shares(
    group: pl.DataFrame,
    windows: Sequence[int],
) -> dict[str, float]:
    """Start-share / minutes features over non-absence appearances only.

    Args:
        group: One player's appearances with ``is_absence``, ``starts``,
            ``minutes`` (chronological).
        windows: Rolling window sizes (e.g. 3, 5, 10).

    Returns:
        Feature map including ``starts_share_*``, ``minutes_mean_*`` for each
        window present, plus ``minutes_mean_5`` / ``minutes_std_5`` and return
        metrics.
    """
    flags = group["is_absence"].to_list() if "is_absence" in group.columns else []
    gsr, alen = return_metrics([bool(flag) for flag in flags])
    available = (
        group.filter(~pl.col("is_absence")) if "is_absence" in group.columns else group
    )
    features: dict[str, float] = {
        "games_since_return": gsr,
        "absence_length": alen,
    }
    for window in windows:
        recent = available.tail(window)
        n = recent.height
        rate = float((recent["starts"] > 0).sum() / n) if n else 0.0
        features[f"starts_share_{window}"] = rate
        features[f"minutes_mean_{window}"] = (
            _series_float(recent["minutes"].mean()) if n else 0.0
        )
    recent5 = available.tail(5)
    if "minutes_mean_5" not in features:
        features["minutes_mean_5"] = (
            _series_float(recent5["minutes"].mean()) if recent5.height else 0.0
        )
    if recent5.height >= 2:
        features["minutes_std_5"] = _series_float(recent5["minutes"].std())
    else:
        features["minutes_std_5"] = 0.0
    return features


def _series_float(value: object) -> float:
    """Coerce a polars aggregation scalar to float (null → 0)."""
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, int | float):
        return float(value)
    if isinstance(value, np.generic):
        return float(value.item())  # type: ignore[arg-type]
    return 0.0


def available_rate(group: pl.DataFrame) -> tuple[float, float]:
    """Empirical start / sub rates over non-absence appearances.

    Args:
        group: Player appearances with ``is_absence``, ``starts``, ``minutes``.

    Returns:
        ``(p_start, p_sub)`` on the available subset (zeros when empty).
    """
    available = (
        group.filter(~pl.col("is_absence")) if "is_absence" in group.columns else group
    )
    n = available.height
    if n == 0:
        return 0.0, 0.0
    start_arr = available["starts"].to_numpy()
    minute_arr = available["minutes"].to_numpy()
    return (
        float(np.sum(start_arr > 0) / n),
        float(np.sum((start_arr == 0) & (minute_arr > 0)) / n),
    )
