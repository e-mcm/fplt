"""Near-deadline predicted-XI override for minutes P(start).

Reads staged ``predicted_lineups`` through a ``PITView``. Far from kickoff
(beyond ``window_hours``) the empirical rates are left alone. Inside the
window, ``strength`` mixes P(start) toward role targets; soft-squad and
hard-11 still run afterward in the usual order.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from fplt.data.pit import PITView
from fplt.data.predicted_lineups import ROLE_BENCH, ROLE_OUT, ROLE_START

# Role targets for P(start) when the source does not publish confidence.
_START_TARGET = 0.92
_BENCH_TARGET = 0.12
_OUT_TARGET = 0.02
_UNLISTED_TARGET = 0.05


def apply_predicted_xi(
    view: PITView,
    context: pl.DataFrame,
    p_start: np.ndarray,
    p_sub: np.ndarray,
    *,
    window_hours: float,
    strength: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Mix start/sub rates toward predicted lineup roles when in-window.

    Args:
        view: Deadline PIT view (lineups + fixtures).
        context: Predict context with ``player_key``, ``fixture_id``,
            ``team_key`` aligned with ``p_start`` / ``p_sub``.
        p_start: Empirical P(start) after availability gating.
        p_sub: Empirical P(sub).
        window_hours: Apply only when kickoff − cutoff ≤ this many hours.
        strength: Mix weight in ``[0, 1]`` toward the role target.

    Returns:
        Adjusted ``(p_start, p_sub)`` arrays (copies when anything changes).
    """
    if strength <= 0 or window_hours < 0 or context.is_empty():
        return p_start, p_sub
    strength = float(np.clip(strength, 0.0, 1.0))
    lineups = _latest_lineups(view)
    if lineups.is_empty():
        return p_start, p_sub
    kickoffs = _fixture_kickoffs(view)
    if kickoffs.is_empty():
        return p_start, p_sub
    cutoff = view.cutoff
    framed = (
        context.with_row_index("_idx")
        .join(kickoffs, on="fixture_id", how="left")
        .with_columns(
            ((pl.col("kickoff_time") - pl.lit(cutoff)).dt.total_hours()).alias(
                "hours_to_ko"
            )
        )
    )
    in_window = framed.filter(
        pl.col("hours_to_ko").is_not_null() & (pl.col("hours_to_ko") <= window_hours)
    )
    if in_window.is_empty():
        return p_start, p_sub

    joined = in_window.join(
        lineups.select("player_key", "fixture_id", "team_key", "role", "confidence"),
        on=["player_key", "fixture_id", "team_key"],
        how="left",
    )
    # Teams with a full predicted XI: demote unlisted teammates lightly.
    full_xi = (
        lineups.filter(pl.col("role") == ROLE_START)
        .group_by(["fixture_id", "team_key"])
        .len()
        .filter(pl.col("len") >= 11)
        .select("fixture_id", "team_key", pl.lit(True).alias("has_full_xi"))
    )
    joined = joined.join(full_xi, on=["fixture_id", "team_key"], how="left")

    start_out = p_start.copy()
    sub_out = p_sub.copy()
    for row in joined.iter_rows(named=True):
        index = int(row["_idx"])
        role = row.get("role")
        confidence = row.get("confidence")
        has_full = bool(row.get("has_full_xi"))
        if role is None and not has_full:
            continue
        if role == ROLE_START:
            target = (
                float(confidence)
                if isinstance(confidence, int | float)
                else _START_TARGET
            )
            target_sub = max(0.0, 0.05 * (1.0 - target))
        elif role == ROLE_BENCH:
            target = (
                float(confidence) * 0.15
                if isinstance(confidence, int | float)
                else _BENCH_TARGET
            )
            target_sub = 0.35
        elif role == ROLE_OUT:
            target = _OUT_TARGET
            target_sub = 0.02
        else:
            # Unlisted on a team with a full predicted XI.
            target = _UNLISTED_TARGET
            target_sub = 0.1
        target = float(np.clip(target, 0.0, 1.0))
        start_out[index] = (1.0 - strength) * start_out[index] + strength * target
        sub_out[index] = (1.0 - strength) * sub_out[index] + strength * target_sub
    return start_out, sub_out


def _latest_lineups(view: PITView) -> pl.DataFrame:
    """Latest visible predicted lineup row per player-fixture-team."""
    frame = view.latest(
        "predicted_lineups", ["season", "team_key", "fixture_id", "player_key"]
    ).collect()
    if frame.is_empty():
        return frame
    return frame.filter(
        pl.col("player_key").is_not_null() & pl.col("fixture_id").is_not_null()
    )


def _fixture_kickoffs(view: PITView) -> pl.DataFrame:
    """Latest kickoff per fixture_id on ``view``."""
    return (
        view.latest("fixtures", ["season", "fixture_id"])
        .select("fixture_id", "kickoff_time")
        .collect()
    )
