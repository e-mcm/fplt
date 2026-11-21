"""Shared rate helpers for involvement components."""

from __future__ import annotations

import polars as pl

from fplt.data.pit import PITView
from fplt.schemas.components import InvolvementSchema

POSITIONS = ("GK", "DEF", "MID", "FWD")


def appearance_history(view: PITView) -> pl.DataFrame:
    """Visible player-fixture stats joined to position and team."""
    stats = view.latest(
        "player_fixture_stats", ["season", "player_key", "fixture_id"]
    ).collect()
    if stats.is_empty():
        return stats
    roster = (
        view.latest("players", ["season", "player_key"])
        .select("season", "player_key", "position", "team_key")
        .collect()
    )
    return stats.join(roster, on=["season", "player_key"], how="left").sort(
        ["player_key", "season", "gw", "fixture_id"]
    )


def current_teams(view: PITView) -> pl.DataFrame:
    """Latest ``team_key`` per player at ``view``."""
    roster = (
        view.latest("players", ["season", "player_key"])
        .select("player_key", "team_key", "available_at")
        .collect()
    )
    if roster.is_empty():
        return pl.DataFrame(schema={"player_key": pl.Int64(), "team_key": pl.Int64()})
    return (
        roster.sort("available_at")
        .unique(subset=["player_key"], keep="last")
        .select("player_key", "team_key")
    )


def latest_rows(history: pl.DataFrame, window: int) -> pl.DataFrame:
    """Keep the last ``window`` rows per ``player_key`` in row order."""
    ranked = history.with_columns(
        pl.col("player_key").cum_count().over("player_key").cast(pl.Int64).alias("_i"),
        pl.len().over("player_key").cast(pl.Int64).alias("_n"),
    )
    return ranked.filter(pl.col("_i") > pl.col("_n") - window).drop("_i", "_n")


def position_goal_assist_rates(
    history: pl.DataFrame,
) -> dict[str, tuple[float, float]]:
    """Per-90 goal and assist rates by position from appearances with minutes."""
    played = history.filter(
        (pl.col("minutes") > 0) & pl.col("position").is_in(list(POSITIONS))
    )
    if played.is_empty():
        return {}
    totals = played.group_by("position").agg(
        pl.col("goals_scored").sum().alias("goals"),
        pl.col("assists").sum().alias("assists"),
        (pl.col("minutes") / 90.0).sum().alias("nineties"),
    )
    rates: dict[str, tuple[float, float]] = {}
    for row in totals.iter_rows(named=True):
        nineties = float(row["nineties"])
        rates[str(row["position"])] = (
            float(row["goals"]) / nineties,
            float(row["assists"]) / nineties,
        )
    return rates


def shrunk_rate(
    events: float, nineties: float, prior: float, shrinkage: float
) -> float:
    """Empirical Bayes shrink of ``events / nineties`` toward ``prior``."""
    denom = nineties + shrinkage
    if denom <= 0:
        return prior
    return (events + shrinkage * prior) / denom


def penalty_takers(
    view: PITView,
    history: pl.DataFrame,
    *,
    use_penalties_order: bool = True,
) -> set[int]:
    """Players who should get the penalty-taker goal bonus.

    When ``use_penalties_order`` is true (default), prefer the latest PIT
    ``player_status.penalties_order == 1``. When that order is null (vaastav
    and other snapshot gaps), fall back to anyone with a visible
    ``penalties_missed > 0``. A non-null order other than 1 gates off the
    miss heuristic for that player.

    When ``use_penalties_order`` is false, ignore order and use the miss
    heuristic only (A/B control / pre-W2 behaviour).
    """
    if not use_penalties_order:
        return _miss_takers(history)
    status = (
        view.latest("player_status", ["season", "player_key"])
        .collect()
        .sort("available_at")
        .unique(subset=["player_key"], keep="last")
    )
    order_by_player: dict[int, int | None] = {}
    if not status.is_empty() and "penalties_order" in status.columns:
        for row in status.select("player_key", "penalties_order").iter_rows(named=True):
            order = row["penalties_order"]
            order_by_player[int(row["player_key"])] = (
                None if order is None else int(order)
            )
    takers = {player for player, order in order_by_player.items() if order == 1}
    for player in _miss_takers(history):
        if order_by_player.get(player) is None:
            takers.add(player)
    return takers


def _miss_takers(history: pl.DataFrame) -> set[int]:
    """Players with a visible ``penalties_missed > 0``."""
    if history.is_empty() or "penalties_missed" not in history.columns:
        return set()
    return {
        int(player)
        for player in history.filter(pl.col("penalties_missed") > 0)["player_key"]
    }


def team_share(rate: str, total: str) -> pl.Expr:
    """Normalise ``rate`` within the team, or split evenly when the total is 0."""
    even = 1.0 / pl.col("squad")
    return pl.when(pl.col(total) > 0).then(pl.col(rate) / pl.col(total)).otherwise(even)


def validated_involvement(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate an involvement frame against ``InvolvementSchema``."""
    checked = InvolvementSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked
