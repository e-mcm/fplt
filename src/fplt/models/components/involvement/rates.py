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


def team_share(rate: str, total: str) -> pl.Expr:
    """Normalise ``rate`` within the team, or split evenly when the total is 0."""
    even = 1.0 / pl.col("squad")
    return pl.when(pl.col(total) > 0).then(pl.col(rate) / pl.col(total)).otherwise(even)


def validated_involvement(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate an involvement frame against ``InvolvementSchema``."""
    checked = InvolvementSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked
