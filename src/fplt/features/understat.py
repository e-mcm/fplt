"""Per-90 xG and xA, and each player's share of their team's totals."""

from __future__ import annotations

import polars as pl

from fplt.data.pit import PITView
from fplt.features.registry import feature
from fplt.schemas.understat import UnderstatShareSchema


@feature(
    name="understat_share",
    version=1,
    grain="player_gw",
    inputs=["understat_player_matches"],
    schema=UnderstatShareSchema,
)
def understat_share(view: PITView) -> pl.DataFrame:
    """Rates from Understat matches visible at ``view``.

    A player's share is their xG (or xA) divided by the sum of their team's
    xG in the matches they appeared in. Per-90 divides by minutes. Unmapped
    Understat ids stay out of the result but still count toward the team
    total. Zero team xG or zero minutes gives a zero rate.

    Args:
        view: Point-in-time view. Only ``understat_player_matches`` is read.

    Returns:
        A frame matching ``UnderstatShareSchema``.
    """
    matches = view.table("understat_player_matches").collect()
    season = view.as_of.season
    gameweek = view.as_of.gw
    if matches.is_empty():
        return _empty(season, gameweek)
    totals = matches.group_by("match_id", "team_key").agg(
        pl.col("xg").sum().alias("team_xg"),
        pl.col("xa").sum().alias("team_xa"),
    )
    mapped = matches.filter(pl.col("player_key").is_not_null()).join(
        totals, on=["match_id", "team_key"], how="left"
    )
    if mapped.is_empty():
        return _empty(season, gameweek)
    summed = mapped.group_by("player_key").agg(
        pl.col("xg").sum().alias("xg"),
        pl.col("xa").sum().alias("xa"),
        pl.col("minutes").sum().alias("minutes"),
        pl.col("team_xg").sum().alias("team_xg"),
        pl.col("team_xa").sum().alias("team_xa"),
    )
    return summed.select(
        pl.lit(season).alias("season"),
        pl.lit(gameweek).alias("gw"),
        "player_key",
        _per_90("xg").alias("xg_per_90"),
        _per_90("xa").alias("xa_per_90"),
        _share("xg", "team_xg").alias("xg_share"),
        _share("xa", "team_xa").alias("xa_share"),
    )


def _per_90(column: str) -> pl.Expr:
    return (
        pl.when(pl.col("minutes") > 0)
        .then(pl.col(column) * 90 / pl.col("minutes"))
        .otherwise(0.0)
    )


def _share(column: str, team: str) -> pl.Expr:
    return pl.when(pl.col(team) > 0).then(pl.col(column) / pl.col(team)).otherwise(0.0)


def _empty(season: str, gameweek: int) -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "season": pl.Utf8(),
            "gw": pl.Int64(),
            "player_key": pl.Int64(),
            "xg_per_90": pl.Float64(),
            "xa_per_90": pl.Float64(),
            "xg_share": pl.Float64(),
            "xa_share": pl.Float64(),
        }
    ).with_columns(pl.lit(season).alias("season"), pl.lit(gameweek).alias("gw"))
