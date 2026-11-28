"""Biggest post-GW prediction misses with reason tags (#394 / Spec PR6).

Top 15 by ``|actual − xpts|``. ``predictable_at_T90`` uses only
``player_status`` / ``predicted_lineups`` with ``available_at`` ≤ T−90m.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import polars as pl

from fplt.data.predicted_lineups import ROLE_BENCH, ROLE_OUT, ROLE_START
from fplt.models.components.minutes.common import UNAVAILABLE
from fplt.schemas.post_gw import PostGwBiggestMissSchema
from fplt.types import GWKey
from fplt.utils.frames import require_dataframe

__all__ = [
    "MISS_REASONS",
    "TOP_N",
    "compute_biggest_misses",
    "empty_biggest_misses",
    "t90_cutoff",
    "tag_miss_reason",
]

TOP_N = 15
MISS_REASONS = (
    "benched",
    "subbed_early",
    "red_card",
    "injury",
    "penalty",
    "dgw",
    "blank",
    "other",
)
_SCHEMA = {
    "model_id": pl.Utf8(),
    "season": pl.Utf8(),
    "gw": pl.Int64(),
    "player_key": pl.Int64(),
    "fixture_id": pl.Int64(),
    "xpts": pl.Float64(),
    "actual_pts": pl.Int64(),
    "abs_error": pl.Float64(),
    "signed_error": pl.Float64(),
    "reason": pl.Utf8(),
    "predictable_at_T90": pl.Boolean(),
    "rank": pl.Int64(),
}


def empty_biggest_misses() -> pl.DataFrame:
    """Return an empty ``PostGwBiggestMissSchema`` frame."""
    return require_dataframe(
        PostGwBiggestMissSchema.validate(pl.DataFrame(schema=_SCHEMA))
    )


def t90_cutoff(deadline: datetime) -> datetime:
    """Return the T−90m PIT cutoff for ``deadline`` (UTC)."""
    if deadline.tzinfo is None:
        deadline = deadline.replace(tzinfo=UTC)
    return deadline.astimezone(UTC) - timedelta(minutes=90)


def tag_miss_reason(
    *,
    minutes: int,
    starts: int,
    red_cards: int,
    penalties_missed: int,
    status: str | None,
    injury_out: bool,
    n_player_fixtures: int,
    n_team_fixtures: int,
) -> str:
    """Return one reason tag (red_card → … → other)."""
    if red_cards > 0:
        return "red_card"
    if penalties_missed > 0:
        return "penalty"
    if n_team_fixtures == 0:
        return "blank"
    if minutes == 0 and (injury_out or status == "i"):
        return "injury"
    if starts == 0 and minutes == 0:
        return "benched"
    if starts > 0 and minutes < 60:
        return "subbed_early"
    if n_player_fixtures > 1:
        return "dgw"
    return "other"


def compute_biggest_misses(
    *,
    model_id: str,
    key: GWKey,
    predictions: pl.DataFrame,
    outcomes: pl.DataFrame,
    stats: pl.DataFrame,
    fixtures: pl.DataFrame,
    player_status: pl.DataFrame,
    predicted_lineups: pl.DataFrame,
    deadline: datetime,
    player_injuries: pl.DataFrame | None = None,
    top_n: int = TOP_N,
) -> pl.DataFrame:
    """Build the top-``top_n`` |actual − xpts| table for one model × GW.

    Returns:
        Validated ``PostGwBiggestMissSchema`` frame (possibly empty).
    """
    if predictions.is_empty() or outcomes.is_empty():
        return empty_biggest_misses()
    season_col = "target_season" if "target_season" in predictions.columns else "season"
    gw_col = "target_gw" if "target_gw" in predictions.columns else "gw"
    pred = predictions.filter(
        (pl.col("model_id") == model_id)
        & (pl.col(season_col) == key.season)
        & (pl.col(gw_col) == key.gw)
    ).select("player_key", "fixture_id", "xpts")
    actual = outcomes.filter(
        (pl.col("season") == key.season) & (pl.col("gw") == key.gw)
    ).select("player_key", "fixture_id", "team_key", "pts", "minutes")
    joined = pred.join(actual, on=["player_key", "fixture_id"], how="inner")
    if joined.is_empty():
        return empty_biggest_misses()

    fx = fixtures.filter((pl.col("season") == key.season) & (pl.col("gw") == key.gw))
    team_n = _team_fixture_counts(fx)
    player_n = joined.group_by("player_key").agg(pl.len().alias("n_player_fixtures"))
    joined = (
        joined.join(team_n, on="team_key", how="left")
        .join(player_n, on="player_key", how="left")
        .join(_latest_stats(stats, key), on=["player_key", "fixture_id"], how="left")
        .with_columns(
            pl.col("n_team_fixtures").fill_null(0).cast(pl.Int64),
            pl.col("starts").fill_null(0).cast(pl.Int64),
            pl.col("red_cards").fill_null(0).cast(pl.Int64),
            pl.col("penalties_missed").fill_null(0).cast(pl.Int64),
            pl.col("minutes").fill_null(0).cast(pl.Int64),
        )
    )
    cutoff = t90_cutoff(deadline)
    joined = (
        joined.join(
            _status_at_t90(player_status, key.season, cutoff),
            on="player_key",
            how="left",
        )
        .join(
            _lineup_at_t90(predicted_lineups, key.season, cutoff),
            on=["player_key", "fixture_id"],
            how="left",
        )
        .join(
            _injury_out_at_t90(player_injuries, key.season, cutoff),
            on="player_key",
            how="left",
        )
        .with_columns(pl.col("injury_out").fill_null(False))
    )

    rows: list[dict[str, object]] = []
    for row in joined.iter_rows(named=True):
        minutes = int(row["minutes"] or 0)
        starts = int(row["starts"] or 0)
        status = None if row.get("status") is None else str(row["status"])
        actual_pts = int(row["pts"])
        xpts = float(row["xpts"])
        rows.append(
            {
                "model_id": model_id,
                "season": key.season,
                "gw": key.gw,
                "player_key": int(row["player_key"]),
                "fixture_id": int(row["fixture_id"]),
                "xpts": xpts,
                "actual_pts": actual_pts,
                "abs_error": abs(actual_pts - xpts),
                "signed_error": float(actual_pts) - xpts,
                "reason": tag_miss_reason(
                    minutes=minutes,
                    starts=starts,
                    red_cards=int(row["red_cards"] or 0),
                    penalties_missed=int(row["penalties_missed"] or 0),
                    status=status,
                    injury_out=bool(row["injury_out"]),
                    n_player_fixtures=int(row["n_player_fixtures"] or 1),
                    n_team_fixtures=int(row["n_team_fixtures"] or 0),
                ),
                "predictable_at_T90": _predictable(
                    actual_pts=actual_pts,
                    xpts=xpts,
                    status=status,
                    chance=(
                        None if row.get("chance") is None else float(row["chance"])
                    ),
                    role=None if row.get("role") is None else str(row["role"]),
                ),
            }
        )
    frame = (
        pl.DataFrame(rows)
        .sort(
            ["abs_error", "player_key", "fixture_id"],
            descending=[True, False, False],
        )
        .head(top_n)
        .with_columns(pl.int_range(1, pl.len() + 1).alias("rank"))
    )
    return require_dataframe(PostGwBiggestMissSchema.validate(frame))


def _team_fixture_counts(fixtures: pl.DataFrame) -> pl.DataFrame:
    if fixtures.is_empty():
        return pl.DataFrame(
            schema={"team_key": pl.Int64(), "n_team_fixtures": pl.Int64()}
        )
    return (
        pl.concat(
            [
                fixtures.select(pl.col("team_h_key").alias("team_key")),
                fixtures.select(pl.col("team_a_key").alias("team_key")),
            ]
        )
        .group_by("team_key")
        .agg(pl.len().alias("n_team_fixtures"))
    )


def _latest_stats(stats: pl.DataFrame, key: GWKey) -> pl.DataFrame:
    cols = ["player_key", "fixture_id", "starts", "red_cards", "penalties_missed"]
    empty = pl.DataFrame(schema={c: pl.Int64() for c in cols})
    if stats.is_empty():
        return empty
    part = stats.filter(pl.col("season") == key.season)
    if "gw" in part.columns:
        part = part.filter(pl.col("gw") == key.gw)
    if part.is_empty():
        return empty
    if "available_at" in part.columns:
        part = part.sort("available_at")
    return part.unique(subset=["player_key", "fixture_id"], keep="last").select(cols)


def _status_at_t90(status: pl.DataFrame, season: str, cutoff: datetime) -> pl.DataFrame:
    empty = pl.DataFrame(
        schema={"player_key": pl.Int64(), "status": pl.Utf8(), "chance": pl.Float64()}
    )
    if status.is_empty():
        return empty
    part = status.filter(
        (pl.col("season") == season) & (pl.col("available_at") <= cutoff)
    )
    if part.is_empty():
        return empty
    chance = (
        pl.coalesce(pl.col("chance_of_playing_this_round"), pl.col("chance_of_playing"))
        if "chance_of_playing_this_round" in part.columns
        else pl.col("chance_of_playing")
    )
    return (
        part.sort("available_at")
        .unique(subset=["player_key"], keep="last")
        .select("player_key", "status", chance.alias("chance"))
    )


def _lineup_at_t90(
    lineups: pl.DataFrame, season: str, cutoff: datetime
) -> pl.DataFrame:
    empty = pl.DataFrame(
        schema={
            "player_key": pl.Int64(),
            "fixture_id": pl.Int64(),
            "role": pl.Utf8(),
        }
    )
    if lineups.is_empty():
        return empty
    part = lineups.filter(
        (pl.col("season") == season)
        & (pl.col("available_at") <= cutoff)
        & pl.col("player_key").is_not_null()
        & pl.col("fixture_id").is_not_null()
    )
    if part.is_empty():
        return empty
    return (
        part.sort("available_at")
        .unique(subset=["player_key", "fixture_id"], keep="last")
        .select(
            pl.col("player_key").cast(pl.Int64),
            pl.col("fixture_id").cast(pl.Int64),
            "role",
        )
    )


def _injury_out_at_t90(
    injuries: pl.DataFrame | None, season: str, cutoff: datetime
) -> pl.DataFrame:
    empty = pl.DataFrame(schema={"player_key": pl.Int64(), "injury_out": pl.Boolean()})
    if injuries is None or injuries.is_empty():
        return empty
    part = injuries.filter(
        (pl.col("season") == season)
        & (pl.col("available_at") <= cutoff)
        & pl.col("player_key").is_not_null()
        & (pl.col("category") == "out")
    )
    if part.is_empty():
        return empty
    return (
        part.select(pl.col("player_key").cast(pl.Int64))
        .unique()
        .with_columns(pl.lit(True).alias("injury_out"))
    )


def _predictable(
    *,
    actual_pts: int,
    xpts: float,
    status: str | None,
    chance: float | None,
    role: str | None,
) -> bool:
    """True when T−90m availability / lineup pointed opposite the miss."""
    if status is None and chance is None and role is None:
        return False
    over = float(actual_pts) < xpts
    under = float(actual_pts) > xpts
    neg = (
        (status in UNAVAILABLE)
        or (chance is not None and chance < 50.0)
        or (role in {ROLE_BENCH, ROLE_OUT})
    )
    pos = (role == ROLE_START) or (status == "a" and (chance is None or chance >= 75.0))
    return bool((over and neg) or (under and pos))
