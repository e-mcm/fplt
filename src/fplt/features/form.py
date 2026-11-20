"""PIT features for the direct LightGBM points model.

Column names follow ``BASE_FEATURE_COLS`` in the legacy predictor, limited to
fields the staged tables actually carry (no BPS / ICT / xG rolling series).
"""

from __future__ import annotations

import polars as pl

from fplt.data.pit import PITView
from fplt.features.registry import feature
from fplt.models.universe import player_fixtures
from fplt.schemas.features import (
    ContextFeatureSchema,
    OpponentStrengthSchema,
    RollingFormSchema,
)
from fplt.types import GWKey, Position

FEATURE_COLUMNS: list[str] = [
    "position_enc",
    "was_home",
    "value",
    "opponent_team",
    "opp_strength_overall",
    "opp_strength_attack",
    "opp_strength_defence",
    "total_points_rolling_3",
    "total_points_rolling_5",
    "total_points_rolling_10",
    "minutes_rolling_3",
    "minutes_rolling_5",
    "minutes_rolling_10",
    "minutes_std_5",
    "goals_scored_rolling_5",
    "assists_rolling_5",
    "clean_sheets_rolling_5",
    "goals_conceded_rolling_5",
    "saves_rolling_5",
    "bonus_rolling_5",
]

_ROLLING_MEAN_COLS = (
    "total_points",
    "minutes",
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "bonus",
)

_WINDOWS = (3, 5, 10)


def _position_enc() -> pl.Expr:
    mapping = {member.name: float(member.value) for member in Position}
    return (
        pl.col("position")
        .replace_strict(mapping, default=None)
        .cast(pl.Float64)
        .alias("position_enc")
    )


def _empty_keys() -> dict[str, pl.DataType]:
    return {"player_key": pl.Int64(), "fixture_id": pl.Int64()}


def _annotate_fixtures(view: PITView, rows: pl.DataFrame) -> pl.DataFrame:
    """Attach position, price, home flag, and opponent to player-fixture rows."""
    if rows.is_empty():
        return rows
    players = (
        view.latest("players", by=["season", "player_key"])
        .select("season", "player_key", "team_key", "position")
        .collect()
    )
    status = (
        view.latest("player_status", by=["season", "player_key"])
        .select("season", "player_key", "price")
        .collect()
    )
    fixtures = (
        view.latest("fixtures", by=["season", "fixture_id"])
        .select("season", "fixture_id", "team_h_key", "team_a_key")
        .collect()
    )
    return (
        rows.join(players, on=["season", "player_key"], how="left")
        .join(status, on=["season", "player_key"], how="left")
        .join(fixtures, on=["season", "fixture_id"], how="left")
        .with_columns(
            (pl.col("team_key") == pl.col("team_h_key")).alias("is_home"),
            pl.when(pl.col("team_key") == pl.col("team_h_key"))
            .then(pl.col("team_a_key"))
            .otherwise(pl.col("team_h_key"))
            .alias("opponent_key"),
        )
        .with_columns(
            _position_enc(),
            pl.col("is_home").cast(pl.Float64).alias("was_home"),
            pl.col("price").cast(pl.Float64).alias("value"),
            pl.col("opponent_key").cast(pl.Float64).alias("opponent_team"),
        )
    )


def latest_form(stats: pl.DataFrame) -> pl.DataFrame:
    """Rolling means of the last appearances visible for each player.

    Windows match the legacy LightGBM starter list. The caller's current
    fixture is never in ``stats``, so these values are safe for prediction.
    """
    schema = {
        "player_key": pl.Int64(),
        "total_points_rolling_3": pl.Float64(),
        "total_points_rolling_5": pl.Float64(),
        "total_points_rolling_10": pl.Float64(),
        "minutes_rolling_3": pl.Float64(),
        "minutes_rolling_5": pl.Float64(),
        "minutes_rolling_10": pl.Float64(),
        "minutes_std_5": pl.Float64(),
        "goals_scored_rolling_5": pl.Float64(),
        "assists_rolling_5": pl.Float64(),
        "clean_sheets_rolling_5": pl.Float64(),
        "goals_conceded_rolling_5": pl.Float64(),
        "saves_rolling_5": pl.Float64(),
        "bonus_rolling_5": pl.Float64(),
    }
    if stats.is_empty():
        return pl.DataFrame(schema=schema)
    ordered = stats.sort(["player_key", "available_at"])
    exprs: list[pl.Expr] = []
    for column in _ROLLING_MEAN_COLS:
        for window in _WINDOWS:
            if column not in {"total_points", "minutes"} and window != 5:
                continue
            exprs.append(
                pl.col(column)
                .rolling_mean(window_size=window, min_samples=1)
                .over("player_key")
                .alias(f"{column}_rolling_{window}")
            )
        if column == "minutes":
            exprs.append(
                pl.col("minutes")
                .rolling_std(window_size=5, min_samples=2)
                .over("player_key")
                .alias("minutes_std_5")
            )
    rolled = ordered.with_columns(exprs)
    return rolled.group_by("player_key").tail(1).select(list(schema))


def shifted_form(stats: pl.DataFrame) -> pl.DataFrame:
    """Add rolling means that exclude the row itself, for training."""
    if stats.is_empty():
        return stats
    ordered = stats.sort(["player_key", "available_at"])
    exprs: list[pl.Expr] = []
    for column in _ROLLING_MEAN_COLS:
        shifted = pl.col(column).shift(1)
        for window in _WINDOWS:
            if column not in {"total_points", "minutes"} and window != 5:
                continue
            exprs.append(
                shifted.rolling_mean(window_size=window, min_samples=1)
                .over("player_key")
                .alias(f"{column}_rolling_{window}")
            )
        if column == "minutes":
            exprs.append(
                shifted.rolling_std(window_size=5, min_samples=2)
                .over("player_key")
                .alias("minutes_std_5")
            )
    return ordered.with_columns(exprs)


def opponent_ratings(fixtures: pl.DataFrame) -> pl.DataFrame:
    """Mean goals for and against from finished fixtures visible to the caller."""
    empty = pl.DataFrame(
        schema={
            "team_key": pl.Int64(),
            "opp_strength_attack": pl.Float64(),
            "opp_strength_defence": pl.Float64(),
            "opp_strength_overall": pl.Float64(),
        }
    )
    finished = fixtures.filter(pl.col("team_h_score").is_not_null())
    if finished.is_empty():
        return empty
    home = finished.select(
        pl.col("team_h_key").alias("team_key"),
        pl.col("team_h_score").alias("gf"),
        pl.col("team_a_score").alias("ga"),
    )
    away = finished.select(
        pl.col("team_a_key").alias("team_key"),
        pl.col("team_a_score").alias("gf"),
        pl.col("team_h_score").alias("ga"),
    )
    return (
        pl.concat([home, away])
        .group_by("team_key")
        .agg(
            pl.col("gf").mean().alias("opp_strength_attack"),
            pl.col("ga").mean().alias("opp_strength_defence"),
        )
        .with_columns(
            (
                (pl.col("opp_strength_attack") + pl.col("opp_strength_defence")) / 2.0
            ).alias("opp_strength_overall")
        )
    )


@feature(
    name="context",
    version=1,
    grain="player_fixture",
    inputs=["players", "player_status", "fixtures"],
    schema=ContextFeatureSchema,
)
def context(view: PITView, horizon: list[GWKey]) -> pl.DataFrame:
    """Position, home flag, price, and opponent id for each horizon fixture."""
    rows = _annotate_fixtures(view, player_fixtures(view, horizon))
    columns = [
        "player_key",
        "fixture_id",
        "position_enc",
        "was_home",
        "value",
        "opponent_team",
    ]
    if rows.is_empty():
        return pl.DataFrame(
            schema={
                **_empty_keys(),
                "position_enc": pl.Float64(),
                "was_home": pl.Float64(),
                "value": pl.Float64(),
                "opponent_team": pl.Float64(),
            }
        )
    return rows.select(columns)


@feature(
    name="rolling_form",
    version=1,
    grain="player_fixture",
    inputs=["player_fixture_stats"],
    schema=RollingFormSchema,
)
def rolling_form(view: PITView, horizon: list[GWKey]) -> pl.DataFrame:
    """Prior appearance means attached to each horizon fixture."""
    rows = player_fixtures(view, horizon)
    form = latest_form(view.table("player_fixture_stats").collect())
    if rows.is_empty():
        return pl.DataFrame(
            schema={
                **_empty_keys(),
                "total_points_rolling_3": pl.Float64(),
                "total_points_rolling_5": pl.Float64(),
                "total_points_rolling_10": pl.Float64(),
                "minutes_rolling_3": pl.Float64(),
                "minutes_rolling_5": pl.Float64(),
                "minutes_rolling_10": pl.Float64(),
                "minutes_std_5": pl.Float64(),
                "goals_scored_rolling_5": pl.Float64(),
                "assists_rolling_5": pl.Float64(),
                "clean_sheets_rolling_5": pl.Float64(),
                "goals_conceded_rolling_5": pl.Float64(),
                "saves_rolling_5": pl.Float64(),
                "bonus_rolling_5": pl.Float64(),
            }
        )
    return rows.select("player_key", "fixture_id").join(
        form, on="player_key", how="left"
    )


@feature(
    name="opponent_strength",
    version=1,
    grain="player_fixture",
    inputs=["fixtures"],
    schema=OpponentStrengthSchema,
)
def opponent_strength(view: PITView, horizon: list[GWKey]) -> pl.DataFrame:
    """Opponent attack and defence from finished fixtures already visible."""
    rows = _annotate_fixtures(view, player_fixtures(view, horizon))
    if rows.is_empty():
        return pl.DataFrame(
            schema={
                **_empty_keys(),
                "opp_strength_overall": pl.Float64(),
                "opp_strength_attack": pl.Float64(),
                "opp_strength_defence": pl.Float64(),
            }
        )
    ratings = opponent_ratings(
        view.latest("fixtures", by=["season", "fixture_id"]).collect()
    )
    return (
        rows.select("player_key", "fixture_id", "opponent_key")
        .join(ratings, left_on="opponent_key", right_on="team_key", how="left")
        .select(
            "player_key",
            "fixture_id",
            "opp_strength_overall",
            "opp_strength_attack",
            "opp_strength_defence",
        )
    )


def training_frame(view: PITView) -> pl.DataFrame:
    """One row per visible appearance. Rolling columns exclude that appearance."""
    stats = shifted_form(view.table("player_fixture_stats").collect())
    empty = pl.DataFrame(
        schema={column: pl.Float64() for column in FEATURE_COLUMNS}
        | {"total_points": pl.Float64()}
    )
    if stats.is_empty():
        return empty
    players = (
        view.latest("players", by=["season", "player_key"])
        .select("season", "player_key", "team_key", "position")
        .collect()
    )
    fixtures = view.latest("fixtures", by=["season", "fixture_id"]).collect()
    status = (
        view.latest("player_status", by=["season", "player_key"])
        .select("season", "player_key", "price")
        .collect()
    )
    ratings = opponent_ratings(fixtures)
    joined = (
        stats.join(players, on=["season", "player_key"], how="left")
        .join(
            fixtures.select("season", "fixture_id", "team_h_key", "team_a_key"),
            on=["season", "fixture_id"],
            how="left",
        )
        .join(status, on=["season", "player_key"], how="left")
        .with_columns(
            (pl.col("team_key") == pl.col("team_h_key")).alias("is_home"),
            pl.when(pl.col("team_key") == pl.col("team_h_key"))
            .then(pl.col("team_a_key"))
            .otherwise(pl.col("team_h_key"))
            .alias("opponent_key"),
        )
        .join(ratings, left_on="opponent_key", right_on="team_key", how="left")
        .with_columns(
            _position_enc(),
            pl.col("is_home").cast(pl.Float64).alias("was_home"),
            pl.col("price").cast(pl.Float64).alias("value"),
            pl.col("opponent_key").cast(pl.Float64).alias("opponent_team"),
            pl.col("opp_strength_attack").fill_null(1.0),
            pl.col("opp_strength_defence").fill_null(1.0),
            pl.col("opp_strength_overall").fill_null(1.0),
            *[
                pl.col(column).fill_null(0.0)
                for column in FEATURE_COLUMNS
                if column.endswith(("_3", "_5", "_10")) or column == "minutes_std_5"
            ],
        )
    )
    return joined.select([*FEATURE_COLUMNS, "total_points"]).drop_nulls(
        subset=FEATURE_COLUMNS[:4]
    )
