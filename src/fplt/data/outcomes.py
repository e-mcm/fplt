"""Stage curated actuals for the ledger view (design §7.2 / data model §3.6).

``player_fixture_outcomes`` and ``fixture_outcomes`` are derived from staged
``player_fixture_stats`` / ``fixtures`` once the target gameweek is
``data_checked``. Corrections re-stage and land in a new curated build rather
than rewriting prediction rows.
"""

from __future__ import annotations

import logging

import polars as pl

from fplt.data.store import DataStore
from fplt.schemas.l2 import SCHEMA_VERSION
from fplt.schemas.outcomes import FixtureOutcomesSchema, PlayerFixtureOutcomesSchema
from fplt.utils.frames import require_dataframe

logger = logging.getLogger(__name__)


def build_player_fixture_outcomes(store: DataStore) -> pl.DataFrame:
    """Build ``player_fixture_outcomes`` for data-checked gameweeks.

    Uses the latest ``player_fixture_stats`` row per
    ``(season, gw, player_key, fixture_id)`` (same semantics the old
    ``fill_actuals`` join used). ``team_key`` comes from the latest
    ``players`` row for that season.

    Args:
        store: Staged tables including stats, gameweeks, and players.

    Returns:
        Rows matching ``PlayerFixtureOutcomesSchema`` (possibly empty).
    """
    checked = _checked_gameweeks(store)
    if checked.is_empty():
        return pl.DataFrame(schema=_player_outcomes_polars())
    stats = (
        store.read("player_fixture_stats")
        .sort("available_at")
        .unique(subset=["season", "gw", "player_key", "fixture_id"], keep="last")
    )
    players = (
        store.read("players")
        .sort("available_at")
        .unique(subset=["season", "player_key"], keep="last")
        .select("season", "player_key", "team_key")
    )
    frame = (
        stats.join(checked, on=["season", "gw"], how="inner")
        .join(players, on=["season", "player_key"], how="left")
        .with_columns(
            pl.col("team_key").fill_null(0),
            pl.lit(SCHEMA_VERSION).alias("schema_version"),
        )
        .select(
            "season",
            "gw",
            "fixture_id",
            "player_key",
            "team_key",
            pl.col("total_points").alias("pts"),
            "minutes",
            pl.col("goals_scored").alias("goals"),
            "assists",
            "clean_sheets",
            "available_at",
            "schema_version",
            "source",
        )
    )
    checked_frame = PlayerFixtureOutcomesSchema.validate(frame)
    checked_frame = require_dataframe(checked_frame)
    return checked_frame


def build_fixture_outcomes(store: DataStore) -> pl.DataFrame:
    """Build ``fixture_outcomes`` for data-checked gameweeks with scores.

    Args:
        store: Staged tables including fixtures and gameweeks.

    Returns:
        Rows matching ``FixtureOutcomesSchema`` (possibly empty).
    """
    checked = _checked_gameweeks(store)
    if checked.is_empty():
        return pl.DataFrame(schema=_fixture_outcomes_polars())
    fixtures = (
        store.read("fixtures")
        .sort("available_at")
        .unique(subset=["season", "fixture_id"], keep="last")
        .filter(
            pl.col("team_h_score").is_not_null() & pl.col("team_a_score").is_not_null()
        )
    )
    frame = (
        fixtures.join(checked, on=["season", "gw"], how="inner")
        .with_columns(pl.lit(SCHEMA_VERSION).alias("schema_version"))
        .select(
            "season",
            "gw",
            "fixture_id",
            "team_h_key",
            "team_a_key",
            "team_h_score",
            "team_a_score",
            "available_at",
            "schema_version",
            "source",
        )
    )
    checked_frame = FixtureOutcomesSchema.validate(frame)
    checked_frame = require_dataframe(checked_frame)
    return checked_frame


def stage_outcomes(store: DataStore) -> dict[str, pl.DataFrame]:
    """Write both outcomes tables into ``store`` and return the frames.

    Args:
        store: Staged data root to read from and write into.

    Returns:
        Mapping of table name to the staged frame (empty tables are skipped).
    """
    written: dict[str, pl.DataFrame] = {}
    player = build_player_fixture_outcomes(store)
    if not player.is_empty():
        written["player_fixture_outcomes"] = store.write(
            "player_fixture_outcomes", player
        )
    fixture = build_fixture_outcomes(store)
    if not fixture.is_empty():
        written["fixture_outcomes"] = store.write("fixture_outcomes", fixture)
    logger.info(
        "staged outcomes player_fixture=%s fixture=%s",
        written.get("player_fixture_outcomes", pl.DataFrame()).height,
        written.get("fixture_outcomes", pl.DataFrame()).height,
    )
    return written


def _checked_gameweeks(store: DataStore) -> pl.DataFrame:
    """Latest ``(season, gw)`` rows whose ``data_checked`` flag is true."""
    return (
        store.read("gameweeks")
        .filter(pl.col("data_checked"))
        .sort("available_at")
        .unique(subset=["season", "gw"], keep="last")
        .select("season", "gw")
    )


def _player_outcomes_polars() -> dict[str, pl.DataType]:
    from fplt.schemas.staged import POLARS_SCHEMAS

    return POLARS_SCHEMAS["player_fixture_outcomes"]


def _fixture_outcomes_polars() -> dict[str, pl.DataType]:
    from fplt.schemas.staged import POLARS_SCHEMAS

    return POLARS_SCHEMAS["fixture_outcomes"]
