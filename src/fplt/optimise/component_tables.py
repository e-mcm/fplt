"""Build layered sub-model tables for predict runs (weekly-ui §3.1).

Generalises explain-depth score / minutes / involvement predicts across the
full horizon. Opaque models write none. ``exp_goals`` is display-only.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import polars as pl

from fplt.data.pit import PITView
from fplt.models.distributions import ScorelineDist
from fplt.models.universe import player_fixtures
from fplt.optimise.explain import LayeredExplainParts, minutes_summary_from_pmf
from fplt.schemas.explain import PLAYER_DEPTH_COLUMNS, PlayerDepthSchema
from fplt.schemas.l2 import SCHEMA_VERSION
from fplt.schemas.run_components import (
    COMPONENT_INVOLVEMENT_COLUMNS,
    COMPONENT_MINUTES_COLUMNS,
    COMPONENT_SCORELINE_COLUMNS,
    ComponentInvolvementSchema,
    ComponentMinutesSchema,
    ComponentScorelineSchema,
)
from fplt.types import GWKey


@dataclass(frozen=True)
class RunComponentTables:
    """Validated minutes / scoreline / involvement frames for one predict run."""

    minutes: pl.DataFrame
    scoreline: pl.DataFrame
    involvement: pl.DataFrame


def build_run_component_tables(
    view: PITView,
    *,
    parts: LayeredExplainParts,
    horizon: list[GWKey],
    run_id: str,
    as_of: GWKey,
    deadline_utc: datetime,
) -> RunComponentTables:
    """Build the three §3.1 tables for every player–fixture in ``horizon``."""
    asked = player_fixtures(view, horizon) if horizon else pl.DataFrame()
    if asked.is_empty():
        return _empty()
    players = (
        view.latest("players", by=["season", "player_key"])
        .select("season", "player_key", "team_key")
        .collect()
    )
    fixtures = (
        view.latest("fixtures", by=["season", "fixture_id"])
        .select("season", "fixture_id", "gw", "team_h_key", "team_a_key")
        .collect()
    )
    annotated = (
        asked.join(players, on=["season", "player_key"], how="inner")
        .join(fixtures, on=["season", "fixture_id"], how="inner")
        .with_columns(
            (pl.col("team_key") == pl.col("team_h_key")).alias("is_home"),
            pl.when(pl.col("team_key") == pl.col("team_h_key"))
            .then(pl.col("team_a_key"))
            .otherwise(pl.col("team_h_key"))
            .alias("opponent_team_key"),
            pl.col("gw").cast(pl.Int64).alias("target_gw"),
            pl.lit(as_of.season).alias("target_season"),
        )
    )
    fixture_rows = (
        fixtures.join(
            asked.select("season", "fixture_id").unique(),
            on=["season", "fixture_id"],
            how="inner",
        )
        .select("season", "fixture_id", "gw", "team_h_key", "team_a_key")
        .unique(subset=["fixture_id"])
    )
    score_dist: ScorelineDist = parts.score.predict(view, fixture_rows)
    eg = score_dist.expected_goals()
    pf = asked.select("player_key", "fixture_id")
    mins = minutes_summary_from_pmf(parts.minutes.predict(view, pf))
    weights = parts.involvement.predict(view, pf)
    keys = _keys(run_id=run_id, as_of=as_of, deadline_utc=deadline_utc)
    return RunComponentTables(
        minutes=_ok(
            ComponentMinutesSchema,
            annotated.join(mins, on=["player_key", "fixture_id"], how="inner")
            .with_columns(*keys)
            .select(list(COMPONENT_MINUTES_COLUMNS)),
        ),
        scoreline=_scoreline(
            fixture_rows,
            score_dist,
            eg,
            run_id=run_id,
            as_of=as_of,
            deadline_utc=deadline_utc,
        ),
        involvement=_ok(
            ComponentInvolvementSchema,
            annotated.join(weights, on=["player_key", "fixture_id"], how="inner")
            .join(eg, on="fixture_id", how="left")
            .with_columns(
                pl.when(pl.col("is_home"))
                .then(pl.col("home_eg"))
                .otherwise(pl.col("away_eg"))
                .alias("team_eg"),
                *keys,
            )
            .with_columns(
                (pl.col("team_eg") * pl.col("goal_weight")).alias("exp_goals"),
                (pl.col("team_eg") * pl.col("assist_weight")).alias("exp_assists"),
                pl.lit("team_eg_x_weight").alias("exp_goals_method"),
            )
            .select(list(COMPONENT_INVOLVEMENT_COLUMNS)),
        ),
    )


def depth_rows_from_component_tables(
    *,
    minutes: pl.DataFrame,
    scoreline: pl.DataFrame,
    involvement: pl.DataFrame,
    player_keys: list[int],
    target_gw: int,
) -> pl.DataFrame:
    """Rebuild explain ``PlayerDepthSchema`` rows from stored component tables."""
    keys = [int(key) for key in player_keys]
    if not keys:
        return _ok(PlayerDepthSchema, pl.DataFrame(schema=PLAYER_DEPTH_COLUMNS))
    mins = minutes.filter(
        pl.col("player_key").is_in(keys) & (pl.col("target_gw") == target_gw)
    )
    inv = involvement.filter(
        pl.col("player_key").is_in(keys) & (pl.col("target_gw") == target_gw)
    )
    score = scoreline.filter(pl.col("target_gw") == target_gw).select(
        "fixture_id",
        "team_key",
        "opponent_team_key",
        "is_home",
        "team_eg",
        "opp_eg",
    )
    return _ok(
        PlayerDepthSchema,
        mins.select("player_key", "fixture_id", "team_key", "xmins", "target_gw")
        .join(
            inv.select("player_key", "fixture_id", "goal_weight", "assist_weight"),
            on=["player_key", "fixture_id"],
            how="left",
        )
        .join(score, on=["fixture_id", "team_key"], how="left")
        .select(list(PLAYER_DEPTH_COLUMNS)),
    )


def load_run_component_tables(
    data_dir: Path,
    *,
    model_name: str,
    season: str,
    gameweek: int,
) -> RunComponentTables | None:
    """Load §3.1 tables from the latest matching predict run, if present."""
    from fplt.store.api import Store

    store = Store(data_dir)
    catalog = store.runs(spec_name=model_name, run_type="predict")
    if catalog.is_empty():
        return None
    matched = catalog.filter(
        (pl.col("as_of_season") == season) & (pl.col("as_of_gw") == gameweek)
    )
    if matched.is_empty():
        return None
    run_id = str(matched.sort("run_id", descending=True)["run_id"][0])
    try:
        minutes = store.load("minutes", run=run_id).collect()
        scoreline = store.load("scoreline", run=run_id).collect()
        involvement = store.load("involvement", run=run_id).collect()
    except FileNotFoundError:
        return None
    return RunComponentTables(
        minutes=_ok(ComponentMinutesSchema, minutes),
        scoreline=_ok(ComponentScorelineSchema, scoreline),
        involvement=_ok(ComponentInvolvementSchema, involvement),
    )


def _scoreline(
    fixture_rows: pl.DataFrame,
    score_dist: ScorelineDist,
    eg: pl.DataFrame,
    *,
    run_id: str,
    as_of: GWKey,
    deadline_utc: datetime,
) -> pl.DataFrame:
    if fixture_rows.is_empty():
        return _ok(
            ComponentScorelineSchema, pl.DataFrame(schema=COMPONENT_SCORELINE_COLUMNS)
        )
    base = fixture_rows.join(eg, on="fixture_id").join(
        pl.DataFrame(
            {
                "fixture_id": np.asarray(score_dist.fixture_ids, dtype=np.int64),
                "p_cs_home": score_dist.p_clean_sheet("home"),
                "p_cs_away": score_dist.p_clean_sheet("away"),
            }
        ),
        on="fixture_id",
    )
    sides: list[pl.DataFrame] = []
    for is_home, team, opp, teg, oeg, pcs in (
        (True, "team_h_key", "team_a_key", "home_eg", "away_eg", "p_cs_home"),
        (False, "team_a_key", "team_h_key", "away_eg", "home_eg", "p_cs_away"),
    ):
        sides.append(
            base.select(
                "fixture_id",
                pl.col("gw").cast(pl.Int64).alias("target_gw"),
                pl.col(team).alias("team_key"),
                pl.col(opp).alias("opponent_team_key"),
                pl.lit(is_home).alias("is_home"),
                pl.col(teg).alias("team_eg"),
                pl.col(oeg).alias("opp_eg"),
                pl.col(pcs).alias("p_cs"),
            )
        )
    frame = (
        pl.concat(sides)
        .with_columns(
            pl.lit(as_of.season).alias("target_season"),
            *_keys(run_id=run_id, as_of=as_of, deadline_utc=deadline_utc),
        )
        .select(list(COMPONENT_SCORELINE_COLUMNS))
    )
    return _ok(ComponentScorelineSchema, frame)


def _keys(*, run_id: str, as_of: GWKey, deadline_utc: datetime) -> list[pl.Expr]:
    return [
        pl.lit(run_id).alias("run_id"),
        pl.lit(as_of.season).alias("as_of_season"),
        pl.lit(as_of.gw).cast(pl.Int64).alias("as_of_gw"),
        pl.lit(deadline_utc).alias("deadline_utc"),
        (pl.col("target_gw") - as_of.gw).cast(pl.Int64).alias("horizon"),
        pl.lit(SCHEMA_VERSION).cast(pl.Int64).alias("schema_version"),
    ]


def _empty() -> RunComponentTables:
    return RunComponentTables(
        minutes=_ok(
            ComponentMinutesSchema, pl.DataFrame(schema=COMPONENT_MINUTES_COLUMNS)
        ),
        scoreline=_ok(
            ComponentScorelineSchema, pl.DataFrame(schema=COMPONENT_SCORELINE_COLUMNS)
        ),
        involvement=_ok(
            ComponentInvolvementSchema,
            pl.DataFrame(schema=COMPONENT_INVOLVEMENT_COLUMNS),
        ),
    )


def _ok(schema: type[object], frame: pl.DataFrame) -> pl.DataFrame:
    validate = getattr(schema, "validate")  # noqa: B009 - schema is type[object]
    checked = validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked
