"""Pandera schemas for layered sub-model tables on predict runs.

Every layered predict run writes these under
``predictions/components/{minutes,scoreline,involvement}.parquet``
(``docs/proposals/weekly-ui-milestone.md`` §3.1). Opaque models write none.
``exp_goals`` / ``exp_assists`` are display-only and must never feed ``xpts``.
"""

from __future__ import annotations

from typing import Literal

import pandera.polars as pa
import polars as pl

from fplt.schemas.l2 import L2StandardKeys, L2TeamFixtureKeys

ExpGoalsMethod = Literal["sim_mean", "team_eg_x_weight"]
EXP_GOALS_METHODS: tuple[str, ...] = ("sim_mean", "team_eg_x_weight")


class ComponentMinutesSchema(L2StandardKeys):
    """Minutes-component summary at player–fixture grain.

    ``xmins`` is ``Σ bin · P(bin)`` from the minutes PMF;
    ``p_played`` is ``P(minutes > 0)``; ``p_60`` is ``P(minutes ≥ 60)``.
    """

    xmins: float = pa.Field(ge=0)
    p_played: float = pa.Field(ge=0, le=1)
    p_60: float = pa.Field(ge=0, le=1)

    class Config(L2StandardKeys.Config):
        unique = [  # noqa: RUF012
            "run_id",
            "as_of_season",
            "as_of_gw",
            "player_key",
            "fixture_id",
        ]  # noqa: RUF012


class ComponentScorelineSchema(L2TeamFixtureKeys):
    """Score-component summary at team–fixture grain (two rows per fixture).

    ``team_eg`` is λ for this team; ``opp_eg`` is λ against; ``p_cs`` is
    P(opponent scores 0) from the scoreline PMF.
    """

    opponent_team_key: int
    is_home: bool
    team_eg: float = pa.Field(ge=0)
    opp_eg: float = pa.Field(ge=0)
    p_cs: float = pa.Field(ge=0, le=1)

    class Config(L2TeamFixtureKeys.Config):
        unique = [  # noqa: RUF012
            "run_id",
            "as_of_season",
            "as_of_gw",
            "fixture_id",
            "team_key",
        ]  # noqa: RUF012


class ComponentInvolvementSchema(L2StandardKeys):
    """Involvement-component summary at player–fixture grain.

    ``goal_weight`` / ``assist_weight`` are team-relative shares.
    ``exp_goals`` / ``exp_assists`` are display-only; ``exp_goals_method``
    records whether they came from sim means or ``team_eg × weight``.
    """

    goal_weight: float = pa.Field(ge=0)
    assist_weight: float = pa.Field(ge=0)
    exp_goals: float = pa.Field(ge=0)
    exp_assists: float = pa.Field(ge=0)
    exp_goals_method: str = pa.Field(isin=list(EXP_GOALS_METHODS))

    class Config(L2StandardKeys.Config):
        unique = [  # noqa: RUF012
            "run_id",
            "as_of_season",
            "as_of_gw",
            "player_key",
            "fixture_id",
        ]  # noqa: RUF012


COMPONENT_MINUTES_COLUMNS: dict[str, pl.DataType] = {
    "run_id": pl.Utf8(),
    "as_of_season": pl.Utf8(),
    "as_of_gw": pl.Int64(),
    "deadline_utc": pl.Datetime(time_zone="UTC"),
    "target_season": pl.Utf8(),
    "target_gw": pl.Int64(),
    "horizon": pl.Int64(),
    "fixture_id": pl.Int64(),
    "player_key": pl.Int64(),
    "team_key": pl.Int64(),
    "schema_version": pl.Int64(),
    "xmins": pl.Float64(),
    "p_played": pl.Float64(),
    "p_60": pl.Float64(),
}

COMPONENT_SCORELINE_COLUMNS: dict[str, pl.DataType] = {
    "run_id": pl.Utf8(),
    "as_of_season": pl.Utf8(),
    "as_of_gw": pl.Int64(),
    "deadline_utc": pl.Datetime(time_zone="UTC"),
    "target_season": pl.Utf8(),
    "target_gw": pl.Int64(),
    "horizon": pl.Int64(),
    "fixture_id": pl.Int64(),
    "team_key": pl.Int64(),
    "schema_version": pl.Int64(),
    "opponent_team_key": pl.Int64(),
    "is_home": pl.Boolean(),
    "team_eg": pl.Float64(),
    "opp_eg": pl.Float64(),
    "p_cs": pl.Float64(),
}

COMPONENT_INVOLVEMENT_COLUMNS: dict[str, pl.DataType] = {
    "run_id": pl.Utf8(),
    "as_of_season": pl.Utf8(),
    "as_of_gw": pl.Int64(),
    "deadline_utc": pl.Datetime(time_zone="UTC"),
    "target_season": pl.Utf8(),
    "target_gw": pl.Int64(),
    "horizon": pl.Int64(),
    "fixture_id": pl.Int64(),
    "player_key": pl.Int64(),
    "team_key": pl.Int64(),
    "schema_version": pl.Int64(),
    "goal_weight": pl.Float64(),
    "assist_weight": pl.Float64(),
    "exp_goals": pl.Float64(),
    "exp_assists": pl.Float64(),
    "exp_goals_method": pl.Utf8(),
}
