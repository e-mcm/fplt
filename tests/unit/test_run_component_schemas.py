"""Pandera contracts for predict-run component tables (§3.1)."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest
from pandera.errors import SchemaError

from fplt.schemas.l2 import SCHEMA_VERSION, L2TeamFixtureKeys
from fplt.schemas.run_components import (
    COMPONENT_INVOLVEMENT_COLUMNS,
    COMPONENT_MINUTES_COLUMNS,
    COMPONENT_SCORELINE_COLUMNS,
    EXP_GOALS_METHODS,
    ComponentInvolvementSchema,
    ComponentMinutesSchema,
    ComponentScorelineSchema,
)

_TS = datetime(2026, 11, 24, 12, 56, 1, tzinfo=UTC)


def _l2_player_keys(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "run_id": ["r_predict"],
        "as_of_season": ["2026-27"],
        "as_of_gw": [12],
        "deadline_utc": [_TS],
        "target_season": ["2026-27"],
        "target_gw": [13],
        "horizon": [1],
        "fixture_id": [100],
        "player_key": [42],
        "team_key": [7],
        "schema_version": [SCHEMA_VERSION],
    }
    base.update(overrides)
    return base


def _l2_team_keys(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "run_id": ["r_predict"],
        "as_of_season": ["2026-27"],
        "as_of_gw": [12],
        "deadline_utc": [_TS],
        "target_season": ["2026-27"],
        "target_gw": [13],
        "horizon": [1],
        "fixture_id": [100],
        "team_key": [7],
        "schema_version": [SCHEMA_VERSION],
    }
    base.update(overrides)
    return base


def test_component_minutes_schema_accepts_valid_row() -> None:
    frame = pl.DataFrame(
        {
            **_l2_player_keys(),
            "xmins": [62.5],
            "p_played": [0.9],
            "p_60": [0.7],
        }
    )
    checked = ComponentMinutesSchema.validate(frame)
    assert checked["xmins"].to_list() == [62.5]
    assert set(checked.columns) == set(COMPONENT_MINUTES_COLUMNS)


def test_component_minutes_rejects_negative_xmins() -> None:
    frame = pl.DataFrame(
        {
            **_l2_player_keys(),
            "xmins": [-1.0],
            "p_played": [0.5],
            "p_60": [0.2],
        }
    )
    with pytest.raises(SchemaError):
        ComponentMinutesSchema.validate(frame)


def test_component_scoreline_schema_two_rows_per_fixture() -> None:
    frame = pl.DataFrame(
        {
            **_l2_team_keys(
                team_key=[7, 11],
                fixture_id=[100, 100],
                run_id=["r_predict", "r_predict"],
                as_of_season=["2026-27", "2026-27"],
                as_of_gw=[12, 12],
                deadline_utc=[_TS, _TS],
                target_season=["2026-27", "2026-27"],
                target_gw=[13, 13],
                horizon=[1, 1],
                schema_version=[SCHEMA_VERSION, SCHEMA_VERSION],
            ),
            "opponent_team_key": [11, 7],
            "is_home": [True, False],
            "team_eg": [1.4, 1.1],
            "opp_eg": [1.1, 1.4],
            "p_cs": [0.3, 0.25],
        }
    )
    checked = ComponentScorelineSchema.validate(frame)
    assert checked.height == 2
    assert set(checked.columns) == set(COMPONENT_SCORELINE_COLUMNS)
    team_only = frame.select(list(L2TeamFixtureKeys.to_schema().columns))
    L2TeamFixtureKeys.validate(team_only)


def test_component_scoreline_rejects_player_key_column() -> None:
    frame = pl.DataFrame(
        {
            **_l2_team_keys(),
            "player_key": [42],
            "opponent_team_key": [11],
            "is_home": [True],
            "team_eg": [1.0],
            "opp_eg": [1.0],
            "p_cs": [0.2],
        }
    )
    with pytest.raises(SchemaError):
        ComponentScorelineSchema.validate(frame)


def test_component_involvement_schema_and_method_enum() -> None:
    frame = pl.DataFrame(
        {
            **_l2_player_keys(),
            "goal_weight": [0.2],
            "assist_weight": [0.15],
            "exp_goals": [0.28],
            "exp_assists": [0.21],
            "exp_goals_method": ["team_eg_x_weight"],
        }
    )
    checked = ComponentInvolvementSchema.validate(frame)
    assert checked["exp_goals_method"].to_list() == ["team_eg_x_weight"]
    assert set(EXP_GOALS_METHODS) == {"sim_mean", "team_eg_x_weight"}
    assert set(checked.columns) == set(COMPONENT_INVOLVEMENT_COLUMNS)


def test_component_involvement_rejects_unknown_method() -> None:
    frame = pl.DataFrame(
        {
            **_l2_player_keys(),
            "goal_weight": [0.2],
            "assist_weight": [0.15],
            "exp_goals": [0.28],
            "exp_assists": [0.21],
            "exp_goals_method": ["chained_xpts"],
        }
    )
    with pytest.raises(SchemaError):
        ComponentInvolvementSchema.validate(frame)


def test_empty_frames_match_column_dicts() -> None:
    assert ComponentMinutesSchema.validate(
        pl.DataFrame(schema=COMPONENT_MINUTES_COLUMNS)
    ).is_empty()
    assert ComponentScorelineSchema.validate(
        pl.DataFrame(schema=COMPONENT_SCORELINE_COLUMNS)
    ).is_empty()
    assert ComponentInvolvementSchema.validate(
        pl.DataFrame(schema=COMPONENT_INVOLVEMENT_COLUMNS)
    ).is_empty()
