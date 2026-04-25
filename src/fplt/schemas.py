"""Pandera DataFrameModel schemas for every major DataFrame in the fplt pipeline.

Schemas are grouped by pipeline stage:

  Raw data      — GameweekSchema, FixtureSchema, TeamSchema
  Features      — FixtureRecordSchema, FeatureSchema
  Model outputs — MinutesPredSchema, BonusPredSchema, CardPredSchema,
                  SavesPredSchema, GoalAllocPredSchema, ComponentPredSchema,
                  TeamRatingsSchema, FeatureImportanceSchema,
                  DixonColesFixtureSchema
  Optimisation  — PlayerPoolSchema
"""

from __future__ import annotations

from typing import Optional

import pandera.pandas as pa
from pandera.typing import Series

# ── Raw data ──────────────────────────────────────────────────────────────────


class GameweekSchema(pa.DataFrameModel):
    """One row per player per gameweek. Output of load_gameweeks / load_all_seasons."""

    name: Series[str] = pa.Field(nullable=True)
    element: Series[int]
    fixture: Series[int]
    GW: Series[int] = pa.Field(ge=1)
    was_home: Series[bool]
    opponent_team: Series[int]
    minutes: Series[int] = pa.Field(ge=0)
    total_points: Series[int]
    value: Series[int] = pa.Field(gt=0)
    goals_scored: Series[int] = pa.Field(ge=0)
    assists: Series[int] = pa.Field(ge=0)
    clean_sheets: Series[int] = pa.Field(ge=0, le=1)
    goals_conceded: Series[int] = pa.Field(ge=0)
    saves: Series[int] = pa.Field(ge=0)
    bonus: Series[int] = pa.Field(ge=0, le=3)
    yellow_cards: Series[int] = pa.Field(ge=0, le=1)
    red_cards: Series[int] = pa.Field(ge=0, le=1)
    season: Series[str]

    # Extended columns present from ~2019-20 onwards
    position: Optional[Series[str]] = pa.Field(nullable=True)
    team: Optional[Series[str]] = pa.Field(nullable=True)
    xP: Optional[Series[float]] = pa.Field(nullable=True)
    starts: Optional[Series[int]] = pa.Field(nullable=True)
    expected_goals: Optional[Series[float]] = pa.Field(nullable=True)
    expected_assists: Optional[Series[float]] = pa.Field(nullable=True)
    expected_goal_involvements: Optional[Series[float]] = pa.Field(nullable=True)
    expected_goals_conceded: Optional[Series[float]] = pa.Field(nullable=True)

    class Config:
        strict = False  # extra columns (e.g. mng_* in 2024-25) are ignored
        coerce = True  # tolerate int/float mismatches from CSV loading


class FixtureSchema(pa.DataFrameModel):
    """One row per fixture. Output of load_fixtures."""

    id: Series[int]
    team_h: Series[int]
    team_a: Series[int]
    season: Series[str]

    team_h_score: Optional[Series[float]] = pa.Field(nullable=True, ge=0)
    team_a_score: Optional[Series[float]] = pa.Field(nullable=True, ge=0)

    class Config:
        strict = False
        coerce = True


class TeamSchema(pa.DataFrameModel):
    """One row per team. Output of load_teams."""

    id: Series[int]
    name: Series[str]
    strength: Series[int]
    season: Series[str]

    class Config:
        strict = False
        coerce = True


# ── Feature engineering ───────────────────────────────────────────────────────


class FixtureRecordSchema(pa.DataFrameModel):
    """Fixture-level training data for DixonColesModel. Output of build_fixture_records."""

    fixture_id: Series[object]  # int or str depending on source
    season: Series[str]
    team_h: Series[object]  # team ID or name (int or str)
    team_a: Series[object]
    team_h_score: Series[int] = pa.Field(ge=0)
    team_a_score: Series[int] = pa.Field(ge=0)

    kickoff_time: Optional[Series[object]]  # tz-aware datetime when present

    class Config:
        strict = False
        coerce = True


class FeatureSchema(pa.DataFrameModel):
    """Engineered player feature matrix. Output of build_features.

    Rolling features are not listed here (strict=False); callers rely on
    LGBMPredictor.feature_cols for the exact feature set used by each model.
    """

    element: Series[int]
    GW: Series[int] = pa.Field(ge=1)
    was_home: Series[bool]
    value: Series[int] = pa.Field(gt=0)
    opponent_team: Series[int]
    position_enc: Series[int] = pa.Field(ge=-1, le=3)
    season: Series[str]

    # Opponent strength — present whenever teams_df has the columns
    opp_strength_overall: Optional[Series[float]] = pa.Field(nullable=True)
    opp_strength_attack: Optional[Series[float]] = pa.Field(nullable=True)
    opp_strength_defence: Optional[Series[float]] = pa.Field(nullable=True)

    class Config:
        strict = False
        coerce = True


# ── Dixon-Coles ───────────────────────────────────────────────────────────────


class DixonColesFixtureSchema(pa.DataFrameModel):
    """Input to DixonColesModel.predict_distribution — one row per fixture."""

    team_h: Series[object]
    team_a: Series[object]

    fixture_id: Optional[Series[object]]  # used as ScoreDistribution identifier

    class Config:
        strict = False
        coerce = True


class TeamRatingsSchema(pa.DataFrameModel):
    """Output of DixonColesModel.team_ratings()."""

    team: Series[object]
    attack: Series[float] = pa.Field(gt=0)
    defence: Series[float] = pa.Field(gt=0)

    class Config:
        strict = False


# ── Model prediction outputs ──────────────────────────────────────────────────


class MinutesPredSchema(pa.DataFrameModel):
    """Output of MinutesModel.predict()."""

    p_starts: Series[float] = pa.Field(ge=0.0, le=1.0)
    e_minutes: Series[float] = pa.Field(ge=0.0, le=90.0)


class BonusPredSchema(pa.DataFrameModel):
    """Output of BonusModel.predict()."""

    e_bonus: Series[float] = pa.Field(ge=0.0, le=3.0)


class CardPredSchema(pa.DataFrameModel):
    """Output of CardModel.predict()."""

    p_yellow: Series[float] = pa.Field(ge=0.0, le=1.0)
    p_red: Series[float] = pa.Field(ge=0.0, le=1.0)


class SavesPredSchema(pa.DataFrameModel):
    """Output of SavesModel.predict()."""

    e_saves: Series[float] = pa.Field(ge=0.0)


class GoalAllocPredSchema(pa.DataFrameModel):
    """Output of GoalAllocationModel.predict()."""

    expected_goals_scored: Series[float] = pa.Field(ge=0.0)


class ComponentPredSchema(pa.DataFrameModel):
    """Output of ComponentPredictor.predict_components()."""

    minutes: Optional[Series[float]] = pa.Field(ge=0.0, le=90.0, nullable=True)
    goals_scored: Optional[Series[float]] = pa.Field(ge=0.0, le=5.0, nullable=True)
    assists: Optional[Series[float]] = pa.Field(ge=0.0, le=3.0, nullable=True)
    clean_sheets: Optional[Series[float]] = pa.Field(ge=0.0, le=1.0, nullable=True)
    saves: Optional[Series[float]] = pa.Field(ge=0.0, le=15.0, nullable=True)
    bonus: Optional[Series[float]] = pa.Field(ge=0.0, le=3.0, nullable=True)
    yellow_cards: Optional[Series[float]] = pa.Field(ge=0.0, le=1.0, nullable=True)

    class Config:
        strict = False


class FeatureImportanceSchema(pa.DataFrameModel):
    """Output of LGBMPredictor.feature_importance() and similar."""

    feature: Series[str]
    importance: Series[float] = pa.Field(ge=0.0)


# ── Optimisation ──────────────────────────────────────────────────────────────


class PlayerPoolSchema(pa.DataFrameModel):
    """Input to select_team and recommend_transfers."""

    element: Series[int]
    position: Series[str]
    team: Series[object]
    value: Series[int] = pa.Field(gt=0)
    xpts: Series[float]

    class Config:
        strict = False
        coerce = True
