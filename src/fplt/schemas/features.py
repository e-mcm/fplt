"""Pandera schemas for point-in-time feature frames."""

from __future__ import annotations

import pandera.polars as pa


class ContextFeatureSchema(pa.DataFrameModel):
    """Position, venue, price, and opponent id at player-fixture grain."""

    player_key: int
    fixture_id: int
    position_enc: float = pa.Field(nullable=True)
    was_home: float = pa.Field(nullable=True)
    value: float = pa.Field(nullable=True)
    opponent_team: float = pa.Field(nullable=True)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["player_key", "fixture_id"]


class RollingFormSchema(pa.DataFrameModel):
    """Shifted rolling appearance stats attached to a player-fixture."""

    player_key: int
    fixture_id: int
    total_points_rolling_3: float = pa.Field(nullable=True)
    total_points_rolling_5: float = pa.Field(nullable=True)
    total_points_rolling_10: float = pa.Field(nullable=True)
    minutes_rolling_3: float = pa.Field(nullable=True)
    minutes_rolling_5: float = pa.Field(nullable=True)
    minutes_rolling_10: float = pa.Field(nullable=True)
    minutes_std_5: float = pa.Field(nullable=True)
    goals_scored_rolling_5: float = pa.Field(nullable=True)
    assists_rolling_5: float = pa.Field(nullable=True)
    clean_sheets_rolling_5: float = pa.Field(nullable=True)
    goals_conceded_rolling_5: float = pa.Field(nullable=True)
    saves_rolling_5: float = pa.Field(nullable=True)
    bonus_rolling_5: float = pa.Field(nullable=True)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["player_key", "fixture_id"]


class OpponentStrengthSchema(pa.DataFrameModel):
    """Opponent strength from finished fixtures already visible at the deadline."""

    player_key: int
    fixture_id: int
    opp_strength_overall: float = pa.Field(nullable=True)
    opp_strength_attack: float = pa.Field(nullable=True)
    opp_strength_defence: float = pa.Field(nullable=True)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["player_key", "fixture_id"]


class AvailabilityFeatureSchema(pa.DataFrameModel):
    """First-party FPL availability tags at player-fixture grain."""

    player_key: int
    fixture_id: int
    status_ord: float = pa.Field(nullable=True)
    chance_next: float = pa.Field(nullable=True)
    chance_this: float = pa.Field(nullable=True)
    chance_gate: float = pa.Field(nullable=True)
    has_news: float = pa.Field(nullable=True)
    has_news_added: float = pa.Field(nullable=True)
    news_pct: float = pa.Field(nullable=True)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["player_key", "fixture_id"]
