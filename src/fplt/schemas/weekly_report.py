"""Pandera schemas for post-GW decision scoring and weekly reports."""

from __future__ import annotations

import pandera.polars as pa

_GW_KINDS = ["normal", "blank", "double"]


class WeeklyDecisionScoreSchema(pa.DataFrameModel):
    """One model × season × gameweek decision-quality row (ops-08)."""

    model_id: str
    season: str
    gw: int = pa.Field(ge=1, le=38)
    gw_kind: str = pa.Field(isin=_GW_KINDS)
    plan_ev: float
    realised_points: int
    realised_minus_plan_ev: float
    hindsight_best: int
    regret: int = pa.Field(ge=0)
    captaincy_regret: int = pa.Field(ge=0)
    hit: float = pa.Field(ge=0)
    hit_payback_to_date: float
    season_points_to_date: int
    points_at_rank: float = pa.Field(nullable=True)
    percentile: float = pa.Field(nullable=True, ge=0)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["model_id", "season", "gw"]


class SeasonToDateSchema(pa.DataFrameModel):
    """Running season-to-date rollup of weekly decision scores."""

    model_id: str
    season: str
    through_gw: int = pa.Field(ge=1, le=38)
    gameweeks: int = pa.Field(ge=0)
    season_points: int
    plan_ev_sum: float
    regret_sum: int = pa.Field(ge=0)
    captaincy_regret_sum: int = pa.Field(ge=0)
    hits_sum: float = pa.Field(ge=0)
    hit_payback_to_date: float
    points_at_rank: float = pa.Field(nullable=True)
    percentile: float = pa.Field(nullable=True, ge=0)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["model_id", "season", "through_gw"]
