"""Pandera schemas for post-gameweek analysis runs (#393 / #394)."""

from __future__ import annotations

import pandera.polars as pa

_GW_KINDS = ["normal", "blank", "double"]
_MISS_REASONS = [
    "benched",
    "subbed_early",
    "red_card",
    "injury",
    "penalty",
    "dgw",
    "blank",
    "other",
]


class PostGwScorecardSchema(pa.DataFrameModel):
    """One model × season × gameweek scorecard row for a post_gw run."""

    model_id: str
    season: str
    gw: int = pa.Field(ge=1, le=38)
    gw_kind: str = pa.Field(isin=_GW_KINDS)
    plan_ev: float
    realised_points: int
    realised_minus_plan_ev: float
    pmf_percentile: float = pa.Field(nullable=True, ge=0.0, le=1.0)
    fpl_average: float = pa.Field(nullable=True)
    fpl_highest: float = pa.Field(nullable=True)
    rank_change: float = pa.Field(nullable=True)
    season_points_to_date: int
    points_at_rank: float = pa.Field(nullable=True)
    percentile: float = pa.Field(nullable=True, ge=0.0)
    realised_ex_chips: int = pa.Field(nullable=True)
    chip: str = pa.Field(nullable=True)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["model_id", "season", "gw"]  # noqa: RUF012


class PostGwAttributionSchema(pa.DataFrameModel):
    """Decision-attribution row wrapping ops-08 ``decision_scoring``."""

    model_id: str
    season: str
    gw: int = pa.Field(ge=1, le=38)
    gw_kind: str = pa.Field(isin=_GW_KINDS)
    plan_ev: float
    realised_points: int
    luck_vs_model: float
    captaincy_regret: int = pa.Field(ge=0)
    xi_bench_regret: int = pa.Field(ge=0)
    transfer_regret: int = pa.Field(ge=0)
    chip_timing_regret: int = pa.Field(nullable=True, ge=0)
    hindsight_best: int
    regret: int = pa.Field(ge=0)
    hit: float = pa.Field(ge=0)
    hit_payback_to_date: float
    chip_value: float = pa.Field(nullable=True)
    intervention_delta: float = pa.Field(nullable=True)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["model_id", "season", "gw"]  # noqa: RUF012


class PostGwBiggestMissSchema(pa.DataFrameModel):
    """Top |actual − xpts| player–fixture misses for a post_gw run (#394)."""

    model_id: str
    season: str
    gw: int = pa.Field(ge=1, le=38)
    player_key: int
    fixture_id: int
    xpts: float
    actual_pts: int
    abs_error: float = pa.Field(ge=0.0)
    signed_error: float
    reason: str = pa.Field(isin=_MISS_REASONS)
    predictable_at_T90: bool
    rank: int = pa.Field(ge=1, le=15)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["model_id", "season", "gw", "player_key", "fixture_id"]  # noqa: RUF012
