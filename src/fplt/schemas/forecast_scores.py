"""Pandera schemas for post-GW forecast evaluation (#395 / Spec §2.3)."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl

from fplt.schemas.error_attribution import PRICE_BANDS
from fplt.schemas.l2 import SCHEMA_VERSION

FORECAST_SCORE_EVENTS: tuple[str, ...] = (
    "played",
    "ge60",
    "goal",
    "assist",
    "cs",
    "haul",
    "blank",
)

AGG_METRICS: tuple[str, ...] = (
    "mean_rps",
    "mean_log_score",
    "rps_skill_clim",
    "pit_coverage_50",
    "pit_coverage_80",
    "pit_coverage_90",
    "pit_hist",
    "selection_bias",
    "selection_bias_delta",
    "mz_intercept",
    "mz_slope",
    "spearman_within_pos",
    "top5_hit_rate",
    "top10_hit_rate",
    "brier_played",
    "brier_ge60",
    "brier_goal",
    "brier_assist",
    "brier_cs",
    "brier_haul",
    "brier_blank",
    "mse_mean",
)


class ForecastScoresSchema(pa.DataFrameModel):
    """One player–fixture proper-score / calibration / event row.

    ``selected_lines`` is a comma-joined sorted list of official line names
    that owned the player in the plan (empty string when none). ``pit_u`` is
    the randomised PIT; ``log_score`` uses ``LOG_SCORE_FLOOR`` from
    ``fplt.backtest.points_scores``.
    """

    model_id: str
    season: str
    gw: int = pa.Field(ge=1, le=38)
    player_key: int
    fixture_id: int
    position: str
    price_band: str = pa.Field(isin=list(PRICE_BANDS))
    minutes_bucket: str
    y: int
    mu_hat: float
    rps: float = pa.Field(ge=0.0)
    log_score: float = pa.Field(ge=0.0)
    rps_clim: float = pa.Field(nullable=True, ge=0.0)
    pit_u: float = pa.Field(ge=0.0, le=1.0)
    p_played: float = pa.Field(ge=0.0, le=1.0)
    p_ge60: float = pa.Field(ge=0.0, le=1.0)
    p_goal: float = pa.Field(ge=0.0, le=1.0)
    p_assist: float = pa.Field(ge=0.0, le=1.0)
    p_cs: float = pa.Field(ge=0.0, le=1.0)
    p_haul: float = pa.Field(ge=0.0, le=1.0)
    p_blank: float = pa.Field(ge=0.0, le=1.0)
    y_played: int = pa.Field(ge=0, le=1)
    y_ge60: int = pa.Field(ge=0, le=1)
    y_goal: int = pa.Field(ge=0, le=1)
    y_assist: int = pa.Field(ge=0, le=1)
    y_cs: int = pa.Field(ge=0, le=1)
    y_haul: int = pa.Field(ge=0, le=1)
    y_blank: int = pa.Field(ge=0, le=1)
    selected_lines: str
    schema_version: int = pa.Field(ge=1)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["model_id", "season", "gw", "player_key", "fixture_id"]  # noqa: RUF012


class ForecastScoreAggSchema(pa.DataFrameModel):
    """Season-to-date (or per-GW) aggregation of ``forecast_scores``.

    ``gw`` is null for season-to-date rows. ``ci_*`` are block-bootstrap-by-GW
    percentile bounds (null when fewer than two gameweeks or for histogram
    bins). ``slice`` names the stratum (``all``, ``position=MID``,
    ``bin=0.0-0.1``, ``line=primary``, …).
    """

    model_id: str
    season: str
    gw: int = pa.Field(ge=1, le=38, nullable=True)
    slice: str
    metric: str = pa.Field(isin=list(AGG_METRICS))
    value: float
    ci_lo: float = pa.Field(nullable=True)
    ci_hi: float = pa.Field(nullable=True)
    n: int = pa.Field(ge=0)
    schema_version: int = pa.Field(ge=1)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


FORECAST_SCORES_COLUMNS: dict[str, pl.DataType] = {
    "model_id": pl.Utf8(),
    "season": pl.Utf8(),
    "gw": pl.Int64(),
    "player_key": pl.Int64(),
    "fixture_id": pl.Int64(),
    "position": pl.Utf8(),
    "price_band": pl.Utf8(),
    "minutes_bucket": pl.Utf8(),
    "y": pl.Int64(),
    "mu_hat": pl.Float64(),
    "rps": pl.Float64(),
    "log_score": pl.Float64(),
    "rps_clim": pl.Float64(),
    "pit_u": pl.Float64(),
    "p_played": pl.Float64(),
    "p_ge60": pl.Float64(),
    "p_goal": pl.Float64(),
    "p_assist": pl.Float64(),
    "p_cs": pl.Float64(),
    "p_haul": pl.Float64(),
    "p_blank": pl.Float64(),
    "y_played": pl.Int64(),
    "y_ge60": pl.Int64(),
    "y_goal": pl.Int64(),
    "y_assist": pl.Int64(),
    "y_cs": pl.Int64(),
    "y_haul": pl.Int64(),
    "y_blank": pl.Int64(),
    "selected_lines": pl.Utf8(),
    "schema_version": pl.Int64(),
}

FORECAST_SCORE_AGG_COLUMNS: dict[str, pl.DataType] = {
    "model_id": pl.Utf8(),
    "season": pl.Utf8(),
    "gw": pl.Int64(),
    "slice": pl.Utf8(),
    "metric": pl.Utf8(),
    "value": pl.Float64(),
    "ci_lo": pl.Float64(),
    "ci_hi": pl.Float64(),
    "n": pl.Int64(),
    "schema_version": pl.Int64(),
}

__all__ = [
    "AGG_METRICS",
    "FORECAST_SCORE_AGG_COLUMNS",
    "FORECAST_SCORE_EVENTS",
    "FORECAST_SCORES_COLUMNS",
    "SCHEMA_VERSION",
    "ForecastScoreAggSchema",
    "ForecastScoresSchema",
]
