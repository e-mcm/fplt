"""Pandera schemas for the Last-week UI page (#396 / #397)."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl


class LastWeekCatalogSchema(pa.DataFrameModel):
    """One succeeded ``purpose=post_gw`` analysis run for the GW selector."""

    run_id: str
    season: str
    gw: int = pa.Field(ge=1, le=38)
    model_id: str
    line: str = pa.Field(nullable=True)
    finished_at: str = pa.Field(nullable=True)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["run_id"]  # noqa: RUF012


class LastWeekLinesStdSchema(pa.DataFrameModel):
    """Season-to-date line totals with chips and ex-chips (#396)."""

    model_id: str
    season: str
    through_gw: int = pa.Field(ge=1, le=38)
    gameweeks: int = pa.Field(ge=0)
    points_with_chips: int
    points_ex_chips: int = pa.Field(nullable=True)
    plan_ev_sum: float = pa.Field(nullable=True)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["model_id", "season", "through_gw"]  # noqa: RUF012


class LastWeekSkillSchema(pa.DataFrameModel):
    """Season-to-date forecast skill / calibration metric with optional CI."""

    metric: str
    label: str
    value: float
    ci_lo: float = pa.Field(nullable=True)
    ci_hi: float = pa.Field(nullable=True)
    n: int = pa.Field(ge=0)
    slice: str

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


class LastWeekPitHistSchema(pa.DataFrameModel):
    """PIT histogram bins for the Plotly ``pit_hist`` renderer."""

    bin_lo: float = pa.Field(ge=0.0, le=1.0)
    bin_hi: float = pa.Field(ge=0.0, le=1.0)
    density: float = pa.Field(ge=0.0)
    label: str

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


class LastWeekReliabilitySchema(pa.DataFrameModel):
    """Event reliability diagram rows (``pred_bin``, ``mean_pred``, ``observed``)."""

    event: str
    pred_bin: str
    mean_pred: float = pa.Field(ge=0.0, le=1.0)
    observed: float = pa.Field(ge=0.0, le=1.0)
    n: int = pa.Field(ge=0)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


class LastWeekSelectionBiasSchema(pa.DataFrameModel):
    """Selection-conditional bias rows (optimiser's curse) with CIs."""

    line: str
    metric: str
    value: float
    ci_lo: float = pa.Field(nullable=True)
    ci_hi: float = pa.Field(nullable=True)
    n: int = pa.Field(ge=0)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


class LastWeekRankContextSchema(pa.DataFrameModel):
    """FPL average / highest and manager rank change (#397 / Spec §2.2 A)."""

    field: str
    label: str
    value: float = pa.Field(nullable=True)
    note: str = pa.Field(nullable=True)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["field"]  # noqa: RUF012


class LastWeekBenchmarksSchema(pa.DataFrameModel):
    """Primary vs ep_next / OpenFPL on shared fixtures (#397 / Spec §2.2 C)."""

    arm: str
    n: int = pa.Field(ge=0)
    rmse: float = pa.Field(nullable=True)
    mae: float = pa.Field(nullable=True)
    mean_error: float = pa.Field(nullable=True)
    note: str = pa.Field(nullable=True)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["arm"]  # noqa: RUF012


class LastWeekOpsReviewSchema(pa.DataFrameModel):
    """Applied checkpoint, freshness, timings, alarms (#397 / Spec §2.2 D)."""

    section: str
    key: str
    value: str = pa.Field(nullable=True)
    detail: str = pa.Field(nullable=True)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


LAST_WEEK_CATALOG_COLUMNS: dict[str, pl.DataType] = {
    "run_id": pl.Utf8(),
    "season": pl.Utf8(),
    "gw": pl.Int64(),
    "model_id": pl.Utf8(),
    "line": pl.Utf8(),
    "finished_at": pl.Utf8(),
}

LAST_WEEK_LINES_STD_COLUMNS: dict[str, pl.DataType] = {
    "model_id": pl.Utf8(),
    "season": pl.Utf8(),
    "through_gw": pl.Int64(),
    "gameweeks": pl.Int64(),
    "points_with_chips": pl.Int64(),
    "points_ex_chips": pl.Int64(),
    "plan_ev_sum": pl.Float64(),
}

LAST_WEEK_SKILL_COLUMNS: dict[str, pl.DataType] = {
    "metric": pl.Utf8(),
    "label": pl.Utf8(),
    "value": pl.Float64(),
    "ci_lo": pl.Float64(),
    "ci_hi": pl.Float64(),
    "n": pl.Int64(),
    "slice": pl.Utf8(),
}

LAST_WEEK_PIT_HIST_COLUMNS: dict[str, pl.DataType] = {
    "bin_lo": pl.Float64(),
    "bin_hi": pl.Float64(),
    "density": pl.Float64(),
    "label": pl.Utf8(),
}

LAST_WEEK_RELIABILITY_COLUMNS: dict[str, pl.DataType] = {
    "event": pl.Utf8(),
    "pred_bin": pl.Utf8(),
    "mean_pred": pl.Float64(),
    "observed": pl.Float64(),
    "n": pl.Int64(),
}

LAST_WEEK_SELECTION_BIAS_COLUMNS: dict[str, pl.DataType] = {
    "line": pl.Utf8(),
    "metric": pl.Utf8(),
    "value": pl.Float64(),
    "ci_lo": pl.Float64(),
    "ci_hi": pl.Float64(),
    "n": pl.Int64(),
}

LAST_WEEK_RANK_CONTEXT_COLUMNS: dict[str, pl.DataType] = {
    "field": pl.Utf8(),
    "label": pl.Utf8(),
    "value": pl.Float64(),
    "note": pl.Utf8(),
}

LAST_WEEK_BENCHMARKS_COLUMNS: dict[str, pl.DataType] = {
    "arm": pl.Utf8(),
    "n": pl.Int64(),
    "rmse": pl.Float64(),
    "mae": pl.Float64(),
    "mean_error": pl.Float64(),
    "note": pl.Utf8(),
}

LAST_WEEK_OPS_REVIEW_COLUMNS: dict[str, pl.DataType] = {
    "section": pl.Utf8(),
    "key": pl.Utf8(),
    "value": pl.Utf8(),
    "detail": pl.Utf8(),
}

__all__ = [
    "LAST_WEEK_BENCHMARKS_COLUMNS",
    "LAST_WEEK_CATALOG_COLUMNS",
    "LAST_WEEK_LINES_STD_COLUMNS",
    "LAST_WEEK_OPS_REVIEW_COLUMNS",
    "LAST_WEEK_PIT_HIST_COLUMNS",
    "LAST_WEEK_RANK_CONTEXT_COLUMNS",
    "LAST_WEEK_RELIABILITY_COLUMNS",
    "LAST_WEEK_SELECTION_BIAS_COLUMNS",
    "LAST_WEEK_SKILL_COLUMNS",
    "LastWeekBenchmarksSchema",
    "LastWeekCatalogSchema",
    "LastWeekLinesStdSchema",
    "LastWeekOpsReviewSchema",
    "LastWeekPitHistSchema",
    "LastWeekRankContextSchema",
    "LastWeekReliabilitySchema",
    "LastWeekSelectionBiasSchema",
    "LastWeekSkillSchema",
]
