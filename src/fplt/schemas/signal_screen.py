"""Pandera schema for the residual signal-screen table (§1.2 toolkit)."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl


class SignalScreenSchema(pa.DataFrameModel):
    """One covariate × component row from the residual signal screen.

    ``effect_size`` is the partial correlation of the residual with the
    covariate controlling for μ̂ and position. ``rejected`` is true when the
    Benjamini–Hochberg q-value is at most the screen's FDR level (default 10%).
    """

    covariate: str
    component: str
    source: str
    effect_size: float = pa.Field(nullable=True)
    se: float = pa.Field(nullable=True, ge=0)
    ci_low: float = pa.Field(nullable=True)
    ci_high: float = pa.Field(nullable=True)
    p_value: float = pa.Field(nullable=True, ge=0, le=1)
    q_value: float = pa.Field(nullable=True, ge=0, le=1)
    n: int = pa.Field(ge=0)
    rank: int = pa.Field(ge=1)
    rejected: bool
    binned_delta: float = pa.Field(nullable=True)
    binned_ci_low: float = pa.Field(nullable=True)
    binned_ci_high: float = pa.Field(nullable=True)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


SIGNAL_SCREEN_COLUMNS: dict[str, pl.DataType] = {
    "covariate": pl.Utf8(),
    "component": pl.Utf8(),
    "source": pl.Utf8(),
    "effect_size": pl.Float64(),
    "se": pl.Float64(),
    "ci_low": pl.Float64(),
    "ci_high": pl.Float64(),
    "p_value": pl.Float64(),
    "q_value": pl.Float64(),
    "n": pl.Int64(),
    "rank": pl.Int64(),
    "rejected": pl.Boolean(),
    "binned_delta": pl.Float64(),
    "binned_ci_low": pl.Float64(),
    "binned_ci_high": pl.Float64(),
}
