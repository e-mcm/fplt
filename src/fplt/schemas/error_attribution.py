"""Pandera schema for one-at-a-time oracle-gain error attribution (toolkit §1.1)."""

from __future__ import annotations

from typing import Literal

import pandera.polars as pa
import polars as pl

OracleComponent = Literal["minutes", "scoreline", "involvement", "other"]
AttributionMetric = Literal["rps", "log_score", "mse_mean"]
PriceBand = Literal["budget", "mid", "premium"]

ORACLE_COMPONENTS: tuple[str, ...] = (
    "minutes",
    "scoreline",
    "involvement",
    "other",
)
ATTRIBUTION_METRICS: tuple[str, ...] = ("rps", "log_score", "mse_mean")
PRICE_BANDS: tuple[str, ...] = ("budget", "mid", "premium")


class ErrorAttributionSchema(pa.DataFrameModel):
    """Oracle gain by season / GW / position / price band.

    ``gain`` is ``score(control) − score(oracle_c)`` (lower-is-better scores).
    ``gw`` is null for season-to-date rows; ``ci_*`` are block-bootstrap-by-GW
    percentile bounds (null when fewer than two gameweeks).
    """

    model_id: str
    season: str
    gw: int = pa.Field(ge=1, le=38, nullable=True)
    position: str
    price_band: str = pa.Field(isin=list(PRICE_BANDS))
    component: str = pa.Field(isin=list(ORACLE_COMPONENTS))
    metric: str = pa.Field(isin=list(ATTRIBUTION_METRICS))
    gain: float
    ci_lo: float = pa.Field(nullable=True)
    ci_hi: float = pa.Field(nullable=True)
    n: int = pa.Field(ge=0)
    schema_version: int = pa.Field(ge=1)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


ERROR_ATTRIBUTION_COLUMNS: dict[str, pl.DataType] = {
    "model_id": pl.Utf8(),
    "season": pl.Utf8(),
    "gw": pl.Int64(),
    "position": pl.Utf8(),
    "price_band": pl.Utf8(),
    "component": pl.Utf8(),
    "metric": pl.Utf8(),
    "gain": pl.Float64(),
    "ci_lo": pl.Float64(),
    "ci_hi": pl.Float64(),
    "n": pl.Int64(),
    "schema_version": pl.Int64(),
}
