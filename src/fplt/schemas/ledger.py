"""Pandera schema for the prediction ledger."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl


class LedgerSchema(pa.DataFrameModel):
    """Predictions joined to actuals once the target gameweek is checked."""

    model_id: str
    run_kind: str
    as_of_season: str
    as_of_gw: int = pa.Field(ge=1, le=38)
    target_season: str
    target_gw: int = pa.Field(ge=1, le=38)
    horizon: int = pa.Field(ge=1)
    player_key: int
    position: str
    fixture_id: int
    xpts: float
    xpts_sd: float = pa.Field(ge=0, nullable=True)
    p_haul: float = pa.Field(ge=0, le=1, nullable=True)
    p_60: float = pa.Field(ge=0, le=1, nullable=True)
    actual_pts: int = pa.Field(nullable=True)
    actual_minutes: int = pa.Field(nullable=True)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


LEDGER_COLUMNS: dict[str, pl.DataType] = {
    "model_id": pl.Utf8(),
    "run_kind": pl.Utf8(),
    "as_of_season": pl.Utf8(),
    "as_of_gw": pl.Int64(),
    "target_season": pl.Utf8(),
    "target_gw": pl.Int64(),
    "horizon": pl.Int64(),
    "player_key": pl.Int64(),
    "position": pl.Utf8(),
    "fixture_id": pl.Int64(),
    "xpts": pl.Float64(),
    "xpts_sd": pl.Float64(),
    "p_haul": pl.Float64(),
    "p_60": pl.Float64(),
    "actual_pts": pl.Int64(),
    "actual_minutes": pl.Int64(),
}
