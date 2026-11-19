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


class ComponentSchema(pa.DataFrameModel):
    """Scoreline probabilities for one fixture, with actuals after the check."""

    model_id: str
    run_kind: str
    as_of_season: str
    as_of_gw: int = pa.Field(ge=1, le=38)
    target_season: str
    target_gw: int = pa.Field(ge=1, le=38)
    fixture_id: int
    team_h_key: int
    team_a_key: int
    p_home: float = pa.Field(ge=0, le=1)
    p_draw: float = pa.Field(ge=0, le=1)
    p_away: float = pa.Field(ge=0, le=1)
    expected_home: float = pa.Field(ge=0)
    expected_away: float = pa.Field(ge=0)
    actual_home: int = pa.Field(nullable=True, ge=0)
    actual_away: int = pa.Field(nullable=True, ge=0)
    rps: float = pa.Field(nullable=True, ge=0)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


COMPONENT_COLUMNS: dict[str, pl.DataType] = {
    "model_id": pl.Utf8(),
    "run_kind": pl.Utf8(),
    "as_of_season": pl.Utf8(),
    "as_of_gw": pl.Int64(),
    "target_season": pl.Utf8(),
    "target_gw": pl.Int64(),
    "fixture_id": pl.Int64(),
    "team_h_key": pl.Int64(),
    "team_a_key": pl.Int64(),
    "p_home": pl.Float64(),
    "p_draw": pl.Float64(),
    "p_away": pl.Float64(),
    "expected_home": pl.Float64(),
    "expected_away": pl.Float64(),
    "actual_home": pl.Int64(),
    "actual_away": pl.Int64(),
    "rps": pl.Float64(),
}
