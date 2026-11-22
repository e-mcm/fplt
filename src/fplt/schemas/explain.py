"""Pandera schemas for forecast boards and plan decision deltas."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl

_POSITIONS = ["GK", "DEF", "MID", "FWD"]


class ForecastBoardSchema(pa.DataFrameModel):
    """Per-player projected-points board with a first-order event decomposition.

    ``appearance_xpts`` / ``goals_xpts`` / ``assists_xpts`` / ``cs_xpts`` are
    readable approximations from stored event probs and ``Rules``. Multi-event
    outcomes, goals-conceded, and other points land in ``residual_xpts`` so
    ``xpts`` still equals the sum of the five contribution columns.
    """

    player_key: int
    position: str = pa.Field(isin=_POSITIONS)
    target_gw: int = pa.Field(ge=1, le=38)
    xpts: float
    xpts_sd: float = pa.Field(ge=0, nullable=True)
    p_played: float = pa.Field(ge=0, le=1, nullable=True)
    p_60: float = pa.Field(ge=0, le=1, nullable=True)
    p_goal: float = pa.Field(ge=0, le=1, nullable=True)
    p_assist: float = pa.Field(ge=0, le=1, nullable=True)
    p_cs: float = pa.Field(ge=0, le=1, nullable=True)
    p_haul: float = pa.Field(ge=0, le=1, nullable=True)
    appearance_xpts: float = pa.Field(nullable=True)
    goals_xpts: float = pa.Field(nullable=True)
    assists_xpts: float = pa.Field(nullable=True)
    cs_xpts: float = pa.Field(nullable=True)
    residual_xpts: float = pa.Field(nullable=True)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["player_key", "target_gw"]


class TransferDeltaSchema(pa.DataFrameModel):
    """One buy/sell pair with opaque xPts and price deltas for the act-on week."""

    player_key_out: int
    player_key_in: int
    position_out: str = pa.Field(isin=_POSITIONS)
    position_in: str = pa.Field(isin=_POSITIONS)
    xpts_out: float
    xpts_in: float
    delta_xpts: float
    price_out: int = pa.Field(ge=0)
    price_in: int = pa.Field(ge=0)
    delta_price: int

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


FORECAST_BOARD_COLUMNS: dict[str, pl.DataType] = {
    "player_key": pl.Int64(),
    "position": pl.Utf8(),
    "target_gw": pl.Int64(),
    "xpts": pl.Float64(),
    "xpts_sd": pl.Float64(),
    "p_played": pl.Float64(),
    "p_60": pl.Float64(),
    "p_goal": pl.Float64(),
    "p_assist": pl.Float64(),
    "p_cs": pl.Float64(),
    "p_haul": pl.Float64(),
    "appearance_xpts": pl.Float64(),
    "goals_xpts": pl.Float64(),
    "assists_xpts": pl.Float64(),
    "cs_xpts": pl.Float64(),
    "residual_xpts": pl.Float64(),
}

TRANSFER_DELTA_COLUMNS: dict[str, pl.DataType] = {
    "player_key_out": pl.Int64(),
    "player_key_in": pl.Int64(),
    "position_out": pl.Utf8(),
    "position_in": pl.Utf8(),
    "xpts_out": pl.Float64(),
    "xpts_in": pl.Float64(),
    "delta_xpts": pl.Float64(),
    "price_out": pl.Int64(),
    "price_in": pl.Int64(),
    "delta_price": pl.Int64(),
}
