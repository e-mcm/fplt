"""Pandera schema for the Understat share feature."""

from __future__ import annotations

import pandera.polars as pa


class UnderstatShareSchema(pa.DataFrameModel):
    """xG and xA rates for one player at one deadline."""

    season: str
    gw: int = pa.Field(ge=1, le=38)
    player_key: int
    xg_per_90: float = pa.Field(ge=0)
    xa_per_90: float = pa.Field(ge=0)
    xg_share: float = pa.Field(ge=0, le=1)
    xa_share: float = pa.Field(ge=0, le=1)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["season", "gw", "player_key"]  # noqa: RUF012
