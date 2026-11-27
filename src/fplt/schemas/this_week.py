"""Pandera schemas for ``fplt.analysis.this_week`` UI tables (§3.3)."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl

_POSITIONS = ["GK", "DEF", "MID", "FWD"]


class ThisWeekPlayersSchema(pa.DataFrameModel):
    """One row per player for the This-week players table.

    Horizon gameweek columns are added dynamically as ``xpts_gw{N}``;
    ``total_xpts`` sums the selected horizon. Missing curated names use a
    placeholder ``web_name`` so prediction-only players are not dropped.
    """

    player_key: int
    web_name: str
    team: str
    position: str = pa.Field(isin=_POSITIONS)
    price: float = pa.Field(nullable=True, ge=0)
    status: str = pa.Field(nullable=True)
    news: str = pa.Field(nullable=True)
    p_played: float = pa.Field(nullable=True, ge=0, le=1)
    total_xpts: float
    in_squad: bool

    class Config(pa.DataFrameModel.Config):
        strict = False
        coerce = True
        unique = ["player_key"]  # noqa: RUF012


class ThisWeekTransferSchema(pa.DataFrameModel):
    """One planned transfer pair (out → in) for step 0."""

    player_key_out: int
    player_key_in: int
    name_out: str
    name_in: str
    price_out: float = pa.Field(nullable=True)
    price_in: float = pa.Field(nullable=True)
    xpts_out: float
    xpts_in: float
    delta_xpts: float

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


class ThisWeekFixtureDetailSchema(pa.DataFrameModel):
    """Per-fixture drill-down row for one player across the horizon."""

    player_key: int
    target_gw: int = pa.Field(ge=1, le=38)
    fixture_id: int
    opponent: str = pa.Field(nullable=True)
    is_home: bool = pa.Field(nullable=True)
    xmins: float = pa.Field(nullable=True, ge=0)
    p_played: float = pa.Field(nullable=True, ge=0, le=1)
    p_60: float = pa.Field(nullable=True, ge=0, le=1)
    team_eg: float = pa.Field(nullable=True, ge=0)
    opp_eg: float = pa.Field(nullable=True, ge=0)
    p_cs: float = pa.Field(nullable=True, ge=0, le=1)
    p_goal: float = pa.Field(nullable=True, ge=0, le=1)
    p_assist: float = pa.Field(nullable=True, ge=0, le=1)
    exp_goals: float = pa.Field(nullable=True, ge=0)
    exp_assists: float = pa.Field(nullable=True, ge=0)
    xpts: float

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["player_key", "fixture_id"]  # noqa: RUF012


THIS_WEEK_PLAYERS_BASE_COLUMNS: dict[str, pl.DataType] = {
    "player_key": pl.Int64(),
    "web_name": pl.Utf8(),
    "team": pl.Utf8(),
    "position": pl.Utf8(),
    "price": pl.Float64(),
    "status": pl.Utf8(),
    "news": pl.Utf8(),
    "p_played": pl.Float64(),
    "total_xpts": pl.Float64(),
    "in_squad": pl.Boolean(),
}

THIS_WEEK_TRANSFER_COLUMNS: dict[str, pl.DataType] = {
    "player_key_out": pl.Int64(),
    "player_key_in": pl.Int64(),
    "name_out": pl.Utf8(),
    "name_in": pl.Utf8(),
    "price_out": pl.Float64(),
    "price_in": pl.Float64(),
    "xpts_out": pl.Float64(),
    "xpts_in": pl.Float64(),
    "delta_xpts": pl.Float64(),
}

THIS_WEEK_FIXTURE_DETAIL_COLUMNS: dict[str, pl.DataType] = {
    "player_key": pl.Int64(),
    "target_gw": pl.Int64(),
    "fixture_id": pl.Int64(),
    "opponent": pl.Utf8(),
    "is_home": pl.Boolean(),
    "xmins": pl.Float64(),
    "p_played": pl.Float64(),
    "p_60": pl.Float64(),
    "team_eg": pl.Float64(),
    "opp_eg": pl.Float64(),
    "p_cs": pl.Float64(),
    "p_goal": pl.Float64(),
    "p_assist": pl.Float64(),
    "exp_goals": pl.Float64(),
    "exp_assists": pl.Float64(),
    "xpts": pl.Float64(),
}
