"""Pandera schemas for the single-gameweek squad selector."""

from __future__ import annotations

import pandera.polars as pa

_POSITIONS = ["GK", "DEF", "MID", "FWD"]


class SquadPlayerSchema(pa.DataFrameModel):
    """One candidate player and the points to optimise for one gameweek."""

    player_key: int
    position: str = pa.Field(isin=_POSITIONS)
    team_key: int
    price: int = pa.Field(ge=0)
    xpts: float

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["player_key"]  # noqa: RUF012


class SquadPickSchema(pa.DataFrameModel):
    """The fifteen selected players, with the eleven and the two armbands."""

    player_key: int
    position: str = pa.Field(isin=_POSITIONS)
    team_key: int
    price: int = pa.Field(ge=0)
    xpts: float
    in_xi: bool
    captain: bool
    vice: bool

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["player_key"]  # noqa: RUF012


class GameweekPointsSchema(pa.DataFrameModel):
    """Player-fixture predictions summed to one player in one gameweek."""

    player_key: int
    xpts: float

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["player_key"]  # noqa: RUF012
