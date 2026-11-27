"""Pandera schemas for a transfer plan and the decision ledger."""

from __future__ import annotations

import pandera.polars as pa

_POSITIONS = ["GK", "DEF", "MID", "FWD"]


class CandidateSchema(pa.DataFrameModel):
    """One player the planner is allowed to buy or keep."""

    player_key: int
    position: str = pa.Field(isin=_POSITIONS)
    team_key: int
    price: int = pa.Field(ge=0)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["player_key"]  # noqa: RUF012


class ExpectedPointsSchema(pa.DataFrameModel):
    """Opaque expected points for one player in one target gameweek."""

    player_key: int
    gw: int = pa.Field(ge=1, le=38)
    xpts: float

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["player_key", "gw"]  # noqa: RUF012


class HorizonPointsSchema(pa.DataFrameModel):
    """Expected points for one player at one step of the horizon."""

    player_key: int
    step: int = pa.Field(ge=0)
    xpts: float

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["player_key", "step"]  # noqa: RUF012


class PriceForecastSchema(pa.DataFrameModel):
    """Predicted list price for one player at one horizon step.

    Step 0 must match the as-of candidate / ``player_status`` list price.
    Later steps are forecasts built only from deadline-visible inputs.
    """

    player_key: int
    step: int = pa.Field(ge=0)
    list_price: int = pa.Field(ge=0)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["player_key", "step"]  # noqa: RUF012


class HorizonPickSchema(pa.DataFrameModel):
    """A squad member or a sale at one step of the horizon."""

    step: int = pa.Field(ge=0)
    player_key: int
    position: str = pa.Field(isin=_POSITIONS)
    team_key: int
    price: int = pa.Field(ge=0)
    xpts: float
    in_xi: bool
    captain: bool
    vice: bool
    bought: bool
    sold: bool

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["step", "player_key"]  # noqa: RUF012


class PlanSchema(pa.DataFrameModel):
    """A saved plan: one row per player kept or sold in each horizon week."""

    model_id: str
    as_of_season: str
    as_of_gw: int = pa.Field(ge=1, le=38)
    target_gw: int = pa.Field(ge=1, le=38)
    step: int = pa.Field(ge=0)
    player_key: int
    position: str = pa.Field(isin=_POSITIONS)
    team_key: int
    in_xi: bool
    captain: bool
    vice: bool
    bought: bool
    sold: bool
    xpts: float
    hit: float = pa.Field(ge=0)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = [  # noqa: RUF012
            "model_id",
            "as_of_season",
            "as_of_gw",
            "target_gw",
            "player_key",
        ]  # noqa: RUF012


class DecisionSchema(pa.DataFrameModel):
    """One season of squad decisions for one model, at player-gameweek grain."""

    model_id: str
    season: str
    gw: int = pa.Field(ge=1, le=38)
    player_key: int
    position: str = pa.Field(isin=_POSITIONS)
    in_xi: bool
    captain: bool
    vice: bool
    bought: bool
    sold: bool
    xpts: float
    actual_pts: int = pa.Field(nullable=True)
    actual_minutes: int = pa.Field(nullable=True, ge=0)
    hit: float = pa.Field(ge=0)
    free_transfers: int = pa.Field(ge=0)
    chip: str = pa.Field(nullable=True)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["model_id", "season", "gw", "player_key"]  # noqa: RUF012


class ChipWindowsSchema(pa.DataFrameModel):
    """Per-deadline ``ChipUseWindows`` logged beside the decision ledger."""

    model_id: str
    season: str
    gw: int = pa.Field(ge=1, le=38)
    allowed: str
    forced: str
    no_chip: str

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["model_id", "season", "gw"]  # noqa: RUF012
