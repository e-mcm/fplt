"""Pandera schemas for the forecast × optimiser 2×2 swap harness."""

from __future__ import annotations

import pandera.polars as pa


class SwapProjectionSchema(pa.DataFrameModel):
    """Opaque expected points labelled by forecast source.

    Both arms of the forecast axis (ours vs benchmark) share this frame.
    ``source`` is a free label (``ours``, ``benchmark``, ``ep_next``, …);
    the harness does not fetch live FPL Review data.
    """

    source: str
    as_of_gw: int = pa.Field(ge=1, le=38)
    target_gw: int = pa.Field(ge=1, le=38)
    player_key: int
    xpts: float

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["source", "as_of_gw", "target_gw", "player_key"]


class SeasonPointsSchema(pa.DataFrameModel):
    """Season totals for one forecast × optimiser arm.

    Suitable as the decisions-suite readout for promote-eval later. Component
    regret columns stay nullable until the N5 attainable/regret work lands.
    """

    arm_id: str
    forecast_source: str
    optimiser_id: str
    season: str
    total_points: int
    gameweeks: int
    mean_ppg: float
    per_gw_json: str
    captaincy_regret: float = pa.Field(nullable=True)
    xi_regret: float = pa.Field(nullable=True)
    transfer_regret: float = pa.Field(nullable=True)
    chip_regret: float = pa.Field(nullable=True)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["arm_id", "season"]
