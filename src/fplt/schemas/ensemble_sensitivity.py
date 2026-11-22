"""Pandera schemas for ensemble-sensitivity (move-frequency) reporting."""

from __future__ import annotations

import pandera.polars as pa


class EnsembleSensitivityRowSchema(pa.DataFrameModel):
    """One gameweek of control vs challenger week-1 action comparison.

    Shared squad state: both arms solve from the same pre-deadline squad;
    the control plan advances ownership. ``move_frequency`` is reported on
    the summary (1 − week-1 agree rate), not per row.
    """

    season: str
    gw: int = pa.Field(ge=1, le=38)
    control_model_id: str
    challenger_model_id: str
    week1_agree: bool
    transfer_agree: bool
    xi_agree: bool
    captain_agree: bool
    vice_agree: bool
    n_transfers_control: int = pa.Field(ge=0)
    n_transfers_challenger: int = pa.Field(ge=0)
    n_transfer_in_symmetric_diff: int = pa.Field(ge=0)
    hit_control: float = pa.Field(ge=0)
    hit_challenger: float = pa.Field(ge=0)
    horizon: int = pa.Field(ge=1)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["season", "gw", "control_model_id", "challenger_model_id"]


class EnsembleSensitivitySummarySchema(pa.DataFrameModel):
    """Season-level move-frequency board for one control × challenger pair."""

    control_model_id: str
    challenger_model_id: str
    n_gw: int = pa.Field(ge=0)
    week1_agree_rate: float
    transfer_agree_rate: float
    xi_agree_rate: float
    captain_agree_rate: float
    vice_agree_rate: float
    move_frequency: float
    transfer_move_frequency: float
    mean_transfer_in_symmetric_diff: float
    mean_abs_hit_diff: float

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["control_model_id", "challenger_model_id"]
