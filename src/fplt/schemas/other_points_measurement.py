"""Pandera schemas for match-conditioned other-points measurement reports."""

from __future__ import annotations

import pandera.polars as pa
import polars as pl


class OtherPointsMeasurementSummarySchema(pa.DataFrameModel):
    """One metric row from ``fplt other-measure``.

    ``slice`` encodes filters such as ``all``, ``decision_pool``,
    ``position=MID``, ``top_xpts_decile``, or match-context labels like
    ``match|team_goals=2+|cs=1``.
    """

    model_name: str
    metric: str
    slice: str
    value: float = pa.Field(nullable=True)
    n: int = pa.Field(ge=0)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


class OtherPointsCaptainRegretSchema(pa.DataFrameModel):
    """Per-GW unconstrained captain regret proxy (pool max-xPts vs max-actual)."""

    season: str
    gw: int = pa.Field(ge=1, le=38)
    chosen_player_key: int
    oracle_player_key: int
    chosen_xpts: float
    chosen_actual: float
    oracle_actual: float
    regret: float = pa.Field(ge=0)
    n_pool: int = pa.Field(ge=1)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


OTHER_POINTS_MEASUREMENT_SUMMARY_COLUMNS: dict[str, pl.DataType] = {
    "model_name": pl.Utf8(),
    "metric": pl.Utf8(),
    "slice": pl.Utf8(),
    "value": pl.Float64(),
    "n": pl.Int64(),
}

OTHER_POINTS_CAPTAIN_REGRET_COLUMNS: dict[str, pl.DataType] = {
    "season": pl.Utf8(),
    "gw": pl.Int64(),
    "chosen_player_key": pl.Int64(),
    "oracle_player_key": pl.Int64(),
    "chosen_xpts": pl.Float64(),
    "chosen_actual": pl.Float64(),
    "oracle_actual": pl.Float64(),
    "regret": pl.Float64(),
    "n_pool": pl.Int64(),
}
