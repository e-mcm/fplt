"""Pandera schema for virtual-line squad snapshots on plan runs.

Plan runs in ``squad_mode=virtual`` write ``plans/squad_state_in.parquet`` and
``plans/squad_state_out.parquet`` so each line's season is an auditable chain
(data model §3.3 / §4.1). One row per snapshot; owned players and chip
inventory are JSON columns matching ``fplt.optimise.state.SquadState``.
"""

from __future__ import annotations

from datetime import datetime

import pandera.polars as pa


class SquadStateTableSchema(pa.DataFrameModel):
    """Squad, bank, free transfers, and chips at plan in or out."""

    run_id: str
    as_of_season: str
    as_of_gw: int = pa.Field(ge=1, le=38)
    deadline_utc: datetime
    schema_version: int = pa.Field(ge=1)
    bank: int = pa.Field(ge=0)
    free_transfers: int = pa.Field(ge=0)
    ft_bank_cap: int = pa.Field(ge=1)
    hit_cost: int = pa.Field(ge=0)
    decay: float = pa.Field(gt=0, le=1)
    bench_weight: float = pa.Field(ge=0)
    owned: str
    chips_remaining: str
    chips_played: str

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["run_id", "as_of_season", "as_of_gw"]  # noqa: RUF012
