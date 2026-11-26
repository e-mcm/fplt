"""Pandera schema for the derived runs catalog (data-model §3.5)."""

from __future__ import annotations

from datetime import datetime

import pandera.polars as pa
import polars as pl


class CatalogRunSchema(pa.DataFrameModel):
    """One row per succeeded run manifest in ``catalog/runs.parquet``.

    Flattened scalars from ``RunManifest`` plus the primary (first) ``as_of``
    deadline. Incomplete runs without ``_SUCCESS`` never appear.
    """

    run_id: str
    kind: str
    mode: str
    purpose: str
    experiment_id: str = pa.Field(nullable=True)
    arm: str = pa.Field(nullable=True)
    run_type: str
    spec_hash: str
    spec_label: str
    model_id: str
    line: str = pa.Field(nullable=True)
    plan_horizon: int = pa.Field(nullable=True, ge=1)
    squad_mode: str = pa.Field(nullable=True)
    prev_run_id: str = pa.Field(nullable=True)
    code_sha: str
    code_dirty: bool
    prod_tag: str = pa.Field(nullable=True)
    fplt_mc_version: str = pa.Field(nullable=True)
    lockfile_hash: str
    build_id: str = pa.Field(nullable=True)
    as_of_season: str
    as_of_gw: int = pa.Field(ge=1, le=38)
    deadline_utc: datetime
    n_as_of: int = pa.Field(ge=1)
    horizon: int = pa.Field(ge=1)
    checkpoint: str = pa.Field(nullable=True)
    host: str
    started_at: datetime
    finished_at: datetime
    status: str
    parent_run_id: str = pa.Field(nullable=True)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


CATALOG_RUN_COLUMNS: dict[str, pl.DataType] = {
    "run_id": pl.Utf8(),
    "kind": pl.Utf8(),
    "mode": pl.Utf8(),
    "purpose": pl.Utf8(),
    "experiment_id": pl.Utf8(),
    "arm": pl.Utf8(),
    "run_type": pl.Utf8(),
    "spec_hash": pl.Utf8(),
    "spec_label": pl.Utf8(),
    "model_id": pl.Utf8(),
    "line": pl.Utf8(),
    "plan_horizon": pl.Int64(),
    "squad_mode": pl.Utf8(),
    "prev_run_id": pl.Utf8(),
    "code_sha": pl.Utf8(),
    "code_dirty": pl.Boolean(),
    "prod_tag": pl.Utf8(),
    "fplt_mc_version": pl.Utf8(),
    "lockfile_hash": pl.Utf8(),
    "build_id": pl.Utf8(),
    "as_of_season": pl.Utf8(),
    "as_of_gw": pl.Int64(),
    "deadline_utc": pl.Datetime(time_zone="UTC"),
    "n_as_of": pl.Int64(),
    "horizon": pl.Int64(),
    "checkpoint": pl.Utf8(),
    "host": pl.Utf8(),
    "started_at": pl.Datetime(time_zone="UTC"),
    "finished_at": pl.Datetime(time_zone="UTC"),
    "status": pl.Utf8(),
    "parent_run_id": pl.Utf8(),
}
