"""L2 run-table standard keys and schema-version constant.

Every L2 parquet under ``runs/run=<run_id>/`` is self-describing: it carries
``run_id`` even though the path already names the run, plus the as-of /
target keys from ``docs/proposals/fplt-data-model.md`` §3.4. Every L1 and L2
table also carries ``schema_version`` so readers can upgrade old files
without rewriting them (§4.2).
"""

from __future__ import annotations

from datetime import datetime

import pandera.polars as pa

# Initial contract version for data-model tables introduced in Stage B2.
# Bump per table when that table's columns change incompatibly; never rewrite
# old files — register an upgrade function instead.
SCHEMA_VERSION = 1


class L2StandardKeys(pa.DataFrameModel):
    """Mixin: standard keys on every player–fixture L2 run table.

    Grain-specific columns (for example ``step`` on plans) are added by the
    concrete table schema. Inherit this mixin rather than re-declaring the
    keys so new L2 tables stay aligned with §3.4.
    """

    run_id: str
    as_of_season: str
    as_of_gw: int = pa.Field(ge=1, le=38)
    deadline_utc: datetime
    target_season: str
    target_gw: int = pa.Field(ge=1, le=38)
    horizon: int = pa.Field(ge=0)
    fixture_id: int
    player_key: int
    team_key: int
    schema_version: int = pa.Field(ge=1)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
