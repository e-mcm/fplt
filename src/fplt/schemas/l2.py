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


class L2AsOfTargetKeys(pa.DataFrameModel):
    """Mixin: as-of / target / fixture keys shared by every L2 run table.

    Player–fixture tables add ``player_key`` via :class:`L2StandardKeys`.
    Team–fixture tables (scoreline) add only ``team_key`` via
    :class:`L2TeamFixtureKeys`.
    """

    run_id: str
    as_of_season: str
    as_of_gw: int = pa.Field(ge=1, le=38)
    deadline_utc: datetime
    target_season: str
    target_gw: int = pa.Field(ge=1, le=38)
    horizon: int = pa.Field(ge=0)
    fixture_id: int
    schema_version: int = pa.Field(ge=1)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


class L2StandardKeys(L2AsOfTargetKeys):
    """Mixin: standard keys on every player–fixture L2 run table.

    Grain-specific columns (for example ``step`` on plans) are added by the
    concrete table schema. Inherit this mixin rather than re-declaring the
    keys so new L2 tables stay aligned with §3.4.
    """

    player_key: int
    team_key: int

    class Config(L2AsOfTargetKeys.Config):
        pass


class L2TeamFixtureKeys(L2AsOfTargetKeys):
    """Mixin: standard keys on every team–fixture L2 run table.

    Used by ``predictions/components/scoreline.parquet`` (two rows per
    fixture). ``player_key`` is absent at this grain.
    """

    team_key: int

    class Config(L2AsOfTargetKeys.Config):
        pass
