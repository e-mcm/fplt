"""Dagster assets over the staged parquet tables.

These assets are not season-gameweek partitioned. ``DataStore`` keeps one
parquet file per table, and ``available_at`` is what makes a row visible.
"""

# Dagster does not ship type stubs.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false

from __future__ import annotations

import dagster as dg
import polars as pl

from fplt.data.raw import ingest_raw
from fplt.data.store import DataStore
from fplt.data.vaastav import import_available
from fplt.schemas.staged import STAGED_SCHEMAS
from fplt.settings import get_settings


@dg.asset(name="raw_snapshots")
def raw_snapshots() -> int:
    """Count gzipped snapshot files already fetched into the data directory."""
    root = get_settings().data_dir / "raw"
    if not root.exists():
        return 0
    return sum(1 for path in root.rglob("*.json.gz") if path.is_file())


@dg.asset(name="staged_ingest", deps=[raw_snapshots])
def staged_ingest() -> int:
    """Ingest raw snapshots and any vaastav season directories."""
    settings = get_settings()
    store = DataStore(settings.data_dir)
    count = ingest_raw(store, settings.data_dir / "raw")
    import_available(store, settings.data_dir)
    return count


def staged_asset(table: str) -> dg.AssetsDefinition:
    """Read one staged table. ``table`` is a key of ``STAGED_SCHEMAS``."""

    def _read() -> pl.DataFrame:
        return DataStore(get_settings().data_dir).read(table)

    _read.__name__ = f"staged_{table}"
    return dg.asset(
        name=f"staged_{table}",
        deps=[staged_ingest],
        description=f"Staged {table} rows.",
    )(_read)


staged_assets = [staged_asset(table) for table in STAGED_SCHEMAS]

refresh_staged = dg.define_asset_job(
    name="refresh_staged",
    selection=dg.AssetSelection.assets(raw_snapshots, staged_ingest, *staged_assets),
)
