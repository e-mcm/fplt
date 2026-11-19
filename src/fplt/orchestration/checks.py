"""Blocking pandera checks for the staged assets."""

# Dagster does not ship type stubs.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false

from __future__ import annotations

import dagster as dg
import polars as pl
from pandera.errors import SchemaError

from fplt.data.store import DataStore
from fplt.orchestration.assets import staged_assets
from fplt.schemas.staged import STAGED_SCHEMAS
from fplt.settings import get_settings


def schema_check_result(table: str, frame: pl.DataFrame) -> dg.AssetCheckResult:
    """Validate ``frame`` against the staged schema for ``table``.

    Args:
        table: A key of ``STAGED_SCHEMAS``.
        frame: The materialised table.

    Returns:
        A passing check, or a failing check whose metadata holds the first
        line of the validation error.
    """
    try:
        STAGED_SCHEMAS[table].validate(frame)
    except (SchemaError, pl.exceptions.PolarsError) as exc:
        message = str(exc).splitlines()[0][:500]
        return dg.AssetCheckResult(
            passed=False,
            metadata={"failure": dg.MetadataValue.text(message)},
        )
    return dg.AssetCheckResult(
        passed=True,
        metadata={"rows": dg.MetadataValue.int(frame.height)},
    )


def pandera_check(asset: dg.AssetsDefinition, table: str) -> dg.AssetChecksDefinition:
    """Build a blocking schema check for one staged asset."""

    def _check() -> dg.AssetCheckResult:
        frame = DataStore(get_settings().data_dir).read(table)
        return schema_check_result(table, frame)

    _check.__name__ = f"{table}_schema"
    return dg.asset_check(asset=asset, blocking=True, name=f"{table}_schema")(_check)


staged_checks = [
    pandera_check(asset, table)
    for asset, table in zip(staged_assets, STAGED_SCHEMAS, strict=True)
]
