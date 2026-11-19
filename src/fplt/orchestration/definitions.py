"""Dagster entrypoint. ``make dagster-dev`` loads ``defs`` from this package."""

# Dagster does not ship type stubs.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import dagster as dg

from fplt.orchestration.assets import (
    raw_snapshots,
    refresh_staged,
    staged_assets,
    staged_ingest,
)
from fplt.orchestration.checks import staged_checks
from fplt.orchestration.sensors import snapshots_sensor

defs = dg.Definitions(
    assets=[raw_snapshots, staged_ingest, *staged_assets],
    asset_checks=staged_checks,
    jobs=[refresh_staged],
    sensors=[snapshots_sensor],
)
