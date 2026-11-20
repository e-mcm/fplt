"""Dagster entrypoint. ``make dagster-dev`` loads ``defs`` from this package."""

# Dagster does not ship type stubs.
# pyright: reportUnknownMemberType=false, reportUnknownArgumentType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

import dagster as dg

from fplt.orchestration.assets import (
    catch_up_job,
    pre_deadline_job,
    pre_deadline_plan,
    raw_snapshots,
    recommendation_report,
    refresh_staged,
    staged_assets,
    staged_ingest,
    understat_ingest,
    understat_ingest_job,
)
from fplt.orchestration.checks import staged_checks
from fplt.orchestration.sensors import (
    catchup_sensor,
    notify_run_failure,
    pre_deadline_sensor,
    snapshots_sensor,
)

defs = dg.Definitions(
    assets=[
        raw_snapshots,
        staged_ingest,
        *staged_assets,
        understat_ingest,
        recommendation_report,
        pre_deadline_plan,
    ],
    asset_checks=staged_checks,
    jobs=[refresh_staged, understat_ingest_job, catch_up_job, pre_deadline_job],
    sensors=[
        snapshots_sensor,
        catchup_sensor,
        pre_deadline_sensor,
        notify_run_failure,
    ],
)
