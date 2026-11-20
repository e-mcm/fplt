"""Sensor that pulls new files from the snapshots branch."""

# Dagster does not ship type stubs.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false

from __future__ import annotations

from datetime import UTC, datetime

import dagster as dg

from fplt.automation.cycle import (
    PRODUCTION_MODELS,
    due_gameweek,
    pending_checked,
    plans_exist,
)
from fplt.automation.notify import notify
from fplt.data.snapshots import SnapshotFetchError, fetch_snapshots
from fplt.data.store import DataStore
from fplt.orchestration.assets import catch_up_job, pre_deadline_job, refresh_staged
from fplt.settings import get_settings

_TEN_MINUTES = 600
_FIFTEEN_MINUTES = 900
_THIRTY_MINUTES = 1800


@dg.sensor(
    job=refresh_staged,
    minimum_interval_seconds=_TEN_MINUTES,
    description="Fetch the snapshots branch and restage when new files arrive.",
)
def snapshots_sensor() -> dg.RunRequest | dg.SkipReason:
    """Copy new snapshot files into ``$FPLT_DATA_DIR/raw`` and request a restage."""
    dest = get_settings().data_dir / "raw"
    try:
        new_files = fetch_snapshots(dest)
    except SnapshotFetchError as exc:
        return dg.SkipReason(str(exc))
    if not new_files:
        return dg.SkipReason("no new snapshots")
    return dg.RunRequest(run_key=new_files[-1].as_posix())


@dg.sensor(
    job=catch_up_job,
    minimum_interval_seconds=_THIRTY_MINUTES,
    description="Plan every checked gameweek that does not have a plan yet.",
)
def catchup_sensor() -> dg.RunRequest | dg.SkipReason:
    """Request a catch-up when a checked gameweek has not been planned."""
    data = get_settings().data_dir
    waiting = pending_checked(DataStore(data), data, PRODUCTION_MODELS)
    if not waiting:
        return dg.SkipReason("no checked gameweek is waiting")
    first = waiting[0]
    return dg.RunRequest(run_key=f"catchup-{first.season}-{first.gw}")


@dg.sensor(
    job=pre_deadline_job,
    minimum_interval_seconds=_FIFTEEN_MINUTES,
    description="Re-plan the deadline that falls inside the next three hours.",
)
def pre_deadline_sensor() -> dg.RunRequest | dg.SkipReason:
    """Request a refresh when a deadline is inside three hours and unplanned."""
    data = get_settings().data_dir
    key = due_gameweek(DataStore(data), datetime.now(UTC))
    if key is None or plans_exist(data, PRODUCTION_MODELS, key):
        return dg.SkipReason("no deadline inside three hours")
    return dg.RunRequest(run_key=f"deadline-{key.season}-{key.gw}")


@dg.run_failure_sensor(name="notify_run_failure")
def notify_run_failure(context: dg.RunFailureSensorContext) -> dg.SkipReason | None:
    """POST the failure to ``FPLT_NTFY_URL`` when that setting is present."""
    url = get_settings().ntfy_url
    if not url:
        return dg.SkipReason("FPLT_NTFY_URL is not set")
    message = context.failure_event.message or "run failed"
    notify(url, "fplt run failed", message)
    return None
