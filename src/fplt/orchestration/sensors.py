"""Sensor that pulls new files from the snapshots branch."""

# Dagster does not ship type stubs.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false

from __future__ import annotations

import dagster as dg

from fplt.data.snapshots import SnapshotFetchError, fetch_snapshots
from fplt.orchestration.assets import refresh_staged
from fplt.settings import get_settings

_TEN_MINUTES = 600


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
