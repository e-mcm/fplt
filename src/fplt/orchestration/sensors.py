"""Sensors that pull snapshots and keep staged feeds / plans current."""

# Dagster does not ship type stubs.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import dagster as dg

from fplt.automation.cycle import (
    PRODUCTION_MODELS,
    due_gameweek,
    pending_checked,
    plans_exist,
)
from fplt.automation.notify import notify
from fplt.data.benchmark_projections import benchmarks_path
from fplt.data.snapshots import SnapshotFetchError, fetch_snapshots
from fplt.data.store import DataStore
from fplt.optimise.plan import DEFAULT_PLAN_HORIZON
from fplt.orchestration.assets import (
    api_football_ingest_job,
    catch_up_job,
    pre_deadline_job,
    refresh_staged,
    understat_ingest_job,
)
from fplt.settings import get_settings

_TEN_MINUTES = 600
_FIFTEEN_MINUTES = 900
_THIRTY_MINUTES = 1800


def _prefix_hits(dest: Path, prefix: str, new_files: list[Path]) -> list[Path]:
    """Return newly fetched paths under ``dest/prefix``.

    API-Football lineup captures share the ``api_football`` tree but are staged
    by ``predicted_lineups_ingest``, so they are excluded here.
    """
    root = dest / prefix
    hits: list[Path] = []
    for path in new_files:
        if not path.is_relative_to(root):
            continue
        relative = path.relative_to(root)
        if prefix == "api_football" and "lineups" in relative.parts:
            continue
        hits.append(path)
    return hits


def _latest_raw(dest: Path, prefix: str) -> Path | None:
    """Newest ``*.json.gz`` under ``dest/prefix``, excluding AF lineups."""
    root = dest / prefix
    if not root.exists():
        return None
    files: list[Path] = []
    for path in sorted(root.rglob("*.json.gz")):
        if not path.is_file():
            continue
        relative = path.relative_to(root)
        if prefix == "api_football" and "lineups" in relative.parts:
            continue
        files.append(path)
    return files[-1] if files else None


def _side_ingest_tick(
    context: dg.SensorEvaluationContext,
    *,
    prefix: str,
    empty_message: str,
) -> dg.RunRequest | dg.SkipReason:
    """Fetch snapshots and request a run when ``prefix`` raw files are new.

    ``snapshots_sensor`` may already have copied the same files in this tick
    window; when fetch returns nothing for the prefix, fall back to the latest
    on-disk path and Dagster's sensor cursor so the ingest is not skipped.
    """
    dest = get_settings().data_dir / "raw"
    try:
        new_files = fetch_snapshots(dest)
    except SnapshotFetchError as exc:
        return dg.SkipReason(str(exc))
    hits = _prefix_hits(dest, prefix, new_files)
    if hits:
        latest = hits[-1]
        context.update_cursor(latest.as_posix())
        return dg.RunRequest(run_key=latest.as_posix())
    latest_local = _latest_raw(dest, prefix)
    if latest_local is None:
        return dg.SkipReason(empty_message)
    key = latest_local.as_posix()
    if context.cursor == key:
        return dg.SkipReason(empty_message)
    context.update_cursor(key)
    return dg.RunRequest(run_key=key)


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
    job=understat_ingest_job,
    minimum_interval_seconds=_TEN_MINUTES,
    description="Fetch snapshots and stage Understat when understat raw files arrive.",
)
def understat_sensor(
    context: dg.SensorEvaluationContext,
) -> dg.RunRequest | dg.SkipReason:
    """Request ``understat_ingest`` when new Understat raw files land."""
    return _side_ingest_tick(
        context, prefix="understat", empty_message="no new understat snapshots"
    )


@dg.sensor(
    job=api_football_ingest_job,
    minimum_interval_seconds=_TEN_MINUTES,
    description=(
        "Fetch snapshots and stage API-Football injuries when injury raw files arrive."
    ),
)
def api_football_sensor(
    context: dg.SensorEvaluationContext,
) -> dg.RunRequest | dg.SkipReason:
    """Request ``api_football_ingest`` when new injury raw files land."""
    return _side_ingest_tick(
        context,
        prefix="api_football",
        empty_message="no new api_football snapshots",
    )


@dg.sensor(
    job=catch_up_job,
    minimum_interval_seconds=_THIRTY_MINUTES,
    description="Plan every checked gameweek that does not have a plan yet.",
)
def catchup_sensor() -> dg.RunRequest | dg.SkipReason:
    """Request a catch-up when a checked gameweek has not been planned."""
    data = get_settings().data_dir
    waiting = pending_checked(
        DataStore(data), data, PRODUCTION_MODELS, horizon=DEFAULT_PLAN_HORIZON
    )
    if not waiting:
        return dg.SkipReason("no checked gameweek is waiting")
    first = waiting[0]
    return dg.RunRequest(run_key=f"catchup-{first.season}-{first.gw}")


@dg.sensor(
    job=pre_deadline_job,
    minimum_interval_seconds=_FIFTEEN_MINUTES,
    description=(
        "Re-plan and archive benchmark_projections for the deadline "
        "inside the next three hours."
    ),
)
def pre_deadline_sensor() -> dg.RunRequest | dg.SkipReason:
    """Request a refresh when plan or benchmark archive is still missing."""
    data = get_settings().data_dir
    key = due_gameweek(DataStore(data), datetime.now(UTC))
    if key is None:
        return dg.SkipReason("no deadline inside three hours")
    planned = plans_exist(data, PRODUCTION_MODELS, key, horizon=DEFAULT_PLAN_HORIZON)
    archived = benchmarks_path(data, key).exists()
    if planned and archived:
        return dg.SkipReason("plan and benchmark_projections already saved")
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
