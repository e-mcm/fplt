"""Sensors that pull snapshots and keep staged feeds / plans current."""

# Dagster does not ship type stubs.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import dagster as dg

from fplt.automation.cycle import due_gameweek, pending_checked, plans_exist
from fplt.automation.notify import notify
from fplt.automation.production import load_production_ops
from fplt.automation.runbook import OPS_03_WORKFLOW_FAILURE
from fplt.data.benchmark_projections import benchmarks_path
from fplt.data.football_data import list_e0_csv_paths
from fplt.data.remote import RemoteError
from fplt.data.snapshots import (
    SnapshotFetchError,
    fetch_raw,
    is_football_data_raw,
)
from fplt.data.store import DataStore
from fplt.orchestration.assets import (
    api_football_ingest_job,
    catch_up_job,
    match_odds_ingest_job,
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
        new_files = fetch_raw(dest)
    except (SnapshotFetchError, RemoteError) as exc:
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
    description=(
        "Fetch raw captures (bucket when configured, else snapshots branch) "
        "and restage when new files arrive."
    ),
)
def snapshots_sensor() -> dg.RunRequest | dg.SkipReason:
    """Copy new snapshot files into ``$FPLT_DATA_DIR/raw`` and request a restage.

    Football-data ``E0.csv`` arrivals are copied but do not trigger
    ``refresh_staged`` — ``match_odds_sensor`` stages those.
    """
    dest = get_settings().data_dir / "raw"
    try:
        new_files = fetch_raw(dest)
    except (SnapshotFetchError, RemoteError) as exc:
        return dg.SkipReason(str(exc))
    refresh_files = [path for path in new_files if not is_football_data_raw(path)]
    if not refresh_files:
        if new_files:
            return dg.SkipReason(
                "only football_data snapshots; match_odds_sensor stages odds"
            )
        return dg.SkipReason("no new snapshots")
    return dg.RunRequest(run_key=refresh_files[-1].as_posix())


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
        "Fetch snapshots and stage API-Football injuries / cup fixtures when "
        "raw files arrive (lineups excluded)."
    ),
)
def api_football_sensor(
    context: dg.SensorEvaluationContext,
) -> dg.RunRequest | dg.SkipReason:
    """Request ``api_football_ingest`` when new AF injury/fixture raw files land."""
    return _side_ingest_tick(
        context,
        prefix="api_football",
        empty_message="no new api_football snapshots",
    )


@dg.sensor(
    job=match_odds_ingest_job,
    minimum_interval_seconds=_THIRTY_MINUTES,
    description=(
        "Stage match_odds when a new football-data E0.csv appears under raw "
        "(monthly GHA captured_at=… or local make fetch-football-data)."
    ),
)
def match_odds_sensor(
    context: dg.SensorEvaluationContext,
) -> dg.RunRequest | dg.SkipReason:
    """Request ``match_odds_ingest`` when a new write-once E0 CSV is present.

    Relies on ``snapshots_sensor`` (or a local fetch) to place files under
    ``$FPLT_DATA_DIR/raw/football_data``. Tracks seen paths in the sensor cursor.
    """
    raw_root = get_settings().data_dir / "raw"
    paths = [path.as_posix() for path in list_e0_csv_paths(raw_root)]
    previous = _cursor_paths(context.cursor)
    new_paths = [path for path in paths if path not in previous]
    context.update_cursor("\n".join(paths))
    if not new_paths:
        return dg.SkipReason("no new football_data E0.csv")
    return dg.RunRequest(run_key=new_paths[-1])


def _cursor_paths(cursor: str | None) -> set[str]:
    if not cursor:
        return set()
    return {line for line in cursor.split("\n") if line}


@dg.sensor(
    job=catch_up_job,
    minimum_interval_seconds=_THIRTY_MINUTES,
    description="Plan every checked gameweek that does not have a plan yet.",
)
def catchup_sensor() -> dg.RunRequest | dg.SkipReason:
    """Request a catch-up when a checked gameweek has not been planned."""
    data = get_settings().data_dir
    ops = load_production_ops()
    waiting = pending_checked(
        DataStore(data), data, ops.models, horizon=ops.plan_horizon
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
    ops = load_production_ops()
    key = due_gameweek(DataStore(data), datetime.now(UTC))
    if key is None:
        return dg.SkipReason("no deadline inside three hours")
    planned = plans_exist(data, ops.models, key, horizon=ops.plan_horizon)
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
    notify(
        url,
        "fplt run failed",
        message,
        runbook_anchor=OPS_03_WORKFLOW_FAILURE,
    )
    return None
