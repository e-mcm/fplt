"""Dagster assets over the staged parquet tables.

These assets are not season-gameweek partitioned. ``DataStore`` keeps one
parquet file per table, and ``available_at`` is what makes a row visible.
"""

# Dagster does not ship type stubs.
# pyright: reportUnknownMemberType=false, reportUnknownVariableType=false
# pyright: reportUnknownArgumentType=false, reportUnknownParameterType=false
# pyright: reportMissingTypeArgument=false

from __future__ import annotations

from datetime import UTC, datetime

import dagster as dg
import polars as pl

from fplt.automation.cycle import PRODUCTION_MODELS, catch_up, pre_deadline, reports
from fplt.automation.notify import notify_recommendation
from fplt.data.football_data import (
    default_team_map_path,
    ingest_football_data,
    load_team_names,
)
from fplt.data.raw import ingest_raw
from fplt.data.store import DataStore
from fplt.data.understat import default_id_maps_path, ingest_understat, load_id_maps
from fplt.data.vaastav import import_available
from fplt.optimise.plan import DEFAULT_PLAN_HORIZON
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


@dg.asset(name="recommendation_report")
def recommendation_report() -> dg.MaterializeResult:
    """Plan every checked gameweek that is still missing one.

    The markdown for those plans is attached as asset metadata.
    """
    settings = get_settings()
    data = settings.data_dir
    done = catch_up(data, PRODUCTION_MODELS, horizon=DEFAULT_PLAN_HORIZON)
    text = reports(data) or "No plan yet."
    if done:
        notify_recommendation(settings, "fplt recommendation", text)
    processed = ", ".join(f"{key.season} gw{key.gw}" for key in done) or "none"
    return dg.MaterializeResult(
        metadata={
            "processed": processed,
            "report": dg.MetadataValue.md(text),
        }
    )


@dg.asset(name="pre_deadline_plan")
def pre_deadline_plan() -> dg.MaterializeResult:
    """Plan the deadline inside the next three hours, if it has no plan yet."""
    settings = get_settings()
    data = settings.data_dir
    key = pre_deadline(
        data, datetime.now(UTC), PRODUCTION_MODELS, horizon=DEFAULT_PLAN_HORIZON
    )
    label = "none" if key is None else f"{key.season} gw{key.gw}"
    text = reports(data) or "No plan yet."
    if key is not None:
        notify_recommendation(settings, "fplt recommendation", text)
    return dg.MaterializeResult(
        metadata={"deadline": label, "report": dg.MetadataValue.md(text)}
    )


catch_up_job = dg.define_asset_job(
    name="catch_up",
    selection=dg.AssetSelection.assets(recommendation_report),
)
pre_deadline_job = dg.define_asset_job(
    name="pre_deadline",
    selection=dg.AssetSelection.assets(pre_deadline_plan),
)


@dg.asset(name="understat_ingest")
def understat_ingest() -> int:
    """Stage Understat player xG and team Deep/PPDA from write-once raw JSON.

    Reads ``$FPLT_DATA_DIR/raw/understat/...`` (match payloads + league
    ``teamsData`` snapshots) and the repo id maps at
    ``configs/understat/id_maps.yaml``. Does not touch FPL snapshot ingest.
    """
    settings = get_settings()
    player_keys, team_keys = load_id_maps(default_id_maps_path())
    return ingest_understat(
        DataStore(settings.data_dir),
        settings.data_dir / "raw",
        player_keys,
        team_keys,
    )


understat_ingest_job = dg.define_asset_job(
    name="understat_ingest",
    selection=dg.AssetSelection.assets(understat_ingest),
)


@dg.asset(name="match_odds_ingest")
def match_odds_ingest() -> int:
    """Stage football-data E0 closing odds from write-once raw CSVs.

    Reads ``$FPLT_DATA_DIR/raw/football_data/season=*/E0.csv`` and the repo
    team map at ``configs/football_data/team_names.yaml``. Rows are labelled
    ``odds_kind=closing_proxy`` with ``available_at=kickoff`` (close ≠
    deadline). Does not touch FPL snapshot ingest.
    """
    settings = get_settings()
    team_keys = load_team_names(default_team_map_path())
    return ingest_football_data(
        DataStore(settings.data_dir),
        settings.data_dir / "raw",
        team_keys,
    )


match_odds_ingest_job = dg.define_asset_job(
    name="match_odds_ingest",
    selection=dg.AssetSelection.assets(match_odds_ingest),
)
