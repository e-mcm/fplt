"""Shared fetch → restage → side-ingest steps for ops-tick and ops-catchup."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from fplt.data.api_football import (
    default_id_maps_path as api_football_id_maps_path,
)
from fplt.data.api_football import (
    ingest_api_football,
)
from fplt.data.api_football import (
    load_id_maps as load_api_football_id_maps,
)
from fplt.data.raw import ingest_raw
from fplt.data.snapshots import RawSource, fetch_raw
from fplt.data.store import DataStore
from fplt.data.understat import default_id_maps_path, ingest_understat, load_id_maps
from fplt.data.vaastav import import_available
from fplt.settings import Settings, get_settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FeedRefreshResult:
    """Counts from one ``refresh_ops_feeds`` call.

    Attributes:
        snapshots_fetched: New raw files copied by ``fetch_raw``.
        staged_snapshots: Bootstrap captures ingested by ``ingest_raw``.
        understat_matches: Understat match files staged.
        api_football_captures: API-Football injury captures staged.
    """

    snapshots_fetched: int
    staged_snapshots: int
    understat_matches: int
    api_football_captures: int


def refresh_ops_feeds(
    data_dir: Path | None = None,
    *,
    source: RawSource = "auto",
    remote: str = "origin",
    branch: str = "snapshots",
    settings: Settings | None = None,
) -> FeedRefreshResult:
    """Fetch raw captures and restage FPL / Understat / API-Football feeds.

    Idempotent: already-local raw objects are skipped, and ingest helpers
    no-op when nothing new is present. Used by ``ops-tick`` and
    ``ops-catchup`` so both stay aligned with the Dagster refresh jobs.

    Args:
        data_dir: Root with ``raw/`` and ``staged/``. Defaults to
            ``Settings.data_dir``.
        source: Raw fetch source: ``auto``, ``bucket``, or ``branch``.
        remote: Git remote (or local path in tests) for the snapshots branch.
        branch: Branch that holds write-once captures (default ``snapshots``).
        settings: Optional settings for bucket credentials. Defaults to
            ``get_settings()``.

    Returns:
        Fetch and ingest counts for the operator status print.

    Raises:
        SnapshotFetchError: The snapshots branch could not be cloned, or
            ``source=bucket`` with no data bucket configured.
        RemoteError: An rclone pull against the bucket failed.
    """
    cfg = get_settings() if settings is None else settings
    root = cfg.data_dir if data_dir is None else data_dir
    raw_root = root / "raw"

    logger.info(
        "ops_feeds: fetch_raw source=%s remote=%s branch=%s",
        source,
        remote,
        branch,
    )
    new_files = fetch_raw(
        raw_root,
        cfg,
        source=source,
        remote=remote,
        branch=branch,
    )

    store = DataStore(root)
    logger.info("ops_feeds: refresh staged")
    staged = ingest_raw(store, raw_root)
    import_available(store, root)

    logger.info("ops_feeds: understat_ingest")
    player_keys, team_keys = load_id_maps(default_id_maps_path())
    understat_n = ingest_understat(store, raw_root, player_keys, team_keys)

    logger.info("ops_feeds: api_football_ingest")
    af_players, af_teams = load_api_football_id_maps(api_football_id_maps_path())
    af_n = ingest_api_football(store, raw_root, af_players, af_teams)

    return FeedRefreshResult(
        snapshots_fetched=len(new_files),
        staged_snapshots=staged,
        understat_matches=understat_n,
        api_football_captures=af_n,
    )
