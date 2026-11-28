"""Read write-once gzipped API snapshots and append them to the store."""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl

from fplt.data.quarantine import capture_relpath, quarantined_capture_paths
from fplt.data.staging import stage_bootstrap, stage_event_live
from fplt.data.store import DataStore
from fplt.utils.timing import (
    TimingSession,
    active_timing_session,
    format_kind_summary,
    timed_phase,
    timing_artifacts,
    timing_session,
    write_timing_record,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class RawSnapshot:
    """One captured bootstrap, with fixtures beside it when present."""

    season: str
    captured_at: datetime
    bootstrap_path: Path
    fixtures_path: Path | None

    @property
    def capture_dir(self) -> Path:
        """Directory holding this capture's bootstrap (and fixtures)."""
        return self.bootstrap_path.parent


@dataclass(frozen=True)
class EventLiveCapture:
    """One ``event/{gw}/live`` capture under ``raw/fpl_api``."""

    season: str
    gw: int
    captured_at: datetime
    path: Path

    @property
    def capture_dir(self) -> Path:
        """Directory holding this ``event_live.json.gz``."""
        return self.path.parent


def _stamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H-%M-%SZ").replace(tzinfo=UTC)


def iter_snapshots(raw_root: Path) -> list[RawSnapshot]:
    """Bootstrap captures under ``raw_root/fpl_api``, oldest first."""
    found: list[RawSnapshot] = []
    base = raw_root / "fpl_api"
    if not base.exists():
        return found
    for path in sorted(base.glob("season=*/captured_at=*/bootstrap_static.json.gz")):
        season = path.parents[1].name.removeprefix("season=")
        captured = _stamp(path.parent.name.removeprefix("captured_at="))
        fixtures = path.parent / "fixtures.json.gz"
        found.append(
            RawSnapshot(
                season=season,
                captured_at=captured,
                bootstrap_path=path,
                fixtures_path=fixtures if fixtures.exists() else None,
            )
        )
    return found


def iter_event_live(raw_root: Path) -> list[EventLiveCapture]:
    """``event_live`` captures under ``raw_root/fpl_api``, oldest first."""
    found: list[EventLiveCapture] = []
    base = raw_root / "fpl_api"
    if not base.exists():
        return found
    for path in sorted(
        base.glob("season=*/event_live/gw=*/captured_at=*/event_live.json.gz")
    ):
        season = path.parents[3].name.removeprefix("season=")
        gw = int(path.parents[1].name.removeprefix("gw="))
        captured = _stamp(path.parent.name.removeprefix("captured_at="))
        found.append(
            EventLiveCapture(
                season=season,
                gw=gw,
                captured_at=captured,
                path=path,
            )
        )
    return found


def read_json(path: Path) -> Any:
    """Read one gzipped JSON file."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def ingest_raw(
    store: DataStore,
    raw_root: Path,
    *,
    snapshots: Sequence[RawSnapshot] | None = None,
    event_live_after: datetime | None = None,
) -> int:
    """Stage bootstrap snapshots and checked live payloads.

    Captures listed in ``store.root/quarantine/`` (ops-06) are skipped so a
    quarantined raw object never reaches ``staged/``. ``event_live`` files are
    staged once each (not once per bootstrap), optionally restricted to
    captures newer than ``event_live_after``.

    Args:
        store: Destination staged store.
        raw_root: Local ``raw/`` tree.
        snapshots: Explicit bootstrap list; defaults to every snapshot under
            ``raw_root``.
        event_live_after: When set, only stage ``event_live`` captures with
            ``captured_at`` strictly after this stamp. ``None`` stages all.

    Returns:
        Number of bootstrap + event_live files consumed.
    """
    skipped = quarantined_capture_paths(store.root)
    outer = active_timing_session()
    if outer is not None:
        with timed_phase("ingest"):
            return _ingest_raw_body(
                store,
                raw_root,
                skipped,
                snapshots=snapshots,
                event_live_after=event_live_after,
            )

    session = TimingSession()
    with timing_artifacts(store.root):
        with timing_session(session):
            with timed_phase("ingest"):
                count = _ingest_raw_body(
                    store,
                    raw_root,
                    skipped,
                    snapshots=snapshots,
                    event_live_after=event_live_after,
                )
        meta = {
            "files": count,
            "raw_root": str(raw_root),
            "quarantine_skipped": len(skipped),
        }
        logger.info(format_kind_summary("ingest", session, meta=meta))
        write_timing_record("ingest", session, meta=meta)
    return count


def _ingest_raw_body(
    store: DataStore,
    raw_root: Path,
    skipped: frozenset[str],
    *,
    snapshots: Sequence[RawSnapshot] | None,
    event_live_after: datetime | None,
) -> int:
    """Stage snapshots and live payloads; return files consumed."""
    count = 0
    snaps = list(snapshots) if snapshots is not None else iter_snapshots(raw_root)
    for snap in snaps:
        rel = capture_relpath(raw_root, snap.capture_dir)
        if rel in skipped:
            logger.info("skipping quarantined capture %s", rel)
            continue
        bootstrap: dict[str, Any] = read_json(snap.bootstrap_path)
        fixtures = cast(
            list[dict[str, Any]],
            read_json(snap.fixtures_path) if snap.fixtures_path else [],
        )
        frames = stage_bootstrap(bootstrap, fixtures, snap.captured_at, snap.season)
        for name, frame in frames.items():
            if frame.is_empty():
                continue
            store.write(name, frame)
        count += 1

    # Stage each event_live capture once (not once per bootstrap).
    for live in iter_event_live(raw_root):
        if event_live_after is not None and live.captured_at <= event_live_after:
            continue
        rel = capture_relpath(raw_root, live.capture_dir)
        if rel in skipped:
            logger.info("skipping quarantined event_live %s", rel)
            continue
        codes = _element_to_player(store, live.season)
        stats = stage_event_live(
            read_json(live.path), live.season, live.gw, live.captured_at, codes
        )
        if not stats.is_empty():
            store.write("player_fixture_stats", stats)
        count += 1
    return count


def _element_to_player(store: DataStore, season: str) -> dict[int, int]:
    """Map ``element_id`` → ``player_key`` from the latest staged players."""
    frame = (
        store.read("players")
        .filter(pl.col("season") == season)
        .sort("available_at")
        .unique(subset=["element_id"], keep="last")
    )
    return {
        int(row["element_id"]): int(row["player_key"])
        for row in frame.iter_rows(named=True)
    }
