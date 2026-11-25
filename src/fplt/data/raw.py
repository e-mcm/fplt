"""Read write-once gzipped API snapshots and append them to the store."""

from __future__ import annotations

import gzip
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from fplt.data.quarantine import capture_relpath, quarantined_capture_paths
from fplt.data.staging import stage_bootstrap, stage_event_live
from fplt.data.store import DataStore
from fplt.utils.timing import (
    TimingSession,
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


def read_json(path: Path) -> Any:
    """Read one gzipped JSON file."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def ingest_raw(store: DataStore, raw_root: Path) -> int:
    """Stage every snapshot and checked live payload. Return files consumed.

    Captures listed in ``store.root/quarantine/`` (ops-06) are skipped so a
    quarantined raw object never reaches ``staged/``.
    """
    skipped = quarantined_capture_paths(store.root)
    session = TimingSession()
    with timing_artifacts(store.root):
        with timing_session(session):
            with timed_phase("ingest"):
                count = 0
                for snap in iter_snapshots(raw_root):
                    rel = capture_relpath(raw_root, snap.capture_dir)
                    if rel in skipped:
                        logger.info("skipping quarantined capture %s", rel)
                        continue
                    bootstrap: dict[str, Any] = read_json(snap.bootstrap_path)
                    fixtures = cast(
                        list[dict[str, Any]],
                        read_json(snap.fixtures_path) if snap.fixtures_path else [],
                    )
                    frames = stage_bootstrap(
                        bootstrap, fixtures, snap.captured_at, snap.season
                    )
                    for name, frame in frames.items():
                        if frame.is_empty():
                            continue
                        store.write(name, frame)
                    codes = {
                        int(row["element_id"]): int(row["player_key"])
                        for row in frames["players"].iter_rows(named=True)
                    }
                    live_root = (
                        raw_root / "fpl_api" / f"season={snap.season}" / "event_live"
                    )
                    if live_root.exists():
                        for path in sorted(
                            live_root.glob("gw=*/captured_at=*/event_live.json.gz")
                        ):
                            gw = int(path.parents[1].name.removeprefix("gw="))
                            captured = _stamp(
                                path.parent.name.removeprefix("captured_at=")
                            )
                            stats = stage_event_live(
                                read_json(path), snap.season, gw, captured, codes
                            )
                            if not stats.is_empty():
                                store.write("player_fixture_stats", stats)
                            count += 1
                    count += 1
        meta = {
            "files": count,
            "raw_root": str(raw_root),
            "quarantine_skipped": len(skipped),
        }
        logger.info(format_kind_summary("ingest", session, meta=meta))
        write_timing_record("ingest", session, meta=meta)
    return count
