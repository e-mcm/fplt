"""Capture FPL API snapshots.

This file imports only the standard library and httpx so it can run in a
minimal GitHub Actions job. Raw responses are written once as gzipped JSON.

Cadence (workflow cron every 15 minutes; this function throttles writes):

- Within 3 hours of the next deadline: at most every 15 minutes.
- Within 6 hours: at most every hour.
- Otherwise: at most every 12 hours.
"""

from __future__ import annotations

import argparse
import gzip
import json
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import httpx

API_ROOT = "https://fantasy.premierleague.com/api"
_DEADLINE_WINDOW = timedelta(hours=6)
_DENSE_WINDOW = timedelta(hours=3)
_DENSE_STALENESS = timedelta(minutes=15)
_NEAR_STALENESS = timedelta(hours=1)
_MAX_STALENESS = timedelta(hours=12)


def season_label(moment: datetime) -> str:
    """FPL season containing ``moment``. A season starts in July."""
    year = moment.year if moment.month >= 7 else moment.year - 1
    return f"{year}-{(year + 1) % 100:02d}"


def _parse_deadline(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def next_deadline(bootstrap: dict[str, Any], now: datetime) -> datetime | None:
    """Earliest event deadline that is still in the future."""
    future: list[datetime] = []
    for event in bootstrap.get("events", []):
        deadline = _parse_deadline(str(event["deadline_time"]))
        if deadline >= now:
            future.append(deadline)
    return min(future) if future else None


def should_snapshot(
    now: datetime,
    deadline: datetime | None,
    last_snapshot: datetime | None,
) -> bool:
    """True when the cadence says a new bootstrap/fixtures capture is due."""
    if deadline is not None:
        until = deadline - now
        if timedelta(0) <= until <= _DENSE_WINDOW:
            return last_snapshot is None or now - last_snapshot >= _DENSE_STALENESS
        if timedelta(0) <= until <= _DEADLINE_WINDOW:
            return last_snapshot is None or now - last_snapshot >= _NEAR_STALENESS
    if last_snapshot is None or now - last_snapshot > _MAX_STALENESS:
        return True
    return False


def checked_gameweeks(bootstrap: dict[str, Any]) -> list[int]:
    """Gameweeks whose results are official (``data_checked``)."""
    return [
        int(event["id"])
        for event in bootstrap.get("events", [])
        if event.get("data_checked")
    ]


def stamp(moment: datetime) -> str:
    """Filesystem-safe UTC timestamp."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def snapshot_path(root: Path, season: str, captured_at: datetime, name: str) -> Path:
    """Path for a bootstrap or fixtures snapshot. Never used for event live."""
    return (
        root
        / "fpl_api"
        / f"season={season}"
        / f"captured_at={stamp(captured_at)}"
        / f"{name}.json.gz"
    )


def event_live_path(root: Path, season: str, gw: int, captured_at: datetime) -> Path:
    """Path for one gameweek's live payload."""
    return (
        root
        / "fpl_api"
        / f"season={season}"
        / "event_live"
        / f"gw={gw}"
        / f"captured_at={stamp(captured_at)}"
        / "event_live.json.gz"
    )


def write_once(path: Path, payload: Any) -> bool:
    """Write gzipped JSON if ``path`` does not exist. Return whether it wrote."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    return True


def latest_snapshot(root: Path, season: str) -> datetime | None:
    """Most recent bootstrap capture time already on disk for ``season``."""
    base = root / "fpl_api" / f"season={season}"
    if not base.exists():
        return None
    stamps = [
        path.parent.name.removeprefix("captured_at=")
        for path in base.glob("captured_at=*/bootstrap_static.json.gz")
    ]
    if not stamps:
        return None
    latest = max(stamps)
    return datetime.strptime(latest, "%Y-%m-%dT%H-%M-%SZ").replace(tzinfo=UTC)


@dataclass(frozen=True)
class CaptureResult:
    """Files written by one capture."""

    wrote_static: bool
    live_gameweeks: list[int]


def capture(
    client: httpx.Client,
    root: Path,
    now: datetime | None = None,
) -> CaptureResult:
    """Snapshot bootstrap and fixtures when due, and any checked live payloads."""
    moment = now or datetime.now(UTC)
    bootstrap = client.get(f"{API_ROOT}/bootstrap-static/").json()
    season = season_label(moment)
    deadline = next_deadline(bootstrap, moment)
    last = latest_snapshot(root, season)
    wrote_static = False
    if should_snapshot(moment, deadline, last):
        fixtures = client.get(f"{API_ROOT}/fixtures/").json()
        captured = snapshot_path(root, season, moment, "bootstrap_static")
        wrote_static = write_once(captured, bootstrap)
        write_once(snapshot_path(root, season, moment, "fixtures"), fixtures)
    live: list[int] = []
    for gw in checked_gameweeks(bootstrap):
        path = event_live_path(root, season, gw, moment)
        if path.exists():
            continue
        # Skip if any earlier capture of this gameweek exists.
        existing = list(
            (root / "fpl_api" / f"season={season}" / "event_live" / f"gw={gw}").glob(
                "captured_at=*/event_live.json.gz"
            )
        )
        if existing:
            continue
        payload = client.get(f"{API_ROOT}/event/{gw}/live/").json()
        if write_once(path, payload):
            live.append(gw)
    return CaptureResult(wrote_static=wrote_static, live_gameweeks=live)


def main(argv: list[str] | None = None) -> None:
    """Capture into ``--out`` (default ``./data/raw``)."""
    parser = argparse.ArgumentParser(prog="snapshot")
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    args = parser.parse_args(argv)
    with httpx.Client(timeout=30.0) as client:
        result = capture(client, args.out)
    print(
        f"static={'wrote' if result.wrote_static else 'skipped'} "
        f"live={result.live_gameweeks}"
    )


if __name__ == "__main__":
    main()
