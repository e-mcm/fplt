"""Capture API-Football (API-Sports) Premier League injuries.

This file imports only the standard library and httpx so it can run in a
minimal GitHub Actions job. Raw responses are written once as gzipped JSON
under ``data/raw/api_football/...``.

The API key is read from the environment only — never from argv:

- ``FPLT_API_FOOTBALL_KEY`` (preferred; matches ``fplt.settings``)
- ``API_FOOTBALL_KEY``
- ``APISPORTS_KEY``

Free tier is ~100 requests/day. Default capture skips when a snapshot for the
same season already exists within ``--min-interval-hours`` (default 4, matching
upstream injury refresh cadence).
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import httpx

API_ROOT = "https://v3.football.api-sports.io"
EPL_LEAGUE_ID = 39
_USER_AGENT = "Mozilla/5.0 (compatible; fplt-api-football-fetch/1.0; +https://github.com/e-mcm/fplt)"
_KEY_ENV_NAMES = (
    "FPLT_API_FOOTBALL_KEY",
    "API_FOOTBALL_KEY",
    "APISPORTS_KEY",
)
_CAPTURED_RE = re.compile(r"captured_at=(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)")


def season_label(moment: datetime) -> str:
    """FPL season containing ``moment``. A season starts in July."""
    year = moment.year if moment.month >= 7 else moment.year - 1
    return f"{year}-{(year + 1) % 100:02d}"


def api_season_year(season: str) -> int:
    """``2024-25`` → ``2024`` (API-Football season start year)."""
    return int(season.split("-", 1)[0])


def stamp(moment: datetime) -> str:
    """Filesystem-safe UTC timestamp."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def injuries_path(root: Path, season: str, captured_at: datetime) -> Path:
    """Write-once path for one league-season injuries snapshot."""
    return (
        root
        / "api_football"
        / f"season={season}"
        / "injuries"
        / f"captured_at={stamp(captured_at)}"
        / "injuries.json.gz"
    )


def write_once(path: Path, payload: Any) -> bool:
    """Write gzipped JSON if ``path`` does not exist. Return whether it wrote."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    return True


def resolve_api_key(environ: dict[str, str] | None = None) -> str | None:
    """Return the first non-empty API-Football key from known env names."""
    env = environ if environ is not None else cast(dict[str, str], os.environ)
    for name in _KEY_ENV_NAMES:
        value = env.get(name, "").strip()
        if value:
            return value
    return None


def parse_captured_at(path: Path) -> datetime | None:
    """Parse ``captured_at=...`` from an injuries snapshot path."""
    match = _CAPTURED_RE.search(path.as_posix())
    if match is None:
        return None
    parts = match.group(1)
    iso = f"{parts[0:10]}T{parts[11:13]}:{parts[14:16]}:{parts[17:19]}+00:00"
    return datetime.fromisoformat(iso).astimezone(UTC)


def latest_capture(root: Path, season: str) -> datetime | None:
    """Newest ``captured_at`` already on disk for ``season``, if any."""
    base = root / "api_football" / f"season={season}" / "injuries"
    if not base.exists():
        return None
    newest: datetime | None = None
    for path in base.glob("captured_at=*/injuries.json.gz"):
        when = parse_captured_at(path)
        if when is None:
            continue
        if newest is None or when > newest:
            newest = when
    return newest


def should_fetch(
    root: Path,
    season: str,
    now: datetime,
    *,
    min_interval: timedelta,
) -> bool:
    """True when no recent write-once snapshot exists for ``season``."""
    previous = latest_capture(root, season)
    if previous is None:
        return True
    return now - previous >= min_interval


def fetch_injuries(
    client: httpx.Client,
    *,
    league: int,
    season_year: int,
) -> dict[str, Any]:
    """GET ``/injuries`` for one league-season and return the JSON body."""
    response = client.get(
        f"{API_ROOT}/injuries",
        params={"league": league, "season": season_year},
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("/injuries must return a JSON object")
    return cast(dict[str, Any], payload)


@dataclass(frozen=True)
class CaptureResult:
    """Outcome of one capture attempt."""

    wrote: bool
    skipped: bool
    path: Path | None


def capture(
    client: httpx.Client,
    root: Path,
    season: str,
    *,
    now: datetime | None = None,
    min_interval_hours: float = 4.0,
    league: int = EPL_LEAGUE_ID,
    delay: float = 0.0,
) -> CaptureResult:
    """Fetch EPL injuries for ``season`` when the rate-limit window allows.

    Args:
        client: Authenticated httpx client (``x-apisports-key`` header set).
        root: ``$FPLT_DATA_DIR/raw``.
        season: FPL season label such as ``2024-25``.
        now: Capture clock (UTC). Defaults to wall clock.
        min_interval_hours: Skip the HTTP call when a newer-than-this snapshot
            already exists (conserves free-tier quota).
        league: API-Football league id (39 = Premier League).
        delay: Optional sleep after a successful write (tests / politeness).

    Returns:
        Whether a new file was written, or skipped for rate limiting / write-once.
    """
    moment = now or datetime.now(UTC)
    interval = timedelta(hours=min_interval_hours)
    path = injuries_path(root, season, moment)
    if not should_fetch(root, season, moment, min_interval=interval):
        return CaptureResult(wrote=False, skipped=True, path=None)
    if path.exists():
        return CaptureResult(wrote=False, skipped=True, path=path)
    payload = fetch_injuries(client, league=league, season_year=api_season_year(season))
    wrote = write_once(path, payload)
    if wrote and delay > 0:
        time.sleep(delay)
    return CaptureResult(wrote=wrote, skipped=not wrote, path=path if wrote else None)


def main(argv: list[str] | None = None) -> int:
    """Capture into ``--out`` (default ``./data/raw``)."""
    parser = argparse.ArgumentParser(prog="fetch_api_football")
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="FPL season label such as 2024-25 (default: current).",
    )
    parser.add_argument(
        "--min-interval-hours",
        type=float,
        default=4.0,
        help="Skip fetch when a snapshot newer than this already exists.",
    )
    parser.add_argument(
        "--league",
        type=int,
        default=EPL_LEAGUE_ID,
        help="API-Football league id (default: 39 Premier League).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.0,
        help="Seconds to sleep after a successful write.",
    )
    args = parser.parse_args(argv)
    key = resolve_api_key()
    if not key:
        names = ", ".join(_KEY_ENV_NAMES)
        print(
            f"missing API-Football key; set one of: {names}",
            file=sys.stderr,
        )
        return 2
    moment = datetime.now(UTC)
    season = args.season or season_label(moment)
    with httpx.Client(
        timeout=30.0,
        headers={
            "User-Agent": _USER_AGENT,
            "x-apisports-key": key,
            "Accept": "application/json",
        },
    ) as client:
        result = capture(
            client,
            args.out,
            season,
            now=moment,
            min_interval_hours=args.min_interval_hours,
            league=args.league,
            delay=args.delay,
        )
    if result.skipped and not result.wrote:
        print(f"season={season} injuries=skipped")
    else:
        print(f"season={season} injuries={'wrote' if result.wrote else 'exists'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
