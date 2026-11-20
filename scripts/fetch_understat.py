"""Capture Understat EPL match JSON.

This file imports only the standard library and httpx so it can run in a
minimal GitHub Actions job. Raw responses are written once as gzipped JSON
under ``data/raw/understat/...``.
"""

from __future__ import annotations

import argparse
import gzip
import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx

API_ROOT = "https://understat.com"
_AJAX = {
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}
_USER_AGENT = (
    "Mozilla/5.0 (compatible; fplt-understat-fetch/1.0; +https://github.com/e-mcm/fplt)"
)


def season_label(moment: datetime) -> str:
    """FPL season containing ``moment``. A season starts in July."""
    year = moment.year if moment.month >= 7 else moment.year - 1
    return f"{year}-{(year + 1) % 100:02d}"


def understat_year(season: str) -> str:
    """``2024-25`` → ``2024``."""
    return season.split("-", 1)[0]


def stamp(moment: datetime) -> str:
    """Filesystem-safe UTC timestamp."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def match_path(root: Path, season: str, match_id: int) -> Path:
    """Path for one finished match payload."""
    return (
        root
        / "understat"
        / f"season={season}"
        / f"match_id={match_id}"
        / "match.json.gz"
    )


def dates_path(root: Path, season: str, captured_at: datetime) -> Path:
    """Path for one league-dates snapshot."""
    return (
        root
        / "understat"
        / f"season={season}"
        / "dates"
        / f"captured_at={stamp(captured_at)}"
        / "dates.json.gz"
    )


def write_once(path: Path, payload: Any) -> bool:
    """Write gzipped JSON if ``path`` does not exist. Return whether it wrote."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    return True


def finished_match_ids(dates: list[dict[str, Any]]) -> list[int]:
    """Understat match ids whose ``isResult`` flag is true."""
    ids: list[int] = []
    for row in dates:
        if row.get("isResult") in (True, "true", "True", 1, "1"):
            ids.append(int(row["id"]))
    return ids


@dataclass(frozen=True)
class CaptureResult:
    """Files written by one capture."""

    wrote_dates: bool
    match_ids: list[int]


def _seed_session(client: httpx.Client, year: str) -> None:
    client.get(f"{API_ROOT}/league/EPL/{year}")


def fetch_league_dates(client: httpx.Client, year: str) -> list[dict[str, Any]]:
    """Return the ``dates`` list from ``getLeagueData`` for EPL ``year``."""
    _seed_session(client, year)
    response = client.get(
        f"{API_ROOT}/getLeagueData/EPL/{year}",
        headers={**_AJAX, "Referer": f"{API_ROOT}/league/EPL/{year}"},
    )
    response.raise_for_status()
    payload = cast(dict[str, Any], response.json())
    dates = payload.get("dates", [])
    if not isinstance(dates, list):
        raise ValueError("getLeagueData.dates must be a list")
    return cast(list[dict[str, Any]], dates)


def fetch_match(client: httpx.Client, match_id: int, year: str) -> dict[str, Any]:
    """Return one ``getMatchData`` payload."""
    response = client.get(
        f"{API_ROOT}/getMatchData/{match_id}",
        headers={**_AJAX, "Referer": f"{API_ROOT}/league/EPL/{year}"},
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError(f"getMatchData/{match_id} must return an object")
    return cast(dict[str, Any], payload)


def capture(
    client: httpx.Client,
    root: Path,
    season: str,
    *,
    now: datetime | None = None,
    delay: float = 0.2,
) -> CaptureResult:
    """Fetch finished EPL matches for ``season`` into write-once raw paths."""
    moment = now or datetime.now(UTC)
    year = understat_year(season)
    dates = fetch_league_dates(client, year)
    wrote_dates = write_once(dates_path(root, season, moment), dates)
    written: list[int] = []
    for match_id in finished_match_ids(dates):
        path = match_path(root, season, match_id)
        if path.exists():
            continue
        payload = fetch_match(client, match_id, year)
        if write_once(path, payload):
            written.append(match_id)
        if delay > 0:
            time.sleep(delay)
    return CaptureResult(wrote_dates=wrote_dates, match_ids=written)


def main(argv: list[str] | None = None) -> None:
    """Capture into ``--out`` (default ``./data/raw``)."""
    parser = argparse.ArgumentParser(prog="fetch_understat")
    parser.add_argument("--out", type=Path, default=Path("data/raw"))
    parser.add_argument(
        "--season",
        type=str,
        default=None,
        help="FPL season label such as 2024-25 (default: current).",
    )
    parser.add_argument(
        "--delay",
        type=float,
        default=0.2,
        help="Seconds to wait between match requests.",
    )
    args = parser.parse_args(argv)
    moment = datetime.now(UTC)
    season = args.season or season_label(moment)
    with httpx.Client(
        timeout=30.0,
        headers={"User-Agent": _USER_AGENT},
        follow_redirects=True,
    ) as client:
        result = capture(client, args.out, season, now=moment, delay=args.delay)
    print(
        f"season={season} dates={'wrote' if result.wrote_dates else 'skipped'} "
        f"matches={result.match_ids}"
    )


if __name__ == "__main__":
    main()
