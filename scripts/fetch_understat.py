"""Capture Understat EPL match JSON and league team Deep/PPDA.

This file imports only the standard library and httpx so it can run in a
minimal GitHub Actions job. Raw responses are written once as gzipped JSON
under ``data/raw/understat/...``.

League dates and team Deep / PPDA come from the AJAX ``getLeagueData``
payload (``dates`` / ``teams``). Older league pages that still embed
``datesData`` / ``teamsData`` remain a fallback. Match rosters still use
``getMatchData``.
"""

from __future__ import annotations

import argparse
import gzip
import importlib.util
import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from types import ModuleType
from typing import Any, cast

import httpx


def _capture_health() -> ModuleType:
    """Load ``fplt.data.capture_health`` without importing heavy ``fplt.data``."""
    name = "fplt_data_capture_health"
    existing = sys.modules.get(name)
    if existing is not None:
        return existing
    path = (
        Path(__file__).resolve().parents[1]
        / "src"
        / "fplt"
        / "data"
        / "capture_health.py"
    )
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise ImportError(f"cannot load capture_health from {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


API_ROOT = "https://understat.com"
_AJAX = {
    "X-Requested-With": "XMLHttpRequest",
    "Accept": "application/json, text/javascript, */*; q=0.01",
}
_USER_AGENT = (
    "Mozilla/5.0 (compatible; fplt-understat-fetch/1.0; +https://github.com/e-mcm/fplt)"
)
_JSON_PARSE = re.compile(r"(\w+)\s*=\s*JSON\.parse\('(.+?)'\)", re.DOTALL)


class UnderstatUnavailable(RuntimeError):
    """Understat returned maintenance HTML or otherwise no league JSON."""


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


def teams_path(root: Path, season: str, captured_at: datetime) -> Path:
    """Path for one league ``teamsData`` snapshot (Deep / PPDA history)."""
    return (
        root
        / "understat"
        / f"season={season}"
        / "teams"
        / f"captured_at={stamp(captured_at)}"
        / "teams.json.gz"
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


def is_maintenance_html(text: str) -> bool:
    """Return whether ``text`` looks like Understat's maintenance page."""
    lowered = text.lower()
    return "come back soon" in lowered or "/css/errors.css" in lowered


def parse_embedded_json(html: str, name: str) -> Any:
    """Parse ``name = JSON.parse('...')`` from an Understat HTML page."""
    for match in _JSON_PARSE.finditer(html):
        if match.group(1) != name:
            continue
        raw = match.group(2).encode("utf-8").decode("unicode_escape")
        return json.loads(raw)
    raise KeyError(f"{name} not found in Understat HTML")


def dates_from_league_data(payload: dict[str, Any]) -> list[dict[str, Any]]:
    """Extract the ``dates`` list from a ``getLeagueData`` JSON object.

    Args:
        payload: Decoded AJAX body with a ``dates`` key (the post-2025 shape).

    Returns:
        League match rows (finished and unfinished).

    Raises:
        KeyError: ``dates`` is missing.
        ValueError: ``dates`` is present but not a list.
    """
    if "dates" not in payload:
        raise KeyError("dates not found in getLeagueData payload")
    dates = payload["dates"]
    if not isinstance(dates, list):
        raise ValueError("getLeagueData.dates must be a list")
    return cast(list[dict[str, Any]], dates)


def teams_from_league_data(payload: dict[str, Any]) -> dict[str, Any]:
    """Extract the ``teams`` object from a ``getLeagueData`` JSON object."""
    teams = payload.get("teams", {})
    if not isinstance(teams, dict):
        raise ValueError("getLeagueData.teams must be an object")
    return cast(dict[str, Any], teams)


@dataclass(frozen=True)
class CaptureResult:
    """Files written by one capture."""

    wrote_dates: bool
    wrote_teams: bool
    match_ids: list[int]


def _seed_session(client: httpx.Client, year: str) -> str:
    response = client.get(f"{API_ROOT}/league/EPL/{year}")
    response.raise_for_status()
    return response.text


def fetch_league_page(client: httpx.Client, year: str) -> str:
    """Return the EPL league HTML for ``year`` (seeds cookies too)."""
    return _seed_session(client, year)


def fetch_league_data(
    client: httpx.Client, year: str, html: str | None = None
) -> dict[str, Any]:
    """Return the ``getLeagueData`` object for EPL ``year``.

    Raises:
        UnderstatUnavailable: The site is on maintenance or returns non-JSON.
        httpx.HTTPError: The AJAX request fails with a non-success status.
    """
    page = html if html is not None else fetch_league_page(client, year)
    if is_maintenance_html(page):
        raise UnderstatUnavailable(
            "Understat league page is a maintenance HTML response"
        )
    response = client.get(
        f"{API_ROOT}/getLeagueData/EPL/{year}",
        headers={**_AJAX, "Referer": f"{API_ROOT}/league/EPL/{year}"},
    )
    response.raise_for_status()
    body = response.text
    if is_maintenance_html(body):
        raise UnderstatUnavailable(
            "Understat getLeagueData is a maintenance HTML response"
        )
    try:
        payload = response.json()
    except json.JSONDecodeError as exc:
        raise UnderstatUnavailable(
            "getLeagueData returned non-JSON; dates are no longer embedded in HTML"
        ) from exc
    if not isinstance(payload, dict):
        raise UnderstatUnavailable("getLeagueData must return a JSON object")
    return cast(dict[str, Any], payload)


def fetch_league_dates(
    client: httpx.Client, year: str, html: str | None = None
) -> list[dict[str, Any]]:
    """Return the ``dates`` list for EPL ``year`` (AJAX, else HTML ``datesData``)."""
    page = html if html is not None else fetch_league_page(client, year)
    try:
        payload = fetch_league_data(client, year, page)
        return dates_from_league_data(payload)
    except UnderstatUnavailable:
        if is_maintenance_html(page):
            raise
        # Legacy pages still embed datesData; keep that fallback for fixtures.
        dates = parse_embedded_json(page, "datesData")
        if not isinstance(dates, list):
            raise ValueError("datesData must be a list") from None
        return cast(list[dict[str, Any]], dates)
    except httpx.HTTPError:
        if is_maintenance_html(page):
            raise UnderstatUnavailable(
                "Understat league page is a maintenance HTML response"
            ) from None
        dates = parse_embedded_json(page, "datesData")
        if not isinstance(dates, list):
            raise ValueError("datesData must be a list") from None
        return cast(list[dict[str, Any]], dates)


def fetch_league_teams(
    html: str, league_data: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Return league ``teams`` from AJAX payload, else HTML ``teamsData``."""
    if league_data is not None:
        teams = teams_from_league_data(league_data)
        if teams:
            return teams
    teams = parse_embedded_json(html, "teamsData")
    if not isinstance(teams, dict):
        raise ValueError("teamsData must be an object")
    return cast(dict[str, Any], teams)


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
    """Fetch finished EPL matches and league team stats for ``season``.

    Raises:
        UnderstatUnavailable: The site serves maintenance HTML / no league JSON.
    """
    moment = now or datetime.now(UTC)
    year = understat_year(season)
    html = fetch_league_page(client, year)
    league_data: dict[str, Any] | None = None
    try:
        league_data = fetch_league_data(client, year, html)
        dates = dates_from_league_data(league_data)
    except UnderstatUnavailable:
        if is_maintenance_html(html):
            raise
        # Legacy HTML still embeds datesData (pre-AJAX pages / recorded fixtures).
        dates = parse_embedded_json(html, "datesData")
        if not isinstance(dates, list):
            raise ValueError("datesData must be a list") from None
        dates = cast(list[dict[str, Any]], dates)
    wrote_dates = write_once(dates_path(root, season, moment), dates)
    wrote_teams = False
    try:
        teams = fetch_league_teams(html, league_data)
        wrote_teams = write_once(teams_path(root, season, moment), teams)
    except (KeyError, ValueError):
        # Maintenance / blocked pages omit teams; dates-only still useful.
        wrote_teams = False
    written: list[int] = []
    for match_id in finished_match_ids(dates):
        path = match_path(root, season, match_id)
        if path.exists():
            continue
        try:
            payload = fetch_match(client, match_id, year)
        except (httpx.HTTPError, ValueError, json.JSONDecodeError):
            continue
        if write_once(path, payload):
            written.append(match_id)
        if delay > 0:
            time.sleep(delay)
    return CaptureResult(
        wrote_dates=wrote_dates, wrote_teams=wrote_teams, match_ids=written
    )


def main(argv: list[str] | None = None) -> int:
    """Capture into ``--out`` (default ``./data/raw``).

    Distinguishes healthy "nothing new" (exit 0, heartbeat ok) from source
    unavailable (warning + status counter; no heartbeat; non-zero after K).
    """
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
    parser.add_argument(
        "--status-dir",
        type=Path,
        default=Path("data/status"),
        help="Local status/ mirror for unavailable counters (R2 status/).",
    )
    parser.add_argument(
        "--unavailable-threshold",
        type=int,
        default=None,
        help="Consecutive unavailable runs before exit non-zero (default: config).",
    )
    args = parser.parse_args(argv)
    health = _capture_health()
    moment = datetime.now(UTC)
    season = args.season or season_label(moment)
    threshold = args.unavailable_threshold
    with httpx.Client(
        timeout=30.0,
        headers={"User-Agent": _USER_AGENT},
        follow_redirects=True,
    ) as client:
        try:
            result = capture(client, args.out, season, now=moment, delay=args.delay)
        except UnderstatUnavailable as exc:
            outcome = health.record_capture_outcome(
                source="understat",
                status_dir=args.status_dir,
                available=False,
                reason=str(exc),
                threshold=threshold,
                now=moment,
            )
            print(f"season={season} unavailable: {exc}")
            return int(outcome.exit_code)
        except httpx.HTTPError as exc:
            outcome = health.record_capture_outcome(
                source="understat",
                status_dir=args.status_dir,
                available=False,
                reason=f"HTTP error: {exc}",
                threshold=threshold,
                now=moment,
            )
            print(f"season={season} unavailable: {exc}")
            return int(outcome.exit_code)
    outcome = health.record_capture_outcome(
        source="understat",
        status_dir=args.status_dir,
        available=True,
        threshold=threshold,
        now=moment,
    )
    print(
        f"season={season} dates={'wrote' if result.wrote_dates else 'skipped'} "
        f"teams={'wrote' if result.wrote_teams else 'skipped'} "
        f"matches={result.match_ids}"
    )
    return int(outcome.exit_code)


if __name__ == "__main__":
    raise SystemExit(main())
