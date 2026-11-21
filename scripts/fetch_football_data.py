"""Download football-data.co.uk E0 CSVs write-once into raw storage.

This file imports only the standard library and httpx so it can run in a
minimal job. Existing ``E0.csv`` files are never overwritten.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass
from pathlib import Path

import httpx

from fplt.data.football_data import e0_raw_path, season_token

API_ROOT = "https://www.football-data.co.uk/mmz4281"
_USER_AGENT = "Mozilla/5.0 (compatible; fplt-football-data-fetch/1.0; +https://github.com/e-mcm/fplt)"


@dataclass(frozen=True)
class CaptureResult:
    """Whether a season CSV was written."""

    season: str
    path: Path
    wrote: bool


def e0_url(season: str) -> str:
    """Public CSV URL for one Premier League season."""
    return f"{API_ROOT}/{season_token(season)}/E0.csv"


def write_once(path: Path, body: str) -> bool:
    """Write UTF-8 text if ``path`` does not exist. Return whether it wrote."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return True


def fetch_season(client: httpx.Client, root: Path, season: str) -> CaptureResult:
    """Download one E0 CSV into ``root`` unless it already exists."""
    path = e0_raw_path(root, season)
    if path.exists():
        return CaptureResult(season=season, path=path, wrote=False)
    response = client.get(e0_url(season))
    response.raise_for_status()
    wrote = write_once(path, response.text)
    return CaptureResult(season=season, path=path, wrote=wrote)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: fetch one or more E0 season CSVs write-once."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Raw root (usually $FPLT_DATA_DIR/raw)",
    )
    parser.add_argument(
        "--season",
        action="append",
        required=True,
        help="FPL season label, e.g. 2024-25 (repeatable)",
    )
    args = parser.parse_args(argv)
    seasons: list[str] = list(args.season)
    # football-data.co.uk 302s www → apex; httpx defaults to not following.
    with httpx.Client(
        headers={"User-Agent": _USER_AGENT},
        timeout=60.0,
        follow_redirects=True,
    ) as client:
        for season in seasons:
            result = fetch_season(client, args.out, season)
            state = "wrote" if result.wrote else "exists"
            print(f"{state} {result.path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
