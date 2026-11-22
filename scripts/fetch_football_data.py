"""Download football-data.co.uk E0 CSVs write-once into raw storage.

This file imports only the standard library and httpx so it can run in a
minimal job. Existing paths are never overwritten. Pass ``--captured-at`` for
monthly refresh captures under ``season=…/captured_at=…/E0.csv``.
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import httpx

API_ROOT = "https://www.football-data.co.uk/mmz4281"
_USER_AGENT = (
    "Mozilla/5.0 (compatible; fplt-football-data-fetch/1.0; "
    "+https://github.com/e-mcm/fplt)"
)
_CAPTURED_AT = re.compile(r"^\d{4}-\d{2}-\d{2}$")


@dataclass(frozen=True)
class CaptureResult:
    """Whether a season CSV was written."""

    season: str
    path: Path
    wrote: bool


def season_token(season: str) -> str:
    """Convert ``2024-25`` to football-data's ``2425`` path token."""
    start, end = season.split("-", 1)
    return f"{int(start) % 100:02d}{int(end) % 100:02d}"


def season_label(moment: datetime) -> str:
    """FPL season containing ``moment``. A season starts in July."""
    year = moment.year if moment.month >= 7 else moment.year - 1
    return f"{year}-{(year + 1) % 100:02d}"


def e0_url(season: str) -> str:
    """Public CSV URL for one Premier League season."""
    return f"{API_ROOT}/{season_token(season)}/E0.csv"


def e0_raw_path(raw_root: Path, season: str, *, captured_at: str | None = None) -> Path:
    """Write-once path for one E0 CSV under ``raw/football_data``."""
    base = raw_root / "football_data" / f"season={season}"
    if captured_at is None:
        return base / "E0.csv"
    return base / f"captured_at={captured_at}" / "E0.csv"


def write_once(path: Path, body: str) -> bool:
    """Write UTF-8 text if ``path`` does not exist. Return whether it wrote."""
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(body, encoding="utf-8")
    return True


def fetch_season(
    client: httpx.Client,
    root: Path,
    season: str,
    *,
    captured_at: str | None = None,
) -> CaptureResult:
    """Download one E0 CSV into ``root`` unless the destination already exists."""
    path = e0_raw_path(root, season, captured_at=captured_at)
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
        help="FPL season label, e.g. 2024-25 (repeatable). Default: current.",
    )
    parser.add_argument(
        "--captured-at",
        default=None,
        help=(
            "UTC calendar day YYYY-MM-DD for a monthly write-once capture path. "
            "Omit for legacy season=/E0.csv."
        ),
    )
    args = parser.parse_args(argv)
    captured_at: str | None = args.captured_at
    if captured_at is not None and _CAPTURED_AT.match(captured_at) is None:
        parser.error("--captured-at must be YYYY-MM-DD")
    seasons: list[str] = (
        list(args.season) if args.season else [season_label(datetime.now(tz=UTC))]
    )
    # football-data.co.uk 302s www → apex; httpx defaults to not following.
    with httpx.Client(
        headers={"User-Agent": _USER_AGENT},
        timeout=60.0,
        follow_redirects=True,
    ) as client:
        for season in seasons:
            result = fetch_season(client, args.out, season, captured_at=captured_at)
            state = "wrote" if result.wrote else "exists"
            print(f"{state} {result.path}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
