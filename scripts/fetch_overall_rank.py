"""Capture FPL overall-league points-at-rank ladders write-once.

Fetches ``bootstrap-static`` (field size) plus a few classic Overall standings
pages (10k, 100k, …) and writes
``raw/fpl_api/season=.../overall_rank/captured_at=.../points_at_rank.json.gz``.

Does not overwrite an existing capture at the same timestamp. Prefer library
calls via ``fplt.data.overall_rank``; this script is the operator entry point.
"""

from __future__ import annotations

import argparse
import sys
from datetime import UTC, datetime
from pathlib import Path

from fplt.data.fpl_client import FPLClient
from fplt.data.overall_rank import (
    DEFAULT_TARGET_RANKS,
    OVERALL_LEAGUE_ID,
    capture_overall_rank,
    season_label,
)


def main(argv: list[str] | None = None) -> int:
    """CLI entry: fetch and cache an overall-rank ladder."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="Raw root (usually $FPLT_DATA_DIR/raw)",
    )
    parser.add_argument(
        "--season",
        default=None,
        help="FPL season label (default: season containing now)",
    )
    parser.add_argument(
        "--rank",
        action="append",
        type=int,
        dest="ranks",
        help="Target rank to sample (repeatable; default includes 10k and 100k)",
    )
    parser.add_argument(
        "--league-id",
        type=int,
        default=OVERALL_LEAGUE_ID,
        help=f"Classic league id (default Overall {OVERALL_LEAGUE_ID})",
    )
    args = parser.parse_args(argv)
    moment = datetime.now(UTC)
    season = args.season or season_label(moment)
    ranks = tuple(args.ranks) if args.ranks else DEFAULT_TARGET_RANKS
    with FPLClient() as client:
        snapshot, wrote = capture_overall_rank(
            client,
            args.out,
            season=season,
            captured_at=moment,
            target_ranks=ranks,
            league_id=args.league_id,
        )
    status = "wrote" if wrote else "exists"
    print(
        f"overall_rank season={snapshot.season} "
        f"total_players={snapshot.total_players} "
        f"ranks={len(snapshot.ranks)} {status}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
