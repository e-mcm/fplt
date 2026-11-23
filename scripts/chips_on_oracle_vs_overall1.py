"""Run chips-on constrained hindsight oracle vs Overall #1 for a season.

Default arm: optimised chips, H=8, ``force_play=False`` (scheduler does not
force-pack remaining half-window chips — with perfect foresight the MILP
should time chips; force-play is a live-forecast heuristic).

Usage::

    export FPLT_DATA_DIR=/workspace/data-chips-on-oracle
    uv run python scripts/chips_on_oracle_vs_overall1.py \\
        --season 2024-25 --from-gw 1 --to-gw 38 \\
        --overall1-points 2810 --output-dir /path/to/artifacts
"""

from __future__ import annotations

import argparse
import json
import logging
import time
from pathlib import Path

import polars as pl

from fplt.data.store import DataStore
from fplt.optimise.attainable import (
    ORACLE_MODEL_ID,
    oracle_ledger_model_id,
    run_hindsight_oracle,
)
from fplt.optimise.decisions import read_decisions
from fplt.optimise.realised import summarise_decisions
from fplt.settings import get_settings

logger = logging.getLogger(__name__)

# Published 2024-25 Overall #1 (Lovro Budišin). Live FPL API only serves the
# current season; pin the completed-season total from contemporaneous reports.
DEFAULT_OVERALL1: dict[str, dict[str, int | str | list[str]]] = {
    "2024-25": {
        "points": 2810,
        "manager": "Lovro Budišin",
        "team": "Aina Krafth Bree",
        "sources": [
            "https://www.sportscasting.com/uk/news/fpl-champion-2024-25-highest-ranked-managers-overall-leaderboard/",
            "https://www.fantasyfootballretro.com/records/fpl-winners/",
            "https://allaboutfpl.com/2026/05/list-of-fpl-winners-since-the-first-fpl-season-in-2002-03/",
        ],
    }
}


def main() -> None:
    """CLI entry: simulate chips-on oracle and compare to Overall #1."""
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--season", default="2024-25")
    parser.add_argument("--from-gw", type=int, default=1)
    parser.add_argument("--to-gw", type=int, default=38)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument(
        "--force-play",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Optimised scheduler force-pack (default: off for hindsight)",
    )
    parser.add_argument(
        "--chip-arm",
        default="optimised",
        choices=("off", "rule_based", "optimised"),
    )
    parser.add_argument(
        "--overall1-points",
        type=int,
        default=None,
        help="Override Overall #1 season points",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Directory for summary JSON / decision parquet copy",
    )
    parser.add_argument(
        "--skip-simulate",
        action="store_true",
        help="Reuse existing decision ledger rows for this oracle model_id",
    )
    args = parser.parse_args()
    data_dir = get_settings().data_dir
    store = DataStore(data_dir)
    _ = store  # ensure path exists / is readable
    meta = DEFAULT_OVERALL1.get(args.season, {})
    if args.overall1_points is not None:
        overall1 = int(args.overall1_points)
    else:
        raw_points = meta.get("points", 0)
        overall1 = raw_points if isinstance(raw_points, int) else 0
    if overall1 <= 0:
        raise SystemExit(
            f"pass --overall1-points for season={args.season} (not in DEFAULT_OVERALL1)"
        )
    model_id = oracle_ledger_model_id(args.chip_arm)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    if not args.skip_simulate:
        logger.info(
            "oracle start season=%s gw=%s-%s H=%s chip_arm=%s force_play=%s",
            args.season,
            args.from_gw,
            args.to_gw,
            args.horizon,
            args.chip_arm,
            args.force_play,
        )
        run_hindsight_oracle(
            data_dir,
            args.season,
            args.from_gw,
            args.to_gw,
            args.horizon,
            chip_arm=args.chip_arm,
            force_play=args.force_play,
            prefer_staged_actuals=True,
        )
    elapsed = time.perf_counter() - started
    decisions = read_decisions(data_dir).filter(
        (pl.col("model_id") == model_id)
        & (pl.col("season") == args.season)
        & (pl.col("gw") >= args.from_gw)
        & (pl.col("gw") <= args.to_gw)
    )
    if decisions.is_empty():
        raise SystemExit(f"no decision rows for model_id={model_id}")
    summary = summarise_decisions(decisions)
    oracle_pts = int(summary["total_points"][0])
    gap = oracle_pts - overall1
    chips = (
        decisions.select("gw", "chip")
        .unique()
        .filter(pl.col("chip").is_not_null())
        .sort("gw")
    )
    chip_plays = [
        {"gw": int(row["gw"]), "chip": str(row["chip"])}
        for row in chips.iter_rows(named=True)
    ]
    payload = {
        "season": args.season,
        "from_gw": args.from_gw,
        "to_gw": args.to_gw,
        "horizon": args.horizon,
        "chip_arm": args.chip_arm,
        "force_play": args.force_play,
        "oracle_model_id": model_id,
        "base_oracle_model_id": ORACLE_MODEL_ID,
        "oracle_points": oracle_pts,
        "overall1_points": overall1,
        "overall1_manager": meta.get("manager"),
        "overall1_team": meta.get("team"),
        "overall1_sources": meta.get("sources"),
        "gap_oracle_minus_overall1": gap,
        "pass": oracle_pts >= overall1,
        "chip_plays": chip_plays,
        "gameweeks": int(summary["gameweeks"][0]),
        "elapsed_seconds": round(elapsed, 1),
        "data_dir": str(data_dir),
        "choice_rationale": (
            "Primary arm is H=8 force_play=False: rolling horizon matches the "
            "candidate chips-on planner; force-off lets the hindsight MILP "
            "time chips without half-window packing."
        ),
    }
    out_json = args.output_dir / "summary.json"
    out_json.write_text(json.dumps(payload, indent=2) + "\n")
    decisions.write_parquet(args.output_dir / "decisions.parquet")
    chips.write_csv(args.output_dir / "chip_plays.csv")
    print(json.dumps(payload, indent=2))
    print(f"wrote {out_json}")


if __name__ == "__main__":
    main()
