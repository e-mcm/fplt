"""Command line entry point for fplt."""

from __future__ import annotations

import argparse
import sys
from typing import Literal

from fplt.backtest.run import run_backtest
from fplt.settings import get_settings

_REFITS = ("every_gw", "every_n_gw", "never")


def _not_implemented(name: str) -> int:
    print(f"fplt {name} is not implemented yet", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch a subcommand."""
    parser = argparse.ArgumentParser(prog="fplt")
    sub = parser.add_subparsers(dest="command")
    backtest = sub.add_parser("backtest", help="Walk-forward backtest a model")
    backtest.add_argument("--season", required=True)
    backtest.add_argument("--from-gw", type=int, default=1)
    backtest.add_argument("--to-gw", type=int, default=38)
    backtest.add_argument(
        "--models",
        nargs="+",
        default=["rolling_mean_v1", "ep_next_v1"],
    )
    backtest.add_argument("--refit", choices=_REFITS)
    backtest.add_argument("--seasons-back", type=int)
    backtest.add_argument("--steps", type=int, default=1)
    sub.add_parser("predict", help="Predict the next gameweeks")
    sub.add_parser("plan", help="Recommend a squad and transfers")
    args = parser.parse_args(argv)
    if args.command is None:
        return
    if args.command != "backtest":
        raise SystemExit(_not_implemented(args.command))
    _backtest(args)


def _backtest(args: argparse.Namespace) -> None:
    refit: Literal["every_gw", "every_n_gw", "never"] | None = args.refit
    try:
        ran = run_backtest(
            get_settings().data_dir,
            args.season,
            args.from_gw,
            args.to_gw,
            list(args.models),
            refit=refit,
            seasons_back=args.seasons_back,
            steps=args.steps,
        )
    except (ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    for spec in ran:
        window = spec.train_window.seasons_back
        print(
            f"{spec.name} refit={spec.refit} seasons_back={window} "
            f"model_id={spec.model_id}"
        )
