"""Command line entry point for fplt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

from fplt.backtest.run import run_backtest
from fplt.optimise.decisions import run_decisions, run_plan
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
    plan = sub.add_parser("plan", help="Recommend a squad and transfers")
    plan.add_argument("--model", required=True)
    plan.add_argument("--season", required=True)
    plan.add_argument("--gw", type=int)
    plan.add_argument("--from-gw", type=int, default=1)
    plan.add_argument("--to-gw", type=int, default=38)
    plan.add_argument("--horizon", type=int, default=2)
    plan.add_argument("--decisions", action="store_true")
    plan.add_argument("--entry-json", type=Path)
    args = parser.parse_args(argv)
    if args.command is None:
        return
    if args.command == "backtest":
        _backtest(args)
        return
    if args.command == "plan":
        _plan(args)
        return
    raise SystemExit(_not_implemented(args.command))


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


def _plan(args: argparse.Namespace) -> None:
    try:
        if args.decisions:
            path = run_decisions(
                get_settings().data_dir,
                args.model,
                args.season,
                args.from_gw,
                args.to_gw,
                args.horizon,
            )
            print(path)
            return
        if args.gw is None:
            raise ValueError("fplt plan needs --gw")
        _solved, path, report = run_plan(
            get_settings().data_dir,
            args.model,
            args.season,
            args.gw,
            args.horizon,
            args.entry_json,
        )
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(report, end="")
    print(path)
