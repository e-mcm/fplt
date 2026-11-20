"""Command line entry point for fplt."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Literal

from fplt.backtest.predict_live import run_predict
from fplt.backtest.run import run_backtest
from fplt.data.fpl_client import FPLAPIError
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.optimise.decisions import resolve_squad_state, run_decisions, run_plan
from fplt.settings import get_settings
from fplt.types import GWKey

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
    predict = sub.add_parser("predict", help="Predict the next gameweeks")
    predict.add_argument("--model", required=True)
    predict.add_argument("--horizon", type=int, default=5)
    predict.add_argument("--season")
    predict.add_argument("--gw", type=int)
    plan = sub.add_parser("plan", help="Recommend a squad and transfers")
    plan.add_argument("--model", required=True)
    plan.add_argument("--season", required=True)
    plan.add_argument("--gw", type=int)
    plan.add_argument("--from-gw", type=int, default=1)
    plan.add_argument("--to-gw", type=int, default=38)
    plan.add_argument("--horizon", type=int, default=2)
    plan.add_argument("--decisions", action="store_true")
    plan.add_argument("--entry-json", type=Path)
    plan.add_argument(
        "--from-api",
        action="store_true",
        help=(
            "Load the live / public squad via resolve_squad_state "
            "(default when FPLT_ENTRY_ID is set; --entry-json still wins)"
        ),
    )
    args = parser.parse_args(argv)
    if args.command is None:
        return
    if args.command == "backtest":
        _backtest(args)
        return
    if args.command == "predict":
        _predict(args)
        return
    if args.command == "plan":
        _plan(args)
        return
    raise SystemExit(_not_implemented(args.command))


def _predict(args: argparse.Namespace) -> None:
    try:
        path = run_predict(
            get_settings().data_dir,
            args.model,
            args.horizon,
            season=args.season,
            gw=args.gw,
        )
    except (FileExistsError, FileNotFoundError, KeyError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(path)


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
        settings = get_settings()
        data_dir = settings.data_dir
        use_resolve = (
            args.from_api
            or args.entry_json is not None
            or settings.entry_id is not None
        )
        if args.from_api and settings.entry_id is None and args.entry_json is None:
            raise ValueError("fplt plan --from-api needs FPLT_ENTRY_ID or --entry-json")
        if use_resolve:
            # Same helper as automation catch_up / pre_deadline.
            view = PITData(DataStore(data_dir)).view(GWKey(args.season, args.gw))
            squad = resolve_squad_state(
                view,
                entry_json=args.entry_json,
                event=args.gw,
                settings=settings,
            )
            _solved, path, report = run_plan(
                data_dir,
                args.model,
                args.season,
                args.gw,
                args.horizon,
                squad=squad,
            )
        else:
            _solved, path, report = run_plan(
                data_dir,
                args.model,
                args.season,
                args.gw,
                args.horizon,
            )
    except (
        FileExistsError,
        FileNotFoundError,
        KeyError,
        ValueError,
        FPLAPIError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(report, end="")
    print(path)
