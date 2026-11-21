"""Command line entry point for fplt."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Literal

from fplt.backtest.compare import CompareMode, compare_models, format_summary_table
from fplt.backtest.eval_set import CORE_MODELS
from fplt.backtest.predict_live import run_predict
from fplt.backtest.run import run_backtest
from fplt.data.fpl_client import FPLAPIError
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.optimise.apply import ApplyError, apply_plan, format_apply_preview
from fplt.optimise.decisions import (
    PlanExistsError,
    resolve_squad_state,
    run_decisions,
    run_plan,
)
from fplt.settings import get_settings
from fplt.types import GWKey

_REFITS = ("every_gw", "every_n_gw", "never")
_COMPARE_MODES = ("walk_forward", "fixed")


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
    compare = sub.add_parser(
        "compare",
        help="Compare models under shared windows (walk-forward + fixed)",
    )
    compare.add_argument("--season", required=True)
    compare.add_argument("--test-from-gw", type=int, default=1)
    compare.add_argument("--test-to-gw", type=int, default=38)
    compare.add_argument(
        "--fit-gw",
        type=int,
        help="Fixed-mode fit deadline (default: test-from-gw)",
    )
    compare.add_argument(
        "--models",
        nargs="+",
        help=f"Override the default set ({', '.join(CORE_MODELS)})",
    )
    compare.add_argument(
        "--modes",
        nargs="+",
        choices=_COMPARE_MODES,
        help="Subset of walk_forward / fixed (default: both)",
    )
    compare.add_argument("--seasons-back", type=int)
    compare.add_argument("--steps", type=int, default=1)
    compare.add_argument("--with-openfpl", action="store_true")
    compare.add_argument("--openfpl-model-dir")
    compare.add_argument(
        "--output-dir",
        type=Path,
        help="Write summary JSON/parquet here (default: $FPLT_DATA_DIR/compare/...)",
    )
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
    apply = sub.add_parser(
        "apply",
        help="Dry-run or POST a saved plan's transfers/lineup to FPL",
    )
    apply.add_argument("--model", required=True)
    apply.add_argument("--season", required=True)
    apply.add_argument("--gw", type=int, required=True)
    apply.add_argument(
        "--horizon",
        type=int,
        default=2,
        help="Horizon used when the plan was written (default: 2)",
    )
    apply.add_argument(
        "--confirm",
        action="store_true",
        help="POST to FPL (default is dry-run). Needs FPLT_FPL_API_TOKEN.",
    )
    apply.add_argument(
        "--transfers-only",
        action="store_true",
        help="Skip the lineup POST",
    )
    apply.add_argument(
        "--lineup-only",
        action="store_true",
        help="Skip the transfers POST",
    )
    args = parser.parse_args(argv)
    if args.command is None:
        return
    if args.command == "backtest":
        _backtest(args)
        return
    if args.command == "compare":
        _compare(args)
        return
    if args.command == "predict":
        _predict(args)
        return
    if args.command == "plan":
        _plan(args)
        return
    if args.command == "apply":
        _apply(args)
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


def _configure_cli_logging() -> None:
    """Show library progress on stderr. Attaches one handler for the process."""
    package = logging.getLogger("fplt")
    package.setLevel(logging.INFO)
    if any(getattr(handler, "_fplt_cli", False) for handler in package.handlers):
        return
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    )
    handler._fplt_cli = True  # type: ignore[attr-defined]
    package.addHandler(handler)


def _compare(args: argparse.Namespace) -> None:
    _configure_cli_logging()
    modes: list[CompareMode] | None = (
        list(args.modes) if args.modes is not None else None
    )
    try:
        result = compare_models(
            get_settings().data_dir,
            args.season,
            models=list(args.models) if args.models else None,
            test_from_gw=args.test_from_gw,
            test_to_gw=args.test_to_gw,
            fit_gw=args.fit_gw,
            seasons_back=args.seasons_back,
            modes=modes,
            steps=args.steps,
            with_openfpl=bool(args.with_openfpl),
            openfpl_model_dir=args.openfpl_model_dir,
            output_dir=args.output_dir,
        )
    except (ValueError, KeyError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    for run in result.runs:
        window = run.spec.train_window.seasons_back
        print(
            f"{run.model_name} mode={run.mode} refit={run.spec.refit} "
            f"seasons_back={window} model_id={run.spec.model_id}"
        )
    print(format_summary_table(result.summary))
    print(f"wrote {result.output_dir / 'summary.json'}")
    if result.skipped_openfpl:
        print(result.skipped_openfpl, file=sys.stderr)


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
    except PlanExistsError as exc:
        print(exc.report, end="")
        print(f"plan already exists: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
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
    print(path.parent / "report.md")


def _apply(args: argparse.Namespace) -> None:
    try:
        result = apply_plan(
            get_settings().data_dir,
            args.model,
            args.season,
            args.gw,
            horizon=int(args.horizon),
            confirm=bool(args.confirm),
            transfers_only=bool(args.transfers_only),
            lineup_only=bool(args.lineup_only),
        )
    except (
        ApplyError,
        FPLAPIError,
        FileNotFoundError,
        KeyError,
        ValueError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(format_apply_preview(result), end="")
