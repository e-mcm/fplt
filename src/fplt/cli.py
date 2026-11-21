"""Command line entry point for fplt."""

from __future__ import annotations

import argparse
import logging
import sys
from getpass import getpass
from pathlib import Path
from typing import Literal

from fplt.auth import clear_credentials, format_status, set_credentials
from fplt.auth import status as auth_status
from fplt.backtest.compare import CompareMode, compare_models, format_summary_table
from fplt.backtest.component_eval import (
    DEFAULT_SIMS,
    format_component_table,
    run_component_eval,
)
from fplt.backtest.eval_set import CORE_MODELS
from fplt.backtest.predict_live import run_predict
from fplt.backtest.run import run_backtest
from fplt.browser_auth import BrowserAuthError, auth_from_browser
from fplt.data.fpl_client import FPLAPIError
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.optimise.apply import ApplyError, apply_plan, format_apply_preview
from fplt.optimise.decisions import (
    PlanExistsError,
    explain_decision_gameweek,
    format_decision_scoreboard,
    resolve_squad_state,
    run_decision_bakeoff,
    run_decisions,
    run_plan,
)
from fplt.optimise.plan import DEFAULT_PLAN_HORIZON
from fplt.optimise.realised import DEFAULT_MAX_MEAN_PPG
from fplt.optimise.state import load_squad_state, rebuild_squad_state
from fplt.settings import get_settings
from fplt.types import GWKey
from fplt.utils.logging import configure_logging

_REFITS = ("every_gw", "every_n_gw", "never")
_COMPARE_MODES = ("walk_forward", "fixed")
logger = logging.getLogger(__name__)


def _not_implemented(name: str) -> int:
    print(f"fplt {name} is not implemented yet", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch a subcommand."""
    log_flags = argparse.ArgumentParser(add_help=False)
    log_flags.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logs (HTTP paths, per-gameweek steps)",
    )
    log_flags.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="WARNING+ only",
    )
    parser = argparse.ArgumentParser(prog="fplt", parents=[log_flags])
    sub = parser.add_subparsers(dest="command")
    backtest = sub.add_parser(
        "backtest",
        parents=[log_flags],
        help="Walk-forward backtest a model",
    )
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
    backtest.add_argument(
        "--time-decay",
        type=float,
        help=(
            "Override TrainWindow.time_decay (calendar λ, inverse days). "
            "LightGBM sample weights; Dixon-Coles keeps using xi."
        ),
    )
    backtest.add_argument("--steps", type=int, default=1)
    compare = sub.add_parser(
        "compare",
        parents=[log_flags],
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
    compare.add_argument(
        "--time-decay",
        type=float,
        help=(
            "Override TrainWindow.time_decay (calendar λ, inverse days). "
            "LightGBM sample weights; Dixon-Coles keeps using xi."
        ),
    )
    compare.add_argument("--steps", type=int, default=1)
    compare.add_argument("--with-openfpl", action="store_true")
    compare.add_argument("--openfpl-model-dir")
    compare.add_argument(
        "--output-dir",
        type=Path,
        help="Write summary JSON/parquet here (default: $FPLT_DATA_DIR/compare/...)",
    )
    predict = sub.add_parser(
        "predict",
        parents=[log_flags],
        help="Predict the next gameweeks",
    )
    predict.add_argument("--model", required=True)
    predict.add_argument("--horizon", type=int, default=5)
    predict.add_argument("--season")
    predict.add_argument("--gw", type=int)
    plan = sub.add_parser(
        "plan",
        parents=[log_flags],
        help="Recommend a squad and transfers",
    )
    plan.add_argument("--model", required=True)
    plan.add_argument("--season", required=True)
    plan.add_argument("--gw", type=int)
    plan.add_argument("--from-gw", type=int, default=1)
    plan.add_argument("--to-gw", type=int, default=38)
    plan.add_argument("--horizon", type=int, default=DEFAULT_PLAN_HORIZON)
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
    plan.add_argument(
        "--rebuild",
        action="store_true",
        help=(
            "Draft fifteen from an empty squad using today's budget "
            "(bank + selling prices). Ignores transfer hits; needs selling "
            "prices from my-team or --entry-json when not using YAML owned"
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
        default=DEFAULT_PLAN_HORIZON,
        help=(
            f"Horizon used when the plan was written (default: {DEFAULT_PLAN_HORIZON})"
        ),
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
    decisions = sub.add_parser(
        "decisions",
        parents=[log_flags],
        help="Multi-model historical decision bakeoff scoreboard",
    )
    decisions.add_argument("--season", required=True)
    decisions.add_argument("--from-gw", type=int, default=1)
    decisions.add_argument("--to-gw", type=int, default=38)
    decisions.add_argument("--horizon", type=int, default=DEFAULT_PLAN_HORIZON)
    decisions.add_argument(
        "--models",
        nargs="+",
        default=list(CORE_MODELS),
        help="YAML spec names (default: CORE_MODELS)",
    )
    decisions.add_argument(
        "--summarise-only",
        action="store_true",
        help="Score existing ledger/decisions rows; skip re-simulating",
    )
    decisions.add_argument(
        "--explain-gw",
        type=int,
        help="Dump XI / captain arithmetic for one gameweek and exit",
    )
    decisions.add_argument(
        "--explain-model",
        help="YAML model name for --explain-gw (default: first --models entry)",
    )
    decisions.add_argument(
        "--max-mean-ppg",
        type=float,
        default=DEFAULT_MAX_MEAN_PPG,
        help=(
            "Fail when any model's mean pts/GW exceeds this "
            f"(default: {DEFAULT_MAX_MEAN_PPG:.0f}; 0 disables)"
        ),
    )
    eval_components = sub.add_parser(
        "eval-components",
        parents=[log_flags],
        help="Layered component metrics (RPS, minutes Brier, involvement, n_sims)",
    )
    eval_components.add_argument("--season", required=True)
    eval_components.add_argument("--from-gw", type=int, default=1)
    eval_components.add_argument("--to-gw", type=int, default=10)
    eval_components.add_argument(
        "--fit-gw",
        type=int,
        help="Deadline for the n_sims sweep fit (default: from-gw)",
    )
    eval_components.add_argument(
        "--score-models",
        nargs="+",
        help="Score YAML specs (default: dixon_coles_v1 poisson_score_v1)",
    )
    eval_components.add_argument(
        "--minutes-model",
        default="empirical_minutes_v1",
        help="Minutes component YAML name",
    )
    eval_components.add_argument(
        "--involvement-model",
        default="involvement_shares_v1",
        help="Involvement component YAML name",
    )
    eval_components.add_argument(
        "--layered-model",
        default="dc_layered_v1",
        help="Composite YAML for the n_sims sweep",
    )
    eval_components.add_argument(
        "--sims",
        nargs="+",
        type=int,
        help=(
            "n_sims values to sweep "
            f"(default: {' '.join(str(n) for n in DEFAULT_SIMS)})"
        ),
    )
    eval_components.add_argument(
        "--skip-sims",
        action="store_true",
        help="Skip the Monte Carlo n_sims sweep",
    )
    eval_components.add_argument(
        "--output-dir",
        type=Path,
        help="Write summary JSON/parquet here",
    )
    auth = sub.add_parser(
        "auth",
        parents=[log_flags],
        help="Store FPL live-team token outside the repo (~/.config/fplt/env)",
    )
    auth_sub = auth.add_subparsers(dest="auth_command")
    auth_sub.add_parser(
        "status",
        parents=[log_flags],
        help="Show stored entry id and token expiry (default)",
    )
    auth_set = auth_sub.add_parser(
        "set",
        parents=[log_flags],
        help="Paste DevTools Bearer token into ~/.config/fplt/env (never the repo)",
    )
    auth_set.add_argument(
        "--token",
        help="Bearer value (or - to read stdin). Prompted when omitted.",
    )
    auth_set.add_argument(
        "--entry-id",
        type=int,
        help="FPL manager entry id (FPLT_ENTRY_ID)",
    )
    auth_clear = auth_sub.add_parser(
        "clear",
        parents=[log_flags],
        help="Remove FPLT_FPL_API_TOKEN from the user env file",
    )
    auth_clear.add_argument(
        "--also-entry-id",
        action="store_true",
        help="Also remove FPLT_ENTRY_ID",
    )
    auth_browser = auth_sub.add_parser(
        "from-browser",
        parents=[log_flags],
        help=(
            "Read FPL oidc.user access_token from local Chrome/Firefox "
            "localStorage into ~/.config/fplt/env (no password; close "
            "browser if the DB is locked)"
        ),
    )
    auth_browser.add_argument(
        "--browser",
        choices=("auto", "chrome", "chromium", "edge", "brave", "firefox"),
        default="auto",
        help="Browser to search (default: auto)",
    )
    auth_browser.add_argument(
        "--profile",
        help="Profile directory name (e.g. Default, Profile 1)",
    )
    auth_browser.add_argument(
        "--entry-id",
        type=int,
        help="FPL manager entry id (FPLT_ENTRY_ID)",
    )
    auth_browser.add_argument(
        "--no-copy",
        action="store_true",
        help="Read Chromium LevelDB in place (default copies to avoid locks)",
    )
    args = parser.parse_args(argv)
    if args.command is None:
        return
    configure_logging(verbose=bool(args.verbose), quiet=bool(args.quiet))
    if args.command == "auth":
        logger.info("fplt auth")
        _auth(args)
        return
    logger.info("fplt %s (data_dir=%s)", args.command, get_settings().data_dir)
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
    if args.command == "decisions":
        _decisions(args)
        return
    if args.command == "eval-components":
        _eval_components(args)
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
        logger.error("predict failed: %s", exc)
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
            time_decay=args.time_decay,
            steps=args.steps,
        )
    except (ValueError, KeyError) as exc:
        logger.error("backtest failed: %s", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    for spec in ran:
        window = spec.train_window.seasons_back
        decay = spec.train_window.time_decay
        print(
            f"{spec.name} refit={spec.refit} seasons_back={window} "
            f"time_decay={decay} model_id={spec.model_id}"
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
            time_decay=args.time_decay,
            modes=modes,
            steps=args.steps,
            with_openfpl=bool(args.with_openfpl),
            openfpl_model_dir=args.openfpl_model_dir,
            output_dir=args.output_dir,
        )
    except (ValueError, KeyError) as exc:
        logger.error("compare failed: %s", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    for run in result.runs:
        window = run.spec.train_window.seasons_back
        decay = run.spec.train_window.time_decay
        print(
            f"{run.model_name} mode={run.mode} refit={run.spec.refit} "
            f"seasons_back={window} time_decay={decay} "
            f"model_id={run.spec.model_id}"
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
            logger.info(
                "resolving squad (entry_id=%s session=%s entry_json=%s)",
                settings.entry_id,
                "set" if settings.fpl_session else "unset",
                args.entry_json,
            )
            view = PITData(DataStore(data_dir)).view(GWKey(args.season, args.gw))
            squad = resolve_squad_state(
                view,
                entry_json=args.entry_json,
                event=args.gw,
                settings=settings,
            )
        else:
            logger.info("planning from YAML squad (no FPLT_ENTRY_ID)")
            squad = load_squad_state()
        if args.rebuild:
            owned_n = len(squad.owned)
            squad = rebuild_squad_state(squad)
            logger.info(
                "rebuild draft: cleared %s owned; bank=%s (ITB + selling prices)",
                owned_n,
                squad.bank,
            )
        _solved, path, report = run_plan(
            data_dir,
            args.model,
            args.season,
            args.gw,
            args.horizon,
            squad=squad,
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
        logger.error("plan failed: %s", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(report, end="")
    print(path)
    print(path.parent / "report.md")


def _eval_components(args: argparse.Namespace) -> None:
    sims: list[int] | tuple[int, ...] | None
    if args.skip_sims:
        sims = []
    elif args.sims is not None:
        sims = list(args.sims)
    else:
        sims = None
    try:
        result = run_component_eval(
            get_settings().data_dir,
            args.season,
            from_gw=int(args.from_gw),
            to_gw=int(args.to_gw),
            fit_gw=args.fit_gw,
            score_models=list(args.score_models) if args.score_models else None,
            minutes_model=str(args.minutes_model),
            involvement_model=str(args.involvement_model),
            layered_model=str(args.layered_model),
            sims=sims,
            output_dir=args.output_dir,
        )
    except (ValueError, KeyError) as exc:
        logger.error("eval-components failed: %s", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(format_component_table(result.summary))
    if not result.sims.is_empty():
        print(result.sims)
    print(f"wrote {result.output_dir / 'summary.json'}")


def _decisions(args: argparse.Namespace) -> None:
    data_dir = get_settings().data_dir
    if args.explain_gw is not None:
        model_name = args.explain_model or list(args.models)[0]
        try:
            print(
                explain_decision_gameweek(
                    data_dir, model_name, args.season, int(args.explain_gw)
                )
            )
        except (KeyError, ValueError) as exc:
            logger.error("decisions explain failed: %s", exc)
            print(str(exc), file=sys.stderr)
            raise SystemExit(2) from exc
        return
    max_mean = float(args.max_mean_ppg)
    try:
        summary = run_decision_bakeoff(
            data_dir,
            list(args.models),
            args.season,
            args.from_gw,
            args.to_gw,
            args.horizon,
            simulate=not bool(args.summarise_only),
            max_mean_ppg=None if max_mean <= 0 else max_mean,
        )
    except (KeyError, ValueError) as exc:
        logger.error("decisions failed: %s", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(format_decision_scoreboard(summary))
    path = data_dir / "ledger" / "decisions" / "part.parquet"
    print(path)


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


def _auth(args: argparse.Namespace) -> None:
    command = getattr(args, "auth_command", None) or "status"
    try:
        if command == "status":
            print(format_status(auth_status()), end="")
            return
        if command == "set":
            token = _read_auth_token(args.token)
            result = set_credentials(token=token, entry_id=args.entry_id)
            get_settings.cache_clear()
            print(format_status(result), end="")
            if result.token is not None and result.token.expired is True:
                print(
                    "warning: pasted token already looks expired",
                    file=sys.stderr,
                )
            return
        if command == "clear":
            result = clear_credentials(
                clear_entry_id=bool(args.also_entry_id),
            )
            get_settings.cache_clear()
            print(format_status(result), end="")
            return
        if command == "from-browser":
            result = auth_from_browser(
                browser=args.browser,
                profile=args.profile,
                entry_id=args.entry_id,
                copy_locked=not bool(args.no_copy),
            )
            get_settings.cache_clear()
            print(format_status(result), end="")
            if result.token is not None and result.token.expired is True:
                print(
                    "warning: browser token already looks expired; "
                    "refresh FPL in the browser and retry",
                    file=sys.stderr,
                )
            return
    except (BrowserAuthError, OSError, ValueError) as exc:
        logger.error("auth failed: %s", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    raise SystemExit(_not_implemented(f"auth {command}"))


def _read_auth_token(raw: str | None) -> str:
    if raw == "-":
        token = sys.stdin.read()
    elif raw is not None:
        token = raw
    elif sys.stdin.isatty():
        print(
            "Paste the Bearer value from DevTools → Network → my-team → "
            "request header x-api-authorization (token only, not 'Bearer ').",
            file=sys.stderr,
        )
        token = getpass("FPL API token: ")
    else:
        token = sys.stdin.read()
    if not token or not token.strip():
        raise ValueError("empty token")
    return token
