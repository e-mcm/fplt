"""Command line entry point for fplt."""

from __future__ import annotations

import argparse
import logging
import sys
from getpass import getpass
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import ValidationError

from fplt.auth import (
    check_token,
    clear_credentials,
    format_auth_check,
    format_status,
    set_credentials,
)
from fplt.auth import status as auth_status
from fplt.backtest.archive_compare import (
    DEFAULT_SOURCES,
    compare_to_archives,
    format_archive_board,
)
from fplt.backtest.bakeoff import (
    bakeoff_from_models,
    execute_bakeoff,
    format_bakeoff_plan,
    load_bakeoff_spec,
    plan_bakeoff,
)
from fplt.backtest.bakeoff import (
    plan_to_json as bakeoff_plan_to_json,
)
from fplt.backtest.compare import CompareMode, compare_models, format_summary_table
from fplt.backtest.component_eval import (
    DEFAULT_SIMS,
    format_component_table,
    run_component_eval,
)
from fplt.backtest.eval_set import CORE_MODELS
from fplt.backtest.horizon_skill import format_skill_curve, run_horizon_skill
from fplt.backtest.horizons import DEFAULT_PROJECTION_HORIZON, DEFAULT_SKILL_STEPS
from fplt.backtest.minutes_measurement import (
    format_misses_markdown,
    run_minutes_measurement,
)
from fplt.backtest.minutes_measurement import (
    format_summary_table as format_minutes_measurement_table,
)
from fplt.backtest.other_points_measurement import (
    format_captain_regret_markdown,
    run_other_points_measurement,
)
from fplt.backtest.other_points_measurement import (
    format_summary_table as format_other_points_measurement_table,
)
from fplt.backtest.predict_live import run_predict
from fplt.backtest.predicted_xi_audit import (
    DEFAULT_MIN_START_ACCURACY,
    format_audit_markdown,
    run_predicted_xi_audit,
)
from fplt.backtest.predicted_xi_audit import (
    format_summary_table as format_predicted_xi_audit_table,
)
from fplt.backtest.produce_ledger import StaleLedgerCacheError
from fplt.backtest.promote_eval import (
    execute_promote_eval,
    format_plan,
    load_eval_spec,
    plan_promote_eval,
    plan_to_json,
)
from fplt.backtest.run import run_backtest
from fplt.backtest.simulated_season import (
    format_simulated_season_report,
    run_chip_arm_sim_season,
)
from fplt.browser_auth import BrowserAuthError, auth_from_browser
from fplt.data.benchmark_projections import (
    BenchmarkArchiveError,
    archive_deadline_benchmarks,
    archive_if_missing,
)
from fplt.data.fpl_client import FPLAPIError
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.optimise.apply import ApplyError, apply_plan, format_apply_preview
from fplt.optimise.attainable import (
    ORACLE_MODEL_ID,
    enrich_decision_summary,
    run_hindsight_oracle,
)
from fplt.optimise.decisions import (
    DEFAULT_LIVE_CHIP_ARM,
    DEFAULT_LIVE_PRICE_FORECAST_METHOD,
    PlanExistsError,
    explain_decision_gameweek,
    format_chip_decision_scoreboard,
    format_decision_scoreboard,
    read_decisions,
    resolve_squad_state,
    run_chip_decision_board,
    run_decision_bakeoff,
    run_decisions,
    run_plan,
)
from fplt.optimise.explain import run_explain
from fplt.optimise.horizon_decay import HorizonDecayMethod, resolve_step_weights
from fplt.optimise.plan import DEFAULT_PLAN_HORIZON
from fplt.optimise.realised import DEFAULT_MAX_MEAN_PPG
from fplt.optimise.regret_board import run_regret_board, write_regret_board
from fplt.optimise.state import load_squad_state, rebuild_squad_state
from fplt.settings import SettingsConfigError, get_settings, secret_value
from fplt.types import GWKey
from fplt.utils.logging import configure_logging
from fplt.utils.process import run_command

_REFITS = ("every_gw", "every_n_gw", "never")
_COMPARE_MODES = ("walk_forward", "fixed")
logger = logging.getLogger(__name__)


def _not_implemented(name: str) -> int:
    print(f"fplt {name} is not implemented yet", file=sys.stderr)
    return 2


def _exit_config_error(exc: BaseException) -> None:
    """Log a config error and exit 2 without a traceback.

    Disables ``logging.raiseExceptions`` for the emit so a broken handler
    (seen under xdist + caplog) cannot dump ``--- Logging error ---`` plus
    the exception class name onto operator stderr.
    """
    previous = logging.raiseExceptions
    logging.raiseExceptions = False
    try:
        logger.error("%s", exc)
    finally:
        logging.raiseExceptions = previous
    print(str(exc), file=sys.stderr)
    raise SystemExit(2) from None


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
    log_flags.add_argument(
        "--log-format",
        choices=("text", "json"),
        default=None,
        help=(
            "Log line format (default: text, or FPLT_LOG_FORMAT). "
            "Workflows set FPLT_LOG_FORMAT=json."
        ),
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
        help=(
            "Compare models under shared windows, or mint a comparison "
            "experiment via repeated --spec (storage; no verdict)"
        ),
    )
    compare.add_argument(
        "--season",
        help="Season for walk-forward compare (required unless --spec)",
    )
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
        "--spec",
        action="append",
        dest="specs",
        metavar="NAME",
        help=(
            "ModelSpec name for a comparison-flavour experiment "
            "(repeat ≥2; pins latest build + as-of, no walk-forward)"
        ),
    )
    compare.add_argument(
        "--gw",
        type=int,
        help="As-of gameweek when minting a --spec comparison (default: 1)",
    )
    compare.add_argument(
        "--question",
        help="Optional one-line question for a --spec comparison experiment",
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
    compare.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_PROJECTION_HORIZON,
        help="Horizon for --spec comparison experiments (default: projection H)",
    )
    compare.add_argument("--with-openfpl", action="store_true")
    compare.add_argument("--openfpl-model-dir")
    compare.add_argument(
        "--output-dir",
        type=Path,
        help="Write summary JSON/parquet here (default: $FPLT_DATA_DIR/compare/...)",
    )
    compare.add_argument(
        "--data-dir",
        type=Path,
        help="Override FPLT_DATA_DIR when minting a --spec comparison",
    )
    predict = sub.add_parser(
        "predict",
        parents=[log_flags],
        help="Predict the next gameweeks",
    )
    predict.add_argument("--model", required=True)
    predict.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_PROJECTION_HORIZON,
        help=(
            "Gameweeks to project starting at the as-of "
            f"(default: {DEFAULT_PROJECTION_HORIZON}; plan default stays "
            f"{DEFAULT_PLAN_HORIZON})"
        ),
    )
    predict.add_argument("--season")
    predict.add_argument("--gw", type=int)
    plan = sub.add_parser(
        "plan",
        parents=[log_flags],
        help=(
            f"Recommend a squad and transfers (default chips: {DEFAULT_LIVE_CHIP_ARM})"
        ),
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
    plan.add_argument(
        "--chip-arm",
        choices=["off", "rule_based", "optimised"],
        default=DEFAULT_LIVE_CHIP_ARM,
        help=(
            "Chip policy for live planning: optimised runs the season chip "
            "scheduler then MILP (default); off disables chips; rule_based "
            "forces landmark windows. Ignored with --decisions (stays chips-off)"
        ),
    )
    plan.add_argument(
        "--chip-rules-season",
        help=(
            "Chip-rules year for --chip-arm (default: --season). "
            "Ignored when --chip-arm=off or with --decisions"
        ),
    )
    plan.add_argument(
        "--no-force-play",
        action="store_true",
        help=(
            "Disable half-window force-packing in the optimised chip "
            "scheduler (ablation; default keeps force-play on). Ignored "
            "when --chip-arm is not optimised or with --decisions"
        ),
    )
    plan.add_argument(
        "--joint-scenarios",
        action="store_true",
        help=(
            "Fit the layered model and pass joint MC samples into step-0 "
            "lineup SAA (FDB-241; opt-in until a keep flip). Ignored with "
            "--decisions"
        ),
    )
    plan.add_argument(
        "--n-scenario-sims",
        type=int,
        default=None,
        help="Draw count for --joint-scenarios (default: 500)",
    )
    plan.add_argument(
        "--horizon-decay",
        choices=["geometric", "skill"],
        default="geometric",
        help=(
            "Planner step weights: geometric decay (live default) or "
            "measured skill-by-horizon shrink (FDB-242 ablation)"
        ),
    )
    explain = sub.add_parser(
        "explain",
        parents=[log_flags],
        help="Forecast board + plan/transfer deltas vs current squad",
    )
    explain.add_argument(
        "--model",
        default="dc_layered_v4",
        help="YAML spec name (default: dc_layered_v4)",
    )
    explain.add_argument("--season", required=True)
    explain.add_argument("--gw", type=int, required=True)
    explain.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_PLAN_HORIZON,
        help=(
            f"Horizon used when the plan was written (default: {DEFAULT_PLAN_HORIZON})"
        ),
    )
    explain.add_argument(
        "--top",
        type=int,
        default=30,
        help="Top-N players for forecast-only mode (ignored when a plan exists)",
    )
    explain.add_argument(
        "--forecast-only",
        action="store_true",
        help="Skip decision/transfer deltas (no plan required)",
    )
    explain.add_argument(
        "--decision-only",
        action="store_true",
        help="Skip the forecast board (plan required)",
    )
    explain.add_argument(
        "--player",
        type=int,
        default=None,
        help="Player key for fixture/involvement depth (skips plan sections)",
    )
    explain.add_argument(
        "--vs",
        type=int,
        default=None,
        dest="vs_player",
        help="Sold player key for A→B transfer-pair depth (requires --player as in)",
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
    decisions.add_argument(
        "--with-oracle",
        action="store_true",
        help=(
            "Run the constrained hindsight oracle and append attainable "
            "share columns to the scoreboard"
        ),
    )
    decisions.add_argument(
        "--chip-board",
        action="store_true",
        help=(
            "Three-arm chip board: chips-off vs rule-based vs optimised MILP "
            "(season points; does not run the full FDB-120 bakeoff)"
        ),
    )
    decisions.add_argument(
        "--chip-rules-season",
        help=(
            "Chip-rules year for --chip-board (default: --season). "
            "Use 2025-26 on a 2024-25 fixture season to label a counterfactual"
        ),
    )
    decisions.add_argument(
        "--chip-arms",
        nargs="+",
        choices=["off", "rule_based", "optimised"],
        default=["off", "rule_based", "optimised"],
        help="Chip arms for --chip-board (default: all three)",
    )
    decisions.add_argument(
        "--no-force-play",
        action="store_true",
        help=(
            "Disable half-window force-packing of remaining chips in the "
            "optimised chip scheduler (ablation; default keeps force-play on)"
        ),
    )
    decisions.add_argument(
        "--no-lineup-reopt",
        action="store_true",
        help=(
            "Keep the raw MILP XI / captain / vice (ablation; default re-opts "
            "lineups via EV after the transfer solve)"
        ),
    )
    decisions.add_argument(
        "--joint-scenarios",
        action="store_true",
        help=(
            "Pass joint MC samples into step-0 lineup SAA (FDB-241; treatment "
            "arm vs default EV reopt). Requires a sample()-capable model"
        ),
    )
    decisions.add_argument(
        "--n-scenario-sims",
        type=int,
        default=None,
        help="Draw count for --joint-scenarios (default: 500)",
    )
    decisions.add_argument(
        "--price-forecast",
        choices=["flat", "mean_gw_delta"],
        default=None,
        help=(
            "Within-horizon list-price forecast for plan_squad "
            "(default: off / as-of flat). mean_gw_delta uses PIT deadline "
            "price history; flat matches omitting the flag"
        ),
    )
    decisions.add_argument(
        "--horizon-decay",
        choices=["geometric", "skill"],
        default="geometric",
        help=(
            "Planner step weights: geometric SquadState.decay (live control) "
            "or measured skill-by-horizon RMSE shrink (FDB-242 treatment; "
            "configs/optimise/horizon_skill_weights.yaml)"
        ),
    )
    regret_board = sub.add_parser(
        "regret-board",
        parents=[log_flags],
        help=(
            "Regret-by-component board: captain / XI-bench / transfers / "
            "chip timing vs constrained attainable (north-star §2)"
        ),
    )
    regret_board.add_argument("--season", required=True)
    regret_board.add_argument("--from-gw", type=int, default=1)
    regret_board.add_argument("--to-gw", type=int, default=38)
    regret_board.add_argument("--horizon", type=int, default=DEFAULT_PLAN_HORIZON)
    regret_board.add_argument(
        "--model",
        default="dc_layered_v2",
        help="YAML policy model (default: dc_layered_v2)",
    )
    regret_board.add_argument(
        "--chip-arm",
        choices=["off", "rule_based", "optimised"],
        default=DEFAULT_LIVE_CHIP_ARM,
        help=f"Policy chip arm (default: {DEFAULT_LIVE_CHIP_ARM})",
    )
    regret_board.add_argument(
        "--chip-rules-season",
        help="Chip-rules year (default: --season)",
    )
    regret_board.add_argument(
        "--no-force-play",
        action="store_true",
        help="Disable force-play on the policy (and oracle unless overridden)",
    )
    regret_board.add_argument(
        "--no-lineup-reopt",
        action="store_true",
        help="Keep raw MILP XI / captain (ablation)",
    )
    regret_board.add_argument(
        "--price-forecast",
        choices=["flat", "mean_gw_delta"],
        default=DEFAULT_LIVE_PRICE_FORECAST_METHOD,
        help=(
            "Policy within-horizon price forecast "
            f"(default: {DEFAULT_LIVE_PRICE_FORECAST_METHOD})"
        ),
    )
    regret_board.add_argument(
        "--oracle-chip-arm",
        choices=["off", "rule_based", "optimised"],
        default=None,
        help="Oracle chip arm (default: same as --chip-arm)",
    )
    regret_board.add_argument(
        "--oracle-no-force-play",
        action="store_true",
        help="Oracle force-play off (hindsight ceiling ablation)",
    )
    regret_board.add_argument(
        "--summarise-only",
        action="store_true",
        help="Reuse ledger policy + oracle rows; skip re-simulating",
    )
    regret_board.add_argument(
        "--policy-parquet",
        type=Path,
        help="Load policy decisions from a parquet (skips policy simulate)",
    )
    regret_board.add_argument(
        "--output-dir",
        type=Path,
        help="Write report.md / summary parquet / meta.json here",
    )
    weekly_report = sub.add_parser(
        "weekly-report",
        parents=[log_flags],
        help=(
            "Post-GW decision scoring report (ops-08): realised vs plan EV, "
            "hindsight regret, captaincy, hit payback, points-at-rank"
        ),
    )
    weekly_report.add_argument("--season", required=True)
    weekly_report.add_argument("--gw", type=int, required=True)
    weekly_report.add_argument(
        "--model-id",
        help="Ledger model_id filter (default: all models for the GW)",
    )
    weekly_report.add_argument(
        "--hindsight-best",
        type=int,
        help="Constrained-hindsight net points for this GW (else local XV opt)",
    )
    weekly_report.add_argument(
        "--n-fixtures",
        type=int,
        help="Fixture count for blank/double tagging (0=blank, >10=double)",
    )
    weekly_report.add_argument(
        "--no-notify",
        action="store_true",
        help="Write the report without sending ntfy/SMTP",
    )
    sim_season = sub.add_parser(
        "sim-season",
        parents=[log_flags],
        help=(
            "Chip-arm simulated-season SE (N6): CRN Monte Carlo season totals "
            "and paired standard errors for off / rule_based / optimised"
        ),
    )
    sim_season.add_argument("--season", required=True)
    sim_season.add_argument("--from-gw", type=int, default=1)
    sim_season.add_argument(
        "--to-gw",
        type=int,
        default=3,
        help="Last gameweek inclusive (default: 3 for a short smoke window)",
    )
    sim_season.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_PLAN_HORIZON,
        help=f"Planner look-ahead (default: {DEFAULT_PLAN_HORIZON})",
    )
    sim_season.add_argument(
        "--models",
        nargs="+",
        default=list(CORE_MODELS)[:1],
        help="YAML spec names (default: first CORE_MODELS entry)",
    )
    sim_season.add_argument(
        "--chip-arms",
        nargs="+",
        choices=["off", "rule_based", "optimised"],
        default=["off", "rule_based", "optimised"],
        help="Chip arms to CRN-compare (default: all three)",
    )
    sim_season.add_argument(
        "--chip-rules-season",
        help="Chip-rules year (default: --season)",
    )
    sim_season.add_argument(
        "--n-paths",
        type=int,
        default=5,
        help="Monte Carlo paths (default: 5; keep small for smoke)",
    )
    sim_season.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Master seed for CRN path draws (default: 0)",
    )
    sim_season.add_argument(
        "--with-attainable",
        action="store_true",
        help="Also report share-of-attainable SEs (N5 chips-off oracle per path)",
    )
    sim_season.add_argument(
        "--no-force-play",
        action="store_true",
        help=(
            "Disable half-window force-packing in the optimised chip "
            "scheduler (ablation; default keeps force-play on). Same-H "
            "force ablation; for H×force CRN pairs (promote case E) build "
            "two DecisionPolicy rows with per-policy horizon/force_play"
        ),
    )
    eval_components = sub.add_parser(
        "eval-components",
        parents=[log_flags],
        help=(
            "Layered component metrics (RPS, minutes Brier, involvement, "
            "n_sims) plus optional points skill-by-horizon"
        ),
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
        help=(
            "Score YAML specs "
            "(default: dixon_coles_v1 poisson_score_v1 odds_poisson_v1)"
        ),
    )
    eval_components.add_argument(
        "--minutes-model",
        nargs="+",
        default=["empirical_minutes_v1"],
        help="Minutes component YAML name(s) (default: empirical_minutes_v1)",
    )
    eval_components.add_argument(
        "--involvement-model",
        nargs="+",
        default=["involvement_shares_v1"],
        help=("Involvement component YAML name(s) (default: involvement_shares_v1)"),
    )
    eval_components.add_argument(
        "--layered-model",
        default="dc_layered_v1",
        help="Composite YAML for the n_sims sweep",
    )
    eval_components.add_argument(
        "--points-models",
        nargs="+",
        help=(
            "PointsPredictor YAML names for skill-by-horizon "
            f"(MAE/RMSE for H=1…--steps; default steps={DEFAULT_SKILL_STEPS})"
        ),
    )
    eval_components.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_SKILL_STEPS,
        help=(
            "Lead times for --points-models "
            f"(default: {DEFAULT_SKILL_STEPS}; ignored without --points-models)"
        ),
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
        "--minutes-only",
        action="store_true",
        help=(
            "Score only minutes Brier (skip score/involvement/sims). "
            "Defaults --minutes-model to empirical_minutes_v2 when omitted. "
            "Read slice source=fpl_api for snapshot-backed gameweeks."
        ),
    )
    eval_components.add_argument(
        "--output-dir",
        type=Path,
        help="Write summary JSON/parquet here",
    )
    minutes_measure = sub.add_parser(
        "minutes-measure",
        parents=[log_flags],
        help=(
            "Snapshot minutes diagnostics: availability/rotation/early-sub "
            "Brier+log, xPts-weighted Brier on the decision pool, weekly misses"
        ),
    )
    minutes_measure.add_argument("--season", required=True)
    minutes_measure.add_argument("--from-gw", type=int, default=1)
    minutes_measure.add_argument("--to-gw", type=int, default=10)
    minutes_measure.add_argument(
        "--minutes-model",
        default="empirical_minutes_v2",
        help="Minutes component YAML name (default: empirical_minutes_v2)",
    )
    minutes_measure.add_argument(
        "--top-misses-per-gw",
        type=int,
        default=25,
        help="Rows kept per GW in the misses report (default: 25)",
    )
    minutes_measure.add_argument(
        "--output-dir",
        type=Path,
        help="Write summary/misses parquet+JSON+markdown here",
    )
    other_measure = sub.add_parser(
        "other-measure",
        parents=[log_flags],
        help=(
            "Other-points / haul diagnostics: haul Brier+log, top xPts-decile "
            "haul calibration, match-context other residual, captain-regret proxy"
        ),
    )
    other_measure.add_argument("--season", required=True)
    other_measure.add_argument("--from-gw", type=int, default=1)
    other_measure.add_argument("--to-gw", type=int, default=10)
    other_measure.add_argument(
        "--points-model",
        default="dc_layered_v4",
        help="Layered points YAML name (default: dc_layered_v4)",
    )
    other_measure.add_argument(
        "--n-sims",
        type=int,
        default=None,
        help="Optional Monte Carlo override (faster smoke / measurement)",
    )
    other_measure.add_argument(
        "--output-dir",
        type=Path,
        help="Write summary/captain_regret parquet+JSON+markdown here",
    )
    predicted_xi_audit = sub.add_parser(
        "predicted-xi-audit",
        parents=[log_flags],
        help=(
            "Predicted XI accuracy vs official starts: precision/recall by "
            "source and hours-to-KO, bench→start rate, full-XI coverage"
        ),
    )
    predicted_xi_audit.add_argument("--season", required=True)
    predicted_xi_audit.add_argument("--from-gw", type=int, default=1)
    predicted_xi_audit.add_argument("--to-gw", type=int, default=38)
    predicted_xi_audit.add_argument(
        "--source",
        default=None,
        help="Optional predicted_lineups.source filter (e.g. touchline_wire)",
    )
    predicted_xi_audit.add_argument(
        "--min-accuracy",
        type=float,
        default=DEFAULT_MIN_START_ACCURACY,
        help=(
            "Start-precision gate for promote smoke "
            f"(default: {DEFAULT_MIN_START_ACCURACY})"
        ),
    )
    predicted_xi_audit.add_argument(
        "--output-dir",
        type=Path,
        help="Write summary/detail parquet+JSON+markdown here",
    )
    horizon_skill = sub.add_parser(
        "horizon-skill",
        parents=[log_flags],
        help=(
            "Walk-forward MAE/RMSE by lead time (H=1…N); wraps compare "
            f"(default steps={DEFAULT_SKILL_STEPS})"
        ),
    )
    horizon_skill.add_argument("--season", required=True)
    horizon_skill.add_argument("--from-gw", type=int, default=1)
    horizon_skill.add_argument("--to-gw", type=int, default=20)
    horizon_skill.add_argument(
        "--steps",
        type=int,
        default=DEFAULT_SKILL_STEPS,
        help=f"Lead times to score (default: {DEFAULT_SKILL_STEPS})",
    )
    horizon_skill.add_argument(
        "--models",
        nargs="+",
        help="PointsPredictor YAML names (default: compare core set)",
    )
    horizon_skill.add_argument("--seasons-back", type=int)
    horizon_skill.add_argument("--with-openfpl", action="store_true")
    horizon_skill.add_argument("--openfpl-model-dir")
    horizon_skill.add_argument(
        "--output-dir",
        type=Path,
        help="Write horizon_skill JSON/parquet here",
    )
    ensemble_sensitivity = sub.add_parser(
        "ensemble-sensitivity",
        parents=[log_flags],
        help=(
            "Shared-state DC vs LGBM (bakeoff pair) move-frequency board: "
            "how often week-1 transfers/XI/captain disagree"
        ),
    )
    ensemble_sensitivity.add_argument("--season", required=True)
    ensemble_sensitivity.add_argument("--from-gw", type=int, default=1)
    ensemble_sensitivity.add_argument("--to-gw", type=int, default=38)
    ensemble_sensitivity.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_PLAN_HORIZON,
        help=f"Planner horizon (default: {DEFAULT_PLAN_HORIZON})",
    )
    ensemble_sensitivity.add_argument(
        "--control",
        default="dc_layered_v4",
        help="YAML ModelSpec for the ownership path (default: dc_layered_v4)",
    )
    ensemble_sensitivity.add_argument(
        "--challengers",
        nargs="+",
        default=["lgbm_direct_v1"],
        help="YAML ModelSpec names to compare (default: lgbm_direct_v1)",
    )
    ensemble_sensitivity.add_argument(
        "--output-dir",
        type=Path,
        help="Write rows/summary parquet + summary.json here",
    )
    pruning_loss = sub.add_parser(
        "pruning-loss",
        parents=[log_flags],
        help=(
            "Measure week-1 action loss from candidate-pool pruning "
            "(reference vs N sweep + current policy)"
        ),
    )
    pruning_loss.add_argument(
        "--n-scenarios",
        type=int,
        default=24,
        help="Synthetic squad states to sample (default: 24; aim 20–30)",
    )
    pruning_loss.add_argument(
        "--seed",
        type=int,
        default=0,
        help="RNG seed for synthetic scenario sampling",
    )
    pruning_loss.add_argument(
        "--horizon",
        type=int,
        default=5,
        help="Plan horizon for each solve (default: 5)",
    )
    pruning_loss.add_argument(
        "--targets",
        type=int,
        nargs="+",
        help="Pool-size targets to sweep (default: 80 100 130 160 200 300)",
    )
    pruning_loss.add_argument(
        "--reference-time-limit",
        type=float,
        default=3600.0,
        help="HiGHS time limit for the full-pool reference (seconds)",
    )
    pruning_loss.add_argument(
        "--sweep-time-limit",
        type=float,
        default=600.0,
        help="HiGHS time limit for pruned arms (seconds)",
    )
    pruning_loss.add_argument(
        "--output-dir",
        type=Path,
        help="Write rows/summary parquet + summary.json here",
    )
    bakeoff = sub.add_parser(
        "bakeoff",
        parents=[log_flags],
        help=(
            "Multi-arm prediction championship on promote-eval/compare "
            "(shared ledger cache + SPA/MCS ranking; not a keep)"
        ),
    )
    bakeoff.add_argument(
        "--bakeoff",
        help="BakeoffSpec name under configs/evals/<name>.yaml",
    )
    bakeoff.add_argument(
        "--models",
        nargs="+",
        help="ModelSpec names (CLI mode; requires --control and --season)",
    )
    bakeoff.add_argument(
        "--control",
        help="SPA benchmark ModelSpec (required with --models)",
    )
    bakeoff.add_argument(
        "--name",
        default="cli_bakeoff",
        help="Artifact name when using --models (default: cli_bakeoff)",
    )
    bakeoff.add_argument("--season", help="Season for CLI --models mode")
    bakeoff.add_argument("--from-gw", type=int)
    bakeoff.add_argument("--to-gw", type=int)
    bakeoff.add_argument("--seasons-back", type=int, default=4)
    bakeoff.add_argument(
        "--modes",
        nargs="+",
        choices=_COMPARE_MODES,
        help="Compare modes (default: walk_forward)",
    )
    bakeoff.add_argument("--steps", type=int, default=1)
    bakeoff.add_argument(
        "--pairwise-evals",
        nargs="+",
        help="Optional EvalSpec names to run after the multi-arm compare",
    )
    bakeoff.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print the suite plan without fitting (default)",
    )
    bakeoff.add_argument(
        "--execute",
        action="store_true",
        help="Run compare + write ranking.md (SPA/MCS)",
    )
    bakeoff.add_argument(
        "--with-openfpl",
        action="store_true",
        help="Allow OpenFPL when listed in --models",
    )
    bakeoff.add_argument("--openfpl-model-dir")
    bakeoff.add_argument(
        "--output-dir",
        type=Path,
        help="Write plan/ranking/summary here",
    )
    bakeoff.add_argument(
        "--json",
        action="store_true",
        help="Print plan JSON instead of the text table (dry-run)",
    )
    promote_eval = sub.add_parser(
        "promote-eval",
        parents=[log_flags],
        help=(
            "Control vs treatment EvalSpec: dry-run plan or execute "
            "(preflight + compare + recommendation)"
        ),
    )
    promote_eval.add_argument(
        "--eval",
        required=True,
        help="EvalSpec name under configs/evals/<name>.yaml",
    )
    promote_eval.add_argument(
        "--dry-run",
        action="store_true",
        default=True,
        help="Print the suite plan without fitting (default)",
    )
    promote_eval.add_argument(
        "--execute",
        action="store_true",
        help="Run preflight and prediction compare; write recommendation.md",
    )
    promote_eval.add_argument(
        "--allow-dirty",
        action="store_true",
        help="Continue with a dirty tree; forces inconclusive",
    )
    promote_eval.add_argument(
        "--tier",
        choices=("S0", "S1", "S2", "S3", "S4"),
        help="Clamp to_gw / disable decisions for S0–S2; S0 is preflight only",
    )
    promote_eval.add_argument("--from-gw", type=int)
    promote_eval.add_argument("--to-gw", type=int)
    promote_eval.add_argument(
        "--seasons",
        nargs="+",
        help="Override window.seasons",
    )
    promote_eval.add_argument(
        "--skip-decisions",
        action="store_true",
        help="Force decisions suite off",
    )
    promote_eval.add_argument(
        "--output-dir",
        type=Path,
        help="Write plan/audit/recommendation here",
    )
    promote_eval.add_argument(
        "--json",
        action="store_true",
        help="Print plan JSON instead of the text table (dry-run)",
    )
    archive_benchmarks = sub.add_parser(
        "archive-benchmarks",
        parents=[log_flags],
        help=(
            "Archive ep_next + OpenFPL into benchmark_projections "
            "(pre-deadline or backfill; no FPL Review scrape)"
        ),
    )
    archive_benchmarks.add_argument("--season")
    archive_benchmarks.add_argument("--gw", type=int)
    archive_benchmarks.add_argument(
        "--skip-openfpl",
        action="store_true",
        help="Do not require OpenFPL rows (ep_next only; not for production)",
    )
    ingest_lineups = sub.add_parser(
        "ingest-lineups",
        parents=[log_flags],
        help=(
            "Stage predicted_lineups from raw/touchline (and raw/api_football/"
            "lineups). Optional live Touchline fetch when "
            "FPLT_TOUCHLINE_LINEUPS_ENABLED=1"
        ),
    )
    ingest_lineups.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch Touchline Wire now (requires FPLT_TOUCHLINE_LINEUPS_ENABLED)",
    )
    ingest_lineups.add_argument(
        "--season",
        help="Season label for a live fetch path (default: from clock)",
    )
    ingest_review = sub.add_parser(
        "ingest-review",
        parents=[log_flags],
        help=(
            "Merge a manual FPL Review CSV/JSON drop into "
            "benchmark_projections (source=fpl_review; no scrape)"
        ),
    )
    ingest_review.add_argument(
        "path",
        type=Path,
        help="Drop file (.csv or .json) with player_key/fixture_id/xpts",
    )
    ingest_review.add_argument(
        "--available-at",
        help="UTC ISO timestamp when the file omits available_at",
    )
    ingest_review.add_argument("--season", help="Override as_of season")
    ingest_review.add_argument("--gw", type=int, help="Override as_of gameweek")
    ingest_review.add_argument(
        "--copy-raw",
        action="store_true",
        help="Also copy the drop under data/raw/fpl_review/ (write-once)",
    )
    archive_compare = sub.add_parser(
        "archive-compare",
        parents=[log_flags],
        help=(
            "Paired prospective board: join ledger forecasts to "
            "benchmark_projections (RMSE headline; no FPL Review scrape)"
        ),
    )
    archive_compare.add_argument("--season", required=True)
    archive_compare.add_argument(
        "--model",
        required=True,
        help="ModelSpec name or ledger model_id for the candidate arm",
    )
    archive_compare.add_argument("--from-gw", type=int, default=1)
    archive_compare.add_argument("--to-gw", type=int, default=38)
    archive_compare.add_argument(
        "--sources",
        nargs="+",
        default=list(DEFAULT_SOURCES),
        help="Archive sources to pair (default: ep_next openfpl)",
    )
    archive_compare.add_argument("--horizon", type=int, default=1)
    archive_compare.add_argument(
        "--output-dir",
        type=Path,
        help="Write summary JSON/parquet/markdown here",
    )
    ops_catchup = sub.add_parser(
        "ops-catchup",
        parents=[log_flags],
        help=(
            "Post-downtime recovery: fetch raw → restage → "
            "understat/AF ingest → catch_up → status"
        ),
    )
    ops_catchup.add_argument(
        "--source",
        choices=("auto", "bucket", "branch"),
        default="auto",
        help=(
            "Raw fetch source: auto (bucket when FPLT_DATA_BUCKET is set, "
            "else branch), bucket, or branch (default: auto)"
        ),
    )
    ops_catchup.add_argument(
        "--remote",
        default="origin",
        help="Git remote (or local path) for the snapshots branch",
    )
    ops_catchup.add_argument(
        "--branch",
        default="snapshots",
        help="Branch that holds write-once captures (default: snapshots)",
    )
    ops_catchup.add_argument(
        "--horizon",
        type=int,
        default=DEFAULT_PLAN_HORIZON,
        help=f"Plan horizon (default: {DEFAULT_PLAN_HORIZON})",
    )
    ops_catchup.add_argument(
        "--no-notify",
        action="store_true",
        help="Skip ntfy/SMTP recommendation push even when plans were made",
    )
    status_cmd = sub.add_parser(
        "status",
        parents=[log_flags],
        help=(
            "Ops health document (captures, plans, deadline, token, "
            "quarantines, prod sha); write status.json under data_dir"
        ),
    )
    status_cmd.add_argument(
        "--json",
        action="store_true",
        help="Emit the status document as JSON (default: short text)",
    )
    status_cmd.add_argument(
        "--public",
        action="store_true",
        help="Emit / write the public whitelist (no token or last_error)",
    )
    status_cmd.add_argument(
        "--data-dir",
        type=Path,
        help="Override FPLT_DATA_DIR (default: settings.data_dir)",
    )
    status_cmd.add_argument(
        "--push",
        action="store_true",
        help="Push status/status.json to FPLT_DATA_BUCKET (mutable overwrite)",
    )
    ops_tick = sub.add_parser(
        "ops-tick",
        parents=[log_flags],
        help=(
            "Idempotent ops tick: fetch → restage → catch_up → "
            "deadline checkpoints → no-plan alarm → pre_deadline → status"
        ),
    )
    ops_tick.add_argument(
        "--source",
        choices=("auto", "bucket", "branch"),
        default="auto",
        help=(
            "Raw fetch source: auto (bucket when FPLT_DATA_BUCKET is set, "
            "else branch), bucket, or branch (default: auto)"
        ),
    )
    ops_tick.add_argument(
        "--remote",
        default="origin",
        help="Git remote (or local path) for the snapshots branch",
    )
    ops_tick.add_argument(
        "--branch",
        default="snapshots",
        help="Branch that holds write-once captures (default: snapshots)",
    )
    ops_tick.add_argument(
        "--horizon",
        type=int,
        default=None,
        help=(
            "Plan horizon (default: configs/ops/production.yaml / FPLT_PLAN_HORIZON)"
        ),
    )
    ops_tick.add_argument(
        "--no-notify",
        action="store_true",
        help="Skip ntfy/SMTP recommendation push even when plans were made",
    )
    ops_tick.add_argument(
        "--run-now",
        action="store_true",
        help=(
            "On-demand official-slate preview under checkpoint=manual-<UTC stamp> "
            "(no T-* markers, applied.json, or recommendation ntfy; #364)"
        ),
    )
    preflight = sub.add_parser(
        "preflight",
        parents=[log_flags],
        help=(
            "Season-rollover checks: scoring/chips, team maps, unmapped "
            "share, GW1 deadline, ModelSpecs, dry-run plan (ops-07)"
        ),
    )
    preflight.add_argument(
        "--season",
        required=True,
        help="Season label, e.g. 2027-28",
    )
    preflight.add_argument(
        "--data-dir",
        type=Path,
        help="Override FPLT_DATA_DIR (default: settings.data_dir)",
    )
    preflight.add_argument(
        "--skip-plan",
        action="store_true",
        help="Skip the dry-run plan check",
    )
    idmap = sub.add_parser(
        "idmap",
        parents=[log_flags],
        help="Suggest Understat / API-Football → player_key map entries",
    )
    idmap_sub = idmap.add_subparsers(dest="idmap_command")
    idmap_suggest = idmap_sub.add_parser(
        "suggest",
        parents=[log_flags],
        help=(
            "Fuzzy-match unmapped players; write high-confidence candidates "
            "for a review PR (nothing is auto-merged)"
        ),
    )
    idmap_suggest.add_argument(
        "--source",
        required=True,
        choices=("understat", "api_football"),
        help="Upstream id map to suggest against",
    )
    idmap_suggest.add_argument(
        "--season",
        required=True,
        help="Season label used to scan raw captures and staged FPL players",
    )
    idmap_suggest.add_argument(
        "--data-dir",
        type=Path,
        help="Override FPLT_DATA_DIR (default: settings.data_dir)",
    )
    idmap_suggest.add_argument(
        "--out",
        type=Path,
        help="Write scored candidates YAML (for a review PR)",
    )
    idmap_suggest.add_argument(
        "--min-score",
        type=float,
        default=None,
        help="Minimum similarity to suggest (default 0.75)",
    )
    idmap_suggest.add_argument(
        "--high-confidence",
        type=float,
        default=None,
        help="Score floor flagged for PR (default 0.92)",
    )
    deps_eq = sub.add_parser(
        "deps-equivalence",
        parents=[log_flags],
        help=(
            "Runtime lockfile PR → environment promote-eval (ops-12). "
            "Rewrites the fixed board from base vs head uv.lock and gates merge."
        ),
    )
    deps_eq.add_argument(
        "--control-ref",
        default=None,
        help=(
            "Control git ref (PR base SHA or main); required unless --check-freeze-only"
        ),
    )
    deps_eq.add_argument(
        "--treatment-ref",
        default=None,
        help=("Treatment git ref (PR head SHA); required unless --check-freeze-only"),
    )
    deps_eq.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory for promote-eval artifacts and comment.md",
    )
    deps_eq.add_argument(
        "--live",
        action="store_true",
        help="Execute promote-eval (otherwise prepare + soft-skip)",
    )
    deps_eq.add_argument(
        "--check-freeze-only",
        action="store_true",
        help="Print freeze status and exit 1 only when prod freeze is active",
    )
    deps_eq.add_argument(
        "--repo-root",
        type=Path,
        default=None,
        help="Repository root (default: cwd)",
    )
    governance = sub.add_parser(
        "governance-check",
        parents=[log_flags],
        help=(
            "CI gate: fail when production.yaml / listed ModelSpecs change "
            "without a promote-eval-run keep link (ops-09)"
        ),
    )
    governance.add_argument(
        "--base",
        required=True,
        help="Git base SHA/ref (PR base)",
    )
    governance.add_argument(
        "--head",
        default="HEAD",
        help="Git head SHA/ref (default: HEAD)",
    )
    governance.add_argument(
        "--pr-body",
        default="",
        help="Pull-request body text (look for promote-eval-run:)",
    )
    governance.add_argument(
        "--pr-body-file",
        type=Path,
        help="Read PR body from a file instead of --pr-body",
    )
    drift_check = sub.add_parser(
        "drift-check",
        parents=[log_flags],
        help=(
            "Compare rolling live RMSE/Brier to backtest distribution "
            "and alert on sustained drift (ops-09; alert only)"
        ),
    )
    drift_check.add_argument(
        "--model",
        action="append",
        dest="models",
        help="ModelSpec name (repeatable; default: production models)",
    )
    drift_check.add_argument(
        "--no-notify",
        action="store_true",
        help="Skip ntfy/SMTP even when alerts fire",
    )
    release = sub.add_parser(
        "release",
        parents=[log_flags],
        help="Promote or roll back the moving prod tag (ops-04)",
    )
    release_sub = release.add_subparsers(dest="release_command")
    release_check = release_sub.add_parser(
        "check",
        parents=[log_flags],
        help="Run CI / freeze / smoke gates without moving tags",
    )
    release_check.add_argument(
        "--ref",
        default="HEAD",
        help="Candidate commit / branch / tag (default: HEAD)",
    )
    release_check.add_argument(
        "--ci-status",
        choices=("success", "failure", "pending", "unknown"),
        default=None,
        help=(
            "Inject CI status (tests / GHA). When omitted, treat as success "
            "locally; release.yml passes the Actions conclusion."
        ),
    )
    release_check.add_argument(
        "--data-dir",
        type=Path,
        help="Data root for staged gameweeks/fixtures (default: settings)",
    )
    release_promote = release_sub.add_parser(
        "promote",
        parents=[log_flags],
        help=(
            "Run gates then move prod (and write prod-YYYYMMDD-<sha>). "
            "Rollback: --ref=<previous immutable tag>."
        ),
    )
    release_promote.add_argument(
        "--ref",
        default="HEAD",
        help="Candidate commit / branch / tag (default: HEAD)",
    )
    release_promote.add_argument(
        "--ci-status",
        choices=("success", "failure", "pending", "unknown"),
        default=None,
        help="Inject CI status (required to be success to promote)",
    )
    release_promote.add_argument(
        "--data-dir",
        type=Path,
        help="Data root for staged gameweeks/fixtures (default: settings)",
    )
    release_promote.add_argument(
        "--dry-run",
        action="store_true",
        help="Print git tag/push commands without running them",
    )
    release_sub.add_parser(
        "smoke",
        parents=[log_flags],
        help="Load production ModelSpec (smoke plan gate only)",
    )
    perf = sub.add_parser(
        "perf-report",
        parents=[log_flags],
        help=(
            "Rank wall-clock timing hotspots from timing/*.json, "
            "runs/*/logs/timing.json, and official/*/ops_timing.json"
        ),
    )
    perf.add_argument(
        "--data-dir",
        type=Path,
        help="Override FPLT_DATA_DIR (default: settings.data_dir)",
    )
    perf.add_argument(
        "--kind",
        choices=(
            "walk_forward",
            "plan",
            "decisions",
            "eval_board",
            "ingest",
            "remote",
            "curate",
            "ops_tick",
            "ops_catchup",
        ),
        help="Only include this timing kind",
    )
    perf.add_argument(
        "--limit",
        type=int,
        default=10,
        help="Max timing records to load (newest first; default 10)",
    )
    perf.add_argument(
        "--hotspots",
        type=int,
        default=20,
        help="Max hotspot rows to print (default 20)",
    )
    perf.add_argument(
        "--json",
        action="store_true",
        help="Emit ranked hotspots as JSON instead of text",
    )
    perf.add_argument(
        "--trend",
        action="store_true",
        help=(
            "Summarise nightly bench history under lab_dir/bench/ "
            "(rolling median baseline; see configs/ops/bench.yaml)"
        ),
    )
    perf.add_argument(
        "--lab-dir",
        type=Path,
        help="Lab root for --trend (default: FPLT_LAB_DIR or ./lab)",
    )
    data = sub.add_parser(
        "data",
        parents=[log_flags],
        help="R2 / snapshots raw helpers (parity during dual-write)",
    )
    data_sub = data.add_subparsers(dest="data_command")
    data_parity = data_sub.add_parser(
        "parity",
        parents=[log_flags],
        help=(
            "One-way check: snapshots branch tip → FPLT_DATA_BUCKET/raw "
            "(rclone check; exits 1 on missing/different keys)"
        ),
    )
    data_parity.add_argument(
        "--remote",
        default="origin",
        help="Git remote (or local path) for the snapshots branch",
    )
    data_parity.add_argument(
        "--branch",
        default="snapshots",
        help="Branch that holds write-once captures (default: snapshots)",
    )
    data_pull = data_sub.add_parser(
        "pull",
        parents=[log_flags],
        help="Pull from FPLT_DATA_BUCKET (--official latest | --gw N)",
    )
    data_pull.add_argument(
        "--official",
        action="store_true",
        help="Pull official pointers, runs, and curated UI tables",
    )
    data_pull.add_argument(
        "selector",
        nargs="?",
        choices=["latest"],
        help="Highest GW with a primary pointer at --checkpoint",
    )
    data_pull.add_argument("--gw", type=int, help="Gameweek (xor latest)")
    data_pull.add_argument("--season", help="Season key (default: max under official/)")
    data_pull.add_argument(
        "--checkpoint",
        default="T-90m",
        help=(
            "Concrete id (default T-90m) or alias: manual (newest manual-*), "
            "latest (newest pointer of any checkpoint)"
        ),
    )
    data_pull.add_argument("--line", help="Line filter (default: all at checkpoint)")
    data_pull.add_argument("--data-dir", type=Path, help="Override FPLT_DATA_DIR")
    data_curate = data_sub.add_parser(
        "curate",
        parents=[log_flags],
        help=(
            "Incremental curated build + status/curated/latest.json "
            "(#345; use --full for weekly parity rebuild)"
        ),
    )
    data_curate.add_argument(
        "--full",
        action="store_true",
        help=(
            "Pull all of raw/, stage from scratch, and fail (ntfy) when table "
            "shas diverge from the latest pointer"
        ),
    )
    data_curate.add_argument(
        "--source",
        choices=("auto", "bucket", "branch"),
        default="auto",
        help=(
            "Raw fetch source for full/bootstrap: auto (bucket when "
            "FPLT_DATA_BUCKET is set, else branch), bucket, or branch "
            "(default: auto)"
        ),
    )
    data_curate.add_argument(
        "--remote",
        default="origin",
        help="Git remote (or local path) for the snapshots branch",
    )
    data_curate.add_argument(
        "--branch",
        default="snapshots",
        help="Branch that holds write-once captures (default: snapshots)",
    )
    data_curate.add_argument(
        "--data-dir",
        type=Path,
        help="Override FPLT_DATA_DIR (default: settings.data_dir)",
    )
    data_curate.add_argument(
        "--no-notify",
        action="store_true",
        help="Skip ntfy on full-rebuild parity mismatch",
    )
    data_gates = data_sub.add_parser(
        "gates",
        parents=[log_flags],
        help=(
            "Capture sanity gates (ops-06): explain quarantine with --since, "
            "or --check new bootstrap captures"
        ),
    )
    data_gates.add_argument(
        "--since",
        help=(
            "Show quarantine rows with available_at on/after this bound "
            "(ISO datetime/date, or relative Nd/Nh/Nm such as 7d)"
        ),
    )
    data_gates.add_argument(
        "--check",
        action="store_true",
        help=(
            "Run gates on new bootstrap captures, quarantine failures, "
            "and alert once per new failure type"
        ),
    )
    data_gates.add_argument(
        "--new-raw",
        type=Path,
        help="Directory of newly written raw captures to gate (GHA temp out)",
    )
    data_gates.add_argument(
        "--data-dir",
        type=Path,
        help="Override FPLT_DATA_DIR (default: settings.data_dir)",
    )
    data_gates.add_argument(
        "--notify",
        action="store_true",
        help="POST ntfy for newly seen failure types (uses FPLT_NTFY_URL)",
    )
    data_gates.add_argument(
        "--no-sync",
        action="store_true",
        help="Skip R2 status/quarantine pull/push around --check",
    )
    exp = sub.add_parser(
        "exp",
        parents=[log_flags],
        help="Experiment / batch submit (Stage C / data-model step 7)",
    )
    exp_sub = exp.add_subparsers(dest="exp_command")
    exp_submit = exp_sub.add_parser(
        "submit",
        parents=[log_flags],
        help=(
            "Submit EvalSpec YAML(s) as ab experiments; with --batch mint a "
            "batch and dedupe shared arms across the batch"
        ),
    )
    exp_submit.add_argument(
        "evals",
        nargs="+",
        help="EvalSpec stems or paths (configs/evals/<name>.yaml)",
    )
    exp_submit.add_argument(
        "--batch",
        required=True,
        metavar="LABEL",
        help="Batch label (mints batch_id; required for this CLI surface)",
    )
    exp_submit.add_argument(
        "--evals-dir",
        type=Path,
        help="Override configs/evals when stems are passed",
    )
    exp_submit.add_argument(
        "--data-dir",
        type=Path,
        help="Override FPLT_DATA_DIR (default: settings.data_dir)",
    )
    exp_submit.add_argument(
        "--build-id",
        help="Pin curated build_id for all arms (default: latest, else none)",
    )
    exp_ls = exp_sub.add_parser(
        "ls",
        parents=[log_flags],
        help="List experiment ids under experiments/",
    )
    exp_ls.add_argument(
        "--data-dir",
        type=Path,
        help="Override FPLT_DATA_DIR (default: settings.data_dir)",
    )
    exp_show = exp_sub.add_parser(
        "show",
        parents=[log_flags],
        help="Show one experiment manifest summary and folded status",
    )
    exp_show.add_argument("experiment_id", help="Experiment id (exp_…)")
    exp_show.add_argument(
        "--data-dir",
        type=Path,
        help="Override FPLT_DATA_DIR (default: settings.data_dir)",
    )
    exp_component = exp_sub.add_parser(
        "component",
        parents=[log_flags],
        help=(
            "Component A/B with cached upstream fits (toolkit §2.4): "
            "score component then player-fixture stages via Measure"
        ),
    )
    exp_component.add_argument(
        "--base",
        required=True,
        help="Base pipeline: production or a registered ModelSpec name",
    )
    exp_component.add_argument(
        "--component",
        required=True,
        choices=("minutes", "score", "involvement", "other"),
        help="Layered component slot under test",
    )
    exp_component.add_argument(
        "--arm",
        action="append",
        required=True,
        dest="arms",
        metavar="PATCH|SPEC",
        help=(
            "Treatment arm (repeatable): JSON patch, path=value, "
            "ModelSpec name for a component swap, or a YAML/JSON file"
        ),
    )
    exp_component.add_argument(
        "--window",
        required=True,
        help="Eval window: season or season:gw-gw (e.g. 2024-25:1-10)",
    )
    exp_component.add_argument(
        "--data-dir",
        type=Path,
        help="Override FPLT_DATA_DIR (default: settings.data_dir)",
    )
    exp_component.add_argument(
        "--build-id",
        default="local",
        help="Curated build id for component cache keys (default: local)",
    )
    exp_component.add_argument(
        "--n-sims",
        type=int,
        help="Override ModelSpec n_sims for all arms (screening / tests)",
    )
    catalog = sub.add_parser(
        "catalog",
        parents=[log_flags],
        help="Derived catalog/ index over run manifests (Stage C / step 8)",
    )
    catalog_sub = catalog.add_subparsers(dest="catalog_command")
    catalog_rebuild = catalog_sub.add_parser(
        "rebuild",
        parents=[log_flags],
        help=(
            "Rescan runs/run=*/ with _SUCCESS into catalog/runs.parquet "
            "(clears catalog/inbox/)"
        ),
    )
    catalog_rebuild.add_argument(
        "--data-dir",
        type=Path,
        help="Override FPLT_DATA_DIR (default: settings.data_dir)",
    )
    sql = sub.add_parser(
        "sql",
        parents=[log_flags],
        help=(
            "DuckDB over catalog + run tables (views: runs, official, "
            "player_fixture, minutes, plans, outcomes, ledger)"
        ),
    )
    sql.add_argument(
        "-c",
        "--command",
        dest="sql_command",
        help="Run one SQL statement and print the result (non-interactive)",
    )
    sql.add_argument(
        "--data-dir",
        type=Path,
        help="Override FPLT_DATA_DIR (default: settings.data_dir)",
    )
    sql.add_argument(
        "--rebuild-catalog",
        action="store_true",
        help="Rescan runs into catalog/runs.parquet before opening DuckDB",
    )
    legacy = sub.add_parser(
        "legacy",
        parents=[log_flags],
        help="Wrap pre-run local outputs as kind=legacy runs (B10)",
    )
    legacy_sub = legacy.add_subparsers(dest="legacy_command")
    legacy_import = legacy_sub.add_parser(
        "import",
        parents=[log_flags],
        help=(
            "Import predictions/, plans/, ledger/, artifacts/, and eval "
            "dirs as kind=legacy runs (run_id=legacy-<model_id>-<as_of>)"
        ),
    )
    legacy_import.add_argument(
        "--data-dir",
        type=Path,
        help="Override FPLT_DATA_DIR (default: settings.data_dir)",
    )
    legacy_import.add_argument(
        "--dry-run",
        action="store_true",
        help="List run ids that would be imported; write nothing",
    )
    legacy_import.add_argument(
        "--push",
        action="store_true",
        help="Push each new run to fplt-lab when the lab bucket is configured",
    )
    ui = sub.add_parser(
        "ui",
        parents=[log_flags],
        help=(
            "Local Streamlit dashboard on localhost (reads $FPLT_DATA_DIR; no writes)"
        ),
    )
    ui.add_argument(
        "--port",
        type=int,
        default=8501,
        help="Streamlit port (default: 8501)",
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
    auth_check = auth_sub.add_parser(
        "check",
        parents=[log_flags],
        help=(
            "Monitor FPL API token expiry (ops-01); exit 1 when missing, "
            "malformed, expired, or inside the warn / deadline window"
        ),
    )
    auth_check.add_argument(
        "--warn-days",
        type=float,
        default=3.0,
        help="Warn when exp falls within this many days (default: 3)",
    )
    auth_check.add_argument(
        "--deadline-aware",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Urgent when exp is before the next staged deadline "
            "(default: on; needs staged gameweeks)"
        ),
    )
    auth_check.add_argument(
        "--allow-missing",
        action="store_true",
        help=(
            "Soft-skip (exit 0) when FPLT_FPL_API_TOKEN is unset so "
            "CI/crons stay green before secrets are configured"
        ),
    )
    auth_check.add_argument(
        "--no-notify",
        action="store_true",
        help="Skip ntfy/SMTP even when the check is warning/urgent",
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
    try:
        configure_logging(
            verbose=bool(args.verbose),
            quiet=bool(args.quiet),
            log_format=getattr(args, "log_format", None),
        )
    except (SettingsConfigError, ValidationError) as exc:
        # Logging may not be up yet; still fail cleanly for empty/invalid env.
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from None
    try:
        _dispatch(args)
    except SettingsConfigError as exc:
        _exit_config_error(exc)


def _dispatch(args: argparse.Namespace) -> None:
    """Route a parsed CLI command after logging is configured."""
    if args.command == "auth":
        logger.info("fplt auth")
        _auth(args)
        return
    if args.command == "ui":
        logger.info("fplt ui (data_dir=%s)", get_settings().data_dir)
        raise SystemExit(_ui(args))
    if args.command == "data":
        logger.info("fplt data")
        _data(args)
        return
    if args.command == "catalog":
        logger.info("fplt catalog")
        _catalog(args)
        return
    if args.command == "exp":
        logger.info("fplt exp")
        _exp(args)
        return
    if args.command == "sql":
        logger.info("fplt sql")
        _sql(args)
        return
    if args.command == "legacy":
        logger.info("fplt legacy")
        _legacy(args)
        return
    logger.info("fplt %s (data_dir=%s)", args.command, get_settings().data_dir)
    if args.command == "ops-catchup":
        _ops_catchup(args)
        return
    if args.command == "status":
        _status(args)
        return
    if args.command == "ops-tick":
        _ops_tick(args)
        return
    if args.command == "preflight":
        _preflight(args)
        return
    if args.command == "idmap":
        _idmap(args)
        return
    if args.command == "deps-equivalence":
        _deps_equivalence(args)
        return
    if args.command == "governance-check":
        _governance_check(args)
        return
    if args.command == "drift-check":
        _drift_check(args)
        return
    if args.command == "release":
        _release(args)
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
    if args.command == "explain":
        _explain(args)
        return
    if args.command == "apply":
        _apply(args)
        return
    if args.command == "decisions":
        _decisions(args)
        return
    if args.command == "regret-board":
        _regret_board(args)
        return
    if args.command == "weekly-report":
        _weekly_report(args)
        return
    if args.command == "sim-season":
        _sim_season(args)
        return
    if args.command == "eval-components":
        _eval_components(args)
        return
    if args.command == "minutes-measure":
        _minutes_measure(args)
        return
    if args.command == "other-measure":
        _other_measure(args)
        return
    if args.command == "predicted-xi-audit":
        _predicted_xi_audit(args)
        return
    if args.command == "horizon-skill":
        _horizon_skill(args)
        return
    if args.command == "pruning-loss":
        _pruning_loss(args)
        return
    if args.command == "ensemble-sensitivity":
        _ensemble_sensitivity(args)
        return
    if args.command == "promote-eval":
        _promote_eval(args)
        return
    if args.command == "bakeoff":
        _bakeoff(args)
        return
    if args.command == "archive-benchmarks":
        _archive_benchmarks(args)
        return
    if args.command == "ingest-lineups":
        _ingest_lineups(args)
        return
    if args.command == "ingest-review":
        _ingest_review(args)
        return
    if args.command == "archive-compare":
        _archive_compare(args)
        return
    if args.command == "perf-report":
        _perf_report(args)
        return
    raise SystemExit(_not_implemented(args.command))


def _ui(args: argparse.Namespace) -> int:
    """Launch the local Streamlit dashboard bound to localhost."""
    from importlib.util import find_spec

    if find_spec("streamlit") is None:
        print(
            "streamlit is not installed; run `uv sync` "
            "(dev dependency) or `uv add --dev streamlit`",
            file=sys.stderr,
        )
        return 2
    app = Path(__file__).resolve().parent / "ui" / "app.py"
    port = int(getattr(args, "port", 8501))
    argv = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(app),
        "--server.address",
        "localhost",
        "--server.headless",
        "true",
        "--server.port",
        str(port),
        "--browser.gatherUsageStats",
        "false",
    ]
    print(
        f"Starting fplt UI on http://localhost:{port}/ "
        f"(data_dir={get_settings().data_dir})",
        flush=True,
    )
    return int(run_command(argv).returncode)


def _preflight(args: argparse.Namespace) -> None:
    from fplt.data.store import DataStore
    from fplt.ops.preflight import format_preflight_report, run_preflight

    data_dir = (
        Path(args.data_dir) if args.data_dir is not None else get_settings().data_dir
    )
    report = run_preflight(
        str(args.season),
        store=DataStore(data_dir),
        skip_plan=bool(args.skip_plan),
    )
    print(format_preflight_report(report))
    if not report.ok:
        raise SystemExit(1)


def _idmap(args: argparse.Namespace) -> None:
    command = getattr(args, "idmap_command", None)
    if command == "suggest":
        _idmap_suggest(args)
        return
    raise SystemExit(_not_implemented(f"idmap {command}"))


def _idmap_suggest(args: argparse.Namespace) -> None:
    from typing import cast

    from fplt.data.api_football import (
        default_id_maps_path as api_football_id_maps_path,
    )
    from fplt.data.api_football import (
        load_id_maps as load_api_football_id_maps,
    )
    from fplt.data.store import DataStore
    from fplt.data.understat import (
        default_id_maps_path as understat_id_maps_path,
    )
    from fplt.data.understat import (
        load_id_maps as load_understat_id_maps,
    )
    from fplt.ops.idmap import (
        DEFAULT_HIGH_CONFIDENCE,
        DEFAULT_MIN_SCORE,
        SourceName,
        collect_api_football_unmapped,
        collect_understat_unmapped,
        format_candidates_report,
        suggest_matches,
        write_candidates_yaml,
    )
    from fplt.ops.preflight import build_fpl_targets

    data_dir = (
        Path(args.data_dir) if args.data_dir is not None else get_settings().data_dir
    )
    raw_root = data_dir / "raw"
    store = DataStore(data_dir)
    season = str(args.season)
    source = cast(SourceName, str(args.source))
    min_score = (
        float(args.min_score) if args.min_score is not None else DEFAULT_MIN_SCORE
    )
    high_confidence = (
        float(args.high_confidence)
        if args.high_confidence is not None
        else DEFAULT_HIGH_CONFIDENCE
    )

    if source == "understat":
        player_keys, team_keys = load_understat_id_maps(understat_id_maps_path())
        unmapped = collect_understat_unmapped(
            raw_root,
            season,
            player_keys=player_keys,
            team_keys=team_keys,
        )
    else:
        player_keys, team_keys = load_api_football_id_maps(api_football_id_maps_path())
        unmapped = collect_api_football_unmapped(
            raw_root,
            season,
            player_keys=player_keys,
            team_keys=team_keys,
        )

    if not store.path("players").is_file():
        print(
            f"idmap suggest: no staged players under {data_dir}",
            file=sys.stderr,
        )
        raise SystemExit(1)
    targets = build_fpl_targets(store.read("players"), season=season)
    candidates = suggest_matches(
        unmapped,
        targets,
        min_score=min_score,
        high_confidence=high_confidence,
    )
    print(format_candidates_report(candidates, source=source))
    if args.out is not None:
        write_candidates_yaml(Path(args.out), candidates, source=source)
        print(f"wrote {args.out}")


def _data(args: argparse.Namespace) -> None:
    command = getattr(args, "data_command", None)
    if command == "parity":
        _data_parity(args)
        return
    if command == "pull":
        _data_pull(args)
        return
    if command == "curate":
        _data_curate(args)
        return
    if command == "gates":
        _data_gates(args)
        return
    raise SystemExit(_not_implemented(f"data {command}"))


def _data_curate(args: argparse.Namespace) -> None:
    from fplt.automation.curate import CurateParityError, run_curate
    from fplt.data.remote import RemoteError
    from fplt.data.snapshots import SnapshotFetchError
    from fplt.store.pull_official import OfficialPullError

    data_dir = (
        Path(args.data_dir) if args.data_dir is not None else get_settings().data_dir
    )
    try:
        result = run_curate(
            data_dir,
            full=bool(args.full),
            source=args.source,
            remote=args.remote,
            branch=args.branch,
            notify_parity=not bool(args.no_notify),
        )
    except (
        SnapshotFetchError,
        RemoteError,
        OfficialPullError,
        CurateParityError,
        ValueError,
    ) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    if result.skipped_no_new:
        print("curate: no new raw; pointer unchanged")
        return
    pointer = result.pointer
    build_id = result.manifest.build_id if result.manifest else "?"
    updated = "yes" if result.pointer_updated else "no"
    mode = "full" if result.full else "incremental"
    print(
        f"curate: mode={mode} build_id={build_id} pointer_updated={updated} "
        f"new_captures={len(result.new_capture_dirs)}"
        + (
            f" raw_cutoff={pointer.raw_cutoff.isoformat()}"
            if pointer is not None
            else ""
        )
        + (f" parity_ok={result.parity_ok}" if result.parity_ok is not None else "")
    )


def _data_pull(args: argparse.Namespace) -> None:
    from fplt.data.remote import RemoteError
    from fplt.store.pull_official import (
        OfficialPullError,
        infer_official_season,
        mirror_official_index,
        pull_official,
        require_data_remote,
        resolve_checkpoint_target,
    )

    if not args.official:
        print("fplt data pull: pass --official", file=sys.stderr)
        raise SystemExit(2)
    checkpoint_arg = str(args.checkpoint)
    alias = checkpoint_arg in ("manual", "latest")
    if alias:
        if args.selector == "latest" and args.gw is not None:
            print(
                "fplt data pull --official: pass at most one of 'latest' or --gw",
                file=sys.stderr,
            )
            raise SystemExit(2)
    elif (args.selector == "latest") == (args.gw is not None):
        print(
            "fplt data pull --official: pass exactly one of 'latest' or --gw "
            "(or use --checkpoint manual|latest)",
            file=sys.stderr,
        )
        raise SystemExit(2)
    data_dir = (
        Path(args.data_dir) if args.data_dir is not None else get_settings().data_dir
    )
    try:
        remote = require_data_remote()
        official_root = mirror_official_index(data_dir, remote)
        season = infer_official_season(official_root, args.season)
        pinned_gw = None if args.gw is None else int(args.gw)
        if alias or args.selector == "latest" or pinned_gw is None:
            gw, checkpoint = resolve_checkpoint_target(
                official_root,
                season=season,
                checkpoint=checkpoint_arg,
                gw=pinned_gw,
            )
        else:
            gw, checkpoint = pinned_gw, checkpoint_arg
        result = pull_official(
            data_dir,
            remote,
            season=season,
            gw=gw,
            checkpoint=checkpoint,
            line=args.line,
        )
    except (RemoteError, OfficialPullError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(
        f"official pull: season={result.season} gw={result.gw} "
        f"checkpoint={result.checkpoint} predict={list(result.predict_run_ids)} "
        f"plan={list(result.plan_run_ids)} builds={list(result.build_ids)} "
        f"new_files={result.n_written}"
    )


def _data_gates(args: argparse.Namespace) -> None:
    from fplt.data.capture_gates import (
        check_new_captures,
        format_gates_status,
        parse_since,
    )
    from fplt.data.remote import RemoteError

    data_dir = (
        Path(args.data_dir) if args.data_dir is not None else get_settings().data_dir
    )
    if args.since:
        try:
            bound = parse_since(str(args.since))
        except ValueError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(2) from exc
        print(format_gates_status(data_dir, since=bound), end="")
        if not args.check:
            return
    if not args.check:
        print(
            "fplt data gates: pass --since to explain quarantine, "
            "or --check to gate new captures",
            file=sys.stderr,
        )
        raise SystemExit(2)
    try:
        result = check_new_captures(
            data_dir,
            new_raw=Path(args.new_raw) if args.new_raw is not None else None,
            notify_alerts=bool(args.notify),
            sync_remote=not bool(args.no_sync),
        )
    except RemoteError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    failed = result.failed
    print(
        f"capture gates: checked={len(result.actions)} "
        f"failed={len(failed)} alerted={list(result.alerted)}"
    )
    for action in failed:
        names = ", ".join(f.name for f in action.report.failures)
        print(f"  quarantined {action.capture_path}: {names}")


def _catalog(args: argparse.Namespace) -> None:
    command = getattr(args, "catalog_command", None)
    if command == "rebuild":
        _catalog_rebuild(args)
        return
    raise SystemExit(_not_implemented(f"catalog {command}"))


def _catalog_rebuild(args: argparse.Namespace) -> None:
    from fplt.store.catalog import catalog_runs_path, rebuild_runs_catalog

    data_dir = (
        Path(args.data_dir) if args.data_dir is not None else get_settings().data_dir
    )
    frame = rebuild_runs_catalog(data_dir)
    print(f"catalog rebuild: rows={frame.height} path={catalog_runs_path(data_dir)}")


def _sql(args: argparse.Namespace) -> None:
    from fplt.store.sql import open_sql, run_sql, sql_repl

    data_dir = (
        Path(args.data_dir) if args.data_dir is not None else get_settings().data_dir
    )
    rebuild = bool(args.rebuild_catalog)
    query = getattr(args, "sql_command", None)
    if query:
        frame = run_sql(str(query), data_dir, rebuild_catalog=rebuild)
        print(frame)
        return
    con = open_sql(data_dir, rebuild_catalog=rebuild)
    try:
        sql_repl(con, write_line=print)
    finally:
        con.close()


def _legacy(args: argparse.Namespace) -> None:
    command = getattr(args, "legacy_command", None)
    if command == "import":
        _legacy_import(args)
        return
    raise SystemExit(_not_implemented(f"legacy {command}"))


def _legacy_import(args: argparse.Namespace) -> None:
    from fplt.store.legacy import import_legacy

    data_dir = (
        Path(args.data_dir) if args.data_dir is not None else get_settings().data_dir
    )
    result = import_legacy(
        data_dir,
        dry_run=bool(args.dry_run),
        push=bool(args.push),
    )
    prefix = "would import" if args.dry_run else "imported"
    print(
        f"legacy import: {prefix}={result.n_imported} "
        f"skipped={result.n_skipped} data_dir={data_dir}"
    )
    for run_id in result.imported:
        print(f"  + {run_id}")
    for run_id in result.skipped:
        print(f"  = {run_id} (exists)")


def _data_parity(args: argparse.Namespace) -> None:
    from fplt.data.parity import ParityError, check_raw_parity

    try:
        result = check_raw_parity(
            remote=str(args.remote),
            branch=str(args.branch),
        )
    except ParityError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc
    print(
        f"raw parity ok: {result.source_files} files on {result.branch} "
        f"→ {result.remote_target}/raw"
    )


def _status(args: argparse.Namespace) -> None:
    import json

    from fplt.automation.status import (
        build_status,
        format_status_human,
        public_status,
        push_status,
        write_status,
    )

    root = args.data_dir if args.data_dir is not None else get_settings().data_dir
    doc = build_status(root)
    write_status(root, doc)
    if args.push:
        push_status(root)
    if args.json:
        payload = public_status(doc) if args.public else doc.model_dump_json_doc()
        print(json.dumps(payload, indent=2, sort_keys=True))
        return
    print(format_status_human(doc), end="")


def _ops_catchup(args: argparse.Namespace) -> None:
    from fplt.automation.ops_catchup import (
        format_ops_catchup_status,
        run_ops_catchup,
    )
    from fplt.data.remote import RemoteError
    from fplt.data.snapshots import RawSource, SnapshotFetchError

    source: RawSource = args.source
    try:
        result = run_ops_catchup(
            source=source,
            remote=str(args.remote),
            branch=str(args.branch),
            horizon=int(args.horizon),
            notify=not bool(args.no_notify),
        )
    except (SnapshotFetchError, RemoteError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(format_ops_catchup_status(result), end="")
    print(format_status(auth_status()), end="")


def _ops_tick(args: argparse.Namespace) -> None:
    from fplt.automation.ops_tick import format_ops_tick_status, run_ops_tick
    from fplt.data.remote import RemoteError
    from fplt.data.snapshots import RawSource, SnapshotFetchError

    source: RawSource = args.source
    horizon = None if args.horizon is None else int(args.horizon)
    try:
        result = run_ops_tick(
            source=source,
            remote=str(args.remote),
            branch=str(args.branch),
            horizon=horizon,
            notify=not bool(args.no_notify),
            run_now=bool(args.run_now),
        )
    except (SnapshotFetchError, RemoteError, ValueError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(format_ops_tick_status(result), end="")
    print(format_status(auth_status()), end="")


def _deps_equivalence(args: argparse.Namespace) -> None:
    """Prepare / run runtime-deps equivalence and write comment.md."""
    from fplt.automation.deps_equivalence import (
        check_prod_freeze,
        format_pr_comment,
        materialise_lock_from_git,
        run_deps_equivalence,
    )

    if bool(args.check_freeze_only):
        freeze = check_prod_freeze()
        print(f"{freeze.state}: {freeze.reason}")
        if freeze.state == "frozen":
            raise SystemExit(1)
        return

    if not args.control_ref or not args.treatment_ref or args.output_dir is None:
        print(
            "deps-equivalence requires --control-ref, --treatment-ref, "
            "and --output-dir (or pass --check-freeze-only)",
            file=sys.stderr,
        )
        raise SystemExit(2)

    repo_root = Path(args.repo_root) if args.repo_root else Path.cwd()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    locks_dir = output_dir / "locks"
    control_lock = materialise_lock_from_git(
        repo_root, str(args.control_ref), locks_dir / "control.uv.lock"
    )
    treatment_lock = materialise_lock_from_git(
        repo_root, str(args.treatment_ref), locks_dir / "treatment.uv.lock"
    )
    gate = run_deps_equivalence(
        repo_root=repo_root,
        control_ref=str(args.control_ref),
        treatment_ref=str(args.treatment_ref),
        data_dir=get_settings().data_dir,
        output_dir=output_dir,
        control_lock=control_lock,
        treatment_lock=treatment_lock,
        live=bool(args.live),
        execute=True,
    )
    comment = format_pr_comment(
        gate,
        packages=gate.packages,
        control_ref=gate.control_ref or str(args.control_ref),
        treatment_ref=gate.treatment_ref or str(args.treatment_ref),
    )
    comment_path = output_dir / "comment.md"
    comment_path.write_text(comment, encoding="utf-8")
    print(comment, end="")
    print(f"wrote {comment_path}", file=sys.stderr)
    if gate.gate == "fail":
        raise SystemExit(1)
    if gate.gate == "skip":
        raise SystemExit(0)


def _governance_check(args: argparse.Namespace) -> None:
    from fplt.automation.governance import check_git_production_governance

    body = str(args.pr_body or "")
    if args.pr_body_file is not None:
        body = Path(args.pr_body_file).read_text(encoding="utf-8")
    repo = Path(__file__).resolve().parents[2]
    result = check_git_production_governance(
        repo=repo,
        base_ref=str(args.base),
        head_ref=str(args.head),
        pr_body=body,
        data_dir=get_settings().data_dir,
    )
    if result.passed:
        print("governance-check: ok")
        return
    for reason in result.reasons:
        print(f"governance-check: {reason}", file=sys.stderr)
    raise SystemExit(1)


def _drift_check(args: argparse.Namespace) -> None:
    from fplt.automation.drift import format_drift_alerts, run_live_drift_monitor

    reports = run_live_drift_monitor(
        get_settings().data_dir,
        model_names=args.models,
        notify=not bool(args.no_notify),
    )
    alerted = False
    for report in reports:
        print(format_drift_alerts(report))
        alerted = alerted or report.alerted
    if not reports:
        print("drift-check: no production models resolved")
    raise SystemExit(1 if alerted else 0)


def _release(args: argparse.Namespace) -> None:
    from datetime import UTC, datetime

    import polars as pl

    from fplt.automation.release import (
        ReleaseGateError,
        evaluate_release_gates,
        promote_prod_tags,
        run_smoke_plan,
    )
    from fplt.data.store import DataStore

    command = getattr(args, "release_command", None)
    if command == "smoke":
        try:
            result = run_smoke_plan()
        except ReleaseGateError as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1) from exc
        print(result.detail)
        return
    if command not in ("check", "promote"):
        print("usage: fplt release {check,promote,smoke}", file=sys.stderr)
        raise SystemExit(2)

    data_dir = Path(args.data_dir) if args.data_dir else get_settings().data_dir
    gameweeks = pl.DataFrame()
    fixtures = pl.DataFrame()
    store = DataStore(data_dir)
    if store.path("gameweeks").exists():
        try:
            gameweeks = store.read("gameweeks")
        except Exception as exc:  # noqa: BLE001 — treat unreadable as empty
            logger.warning("release: could not read staged gameweeks: %s", exc)
    if store.path("fixtures").exists():
        try:
            fixtures = store.read("fixtures")
        except Exception as exc:  # noqa: BLE001 — treat unreadable as empty
            logger.warning("release: could not read staged fixtures: %s", exc)

    status = args.ci_status or "success"

    def _status(_sha: str) -> str:
        return str(status)

    try:
        checked = evaluate_release_gates(
            str(args.ref),
            now=datetime.now(UTC),
            gameweeks=gameweeks,
            fixtures=fixtures,
            status_fn=_status,  # type: ignore[arg-type]
        )
    except ReleaseGateError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1) from exc

    print(
        f"release gates ok: ref={checked.candidate.ref} "
        f"sha={checked.candidate.sha} "
        f"history_tag={checked.history_tag} "
        f"smoke={checked.smoke.model}"
    )
    if command == "check":
        return
    cmds = promote_prod_tags(
        checked.candidate,
        history_tag=checked.history_tag,
        dry_run=bool(args.dry_run),
    )
    for line in cmds:
        print(line)


def _ingest_lineups(args: argparse.Namespace) -> None:
    from datetime import UTC, datetime

    from fplt.data.overall_rank import season_label
    from fplt.data.predicted_lineups import (
        PredictedLineupsDisabled,
        PredictedLineupsRateLimit,
        capture_touchline_lineups,
        ingest_predicted_lineups,
    )
    from fplt.data.store import DataStore
    from fplt.settings import get_settings

    settings = get_settings()
    raw_root = settings.data_dir / "raw"
    store = DataStore(settings.data_dir)
    if args.fetch:
        season = args.season or season_label(datetime.now(tz=UTC))
        try:
            path, written = capture_touchline_lineups(
                raw_root,
                season,
                enabled=settings.touchline_lineups_enabled,
                min_interval_seconds=settings.touchline_lineups_min_interval_seconds,
            )
        except (PredictedLineupsDisabled, PredictedLineupsRateLimit) as exc:
            logger.error("%s", exc)
            raise SystemExit(2) from exc
        logger.info(
            "touchline raw %s (%s)",
            path,
            "wrote" if written else "exists",
        )
    count = ingest_predicted_lineups(store, raw_root)
    logger.info("staged predicted_lineups from %s raw file(s)", count)


def _ingest_review(args: argparse.Namespace) -> None:
    """Merge a manual FPL Review drop into benchmark_projections."""
    from datetime import datetime

    from fplt.data.fpl_review import (
        copy_drop_to_raw,
        ingest_fpl_review_drop,
        load_fpl_review_drop,
    )

    path = args.path.expanduser()
    as_of: GWKey | None = None
    if args.season is not None or args.gw is not None:
        if args.season is None or args.gw is None:
            print("--season and --gw must be set together", file=sys.stderr)
            raise SystemExit(2)
        as_of = GWKey(args.season, args.gw)
    try:
        frame = load_fpl_review_drop(path, available_at=args.available_at, as_of=as_of)
        key = GWKey(str(frame["as_of_season"][0]), int(frame["as_of_gw"][0]))
        if args.copy_raw:
            stamp = frame["available_at"][0]
            if not isinstance(stamp, datetime):
                raise ValueError("drop is missing available_at")
            dest = copy_drop_to_raw(
                get_settings().data_dir,
                path,
                as_of=key,
                captured_at=stamp,
            )
            print(f"raw: {dest}")
        out, n = ingest_fpl_review_drop(
            get_settings().data_dir,
            path,
            available_at=args.available_at,
            as_of=as_of,
        )
    except (FileExistsError, FileNotFoundError, ValueError) as exc:
        logger.error("ingest-review failed: %s", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(f"staged {n} fpl_review rows into {out}")


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
    specs = list(args.specs) if getattr(args, "specs", None) else []
    if specs:
        _compare_experiment(args, specs)
        return
    if not args.season:
        print(
            "compare requires --season (walk-forward) or repeated --spec "
            "(comparison experiment)",
            file=sys.stderr,
        )
        raise SystemExit(2)
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
    except (ValueError, KeyError, StaleLedgerCacheError) as exc:
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


def _compare_experiment(args: argparse.Namespace, specs: list[str]) -> None:
    """Mint a comparison-flavour experiment from repeated ``--spec``."""
    from fplt.store.experiment_submit import submit_comparison

    data_dir = (
        Path(args.data_dir) if args.data_dir is not None else get_settings().data_dir
    )
    try:
        result = submit_comparison(
            data_dir,
            specs,
            horizon=int(args.horizon),
            question=args.question,
            season=args.season,
            gw=int(args.gw) if args.gw is not None else None,
        )
    except (ValueError, FileNotFoundError) as exc:
        logger.error("compare --spec failed: %s", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(f"experiment_id={result.experiment_id}")
    print(f"build_id={result.build_id}")
    print(f"fresh_runs={result.fresh_runs} reused_runs={result.reused_runs}")
    for arm in result.arms:
        flag = "reused" if arm.reused else "fresh"
        print(f"  {arm.arm} run_id={arm.run_id} ({flag})")


def _exp(args: argparse.Namespace) -> None:
    command = getattr(args, "exp_command", None)
    if command == "submit":
        _exp_submit(args)
        return
    if command == "ls":
        _exp_ls(args)
        return
    if command == "show":
        _exp_show(args)
        return
    if command == "component":
        _exp_component(args)
        return
    raise SystemExit(_not_implemented(f"exp {command}"))


def _exp_component(args: argparse.Namespace) -> None:
    """Run toolkit §2.4 component A/B and print the experiment id."""
    from fplt.backtest.component_experiment import run_component_experiment

    data_dir = (
        Path(args.data_dir) if args.data_dir is not None else get_settings().data_dir
    )
    try:
        result = run_component_experiment(
            data_dir,
            base=str(args.base),
            component=str(args.component),
            arms=list(args.arms),
            window=str(args.window),
            build_id=str(args.build_id),
            n_sims_override=args.n_sims,
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("exp component failed: %s", exc)
        raise SystemExit(1) from exc
    print(f"experiment_id={result.experiment_id}")
    print(f"arms={','.join(result.arms)}")
    print(f"window={result.window.label()}")
    if not result.arm_metrics.is_empty():
        print(f"arm_metrics_rows={result.arm_metrics.height}")
    hits = sum(result.cache.hits.values())
    misses = sum(result.cache.misses.values())
    print(f"cache_hits={hits} cache_misses={misses}")
    if "exp_component" in result.phases:
        print(f"exp_component_s={result.phases['exp_component']:.3f}")


def _exp_submit(args: argparse.Namespace) -> None:
    from fplt.store.experiment_submit import load_eval_ref, submit_batch

    data_dir = (
        Path(args.data_dir) if args.data_dir is not None else get_settings().data_dir
    )
    evals_dir = Path(args.evals_dir) if args.evals_dir is not None else None
    try:
        specs = [load_eval_ref(ref, evals_dir=evals_dir) for ref in args.evals]
        result = submit_batch(
            data_dir,
            specs,
            batch_label=str(args.batch),
            build_id=args.build_id,
        )
    except (ValueError, FileNotFoundError) as exc:
        logger.error("exp submit failed: %s", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(f"batch_id={result.batch_id}")
    print(
        f"experiments={len(result.experiment_ids)} "
        f"fresh_runs={result.fresh_runs} reused_runs={result.reused_runs}"
    )
    for exp_id in result.experiment_ids:
        print(f"  experiment_id={exp_id}")
    for arm in result.arms:
        flag = "reused" if arm.reused else "fresh"
        print(f"  {arm.experiment_id} arm={arm.arm} run_id={arm.run_id} ({flag})")


def _exp_ls(args: argparse.Namespace) -> None:
    from fplt.store.experiments import list_experiment_ids

    data_dir = (
        Path(args.data_dir) if args.data_dir is not None else get_settings().data_dir
    )
    for exp_id in list_experiment_ids(data_dir):
        print(exp_id)


def _exp_show(args: argparse.Namespace) -> None:
    from fplt.store.api import Store

    data_dir = (
        Path(args.data_dir) if args.data_dir is not None else get_settings().data_dir
    )
    try:
        handle = Store(data_dir).experiment(str(args.experiment_id))
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    manifest = handle.manifest
    print(f"experiment_id={handle.experiment_id}")
    print(f"flavour={manifest.flavour}")
    print(f"question={manifest.question}")
    print(f"status={handle.status.label}")
    print(f"batch_id={manifest.batch_id}")
    print(f"eval_spec={manifest.eval_spec}")
    print(f"horizon={manifest.horizon}")
    for arm, run_id in handle.arms.items():
        print(f"  arm={arm} run_id={run_id}")


def _explain(args: argparse.Namespace) -> None:
    if args.forecast_only and args.decision_only:
        print(
            "choose at most one of --forecast-only / --decision-only",
            file=sys.stderr,
        )
        raise SystemExit(2)
    if args.vs_player is not None and args.player is None:
        print("--vs requires --player", file=sys.stderr)
        raise SystemExit(2)
    try:
        text = run_explain(
            get_settings().data_dir,
            args.model,
            args.season,
            int(args.gw),
            horizon=int(args.horizon),
            top_n=int(args.top),
            include_forecast=not bool(args.decision_only),
            include_decision=not bool(args.forecast_only),
            player_key=None if args.player is None else int(args.player),
            vs_player_key=(None if args.vs_player is None else int(args.vs_player)),
        )
    except (FileNotFoundError, KeyError, ValueError) as exc:
        logger.error("explain failed: %s", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(text, end="")


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
                joint_scenarios=bool(args.joint_scenarios),
                n_scenario_sims=args.n_scenario_sims,
                step_weights=_step_weights_for_decay(
                    str(args.horizon_decay), int(args.horizon)
                ),
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
                "set" if secret_value(settings.fpl_session) else "unset",
                args.entry_json,
            )
            view = PITData(DataStore(data_dir)).view(GWKey(args.season, args.gw))
            resolved = resolve_squad_state(
                view,
                entry_json=args.entry_json,
                event=args.gw,
                as_of_gw=args.gw,
                settings=settings,
            )
            squad = resolved.state
            squad_source = resolved.source
            logger.info("resolved squad source=%s", squad_source)
        else:
            logger.info("planning from YAML squad (no FPLT_ENTRY_ID)")
            squad = load_squad_state()
            squad_source = "yaml"
        if args.rebuild:
            owned_n = len(squad.owned)
            squad = rebuild_squad_state(squad)
            logger.info(
                "rebuild draft: cleared %s owned; bank=%s (ITB + selling prices)",
                owned_n,
                squad.bank,
            )
        _solved, path, report, _run_id = run_plan(
            data_dir,
            args.model,
            args.season,
            args.gw,
            args.horizon,
            squad=squad,
            squad_source=squad_source,
            chip_arm=args.chip_arm,
            chip_rules_season=args.chip_rules_season,
            force_play=not bool(args.no_force_play),
            joint_scenarios=bool(args.joint_scenarios),
            n_scenario_sims=args.n_scenario_sims,
            step_weights=_step_weights_for_decay(
                str(args.horizon_decay), int(args.horizon)
            ),
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


def _minutes_measure(args: argparse.Namespace) -> None:
    try:
        result = run_minutes_measurement(
            get_settings().data_dir,
            args.season,
            from_gw=int(args.from_gw),
            to_gw=int(args.to_gw),
            minutes_model=str(args.minutes_model),
            output_dir=args.output_dir,
            top_misses_per_gw=int(args.top_misses_per_gw),
        )
    except (ValueError, KeyError) as exc:
        logger.error("minutes-measure failed: %s", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(format_minutes_measurement_table(result.summary))
    print(format_misses_markdown(result.misses))
    print(f"wrote {result.output_dir / 'summary.json'}")


def _other_measure(args: argparse.Namespace) -> None:
    try:
        result = run_other_points_measurement(
            get_settings().data_dir,
            args.season,
            from_gw=int(args.from_gw),
            to_gw=int(args.to_gw),
            points_model=str(args.points_model),
            n_sims=args.n_sims,
            output_dir=args.output_dir,
        )
    except (ValueError, KeyError) as exc:
        logger.error("other-measure failed: %s", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(format_other_points_measurement_table(result.summary))
    print(format_captain_regret_markdown(result.captain_regret))
    print(f"wrote {result.output_dir / 'summary.json'}")


def _predicted_xi_audit(args: argparse.Namespace) -> None:
    try:
        result = run_predicted_xi_audit(
            get_settings().data_dir,
            args.season,
            from_gw=int(args.from_gw),
            to_gw=int(args.to_gw),
            source=args.source,
            min_accuracy=float(args.min_accuracy),
            output_dir=args.output_dir,
        )
    except (ValueError, KeyError) as exc:
        logger.error("predicted-xi-audit failed: %s", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(format_predicted_xi_audit_table(result.summary))
    print(
        format_audit_markdown(
            result.summary,
            passed=result.passed_min_accuracy,
            min_accuracy=float(args.min_accuracy),
        )
    )
    print(f"wrote {result.output_dir / 'summary.json'}")
    if not result.passed_min_accuracy:
        raise SystemExit(1)


def _eval_components(args: argparse.Namespace) -> None:
    sims: list[int] | tuple[int, ...] | None
    minutes_only = bool(getattr(args, "minutes_only", False))
    if minutes_only or args.skip_sims:
        sims = []
    elif args.sims is not None:
        sims = list(args.sims)
    else:
        sims = None
    if minutes_only:
        score_models: list[str] | None = []
        involvement_model: list[str] | None = []
        # argparse default is empirical_minutes_v1; swap to v2 for the board.
        minutes_model = list(args.minutes_model) if args.minutes_model else []
        if minutes_model == ["empirical_minutes_v1"]:
            minutes_model = ["empirical_minutes_v2"]
    else:
        score_models = list(args.score_models) if args.score_models else None
        involvement_model = (
            list(args.involvement_model) if args.involvement_model else []
        )
        minutes_model = list(args.minutes_model) if args.minutes_model else []
    try:
        result = run_component_eval(
            get_settings().data_dir,
            args.season,
            from_gw=int(args.from_gw),
            to_gw=int(args.to_gw),
            fit_gw=args.fit_gw,
            score_models=score_models,
            minutes_model=minutes_model,
            involvement_model=involvement_model,
            layered_model=str(args.layered_model),
            sims=sims,
            output_dir=args.output_dir,
            points_models=list(args.points_models) if args.points_models else None,
            steps=int(args.steps),
        )
    except (ValueError, KeyError, StaleLedgerCacheError) as exc:
        logger.error("eval-components failed: %s", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(format_component_table(result.summary))
    if not result.sims.is_empty():
        print(result.sims)
    if not result.horizon_skill.is_empty():
        print(format_skill_curve(result.horizon_skill))
    print(f"wrote {result.output_dir / 'summary.json'}")


def _horizon_skill(args: argparse.Namespace) -> None:
    try:
        result = run_horizon_skill(
            get_settings().data_dir,
            args.season,
            models=list(args.models) if args.models else None,
            from_gw=int(args.from_gw),
            to_gw=int(args.to_gw),
            steps=int(args.steps),
            seasons_back=args.seasons_back,
            with_openfpl=bool(args.with_openfpl),
            openfpl_model_dir=args.openfpl_model_dir,
            output_dir=args.output_dir,
        )
    except (ValueError, KeyError, StaleLedgerCacheError) as exc:
        logger.error("horizon-skill failed: %s", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(format_skill_curve(result.curve))
    print(f"wrote {result.output_dir / 'horizon_skill.json'}")


def _ensemble_sensitivity(args: argparse.Namespace) -> None:
    from fplt.optimise.ensemble_sensitivity import (
        format_ensemble_sensitivity_board,
        run_ensemble_sensitivity_from_data_dir,
        write_ensemble_sensitivity_report,
    )

    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else get_settings().data_dir / "ensemble_sensitivity"
    )
    try:
        report = run_ensemble_sensitivity_from_data_dir(
            get_settings().data_dir,
            season=str(args.season),
            from_gw=int(args.from_gw),
            to_gw=int(args.to_gw),
            horizon=int(args.horizon),
            control=str(args.control),
            challengers=list(args.challengers),
        )
        paths = write_ensemble_sensitivity_report(report, output_dir)
    except (KeyError, ValueError) as exc:
        logger.error("ensemble-sensitivity failed: %s", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(format_ensemble_sensitivity_board(report.summary))
    print(f"wrote {paths['json']}")


def _pruning_loss(args: argparse.Namespace) -> None:
    from fplt.optimise.highs_options import HighsSolveOptions
    from fplt.optimise.pruning_loss import (
        DEFAULT_POOL_TARGETS,
        default_sweep_arms,
        format_pruning_loss_summary,
        run_pruning_loss_harness,
        sample_synthetic_scenarios,
        write_pruning_loss_report,
    )

    targets = (
        tuple(int(value) for value in args.targets)
        if args.targets
        else DEFAULT_POOL_TARGETS
    )
    output_dir = (
        Path(args.output_dir)
        if args.output_dir is not None
        else get_settings().data_dir / "pruning_loss"
    )
    try:
        scenarios = sample_synthetic_scenarios(
            int(args.n_scenarios),
            seed=int(args.seed),
            horizon=int(args.horizon),
        )
        report = run_pruning_loss_harness(
            scenarios,
            arms=default_sweep_arms(targets),
            reference_options=HighsSolveOptions(
                time_limit=float(args.reference_time_limit),
                mip_rel_gap=0.0,
                accept_time_limit=True,
            ),
            sweep_options=HighsSolveOptions(
                time_limit=float(args.sweep_time_limit),
                mip_rel_gap=0.0,
                accept_time_limit=True,
            ),
        )
        paths = write_pruning_loss_report(report, output_dir)
    except ValueError as exc:
        logger.error("pruning-loss failed: %s", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(format_pruning_loss_summary(report.summary))
    print(f"wrote {paths['json']}")


def _promote_eval(args: argparse.Namespace) -> None:
    dry_run = not bool(args.execute)
    try:
        spec = load_eval_spec(str(args.eval))
        if dry_run:
            plan = plan_promote_eval(
                spec,
                dry_run=True,
                tier=args.tier,
                from_gw=args.from_gw,
                to_gw=args.to_gw,
                seasons=list(args.seasons) if args.seasons else None,
                skip_decisions=bool(args.skip_decisions),
            )
            if args.json:
                print(plan_to_json(plan), end="")
            else:
                print(format_plan(plan))
            return
        result = execute_promote_eval(
            spec,
            data_dir=get_settings().data_dir,
            dry_run=False,
            tier=args.tier,
            from_gw=args.from_gw,
            to_gw=args.to_gw,
            seasons=list(args.seasons) if args.seasons else None,
            skip_decisions=bool(args.skip_decisions),
            allow_dirty=bool(args.allow_dirty),
            output_dir=args.output_dir,
            argv=sys.argv[1:],
        )
    except (FileNotFoundError, ValueError, KeyError) as exc:
        logger.error("promote-eval failed: %s", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(result.recommendation, end="")
    print(f"wrote {result.output_dir}", file=sys.stderr)
    if result.preflight.failures and result.exit_code == 2:
        print(
            "preflight failed: " + "; ".join(result.preflight.failures),
            file=sys.stderr,
        )
    if result.exit_code != 0:
        raise SystemExit(result.exit_code)


def _bakeoff(args: argparse.Namespace) -> None:
    dry_run = not bool(args.execute)
    try:
        if args.bakeoff:
            spec = load_bakeoff_spec(str(args.bakeoff))
        elif args.models and args.control and args.season:
            from_gw = int(args.from_gw) if args.from_gw is not None else 1
            to_gw = int(args.to_gw) if args.to_gw is not None else 20
            spec = bakeoff_from_models(
                name=str(args.name),
                control=str(args.control),
                models=list(args.models),
                seasons=[str(args.season)],
                from_gw=from_gw,
                to_gw=to_gw,
                seasons_back=int(args.seasons_back),
                modes=list(args.modes) if args.modes else None,
                steps=int(args.steps),
                pairwise_evals=list(args.pairwise_evals)
                if args.pairwise_evals
                else None,
            )
        else:
            raise ValueError(
                "pass --bakeoff <name> or --models … --control … --season …"
            )
        if dry_run:
            plan = plan_bakeoff(
                spec,
                dry_run=True,
                from_gw=args.from_gw,
                to_gw=args.to_gw,
            )
            if args.json:
                print(bakeoff_plan_to_json(plan), end="")
            else:
                print(format_bakeoff_plan(plan), end="")
            return
        result = execute_bakeoff(
            spec,
            data_dir=get_settings().data_dir,
            dry_run=False,
            from_gw=args.from_gw,
            to_gw=args.to_gw,
            output_dir=args.output_dir,
            with_openfpl=bool(args.with_openfpl),
            openfpl_model_dir=args.openfpl_model_dir,
        )
    except (FileNotFoundError, ValueError, KeyError, StaleLedgerCacheError) as exc:
        logger.error("bakeoff failed: %s", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(result.ranking, end="")
    print(f"wrote {result.output_dir}", file=sys.stderr)
    if result.exit_code != 0:
        raise SystemExit(result.exit_code)


def _sim_season(args: argparse.Namespace) -> None:
    data_dir = get_settings().data_dir
    try:
        for model_name in list(args.models):
            report = run_chip_arm_sim_season(
                data_dir,
                model_name,
                args.season,
                int(args.from_gw),
                int(args.to_gw),
                int(args.horizon),
                arms=list(args.chip_arms),
                chip_rules_season=args.chip_rules_season,
                n_paths=int(args.n_paths),
                seed=int(args.seed),
                with_attainable=bool(args.with_attainable),
                force_play=not bool(args.no_force_play),
            )
            print(f"## model={model_name}")
            print(format_simulated_season_report(report))
            print()
    except (KeyError, ValueError) as exc:
        logger.error("sim-season failed: %s", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc


def _regret_board(args: argparse.Namespace) -> None:
    data_dir = get_settings().data_dir
    force_play = not bool(args.no_force_play)
    oracle_force: bool | None
    if bool(args.oracle_no_force_play):
        oracle_force = False
    else:
        oracle_force = None
    price = args.price_forecast
    if price == "flat":
        price = None
    try:
        result = run_regret_board(
            data_dir,
            season=str(args.season),
            from_gw=int(args.from_gw),
            to_gw=int(args.to_gw),
            horizon=int(args.horizon),
            model=str(args.model),
            chip_arm=args.chip_arm,
            chip_rules_season=args.chip_rules_season,
            force_play=force_play,
            lineup_reopt=not bool(args.no_lineup_reopt),
            price_forecast_method=price,
            oracle_chip_arm=args.oracle_chip_arm,
            oracle_force_play=oracle_force,
            simulate_policy=not bool(args.summarise_only)
            and args.policy_parquet is None,
            simulate_oracle=not bool(args.summarise_only),
            policy_parquet=args.policy_parquet,
        )
    except (KeyError, ValueError) as exc:
        logger.error("regret-board failed: %s", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(result.report_md)
    if args.output_dir is not None:
        path = write_regret_board(result, Path(args.output_dir))
        print(path)


def _weekly_report(args: argparse.Namespace) -> None:
    from fplt.optimise.weekly_report import run_weekly_report

    data_dir = get_settings().data_dir
    try:
        result = run_weekly_report(
            data_dir,
            season=str(args.season),
            gw=int(args.gw),
            model_id=args.model_id,
            hindsight_best=args.hindsight_best,
            n_fixtures=args.n_fixtures,
            notify=not bool(args.no_notify),
            settings=get_settings(),
        )
    except ValueError as exc:
        logger.error("weekly-report failed: %s", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(result.report_md, end="")
    print(result.report_dir / "weekly.md")
    if result.notified:
        print("notified: yes")
    else:
        print(f"notified: no ({result.notify_skipped_reason})")


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
        if bool(args.chip_board):
            if bool(args.with_oracle):
                raise ValueError("--with-oracle is not supported with --chip-board")
            summary = run_chip_decision_board(
                data_dir,
                list(args.models),
                args.season,
                args.from_gw,
                args.to_gw,
                args.horizon,
                arms=list(args.chip_arms),
                chip_rules_season=args.chip_rules_season,
                simulate=not bool(args.summarise_only),
                max_mean_ppg=None if max_mean <= 0 else max_mean,
                force_play=not bool(args.no_force_play),
                lineup_reopt=not bool(args.no_lineup_reopt),
                price_forecast_method=args.price_forecast,
                joint_scenarios=bool(args.joint_scenarios),
                n_scenario_sims=args.n_scenario_sims,
                step_weights=_step_weights_for_decay(
                    str(args.horizon_decay), int(args.horizon)
                ),
            )
            print(format_chip_decision_scoreboard(summary))
        else:
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
            if bool(args.with_oracle):
                if not bool(args.summarise_only):
                    run_hindsight_oracle(
                        data_dir,
                        args.season,
                        args.from_gw,
                        args.to_gw,
                        args.horizon,
                        actuals_model=list(args.models)[0],
                    )
                oracle = read_decisions(data_dir).filter(
                    (pl.col("model_id") == ORACLE_MODEL_ID)
                    & (pl.col("season") == args.season)
                )
                if args.from_gw > 1 or args.to_gw < 38:
                    oracle = oracle.filter(
                        (pl.col("gw") >= args.from_gw) & (pl.col("gw") <= args.to_gw)
                    )
                summary = enrich_decision_summary(summary, oracle)
            print(format_decision_scoreboard(summary))
    except (KeyError, ValueError) as exc:
        logger.error("decisions failed: %s", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    path = data_dir / "ledger" / "decisions" / "part.parquet"
    print(path)


def _perf_report(args: argparse.Namespace) -> None:
    """Print a hotspot ranking or nightly bench trend under ``data_dir``."""
    import json
    import os

    from fplt.utils.timing import (
        format_hotspot_report,
        load_timing_records,
        rank_hotspots,
    )

    if bool(getattr(args, "trend", False)):
        from fplt.automation.bench import format_trend_report, list_bench_results

        lab_dir = (
            Path(args.lab_dir)
            if getattr(args, "lab_dir", None)
            else Path(os.environ.get("FPLT_LAB_DIR", "lab"))
        )
        history = list_bench_results(lab_dir, limit=int(args.limit) or None)
        if bool(args.json):
            payload = {
                "lab_dir": str(lab_dir),
                "n_nights": len(history),
                "nights": [item.model_dump_json_doc() for item in history],
            }
            print(json.dumps(payload, indent=2))
            return
        print(format_trend_report(history, limit=int(args.limit) or 14))
        return

    data_dir = Path(args.data_dir) if args.data_dir else get_settings().data_dir
    records = load_timing_records(
        data_dir,
        kind=args.kind,
        limit=int(args.limit),
    )
    if bool(args.json):
        hotspots = rank_hotspots(records)[: int(args.hotspots)]
        payload = {
            "data_dir": str(data_dir),
            "n_records": len(records),
            "records": [
                {
                    "kind": record.get("kind"),
                    "total_s": record.get("total_s"),
                    "meta": record.get("meta"),
                    "path": record.get("_path"),
                }
                for record in records
            ],
            "hotspots": [
                {
                    "phase": spot.phase,
                    "seconds": spot.seconds,
                    "share_pct": spot.share_pct,
                    "category": spot.category,
                }
                for spot in hotspots
            ],
        }
        print(json.dumps(payload, indent=2))
        return
    print(
        format_hotspot_report(
            records,
            limit=int(args.hotspots),
        )
    )


def _archive_benchmarks(args: argparse.Namespace) -> None:
    data_dir = get_settings().data_dir
    include_openfpl = not bool(args.skip_openfpl)
    try:
        if args.season is None and args.gw is None:
            from datetime import UTC, datetime

            from fplt.automation.cycle import due_gameweek

            key = due_gameweek(DataStore(data_dir), datetime.now(UTC))
            if key is None:
                print("no deadline inside three hours", file=sys.stderr)
                raise SystemExit(1)
            path = archive_if_missing(data_dir, key, include_openfpl=include_openfpl)
            if path is None:
                print(f"already archived: {key.season} gw{key.gw}")
                return
        else:
            if args.season is None or args.gw is None:
                raise ValueError("--season and --gw must be given together")
            key = GWKey(str(args.season), int(args.gw))
            path = archive_deadline_benchmarks(
                data_dir, key, include_openfpl=include_openfpl
            )
    except (
        BenchmarkArchiveError,
        FileExistsError,
        KeyError,
        NotImplementedError,
        ValueError,
    ) as exc:
        logger.error("archive-benchmarks failed: %s", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(path)


def _archive_compare(args: argparse.Namespace) -> None:
    _configure_cli_logging()
    try:
        result = compare_to_archives(
            get_settings().data_dir,
            season=str(args.season),
            model=str(args.model),
            from_gw=int(args.from_gw),
            to_gw=int(args.to_gw),
            sources=list(args.sources),
            horizon=int(args.horizon),
            output_dir=args.output_dir,
        )
    except (KeyError, ValueError) as exc:
        logger.error("archive-compare failed: %s", exc)
        print(str(exc), file=sys.stderr)
        raise SystemExit(2) from exc
    print(format_archive_board(result))
    print(result.output_dir)


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
    except SettingsConfigError:
        raise
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
        if command == "check":
            raise SystemExit(_auth_check(args))
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


def _auth_check(args: argparse.Namespace) -> int:
    """Run ops-01 token expiry check; optionally notify and set exit code."""
    from datetime import UTC, datetime

    from fplt.automation.notify import notify_recommendation
    from fplt.automation.runbook import OPS_01_TOKEN_EXPIRY
    from fplt.data.store import DataStore

    deadline = None
    if bool(args.deadline_aware):
        try:
            from fplt.automation.checkpoints import next_upcoming_deadline

            upcoming = next_upcoming_deadline(
                DataStore(get_settings().data_dir),
                datetime.now(tz=UTC),
            )
            if upcoming is not None:
                deadline = upcoming[1]
        except (OSError, ValueError, KeyError) as exc:
            logger.info("auth check: no staged deadline (%s)", exc)

    result = check_token(
        warn_days=float(args.warn_days),
        next_deadline=deadline,
        deadline_aware=bool(args.deadline_aware),
        allow_missing=bool(args.allow_missing),
    )
    print(format_auth_check(result), end="")
    if not bool(args.no_notify) and result.level in ("warning", "urgent"):
        priority = "5" if result.level == "urgent" else None
        notify_recommendation(
            get_settings(),
            f"fplt auth {result.level}",
            result.message,
            priority=priority,
            runbook_anchor=OPS_01_TOKEN_EXPIRY,
        )
    return result.exit_code


def _step_weights_for_decay(
    method: str,
    horizon: int,
) -> tuple[float, ...] | None:
    """Resolve CLI ``--horizon-decay`` into ``plan_squad`` step weights.

    Geometric (live control) returns ``None`` so ``SquadState.decay`` applies.
    Skill loads measured RMSE ratios from
    ``configs/optimise/horizon_skill_weights.yaml``.
    """
    chosen: HorizonDecayMethod
    if method == "geometric":
        return None
    if method == "skill":
        chosen = "skill"
    else:
        raise ValueError(f"unknown horizon decay method {method!r}")
    # decay unused on the skill path; pass 1.0 as a placeholder.
    return resolve_step_weights(horizon, decay=1.0, method=chosen)


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
