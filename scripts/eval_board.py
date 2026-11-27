"""Run walk-forward backtests for the core leaderboard models.

Example::

    uv run python scripts/eval_board.py --season 2024-25
    make eval-board SEASON=2024-25

Core set: rolling_mean_v1, ep_next_v1, dc_layered_v1, lgbm_direct_v1.

OpenFPL (openfpl_v1) is optional. Include it with ``--with-openfpl``,
``--openfpl-model-dir``, or a YAML ``model_dir`` other than the fetched
default checkout. Only seasons from 2024-25 onwards are valid (earlier
seasons are in OpenFPL's training data). The first real score fetches the
ensemble into ``external/openfpl/OpenFPL`` and syncs that project's venv.

After a successful run, open the Streamlit board with ``make leaderboard``.

Note: ``lgbm_direct`` must be registered for end-to-end LightGBM eval. On
``main`` that lands via PR #27 (``import fplt.models.direct`` in
``backtest/run.py``); this script also imports the package so a local run
works before that merge.
"""

from __future__ import annotations

import argparse
import logging
import sys
from typing import Literal

from fplt.orchestration.eval_job import (
    CORE_MODELS,
    OPENFPL_COMPARE_FROM,
    OPENFPL_MODEL,
    run_eval_board,
)
from fplt.settings import get_settings
from fplt.utils.logging import configure_logging

_REFITS = ("every_gw", "every_n_gw", "never")
logger = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    """Build the ``eval_board`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="eval_board",
        description=(
            "Walk-forward backtest the core models into the prediction ledger "
            f"(optional {OPENFPL_MODEL} from {OPENFPL_COMPARE_FROM}+)."
        ),
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="DEBUG logs",
    )
    parser.add_argument(
        "-q",
        "--quiet",
        action="store_true",
        help="WARNING+ only",
    )
    parser.add_argument("--season", required=True, help="Evaluation season")
    parser.add_argument("--from-gw", type=int, default=1)
    parser.add_argument("--to-gw", type=int, default=38)
    parser.add_argument(
        "--models",
        nargs="+",
        help=f"Override the default set ({', '.join(CORE_MODELS)})",
    )
    parser.add_argument(
        "--with-openfpl",
        action="store_true",
        help=f"Include {OPENFPL_MODEL} in the default set",
    )
    parser.add_argument(
        "--openfpl-model-dir",
        help="Override openfpl_v1 params.model_dir (also implies including it)",
    )
    parser.add_argument("--refit", choices=_REFITS)
    parser.add_argument("--seasons-back", type=int)
    parser.add_argument("--steps", type=int, default=1)
    return parser


def main(argv: list[str] | None = None) -> int:
    """Parse args, run the eval board, print model ids."""
    args = build_parser().parse_args(argv)
    configure_logging(verbose=bool(args.verbose), quiet=bool(args.quiet))
    refit: Literal["every_gw", "every_n_gw", "never"] | None = args.refit
    try:
        result = run_eval_board(
            get_settings().data_dir,
            args.season,
            args.from_gw,
            args.to_gw,
            models=list(args.models) if args.models else None,
            with_openfpl=bool(args.with_openfpl),
            openfpl_model_dir=args.openfpl_model_dir,
            refit=refit,
            seasons_back=args.seasons_back,
            steps=args.steps,
        )
    except (ValueError, KeyError) as exc:
        logger.exception("eval_board failed: %s", exc)
        print(str(exc), file=sys.stderr)
        return 2
    for spec in result.specs:
        window = spec.train_window.seasons_back
        print(
            f"{spec.name} refit={spec.refit} seasons_back={window} "
            f"model_id={spec.model_id}"
        )
    if result.skipped_openfpl:
        print(result.skipped_openfpl, file=sys.stderr)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
