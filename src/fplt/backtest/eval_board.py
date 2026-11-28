"""Multi-model walk-forward eval into the prediction ledger.

Runs the core leaderboard set (baselines, layered, LightGBM) for one season
and optionally OpenFPL when configured. Afterward ``make leaderboard`` reads
the same ledger. This is not a CI quality gate.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

# Side-effect imports so kinds resolve even before backtest/run.py registers
# them (see PR #27 for the shared import on the CLI path).
import fplt.models.direct  # noqa: F401  # pyright: ignore[reportUnusedImport]
import fplt.models.ensemble  # noqa: F401  # pyright: ignore[reportUnusedImport]
import fplt.models.external.openfpl  # noqa: F401  # pyright: ignore[reportUnusedImport]
from fplt.backtest.eval_set import (
    CORE_MODELS,
    OPENFPL_COMPARE_FROM,
    OPENFPL_MODEL,
    openfpl_skip_note,
    resolve_eval_models,
)
from fplt.backtest.run import run_backtest
from fplt.backtest.walk_forward import walk_forward
from fplt.data.outcomes import stage_outcomes
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.spec import ModelSpec, get_model, load_model_specs
from fplt.store.runs import write_backtest_run
from fplt.types import GWKey
from fplt.utils.timing import (
    TimingSession,
    binds_timing_artifacts,
    format_kind_summary,
    timed_phase,
    timing_session,
    write_timing_record,
)

logger = logging.getLogger(__name__)

Refit = Literal["every_gw", "every_n_gw", "never"]

# Re-exports for scripts and tests that import from this module.
__all__ = [
    "CORE_MODELS",
    "OPENFPL_COMPARE_FROM",
    "OPENFPL_MODEL",
    "EvalResult",
    "resolve_eval_models",
    "run_eval_board",
]


@dataclass(frozen=True)
class EvalResult:
    """Specs that wrote predictions during one eval run."""

    specs: tuple[ModelSpec, ...]
    skipped_openfpl: str | None = None


@binds_timing_artifacts()
def run_eval_board(
    data_dir: Path,
    season: str,
    from_gw: int = 1,
    to_gw: int = 38,
    *,
    models: list[str] | None = None,
    with_openfpl: bool = False,
    openfpl_model_dir: str | None = None,
    refit: Refit | None = None,
    seasons_back: int | None = None,
    time_decay: float | None = None,
    steps: int = 1,
) -> EvalResult:
    """Walk-forward backtest the eval set into ``data_dir``'s ledger.

    OpenFPL uses ``openfpl_model_dir`` when provided (overrides the YAML
    ``model_dir``). The default YAML path is fetched on first score. An
    empty ``model_dir`` raises rather than recording zeros.

    Args:
        data_dir: Root with ``staged/``; receives ``ledger/``.
        season: Evaluation season.
        from_gw: First as-of gameweek.
        to_gw: Last as-of gameweek, inclusive.
        models: Explicit specs; default is the core set (± OpenFPL).
        with_openfpl: Include OpenFPL in the default set.
        openfpl_model_dir: Optional override for OpenFPL's fitted ensemble.
        refit: Override each spec's refit policy.
        seasons_back: Override each spec's training window.
        time_decay: Override ``TrainWindow.time_decay`` (calendar λ).
        steps: Horizon length passed to ``walk_forward``.

    Returns:
        Specs that ran, plus a skip note when OpenFPL was declined.

    Raises:
        ValueError: Invalid range, unknown spec, or OpenFPL season floor.
        KeyError: A spec kind is not registered.
    """
    chosen = resolve_eval_models(
        season,
        models=models,
        with_openfpl=with_openfpl,
        openfpl_model_dir=openfpl_model_dir,
    )
    skipped = (
        None
        if models is not None
        else openfpl_skip_note(
            season,
            chosen,
            with_openfpl=with_openfpl,
            openfpl_model_dir=openfpl_model_dir,
        )
    )
    others = [name for name in chosen if name != OPENFPL_MODEL]
    include_openfpl = OPENFPL_MODEL in chosen
    ran: list[ModelSpec] = []
    board_timings = TimingSession()
    with timing_session(board_timings):
        if others:
            with timed_phase("board/core"):
                ran.extend(
                    run_backtest(
                        data_dir,
                        season,
                        from_gw,
                        to_gw,
                        others,
                        refit=refit,
                        seasons_back=seasons_back,
                        time_decay=time_decay,
                        steps=steps,
                    )
                )
        if include_openfpl:
            with timed_phase("board/openfpl"):
                ran.append(
                    _run_openfpl(
                        data_dir,
                        season,
                        from_gw,
                        to_gw,
                        model_dir=openfpl_model_dir,
                        refit=refit,
                        seasons_back=seasons_back,
                        time_decay=time_decay,
                        steps=steps,
                    )
                )
                stage_outcomes(DataStore(data_dir))
    meta = {
        "season": season,
        "from_gw": from_gw,
        "to_gw": to_gw,
        "n_models": len(ran),
        "steps": steps,
    }
    logger.info(format_kind_summary("eval_board", board_timings, meta=meta))
    write_timing_record("eval_board", board_timings, meta=meta)
    if skipped:
        logger.info(skipped)
    return EvalResult(specs=tuple(ran), skipped_openfpl=skipped)


def _run_openfpl(
    data_dir: Path,
    season: str,
    from_gw: int,
    to_gw: int,
    *,
    model_dir: str | None,
    refit: Refit | None,
    seasons_back: int | None,
    steps: int,
    time_decay: float | None = None,
) -> ModelSpec:
    """Backtest OpenFPL, optionally overriding ``params.model_dir``."""
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    if seasons_back is not None and seasons_back < 1:
        raise ValueError(f"seasons_back must be positive, got {seasons_back}")
    if time_decay is not None and time_decay < 0:
        raise ValueError(f"time_decay must be non-negative, got {time_decay}")
    if not 1 <= from_gw <= to_gw <= 38:
        raise ValueError(f"gameweeks must be in 1..38, got {from_gw}..{to_gw}")
    specs = load_model_specs()
    if OPENFPL_MODEL not in specs:
        raise ValueError(f"unknown model spec {OPENFPL_MODEL!r}")
    spec = _configured(specs[OPENFPL_MODEL], refit, seasons_back, time_decay)
    if model_dir is not None:
        params = dict(spec.params)
        params["model_dir"] = model_dir
        spec = spec.model_copy(update={"params": params})
    if not str(spec.params.get("model_dir") or "").strip():
        logger.warning(
            "openfpl_v1 model_dir is empty; scoring raises instead of "
            "writing zeros. Set params.model_dir or --openfpl-model-dir."
        )
    store = DataStore(data_dir)
    pit = PITData(store)
    as_ofs = [GWKey(season, gw) for gw in range(from_gw, to_gw + 1)]
    model = get_model(spec.kind)(spec)
    frame = walk_forward(model, pit, as_ofs, steps=steps)
    deadlines = [(key, store.deadline(key)) for key in as_ofs]
    write_backtest_run(
        data_dir,
        spec=spec,
        as_ofs=deadlines,
        horizon=steps,
        frame=frame,
        purpose=f"eval_board openfpl {season}",
    )
    return spec


def _configured(
    spec: ModelSpec,
    refit: Refit | None,
    seasons_back: int | None,
    time_decay: float | None = None,
) -> ModelSpec:
    updates: dict[str, object] = {}
    if refit is not None:
        updates["refit"] = refit
    window_updates: dict[str, object] = {}
    if seasons_back is not None:
        window_updates["seasons_back"] = seasons_back
    if time_decay is not None:
        window_updates["time_decay"] = time_decay
    if window_updates:
        updates["train_window"] = spec.train_window.model_copy(update=window_updates)
    if not updates:
        return spec
    return spec.model_copy(update=updates)


# Alias kept so older tests/scripts can import the skip helper from here.
_openfpl_skip_note = openfpl_skip_note
