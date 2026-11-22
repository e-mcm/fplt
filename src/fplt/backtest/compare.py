"""Fair multi-model comparison under shared train/test windows.

Runs every selected model over the same test gameweeks in two modes:

* ``walk_forward`` — ``refit=every_gw`` (existing backtest style).
* ``fixed`` — fit once at ``fit_gw`` (start of test / end of train) and hold
  parameters for the whole test window (``refit=never``).

Internally each arm × mode is produced via ``produce_ledger`` (P2.5 content-
addressed parquet cache), then appended to the shared ledger. Predictions land
in the usual ledger. A JSON + parquet summary (RMSE headline, MAE secondary,
Spearman, mean per-GW top-10 precision, Brier) is written for Streamlit or
ad-hoc review, scored on the decision-relevant player pool. The ``fplt
compare`` CLI surface is unchanged.
"""

from __future__ import annotations

import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import polars as pl

# Side-effect imports so YAML kinds resolve when compare is used alone.
import fplt.models.baselines  # noqa: F401  # pyright: ignore[reportUnusedImport]
import fplt.models.composite  # noqa: F401  # pyright: ignore[reportUnusedImport]
import fplt.models.direct  # noqa: F401  # pyright: ignore[reportUnusedImport]
import fplt.models.ensemble  # noqa: F401  # pyright: ignore[reportUnusedImport]
import fplt.models.external.openfpl  # noqa: F401  # pyright: ignore[reportUnusedImport]
from fplt.backtest.decision_pool import (
    HEADLINE_METRIC,
    POOL_FILTER_DESCRIPTION,
    filter_decision_pool,
)
from fplt.backtest.eval_set import (
    CORE_MODELS,
    OPENFPL_MODEL,
    openfpl_skip_note,
    resolve_eval_models,
)
from fplt.backtest.ledger import append_predictions, fill_actuals, read_predictions
from fplt.backtest.metrics import (
    calibration_tables,
    format_calibration_markdown,
    overall_summary,
    summarise,
)
from fplt.backtest.produce_ledger import produce_ledger
from fplt.data.store import DataStore
from fplt.models.spec import ModelSpec, load_model_specs

logger = logging.getLogger(__name__)

CompareMode = Literal["walk_forward", "fixed"]
_MODE_REFIT: dict[CompareMode, Literal["every_gw", "never"]] = {
    "walk_forward": "every_gw",
    "fixed": "never",
}
DEFAULT_MODES: tuple[CompareMode, ...] = ("walk_forward", "fixed")


@dataclass(frozen=True)
class CompareRun:
    """One model × mode that wrote predictions during a compare."""

    model_name: str
    mode: CompareMode
    spec: ModelSpec
    cache_hit: bool = False


@dataclass(frozen=True)
class CompareResult:
    """Artifacts from ``compare_models``."""

    season: str
    test_from_gw: int
    test_to_gw: int
    fit_gw: int
    seasons_back: int | None
    modes: tuple[CompareMode, ...]
    runs: tuple[CompareRun, ...]
    summary: pl.DataFrame
    detail: pl.DataFrame
    output_dir: Path
    calibration: pl.DataFrame | None = None
    time_decay: float | None = None
    skipped_openfpl: str | None = None


def compare_models(
    data_dir: Path,
    season: str,
    *,
    models: list[str] | None = None,
    test_from_gw: int = 1,
    test_to_gw: int = 38,
    fit_gw: int | None = None,
    seasons_back: int | None = None,
    time_decay: float | None = None,
    modes: list[CompareMode] | tuple[CompareMode, ...] | None = None,
    steps: int = 1,
    with_openfpl: bool = False,
    openfpl_model_dir: str | None = None,
    output_dir: Path | None = None,
    use_cache: bool = True,
) -> CompareResult:
    """Backtest selected models under shared windows and both compare modes.

    All models share ``seasons_back``, ``time_decay``, ``fit_gw``, and the test
    gameweek range. ``walk_forward`` refits each deadline; ``fixed`` fits once at
    ``fit_gw`` and reuses that fit. When ``fit_gw`` is before ``test_from_gw``,
    the fit deadline is included only to train — its prediction rows are dropped
    from the ledger append for that mode.

    Each arm × mode is produced via ``produce_ledger`` (content-addressed cache
    under ``data_dir/ledger_cache/``), then appended to the shared ledger.

    Args:
        data_dir: Root with ``staged/``; receives ``ledger/`` and compare outputs.
        season: Evaluation season such as ``"2024-25"``.
        models: Explicit YAML specs; default is the eval-board core set.
        test_from_gw: First as-of gameweek scored in the test window.
        test_to_gw: Last as-of gameweek, inclusive.
        fit_gw: Deadline used for the fixed-parameter fit. Defaults to
            ``test_from_gw`` (start of test / end of train).
        seasons_back: Override each spec's training window length.
        time_decay: Override ``TrainWindow.time_decay`` (calendar λ). Honoured
            by LightGBM sample weights; ignored by Dixon–Coles (use ``xi``).
        modes: Subset of ``walk_forward`` / ``fixed``. Default is both.
        steps: Horizon length passed to ``walk_forward``.
        with_openfpl: Include OpenFPL in the default set.
        openfpl_model_dir: Optional OpenFPL ``model_dir`` override.
        output_dir: Where to write summary JSON/parquet. Defaults under
            ``data_dir/compare/``.
        use_cache: When true (default), reuse content-addressed ledgers.

    Returns:
        Runs performed plus overall and per-position summary frames.

    Raises:
        ValueError: Invalid windows, empty models, or OpenFPL season floor.
        KeyError: A spec kind is not registered.
        StaleLedgerCacheError: Cached key.json does not match the computed key.
    """
    chosen_modes = _resolve_modes(modes)
    fit_at = test_from_gw if fit_gw is None else fit_gw
    _validate_window(test_from_gw, test_to_gw, fit_at, steps, seasons_back, time_decay)

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

    specs = load_model_specs()
    missing = [name for name in chosen if name not in specs]
    if missing:
        raise ValueError(f"unknown model spec {missing[0]!r}")

    store = DataStore(data_dir)
    out = output_dir or _default_output_dir(data_dir, season, test_from_gw, test_to_gw)
    out.mkdir(parents=True, exist_ok=True)
    logger.info(
        "compare %s gw %s-%s fit_gw=%s steps=%s modes=%s models=%s out=%s",
        season,
        test_from_gw,
        test_to_gw,
        fit_at,
        steps,
        ",".join(chosen_modes),
        ",".join(chosen),
        out,
    )

    runs: list[CompareRun] = []
    for mode in chosen_modes:
        for name in chosen:
            spec = _configured(
                specs[name],
                refit=_MODE_REFIT[mode],
                seasons_back=seasons_back,
                time_decay=time_decay,
                openfpl_model_dir=openfpl_model_dir if name == OPENFPL_MODEL else None,
            )
            model_dir = str(spec.params.get("model_dir") or "").strip()
            if name == OPENFPL_MODEL and not model_dir:
                logger.warning(
                    "openfpl_v1 model_dir is empty; scoring raises instead of "
                    "writing zeros. Set params.model_dir or --openfpl-model-dir."
                )
            logger.info(
                "compare run model=%s mode=%s refit=%s model_id=%s",
                name,
                mode,
                spec.refit,
                spec.model_id,
            )
            logger.info(
                "%s %s starting refit=%s seasons_back=%s",
                mode,
                name,
                spec.refit,
                spec.train_window.seasons_back,
            )
            started = time.perf_counter()
            produced = produce_ledger(
                data_dir,
                spec,
                season,
                mode=mode,
                test_from_gw=test_from_gw,
                test_to_gw=test_to_gw,
                fit_gw=fit_at,
                steps=steps,
                seasons_back=seasons_back,
                use_cache=use_cache,
            )
            logger.info("%s/%s wrote %s rows", name, mode, produced.frame.height)
            append_predictions(data_dir, produced.frame)
            logger.info(
                "%s %s wrote %s rows in %.1fs (cache_hit=%s)",
                mode,
                name,
                produced.frame.height,
                time.perf_counter() - started,
                produced.cache_hit,
            )
            runs.append(
                CompareRun(
                    model_name=name,
                    mode=mode,
                    spec=spec,
                    cache_hit=produced.cache_hit,
                )
            )

    logger.info("filling actuals and writing summary under %s", out)
    fill_actuals(data_dir, store)
    ledger = read_predictions(data_dir)
    model_ids = {run.spec.model_id for run in runs}
    scored = ledger.filter(pl.col("model_id").is_in(sorted(model_ids)))
    pool = filter_decision_pool(scored, store)
    logger.info(
        "decision pool: %s/%s scored rows (%s)",
        pool.height,
        scored.height,
        POOL_FILTER_DESCRIPTION,
    )
    id_meta = pl.DataFrame(
        {
            "model_id": [run.spec.model_id for run in runs],
            "model_name": [run.model_name for run in runs],
            "mode": [run.mode for run in runs],
        }
    )
    detail = _attach_meta(summarise(pool), id_meta)
    summary = _attach_meta(overall_summary(pool), id_meta)
    calibration = _attach_meta(calibration_tables(pool), id_meta)
    result = CompareResult(
        season=season,
        test_from_gw=test_from_gw,
        test_to_gw=test_to_gw,
        fit_gw=fit_at,
        seasons_back=seasons_back,
        modes=chosen_modes,
        runs=tuple(runs),
        summary=summary,
        detail=detail,
        output_dir=out,
        calibration=calibration,
        time_decay=time_decay,
        skipped_openfpl=skipped,
    )
    logger.info("writing %s", out)
    _write_outputs(result, pool_n=pool.height, full_n=scored.height)
    if skipped:
        logger.info(skipped)
    return result


def format_summary_table(summary: pl.DataFrame) -> str:
    """Render the overall summary as a plain-text table for the CLI.

    Column order puts RMSE (headline) before MAE (continuity).
    """
    if summary.is_empty():
        return "(no scored rows — check actuals / data_checked)"
    cols = [
        c
        for c in (
            "model_name",
            "mode",
            "horizon",
            "rmse",
            "mae",
            "spearman",
            "top10_precision",
            "brier_played",
            "brier_60",
            "brier_goal",
            "brier_assist",
            "brier_cs",
            "brier_haul",
            "log_played",
            "log_60",
            "log_goal",
            "log_assist",
            "log_cs",
            "log_haul",
            "n",
            "model_id",
        )
        if c in summary.columns
    ]
    return str(summary.select(cols).sort("mode", "model_name", "horizon"))


def _configured(
    spec: ModelSpec,
    *,
    refit: Literal["every_gw", "never"],
    seasons_back: int | None,
    time_decay: float | None = None,
    openfpl_model_dir: str | None,
) -> ModelSpec:
    updates: dict[str, object] = {"refit": refit}
    window_updates: dict[str, object] = {}
    if seasons_back is not None:
        window_updates["seasons_back"] = seasons_back
    if time_decay is not None:
        window_updates["time_decay"] = time_decay
    if window_updates:
        updates["train_window"] = spec.train_window.model_copy(update=window_updates)
    if openfpl_model_dir is not None:
        params = dict(spec.params)
        params["model_dir"] = openfpl_model_dir
        updates["params"] = params
    return spec.model_copy(update=updates)


def _resolve_modes(
    modes: list[CompareMode] | tuple[CompareMode, ...] | None,
) -> tuple[CompareMode, ...]:
    if modes is None:
        return DEFAULT_MODES
    if not modes:
        raise ValueError("modes must be non-empty")
    seen: list[CompareMode] = []
    for mode in modes:
        if mode not in _MODE_REFIT:
            raise ValueError(f"unknown compare mode {mode!r}")
        if mode not in seen:
            seen.append(mode)
    return tuple(seen)


def _validate_window(
    test_from_gw: int,
    test_to_gw: int,
    fit_gw: int,
    steps: int,
    seasons_back: int | None,
    time_decay: float | None = None,
) -> None:
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    if seasons_back is not None and seasons_back < 1:
        raise ValueError(f"seasons_back must be positive, got {seasons_back}")
    if time_decay is not None and time_decay < 0:
        raise ValueError(f"time_decay must be non-negative, got {time_decay}")
    if not 1 <= test_from_gw <= test_to_gw <= 38:
        raise ValueError(
            f"test gameweeks must be in 1..38, got {test_from_gw}..{test_to_gw}"
        )
    if not 1 <= fit_gw <= 38:
        raise ValueError(f"fit_gw must be in 1..38, got {fit_gw}")
    if fit_gw > test_from_gw:
        raise ValueError(
            f"fit_gw ({fit_gw}) must be at or before test_from_gw ({test_from_gw})"
        )


def _default_output_dir(
    data_dir: Path, season: str, test_from_gw: int, test_to_gw: int
) -> Path:
    stamp = f"{season}_gw{test_from_gw:02d}-{test_to_gw:02d}"
    return data_dir / "compare" / stamp


def _attach_meta(frame: pl.DataFrame, id_meta: pl.DataFrame) -> pl.DataFrame:
    if frame.is_empty():
        return frame
    return frame.join(id_meta, on="model_id", how="left")


def _write_outputs(
    result: CompareResult,
    *,
    pool_n: int | None = None,
    full_n: int | None = None,
) -> None:
    out = result.output_dir
    summary_path = out / "summary.parquet"
    detail_path = out / "detail.parquet"
    calibration_path = out / "calibration.parquet"
    json_path = out / "summary.json"
    md_path = out / "summary.md"
    result.summary.write_parquet(summary_path)
    result.detail.write_parquet(detail_path)
    calibration = (
        result.calibration if result.calibration is not None else pl.DataFrame()
    )
    calibration.write_parquet(calibration_path)
    payload = {
        "season": result.season,
        "test_from_gw": result.test_from_gw,
        "test_to_gw": result.test_to_gw,
        "fit_gw": result.fit_gw,
        "seasons_back": result.seasons_back,
        "time_decay": result.time_decay,
        "modes": list(result.modes),
        "models": [run.model_name for run in result.runs],
        "headline_metric": HEADLINE_METRIC,
        "player_pool": POOL_FILTER_DESCRIPTION,
        "player_pool_n": pool_n,
        "player_pool_full_n": full_n,
        "runs": [
            {
                "model_name": run.model_name,
                "mode": run.mode,
                "model_id": run.spec.model_id,
                "refit": run.spec.refit,
                "seasons_back": run.spec.train_window.seasons_back,
                "time_decay": run.spec.train_window.time_decay,
                "cache_hit": run.cache_hit,
            }
            for run in result.runs
        ],
        "summary": result.summary.to_dicts(),
        "calibration": calibration.to_dicts(),
        "skipped_openfpl": result.skipped_openfpl,
        "files": {
            "summary_parquet": str(summary_path),
            "detail_parquet": str(detail_path),
            "calibration_parquet": str(calibration_path),
            "summary_json": str(json_path),
            "summary_md": str(md_path),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(_format_summary_markdown(result, calibration), encoding="utf-8")


def _format_summary_markdown(result: CompareResult, calibration: pl.DataFrame) -> str:
    """Markdown companion to ``summary.json`` for compare / promote-eval."""
    lines = [
        f"# Compare {result.season} gw {result.test_from_gw}-{result.test_to_gw}",
        "",
        "Headline point metric is RMSE (MAE secondary). Event Brier / log "
        "scores cover played, 60+, goal, assist, clean sheet, and haul when "
        "the arm emits those probabilities.",
        "",
        "## Summary",
        "",
        "```",
        format_summary_table(result.summary),
        "```",
        "",
        "## Calibration (xPts decile and position)",
        "",
        format_calibration_markdown(calibration),
        "",
    ]
    return "\n".join(lines) + "\n"


__all__ = [
    "CORE_MODELS",
    "DEFAULT_MODES",
    "CompareMode",
    "CompareResult",
    "CompareRun",
    "compare_models",
    "format_summary_table",
]
