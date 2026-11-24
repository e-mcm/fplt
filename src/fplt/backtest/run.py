"""Run a historical backtest into the prediction ledger."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Literal

# Importing the package registers the predictors with get_model.
import fplt.models.baselines  # noqa: F401  # pyright: ignore[reportUnusedImport]
import fplt.models.composite  # noqa: F401  # pyright: ignore[reportUnusedImport]
import fplt.models.direct  # noqa: F401  # pyright: ignore[reportUnusedImport]
import fplt.models.ensemble  # noqa: F401  # pyright: ignore[reportUnusedImport]
import fplt.models.external.openfpl  # noqa: F401  # pyright: ignore[reportUnusedImport]
from fplt.backtest.ledger import append_predictions
from fplt.backtest.walk_forward import walk_forward
from fplt.data.outcomes import stage_outcomes
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.spec import ModelSpec, get_model, load_model_specs
from fplt.store.runs import write_backtest_run
from fplt.types import GWKey
from fplt.utils.timing import binds_timing_artifacts

Refit = Literal["every_gw", "every_n_gw", "never"]
logger = logging.getLogger(__name__)


@binds_timing_artifacts()
def run_backtest(
    data_dir: Path,
    season: str,
    from_gw: int,
    to_gw: int,
    models: list[str],
    *,
    refit: Refit | None = None,
    seasons_back: int | None = None,
    time_decay: float | None = None,
    steps: int = 1,
) -> list[ModelSpec]:
    """Fit and score ``models`` across one season, then stage checked outcomes.

    A copied spec is used when ``refit``, ``seasons_back``, or ``time_decay`` is
    set, so the ledger id differs from the YAML spec. ``refit="never"`` fits at
    the first requested deadline and reuses that fit. ``steps`` is the horizon
    length passed to ``walk_forward``; the library default of 5 is left unchanged.
    Actuals are read via ``Store.ledger()`` (predictions ⋈ outcomes), not by
    mutating the transitional ledger parquet.

    Args:
        data_dir: Root that contains ``staged/`` and receives ``ledger/``.
        season: Evaluation season, such as ``"2024-25"``.
        from_gw: First as-of gameweek.
        to_gw: Last as-of gameweek, inclusive.
        models: Spec names from ``configs/models``.
        refit: Override the spec's refit policy.
        seasons_back: Override how many seasons the fit may see.
        time_decay: Override calendar decay λ (inverse days) on
            ``TrainWindow.time_decay``. Trainers that honour it (LightGBM)
            weight rows by ``exp(-λ × days_before_as_of)``; Dixon–Coles keeps
            using ``xi`` and ignores this field.
        steps: Horizon length. Horizon 1 is the as-of gameweek.

    Returns:
        The spec used for each model, in request order.

    Raises:
        ValueError: The gameweek range, horizon, window, or a spec name is
            invalid.
        KeyError: A spec's kind is not a registered points predictor.
    """
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    if seasons_back is not None and seasons_back < 1:
        raise ValueError(f"seasons_back must be positive, got {seasons_back}")
    if time_decay is not None and time_decay < 0:
        raise ValueError(f"time_decay must be non-negative, got {time_decay}")
    if not 1 <= from_gw <= to_gw <= 38:
        raise ValueError(f"gameweeks must be in 1..38, got {from_gw}..{to_gw}")
    specs = load_model_specs()
    missing = [name for name in models if name not in specs]
    if missing:
        raise ValueError(f"unknown model spec {missing[0]!r}")
    store = DataStore(data_dir)
    pit = PITData(store)
    as_ofs = [GWKey(season, gw) for gw in range(from_gw, to_gw + 1)]
    logger.info(
        "backtest season=%s gw=%s..%s models=%s steps=%s refit=%s "
        "seasons_back=%s time_decay=%s",
        season,
        from_gw,
        to_gw,
        models,
        steps,
        refit,
        seasons_back,
        time_decay,
    )
    ran: list[ModelSpec] = []
    for name in models:
        spec = _configured(specs[name], refit, seasons_back, time_decay)
        logger.info(
            "fitting %s kind=%s refit=%s model_id=%s",
            spec.name,
            spec.kind,
            spec.refit,
            spec.model_id,
        )
        model = get_model(spec.kind)(spec)
        frame = walk_forward(model, pit, as_ofs, steps=steps)
        logger.info(
            "%s wrote %s prediction rows",
            spec.name,
            frame.height,
        )
        append_predictions(data_dir, frame)
        deadlines = [(key, store.deadline(key)) for key in as_ofs]
        run = write_backtest_run(
            data_dir,
            spec=spec,
            as_ofs=deadlines,
            horizon=steps,
            frame=frame,
        )
        logger.info("dual-wrote backtest run_id=%s model=%s", run.run_id, spec.name)
        ran.append(spec)
    logger.info("staging outcomes for checked gameweeks")
    stage_outcomes(store)
    return ran


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
