"""Run the official T−90m slate: curated build, predict, plan, pointers.

Implements the B12 caller contract on ``ProductionOps.official_slate``:
one predict per distinct ModelSpec (longest horizon), one plan per line,
then ``finalize_official_checkpoint``. Writes ``points_pmf`` /
``points_pmf_gw`` on official predict runs from ``sample()`` draws.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from fplt.automation.production import OfficialLine, ProductionOps
from fplt.backtest.predict_live import component_builder_for, fit_or_load_predictor
from fplt.backtest.produce_ledger import n_sims_of
from fplt.backtest.walk_forward import allowed_seasons, horizon_for, ledger_rows
from fplt.data.pit import PITData, PITView
from fplt.data.store import DataStore
from fplt.models.contracts import SampledPointsPredictor
from fplt.models.predictions import require_scored_predictions, write_predictions
from fplt.models.spec import load_model_specs
from fplt.optimise.decisions import PlanExistsError, resolve_squad_state, run_plan
from fplt.optimise.state import SquadState
from fplt.settings import Settings, get_settings
from fplt.store.builds import build_from_staged
from fplt.store.official import (
    LineRunPair,
    finalize_official_checkpoint,
    official_checkpoint_dir,
)
from fplt.store.points_pmf import points_pmf_from_draws
from fplt.store.push import maybe_push_official
from fplt.store.runs import write_plan_run, write_predict_run
from fplt.types import GWKey

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class OfficialCheckpointResult:
    """Outcome of one ``run_official_checkpoint`` call.

    Attributes:
        season: Season of the deadline.
        gw: Gameweek of the deadline.
        checkpoint: Checkpoint label (default ``T-90m``).
        pairs: Per-line predict/plan run ids written to pointers.
        checkpoint_dir: Local ``official/.../checkpoint=…`` directory.
        build_id: Curated build id attached to the predict runs.
    """

    season: str
    gw: int
    checkpoint: str
    pairs: tuple[LineRunPair, ...]
    checkpoint_dir: Path
    build_id: str


def run_official_checkpoint(
    data_dir: Path,
    ops: ProductionOps,
    key: GWKey,
    *,
    settings: Settings | None = None,
    chip_arm: str = "optimised",
) -> OfficialCheckpointResult:
    """Build curated snapshot, run the official slate, finalize, and push.

    Args:
        data_dir: Local data root with staged tables.
        ops: Production config (lines + predict groups).
        key: Deadline being finalized (upcoming GW).
        settings: Optional settings for bucket push.
        chip_arm: Chip policy for each plan (default live ``optimised``).

    Returns:
        Paths and run ids for the written checkpoint.

    Raises:
        ValueError: A predict model cannot ``sample`` for points PMFs.
        BuildExistsError: Propagated only when a conflicting build exists
            with different contents (same id is reused).
        OfficialExistsError: Pointers for this checkpoint already exist.
    """
    cfg = get_settings() if settings is None else settings
    predict_horizons, lines, checkpoint = ops.official_slate()
    build = build_from_staged(data_dir)
    logger.info(
        "official_slate: build_id=%s key=%s predicts=%s lines=%s",
        build.build_id,
        key,
        predict_horizons,
        [line.line for line in lines],
    )

    predict_ids: dict[str, str] = {}
    for spec_name, horizon in predict_horizons.items():
        predict_ids[spec_name] = _official_predict(
            data_dir,
            spec_name,
            key=key,
            horizon=horizon,
            checkpoint=checkpoint,
        )

    view = PITData(DataStore(data_dir)).view(key)
    squad = resolve_squad_state(view, event=key.gw)
    pairs: list[LineRunPair] = []
    for line in lines:
        plan_id = _official_plan(
            data_dir,
            line,
            key=key,
            parent_run_id=predict_ids[line.spec],
            checkpoint=checkpoint,
            squad=squad,
            chip_arm=chip_arm,
        )
        pairs.append(
            LineRunPair(
                line=line.line,
                role=line.role,
                squad_mode=line.squad_mode,
                predict_run_id=predict_ids[line.spec],
                plan_run_id=plan_id,
            )
        )

    finalize_official_checkpoint(
        data_dir,
        season=key.season,
        gw=key.gw,
        pairs=pairs,
        checkpoint=checkpoint,
    )
    cp_dir = official_checkpoint_dir(
        data_dir, season=key.season, gw=key.gw, checkpoint=checkpoint
    )
    maybe_push_official(
        data_dir,
        season=key.season,
        gw=key.gw,
        checkpoint=checkpoint,
        settings=cfg,
    )
    return OfficialCheckpointResult(
        season=key.season,
        gw=key.gw,
        checkpoint=checkpoint,
        pairs=tuple(pairs),
        checkpoint_dir=cp_dir,
        build_id=build.build_id,
    )


def _official_predict(
    data_dir: Path,
    model_name: str,
    *,
    key: GWKey,
    horizon: int,
    checkpoint: str,
) -> str:
    """Fit/load, predict, write official run with components and points PMFs."""
    specs = load_model_specs()
    if model_name not in specs:
        raise ValueError(f"unknown model spec {model_name!r}")
    spec = specs[model_name]
    store = DataStore(data_dir)
    pit = PITData(store)
    seasons = allowed_seasons(key, spec.train_window.seasons_back)
    view = pit.view(key).within_seasons(seasons)
    model = fit_or_load_predictor(data_dir, spec, key, view)
    predicted = model.predict(view, horizon_for(key, horizon))
    require_scored_predictions(predicted, model_name)
    try:
        write_predictions(data_dir, spec, key, predicted)
    except FileExistsError as exc:
        logger.info(
            "official_slate: legacy predictions exist for %s; continuing (%s)",
            key,
            exc,
        )
    rows = ledger_rows(predicted, view, key, "live")
    deadline_utc = store.deadline(key)
    n_sims = n_sims_of(spec) or 100
    pmf_builder = _pmf_builder(
        model,
        view,
        as_of=key,
        deadline_utc=deadline_utc,
        horizon=horizon,
        n_sims=n_sims,
    )
    run = write_predict_run(
        data_dir,
        spec=spec,
        as_of=key,
        deadline_utc=deadline_utc,
        horizon=horizon,
        frame=rows,
        kind="official",
        checkpoint=checkpoint,
        purpose=f"official {checkpoint} {spec.name} {key.season} gw{key.gw}",
        components_for_run_id=component_builder_for(
            model, view, as_of=key, deadline_utc=deadline_utc, horizon=horizon
        ),
        points_pmf_for_run_id=pmf_builder,
    )
    logger.info("official_slate: predict run_id=%s spec=%s", run.run_id, model_name)
    return run.run_id


def _official_plan(
    data_dir: Path,
    line: OfficialLine,
    *,
    key: GWKey,
    parent_run_id: str,
    checkpoint: str,
    squad: object,
    chip_arm: str,
) -> str:
    """Plan one official line; dual-write even when legacy plan already exists."""
    if not isinstance(squad, SquadState):
        raise TypeError(f"expected SquadState, got {type(squad).__name__}")
    try:
        _plan, _path, _report, run_id = run_plan(
            data_dir,
            line.spec,
            key.season,
            key.gw,
            line.horizon,
            squad=squad,
            chip_arm=chip_arm,  # type: ignore[arg-type]
            run_kind="official",
            checkpoint=checkpoint,
            line=line.line,
            squad_mode=line.squad_mode,
            parent_run_id=parent_run_id,
        )
        return run_id
    except PlanExistsError as exc:
        logger.info(
            "official_slate: legacy plan exists for %s h=%s; writing official run",
            key,
            line.horizon,
        )
        frame = pl.read_parquet(Path(str(exc)))
        spec = load_model_specs()[line.spec]
        store = DataStore(data_dir)
        run = write_plan_run(
            data_dir,
            spec=spec,
            as_of=key,
            deadline_utc=store.deadline(key),
            horizon=line.horizon,
            frame=frame,
            kind="official",
            checkpoint=checkpoint,
            line=line.line,
            squad_mode=line.squad_mode,
            parent_run_id=parent_run_id,
            purpose=(
                f"official {checkpoint} {line.line} {key.season} gw{key.gw} "
                f"h{line.horizon}"
            ),
        )
        return run.run_id


def _pmf_builder(
    model: object,
    view: PITView,
    *,
    as_of: GWKey,
    deadline_utc: datetime,
    horizon: int,
    n_sims: int,
):
    """Return ``run_id -> (points_pmf, points_pmf_gw)`` from MC ``sample()``."""

    def _build(run_id: str) -> tuple[pl.DataFrame, pl.DataFrame]:
        if not isinstance(model, SampledPointsPredictor):
            raise ValueError(
                "official predict requires SampledPointsPredictor.sample() "
                f"for points PMFs; got {type(model).__name__}"
            )
        keys = horizon_for(as_of, horizon)
        samples = model.sample(view, keys, n_sims)
        if samples.is_empty():
            raise ValueError(
                f"official predict sample() returned no rows for {as_of} "
                f"horizon={horizon}; cannot write points_pmf"
            )
        players: pl.DataFrame = (
            view.latest("players", by=["season", "player_key"])
            .select("player_key", "team_key")
            .unique(subset=["player_key"], keep="last")
            .collect()
        )
        draws = (
            samples.join(players, on="player_key", how="left")
            .with_columns(
                pl.col("sim").alias("draw_id"),
                (pl.col("target_gw") - as_of.gw).cast(pl.Int64).alias("horizon"),
            )
            .select(
                "player_key",
                "team_key",
                "fixture_id",
                "target_season",
                "target_gw",
                "horizon",
                "draw_id",
                "points",
            )
        )
        if draws["team_key"].null_count() > 0:
            raise ValueError("points_pmf draws missing team_key after player join")
        fx = points_pmf_from_draws(
            draws,
            run_id=run_id,
            as_of_season=as_of.season,
            as_of_gw=as_of.gw,
            deadline_utc=deadline_utc,
            grain="fixture",
        )
        gw = points_pmf_from_draws(
            draws,
            run_id=run_id,
            as_of_season=as_of.season,
            as_of_gw=as_of.gw,
            deadline_utc=deadline_utc,
            grain="gameweek",
        )
        return fx, gw

    return _build
