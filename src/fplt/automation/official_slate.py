"""Run the official T−90m slate: curated build, predict, plan, pointers.

Implements the B12 caller contract on ``ProductionOps.official_slate``:
one predict per distinct ModelSpec (longest horizon), one plan per line,
then ``finalize_official_checkpoint``. Writes ``points_pmf`` /
``points_pmf_gw`` on official predict runs from ``sample()`` draws.

#346: uses a pulled curated ``build_id`` (never ``build_from_staged`` /
never ``raw/``). Run manifests carry ``fp`` + ``build_id``; skip-if-done is
decided by the caller via ``runs/by_fp/``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from fplt.automation.ops_data import (
    LineFpDecision,
    decisions_for_slate,
    pull_required_tables,
    push_fp_index,
    record_line_run,
    tables_for_slate,
)
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
from fplt.schemas.manifests import BuildManifest
from fplt.settings import Settings, get_settings
from fplt.store.builds import load_build_manifest
from fplt.store.fingerprint import fit_fingerprint, last_completed_gw
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


def ensure_slate_runs(
    data_dir: Path,
    ops: ProductionOps,
    key: GWKey,
    *,
    settings: Settings | None = None,
    chip_arm: str = "optimised",
    build_id: str | None = None,
    prod_tag: str | None = None,
    decisions: list[LineFpDecision] | None = None,
    checkpoint: str | None = None,
    now: datetime | None = None,
) -> tuple[str, tuple[LineRunPair, ...]]:
    """Fit/predict/plan for the slate; write ``by_fp``; do not touch official/.

    Used by mid-week ticks (#346) so a fingerprint run exists before the
    T−90m pointer is written. Checkpoints call ``run_official_checkpoint``,
    which wraps this and finalizes pointers.

    Returns:
        ``(build_id, pairs)``.
    """
    cfg = get_settings() if settings is None else settings
    moment = datetime.now(UTC) if now is None else now
    predict_horizons, lines, _default_cp = ops.official_slate()
    # Mid-week ensure passes checkpoint=None; T−90m passes ``T-90m``.
    # Manifests accept None; helpers below take ``str | None`` via cast at call.
    cp_label: str | None = checkpoint
    resolved_build_id = _resolve_build_id(data_dir, build_id)
    build = load_build_manifest(data_dir, resolved_build_id)
    logger.info(
        "official_slate: build_id=%s key=%s predicts=%s lines=%s",
        resolved_build_id,
        key,
        predict_horizons,
        [line.line for line in lines],
    )

    slate_decisions = decisions
    if slate_decisions is None:
        slate_decisions = decisions_for_slate(
            ops,
            build,
            key,
            data_dir=data_dir,
            prod_tag=prod_tag,
        )
    by_line = {item.line: item for item in slate_decisions}

    pull_required_tables(
        data_dir,
        resolved_build_id,
        tables_for_slate(ops),
        settings=cfg,
    )

    predict_ids: dict[str, str] = {}
    for spec_name, horizon in predict_horizons.items():
        reused = _reuse_predict_id(spec_name, slate_decisions)
        if reused is not None:
            predict_ids[spec_name] = reused
            continue
        line_fp = next(
            item.predict_fp for item in slate_decisions if item.spec_name == spec_name
        )
        predict_ids[spec_name] = _official_predict(
            data_dir,
            spec_name,
            key=key,
            horizon=horizon,
            checkpoint=cp_label,
            build_id=resolved_build_id,
            fp=line_fp,
            prod_tag=prod_tag,
            build=build,
        )
        push_fp_index(data_dir, line_fp, settings=cfg)

    view = PITData(DataStore(data_dir)).view(key)
    squad = resolve_squad_state(view, event=key.gw)
    pairs: list[LineRunPair] = []
    for line in lines:
        decision = by_line[line.line]
        if decision.skip and decision.plan_run_id is not None:
            plan_id = decision.plan_run_id
        else:
            plan_id = _official_plan(
                data_dir,
                line,
                key=key,
                parent_run_id=predict_ids[line.spec],
                checkpoint=cp_label,
                squad=squad,
                chip_arm=chip_arm,
                build_id=resolved_build_id,
                fp=decision.plan_fp,
                prod_tag=prod_tag,
            )
            push_fp_index(data_dir, decision.plan_fp, settings=cfg)
            record_line_run(data_dir, line.line, now=moment)
        pairs.append(
            LineRunPair(
                line=line.line,
                role=line.role,
                squad_mode=line.squad_mode,
                predict_run_id=predict_ids[line.spec],
                plan_run_id=plan_id,
            )
        )
    return resolved_build_id, tuple(pairs)


def run_official_checkpoint(
    data_dir: Path,
    ops: ProductionOps,
    key: GWKey,
    *,
    settings: Settings | None = None,
    chip_arm: str = "optimised",
    build_id: str | None = None,
    prod_tag: str | None = None,
    decisions: list[LineFpDecision] | None = None,
    now: datetime | None = None,
) -> OfficialCheckpointResult:
    """Run the official slate against a curated build, finalize, and push.

    Args:
        data_dir: Local data root (staged tables hydrated from the build).
        ops: Production config (lines + predict groups).
        key: Deadline being finalized (upcoming GW).
        settings: Optional settings for bucket push.
        chip_arm: Chip policy for each plan (default live ``optimised``).
        build_id: Curated build id (#346). Required — callers pull/hydrate
            first. When omitted, falls back to ``status/curated/latest.json``.
        prod_tag: Moving ops tag for fingerprints / manifests.
        decisions: Optional precomputed skip-if-done decisions.
        now: Clock for throttle recording.

    Returns:
        Paths and run ids for the written checkpoint.

    Raises:
        ValueError: A predict model cannot ``sample`` for points PMFs, or
            ``build_id`` cannot be resolved.
        OfficialExistsError: Pointers for this checkpoint already exist.
        FileNotFoundError: Build manifest or required tables are missing.
    """
    cfg = get_settings() if settings is None else settings
    _, _, checkpoint = ops.official_slate()
    resolved_build_id, pairs = ensure_slate_runs(
        data_dir,
        ops,
        key,
        settings=cfg,
        chip_arm=chip_arm,
        build_id=build_id,
        prod_tag=prod_tag,
        decisions=decisions,
        checkpoint=checkpoint,
        now=now,
    )

    finalize_official_checkpoint(
        data_dir,
        season=key.season,
        gw=key.gw,
        pairs=list(pairs),
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
        pairs=pairs,
        checkpoint_dir=cp_dir,
        build_id=resolved_build_id,
    )


def _resolve_build_id(data_dir: Path, build_id: str | None) -> str:
    if build_id is not None:
        return build_id
    from fplt.automation.curate import load_curated_latest

    pointer = load_curated_latest(data_dir)
    if pointer is not None:
        return pointer.build_id
    raise ValueError(
        "run_official_checkpoint requires build_id (#346); "
        "status/curated/latest.json is also missing"
    )


def _reuse_predict_id(spec_name: str, decisions: list[LineFpDecision]) -> str | None:
    matching = [item for item in decisions if item.spec_name == spec_name]
    if not matching:
        return None
    if all(item.skip and item.predict_run_id for item in matching):
        return matching[0].predict_run_id
    return None


def _official_predict(
    data_dir: Path,
    model_name: str,
    *,
    key: GWKey,
    horizon: int,
    checkpoint: str | None,
    build_id: str,
    fp: str,
    prod_tag: str | None,
    build: BuildManifest,
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
    completed = last_completed_gw(store.read("gameweeks"))
    fit_fp = fit_fingerprint(spec=spec, build=build, last_completed=completed)
    model = fit_or_load_predictor(data_dir, spec, key, view, fit_fp=fit_fp)
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
        build_id=build_id,
        fp=fp,
        prod_tag=prod_tag,
    )
    logger.info(
        "official_slate: predict run_id=%s spec=%s fp=%s",
        run.run_id,
        model_name,
        fp[:12],
    )
    return run.run_id


def _official_plan(
    data_dir: Path,
    line: OfficialLine,
    *,
    key: GWKey,
    parent_run_id: str,
    checkpoint: str | None,
    squad: object,
    chip_arm: str,
    build_id: str,
    fp: str,
    prod_tag: str | None,
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
            build_id=build_id,
            fp=fp,
            prod_tag=prod_tag,
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
            build_id=build_id,
            fp=fp,
            prod_tag=prod_tag,
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
