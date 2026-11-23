"""Walk-forward fitting and prediction.

Training rows are whatever the view shows at the deadline, limited to the
spec's ``train_window.seasons_back``. Actuals are filled later by the ledger.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path

import polars as pl

from fplt.backtest.horizons import DEFAULT_PROJECTION_HORIZON
from fplt.data.pit import PITData, PITView
from fplt.models.contracts import PointsPredictor
from fplt.models.spec import ModelSpec
from fplt.schemas.ledger import LEDGER_COLUMNS, LedgerSchema
from fplt.types import GWKey
from fplt.utils.timing import (
    TimingSession,
    format_gw_breakdown,
    format_run_summary,
    timed_phase,
    timing_session,
    write_timing_record,
)

logger = logging.getLogger(__name__)


def horizon_for(as_of: GWKey, steps: int) -> list[GWKey]:
    """The next ``steps`` gameweeks, starting at ``as_of`` itself.

    The list stops at gameweek 38. Horizon 1 is ``as_of``.
    """
    keys: list[GWKey] = []
    for offset in range(steps):
        gw = as_of.gw + offset
        if gw > 38:
            break
        keys.append(GWKey(as_of.season, gw))
    return keys


def allowed_seasons(as_of: GWKey, seasons_back: int) -> set[str]:
    """Season labels the training window may see, including ``as_of``."""
    start = int(as_of.season.split("-")[0])
    count = max(seasons_back, 1)
    return {_season_label(start - offset) for offset in range(count)}


def should_refit(spec: ModelSpec, index: int) -> bool:
    """Whether step ``index`` refits, following the spec's policy.

    ``every_gw`` refits every step. ``every_n_gw`` refits when ``index`` is a
    multiple of ``refit_every``. ``never`` refits only on the first step.
    """
    if spec.refit == "never":
        return index == 0
    if spec.refit == "every_n_gw":
        return index % max(spec.refit_every, 1) == 0
    return True


def walk_forward(
    model: PointsPredictor,
    pit: PITData,
    as_ofs: list[GWKey],
    steps: int = DEFAULT_PROJECTION_HORIZON,
    run_kind: str = "backtest",
    timing_dir: Path | None = None,
) -> pl.DataFrame:
    """Fit and predict at each as-of deadline.

    Logs per-gameweek wall times split into setup / fit / predict / write, and
    an end-of-run summary with totals and shares. Nested model phases (when
    recorded via ``timed_phase``) appear after ``|`` on the summary line.
    When ``timing_dir`` is set, also writes a JSON timing artifact under
    ``timing_dir/timing/`` for ``fplt perf-report``.

    Args:
        model: Predictor whose ``spec`` supplies the refit policy and window.
        pit: Source of point-in-time views.
        as_ofs: Deadlines to evaluate, in order.
        steps: Horizon length. Default is ``DEFAULT_PROJECTION_HORIZON`` (10)
            so planning can request H=8–10 without re-predicting; CLI
            backtest/compare still pass an explicit ``steps``.
        run_kind: ``backtest`` or ``live``.
        timing_dir: Optional data root that receives ``timing/*.json``.

    Returns:
        A ``LedgerSchema`` frame. ``actual_pts`` and ``actual_minutes`` are null.
    """
    logger.info(
        "walk_forward model=%s as_ofs=%s steps=%s run_kind=%s",
        model.spec.name,
        len(as_ofs),
        steps,
        run_kind,
    )
    frames: list[pl.DataFrame] = []
    total = len(as_ofs)
    run_timings = TimingSession()
    for index, as_of in enumerate(as_ofs):
        refitting = should_refit(model.spec, index)
        logger.info(
            "%s %s gw %s (%s/%s) %s",
            model.spec.name,
            as_of.season,
            as_of.gw,
            index + 1,
            total,
            "fitting" if refitting else "predicting",
        )
        started = time.perf_counter()
        gw_timings = TimingSession()
        with timing_session(gw_timings):
            with timed_phase("setup"):
                seasons = allowed_seasons(as_of, model.spec.train_window.seasons_back)
                view = pit.view(as_of).within_seasons(seasons)
            if refitting:
                logger.debug("fit at %s (index=%s)", as_of, index)
                with timed_phase("fit"):
                    model.fit(view)
            else:
                logger.debug("reuse fit at %s (index=%s)", as_of, index)
            with timed_phase("predict"):
                predicted = model.predict(view, horizon_for(as_of, steps))
            with timed_phase("write"):
                recorded = ledger_rows(predicted, view, as_of, run_kind)
        run_timings.merge(gw_timings)
        logger.info(
            "%s %s gw %s done in %.1fs (%s), %s rows",
            model.spec.name,
            as_of.season,
            as_of.gw,
            time.perf_counter() - started,
            format_gw_breakdown(gw_timings),
            recorded.height,
        )
        if not recorded.is_empty():
            frames.append(recorded)
    logger.info(format_run_summary(model.spec.name, run_timings, n_gw=total))
    if timing_dir is not None:
        season = as_ofs[0].season if as_ofs else None
        write_timing_record(
            timing_dir,
            "walk_forward",
            run_timings,
            meta={
                "model": model.spec.name,
                "model_id": model.spec.model_id,
                "n_gw": total,
                "steps": steps,
                "run_kind": run_kind,
                "season": season,
            },
        )
    combined = (
        pl.DataFrame(schema=LEDGER_COLUMNS)
        if not frames
        else pl.concat(frames, how="vertical")
    )
    checked = LedgerSchema.validate(combined)
    assert isinstance(checked, pl.DataFrame)
    return checked


def ledger_rows(
    predicted: pl.DataFrame, view: PITView, as_of: GWKey, run_kind: str
) -> pl.DataFrame:
    """Attach ledger columns to a ``PredictionSchema`` frame.

    Args:
        predicted: Rows from ``PointsPredictor.predict``.
        view: Point-in-time view used for the prediction.
        as_of: Deadline the prediction was made at.
        run_kind: ``backtest`` or ``live``.

    Returns:
        A ``LedgerSchema`` frame with null actuals. Empty when ``predicted`` is.
    """
    if predicted.is_empty():
        return pl.DataFrame(schema=LEDGER_COLUMNS)
    players = (
        view.latest("players", by=["season", "player_key"])
        .select(
            pl.col("season").alias("target_season"),
            "player_key",
            "position",
        )
        .collect()
    )
    frame = (
        predicted.join(players, on=["target_season", "player_key"], how="left")
        .with_columns(
            pl.lit(run_kind).alias("run_kind"),
            (pl.col("target_gw") - as_of.gw + 1).alias("horizon"),
            pl.col("position").fill_null("UNK"),
            pl.lit(None).cast(pl.Int64).alias("actual_pts"),
            pl.lit(None).cast(pl.Int64).alias("actual_minutes"),
            pl.lit(None).cast(pl.Int64).alias("actual_goals"),
            pl.lit(None).cast(pl.Int64).alias("actual_assists"),
            pl.lit(None).cast(pl.Int64).alias("actual_cs"),
        )
        .select(list(LEDGER_COLUMNS))
    )
    return frame


def _season_label(start_year: int) -> str:
    return f"{start_year}-{(start_year + 1) % 100:02d}"
