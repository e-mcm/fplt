"""Walk-forward fitting and prediction.

Training rows are whatever the view shows at the deadline, limited to the
spec's ``train_window.seasons_back``. Actuals are filled later by the ledger.
"""

from __future__ import annotations

import polars as pl

from fplt.data.pit import PITData, PITView
from fplt.models.contracts import PointsPredictor
from fplt.models.spec import ModelSpec
from fplt.schemas.ledger import LEDGER_COLUMNS, LedgerSchema
from fplt.types import GWKey


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
    steps: int = 5,
    run_kind: str = "backtest",
) -> pl.DataFrame:
    """Fit and predict at each as-of deadline.

    Args:
        model: Predictor whose ``spec`` supplies the refit policy and window.
        pit: Source of point-in-time views.
        as_ofs: Deadlines to evaluate, in order.
        steps: Horizon length. The default is 5.
        run_kind: ``backtest`` or ``live``.

    Returns:
        A ``LedgerSchema`` frame. ``actual_pts`` and ``actual_minutes`` are null.
    """
    frames: list[pl.DataFrame] = []
    for index, as_of in enumerate(as_ofs):
        seasons = allowed_seasons(as_of, model.spec.train_window.seasons_back)
        view = pit.view(as_of).within_seasons(seasons)
        if should_refit(model.spec, index):
            model.fit(view)
        predicted = model.predict(view, horizon_for(as_of, steps))
        recorded = ledger_rows(predicted, view, as_of, run_kind)
        if not recorded.is_empty():
            frames.append(recorded)
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
        )
        .select(list(LEDGER_COLUMNS))
    )
    return frame


def _season_label(start_year: int) -> str:
    return f"{start_year}-{(start_year + 1) % 100:02d}"
