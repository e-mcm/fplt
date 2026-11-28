"""Validate player-fixture predictions."""

from __future__ import annotations

import polars as pl

from fplt.models.spec import ModelSpec
from fplt.schemas.predictions import PREDICTION_COLUMNS, PredictionSchema
from fplt.types import GWKey
from fplt.utils.frames import require_dataframe


def empty_predictions() -> pl.DataFrame:
    """An empty frame that still matches ``PredictionSchema``."""
    return pl.DataFrame(schema=PREDICTION_COLUMNS)


def require_scored_predictions(frame: pl.DataFrame, model_name: str) -> None:
    """Reject a non-empty prediction frame whose every ``xpts`` is zero.

    An empty frame is a horizon with no fixtures and is left alone. Walk-forward
    backtests still record a cold start; this guard is for live ``fplt predict``.

    Args:
        frame: A ``PredictionSchema`` frame.
        model_name: Spec name used in the error.

    Raises:
        ValueError: ``frame`` has rows and every ``xpts`` is 0.
    """
    if frame.is_empty():
        return
    peak = frame.select(pl.col("xpts").abs().max()).item()
    if peak is None or float(peak) == 0.0:
        raise ValueError(
            f"{model_name} predicted 0 for every player; refusing to record that"
        )


def prediction_frame(
    spec: ModelSpec,
    as_of: GWKey,
    rows: pl.DataFrame,
    xpts: pl.Expr,
    xpts_sd: pl.Expr,
    p_haul: pl.Expr,
    p_60: pl.Expr,
    *,
    p_played: pl.Expr | None = None,
    p_goal: pl.Expr | None = None,
    p_assist: pl.Expr | None = None,
    p_cs: pl.Expr | None = None,
) -> pl.DataFrame:
    """Attach prediction columns and validate them.

    Event probabilities default to null when the caller omits them. Layered
    Monte Carlo models fill them cheaply from the same sims that produce
    ``p_haul`` / ``p_60``; point-estimate baselines leave them null.

    Args:
        spec: Model that produced the rows.
        as_of: Deadline the prediction was made at.
        rows: Player-fixture rows with ``season``, ``gw``, ``player_key``,
            and ``fixture_id``.
        xpts: Expression for expected points.
        xpts_sd: Expression for the standard deviation, or null.
        p_haul: Expression for ``P(points >= 10)``, or null.
        p_60: Expression for ``P(minutes >= 60)``, or null.
        p_played: Expression for ``P(minutes > 0)``, or null.
        p_goal: Expression for ``P(goals >= 1)``, or null.
        p_assist: Expression for ``P(assists >= 1)``, or null.
        p_cs: Expression for ``P(clean sheet)``, or null.

    Returns:
        A frame matching ``PredictionSchema``.
    """
    if rows.is_empty():
        return empty_predictions()
    null = pl.lit(None)
    frame = (
        rows.with_columns(
            pl.lit(spec.model_id).alias("model_id"),
            pl.lit(as_of.season).alias("as_of_season"),
            pl.lit(as_of.gw).alias("as_of_gw"),
            pl.col("season").alias("target_season"),
            pl.col("gw").alias("target_gw"),
            xpts.alias("xpts"),
            xpts_sd.alias("xpts_sd"),
            (p_played if p_played is not None else null).alias("p_played"),
            p_60.alias("p_60"),
            (p_goal if p_goal is not None else null).alias("p_goal"),
            (p_assist if p_assist is not None else null).alias("p_assist"),
            (p_cs if p_cs is not None else null).alias("p_cs"),
            p_haul.alias("p_haul"),
        )
        .with_columns(
            pl.col("xpts").cast(pl.Float64),
            pl.col("xpts_sd").cast(pl.Float64),
            pl.col("p_played").cast(pl.Float64),
            pl.col("p_60").cast(pl.Float64),
            pl.col("p_goal").cast(pl.Float64),
            pl.col("p_assist").cast(pl.Float64),
            pl.col("p_cs").cast(pl.Float64),
            pl.col("p_haul").cast(pl.Float64),
        )
        .select(list(PREDICTION_COLUMNS))
    )
    checked = PredictionSchema.validate(frame)
    checked = require_dataframe(checked)
    return checked
