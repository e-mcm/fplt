"""Metrics scanned from the prediction ledger.

These numbers are for the leaderboard. Pytest does not use them as a model
quality gate.

Mean-seeking boards (compare / promote-eval / bakeoff) treat **RMSE** as the
headline point metric and keep **MAE** as a secondary continuity metric. Callers
should filter to the decision-relevant pool (``filter_decision_pool``) before
summarising when matching the evaluation north star.
"""

from __future__ import annotations

from typing import cast

import numpy as np
import polars as pl

# Decision-hinge binary events: predicted prob column → outcome column.
# Haul stays P(points >= 10) to match ``p_haul`` / design.md.
_EVENT_SPECS: tuple[tuple[str, str, str], ...] = (
    ("played", "p_played", "played"),
    ("60", "p_60", "is_60"),
    ("goal", "p_goal", "scored"),
    ("assist", "p_assist", "assisted"),
    ("cs", "p_cs", "kept_cs"),
    ("haul", "p_haul", "hauled"),
)

_LOG_EPS = 1.0e-15


def summarise(ledger: pl.DataFrame) -> pl.DataFrame:
    """RMSE, MAE, Spearman, top-10 precision and event scores by slice.

    Rows with a null ``actual_pts`` are skipped. Spearman correlates ranks
    within position and gameweek. ``top10_precision`` is the mean of per-
    gameweek top-k precision (k = min(10, n_gw)), not a single pooled top-10.
    Brier / log scores are null when the probability column is missing or null.

    Args:
        ledger: A prediction ledger, usually ``read_predictions``.

    Returns:
        One row per model, horizon, position and target season.
    """
    return _summarise(
        ledger,
        keys=["model_id", "horizon", "position", "target_season"],
        rank_over=["model_id", "horizon", "position", "target_season", "target_gw"],
        include_position=True,
    )


def overall_summary(ledger: pl.DataFrame) -> pl.DataFrame:
    """Pool positions: one row per model, horizon and target season.

    Spearman ranks within gameweek across all positions. ``top10_precision`` is
    the mean of per-gameweek top-k precision over the pooled player set.
    Includes ``n``, the number of scored rows in the slice.

    Args:
        ledger: A prediction ledger, usually ``read_predictions``.

    Returns:
        Overall metric rows suitable for ``fplt compare`` / Streamlit tables.
    """
    return _summarise(
        ledger,
        keys=["model_id", "horizon", "target_season"],
        rank_over=["model_id", "horizon", "target_season", "target_gw"],
        include_position=False,
        include_n=True,
    )


def calibration_tables(ledger: pl.DataFrame, *, n_deciles: int = 10) -> pl.DataFrame:
    """Reliability tables by xPts decile and by position for event probs.

    Each row is one model × horizon × season × event × slice. ``mean_p`` is
    the mean predicted probability; ``mean_y`` is the realised rate. Brier is
    the mean squared error inside the slice. Decile cuts use ``xpts`` ranks
    within the model/horizon/season pool (not predicted-prob bins).

    Args:
        ledger: Scored prediction ledger rows.
        n_deciles: Number of xPts bins (default 10).

    Returns:
        Calibration rows, empty when no scored rows or event probs exist.
    """
    scored = _with_outcomes(ledger.filter(pl.col("actual_pts").is_not_null()))
    if scored.is_empty():
        return _empty_calibration()
    parts = scored.partition_by(["model_id", "horizon", "target_season"], as_dict=True)
    rows: list[dict[str, object]] = []
    for key, group in parts.items():
        parsed = _parse_key(key, include_position=False)
        for event, p_col, y_col in _EVENT_SPECS:
            if p_col not in group.columns or y_col not in group.columns:
                continue
            usable = group.filter(
                pl.col(p_col).is_not_null() & pl.col(y_col).is_not_null()
            )
            if usable.is_empty():
                continue
            rows.extend(
                _calibration_grouped_rows(
                    usable,
                    parsed=parsed,
                    event=event,
                    p_col=p_col,
                    y_col=y_col,
                    slice_kind="position",
                    slice_col="position",
                )
            )
            rows.extend(
                _calibration_decile_rows(
                    usable,
                    parsed=parsed,
                    event=event,
                    p_col=p_col,
                    y_col=y_col,
                    n_deciles=n_deciles,
                )
            )
    if not rows:
        return _empty_calibration()
    return pl.DataFrame(rows)


def format_calibration_markdown(calibration: pl.DataFrame, *, limit: int = 40) -> str:
    """Render a short markdown table of calibration rows for reports."""
    if calibration.is_empty():
        return "_No calibration rows (missing event probabilities or actuals)._"
    cols = [
        c
        for c in (
            "model_id",
            "horizon",
            "event",
            "slice_kind",
            "slice_value",
            "n",
            "mean_p",
            "mean_y",
            "brier",
        )
        if c in calibration.columns
    ]
    head = calibration.select(cols).head(limit)
    return "```\n" + str(head) + "\n```"


def _summarise(
    ledger: pl.DataFrame,
    *,
    keys: list[str],
    rank_over: list[str],
    include_position: bool,
    include_n: bool = False,
) -> pl.DataFrame:
    scored = _with_outcomes(ledger.filter(pl.col("actual_pts").is_not_null()))
    if scored.is_empty():
        return _empty_summary(include_position=include_position, include_n=include_n)
    ranked = scored.with_columns(
        pl.col("xpts").rank().over(rank_over).alias("pred_rank"),
        pl.col("actual_pts").rank().over(rank_over).alias("actual_rank"),
    )
    parts = ranked.partition_by(keys, as_dict=True)
    rows: list[dict[str, object]] = []
    for key, group in parts.items():
        parsed = _parse_key(key, include_position=include_position)
        error = (group["xpts"] - group["actual_pts"]).cast(pl.Float64)
        row: dict[str, object] = {
            "model_id": parsed["model_id"],
            "horizon": parsed["horizon"],
            "target_season": parsed["target_season"],
            "rmse": _mean(error.pow(2)) ** 0.5,
            "mae": _mean(error.abs()),
            "spearman": group.select(pl.corr("pred_rank", "actual_rank")).item(),
            "top10_precision": _mean_topk_precision(group),
        }
        for event, p_col, y_col in _EVENT_SPECS:
            brier_key = f"brier_{event}"
            log_key = f"log_{event}"
            if p_col not in group.columns or y_col not in group.columns:
                row[brier_key] = None
                row[log_key] = None
                continue
            usable = group.filter(
                pl.col(p_col).is_not_null() & pl.col(y_col).is_not_null()
            )
            if usable.is_empty():
                row[brier_key] = None
                row[log_key] = None
                continue
            probs = usable[p_col].cast(pl.Float64)
            outcomes = usable[y_col].cast(pl.Float64)
            row[brier_key] = _brier(probs, outcomes)
            row[log_key] = _log_score(probs, outcomes)
        if include_position:
            row["position"] = parsed["position"]
        if include_n:
            row["n"] = group.height
        rows.append(row)
    return pl.DataFrame(rows)


def _with_outcomes(frame: pl.DataFrame) -> pl.DataFrame:
    """Attach binary outcome columns used by event scores."""
    if frame.is_empty():
        return frame
    extras: list[pl.Expr] = []
    if "actual_minutes" in frame.columns:
        extras.extend(
            [
                (pl.col("actual_minutes") > 0).cast(pl.Float64).alias("played"),
                (pl.col("actual_minutes") >= 60).cast(pl.Float64).alias("is_60"),
            ]
        )
    else:
        extras.extend(
            [
                pl.lit(None).cast(pl.Float64).alias("played"),
                pl.lit(None).cast(pl.Float64).alias("is_60"),
            ]
        )
    if "actual_goals" in frame.columns:
        extras.append((pl.col("actual_goals") >= 1).cast(pl.Float64).alias("scored"))
    else:
        extras.append(pl.lit(None).cast(pl.Float64).alias("scored"))
    if "actual_assists" in frame.columns:
        extras.append(
            (pl.col("actual_assists") >= 1).cast(pl.Float64).alias("assisted")
        )
    else:
        extras.append(pl.lit(None).cast(pl.Float64).alias("assisted"))
    if "actual_cs" in frame.columns:
        extras.append((pl.col("actual_cs") >= 1).cast(pl.Float64).alias("kept_cs"))
    else:
        extras.append(pl.lit(None).cast(pl.Float64).alias("kept_cs"))
    extras.append((pl.col("actual_pts") >= 10).cast(pl.Float64).alias("hauled"))
    return frame.with_columns(extras)


def _mean_topk_precision(group: pl.DataFrame, *, k: int = 10) -> float:
    """Average top-k precision across target gameweeks in ``group``."""
    if group.is_empty():
        return 0.0
    precisions: list[float] = []
    for gw_frame in group.partition_by("target_gw", as_dict=True).values():
        n = gw_frame.height
        if n == 0:
            continue
        top = min(k, n)
        pred_top = set(
            gw_frame.sort("xpts", descending=True).head(top)["player_key"].to_list()
        )
        actual_top = set(
            gw_frame.sort("actual_pts", descending=True)
            .head(top)["player_key"]
            .to_list()
        )
        precisions.append(len(pred_top & actual_top) / top)
    if not precisions:
        return 0.0
    return float(sum(precisions) / len(precisions))


def _brier(probs: pl.Series, outcomes: pl.Series) -> float:
    residual = probs.cast(pl.Float64) - outcomes.cast(pl.Float64)
    return _mean(residual.pow(2))


def _log_score(probs: pl.Series, outcomes: pl.Series) -> float:
    """Mean Bernoulli log loss; lower is better (same orientation as Brier)."""
    p = np.clip(probs.to_numpy().astype(np.float64), _LOG_EPS, 1.0 - _LOG_EPS)
    y = outcomes.to_numpy().astype(np.float64)
    return float(-np.mean(y * np.log(p) + (1.0 - y) * np.log(1.0 - p)))


def _calibration_grouped_rows(
    frame: pl.DataFrame,
    *,
    parsed: dict[str, object],
    event: str,
    p_col: str,
    y_col: str,
    slice_kind: str,
    slice_col: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for value, part in frame.partition_by(slice_col, as_dict=True).items():
        rows.append(
            _calibration_row(
                part,
                parsed=parsed,
                event=event,
                p_col=p_col,
                y_col=y_col,
                slice_kind=slice_kind,
                slice_value=str(value[0]),
            )
        )
    return rows


def _calibration_decile_rows(
    frame: pl.DataFrame,
    *,
    parsed: dict[str, object],
    event: str,
    p_col: str,
    y_col: str,
    n_deciles: int,
) -> list[dict[str, object]]:
    if frame.height < n_deciles or n_deciles < 1:
        return [
            _calibration_row(
                frame,
                parsed=parsed,
                event=event,
                p_col=p_col,
                y_col=y_col,
                slice_kind="xpts_decile",
                slice_value="all",
            )
        ]
    height = float(frame.height)
    # Rank-based deciles stay stable when xPts ties collapse qcut bins.
    labeled = frame.with_columns(
        (
            ((pl.col("xpts").rank(method="average") - 1.0) * n_deciles / height)
            .floor()
            .clip(0, n_deciles - 1)
            .cast(pl.Int64)
            + 1
        )
        .cast(pl.Utf8)
        .alias("_decile")
    )
    return _calibration_grouped_rows(
        labeled,
        parsed=parsed,
        event=event,
        p_col=p_col,
        y_col=y_col,
        slice_kind="xpts_decile",
        slice_col="_decile",
    )


def _calibration_row(
    frame: pl.DataFrame,
    *,
    parsed: dict[str, object],
    event: str,
    p_col: str,
    y_col: str,
    slice_kind: str,
    slice_value: str,
) -> dict[str, object]:
    probs = frame[p_col].cast(pl.Float64)
    outcomes = frame[y_col].cast(pl.Float64)
    return {
        "model_id": parsed["model_id"],
        "horizon": parsed["horizon"],
        "target_season": parsed["target_season"],
        "event": event,
        "slice_kind": slice_kind,
        "slice_value": slice_value,
        "n": frame.height,
        "mean_p": _mean(probs),
        "mean_y": _mean(outcomes),
        "brier": _brier(probs, outcomes),
    }


def _mean(series: pl.Series) -> float:
    value = series.mean()
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)


def _parse_key(key: object, *, include_position: bool) -> dict[str, object]:
    if not isinstance(key, tuple):
        raise TypeError(f"unexpected partition key {key!r}")
    if include_position:
        model_id, horizon, position, season = cast(
            tuple[object, object, object, object], key
        )
        return {
            "model_id": str(model_id),
            "horizon": _as_int(horizon),
            "position": str(position),
            "target_season": str(season),
        }
    model_id, horizon, season = cast(tuple[object, object, object], key)
    return {
        "model_id": str(model_id),
        "horizon": _as_int(horizon),
        "target_season": str(season),
    }


def _as_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise TypeError(f"expected a number, got {value!r}")
    return int(value)


def _event_metric_names() -> list[str]:
    names: list[str] = []
    for event, _, _ in _EVENT_SPECS:
        names.append(f"brier_{event}")
        names.append(f"log_{event}")
    return names


def _empty_summary(
    *, include_position: bool = True, include_n: bool = False
) -> pl.DataFrame:
    schema: dict[str, pl.DataType] = {
        "model_id": pl.Utf8(),
        "horizon": pl.Int64(),
        "target_season": pl.Utf8(),
        "rmse": pl.Float64(),
        "mae": pl.Float64(),
        "spearman": pl.Float64(),
        "top10_precision": pl.Float64(),
    }
    for name in _event_metric_names():
        schema[name] = pl.Float64()
    if include_position:
        # Keep column order stable: identity → RMSE headline → MAE → rest.
        ordered: dict[str, pl.DataType] = {
            "model_id": pl.Utf8(),
            "horizon": pl.Int64(),
            "position": pl.Utf8(),
            "target_season": pl.Utf8(),
            "rmse": pl.Float64(),
            "mae": pl.Float64(),
            "spearman": pl.Float64(),
            "top10_precision": pl.Float64(),
        }
        for name in _event_metric_names():
            ordered[name] = pl.Float64()
        schema = ordered
    if include_n:
        schema["n"] = pl.Int64()
    return pl.DataFrame(schema=schema)


def _empty_calibration() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "model_id": pl.Utf8(),
            "horizon": pl.Int64(),
            "target_season": pl.Utf8(),
            "event": pl.Utf8(),
            "slice_kind": pl.Utf8(),
            "slice_value": pl.Utf8(),
            "n": pl.Int64(),
            "mean_p": pl.Float64(),
            "mean_y": pl.Float64(),
            "brier": pl.Float64(),
        }
    )


__all__ = [
    "calibration_tables",
    "format_calibration_markdown",
    "overall_summary",
    "summarise",
]
