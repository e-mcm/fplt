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

import polars as pl


def summarise(ledger: pl.DataFrame) -> pl.DataFrame:
    """RMSE, MAE, Spearman, top-10 precision and Brier scores by slice.

    Rows with a null ``actual_pts`` are skipped. Spearman correlates ranks
    within position and gameweek. ``top10_precision`` is the mean of per-
    gameweek top-k precision (k = min(10, n_gw)), not a single pooled top-10.
    Brier scores are null when the probability column is null.

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


def _summarise(
    ledger: pl.DataFrame,
    *,
    keys: list[str],
    rank_over: list[str],
    include_position: bool,
    include_n: bool = False,
) -> pl.DataFrame:
    scored = ledger.filter(pl.col("actual_pts").is_not_null())
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
        haul = group.filter(pl.col("p_haul").is_not_null())
        played = group.filter(
            pl.col("p_60").is_not_null() & pl.col("actual_minutes").is_not_null()
        )
        row: dict[str, object] = {
            "model_id": parsed["model_id"],
            "horizon": parsed["horizon"],
            "target_season": parsed["target_season"],
            "rmse": _mean(error.pow(2)) ** 0.5,
            "mae": _mean(error.abs()),
            "spearman": group.select(pl.corr("pred_rank", "actual_rank")).item(),
            "top10_precision": _mean_topk_precision(group),
            "brier_haul": _brier(haul, "p_haul", haul["actual_pts"] >= 10)
            if haul.height
            else None,
            "brier_60": _brier(played, "p_60", played["actual_minutes"] >= 60)
            if played.height
            else None,
        }
        if include_position:
            row["position"] = parsed["position"]
        if include_n:
            row["n"] = group.height
        rows.append(row)
    return pl.DataFrame(rows)


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


def _brier(frame: pl.DataFrame, column: str, outcome: pl.Series) -> float:
    residual = frame[column] - outcome.cast(pl.Float64)
    return _mean(residual.cast(pl.Float64).pow(2))


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
        "brier_haul": pl.Float64(),
        "brier_60": pl.Float64(),
    }
    if include_position:
        # Keep column order stable: identity → RMSE headline → MAE → rest.
        schema = {
            "model_id": pl.Utf8(),
            "horizon": pl.Int64(),
            "position": pl.Utf8(),
            "target_season": pl.Utf8(),
            "rmse": pl.Float64(),
            "mae": pl.Float64(),
            "spearman": pl.Float64(),
            "top10_precision": pl.Float64(),
            "brier_haul": pl.Float64(),
            "brier_60": pl.Float64(),
        }
    if include_n:
        schema["n"] = pl.Int64()
    return pl.DataFrame(schema=schema)


__all__ = ["overall_summary", "summarise"]
