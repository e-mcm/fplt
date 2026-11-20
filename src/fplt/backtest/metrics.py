"""Metrics scanned from the prediction ledger.

These numbers are for the leaderboard. Pytest does not use them as a model
quality gate.
"""

from __future__ import annotations

from typing import cast

import polars as pl


def summarise(ledger: pl.DataFrame) -> pl.DataFrame:
    """MAE, RMSE, Spearman, top-10 precision and Brier scores by slice.

    Rows with a null ``actual_pts`` are skipped. Spearman correlates ranks
    within position and gameweek. Top-10 uses fewer than 10 players when the
    slice is smaller. Brier scores are null when the probability column is null.

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

    Spearman ranks within gameweek across all positions. Top-10 is over the
    pooled player set. Includes ``n``, the number of scored rows in the slice.

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
        k = min(10, group.height)
        predicted = group.sort("xpts", descending=True).head(k)
        pred_top = set(predicted["player_key"].to_list())
        actual_top = set(
            group.sort("actual_pts", descending=True).head(k)["player_key"].to_list()
        )
        haul = group.filter(pl.col("p_haul").is_not_null())
        played = group.filter(
            pl.col("p_60").is_not_null() & pl.col("actual_minutes").is_not_null()
        )
        error = (group["xpts"] - group["actual_pts"]).cast(pl.Float64)
        row: dict[str, object] = {
            "model_id": parsed["model_id"],
            "horizon": parsed["horizon"],
            "target_season": parsed["target_season"],
            "mae": _mean(error.abs()),
            "rmse": _mean(error.pow(2)) ** 0.5,
            "spearman": group.select(pl.corr("pred_rank", "actual_rank")).item(),
            "top10_precision": len(pred_top & actual_top) / k,
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
        "mae": pl.Float64(),
        "rmse": pl.Float64(),
        "spearman": pl.Float64(),
        "top10_precision": pl.Float64(),
        "brier_haul": pl.Float64(),
        "brier_60": pl.Float64(),
    }
    if include_position:
        # Keep column order stable with the historical summarise frame.
        schema = {
            "model_id": pl.Utf8(),
            "horizon": pl.Int64(),
            "position": pl.Utf8(),
            "target_season": pl.Utf8(),
            "mae": pl.Float64(),
            "rmse": pl.Float64(),
            "spearman": pl.Float64(),
            "top10_precision": pl.Float64(),
            "brier_haul": pl.Float64(),
            "brier_60": pl.Float64(),
        }
    if include_n:
        schema["n"] = pl.Int64()
    return pl.DataFrame(schema=schema)
