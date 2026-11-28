"""Proper scores for discrete points PMFs: RPS, log score, MSE of the mean.

Joins outcomes only on ``(player_key, fixture_id)`` (PIT-safe). Bins match
official ``points_pmf`` (−5…30+). Shared by error attribution and #395.
"""

from __future__ import annotations

import numpy as np
import polars as pl
from numpy.typing import NDArray

from fplt.store.points_pmf import POINTS_PMF_MAX, POINTS_PMF_MIN

LOG_SCORE_FLOOR = 1e-15
_WIDTH = POINTS_PMF_MAX - POINTS_PMF_MIN + 1
FloatArray = NDArray[np.floating]


def _clip(y: int) -> int:
    return int(min(POINTS_PMF_MAX, max(POINTS_PMF_MIN, y)))


def _dense_pmf(bins: NDArray[np.integer], probs: FloatArray) -> FloatArray:
    dense = np.zeros(_WIDTH, dtype=np.float64)
    for bin_k, prob in zip(bins.tolist(), probs.tolist(), strict=True):
        idx = int(bin_k) - POINTS_PMF_MIN
        if 0 <= idx < _WIDTH:
            dense[idx] += float(prob)
    total = float(dense.sum())
    if total > 0.0:
        dense /= total
    return dense


def discrete_rps(pmf: FloatArray, y: int) -> float:
    """Ranked probability score Σₖ (F̂(k) − 𝟙{y ≤ k})² on points bins."""
    dense = np.asarray(pmf, dtype=np.float64).reshape(-1)
    if dense.shape != (_WIDTH,):
        raise ValueError(f"pmf must have length {_WIDTH}, got {dense.shape}")
    cdf = np.cumsum(dense)
    observed = (np.arange(POINTS_PMF_MIN, POINTS_PMF_MAX + 1) >= _clip(y)).astype(
        np.float64
    )
    return float(np.sum((cdf - observed) ** 2))


def log_score_pmf(pmf: FloatArray, y: int, *, floor: float = LOG_SCORE_FLOOR) -> float:
    """``-log(max(p̂(y), floor))`` on the official points bins."""
    dense = np.asarray(pmf, dtype=np.float64).reshape(-1)
    if dense.shape != (_WIDTH,):
        raise ValueError(f"pmf must have length {_WIDTH}, got {dense.shape}")
    return float(-np.log(max(float(dense[_clip(y) - POINTS_PMF_MIN]), floor)))


def mse_mean_pmf(pmf: FloatArray, y: int) -> float:
    """``(μ̂ − y)²`` with ``μ̂ = Σ k · p̂(k)``."""
    dense = np.asarray(pmf, dtype=np.float64).reshape(-1)
    if dense.shape != (_WIDTH,):
        raise ValueError(f"pmf must have length {_WIDTH}, got {dense.shape}")
    bins = np.arange(POINTS_PMF_MIN, POINTS_PMF_MAX + 1, dtype=np.float64)
    return float((float(np.dot(bins, dense)) - float(y)) ** 2)


def score_points_pmf(pmf: pl.DataFrame, outcomes: pl.DataFrame) -> pl.DataFrame:
    """Score each player–fixture PMF against that fixture's realised points.

    Blank GWs yield no rows; DGWs keep one row per fixture. Pass-through
    slice columns from ``outcomes`` when present.
    """
    pts_col = "pts" if "pts" in outcomes.columns else "actual_pts"
    if pts_col not in outcomes.columns:
        raise ValueError("outcomes must include pts or actual_pts")
    required = {"player_key", "fixture_id", "points_bin", "probability"}
    missing = required - set(pmf.columns)
    if missing:
        raise ValueError(f"pmf missing columns: {sorted(missing)}")

    keep = [
        c
        for c in (
            "model_id",
            "season",
            "gw",
            "target_season",
            "target_gw",
            "position",
            "price_band",
            "player_key",
            "fixture_id",
            pts_col,
        )
        if c in outcomes.columns
    ]
    scored_out = outcomes.select(keep).unique(
        subset=["player_key", "fixture_id"], keep="first"
    )
    rows: list[dict[str, object]] = []
    for keys, group in pmf.group_by(["player_key", "fixture_id"]):
        player_key, fixture_id = int(keys[0]), int(keys[1])  # type: ignore[index]
        match = scored_out.filter(
            (pl.col("player_key") == player_key) & (pl.col("fixture_id") == fixture_id)
        )
        if match.is_empty():
            continue
        y = int(match[pts_col][0])
        dense = _dense_pmf(
            group["points_bin"].to_numpy(), group["probability"].to_numpy()
        )
        row: dict[str, object] = {
            "player_key": player_key,
            "fixture_id": fixture_id,
            "rps": discrete_rps(dense, y),
            "log_score": log_score_pmf(dense, y),
            "mse_mean": mse_mean_pmf(dense, y),
        }
        for col in keep:
            if col not in ("player_key", "fixture_id", pts_col):
                row[col] = match[col][0]
        if "season" not in row and "target_season" in match.columns:
            row["season"] = match["target_season"][0]
        if "gw" not in row and "target_gw" in match.columns:
            row["gw"] = match["target_gw"][0]
        rows.append(row)
    if not rows:
        return pl.DataFrame(
            schema={
                "player_key": pl.Int64(),
                "fixture_id": pl.Int64(),
                "rps": pl.Float64(),
                "log_score": pl.Float64(),
                "mse_mean": pl.Float64(),
            }
        )
    return pl.DataFrame(rows)
