"""One-at-a-time oracle-gain error attribution by component (toolkit §1.1).

Oracle gain of ``c`` = ``score(control) − score(oracle_c)`` for RPS, log
score and MSE of the mean. Without ``forecast_scores`` (#395), control
scores come from run ``points_pmf`` ⋈ ledger outcomes.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl
from numpy.typing import NDArray

from fplt.backtest.intervals import (
    DEFAULT_BOOTSTRAP_REPS,
    DEFAULT_BOOTSTRAP_SEED,
    paired_interval,
)
from fplt.backtest.other_points_measurement import minutes_bucket
from fplt.backtest.points_scores import score_points_pmf
from fplt.models.simulation import simulate_match
from fplt.schemas.error_attribution import (
    ATTRIBUTION_METRICS,
    ERROR_ATTRIBUTION_COLUMNS,
    ORACLE_COMPONENTS,
    ErrorAttributionSchema,
)
from fplt.schemas.l2 import SCHEMA_VERSION
from fplt.scoring.rules import Rules
from fplt.store.points_pmf import POINTS_PMF_MAX, POINTS_PMF_MIN
from fplt.utils.frames import require_dataframe

FloatArray = NDArray[np.floating]
IntArray = NDArray[np.integer]
OracleComponent = Literal["minutes", "scoreline", "involvement", "other"]
_BUDGET_LT = 55
_PREMIUM_GE = 75


def price_band(price_tenths: int) -> str:
    """Map an FPL list price (tenths) to ``budget`` / ``mid`` / ``premium``."""
    if int(price_tenths) < _BUDGET_LT:
        return "budget"
    if int(price_tenths) < _PREMIUM_GE:
        return "mid"
    return "premium"


def realised_involvement_weights(outcomes: pl.DataFrame) -> pl.DataFrame:
    """Team-relative goal/assist shares from realised player–fixture outcomes."""
    required = {"player_key", "fixture_id", "team_key", "goals", "assists"}
    missing = required - set(outcomes.columns)
    if missing:
        raise ValueError(f"outcomes missing columns: {sorted(missing)}")
    totals = outcomes.group_by(["fixture_id", "team_key"]).agg(
        pl.col("goals").sum().alias("_tg"),
        pl.col("assists").sum().alias("_ta"),
    )
    joined = outcomes.join(totals, on=["fixture_id", "team_key"], how="left")
    return joined.select(
        "player_key",
        "fixture_id",
        pl.when(pl.col("_tg") > 0)
        .then(pl.col("goals") / pl.col("_tg"))
        .otherwise(0.0)
        .alias("goal_weight"),
        pl.when(pl.col("_ta") > 0)
        .then(pl.col("assists") / pl.col("_ta"))
        .otherwise(0.0)
        .alias("assist_weight"),
    )


def replace_component_inputs(
    *,
    component: OracleComponent | str,
    minutes: IntArray,
    scorelines: IntArray,
    sim_players: pl.DataFrame,
    other_grid: FloatArray,
    realised_minutes: IntArray | None = None,
    realised_scoreline: tuple[int, int] | None = None,
    realised_weights: pl.DataFrame | None = None,
    realised_other_total: FloatArray | None = None,
) -> tuple[IntArray, IntArray, pl.DataFrame, FloatArray]:
    """Copy simulator inputs with one component replaced by realised values."""
    if component not in ORACLE_COMPONENTS:
        raise ValueError(f"unknown component: {component!r}")
    minutes_out = np.array(minutes, dtype=np.int64, copy=True)
    scorelines_out = np.array(scorelines, dtype=np.int64, copy=True)
    other_out = np.array(other_grid, dtype=np.float64, copy=True)
    players_out = sim_players

    if component == "minutes":
        if realised_minutes is None:
            raise ValueError("realised_minutes required for minutes oracle")
        mins = np.asarray(realised_minutes, dtype=np.int64).reshape(-1)
        if mins.shape[0] != minutes_out.shape[0]:
            raise ValueError("realised_minutes length must match players")
        minutes_out = np.broadcast_to(mins[:, None], minutes_out.shape).copy()
    elif component == "scoreline":
        if realised_scoreline is None:
            raise ValueError("realised_scoreline required for scoreline oracle")
        scorelines_out[:, 0] = int(realised_scoreline[0])
        scorelines_out[:, 1] = int(realised_scoreline[1])
    elif component == "involvement":
        if realised_weights is None:
            raise ValueError("realised_weights required for involvement oracle")
        players_out = (
            sim_players.drop(["goal_weight", "assist_weight"])
            .join(
                realised_weights.select("player_key", "goal_weight", "assist_weight"),
                on="player_key",
                how="left",
            )
            .with_columns(
                pl.col("goal_weight").fill_null(0.0),
                pl.col("assist_weight").fill_null(0.0),
            )
            .select("player_key", "side", "position", "goal_weight", "assist_weight")
        )
    else:
        if realised_other_total is None:
            raise ValueError("realised_other_total required for other oracle")
        totals = np.asarray(realised_other_total, dtype=np.float64).reshape(-1)
        if totals.shape[0] != other_out.shape[0]:
            raise ValueError("realised_other_total length must match players")
        if realised_minutes is not None:
            bucket_mins = np.asarray(realised_minutes, dtype=np.int64).reshape(-1)
        else:
            bucket_mins = np.asarray(np.mean(minutes_out, axis=1), dtype=np.int64)
        other_out[:] = 0.0
        bucket_idx = {"0": 0, "1-59": 1, "60+": 2}
        for i, mins in enumerate(bucket_mins.tolist()):
            other_out[i, bucket_idx[minutes_bucket(int(mins))]] = totals[i]

    return minutes_out, scorelines_out, players_out, other_out


def points_pmf_from_simulation(
    simulation_points: FloatArray,
    player_keys: IntArray,
    *,
    fixture_id: int,
) -> pl.DataFrame:
    """Bin per-player simulation points into a long ``points_pmf`` frame."""
    points = np.asarray(simulation_points, dtype=np.float64)
    keys = np.asarray(player_keys, dtype=np.int64).reshape(-1)
    if points.ndim != 2 or points.shape[0] != keys.shape[0]:
        raise ValueError("points shape must be (players, n_sims)")
    width = POINTS_PMF_MAX - POINTS_PMF_MIN + 1
    rows: list[dict[str, object]] = []
    for i, key in enumerate(keys.tolist()):
        clipped = np.clip(np.rint(points[i]), POINTS_PMF_MIN, POINTS_PMF_MAX)
        counts = np.bincount(clipped.astype(np.int64) - POINTS_PMF_MIN, minlength=width)
        total = float(counts.sum())
        if total <= 0.0:
            continue
        for offset, count in enumerate(counts.tolist()):
            if count:
                rows.append(
                    {
                        "player_key": int(key),
                        "fixture_id": int(fixture_id),
                        "points_bin": POINTS_PMF_MIN + offset,
                        "probability": float(count) / total,
                    }
                )
    if not rows:
        return pl.DataFrame(
            schema={
                "player_key": pl.Int64(),
                "fixture_id": pl.Int64(),
                "points_bin": pl.Int64(),
                "probability": pl.Float64(),
            }
        )
    return pl.DataFrame(rows)


def oracle_fixture_pmf(
    *,
    component: OracleComponent | str,
    minutes: IntArray,
    scorelines: IntArray,
    sim_players: pl.DataFrame,
    other_grid: FloatArray,
    rules: Rules,
    rng: np.random.Generator,
    fixture_id: int,
    realised_minutes: IntArray | None = None,
    realised_scoreline: tuple[int, int] | None = None,
    realised_weights: pl.DataFrame | None = None,
    realised_other_total: FloatArray | None = None,
) -> pl.DataFrame:
    """Re-simulate one fixture with ``component`` replaced by realised values."""
    mins, scores, players, other = replace_component_inputs(
        component=component,
        minutes=minutes,
        scorelines=scorelines,
        sim_players=sim_players,
        other_grid=other_grid,
        realised_minutes=realised_minutes,
        realised_scoreline=realised_scoreline,
        realised_weights=realised_weights,
        realised_other_total=realised_other_total,
    )
    sim = simulate_match(scores, players, mins, other, rules, rng)
    return points_pmf_from_simulation(sim.points, sim.player_key, fixture_id=fixture_id)


def oracle_gains(
    control_scores: pl.DataFrame,
    oracle_scores: pl.DataFrame,
    *,
    component: OracleComponent | str,
) -> pl.DataFrame:
    """Per player–fixture gains ``control − oracle`` for the three metrics."""
    if component not in ORACLE_COMPONENTS:
        raise ValueError(f"unknown component: {component!r}")
    joined = control_scores.join(
        oracle_scores.select(
            "player_key",
            "fixture_id",
            pl.col("rps").alias("rps_o"),
            pl.col("log_score").alias("log_score_o"),
            pl.col("mse_mean").alias("mse_mean_o"),
        ),
        on=["player_key", "fixture_id"],
        how="inner",
    )
    return joined.with_columns(
        pl.lit(component).alias("component"),
        (pl.col("rps") - pl.col("rps_o")).alias("gain_rps"),
        (pl.col("log_score") - pl.col("log_score_o")).alias("gain_log_score"),
        (pl.col("mse_mean") - pl.col("mse_mean_o")).alias("gain_mse_mean"),
    ).drop(["rps_o", "log_score_o", "mse_mean_o"])


def attribute_errors(
    gains: pl.DataFrame,
    *,
    model_id: str,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    per_gw: bool = False,
) -> pl.DataFrame:
    """Aggregate per-row gains into an ``ErrorAttributionSchema`` table."""
    required = {
        "season",
        "gw",
        "position",
        "price_band",
        "component",
        "gain_rps",
        "gain_log_score",
        "gain_mse_mean",
    }
    missing = required - set(gains.columns)
    if missing:
        raise ValueError(f"gains missing columns: {sorted(missing)}")
    if gains.is_empty():
        empty = pl.DataFrame(schema=ERROR_ATTRIBUTION_COLUMNS)
        return require_dataframe(ErrorAttributionSchema.validate(empty))

    metric_cols = {
        "rps": "gain_rps",
        "log_score": "gain_log_score",
        "mse_mean": "gain_mse_mean",
    }
    out_rows: list[dict[str, object]] = []
    for keys, group in gains.group_by(
        ["season", "position", "price_band", "component"]
    ):
        season, position, band, component = (
            str(keys[0]),  # type: ignore[index]
            str(keys[1]),
            str(keys[2]),
            str(keys[3]),
        )
        for metric in ATTRIBUTION_METRICS:
            col = metric_cols[metric]
            gw_means = (
                group.group_by("gw")
                .agg(pl.col(col).mean().alias("gain"), pl.len().alias("n"))
                .sort("gw")
            )
            interval = paired_interval(
                gw_means["gain"].to_numpy(),
                metric=metric,
                bootstrap_reps=bootstrap_reps,
                bootstrap_seed=bootstrap_seed,
            )
            if np.isfinite(interval.mean):
                mean_gain = float(interval.mean)
            else:
                mean_gain = float(np.mean(group[col].to_numpy()))
            out_rows.append(
                {
                    "model_id": model_id,
                    "season": season,
                    "gw": None,
                    "position": position,
                    "price_band": band,
                    "component": component,
                    "metric": metric,
                    "gain": mean_gain,
                    "ci_lo": interval.bootstrap_ci_low,
                    "ci_hi": interval.bootstrap_ci_high,
                    "n": int(group.height),
                    "schema_version": SCHEMA_VERSION,
                }
            )
            if per_gw:
                for row in gw_means.iter_rows(named=True):
                    out_rows.append(
                        {
                            "model_id": model_id,
                            "season": season,
                            "gw": int(row["gw"]),
                            "position": position,
                            "price_band": band,
                            "component": component,
                            "metric": metric,
                            "gain": float(row["gain"]),
                            "ci_lo": None,
                            "ci_hi": None,
                            "n": int(row["n"]),
                            "schema_version": SCHEMA_VERSION,
                        }
                    )

    frame = pl.DataFrame(out_rows, schema=ERROR_ATTRIBUTION_COLUMNS)
    return require_dataframe(ErrorAttributionSchema.validate(frame))


def scores_from_ledger_and_pmf(
    points_pmf: pl.DataFrame, ledger: pl.DataFrame
) -> pl.DataFrame:
    """Score a run's ``points_pmf`` against ledger outcomes (no #395 table)."""
    outcomes = ledger.filter(pl.col("actual_pts").is_not_null())
    rename: dict[str, str] = {}
    if "target_season" in outcomes.columns and "season" not in outcomes.columns:
        rename["target_season"] = "season"
    if "target_gw" in outcomes.columns and "gw" not in outcomes.columns:
        rename["target_gw"] = "gw"
    if rename:
        outcomes = outcomes.rename(rename)
    return score_points_pmf(points_pmf, outcomes)
