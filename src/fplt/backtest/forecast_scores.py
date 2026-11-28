"""Forecast evaluation: ``forecast_scores`` + STD aggregations (#395 / §2.3).

Proper scores and randomised PIT are stored per player–fixture; season-to-date
slices reuse that table with block-bootstrap-by-GW CIs via ``paired_interval``.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence, Set
from typing import Any

import numpy as np
import polars as pl
from numpy.typing import NDArray

from fplt.backtest.error_attribution import price_band
from fplt.backtest.intervals import (
    DEFAULT_BOOTSTRAP_REPS,
    DEFAULT_BOOTSTRAP_SEED,
    paired_interval,
)
from fplt.backtest.points_scores import (
    dense_pmf,
    discrete_rps,
    log_score_pmf,
    pmf_event_probs,
    randomised_pit,
)
from fplt.schemas.forecast_scores import (
    FORECAST_SCORE_AGG_COLUMNS,
    FORECAST_SCORES_COLUMNS,
    SCHEMA_VERSION,
    ForecastScoreAggSchema,
    ForecastScoresSchema,
)
from fplt.store.points_pmf import POINTS_PMF_MAX, POINTS_PMF_MIN
from fplt.types import GWKey
from fplt.utils.frames import require_dataframe
from fplt.utils.rng import seed_for

FloatArray = NDArray[np.floating]
_PIT_BINS = 10
_TOP_K = (5, 10)


def _as_float(value: object, default: float = 0.0) -> float:
    """Coerce a polars scalar / numpy mean to float."""
    if value is None or isinstance(value, bool):
        return default
    if isinstance(value, int | float):
        return float(value)
    return default


__all__ = [
    "aggregate_forecast_scores",
    "compute_forecast_scores",
    "empty_forecast_score_agg",
    "empty_forecast_scores",
    "fit_climatology",
    "minutes_likely_bucket",
    "selection_conditional_bias",
]


def empty_forecast_scores() -> pl.DataFrame:
    """Return an empty validated ``ForecastScoresSchema`` frame."""
    return require_dataframe(
        ForecastScoresSchema.validate(pl.DataFrame(schema=FORECAST_SCORES_COLUMNS))
    )


def empty_forecast_score_agg() -> pl.DataFrame:
    """Return an empty validated ``ForecastScoreAggSchema`` frame."""
    return require_dataframe(
        ForecastScoreAggSchema.validate(pl.DataFrame(schema=FORECAST_SCORE_AGG_COLUMNS))
    )


def minutes_likely_bucket(p_60: float | None) -> str:
    """Map P(≥60) onto climatology strata aligned with ``minutes_bucket``."""
    if p_60 is None or not np.isfinite(p_60) or float(p_60) < 0.25:
        return "0"
    if float(p_60) < 0.75:
        return "1-59"
    return "60+"


def fit_climatology(history: pl.DataFrame) -> pl.DataFrame:
    """Fit an empirical points PMF on prior-season rows only.

    Expects ``position``, ``price_band``, ``minutes_bucket``, ``pts`` (or ``y``).
    Returns long ``(position, price_band, minutes_bucket, points_bin, probability)``.
    """
    pts = "pts" if "pts" in history.columns else "y"
    required = {"position", "price_band", "minutes_bucket", pts}
    missing = required - set(history.columns)
    if missing:
        raise ValueError(f"climatology history missing: {sorted(missing)}")
    if history.is_empty():
        return pl.DataFrame(
            schema={
                "position": pl.Utf8(),
                "price_band": pl.Utf8(),
                "minutes_bucket": pl.Utf8(),
                "points_bin": pl.Int64(),
                "probability": pl.Float64(),
            }
        )
    clipped = history.select(
        "position",
        "price_band",
        "minutes_bucket",
        pl.col(pts)
        .clip(POINTS_PMF_MIN, POINTS_PMF_MAX)
        .cast(pl.Int64)
        .alias("points_bin"),
    )
    counts = clipped.group_by(
        ["position", "price_band", "minutes_bucket", "points_bin"]
    ).agg(pl.len().alias("n"))
    totals = counts.group_by(["position", "price_band", "minutes_bucket"]).agg(
        pl.col("n").sum().alias("total")
    )
    return (
        counts.join(totals, on=["position", "price_band", "minutes_bucket"])
        .with_columns((pl.col("n") / pl.col("total")).alias("probability"))
        .select(
            "position",
            "price_band",
            "minutes_bucket",
            "points_bin",
            "probability",
        )
    )


def selection_conditional_bias(
    scores: pl.DataFrame, *, line: str = "primary"
) -> tuple[float, float]:
    """Return ``(mean(μ̂−y|selected), mean(μ̂−y|selected)−mean(μ̂−y|all))``."""
    if scores.is_empty():
        return 0.0, 0.0
    bias = (pl.col("mu_hat") - pl.col("y")).alias("bias")
    frame = scores.with_columns(bias)
    overall = _as_float(frame["bias"].mean())
    selected = frame.filter(_selected_mask(line))
    if selected.is_empty():
        return 0.0, 0.0 - overall
    sel = _as_float(selected["bias"].mean())
    return sel, sel - overall


def compute_forecast_scores(
    *,
    model_id: str,
    key: GWKey,
    points_pmf: pl.DataFrame,
    predictions: pl.DataFrame,
    outcomes: pl.DataFrame,
    selected_by_line: Mapping[str, Set[int]] | None = None,
    climatology: pl.DataFrame | None = None,
    meta: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Score each player–fixture PMF into a ``ForecastScoresSchema`` table.

    ``meta`` supplies ``player_key``, ``position``, and ``price`` (tenths) when
    outcomes lack them. Blank GWs yield no rows; DGWs keep one row per fixture.
    """
    if points_pmf.is_empty() or outcomes.is_empty():
        return empty_forecast_scores()
    pts_col = "pts" if "pts" in outcomes.columns else "actual_pts"
    out = outcomes.filter((pl.col("season") == key.season) & (pl.col("gw") == key.gw))
    if out.is_empty():
        return empty_forecast_scores()
    out = _attach_meta(out, meta)
    pred = _filter_predictions(predictions, model_id, key)
    pmf = _filter_pmf(points_pmf, key)
    if pmf.is_empty():
        return empty_forecast_scores()
    clim_map = _climatology_lookup(climatology)
    selected = selected_by_line or {}
    rows: list[dict[str, Any]] = []
    for keys, group in pmf.group_by(["player_key", "fixture_id"]):
        player_key, fixture_id = int(keys[0]), int(keys[1])  # type: ignore[index]
        match = out.filter(
            (pl.col("player_key") == player_key) & (pl.col("fixture_id") == fixture_id)
        )
        if match.is_empty():
            continue
        row = match.row(0, named=True)
        y = int(row[pts_col])
        minutes = int(row.get("minutes") or 0)
        dense = dense_pmf(
            group["points_bin"].to_numpy(), group["probability"].to_numpy()
        )
        derived = pmf_event_probs(dense)
        p_row = (
            pred.filter(
                (pl.col("player_key") == player_key)
                & (pl.col("fixture_id") == fixture_id)
            )
            if not pred.is_empty()
            else pred
        )
        probs = _event_probs(p_row, derived)
        position = str(row.get("position") or "MID")
        band = str(row.get("price_band") or price_band(int(row.get("price") or 60)))
        p_60 = probs["p_ge60"]
        bucket = minutes_likely_bucket(p_60)
        clim = clim_map.get((position, band, bucket))
        rps_clim = None if clim is None else discrete_rps(clim, y)
        rng = seed_for(model_id, key, fixture_id)
        lines = sorted(
            name for name, keys_set in selected.items() if player_key in keys_set
        )
        rows.append(
            {
                "model_id": model_id,
                "season": key.season,
                "gw": key.gw,
                "player_key": player_key,
                "fixture_id": fixture_id,
                "position": position,
                "price_band": band,
                "minutes_bucket": bucket,
                "y": y,
                "mu_hat": float(
                    p_row["xpts"][0]
                    if not p_row.is_empty() and "xpts" in p_row.columns
                    else derived["mu_hat"]
                ),
                "rps": discrete_rps(dense, y),
                "log_score": log_score_pmf(dense, y),
                "rps_clim": rps_clim,
                "pit_u": randomised_pit(dense, y, rng),
                **probs,
                "y_played": int(minutes > 0),
                "y_ge60": int(minutes >= 60),
                "y_goal": int(int(row.get("goals") or 0) > 0),
                "y_assist": int(int(row.get("assists") or 0) > 0),
                "y_cs": int(int(row.get("clean_sheets") or 0) > 0),
                "y_haul": int(y >= 10),
                "y_blank": int(y <= 2),
                "selected_lines": ",".join(lines),
                "schema_version": SCHEMA_VERSION,
            }
        )
    if not rows:
        return empty_forecast_scores()
    return require_dataframe(ForecastScoresSchema.validate(pl.DataFrame(rows)))


def aggregate_forecast_scores(
    scores: pl.DataFrame,
    *,
    model_id: str | None = None,
    season: str | None = None,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
    lines: Sequence[str] = ("primary",),
) -> pl.DataFrame:
    """Aggregate ``forecast_scores`` with block-bootstrap-by-GW CIs."""
    if scores.is_empty():
        return empty_forecast_score_agg()
    frame = scores
    if model_id is not None:
        frame = frame.filter(pl.col("model_id") == model_id)
    if season is not None:
        frame = frame.filter(pl.col("season") == season)
    if frame.is_empty():
        return empty_forecast_score_agg()
    mid = str(frame["model_id"][0])
    seas = str(frame["season"][0])
    out: list[dict[str, Any]] = []
    out.extend(
        _std_metrics(
            frame,
            model_id=mid,
            season=seas,
            slice_name="all",
            bootstrap_reps=bootstrap_reps,
            bootstrap_seed=bootstrap_seed,
        )
    )
    for pos in sorted(frame["position"].unique().to_list()):
        part = frame.filter(pl.col("position") == pos)
        out.extend(
            _std_metrics(
                part,
                model_id=mid,
                season=seas,
                slice_name=f"position={pos}",
                bootstrap_reps=bootstrap_reps,
                bootstrap_seed=bootstrap_seed,
            )
        )
    for line in lines:
        sel = frame.filter(_selected_mask(line))
        if sel.is_empty():
            continue
        bias_rows = _selection_agg(
            frame,
            sel,
            model_id=mid,
            season=seas,
            line=line,
            bootstrap_reps=bootstrap_reps,
            bootstrap_seed=bootstrap_seed,
        )
        out.extend(bias_rows)
    out.extend(_pit_histogram(frame, model_id=mid, season=seas))
    return require_dataframe(
        ForecastScoreAggSchema.validate(
            pl.DataFrame(out, schema=FORECAST_SCORE_AGG_COLUMNS)
        )
    )


def _attach_meta(outcomes: pl.DataFrame, meta: pl.DataFrame | None) -> pl.DataFrame:
    frame = outcomes
    if meta is not None and not meta.is_empty():
        cols = [c for c in ("player_key", "position", "price") if c in meta.columns]
        if "player_key" in cols and len(cols) > 1:
            frame = frame.join(
                meta.select(cols).unique(subset=["player_key"], keep="last"),
                on="player_key",
                how="left",
            )
    if "price_band" not in frame.columns and "price" in frame.columns:
        frame = frame.with_columns(
            pl.col("price")
            .map_elements(lambda p: price_band(int(p or 60)), return_dtype=pl.Utf8)
            .alias("price_band")
        )
    if "price_band" not in frame.columns:
        frame = frame.with_columns(pl.lit("mid").alias("price_band"))
    if "position" not in frame.columns:
        frame = frame.with_columns(pl.lit("MID").alias("position"))
    return frame


def _filter_predictions(
    predictions: pl.DataFrame, model_id: str, key: GWKey
) -> pl.DataFrame:
    if predictions.is_empty():
        return predictions
    season_col = "target_season" if "target_season" in predictions.columns else "season"
    gw_col = "target_gw" if "target_gw" in predictions.columns else "gw"
    part = predictions
    if "model_id" in part.columns:
        part = part.filter(pl.col("model_id") == model_id)
    return part.filter((pl.col(season_col) == key.season) & (pl.col(gw_col) == key.gw))


def _filter_pmf(points_pmf: pl.DataFrame, key: GWKey) -> pl.DataFrame:
    frame = points_pmf
    if "target_season" in frame.columns:
        frame = frame.filter(pl.col("target_season") == key.season)
    elif "season" in frame.columns:
        frame = frame.filter(pl.col("season") == key.season)
    if "target_gw" in frame.columns:
        frame = frame.filter(pl.col("target_gw") == key.gw)
    elif "gw" in frame.columns:
        frame = frame.filter(pl.col("gw") == key.gw)
    return frame


def _selected_mask(line: str) -> pl.Expr:
    """True when ``selected_lines`` contains ``line`` (comma-joined)."""
    return (
        (pl.col("selected_lines") == line)
        | pl.col("selected_lines").str.starts_with(f"{line},")
        | pl.col("selected_lines").str.ends_with(f",{line}")
        | pl.col("selected_lines").str.contains(f",{line},", literal=True)
    )


def _event_probs(
    pred_row: pl.DataFrame, derived: Mapping[str, float]
) -> dict[str, float]:
    def _get(name: str, fallback: float) -> float:
        if pred_row.is_empty() or name not in pred_row.columns:
            return fallback
        value = pred_row[name][0]
        if value is None or not np.isfinite(float(value)):
            return fallback
        return float(value)

    p_played = _get("p_played", derived.get("p_played", 0.5))
    p_60 = _get("p_60", derived.get("p_ge60", 0.5))
    return {
        "p_played": p_played,
        "p_ge60": p_60,
        "p_goal": _get("p_goal", 0.0),
        "p_assist": _get("p_assist", 0.0),
        "p_cs": _get("p_cs", 0.0),
        "p_haul": _get("p_haul", derived["p_haul"]),
        "p_blank": derived["p_blank"],
    }


def _climatology_lookup(
    climatology: pl.DataFrame | None,
) -> dict[tuple[str, str, str], FloatArray]:
    if climatology is None or climatology.is_empty():
        return {}
    out: dict[tuple[str, str, str], FloatArray] = {}
    for keys, group in climatology.group_by(
        ["position", "price_band", "minutes_bucket"]
    ):
        position, band, bucket = str(keys[0]), str(keys[1]), str(keys[2])  # type: ignore[index]
        out[(position, band, bucket)] = dense_pmf(
            group["points_bin"].to_numpy(), group["probability"].to_numpy()
        )
    return out


def _gw_means(frame: pl.DataFrame, expr: pl.Expr, alias: str = "v") -> np.ndarray:
    series = (
        frame.group_by("gw")
        .agg(expr.alias(alias))
        .sort("gw")[alias]
        .to_numpy()
        .astype(np.float64)
    )
    return series


def _boot_row(
    *,
    model_id: str,
    season: str,
    slice_name: str,
    metric: str,
    series: np.ndarray,
    n: int,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    interval = paired_interval(
        series,
        metric=metric,
        bootstrap_reps=bootstrap_reps,
        bootstrap_seed=bootstrap_seed,
    )
    mean = float(np.mean(series)) if series.size else float("nan")
    return {
        "model_id": model_id,
        "season": season,
        "gw": None,
        "slice": slice_name,
        "metric": metric,
        "value": mean,
        "ci_lo": interval.bootstrap_ci_low,
        "ci_hi": interval.bootstrap_ci_high,
        "n": n,
        "schema_version": SCHEMA_VERSION,
    }


def _std_metrics(
    frame: pl.DataFrame,
    *,
    model_id: str,
    season: str,
    slice_name: str,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    n = int(frame.height)
    rows = [
        _boot_row(
            model_id=model_id,
            season=season,
            slice_name=slice_name,
            metric="mean_rps",
            series=_gw_means(frame, pl.col("rps").mean()),
            n=n,
            bootstrap_reps=bootstrap_reps,
            bootstrap_seed=bootstrap_seed,
        ),
        _boot_row(
            model_id=model_id,
            season=season,
            slice_name=slice_name,
            metric="mean_log_score",
            series=_gw_means(frame, pl.col("log_score").mean()),
            n=n,
            bootstrap_reps=bootstrap_reps,
            bootstrap_seed=bootstrap_seed,
        ),
        _boot_row(
            model_id=model_id,
            season=season,
            slice_name=slice_name,
            metric="mse_mean",
            series=_gw_means(frame, ((pl.col("mu_hat") - pl.col("y")) ** 2).mean()),
            n=n,
            bootstrap_reps=bootstrap_reps,
            bootstrap_seed=bootstrap_seed,
        ),
    ]
    clim = frame.filter(pl.col("rps_clim").is_not_null())
    if not clim.is_empty():
        skill = clim.with_columns(
            pl.when(pl.col("rps_clim") > 0)
            .then(1.0 - pl.col("rps") / pl.col("rps_clim"))
            .otherwise(0.0)
            .alias("skill")
        )
        rows.append(
            _boot_row(
                model_id=model_id,
                season=season,
                slice_name=slice_name,
                metric="rps_skill_clim",
                series=_gw_means(skill, pl.col("skill").mean()),
                n=int(clim.height),
                bootstrap_reps=bootstrap_reps,
                bootstrap_seed=bootstrap_seed,
            )
        )
    for level, name in (
        (0.5, "pit_coverage_50"),
        (0.8, "pit_coverage_80"),
        (0.9, "pit_coverage_90"),
    ):
        lo, hi = (1.0 - level) / 2.0, 1.0 - (1.0 - level) / 2.0
        covered = ((pl.col("pit_u") >= lo) & (pl.col("pit_u") <= hi)).cast(pl.Float64)
        rows.append(
            _boot_row(
                model_id=model_id,
                season=season,
                slice_name=slice_name,
                metric=name,
                series=_gw_means(
                    frame.with_columns(covered.alias("c")), pl.col("c").mean()
                ),
                n=n,
                bootstrap_reps=bootstrap_reps,
                bootstrap_seed=bootstrap_seed,
            )
        )
    for event in ("played", "ge60", "goal", "assist", "cs", "haul", "blank"):
        metric = f"brier_{event}"
        p_col, y_col = f"p_{event}" if event != "ge60" else "p_ge60", f"y_{event}"
        if event == "ge60":
            p_col = "p_ge60"
        series = _gw_means(
            frame,
            ((pl.col(p_col) - pl.col(y_col)) ** 2).mean(),
        )
        rows.append(
            _boot_row(
                model_id=model_id,
                season=season,
                slice_name=slice_name,
                metric=metric,
                series=series,
                n=n,
                bootstrap_reps=bootstrap_reps,
                bootstrap_seed=bootstrap_seed,
            )
        )
    rows.extend(
        _discrimination(frame, model_id=model_id, season=season, slice_name=slice_name)
    )
    rows.extend(
        _mincer_zarnowitz(
            frame, model_id=model_id, season=season, slice_name=slice_name
        )
    )
    return rows


def _spearman(x: FloatArray, y: FloatArray) -> float:
    """Pearson correlation of average ranks (Spearman), NaN if degenerate."""
    if x.size < 2:
        return float("nan")
    rx = pl.Series(x).rank(method="average").to_numpy().astype(np.float64)
    ry = pl.Series(y).rank(method="average").to_numpy().astype(np.float64)
    if float(np.std(rx)) == 0.0 or float(np.std(ry)) == 0.0:
        return float("nan")
    corr = np.corrcoef(rx, ry)[0, 1]
    return float(corr)


def _discrimination(
    frame: pl.DataFrame,
    *,
    model_id: str,
    season: str,
    slice_name: str,
) -> list[dict[str, Any]]:
    spearman_gw: list[float] = []
    top_rates: dict[int, list[float]] = {k: [] for k in _TOP_K}
    for _, gw_frame in frame.group_by("gw"):
        for pos in gw_frame["position"].unique().to_list():
            part = gw_frame.filter(pl.col("position") == pos)
            if part.height < 2:
                continue
            rho = _spearman(
                part["mu_hat"].to_numpy().astype(np.float64),
                part["y"].to_numpy().astype(np.float64),
            )
            if np.isfinite(rho):
                spearman_gw.append(rho)
            ranked = part.sort("mu_hat", descending=True)
            for k in _TOP_K:
                head = ranked.head(min(k, ranked.height))
                # Hit = mean y of predicted top-k (descriptive; not quality gate).
                top_rates[k].append(_as_float(head["y"].mean()))
    rows: list[dict[str, Any]] = []
    if spearman_gw:
        rows.append(
            {
                "model_id": model_id,
                "season": season,
                "gw": None,
                "slice": slice_name,
                "metric": "spearman_within_pos",
                "value": float(np.mean(spearman_gw)),
                "ci_lo": None,
                "ci_hi": None,
                "n": int(frame.height),
                "schema_version": SCHEMA_VERSION,
            }
        )
    for k, values in top_rates.items():
        if not values:
            continue
        rows.append(
            {
                "model_id": model_id,
                "season": season,
                "gw": None,
                "slice": slice_name,
                "metric": f"top{k}_hit_rate",
                "value": float(np.mean(values)),
                "ci_lo": None,
                "ci_hi": None,
                "n": int(frame.height),
                "schema_version": SCHEMA_VERSION,
            }
        )
    return rows


def _mincer_zarnowitz(
    frame: pl.DataFrame,
    *,
    model_id: str,
    season: str,
    slice_name: str,
) -> list[dict[str, Any]]:
    if frame.height < 3:
        return []
    y = frame["y"].to_numpy().astype(np.float64)
    x = frame["mu_hat"].to_numpy().astype(np.float64)
    design = np.column_stack([np.ones(len(x)), x])
    try:
        beta, _, _, _ = np.linalg.lstsq(design, y, rcond=None)
    except np.linalg.LinAlgError:
        return []
    return [
        {
            "model_id": model_id,
            "season": season,
            "gw": None,
            "slice": slice_name,
            "metric": "mz_intercept",
            "value": float(beta[0]),
            "ci_lo": None,
            "ci_hi": None,
            "n": int(frame.height),
            "schema_version": SCHEMA_VERSION,
        },
        {
            "model_id": model_id,
            "season": season,
            "gw": None,
            "slice": slice_name,
            "metric": "mz_slope",
            "value": float(beta[1]),
            "ci_lo": None,
            "ci_hi": None,
            "n": int(frame.height),
            "schema_version": SCHEMA_VERSION,
        },
    ]


def _selection_agg(
    all_rows: pl.DataFrame,
    selected: pl.DataFrame,
    *,
    model_id: str,
    season: str,
    line: str,
    bootstrap_reps: int,
    bootstrap_seed: int,
) -> list[dict[str, Any]]:
    slice_name = f"line={line}"
    bias_expr = (pl.col("mu_hat") - pl.col("y")).mean()
    sel_series = _gw_means(selected, bias_expr)
    # Align on shared GW labels via recompute delta per GW.
    gws = sorted(
        {int(g) for g in selected["gw"].to_list()}
        & {int(g) for g in all_rows["gw"].to_list()}
    )
    delta: list[float] = []
    for gw in gws:
        sel_m = _as_float(
            selected.filter(pl.col("gw") == gw)
            .select((pl.col("mu_hat") - pl.col("y")).mean())
            .item()
        )
        all_m = _as_float(
            all_rows.filter(pl.col("gw") == gw)
            .select((pl.col("mu_hat") - pl.col("y")).mean())
            .item()
        )
        delta.append(sel_m - all_m)
    return [
        _boot_row(
            model_id=model_id,
            season=season,
            slice_name=slice_name,
            metric="selection_bias",
            series=sel_series,
            n=int(selected.height),
            bootstrap_reps=bootstrap_reps,
            bootstrap_seed=bootstrap_seed,
        ),
        _boot_row(
            model_id=model_id,
            season=season,
            slice_name=slice_name,
            metric="selection_bias_delta",
            series=np.asarray(delta, dtype=np.float64),
            n=int(selected.height),
            bootstrap_reps=bootstrap_reps,
            bootstrap_seed=bootstrap_seed,
        ),
    ]


def _pit_histogram(
    frame: pl.DataFrame, *, model_id: str, season: str
) -> list[dict[str, Any]]:
    u = frame["pit_u"].to_numpy().astype(np.float64)
    counts, edges = np.histogram(u, bins=_PIT_BINS, range=(0.0, 1.0))
    total = max(int(u.size), 1)
    rows: list[dict[str, Any]] = []
    for i, count in enumerate(counts.tolist()):
        rows.append(
            {
                "model_id": model_id,
                "season": season,
                "gw": None,
                "slice": f"bin={edges[i]:.1f}-{edges[i + 1]:.1f}",
                "metric": "pit_hist",
                "value": float(count) / float(total),
                "ci_lo": None,
                "ci_hi": None,
                "n": int(frame.height),
                "schema_version": SCHEMA_VERSION,
            }
        )
    return rows
