"""Signal screen: residuals vs unused PIT-safe covariates (BH-FDR).

Toolkit §1.2 / #402. Residuals default to the prediction-ledger view
(``actual_pts − xpts``) until ``forecast_scores`` (#395) lands. Covariate
values with ``available_at`` after the row cut-off are excluded (PIT guard).
"""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import datetime
from typing import Literal

import numpy as np
import polars as pl
import statsmodels.api as sm
from numpy.typing import NDArray
from scipy import stats
from statsmodels.stats.sandwich_covariance import cov_cluster_2groups

from fplt.schemas.signal_screen import SIGNAL_SCREEN_COLUMNS, SignalScreenSchema
from fplt.utils.frames import require_dataframe

FloatArray = NDArray[np.floating]

DEFAULT_FDR_ALPHA = 0.10
_MIN_ROWS = 8
_Z_95 = 1.959963984540054

SourceKind = Literal["staged", "derived"]

DERIVED_COVARIATES: tuple[str, ...] = (
    "rest_days",
    "predicted_lineup_confidence",
    "odds_implied_team_goals",
    "set_piece_duty_change",
    "rotation_index",
    "minutes_last_3",
)


def residuals_from_ledger(
    ledger: pl.DataFrame,
    *,
    component: str = "points",
) -> pl.DataFrame:
    """Build a residual panel from the prediction-ledger view.

    Until ``forecast_scores`` (#395) exists, points residuals are
    ``actual_pts − xpts`` on scored rows. Other components require a
    pre-built residual column and are not derived here.

    Args:
        ledger: ``LedgerSchema`` rows (predictions ⋈ outcomes).
        component: Residual family; only ``\"points\"`` is supported.

    Returns:
        Panel with ``residual``, ``mu_hat``, ``position``, ``fixture_id``,
        ``gw``, ``season``, ``player_key``, and ``component``.

    Raises:
        ValueError: ``component`` is not ``\"points\"``.
    """
    if component != "points":
        raise ValueError(
            "residuals_from_ledger only supports component='points' until "
            "forecast_scores (#395); pass a pre-built residual panel instead"
        )
    scored = ledger.filter(pl.col("actual_pts").is_not_null())
    if scored.is_empty():
        return pl.DataFrame(
            schema={
                "season": pl.Utf8(),
                "gw": pl.Int64(),
                "player_key": pl.Int64(),
                "fixture_id": pl.Int64(),
                "position": pl.Utf8(),
                "mu_hat": pl.Float64(),
                "residual": pl.Float64(),
                "component": pl.Utf8(),
            }
        )
    return scored.select(
        pl.col("target_season").alias("season"),
        pl.col("target_gw").alias("gw"),
        "player_key",
        "fixture_id",
        "position",
        pl.col("xpts").cast(pl.Float64()).alias("mu_hat"),
        (pl.col("actual_pts").cast(pl.Float64()) - pl.col("xpts")).alias("residual"),
        pl.lit(component).alias("component"),
    )


def apply_pit_guard(
    values: pl.Series,
    available_at: pl.Series,
    cutoff: pl.Series | datetime,
) -> pl.Series:
    """Null covariate values whose ``available_at`` is after the row cut-off.

    Args:
        values: Covariate values aligned to residual rows.
        available_at: UTC timestamps for each value.
        cutoff: Per-row deadline or a single shared cut-off.

    Returns:
        ``values`` with late rows set to null (PIT-safe).

    Raises:
        ValueError: Series lengths differ.
    """
    if len(values) != len(available_at):
        raise ValueError("values and available_at must have the same length")
    if isinstance(cutoff, datetime):
        cut = pl.Series("cutoff", [cutoff] * len(values))
    else:
        if len(cutoff) != len(values):
            raise ValueError("cutoff series must match values length")
        cut = cutoff
    guarded = (
        pl.DataFrame({"value": values, "available_at": available_at, "cutoff": cut})
        .select(
            pl.when(pl.col("available_at") <= pl.col("cutoff"))
            .then(pl.col("value"))
            .otherwise(None)
            .alias(values.name)
        )
        .to_series()
    )
    return guarded


def benjamini_hochberg(
    p_values: Sequence[float] | FloatArray,
    *,
    alpha: float = DEFAULT_FDR_ALPHA,
) -> tuple[FloatArray, NDArray[np.bool_]]:
    """Benjamini–Hochberg FDR q-values and reject flags.

    Args:
        p_values: Raw p-values (NaN entries stay non-rejected).
        alpha: FDR level (default 10%).

    Returns:
        ``(q_values, rejected)`` aligned to the input order.
    """
    p = np.asarray(p_values, dtype=np.float64)
    m = p.size
    q = np.full(m, np.nan, dtype=np.float64)
    rejected = np.zeros(m, dtype=bool)
    if m == 0:
        return q, rejected
    valid = np.isfinite(p)
    if not valid.any():
        return q, rejected
    idx = np.where(valid)[0]
    order = idx[np.argsort(p[idx], kind="mergesort")]
    ranked = p[order]
    n_valid = ranked.size
    # Step-up BH: q_(i) = min_{j≥i} (p_(j) * m / j), capped at 1.
    raw = ranked * n_valid / np.arange(1, n_valid + 1, dtype=np.float64)
    adj = np.minimum.accumulate(raw[::-1])[::-1]
    adj = np.clip(adj, 0.0, 1.0)
    q[order] = adj
    rejected[order] = adj <= alpha
    return q, rejected


def unused_staged_tables(required: set[str], staged: set[str]) -> set[str]:
    """Return staged tables not required by the model features.

    Args:
        required: ``required_tables(spec)`` (or a test stand-in).
        staged: Keys of ``STAGED_SCHEMAS`` (or a test stand-in).

    Returns:
        ``staged − required``.
    """
    return set(staged) - set(required)


def screen_signals(
    panel: pl.DataFrame,
    covariates: Mapping[str, pl.Series] | pl.DataFrame,
    *,
    component: str = "points",
    sources: Mapping[str, SourceKind] | None = None,
    fdr_alpha: float = DEFAULT_FDR_ALPHA,
) -> pl.DataFrame:
    """Rank covariates by partial correlation with residuals (BH-FDR).

    Controls for μ̂ and position. Standard errors are two-way cluster-robust
    by fixture and gameweek. Output validates as ``SignalScreenSchema``.

    Args:
        panel: Residual panel with ``residual``, ``mu_hat``, ``position``,
            ``fixture_id``, and ``gw``.
        covariates: Name → series aligned to ``panel``, or a wide frame of
            the same height.
        component: Residual family label written on every row.
        sources: Optional ``staged`` / ``derived`` label per covariate;
            defaults to ``derived`` when the name is in
            ``DERIVED_COVARIATES``, else ``staged``.
        fdr_alpha: Benjamini–Hochberg FDR level (default 0.10).

    Returns:
        Ranked ``SignalScreenSchema`` table (lowest q-value first).

    Raises:
        ValueError: Panel is missing required columns or covariate length
            mismatches the panel.
    """
    required = ("residual", "mu_hat", "position", "fixture_id", "gw")
    missing = [c for c in required if c not in panel.columns]
    if missing:
        raise ValueError(f"panel missing columns: {missing}")

    if isinstance(covariates, pl.DataFrame):
        cov_map = {name: covariates[name] for name in covariates.columns}
    else:
        cov_map = dict(covariates)
    if not cov_map:
        empty = pl.DataFrame(schema=SIGNAL_SCREEN_COLUMNS)
        return require_dataframe(SignalScreenSchema.validate(empty))

    rows: list[dict[str, object]] = []
    for name, series in cov_map.items():
        if len(series) != panel.height:
            raise ValueError(
                f"covariate {name!r} length {len(series)} != panel {panel.height}"
            )
        source: SourceKind
        if sources is not None and name in sources:
            source = sources[name]
        elif name in DERIVED_COVARIATES:
            source = "derived"
        else:
            source = "staged"
        rows.append(
            _screen_one(
                panel,
                series,
                name=name,
                component=component,
                source=source,
            )
        )

    p_values = np.array(
        [row["p_value"] if row["p_value"] is not None else np.nan for row in rows],
        dtype=np.float64,
    )
    q_values, rejected = benjamini_hochberg(p_values, alpha=fdr_alpha)
    for i, row in enumerate(rows):
        q = float(q_values[i]) if np.isfinite(q_values[i]) else None
        row["q_value"] = q
        row["rejected"] = bool(rejected[i])

    # Rank by q-value (missing last), then |effect_size| descending.
    def sort_key(item: tuple[int, dict[str, object]]) -> tuple[float, float]:
        _, row = item
        q = row["q_value"]
        effect = row["effect_size"]
        q_key = float(q) if isinstance(q, float) else 2.0
        mag = -abs(float(effect)) if isinstance(effect, float) else 0.0
        return (q_key, mag)

    ordered = [rows[i] for i, _ in sorted(enumerate(rows), key=sort_key)]
    for rank, row in enumerate(ordered, start=1):
        row["rank"] = rank

    frame = pl.DataFrame(ordered, schema=SIGNAL_SCREEN_COLUMNS)
    return require_dataframe(SignalScreenSchema.validate(frame))


def _screen_one(
    panel: pl.DataFrame,
    covariate: pl.Series,
    *,
    name: str,
    component: str,
    source: SourceKind,
) -> dict[str, object]:
    """Estimate one covariate's partial correlation and binned residual delta."""
    base: dict[str, object] = {
        "covariate": name,
        "component": component,
        "source": source,
        "effect_size": None,
        "se": None,
        "ci_low": None,
        "ci_high": None,
        "p_value": None,
        "q_value": None,
        "n": 0,
        "rank": 1,
        "rejected": False,
        "binned_delta": None,
        "binned_ci_low": None,
        "binned_ci_high": None,
    }
    work = panel.select(
        "residual",
        "mu_hat",
        "position",
        "fixture_id",
        "gw",
    ).with_columns(covariate.alias("x"))
    work = work.filter(
        pl.col("residual").is_not_null()
        & pl.col("mu_hat").is_not_null()
        & pl.col("x").is_not_null()
        & pl.col("position").is_not_null()
        & pl.col("fixture_id").is_not_null()
        & pl.col("gw").is_not_null()
    )
    n = work.height
    base["n"] = n
    if n < _MIN_ROWS:
        return base
    if work["x"].n_unique() < 2:
        return base

    y = work["residual"].to_numpy().astype(np.float64)
    x = work["x"].to_numpy().astype(np.float64)
    mu = work["mu_hat"].to_numpy().astype(np.float64)
    positions = work["position"].to_list()
    fixture = work["fixture_id"].to_numpy()
    gw = work["gw"].to_numpy()

    z = _control_matrix(mu, positions)
    e_y = _residualize(y, z)
    e_x = _residualize(x, z)
    if np.std(e_y) < 1e-12 or np.std(e_x) < 1e-12:
        return base

    # Standardise so the slope equals the partial correlation.
    e_y_s = (e_y - e_y.mean()) / e_y.std(ddof=1)
    e_x_s = (e_x - e_x.mean()) / e_x.std(ddof=1)
    exog = e_x_s.reshape(-1, 1)
    ols = sm.OLS(e_y_s, exog).fit()
    cov_tuple = cov_cluster_2groups(ols, fixture, gw)
    cov = np.asarray(cov_tuple[0], dtype=np.float64)
    se = float(np.sqrt(max(cov[0, 0], 0.0)))
    beta = float(ols.params[0])
    if se > 0:
        t_stat = beta / se
        df = max(n - z.shape[1] - 1, 1)
        p_value = float(2.0 * stats.t.sf(abs(t_stat), df))
        ci_low = beta - _Z_95 * se
        ci_high = beta + _Z_95 * se
    else:
        p_value = 1.0
        ci_low = beta
        ci_high = beta

    binned_delta, b_lo, b_hi = _binned_residual_delta(y, x, gw)

    base.update(
        {
            "effect_size": beta,
            "se": se,
            "ci_low": ci_low,
            "ci_high": ci_high,
            "p_value": min(max(p_value, 0.0), 1.0),
            "binned_delta": binned_delta,
            "binned_ci_low": b_lo,
            "binned_ci_high": b_hi,
        }
    )
    return base


def _control_matrix(mu_hat: FloatArray, positions: Sequence[str]) -> FloatArray:
    """Build the control matrix ``[1, μ̂, position dummies]`` (drop-first)."""
    levels = sorted(set(positions))
    cols: list[FloatArray] = [
        np.ones(len(positions), dtype=np.float64),
        np.asarray(mu_hat, dtype=np.float64),
    ]
    for level in levels[1:]:
        dummy = [1.0 if p == level else 0.0 for p in positions]
        cols.append(np.asarray(dummy, dtype=np.float64))
    return np.column_stack(cols)


def _residualize(y: FloatArray, z: FloatArray) -> FloatArray:
    """OLS residuals of ``y`` on ``z``."""
    beta, _, _, _ = np.linalg.lstsq(z, y, rcond=None)
    return y - z @ beta


def _binned_residual_delta(
    residual: FloatArray,
    covariate: FloatArray,
    gw: NDArray[np.integer],
) -> tuple[float | None, float | None, float | None]:
    """Top-minus-bottom tercile mean residual with a GW-clustered SE CI."""
    if residual.size < 2 or covariate.size < 2:
        return None, None, None
    q_low, q_high = np.quantile(covariate, [1 / 3, 2 / 3])
    if not np.isfinite(q_low) or not np.isfinite(q_high) or q_low == q_high:
        return None, None, None
    low = residual[covariate <= q_low]
    high = residual[covariate >= q_high]
    if low.size < 2 or high.size < 2:
        return None, None, None
    delta = float(high.mean() - low.mean())
    # Cluster by GW on the per-row contrast contribution via group means.
    gw_ids = np.unique(gw)
    diffs: list[float] = []
    for g in gw_ids:
        mask = gw == g
        x_g = covariate[mask]
        y_g = residual[mask]
        low_g = y_g[x_g <= q_low]
        high_g = y_g[x_g >= q_high]
        if low_g.size == 0 or high_g.size == 0:
            continue
        diffs.append(float(high_g.mean() - low_g.mean()))
    if len(diffs) < 2:
        return delta, None, None
    se = float(np.std(diffs, ddof=1) / np.sqrt(len(diffs)))
    return delta, delta - _Z_95 * se, delta + _Z_95 * se
