"""Paired forecast intervals for promote-eval (§9).

Per-gameweek loss differentials use Diebold–Mariano / Giacomini–White via
``statsmodels`` OLS with HAC covariance, plus a stationary bootstrap CI from
``arch``. SPA and MCS run only when several arms share a window. Intervals are
descriptive for improvement levers; for equivalence, an interval entirely on
the bad side may ``drop`` (never a ``keep`` rule).
"""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, cast

import numpy as np
import polars as pl
import statsmodels.api as sm
from arch.bootstrap import MCS, SPA, StationaryBootstrap
from numpy.typing import NDArray

MetricName = Literal[
    "mae",
    "rmse",
    "spearman",
    "top10_precision",
    "brier_haul",
    "brier_60",
]

DEFAULT_BOOTSTRAP_REPS = 2000
DEFAULT_BOOTSTRAP_SEED = 0
DEFAULT_SPA_REPS = 1000
DEFAULT_MCS_SIZE = 0.10

FloatArray = NDArray[np.floating[Any]]


@dataclass(frozen=True)
class PairedInterval:
    """HAC OLS and stationary-bootstrap summary for one ``d_t`` series.

    ``d_t`` is signed so a positive mean means the treatment (or candidate) is
    better than the control (or benchmark).
    """

    metric: str
    n_gw: int
    mean: float
    hac_se: float | None
    hac_pvalue: float | None
    hac_ci_low: float | None
    hac_ci_high: float | None
    bootstrap_ci_low: float | None
    bootstrap_ci_high: float | None
    bootstrap_reps: int
    bootstrap_seed: int
    position: str | None = None

    @property
    def entirely_bad(self) -> bool:
        """True when both interval upper bounds are strictly below zero.

        Equivalence uses this to ``drop`` a candidate that is worse. Missing
        bounds (too few gameweeks) do not trigger a drop.
        """
        return interval_entirely_bad(
            self.hac_ci_low,
            self.hac_ci_high,
            self.bootstrap_ci_low,
            self.bootstrap_ci_high,
        )


@dataclass(frozen=True)
class SpaResult:
    """Hansen SPA ("does any arm beat the control?")."""

    pvalue_consistent: float
    pvalue_lower: float
    pvalue_upper: float
    n_gw: int
    n_models: int


@dataclass(frozen=True)
class McsResult:
    """Hansen–Lunde–Nason model confidence set."""

    included: tuple[str, ...]
    excluded: tuple[str, ...]
    size: float
    n_gw: int


def interval_entirely_bad(
    hac_low: float | None,
    hac_high: float | None,
    boot_low: float | None,
    boot_high: float | None,
) -> bool:
    """Return whether every available interval lies entirely below zero.

    Positive ``d_t`` means the candidate is better. An interval entirely on the
    bad side has upper bound ``< 0``. Both HAC and bootstrap must agree when
    both are present; a single available interval is enough when the other is
    missing.

    Args:
        hac_low: HAC CI lower bound (unused for the bad-side test).
        hac_high: HAC CI upper bound.
        boot_low: Bootstrap CI lower bound (unused for the bad-side test).
        boot_high: Bootstrap CI upper bound.

    Returns:
        True when every non-null upper bound is strictly negative.
    """
    del hac_low, boot_low
    uppers = [u for u in (hac_high, boot_high) if u is not None]
    if not uppers:
        return False
    return all(u < 0.0 for u in uppers)


def equivalence_drop_from_intervals(intervals: list[PairedInterval]) -> bool:
    """Return True when any primary-metric interval is entirely on the bad side.

    This is the §3.2 / §9 equivalence ``drop`` rule. It is never a ``keep``
    rule, and improvement levers must not call it for promotion.

    Args:
        intervals: Paired intervals signed so positive means candidate better.

    Returns:
        True if at least one interval has ``entirely_bad``.
    """
    return any(interval.entirely_bad for interval in intervals)


def per_gw_losses(
    ledger: pl.DataFrame,
    *,
    model_id: str,
    metric: MetricName,
    horizon: int = 1,
    position: str | None = None,
) -> pl.DataFrame:
    """Compute one loss (or negated score) per target gameweek.

    Rows with null ``actual_pts`` are skipped. The unit is the gameweek so
    doubles and blanks fall out of the pairing. For metrics where higher is
    better (Spearman, top-10), the series is negated so smaller is always
    better for SPA/MCS.

    Args:
        ledger: Scored prediction ledger (``PredictionSchema`` columns).
        model_id: Model artifact id to filter.
        metric: Primary metric name.
        horizon: Prediction horizon (default 1).
        position: Optional position slice (``GKP``/``DEF``/``MID``/``FWD``).

    Returns:
        Frame with columns ``target_season``, ``target_gw``, ``loss``.
    """
    scored = ledger.filter(
        (pl.col("model_id") == model_id)
        & (pl.col("horizon") == horizon)
        & pl.col("actual_pts").is_not_null()
    )
    if position is not None and "position" in scored.columns:
        scored = scored.filter(pl.col("position") == position)
    if scored.is_empty():
        return pl.DataFrame(
            schema={
                "target_season": pl.Utf8(),
                "target_gw": pl.Int64(),
                "loss": pl.Float64(),
            }
        )

    rows: list[dict[str, object]] = []
    for (season, gw), group in scored.group_by(["target_season", "target_gw"]):
        loss = _gw_loss(group, metric)
        if loss is None:
            continue
        rows.append(
            {
                "target_season": str(season),
                "target_gw": int(gw),  # type: ignore[arg-type]
                "loss": float(loss),
            }
        )
    if not rows:
        return pl.DataFrame(
            schema={
                "target_season": pl.Utf8(),
                "target_gw": pl.Int64(),
                "loss": pl.Float64(),
            }
        )
    return pl.DataFrame(rows).sort(["target_season", "target_gw"])


def loss_differential(
    control_losses: pl.DataFrame,
    treatment_losses: pl.DataFrame,
) -> FloatArray:
    """Align per-GW losses and return ``d_t = control - treatment``.

    Positive values mean the treatment has lower loss (is better). Gameweeks
    present on only one arm are dropped.

    Args:
        control_losses: Output of ``per_gw_losses`` for the control.
        treatment_losses: Output of ``per_gw_losses`` for the treatment.

    Returns:
        1-d array of paired differentials.
    """
    if control_losses.is_empty() or treatment_losses.is_empty():
        return np.asarray([], dtype=np.float64)
    joined = control_losses.join(
        treatment_losses,
        on=["target_season", "target_gw"],
        how="inner",
        suffix="_t",
    ).sort(["target_season", "target_gw"])
    if joined.is_empty():
        return np.asarray([], dtype=np.float64)
    d = joined["loss"].to_numpy() - joined["loss_t"].to_numpy()
    return np.asarray(d, dtype=np.float64)


def paired_interval(
    d_t: FloatArray | list[float],
    *,
    metric: str,
    position: str | None = None,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> PairedInterval:
    """Fit HAC OLS and a stationary-bootstrap CI on ``d_t``.

    Args:
        d_t: Per-gameweek differentials (positive ⇒ treatment better).
        metric: Metric label stored on the result.
        position: Optional position slice label.
        bootstrap_reps: Stationary bootstrap replications (default 2000).
        bootstrap_seed: Fixed seed for the bootstrap.

    Returns:
        ``PairedInterval`` with mean, HAC SE/p/CI, and bootstrap CI.
    """
    series = np.asarray(d_t, dtype=np.float64).reshape(-1)
    n = int(series.size)
    if n == 0 or not np.isfinite(series).all():
        return PairedInterval(
            metric=metric,
            n_gw=n,
            mean=float("nan"),
            hac_se=None,
            hac_pvalue=None,
            hac_ci_low=None,
            hac_ci_high=None,
            bootstrap_ci_low=None,
            bootstrap_ci_high=None,
            bootstrap_reps=bootstrap_reps,
            bootstrap_seed=bootstrap_seed,
            position=position,
        )

    mean = float(np.mean(series))
    hac_se: float | None = None
    hac_p: float | None = None
    hac_lo: float | None = None
    hac_hi: float | None = None
    if n >= 2:
        maxlags = _hac_maxlags(n)
        ols = sm.OLS(series, np.ones(n)).fit(
            cov_type="HAC", cov_kwds={"maxlags": maxlags}
        )
        hac_se = float(ols.bse[0])
        hac_p = float(ols.pvalues[0])
        ci = ols.conf_int(alpha=0.05)
        hac_lo = float(ci[0, 0])
        hac_hi = float(ci[0, 1])

    boot_lo: float | None = None
    boot_hi: float | None = None
    if n >= 2:
        block = max(1, int(np.sqrt(n)))
        bs = StationaryBootstrap(block, series, seed=bootstrap_seed)
        ci_boot = bs.conf_int(
            np.mean, reps=bootstrap_reps, method="percentile", size=0.95
        )
        boot_lo = float(ci_boot[0, 0])
        boot_hi = float(ci_boot[1, 0])

    return PairedInterval(
        metric=metric,
        n_gw=n,
        mean=mean,
        hac_se=hac_se,
        hac_pvalue=hac_p,
        hac_ci_low=hac_lo,
        hac_ci_high=hac_hi,
        bootstrap_ci_low=boot_lo,
        bootstrap_ci_high=boot_hi,
        bootstrap_reps=bootstrap_reps,
        bootstrap_seed=bootstrap_seed,
        position=position,
    )


def intervals_for_pair(
    ledger: pl.DataFrame,
    *,
    control_id: str,
    treatment_id: str,
    metrics: tuple[MetricName, ...] = ("mae", "rmse", "spearman"),
    horizon: int = 1,
    positions: tuple[str, ...] | None = None,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> list[PairedInterval]:
    """Build paired intervals for each metric (and optional position slices).

    Args:
        ledger: Scored ledger covering both arms.
        control_id: Control ``model_id``.
        treatment_id: Treatment ``model_id``.
        metrics: Primary metrics to report.
        horizon: Prediction horizon.
        positions: When set, also compute a slice per position name.
        bootstrap_reps: Bootstrap replications.
        bootstrap_seed: Bootstrap seed.

    Returns:
        List of ``PairedInterval`` (overall first, then position slices).
    """
    results: list[PairedInterval] = []
    slices: list[str | None] = [None]
    if positions:
        slices.extend(positions)
    for position in slices:
        for metric in metrics:
            control = per_gw_losses(
                ledger,
                model_id=control_id,
                metric=metric,
                horizon=horizon,
                position=position,
            )
            treatment = per_gw_losses(
                ledger,
                model_id=treatment_id,
                metric=metric,
                horizon=horizon,
                position=position,
            )
            d_t = loss_differential(control, treatment)
            results.append(
                paired_interval(
                    d_t,
                    metric=metric,
                    position=position,
                    bootstrap_reps=bootstrap_reps,
                    bootstrap_seed=bootstrap_seed,
                )
            )
    return results


def spa_test(
    benchmark_losses: FloatArray | list[float],
    model_losses: FloatArray | list[list[float]],
    *,
    reps: int = DEFAULT_SPA_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> SpaResult:
    """Hansen SPA: does any alternative beat the benchmark?

    Args:
        benchmark_losses: Length-T control / benchmark losses (smaller better).
        model_losses: T×k alternative losses (columns are arms).
        reps: Bootstrap replications.
        seed: Bootstrap seed.

    Returns:
        ``SpaResult`` with lower / consistent / upper p-values.

    Raises:
        ValueError: When shapes are incompatible or there are fewer than two
            gameweeks.
    """
    benchmark = np.asarray(benchmark_losses, dtype=np.float64).reshape(-1)
    models = np.asarray(model_losses, dtype=np.float64)
    if models.ndim == 1:
        models = models.reshape(-1, 1)
    if benchmark.size < 2 or models.shape[0] != benchmark.size:
        raise ValueError(
            "SPA needs T>=2 and model_losses shaped (T, k) matching benchmark"
        )
    if models.shape[1] < 1:
        raise ValueError("SPA needs at least one alternative arm")
    spa = SPA(benchmark, models, reps=reps, bootstrap="stationary", seed=seed)
    spa.compute()
    pvalues = cast(dict[str, float], dict(spa.pvalues))
    return SpaResult(
        pvalue_consistent=float(pvalues["consistent"]),
        pvalue_lower=float(pvalues["lower"]),
        pvalue_upper=float(pvalues["upper"]),
        n_gw=int(benchmark.size),
        n_models=int(models.shape[1]),
    )


def mcs_set(
    losses: FloatArray | list[list[float]],
    *,
    names: tuple[str, ...] | list[str],
    size: float = DEFAULT_MCS_SIZE,
    reps: int = DEFAULT_SPA_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> McsResult:
    """Model confidence set over several arms on a shared window.

    Args:
        losses: T×k loss matrix (smaller better); columns align with ``names``.
        names: Arm labels for each column.
        size: MCS test size (default 0.10).
        reps: Bootstrap replications.
        seed: Bootstrap seed.

    Returns:
        ``McsResult`` with included / excluded arm names.

    Raises:
        ValueError: On shape / name mismatches or T < 2.
    """
    frame = np.asarray(losses, dtype=np.float64)
    if frame.ndim == 1:
        frame = frame.reshape(-1, 1)
    labels = tuple(names)
    if frame.shape[0] < 2:
        raise ValueError("MCS needs T>=2 gameweeks")
    if frame.shape[1] != len(labels):
        raise ValueError("MCS losses columns must match names")
    if frame.shape[1] < 2:
        raise ValueError("MCS needs at least two arms")
    mcs = MCS(frame, size=size, reps=reps, bootstrap="stationary", seed=seed)
    mcs.compute()
    included_raw = cast(list[object], list(mcs.included))
    included_idx = {int(cast(int | str, i)) for i in included_raw}
    included = tuple(labels[i] for i in range(len(labels)) if i in included_idx)
    excluded = tuple(labels[i] for i in range(len(labels)) if i not in included_idx)
    return McsResult(
        included=included,
        excluded=excluded,
        size=size,
        n_gw=int(frame.shape[0]),
    )


def multi_arm_selection(
    loss_by_arm: dict[str, FloatArray],
    *,
    control: str,
    reps: int = DEFAULT_SPA_REPS,
    seed: int = DEFAULT_BOOTSTRAP_SEED,
    mcs_size: float = DEFAULT_MCS_SIZE,
) -> tuple[SpaResult, McsResult] | None:
    """Run SPA (vs control) and MCS when several arms share a window.

    Args:
        loss_by_arm: Map of arm name → length-T loss series (aligned).
        control: Benchmark arm name (must be a key).
        reps: Bootstrap replications for SPA and MCS.
        seed: Shared seed.
        mcs_size: MCS size.

    Returns:
        ``(SpaResult, McsResult)`` when there are at least two non-control arms
        plus the control (three series total), else ``None``.
    """
    if control not in loss_by_arm:
        raise ValueError(f"control {control!r} missing from loss_by_arm")
    others = [name for name in loss_by_arm if name != control]
    if len(others) < 2:
        return None
    names = (control, *others)
    columns = [
        np.asarray(loss_by_arm[name], dtype=np.float64).reshape(-1) for name in names
    ]
    lengths = {col.size for col in columns}
    if len(lengths) != 1:
        raise ValueError("all arm loss series must share the same length")
    matrix = np.column_stack(columns)
    spa = spa_test(matrix[:, 0], matrix[:, 1:], reps=reps, seed=seed)
    mcs = mcs_set(matrix, names=names, size=mcs_size, reps=reps, seed=seed)
    return spa, mcs


def per_gw_wins_from_losses(
    control_losses: pl.DataFrame,
    treatment_losses: pl.DataFrame,
) -> dict[str, int]:
    """Count gameweeks won by control / treatment on a loss series.

    Args:
        control_losses: Control per-GW losses.
        treatment_losses: Treatment per-GW losses.

    Returns:
        ``{"control", "treatment", "ties"}`` win counts.
    """
    wins = {"control": 0, "treatment": 0, "ties": 0}
    if control_losses.is_empty() or treatment_losses.is_empty():
        return wins
    joined = control_losses.join(
        treatment_losses,
        on=["target_season", "target_gw"],
        how="inner",
        suffix="_t",
    )
    for row in joined.to_dicts():
        c = float(row["loss"])
        t = float(row["loss_t"])
        if c < t:
            wins["control"] += 1
        elif t < c:
            wins["treatment"] += 1
        else:
            wins["ties"] += 1
    return wins


def interval_to_dict(interval: PairedInterval) -> dict[str, Any]:
    """Serialize a ``PairedInterval`` for JSON / recommendation tables."""
    return {
        "metric": interval.metric,
        "position": interval.position,
        "n_gw": interval.n_gw,
        "mean": interval.mean,
        "hac_se": interval.hac_se,
        "hac_pvalue": interval.hac_pvalue,
        "hac_ci": [interval.hac_ci_low, interval.hac_ci_high],
        "bootstrap_ci": [interval.bootstrap_ci_low, interval.bootstrap_ci_high],
        "bootstrap_reps": interval.bootstrap_reps,
        "bootstrap_seed": interval.bootstrap_seed,
        "entirely_bad": interval.entirely_bad,
    }


def _gw_loss(group: pl.DataFrame, metric: MetricName) -> float | None:
    if group.is_empty():
        return None
    error = (group["xpts"] - group["actual_pts"]).cast(pl.Float64)
    if metric == "mae":
        return _mean_num(error.abs())
    if metric == "rmse":
        return float(_mean_num(error.pow(2)) ** 0.5)
    if metric == "spearman":
        ranked = group.with_columns(
            pl.col("xpts").rank().alias("pred_rank"),
            pl.col("actual_pts").rank().alias("actual_rank"),
        )
        corr = ranked.select(pl.corr("pred_rank", "actual_rank")).item()
        if corr is None:
            return None
        # Negate so smaller loss is better (aligns with SPA/MCS).
        return -float(corr)
    if metric == "top10_precision":
        k = min(10, group.height)
        if k == 0:
            return None
        pred_top = set(
            group.sort("xpts", descending=True).head(k)["player_key"].to_list()
        )
        actual_top = set(
            group.sort("actual_pts", descending=True).head(k)["player_key"].to_list()
        )
        return -(len(pred_top & actual_top) / k)
    if metric == "brier_haul":
        haul = group.filter(pl.col("p_haul").is_not_null())
        if haul.is_empty():
            return None
        residual = haul["p_haul"] - (haul["actual_pts"] >= 10).cast(pl.Float64)
        return _mean_num(residual.cast(pl.Float64).pow(2))
    if metric == "brier_60":
        played = group.filter(
            pl.col("p_60").is_not_null() & pl.col("actual_minutes").is_not_null()
        )
        if played.is_empty():
            return None
        residual = played["p_60"] - (played["actual_minutes"] >= 60).cast(pl.Float64)
        return _mean_num(residual.cast(pl.Float64).pow(2))
    raise ValueError(f"unknown metric {metric!r}")


def _mean_num(series: pl.Series) -> float:
    value = series.mean()
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)


def _hac_maxlags(n: int) -> int:
    """Newey–West-style lag length for small walk-forward samples."""
    return max(1, int(n ** (1.0 / 3.0)))


__all__ = [
    "DEFAULT_BOOTSTRAP_REPS",
    "DEFAULT_BOOTSTRAP_SEED",
    "DEFAULT_MCS_SIZE",
    "DEFAULT_SPA_REPS",
    "McsResult",
    "PairedInterval",
    "SpaResult",
    "equivalence_drop_from_intervals",
    "interval_entirely_bad",
    "interval_to_dict",
    "intervals_for_pair",
    "loss_differential",
    "mcs_set",
    "multi_arm_selection",
    "paired_interval",
    "per_gw_losses",
    "per_gw_wins_from_losses",
    "spa_test",
]
