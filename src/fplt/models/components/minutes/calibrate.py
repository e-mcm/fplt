"""Calibrate FPL chance flags from snapshot ``player_status`` (P1 / P2).

FPL publishes discrete chance-of-playing flags (0 / 25 / 50 / 75 / 100).
Using them at face value (``pct / 100``) mis-calibrates P(played) (P1), and
the availability gate historically left logistic P(≥60|start) ungated (P2).

When PIT-visible ``source=fpl_api`` status rows exist, fit empirical maps:

* ``play_gate[bucket]`` — replaces face-value as the availability multiplier
* ``long_factor[bucket]`` — multiplies P(≥60|start) relative to fully-available
  starters

Sparse buckets shrink toward face-value / 1.0. Vaastav-only windows keep the
identity maps so behaviour matches the pre-calibration gate.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

from fplt.data.pit import PITView
from fplt.models.components.minutes.common import UNAVAILABLE

_SNAPSHOT = "fpl_api"
_BUCKETS = (0, 25, 50, 75, 100)
_DEFAULT_MIN_COUNT = 20
_FULL_BUCKET = 100


@dataclass(frozen=True)
class ChanceCalibration:
    """Empirical play-gate and ≥60|start factors keyed by chance bucket."""

    play_gate: dict[int, float]
    long_factor: dict[int, float]

    @staticmethod
    def identity() -> ChanceCalibration:
        """Face-value play gate and no ≥60|start adjustment."""
        return ChanceCalibration(play_gate={}, long_factor={})


def snap_chance_bucket(pct: float | None) -> int | None:
    """Snap an FPL chance percent to the nearest published flag bucket.

    Args:
        pct: Chance-of-playing in ``[0, 100]``, or null.

    Returns:
        One of ``0, 25, 50, 75, 100``, or ``None`` when ``pct`` is null.
    """
    if pct is None:
        return None
    value = float(pct)
    return min(_BUCKETS, key=lambda bucket: abs(bucket - value))


def effective_chance_pct(
    status: str | None,
    chance_next: float | None,
    *,
    chance_this: float | None = None,
) -> float | None:
    """Effective chance percent preferring this-round over next-round.

    Unavailable statuses without a numeric chance map to 0 so the calibrated
    gate can hard-zero them. Fully available null chance stays null (gate 1).
    """
    playing = chance_this if chance_this is not None else chance_next
    if playing is not None:
        return float(playing)
    if status in UNAVAILABLE:
        return 0.0
    return None


def calibrated_play_gate(
    pct: float | None,
    face_value: float,
    calibration: ChanceCalibration,
) -> float:
    """Replace face-value chance with the fitted play gate when mapped.

    Args:
        pct: Raw FPL chance percent (pre-snap).
        face_value: ``pct / 100`` (or status fallback) already in ``[0, 1]``.
        calibration: Fitted maps; missing buckets keep ``face_value``.

    Returns:
        Multiplier in ``[0, 1]``.
    """
    bucket = snap_chance_bucket(pct)
    if bucket is None:
        return float(face_value)
    mapped = calibration.play_gate.get(bucket)
    if mapped is None:
        return float(face_value)
    return float(mapped)


def calibrated_long_factor(
    pct: float | None,
    calibration: ChanceCalibration,
) -> float:
    """P(≥60|start) multiplier for a chance flag (P2).

    Fully available / unmapped buckets return 1.0 so healthy starts keep the
    logistic conditional. Flagged buckets scale toward the empirical ratio of
    P(≥60|start | flag) to P(≥60|start | 100%).
    """
    bucket = snap_chance_bucket(pct)
    if bucket is None or bucket >= _FULL_BUCKET:
        return 1.0
    return float(calibration.long_factor.get(bucket, 1.0))


def fit_chance_calibration(
    view: PITView,
    stats: pl.DataFrame,
    *,
    min_count: int = _DEFAULT_MIN_COUNT,
) -> ChanceCalibration:
    """Fit play-gate and ≥60|start maps from snapshot status at each deadline.

    Joins PIT ``player_status`` as-of each appearance's GW deadline, keeps
    ``source=fpl_api`` rows only, and estimates per-bucket rates. Buckets with
    fewer than ``min_count`` rows shrink toward face-value (play) or 1.0
    (long factor).

    Args:
        view: Deadline PIT view (status + gameweeks visible at ``as_of``).
        stats: ``player_fixture_stats`` rows with ``season``, ``gw``,
            ``player_key``, ``minutes``, ``starts``, and ideally ``source``.
        min_count: Pseudo-count target for shrinkage toward the prior.

    Returns:
        ``ChanceCalibration`` (identity when no snapshot rows are usable).
    """
    if min_count < 1:
        raise ValueError(f"min_count must be positive, got {min_count}")
    if stats.is_empty():
        return ChanceCalibration.identity()
    labelled = _status_at_deadline(view, stats)
    if labelled.is_empty():
        return ChanceCalibration.identity()
    if "source" in labelled.columns:
        labelled = labelled.filter(pl.col("source") == _SNAPSHOT)
    if labelled.is_empty():
        return ChanceCalibration.identity()
    chance_expr = (
        pl.coalesce(
            pl.col("chance_of_playing_this_round"),
            pl.col("chance_of_playing"),
        )
        if "chance_of_playing_this_round" in labelled.columns
        else pl.col("chance_of_playing")
    )
    # Unavailable with null chance → bucket 0; available null → drop (no flag).
    with_pct = labelled.with_columns(
        pl.when(chance_expr.is_not_null())
        .then(chance_expr)
        .when(pl.col("status").is_in(list(UNAVAILABLE)))
        .then(pl.lit(0.0))
        .otherwise(None)
        .alias("_chance_pct")
    ).filter(pl.col("_chance_pct").is_not_null())
    if with_pct.is_empty():
        return ChanceCalibration.identity()
    bucketed = with_pct.with_columns(
        ((pl.col("_chance_pct") / 25.0).round() * 25.0)
        .clip(0.0, 100.0)
        .cast(pl.Int64)
        .alias("bucket"),
        (pl.col("minutes") > 0).cast(pl.Float64).alias("played"),
        (pl.col("starts") > 0).cast(pl.Float64).alias("started"),
        (pl.col("minutes") >= 60).cast(pl.Float64).alias("long"),
    )
    play_gate: dict[int, float] = {}
    long_rate: dict[int, float] = {}
    for bucket in _BUCKETS:
        rows = bucketed.filter(pl.col("bucket") == bucket)
        n = rows.height
        if n == 0:
            continue
        emp_play = _series_mean(rows["played"])
        face = bucket / 100.0
        weight = min(1.0, n / float(min_count))
        play_gate[bucket] = weight * emp_play + (1.0 - weight) * face
        starters = rows.filter(pl.col("started") > 0.5)
        if starters.height > 0:
            long_rate[bucket] = _series_mean(starters["long"])
    baseline = long_rate.get(_FULL_BUCKET)
    if baseline is None or baseline <= 0.0:
        # Fall back to all fully-available starts if bucket 100 is empty.
        full = bucketed.filter(
            (pl.col("bucket") == _FULL_BUCKET) & (pl.col("started") > 0.5)
        )
        baseline = _series_mean(full["long"]) if full.height > 0 else None
    long_factor: dict[int, float] = {}
    if baseline is not None and baseline > 0.0:
        for bucket, rate in long_rate.items():
            if bucket >= _FULL_BUCKET:
                long_factor[bucket] = 1.0
                continue
            n_starts = bucketed.filter(
                (pl.col("bucket") == bucket) & (pl.col("started") > 0.5)
            ).height
            weight = min(1.0, n_starts / float(min_count))
            ratio = min(1.0, rate / baseline)
            long_factor[bucket] = weight * ratio + (1.0 - weight) * 1.0
    return ChanceCalibration(play_gate=play_gate, long_factor=long_factor)


def _series_mean(series: pl.Series) -> float:
    """Mean of a float series, treating an empty/null mean as 0."""
    value = series.mean()
    if value is None:
        return 0.0
    if isinstance(value, bool) or not isinstance(value, int | float):
        return 0.0
    return float(value)


def _status_at_deadline(view: PITView, stats: pl.DataFrame) -> pl.DataFrame:
    """Attach PIT status / chance columns as of each appearance's GW deadline."""
    deadlines = (
        view.latest("gameweeks", ["season", "gw"])
        .select("season", "gw", "deadline_time")
        .collect()
    )
    status_lf = view.table("player_status")
    status_cols = [
        "season",
        "player_key",
        "status",
        "chance_of_playing",
        "available_at",
    ]
    schema_names = status_lf.collect_schema().names()
    if "chance_of_playing_this_round" in schema_names:
        status_cols.append("chance_of_playing_this_round")
    status = (
        status_lf.select(status_cols)
        .collect()
        .sort(["season", "player_key", "available_at"])
    )
    with_deadline = stats.join(deadlines, on=["season", "gw"], how="left")
    if status.is_empty() or "deadline_time" not in with_deadline.columns:
        return stats.head(0)
    ordered = with_deadline.sort(["season", "player_key", "deadline_time"])
    return ordered.join_asof(
        status,
        left_on="deadline_time",
        right_on="available_at",
        by=["season", "player_key"],
        strategy="backward",
        check_sortedness=False,
    )
