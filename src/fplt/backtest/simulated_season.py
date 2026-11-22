"""Simulated-season decision evaluation with CRN and paired SEs (N6).

Evaluation north star §2 (Uncertainty): a single real season is one
path-dependent draw. This harness draws many outcome paths from a held-out
truth intensity (or accepts pre-drawn actuals), runs each decision policy on
**the same** path (common random numbers), and reports Monte Carlo standard
errors on paired policy differences for season points and, optionally, share
of attainable points (N5).

Reuses ``simulate_decisions`` / ``summarise_decisions`` and the constrained
hindsight oracle — the same decision sim as promote-eval's decisions suite.
Does not change ``PRODUCTION_MODELS`` or fetch FPL Review.
"""

from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import polars as pl

from fplt.optimise.attainable import attainable_share, simulate_hindsight_oracle
from fplt.optimise.decisions import simulate_decisions
from fplt.optimise.realised import summarise_decisions
from fplt.optimise.state import SquadState
from fplt.schemas.simulated_season import (
    PolicyDiffSESchema,
    SimulatedPathActualsSchema,
    SimulatedSeasonPathSchema,
    TruthRateSchema,
)

ORACLE_PATH_MODEL_ID = "hindsight_oracle_sim_v1"


@dataclass(frozen=True)
class DecisionPolicy:
    """One decision arm: opaque planner ``xpts`` under a stable id."""

    policy_id: str
    predictions: pl.DataFrame


@dataclass(frozen=True)
class SimulatedSeasonReport:
    """Path-level totals plus pairwise SEs for a simulated-season run."""

    paths: pl.DataFrame
    diffs: pl.DataFrame
    n_paths: int
    seed: int | None
    with_attainable: bool


def path_seed(seed: int, path_id: int) -> int:
    """Derive a reproducible per-path seed from the eval seed and path index."""
    payload = f"sim-season|{int(seed)}|{int(path_id)}".encode()
    digest = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return digest & 0x7FFF_FFFF_FFFF_FFFF


def draw_truth_paths(
    rates: pl.DataFrame,
    n_paths: int,
    seed: int,
) -> pl.DataFrame:
    """Draw ``n_paths`` CRN-ready actuals frames from held-out truth rates.

    Each path uses ``path_seed(seed, path_id)`` so re-running with the same
    seed regenerates identical paths, and every policy that consumes a path
    sees the same outcomes (common random numbers).

    Args:
        rates: Rows matching ``TruthRateSchema`` (``rate`` = Poisson mean for
            points; optional ``minutes_rate`` defaults to 90 when ``rate>0``).
        n_paths: Number of independent season paths (``path_id`` 0..n-1).
        seed: Master eval seed.

    Returns:
        A frame matching ``SimulatedPathActualsSchema``.

    Raises:
        ValueError: ``n_paths`` is not positive.
    """
    if n_paths < 1:
        raise ValueError("n_paths must be >= 1")
    prepared = rates
    if "minutes_rate" not in prepared.columns:
        prepared = prepared.with_columns(
            pl.lit(None).cast(pl.Float64).alias("minutes_rate")
        )
    checked = TruthRateSchema.validate(prepared)
    assert isinstance(checked, pl.DataFrame)
    rows: list[dict[str, object]] = []
    base = checked.select("gw", "player_key", "rate", "minutes_rate").sort(
        "gw", "player_key"
    )
    for path_id in range(n_paths):
        rng = np.random.default_rng(path_seed(seed, path_id))
        for record in base.iter_rows(named=True):
            rate = float(record["rate"])
            pts = int(rng.poisson(rate)) if rate > 0 else 0
            minutes_rate = record["minutes_rate"]
            if minutes_rate is None:
                minutes = 90 if rate > 0 else 0
            else:
                minutes = int(min(90, max(0, round(float(minutes_rate)))))
            rows.append(
                {
                    "path_id": path_id,
                    "gw": int(record["gw"]),
                    "player_key": int(record["player_key"]),
                    "actual_pts": pts,
                    "actual_minutes": minutes,
                }
            )
    return _path_actuals(pl.DataFrame(rows) if rows else _empty_path_actuals())


def paired_policy_se(
    paths: pl.DataFrame,
    *,
    control_id: str,
    treatment_id: str,
    metric: str = "season_points",
) -> pl.DataFrame:
    """Compute mean paired difference and Monte Carlo SE for one metric.

    Args:
        paths: Rows matching ``SimulatedSeasonPathSchema``.
        control_id: Baseline ``policy_id``.
        treatment_id: Contrasting ``policy_id``.
        metric: ``season_points`` or ``share_of_attainable``.

    Returns:
        A one-row frame matching ``PolicyDiffSESchema``.

    Raises:
        ValueError: Unknown metric, missing policy, or empty paired set.
    """
    checked = SimulatedSeasonPathSchema.validate(paths)
    assert isinstance(checked, pl.DataFrame)
    if metric == "season_points":
        column = "total_points"
    elif metric == "share_of_attainable":
        column = "share_of_attainable"
    else:
        raise ValueError(
            f"metric must be 'season_points' or 'share_of_attainable', got {metric!r}"
        )
    control = checked.filter(pl.col("policy_id") == control_id).select(
        "path_id", pl.col(column).alias("control")
    )
    treatment = checked.filter(pl.col("policy_id") == treatment_id).select(
        "path_id", pl.col(column).alias("treatment")
    )
    if control.is_empty():
        raise ValueError(f"no paths for control_id={control_id!r}")
    if treatment.is_empty():
        raise ValueError(f"no paths for treatment_id={treatment_id!r}")
    joined = control.join(treatment, on="path_id", how="inner").sort("path_id")
    if joined.is_empty():
        raise ValueError("no overlapping CRN paths for the policy pair")
    if column == "share_of_attainable" and joined["control"].null_count():
        raise ValueError(
            "share_of_attainable is null; re-run with with_attainable=True"
        )
    diffs = (joined["treatment"] - joined["control"]).to_numpy()
    n = int(diffs.shape[0])
    mean_diff = float(np.mean(diffs))
    if n == 1:
        sd = 0.0
        se = 0.0
    else:
        sd = float(np.std(diffs, ddof=1))
        se = sd / math.sqrt(n)
    return _diffs(
        pl.DataFrame(
            {
                "metric": [metric],
                "control_id": [control_id],
                "treatment_id": [treatment_id],
                "n_paths": [n],
                "mean_diff": [mean_diff],
                "se": [se],
                "sd": [sd],
            }
        )
    )


def policy_diff_board(
    paths: pl.DataFrame,
    *,
    pairs: Sequence[tuple[str, str]] | None = None,
    metrics: Sequence[str] = ("season_points",),
) -> pl.DataFrame:
    """Build pairwise SE rows for every requested policy pair and metric.

    Args:
        paths: Path-level policy totals.
        pairs: ``(control_id, treatment_id)`` pairs. Defaults to every ordered
            pair of distinct ``policy_id`` values present.
        metrics: Metrics to difference (``season_points`` and/or
            ``share_of_attainable``).

    Returns:
        Rows matching ``PolicyDiffSESchema``.

    Raises:
        ValueError: Fewer than two policies when ``pairs`` is omitted.
    """
    checked = SimulatedSeasonPathSchema.validate(paths)
    assert isinstance(checked, pl.DataFrame)
    policy_ids = sorted(checked["policy_id"].unique().to_list())
    if pairs is not None:
        chosen = list(pairs)
    else:
        chosen = [
            (left, right)
            for i, left in enumerate(policy_ids)
            for right in policy_ids[i + 1 :]
        ]
    if not chosen:
        raise ValueError("need at least one policy pair")
    frames = [
        paired_policy_se(
            checked,
            control_id=control,
            treatment_id=treatment,
            metric=metric,
        )
        for control, treatment in chosen
        for metric in metrics
    ]
    return _diffs(pl.concat(frames))


def run_simulated_season_eval(
    candidates: pl.DataFrame,
    policies: Sequence[DecisionPolicy],
    state: SquadState,
    *,
    season: str,
    gameweeks: Sequence[int],
    horizon: int,
    n_paths: int | None = None,
    seed: int | None = None,
    truth_rates: pl.DataFrame | None = None,
    path_actuals: pl.DataFrame | None = None,
    with_attainable: bool = False,
    pairs: Sequence[tuple[str, str]] | None = None,
) -> SimulatedSeasonReport:
    """Run every policy on shared outcome paths and report paired SEs.

    Supply either ``path_actuals`` (pre-drawn / existing sim draws labelled by
    ``path_id``) **or** ``truth_rates`` + ``n_paths`` + ``seed`` to draw Poisson
    paths from a held-out truth model.

    Args:
        candidates: ``gw``, ``player_key``, ``position``, ``team_key``, ``price``.
        policies: Decision arms with planner predictions.
        state: Starting squad (typically empty owned, bank 1000).
        season: Season label written onto decision rows.
        gameweeks: Deadlines to play, in order.
        horizon: Planner look-ahead.
        n_paths: Required when drawing from ``truth_rates``.
        seed: Master seed for path draws.
        truth_rates: Held-out intensities (``TruthRateSchema``).
        path_actuals: Pre-drawn paths (``SimulatedPathActualsSchema``).
        with_attainable: When true, run the N5 hindsight oracle on each path
            and attach share of attainable.
        pairs: Optional explicit ``(control, treatment)`` pairs for the SE
            board; default is every unordered pair once (control < treatment
            lexicographically via ``policy_diff_board``).

    Returns:
        ``SimulatedSeasonReport`` with path totals and pairwise SEs.

    Raises:
        ValueError: Bad inputs, empty policies/gameweeks, or missing draw args.
    """
    weeks = list(gameweeks)
    if not weeks:
        raise ValueError("gameweeks must be non-empty")
    if not policies:
        raise ValueError("policies must be non-empty")
    ids = [policy.policy_id for policy in policies]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate policy_id: {ids}")

    drawn = _resolve_paths(
        truth_rates=truth_rates,
        path_actuals=path_actuals,
        n_paths=n_paths,
        seed=seed,
    )
    path_ids = sorted(drawn["path_id"].unique().to_list())
    metrics: list[str] = ["season_points"]
    if with_attainable:
        metrics.append("share_of_attainable")

    rows: list[dict[str, object]] = []
    for path_id in path_ids:
        actuals = (
            drawn.filter(pl.col("path_id") == path_id)
            .drop("path_id")
            .select("gw", "player_key", "actual_pts", "actual_minutes")
        )
        attainable: int | None = None
        if with_attainable:
            oracle = simulate_hindsight_oracle(
                candidates,
                actuals,
                state,
                season=season,
                gameweeks=weeks,
                horizon=horizon,
                model_id=ORACLE_PATH_MODEL_ID,
            )
            oracle_summary = summarise_decisions(oracle)
            if oracle_summary.is_empty():
                attainable = 0
            else:
                attainable = int(oracle_summary["total_points"][0])
        for policy in policies:
            decisions = simulate_decisions(
                candidates,
                policy.predictions,
                actuals,
                state,
                model_id=policy.policy_id,
                season=season,
                gameweeks=weeks,
                horizon=horizon,
            )
            summary = summarise_decisions(decisions)
            total = int(summary["total_points"][0]) if not summary.is_empty() else 0
            n_gw = int(summary["gameweeks"][0]) if not summary.is_empty() else 0
            share = None if attainable is None else attainable_share(total, attainable)
            rows.append(
                {
                    "path_id": int(path_id),
                    "policy_id": policy.policy_id,
                    "season": season,
                    "total_points": total,
                    "gameweeks": n_gw,
                    "attainable_points": attainable,
                    "share_of_attainable": share,
                }
            )

    path_frame = _season_paths(pl.DataFrame(rows) if rows else _empty_season_paths())
    diffs = policy_diff_board(path_frame, pairs=pairs, metrics=tuple(metrics))
    return SimulatedSeasonReport(
        paths=path_frame,
        diffs=diffs,
        n_paths=len(path_ids),
        seed=seed,
        with_attainable=with_attainable,
    )


def format_simulated_season_report(report: SimulatedSeasonReport) -> str:
    """Render a compact markdown scoreboard of paired policy SEs."""
    lines = [
        f"# Simulated-season decision eval (n_paths={report.n_paths}",
    ]
    if report.seed is not None:
        lines[0] += f", seed={report.seed}"
    lines[0] += ")"
    lines.append("")
    lines.append("| metric | control | treatment | mean_diff | SE | sd | n |")
    lines.append("|---|---|---|---:|---:|---:|---:|")
    for row in report.diffs.iter_rows(named=True):
        lines.append(
            "| {metric} | {control} | {treatment} | {mean:.3f} | {se:.3f} | "
            "{sd:.3f} | {n} |".format(
                metric=row["metric"],
                control=row["control_id"],
                treatment=row["treatment_id"],
                mean=float(row["mean_diff"]),
                se=float(row["se"]),
                sd=float(row["sd"]),
                n=int(row["n_paths"]),
            )
        )
    if report.with_attainable:
        lines.extend(["", "_Share-of-attainable diffs use the N5 path oracle._"])
    return "\n".join(lines)


def mean_policy_totals(paths: pl.DataFrame) -> Mapping[str, float]:
    """Mean season points per policy across paths (descriptive, not an SE)."""
    checked = SimulatedSeasonPathSchema.validate(paths)
    assert isinstance(checked, pl.DataFrame)
    grouped = (
        checked.group_by("policy_id")
        .agg(pl.col("total_points").mean().alias("mean_points"))
        .sort("policy_id")
    )
    return {
        str(row["policy_id"]): float(row["mean_points"])
        for row in grouped.iter_rows(named=True)
    }


def _resolve_paths(
    *,
    truth_rates: pl.DataFrame | None,
    path_actuals: pl.DataFrame | None,
    n_paths: int | None,
    seed: int | None,
) -> pl.DataFrame:
    if path_actuals is not None and truth_rates is not None:
        raise ValueError("pass path_actuals or truth_rates, not both")
    if path_actuals is not None:
        return _path_actuals(path_actuals)
    if truth_rates is None:
        raise ValueError("provide truth_rates (+ n_paths, seed) or path_actuals")
    if n_paths is None or seed is None:
        raise ValueError("drawing from truth_rates requires n_paths and seed")
    return draw_truth_paths(truth_rates, n_paths, seed)


def _path_actuals(frame: pl.DataFrame) -> pl.DataFrame:
    checked = SimulatedPathActualsSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked


def _season_paths(frame: pl.DataFrame) -> pl.DataFrame:
    checked = SimulatedSeasonPathSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked


def _diffs(frame: pl.DataFrame) -> pl.DataFrame:
    checked = PolicyDiffSESchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked


def _empty_path_actuals() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "path_id": pl.Int64(),
            "gw": pl.Int64(),
            "player_key": pl.Int64(),
            "actual_pts": pl.Int64(),
            "actual_minutes": pl.Int64(),
        }
    )


def _empty_season_paths() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "path_id": pl.Int64(),
            "policy_id": pl.Utf8(),
            "season": pl.Utf8(),
            "total_points": pl.Int64(),
            "gameweeks": pl.Int64(),
            "attainable_points": pl.Int64(),
            "share_of_attainable": pl.Float64(),
        }
    )
