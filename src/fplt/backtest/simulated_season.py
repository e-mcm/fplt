"""Simulated-season decision evaluation with CRN and paired SEs (N6).

Evaluation north star §2 (Uncertainty): a single real season is one
path-dependent draw. This harness runs each decision policy on **shared**
outcome paths (common random numbers) and reports Monte Carlo standard
errors on paired policy differences for season points and, optionally,
share of attainable points (N5).

**Truth paths.** Prefer model-based **joint** draws (scoreline-correlated
player outcomes) via ``path_actuals_from_joint_sims`` / pre-drawn
``path_actuals``. Independent Poisson rates (``truth_rates_from_actuals`` /
``draw_truth_paths``) remain for tiny wiring smokes only — they are not a
generative truth model and must not back promote SE claims.

**Attainable oracle.** When ``with_attainable=True``, freeze horizon,
decay, and chip arm with :class:`~fplt.optimise.attainable.OraclePin` so
every policy shares one denominator on each path.

Supports the C6 chip board arms (``off`` / ``rule_based`` / ``optimised``)
via ``DecisionPolicy.chip_arm`` and the same ``chip_windows`` / scheduler
contract as ``simulate_decisions``. Chips-off remains the default.

Reuses ``simulate_decisions`` / ``summarise_decisions`` and the constrained
hindsight oracle — the same decision sim as promote-eval's decisions suite.
Does not change ``PRODUCTION_MODELS`` or fetch FPL Review.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import polars as pl

from fplt.backtest.ledger import read_predictions
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.spec import load_model_specs
from fplt.optimise.attainable import (
    OraclePin,
    attainable_share,
    resolve_oracle_pin,
    simulate_hindsight_oracle,
)
from fplt.optimise.chip_windows import ChipArm, ChipUseWindows, chip_arm_model_id
from fplt.optimise.decisions import (
    actual_rows_from_ledger,
    candidates_as_of,
    prediction_rows_from_ledger,
    simulate_decisions,
)
from fplt.optimise.realised import summarise_decisions
from fplt.optimise.state import SquadState, load_squad_state
from fplt.schemas.simulated_season import (
    JointTruthSimSchema,
    PolicyDiffSESchema,
    SimulatedPathActualsSchema,
    SimulatedSeasonPathSchema,
    TruthRateSchema,
)
from fplt.types import GWKey
from fplt.utils.frames import require_dataframe, require_not_none

ORACLE_PATH_MODEL_ID = "hindsight_oracle_sim_v1"
_CHIP_ARMS: tuple[ChipArm, ...] = ("off", "rule_based", "optimised")
logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class DecisionPolicy:
    """One decision arm: opaque planner ``xpts`` under a stable id.

    ``chip_arm`` defaults to chips-off. Non-off arms use the same windows /
    scheduler contract as the C6 chip decision board
    (``simulate_decisions(..., chip_arm=...)``).

    ``force_play`` is forwarded to the optimised chip scheduler (default on).
    ``horizon`` overrides the eval-level look-ahead when set — needed to CRN-
    compare H=3 force-on vs H=8 force-off in one N6 run (promote case E).
    """

    policy_id: str
    predictions: pl.DataFrame
    chip_arm: ChipArm = "off"
    chip_windows: ChipUseWindows | None = None
    chip_rules_season: str | None = None
    force_play: bool = True
    horizon: int | None = None


@dataclass(frozen=True)
class SimulatedSeasonReport:
    """Path-level totals plus pairwise SEs for a simulated-season run."""

    paths: pl.DataFrame
    diffs: pl.DataFrame
    n_paths: int
    seed: int | None
    with_attainable: bool
    oracle_pin: OraclePin | None = None


def path_seed(seed: int, path_id: int) -> int:
    """Derive a reproducible per-path seed from the eval seed and path index."""
    payload = f"sim-season|{int(seed)}|{int(path_id)}".encode()
    digest = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return digest & 0x7FFF_FFFF_FFFF_FFFF


def path_actuals_from_joint_sims(frame: pl.DataFrame) -> pl.DataFrame:
    """Convert joint truth sims into CRN path actuals.

    Maps ``sim`` → ``path_id``, ``points`` → ``actual_pts`` (rounded), and
    ``minutes`` → ``actual_minutes``. Rows that share ``sim`` are treated as
    one common-random-number season path (joint within each ``gw``).

    Args:
        frame: Rows matching ``JointTruthSimSchema``.

    Returns:
        A frame matching ``SimulatedPathActualsSchema``.

    Raises:
        ValueError: Validation fails or the frame is empty.
    """
    if frame.is_empty():
        raise ValueError("joint truth frame must be non-empty")
    checked = JointTruthSimSchema.validate(frame)
    checked = require_dataframe(checked)
    mapped = checked.select(
        pl.col("sim").cast(pl.Int64).alias("path_id"),
        pl.col("gw").cast(pl.Int64),
        pl.col("player_key").cast(pl.Int64),
        pl.col("points").round(0).cast(pl.Int64).clip(0, None).alias("actual_pts"),
        pl.col("minutes").cast(pl.Int64).clip(0, 90).alias("actual_minutes"),
    )
    return _path_actuals(mapped)


def draw_truth_paths(
    rates: pl.DataFrame,
    n_paths: int,
    seed: int,
) -> pl.DataFrame:
    """Draw independent Poisson paths from rates (smoke / legacy only).

    Prefer :func:`path_actuals_from_joint_sims` for SE claims. Independent
    Poisson noise around realised (or other) rates does not reproduce
    scoreline correlation and must not back a promote decision SE.

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
    checked = require_dataframe(checked)
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
    checked = require_dataframe(checked)
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
    checked = require_dataframe(checked)
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


def truth_rates_from_actuals(actuals: pl.DataFrame) -> pl.DataFrame:
    """Build independent Poisson rates from observed player–GW actuals.

    ``actual_pts`` becomes the Poisson mean; ``actual_minutes`` (when present)
    becomes ``minutes_rate``. **Smoke / legacy only** — Poisson around
    realised points is not a model-based joint truth. Prefer joint sims via
    ``path_actuals_from_joint_sims`` for SE claims.

    Args:
        actuals: ``gw``, ``player_key``, ``actual_pts``, optional
            ``actual_minutes``.

    Returns:
        Rows matching ``TruthRateSchema``.

    Raises:
        ValueError: Required columns are missing or the frame is empty.
    """
    required = {"gw", "player_key", "actual_pts"}
    missing = required - set(actuals.columns)
    if missing:
        raise ValueError(f"actuals missing columns: {sorted(missing)}")
    if actuals.is_empty():
        raise ValueError("actuals must be non-empty")
    cols: list[pl.Expr] = [
        pl.col("gw").cast(pl.Int64),
        pl.col("player_key").cast(pl.Int64),
        pl.col("actual_pts").cast(pl.Float64).clip(0.0, None).alias("rate"),
    ]
    if "actual_minutes" in actuals.columns:
        cols.append(
            pl.col("actual_minutes")
            .cast(pl.Float64)
            .clip(0.0, 90.0)
            .alias("minutes_rate")
        )
    else:
        cols.append(pl.lit(None).cast(pl.Float64).alias("minutes_rate"))
    frame = actuals.select(cols)
    # One rate per player–GW (DGW actuals should already be collapsed).
    frame = frame.group_by("gw", "player_key").agg(
        pl.col("rate").sum(),
        pl.col("minutes_rate").max(),
    )
    checked = TruthRateSchema.validate(frame)
    checked = require_dataframe(checked)
    return checked


def chip_arm_policies(
    predictions: pl.DataFrame,
    *,
    base_model_id: str,
    arms: Sequence[ChipArm] = _CHIP_ARMS,
    chip_rules_season: str | None = None,
    chip_windows: Mapping[ChipArm, ChipUseWindows] | None = None,
    force_play: bool = True,
) -> list[DecisionPolicy]:
    """Build decision policies for the C6 chip arms on shared predictions.

    Ledger-style ``policy_id`` values match the chip board: chips-off keeps
    ``base_model_id``; other arms use ``chip_arm_model_id``.

    Args:
        predictions: Shared planner ``xpts`` for every arm.
        base_model_id: Model id without a ``::chip_`` suffix.
        arms: Chip arms to include (default all three).
        chip_rules_season: Chip-rules year for non-off arms.
        chip_windows: Optional precomputed windows per arm (otherwise
            rule-based builds landmarks and optimised schedules weekly).
        force_play: Forwarded to optimised arms (default true; ablation off).

    Returns:
        One ``DecisionPolicy`` per arm.

    Raises:
        ValueError: ``arms`` is empty or contains an unknown arm.
    """
    if not arms:
        raise ValueError("arms must be non-empty")
    policies: list[DecisionPolicy] = []
    for arm in arms:
        if arm not in _CHIP_ARMS:
            raise ValueError(f"unknown chip_arm {arm!r}")
        policy_id = (
            base_model_id if arm == "off" else chip_arm_model_id(base_model_id, arm)
        )
        windows = None if chip_windows is None else chip_windows.get(arm)
        policies.append(
            DecisionPolicy(
                policy_id=policy_id,
                predictions=predictions,
                chip_arm=arm,
                chip_windows=windows,
                chip_rules_season=chip_rules_season,
                force_play=force_play,
            )
        )
    return policies


def chip_arm_control_pairs(policies: Sequence[DecisionPolicy]) -> list[tuple[str, str]]:
    """Pair each non-off chip arm against the chips-off control when present.

    Falls back to every unordered pair when no chips-off policy is in the set.
    """
    by_arm = {policy.chip_arm: policy.policy_id for policy in policies}
    control = by_arm.get("off")
    if control is None:
        return []
    return [
        (control, policy.policy_id)
        for policy in policies
        if policy.chip_arm != "off" and policy.policy_id != control
    ]


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
    oracle_pin: OraclePin | None = None,
    pairs: Sequence[tuple[str, str]] | None = None,
    fixtures: pl.DataFrame | None = None,
) -> SimulatedSeasonReport:
    """Run every policy on shared outcome paths and report paired SEs.

    Supply either ``path_actuals`` (joint / pre-drawn sims labelled by
    ``path_id``) **or** ``truth_rates`` + ``n_paths`` + ``seed`` to draw
    independent Poisson paths (smoke only).

    Args:
        candidates: ``gw``, ``player_key``, ``position``, ``team_key``, ``price``.
        policies: Decision arms with planner predictions (and optional
            ``chip_arm`` / windows).
        state: Starting squad (typically empty owned, bank 1000).
        season: Season label written onto decision rows.
        gameweeks: Deadlines to play, in order.
        horizon: Default planner look-ahead; each policy may override via
            ``DecisionPolicy.horizon``.
        n_paths: Required when drawing from ``truth_rates``.
        seed: Master seed for path draws.
        truth_rates: Independent Poisson intensities (``TruthRateSchema``;
            smoke / legacy only).
        path_actuals: Pre-drawn paths (``SimulatedPathActualsSchema``),
            preferably from joint truth sims.
        with_attainable: When true, run the N5 hindsight oracle on each path
            under a single :class:`OraclePin` and attach share of attainable.
        oracle_pin: Explicit H / decay / chip_arm for the path oracle.
            Required when ``with_attainable`` and policies use divergent
            horizons; otherwise defaults to eval ``horizon``, ``state.decay``,
            chips-off.
        pairs: Optional explicit ``(control, treatment)`` pairs for the SE
            board; default is every unordered pair once (control < treatment
            lexicographically via ``policy_diff_board``), unless chip arms
            include chips-off — then each non-off arm is paired against off.
        fixtures: Staged fixtures for rule-based landmarks / optimised
            ``schedule_chip_windows`` (required when any policy or the
            oracle pin is not chips-off).

    Returns:
        ``SimulatedSeasonReport`` with path totals and pairwise SEs.

    Raises:
        ValueError: Bad inputs, empty policies/gameweeks, missing fixtures for
            chip arms, divergent horizons without a pin, or missing draw args.
    """
    weeks = list(gameweeks)
    if not weeks:
        raise ValueError("gameweeks must be non-empty")
    if not policies:
        raise ValueError("policies must be non-empty")
    ids = [policy.policy_id for policy in policies]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate policy_id: {ids}")

    resolved_pin: OraclePin | None = None
    if with_attainable:
        resolved_pin = resolve_oracle_pin(
            horizon=horizon,
            state=state,
            policy_horizons=[policy.horizon for policy in policies],
            pin=oracle_pin,
        )

    needs_fixtures = any(policy.chip_arm != "off" for policy in policies)
    if resolved_pin is not None and resolved_pin.chip_arm != "off":
        needs_fixtures = True
    if needs_fixtures and (fixtures is None or fixtures.is_empty()):
        raise ValueError("chip arms require non-empty fixtures")

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
    chosen_pairs = list(pairs) if pairs is not None else None
    if chosen_pairs is None:
        chip_pairs = chip_arm_control_pairs(policies)
        if chip_pairs:
            chosen_pairs = chip_pairs

    oracle_state = state
    if resolved_pin is not None and resolved_pin.decay != float(state.decay):
        oracle_state = state.model_copy(update={"decay": resolved_pin.decay})

    rows: list[dict[str, object]] = []
    for path_id in path_ids:
        actuals = (
            drawn.filter(pl.col("path_id") == path_id)
            .drop("path_id")
            .select("gw", "player_key", "actual_pts", "actual_minutes")
        )
        attainable: int | None = None
        if with_attainable:
            resolved_pin = require_not_none(resolved_pin, what="resolved_pin")
            oracle = simulate_hindsight_oracle(
                candidates,
                actuals,
                oracle_state,
                season=season,
                gameweeks=weeks,
                horizon=resolved_pin.horizon,
                model_id=ORACLE_PATH_MODEL_ID,
                chip_arm=resolved_pin.chip_arm,
                fixtures=fixtures,
                force_play=resolved_pin.force_play,
            )
            oracle_summary = summarise_decisions(oracle)
            if oracle_summary.is_empty():
                attainable = 0
            else:
                attainable = int(oracle_summary["total_points"][0])
        for policy in policies:
            policy_horizon = horizon if policy.horizon is None else policy.horizon
            decisions = simulate_decisions(
                candidates,
                policy.predictions,
                actuals,
                state,
                model_id=policy.policy_id,
                season=season,
                gameweeks=weeks,
                horizon=policy_horizon,
                chip_arm=policy.chip_arm,
                chip_rules_season=policy.chip_rules_season,
                chip_windows=policy.chip_windows,
                fixtures=fixtures,
                force_play=policy.force_play,
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
    diffs = policy_diff_board(path_frame, pairs=chosen_pairs, metrics=tuple(metrics))
    return SimulatedSeasonReport(
        paths=path_frame,
        diffs=diffs,
        n_paths=len(path_ids),
        seed=seed,
        with_attainable=with_attainable,
        oracle_pin=resolved_pin,
    )


def run_chip_arm_sim_season(
    data_dir: Path,
    model_name: str,
    season: str,
    from_gw: int,
    to_gw: int,
    horizon: int,
    *,
    arms: Sequence[ChipArm] = _CHIP_ARMS,
    chip_rules_season: str | None = None,
    n_paths: int = 5,
    seed: int = 0,
    with_attainable: bool = False,
    force_play: bool = True,
) -> SimulatedSeasonReport:
    """Run chip-arm simulated-season SE from staged data + the prediction ledger.

    Builds independent Poisson rates from ledger actuals (smoke only — not
    joint truth), then CRN-runs the requested chip arms. Intended for short
    wiring smokes (few GWs × small ``n_paths``), not promote SE boards.
    Joint truth paths land via ``path_actuals_from_joint_sims`` (follow-up:
    held-out layered sampler).

    Args:
        data_dir: Root with ``staged/`` and ``ledger/``.
        model_name: YAML spec name.
        season: Fixture / prediction season.
        from_gw: First gameweek.
        to_gw: Last gameweek, inclusive.
        horizon: Planner horizon (does not flip ``DEFAULT_PLAN_HORIZON``).
        arms: Chip arms (default all three).
        chip_rules_season: Chip-rules year (default ``season``).
        n_paths: Monte Carlo path count.
        seed: Master draw seed.
        with_attainable: Optional N5 share SE under a chips-off
            :class:`OraclePin` (eval horizon + state decay).
        force_play: Optimised-arm half-window packing (default true; CLI
            ``--no-force-play`` sets false). Does not flip the live default.

    Returns:
        ``SimulatedSeasonReport`` with path totals and paired SEs.

    Raises:
        KeyError: Unknown model spec or missing deadline.
        ValueError: Empty arms, missing ledger actuals, or chip fixtures.
    """
    if not arms:
        raise ValueError("arms must be non-empty")
    spec = load_model_specs()[model_name]
    rules_season = chip_rules_season if chip_rules_season is not None else season
    squad_rules = load_squad_state()
    draft = squad_rules.model_copy(
        update={"owned": [], "bank": 1000, "free_transfers": 1}
    )
    store = DataStore(data_dir)
    gameweeks = list(range(from_gw, to_gw + 1))
    pools = [
        candidates_as_of(PITData(store).view(GWKey(season, gw))).with_columns(
            pl.lit(gw).alias("gw")
        )
        for gw in gameweeks
    ]
    ledger = read_predictions(data_dir)
    predictions = prediction_rows_from_ledger(ledger, spec.model_id, season)
    actuals = actual_rows_from_ledger(ledger, spec.model_id, season)
    if actuals.is_empty():
        raise ValueError(
            f"no ledger actuals for model_id={spec.model_id!r} season={season!r}"
        )
    week_actuals = actuals.filter(pl.col("gw").is_in(gameweeks))
    if week_actuals.is_empty():
        raise ValueError(
            f"no ledger actuals in GW{from_gw}–{to_gw} for {spec.model_id!r}"
        )
    fixtures = store.read("fixtures").filter(pl.col("season") == season)
    policies = chip_arm_policies(
        predictions,
        base_model_id=spec.model_id,
        arms=arms,
        chip_rules_season=rules_season,
        force_play=force_play,
    )
    logger.info(
        "chip-arm sim-season model=%s season=%s gw=%s-%s arms=%s "
        "force_play=%s n_paths=%s",
        model_name,
        season,
        from_gw,
        to_gw,
        list(arms),
        force_play,
        n_paths,
    )
    return run_simulated_season_eval(
        pl.concat(pools),
        policies,
        draft,
        season=season,
        gameweeks=gameweeks,
        horizon=horizon,
        n_paths=n_paths,
        seed=seed,
        truth_rates=truth_rates_from_actuals(week_actuals),
        with_attainable=with_attainable,
        fixtures=fixtures,
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
        pin = report.oracle_pin
        if pin is None:
            lines.extend(["", "_Share-of-attainable diffs use the N5 path oracle._"])
        else:
            lines.extend(
                [
                    "",
                    (
                        f"_Share-of-attainable oracle pin: H={pin.horizon}, "
                        f"decay={pin.decay}, chip_arm={pin.chip_arm}_"
                    ),
                ]
            )
    return "\n".join(lines)


def mean_policy_totals(paths: pl.DataFrame) -> Mapping[str, float]:
    """Mean season points per policy across paths (descriptive, not an SE)."""
    checked = SimulatedSeasonPathSchema.validate(paths)
    checked = require_dataframe(checked)
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
    checked = require_dataframe(checked)
    return checked


def _season_paths(frame: pl.DataFrame) -> pl.DataFrame:
    checked = SimulatedSeasonPathSchema.validate(frame)
    checked = require_dataframe(checked)
    return checked


def _diffs(frame: pl.DataFrame) -> pl.DataFrame:
    checked = PolicyDiffSESchema.validate(frame)
    checked = require_dataframe(checked)
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
