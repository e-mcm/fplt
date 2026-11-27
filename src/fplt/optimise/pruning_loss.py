"""Pruning-loss harness: measure week-1 action cost of candidate-pool size.

Builds a reference solve (full pool, long time limit, ``mip_rel_gap=0``),
then sweeps target pool sizes and the current ``DEFAULT_PRUNE_POLICY``.
Primary metric is week-1 action agreement (transfers, XI, captain, vice);
objective gap, solve time, and MIP status are secondary.

A future prune-policy change is an equivalence check: identical week-1
actions versus the reference (or versus the previous policy) via
:func:`check_week1_equivalence`. Do not assert objective quality in pytest.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import polars as pl

from fplt.optimise.candidate_prune import (
    DEFAULT_PRUNE_POLICY,
    DEFAULT_TARGET_POOL,
    MIN_POS_POOL,
    CandidatePrunePolicy,
)
from fplt.optimise.highs_options import HighsSolveOptions
from fplt.optimise.plan import GameweekPlan, plan_squad
from fplt.optimise.state import OwnedPlayer, SquadState, rebuild_squad_state
from fplt.schemas.pruning_loss import PruningLossRowSchema
from fplt.utils.frames import require_dataframe

_log = logging.getLogger(__name__)

# Pool sizes to sweep (§3.1). ``current`` is recorded as a separate arm.
DEFAULT_POOL_TARGETS: tuple[int, ...] = (80, 100, 130, 160, 200, 300)

# Reference solve: long wall clock, prove optimality when it finishes.
REFERENCE_HIGHS_OPTIONS = HighsSolveOptions(
    time_limit=3600.0,
    mip_rel_gap=0.0,
    accept_time_limit=True,
)

# Shorter default for non-reference arms (still gap 0).
SWEEP_HIGHS_OPTIONS = HighsSolveOptions(
    time_limit=600.0,
    mip_rel_gap=0.0,
    accept_time_limit=True,
)

SCENARIO_TAGS: tuple[str, ...] = (
    "early_season",
    "dgw_runup",
    "rebuild",
    "mid_banked_fts",
)


@dataclass(frozen=True)
class Week1Actions:
    """Executable first-step decisions compared for pruning loss."""

    transfers_in: tuple[int, ...]
    transfers_out: tuple[int, ...]
    xi: tuple[int, ...]
    captain: int | None
    vice: int | None

    def mismatches(self, other: Week1Actions) -> tuple[str, ...]:
        """Field names that differ from ``other``."""
        fields: list[str] = []
        if self.transfers_in != other.transfers_in:
            fields.append("transfers_in")
        if self.transfers_out != other.transfers_out:
            fields.append("transfers_out")
        if self.xi != other.xi:
            fields.append("xi")
        if self.captain != other.captain:
            fields.append("captain")
        if self.vice != other.vice:
            fields.append("vice")
        return tuple(fields)


@dataclass(frozen=True)
class Week1EquivalenceResult:
    """Equivalence-style readout for a prune-policy change (§3.3)."""

    identical: bool
    reference: Week1Actions
    candidate: Week1Actions
    mismatches: tuple[str, ...]


@dataclass(frozen=True)
class PruningScenario:
    """One squad state + horizon points for the pruning-loss sweep."""

    scenario_id: str
    tag: str
    players: pl.DataFrame
    points: pl.DataFrame
    state: SquadState
    horizon: int
    initial: bool


@dataclass(frozen=True)
class PruneArm:
    """One prune setting in the sweep.

    ``policy`` None means the reference arm (no pruning). ``target_pool`` is
    the nominal N for scaled arms; None for ``reference`` / ``current``.
    """

    arm_id: str
    policy: CandidatePrunePolicy | None
    target_pool: int | None = None


@dataclass(frozen=True)
class PruningLossReport:
    """Sweep rows plus agreement / gap summary by arm."""

    rows: pl.DataFrame
    summary: pl.DataFrame
    scenarios: tuple[PruningScenario, ...]
    arms: tuple[PruneArm, ...]


def week1_actions_from_plan(
    plan: GameweekPlan,
    *,
    initial: bool,
) -> Week1Actions:
    """Extract week-1 actions from a solved ``GameweekPlan``.

    Args:
        plan: Solved horizon (``HorizonPickSchema`` picks).
        initial: True when step 0 is a draft (buys are not transfers).

    Returns:
        Sorted transfer / XI keys and armbands for step 0.
    """
    step0 = plan.picks.filter(pl.col("step") == 0)
    if initial:
        transfers_in: tuple[int, ...] = ()
        transfers_out: tuple[int, ...] = ()
    else:
        transfers_in = _sorted_keys(step0.filter(pl.col("bought")))
        transfers_out = _sorted_keys(step0.filter(pl.col("sold")))
    xi = _sorted_keys(step0.filter(pl.col("in_xi")))
    captains = _sorted_keys(step0.filter(pl.col("captain")))
    vices = _sorted_keys(step0.filter(pl.col("vice")))
    return Week1Actions(
        transfers_in=transfers_in,
        transfers_out=transfers_out,
        xi=xi,
        captain=captains[0] if captains else None,
        vice=vices[0] if vices else None,
    )


def week1_actions_agree(left: Week1Actions, right: Week1Actions) -> bool:
    """True when transfers, XI, captain, and vice match."""
    return left == right


def check_week1_equivalence(
    reference: GameweekPlan,
    candidate: GameweekPlan,
    *,
    initial: bool,
) -> Week1EquivalenceResult:
    """Equivalence check for a prune-policy change: identical week-1 actions.

    Pass condition for a ``kind: refactor``-style prune lever: ``identical``
    is True. Objective gap is intentionally not part of the pass rule.

    Args:
        reference: Plan from the baseline pool / policy.
        candidate: Plan from the candidate pool / policy.
        initial: Whether step 0 is a draft.

    Returns:
        Structured agreement result with mismatch field names.
    """
    ref_actions = week1_actions_from_plan(reference, initial=initial)
    cand_actions = week1_actions_from_plan(candidate, initial=initial)
    mismatches = ref_actions.mismatches(cand_actions)
    return Week1EquivalenceResult(
        identical=not mismatches,
        reference=ref_actions,
        candidate=cand_actions,
        mismatches=mismatches,
    )


def policy_for_target_pool(
    target: int,
    *,
    base: CandidatePrunePolicy | None = None,
    reference_n: int = DEFAULT_TARGET_POOL,
) -> CandidatePrunePolicy:
    """Scale ``base`` keep quotas toward nominal pool ``target``.

    Args:
        target: Desired approximate pool size.
        base: Policy to scale. Defaults to ``DEFAULT_PRUNE_POLICY``.
        reference_n: Nominal size ``base`` was tuned for.

    Returns:
        A new ``CandidatePrunePolicy`` with scaled top-xPts, top-value,
        per-step, and window keep counts. Cheapest fillers and
        ``window_lengths`` are left unchanged.

    Raises:
        ValueError: ``target`` or ``reference_n`` is below squad minima.
    """
    minimum = sum(MIN_POS_POOL.values())
    if target < minimum:
        raise ValueError(f"target pool {target} is below squad minima ({minimum})")
    if reference_n < minimum:
        raise ValueError(f"reference_n {reference_n} is below squad minima ({minimum})")
    policy = DEFAULT_PRUNE_POLICY if base is None else base
    factor = target / float(reference_n)
    return CandidatePrunePolicy(
        top_xpts=_scale_counts(policy.top_xpts, factor),
        top_value=_scale_counts(policy.top_value, factor),
        top_per_step=_scale_counts(policy.top_per_step, factor),
        top_window=_scale_counts(policy.top_window, factor),
        window_lengths=policy.window_lengths,
        keep_cheapest=policy.keep_cheapest,
        min_p_60=policy.min_p_60,
        auto_min_horizon=policy.auto_min_horizon,
    )


def default_sweep_arms(
    targets: Sequence[int] = DEFAULT_POOL_TARGETS,
) -> tuple[PruneArm, ...]:
    """Reference, scaled ``n=…`` arms, and the current default policy."""
    arms: list[PruneArm] = [PruneArm(arm_id="reference", policy=None, target_pool=None)]
    for target in targets:
        arms.append(
            PruneArm(
                arm_id=f"n={int(target)}",
                policy=policy_for_target_pool(int(target)),
                target_pool=int(target),
            )
        )
    arms.append(
        PruneArm(arm_id="current", policy=DEFAULT_PRUNE_POLICY, target_pool=None)
    )
    return tuple(arms)


def sample_synthetic_scenarios(
    n: int = 24,
    *,
    seed: int = 0,
    horizon: int = 5,
    pool_counts: Mapping[str, int] | None = None,
) -> tuple[PruningScenario, ...]:
    """Build ``n`` synthetic squad states across the §3.1 scenario tags.

    Tags cycle through early season, DGW run-up, ``--rebuild``, and
    mid-season with banked free transfers. Default pools are large enough
    to prune (~200 raw players before dominance). Pass ``pool_counts`` to
    shrink for fast tests.

    Args:
        n: Number of scenarios (typically 20–30).
        seed: RNG seed for owned picks and xPts jitter.
        horizon: Plan horizon (prune auto-applies at H≥3).
        pool_counts: Optional per-position pool sizes. Defaults to a ~200
            player universe (use ~640 for quota-sensitive N sweeps).

    Returns:
        Frozen scenarios ready for :func:`run_pruning_loss_harness`.

    Raises:
        ValueError: ``n`` is outside 1..60 or ``horizon`` < 1.
    """
    if n < 1 or n > 60:
        raise ValueError(f"n must be in 1..60, got {n}")
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    rng = np.random.default_rng(seed)
    players = _synthetic_pool(n_per_pos=pool_counts)
    scenarios: list[PruningScenario] = []
    for index in range(n):
        tag = SCENARIO_TAGS[index % len(SCENARIO_TAGS)]
        points = _synthetic_points(players, horizon=horizon, rng=rng, tag=tag)
        state, initial = _state_for_tag(players, points, tag=tag, rng=rng)
        scenarios.append(
            PruningScenario(
                scenario_id=f"{tag}_{index:02d}",
                tag=tag,
                players=players,
                points=points,
                state=state,
                horizon=horizon,
                initial=initial,
            )
        )
    return tuple(scenarios)


def run_pruning_loss_harness(
    scenarios: Sequence[PruningScenario],
    *,
    arms: Sequence[PruneArm] | None = None,
    reference_options: HighsSolveOptions | None = None,
    sweep_options: HighsSolveOptions | None = None,
) -> PruningLossReport:
    """Solve reference + prune arms; record agreement, gap, time, status.

    Args:
        scenarios: Squad states to evaluate.
        arms: Sweep arms. Defaults to :func:`default_sweep_arms`.
        reference_options: HiGHS knobs for the full-pool reference.
        sweep_options: HiGHS knobs for pruned arms.

    Returns:
        Validated row frame, per-arm summary, and the inputs used.

    Raises:
        ValueError: Empty scenarios / arms, or a solve fails.
    """
    if not scenarios:
        raise ValueError("scenarios must be non-empty")
    active_arms = tuple(arms) if arms is not None else default_sweep_arms()
    if not active_arms:
        raise ValueError("arms must be non-empty")
    if active_arms[0].arm_id != "reference" or active_arms[0].policy is not None:
        raise ValueError("arms[0] must be the reference arm (policy=None)")
    ref_opts = (
        REFERENCE_HIGHS_OPTIONS if reference_options is None else reference_options
    )
    arm_opts = SWEEP_HIGHS_OPTIONS if sweep_options is None else sweep_options

    rows: list[dict[str, object]] = []
    for scenario in scenarios:
        ref_plan = _solve(scenario, policy=None, options=ref_opts)
        ref_actions = week1_actions_from_plan(ref_plan, initial=scenario.initial)
        for arm in active_arms:
            options = ref_opts if arm.policy is None else arm_opts
            plan = (
                ref_plan
                if arm.policy is None
                else _solve(scenario, policy=arm.policy, options=options)
            )
            actions = week1_actions_from_plan(plan, initial=scenario.initial)
            agree = week1_actions_agree(ref_actions, actions)
            rows.append(
                {
                    "scenario_id": scenario.scenario_id,
                    "scenario_tag": scenario.tag,
                    "arm_id": arm.arm_id,
                    "target_pool": arm.target_pool,
                    "pool_size": (
                        int(plan.pool_size)
                        if plan.pool_size is not None
                        else int(scenario.players.height)
                    ),
                    "week1_agree": agree,
                    "objective": float(plan.points),
                    "objective_gap": float(ref_plan.points - plan.points),
                    "solve_elapsed": plan.solve_elapsed,
                    "mip_status": plan.mip_status,
                    "horizon": int(scenario.horizon),
                }
            )
            _log.info(
                "pruning-loss %s arm=%s agree=%s gap=%.4f pool=%s status=%s",
                scenario.scenario_id,
                arm.arm_id,
                agree,
                ref_plan.points - plan.points,
                plan.pool_size,
                plan.mip_status,
            )

    frame = pl.DataFrame(rows)
    checked = PruningLossRowSchema.validate(frame)
    checked = require_dataframe(checked)
    summary = summarise_pruning_loss(checked)
    return PruningLossReport(
        rows=checked,
        summary=summary,
        scenarios=tuple(scenarios),
        arms=active_arms,
    )


def summarise_pruning_loss(rows: pl.DataFrame) -> pl.DataFrame:
    """Per-arm agreement rate and mean objective gap (descriptive).

    Args:
        rows: Frame matching ``PruningLossRowSchema``.

    Returns:
        One row per ``arm_id`` with ``n``, ``agree_rate``, ``mean_objective_gap``,
        and ``mean_solve_elapsed``.
    """
    checked = PruningLossRowSchema.validate(rows)
    checked = require_dataframe(checked)
    return (
        checked.group_by("arm_id", maintain_order=True)
        .agg(
            pl.len().alias("n"),
            pl.col("week1_agree").mean().alias("agree_rate"),
            pl.col("objective_gap").mean().alias("mean_objective_gap"),
            pl.col("solve_elapsed").mean().alias("mean_solve_elapsed"),
            pl.col("pool_size").mean().alias("mean_pool_size"),
        )
        .sort("arm_id")
    )


def write_pruning_loss_report(
    report: PruningLossReport,
    output_dir: Path,
) -> dict[str, Path]:
    """Write parquet + JSON summary under ``output_dir``.

    Args:
        report: Harness result.
        output_dir: Destination directory (created if missing).

    Returns:
        Paths for ``rows.parquet``, ``summary.parquet``, and ``summary.json``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.parquet"
    summary_path = output_dir / "summary.parquet"
    json_path = output_dir / "summary.json"
    report.rows.write_parquet(rows_path)
    report.summary.write_parquet(summary_path)
    payload = {
        "arms": [
            {
                "arm_id": arm.arm_id,
                "target_pool": arm.target_pool,
                "policy": None
                if arm.policy is None
                else arm.policy.model_dump(mode="json"),
            }
            for arm in report.arms
        ],
        "n_scenarios": len(report.scenarios),
        "scenario_ids": [scenario.scenario_id for scenario in report.scenarios],
        "summary": report.summary.to_dicts(),
        "equivalence_pass_rule": (
            "identical week-1 actions (transfers_in/out, xi, captain, vice)"
        ),
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {"rows": rows_path, "summary": summary_path, "json": json_path}


def format_pruning_loss_summary(summary: pl.DataFrame) -> str:
    """Render a compact text table of per-arm agreement and gap."""
    lines = [
        "arm_id  n  agree_rate  mean_obj_gap  mean_pool  mean_elapsed",
        "------  -  ----------  ------------  ---------  ------------",
    ]
    for row in summary.iter_rows(named=True):
        elapsed = row["mean_solve_elapsed"]
        elapsed_s = "n/a" if elapsed is None else f"{float(elapsed):.2f}s"
        lines.append(
            f"{row['arm_id']:<8} {int(row['n']):>2}  "
            f"{float(row['agree_rate']):>10.3f}  "
            f"{float(row['mean_objective_gap']):>12.4f}  "
            f"{float(row['mean_pool_size']):>9.1f}  "
            f"{elapsed_s:>12}"
        )
    return "\n".join(lines)


def _sorted_keys(frame: pl.DataFrame) -> tuple[int, ...]:
    if frame.is_empty():
        return ()
    return tuple(int(key) for key in frame["player_key"].sort().to_list())


def _scale_counts(
    counts: Mapping[str, int] | None,
    factor: float,
) -> dict[str, int] | None:
    if counts is None:
        return None
    scaled: dict[str, int] = {}
    for position, quota in counts.items():
        if quota <= 0:
            scaled[position] = 0
            continue
        value = max(MIN_POS_POOL[position], int(round(quota * factor)))
        scaled[position] = value
    return scaled


def _solve(
    scenario: PruningScenario,
    *,
    policy: CandidatePrunePolicy | None,
    options: HighsSolveOptions,
) -> GameweekPlan:
    return plan_squad(
        scenario.players,
        scenario.points,
        scenario.state,
        initial=scenario.initial,
        horizon=scenario.horizon,
        prune=policy is not None,
        prune_policy=policy,
        highs_options=options,
        # Equivalence harness measures the transfer MILP, not D1 lineup re-opt.
        lineup_reopt=False,
    )


def _synthetic_pool(*, n_per_pos: Mapping[str, int] | None = None) -> pl.DataFrame:
    counts = dict(n_per_pos or {"GK": 20, "DEF": 60, "MID": 80, "FWD": 40})
    rows: list[dict[str, object]] = []
    key = 1
    for position, count in counts.items():
        for index in range(count):
            rows.append(
                {
                    "player_key": key,
                    "position": position,
                    "team_key": 1 + (index % 20),
                    "price": 40 + (index % 60),
                }
            )
            key += 1
    return pl.DataFrame(rows)


def _synthetic_points(
    players: pl.DataFrame,
    *,
    horizon: int,
    rng: np.random.Generator,
    tag: str,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for row in players.iter_rows(named=True):
        base = 1.5 + 0.05 * (80 - (int(row["player_key"]) % 80))
        for step in range(horizon):
            jitter = float(rng.normal(0.0, 0.25))
            xpts = max(0.0, base + jitter - 0.05 * step)
            # DGW run-up: inflate step 0 for a slice of attackers.
            if tag == "dgw_runup" and step == 0 and row["position"] in {"MID", "FWD"}:
                if int(row["player_key"]) % 7 == 0:
                    xpts += 4.0
            rows.append(
                {
                    "player_key": int(row["player_key"]),
                    "step": step,
                    "xpts": xpts,
                }
            )
    return pl.DataFrame(rows)


def _state_for_tag(
    players: pl.DataFrame,
    points: pl.DataFrame,
    *,
    tag: str,
    rng: np.random.Generator,
) -> tuple[SquadState, bool]:
    decay = 0.9
    if tag == "early_season":
        state = SquadState(
            bank=1000,
            free_transfers=1,
            ft_bank_cap=5,
            hit_cost=4,
            decay=decay,
            bench_weight=0.0001,
            owned=[],
        )
        return state, True

    owned = _pick_owned(players, points, rng=rng)
    bank = max(0, 1000 - sum(player.selling_price for player in owned))
    if tag == "rebuild":
        base = SquadState(
            bank=bank,
            free_transfers=1,
            ft_bank_cap=5,
            hit_cost=4,
            decay=decay,
            bench_weight=0.0001,
            owned=owned,
        )
        return rebuild_squad_state(base), True

    free_transfers = 2 if tag == "mid_banked_fts" else 1
    state = SquadState(
        bank=bank,
        free_transfers=free_transfers,
        ft_bank_cap=5,
        hit_cost=4,
        decay=decay,
        bench_weight=0.0001,
        owned=owned,
    )
    return state, False


def _pick_owned(
    players: pl.DataFrame,
    points: pl.DataFrame,
    *,
    rng: np.random.Generator,
) -> list[OwnedPlayer]:
    """Pick a legal-ish fifteen biased toward higher Σ xPts."""
    totals = points.group_by("player_key").agg(pl.col("xpts").sum().alias("sum_xpts"))
    ranked = players.join(totals, on="player_key", how="left").with_columns(
        pl.col("sum_xpts").fill_null(0.0)
    )
    owned: list[OwnedPlayer] = []
    club_counts: dict[int, int] = {}
    for position, need in MIN_POS_POOL.items():
        part = ranked.filter(pl.col("position") == position).sort(
            "sum_xpts", descending=True
        )
        keys = part["player_key"].to_list()
        # Mild shuffle of the top band so scenarios diverge.
        top = keys[: max(need * 3, need)]
        rng.shuffle(top)
        taken = 0
        for key in top:
            row = part.filter(pl.col("player_key") == key).row(0, named=True)
            team = int(row["team_key"])
            if club_counts.get(team, 0) >= 3:
                continue
            price = int(row["price"])
            owned.append(
                OwnedPlayer(
                    player_key=int(key),
                    purchase_price=price,
                    selling_price=price,
                )
            )
            club_counts[team] = club_counts.get(team, 0) + 1
            taken += 1
            if taken >= need:
                break
        if taken < need:
            raise ValueError(f"could not fill {position}: need {need}, got {taken}")
    return owned


# Re-export for callers that want a stable dump of Week1Actions.
def week1_actions_dict(actions: Week1Actions) -> dict[str, object]:
    """JSON-friendly dict of week-1 actions."""
    return asdict(actions)
