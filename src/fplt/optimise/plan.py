"""Multi-period squad plan: transfers, free-transfer banking, and hits.

The first period is the only one a caller should act on. Later periods exist
so a free transfer is kept when it is worth more next week. Hits are four
points (or ``SquadState.hit_cost``) for each transfer beyond the free ones,
and they are not discounted. Points are. A blank ``owned`` squad is an
initial draft and those buys are not transfers.

Pass ``season`` and ``gameweeks`` to enable chips from ``ChipRules`` and
``SquadState.chips``: Bench Boost, Triple Captain, Wildcard (hit
suppression; next free transfer resets to 1), and Free Hit (temporary
squad; step ``t+1`` continues from ``t-1``; free-transfer banking passes
through when ``free_hit_keeps_free_transfers``). Optional ``chip_windows``
restricts or forces chips by gameweek (Wave C3 hooks for C5).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace

import highspy
import numpy as np
import polars as pl

from fplt.optimise.candidate_prune import (
    DEFAULT_KEEP_CHEAPEST as CANDIDATE_KEEP_CHEAPEST,
)
from fplt.optimise.candidate_prune import (
    DEFAULT_PRUNE_POLICY,
    LEGACY_TOP_K,
    CandidatePrunePolicy,
    prune_candidates,
)
from fplt.optimise.candidate_prune import (
    PRUNE_AUTO_MIN_HORIZON as PRUNE_AUTO_MIN_HORIZON,
)
from fplt.optimise.chip_windows import ChipUseWindows
from fplt.optimise.highs_options import (
    DEFAULT_HIGHS_OPTIONS,
    HighsSolveOptions,
    log_solve_status,
)
from fplt.optimise.horizon_decay import (
    geometric_step_weights,
    validate_step_weights,
)
from fplt.optimise.lineup import (
    DEFAULT_HAUL_WEIGHT,
    reoptimize_plan_lineups,
)
from fplt.optimise.long_horizon import (
    DEFAULT_LONG_HORIZON_POLICY,
    LongHorizonPolicy,
    coarsen_steps,
    shift_horizon_picks,
)
from fplt.optimise.prices import list_price_matrix, sell_price_matrix
from fplt.optimise.scenarios import ScenarioBundle
from fplt.optimise.squad import (
    _MAX_PER_CLUB,  # pyright: ignore[reportPrivateUsage]
    _SQUAD,  # pyright: ignore[reportPrivateUsage]
    _VICE_WEIGHT,  # pyright: ignore[reportPrivateUsage]
    _XI_MAX,  # pyright: ignore[reportPrivateUsage]
    _XI_MIN,  # pyright: ignore[reportPrivateUsage]
    _XI_SIZE,  # pyright: ignore[reportPrivateUsage]
    _add_binaries,  # pyright: ignore[reportPrivateUsage]
    _at_most,  # pyright: ignore[reportPrivateUsage]
    _pair,  # pyright: ignore[reportPrivateUsage]
    _row,  # pyright: ignore[reportPrivateUsage]
    _sum,  # pyright: ignore[reportPrivateUsage]
)
from fplt.optimise.squad_integrity import (
    inspect_horizon_picks,
    locked_fifteen_issue,
    log_horizon_integrity,
    require_locked_steps,
)
from fplt.optimise.state import ChipInventory, OwnedPlayer, SquadState
from fplt.schemas.plan import (
    CandidateSchema,
    HorizonPickSchema,
    HorizonPointsSchema,
    PlanSchema,
    PriceForecastSchema,
)
from fplt.scoring.chips import ChipName, ChipRules, ChipWindow
from fplt.utils.frames import require_dataframe
from fplt.utils.timing import timed_phase

_Schema = (
    type[CandidateSchema]
    | type[HorizonPickSchema]
    | type[HorizonPointsSchema]
    | type[PlanSchema]
    | type[PriceForecastSchema]
)

_MAX_MOVES = 15
_SQUAD_SIZE = 15
_log = logging.getLogger(__name__)

# Default look-ahead for ``fplt plan`` / ``fplt decisions`` / ``fplt apply``.
# Chosen from the 2024-25 chips-off H∈{1..4} season-points sweep (H=3 best
# cross-model mean; H=4 matched H=3 actions at higher MILP cost).
DEFAULT_PLAN_HORIZON = 3

# Re-exports for callers / tests that imported prune knobs from this module.
DEFAULT_CANDIDATE_TOP_K: Mapping[str, int] = LEGACY_TOP_K
DEFAULT_KEEP_CHEAPEST = CANDIDATE_KEEP_CHEAPEST


@dataclass(frozen=True)
class GameweekPlan:
    """Solved horizon. ``points`` is decayed XI plus captain, minus hits.

    ``chips`` is one entry per horizon step: ``\"wildcard\"``, ``\"freehit\"``,
    ``\"bboost\"``, ``\"3xc\"``, or ``None``.

    ``solve_elapsed`` / ``mip_status`` / ``pool_size`` are filled by
    ``plan_squad`` for the pruning-loss harness and logging; older
    constructors may leave them unset.
    """

    picks: pl.DataFrame
    hits: tuple[float, ...]
    free_transfers: tuple[int, ...]
    points: float
    chips: tuple[str | None, ...] = ()
    solve_elapsed: float | None = None
    mip_status: str | None = None
    pool_size: int | None = None


def plan_squad(
    players: pl.DataFrame,
    points: pl.DataFrame,
    state: SquadState,
    *,
    initial: bool,
    horizon: int,
    season: str | None = None,
    gameweeks: list[int] | None = None,
    chip_windows: ChipUseWindows | None = None,
    chip_rules_season: str | None = None,
    prune: bool | None = None,
    prune_policy: CandidatePrunePolicy | None = None,
    highs_options: HighsSolveOptions | None = None,
    warm_start: GameweekPlan | None = None,
    warm_start_shift: int = 1,
    long_horizon_policy: LongHorizonPolicy | None = None,
    lineup_reopt: bool = True,
    scenarios: ScenarioBundle | None = None,
    p_haul: pl.DataFrame | None = None,
    haul_weight: float = DEFAULT_HAUL_WEIGHT,
    price_forecast: pl.DataFrame | None = None,
    step_weights: Sequence[float] | None = None,
) -> GameweekPlan:
    """Solve one squad over ``horizon`` gameweeks.

    Args:
        players: Candidates matching ``CandidateSchema``.
        points: ``HorizonPointsSchema``. A missing step is zero points.
            Rows for ``player_key`` values absent from ``players`` are
            ignored (common for mid-season additions that appear in the
            prediction universe before they have a PIT price).
        state: Bank, free transfers, owned selling prices, and chip
            inventory.
        initial: The owned list is empty and the first week is a draft.
        horizon: Number of gameweeks, including the one to act on.
        season: Season label for ``ChipRules.for_season`` when
            ``chip_rules_season`` is omitted. Required with ``gameweeks``
            to allow chips (Wildcard, Free Hit, Bench Boost, Triple
            Captain).
        gameweeks: Target gameweek for each horizon step, in order. Must
            match ``horizon`` when set.
        chip_windows: Optional allowed / forced / no-chip hooks applied
            on top of ``ChipRules`` (Wave C3 / C5).
        chip_rules_season: Optional season whose ``configs/chips`` file
            to load. Defaults to ``season``. Use to score an older fixture
            season under current chip rules (counterfactual).
        prune: Drop low-value candidates before the MILP. ``None``
            auto-enables when ``horizon >= policy.auto_min_horizon``;
            ``True`` / ``False`` force on / off.
        prune_policy: Keep rules (top xPts, top value, cheapest per
            position). Defaults to ``DEFAULT_PRUNE_POLICY`` (~130 players).
        highs_options: HiGHS time limit / gap / parallel knobs. Defaults
            to ``DEFAULT_HIGHS_OPTIONS`` (600s, gap 1e-3, accept
            time-limit incumbents).
        warm_start: Optional prior ``GameweekPlan`` used as a HiGHS MIP
            start. Pass last week's plan (default ``warm_start_shift=1``)
            or an already-shifted plan with ``warm_start_shift=0``.
        warm_start_shift: Leading steps to drop from ``warm_start`` before
            seeding (1 maps prior step 1 → this step 0).
        long_horizon_policy: H≥6 MILP shrink knobs. Defaults to
            ``DEFAULT_LONG_HORIZON_POLICY`` (relax-and-fix on and
            captain/vice coarsening from step 3 when
            ``horizon >= auto_min_horizon``). Forced-chip gameweeks keep
            armband detail so TC/FH landmarks stay feasible.
        lineup_reopt: After the transfer MILP, re-choose XI / captain /
            vice on each *locked* fifteen via sample-average (or EV +
            ``p_haul``). Default on (Wave D1); set ``False`` to keep the
            MILP lineup only. Continuous far steps under relax-and-fix
            are not locked fifteens — reopt skips them when 0.5-thresholded
            picks fail formation (look-ahead only); step 0 must stay legal.
        scenarios: Optional shared scenarios for step-0 lineup SAA.
        p_haul: Optional ``player_key``, ``step``, ``p_haul`` for captaincy.
        haul_weight: Weight on ``p_haul`` in the EV lineup path.
        price_forecast: Optional ``PriceForecastSchema`` list prices per
            ``player_key`` × ``step``. ``None`` keeps as-of prices for
            every step (default). Step 0 must match candidate prices.
        step_weights: Optional per-step objective weights (0-indexed).
            ``None`` keeps geometric ``state.decay ** step`` (live
            control). Skill-by-horizon treatment passes measured weights
            from ``fplt.optimise.horizon_decay`` (FDB-242).

    Returns:
        Squad members and sales at every step.

    Raises:
        ValueError: The inputs disagree, no squad meets the rules, or the
            actionable (step-0) squad is not a legal locked fifteen after
            solve.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    weights = _resolve_plan_weights(state, horizon, step_weights)
    targets = _chip_gameweeks(season, gameweeks, horizon)
    rules_season = chip_rules_season if chip_rules_season is not None else season
    if chip_windows is not None and targets is None:
        raise ValueError("chip_windows requires season and gameweeks")
    rules = ChipRules.for_season(rules_season) if rules_season is not None else None
    pool = _frame(players.sort("player_key"), CandidateSchema)
    # HorizonPointsSchema is strict; optional p_60 is only for prune fillers.
    scored = _frame(points.select("player_key", "step", "xpts"), HorizonPointsSchema)
    owned = _owned(state, initial)
    policy = DEFAULT_PRUNE_POLICY if prune_policy is None else prune_policy
    do_prune = policy.should_auto_apply(horizon) if prune is None else prune
    raw_n = pool.height
    if do_prune:
        prune_points = scored
        if "p_60" in points.columns:
            prune_points = scored.join(
                points.select("player_key", "step", "p_60"),
                on=["player_key", "step"],
                how="left",
            )
        pool = prune_candidates(
            pool,
            prune_points,
            owned_keys=[player.player_key for player in owned],
            policy=policy,
            decay=state.decay,
            step_weights=weights,
        )
        _log.info(
            "candidate prune horizon=%s raw=%s kept=%s (%.0f%%)",
            horizon,
            raw_n,
            pool.height,
            100.0 * pool.height / raw_n if raw_n else 0.0,
        )
    keys = pool["player_key"].to_list()
    index = {int(key): pos for pos, key in enumerate(keys)}
    missing = [player.player_key for player in owned if player.player_key not in index]
    if missing:
        raise ValueError(f"owned players are not in the pool: {missing}")
    count = pool.height
    expected = _expected(scored, index, count, horizon)
    as_of_prices = {
        int(key): int(price)
        for key, price in pool.select("player_key", "price").iter_rows()
    }
    forecast = (
        None if price_forecast is None else _frame(price_forecast, PriceForecastSchema)
    )
    list_price = list_price_matrix(keys, horizon, as_of_prices, price_forecast=forecast)
    # Owned sell credits follow 50% sell-on at each step's list price.
    # Non-owned mid-horizon churn uses list price (v1 approx).
    sell_price = sell_price_matrix(list_price, owned, index)
    owned_at = np.zeros(count, dtype=bool)
    for player in owned:
        owned_at[index[player.player_key]] = True
    options = DEFAULT_HIGHS_OPTIONS if highs_options is None else highs_options
    lh_policy = (
        DEFAULT_LONG_HORIZON_POLICY
        if long_horizon_policy is None
        else long_horizon_policy
    )
    coarsen_from = lh_policy.effective_coarsen_from(horizon)
    coarse_steps = coarsen_steps(
        horizon,
        coarsen_from,
        gameweeks=targets,
        chip_windows=chip_windows,
    )
    locked_steps = _integral_steps(lh_policy, horizon)
    _assert_owned_locked_fifteen(pool, owned, initial=initial)
    _log.info(
        "plan_squad long_horizon raf=%s integral_steps=%s coarsen_steps=%s "
        "lineup_reopt=%s",
        _raf_active(lh_policy, horizon),
        locked_steps,
        sorted(coarse_steps),
        lineup_reopt,
    )
    columns = _columns(
        expected,
        state,
        horizon,
        initial,
        coarsen_steps=coarse_steps,
        step_weights=weights,
    )
    solver = _solver(columns.costs, options)
    _constrain(
        solver,
        pool,
        columns,
        owned_at,
        list_price,
        sell_price,
        state,
        initial,
        horizon,
        rules,
        targets,
        chip_windows,
        coarsen_steps=coarse_steps,
    )
    if warm_start is not None:
        _apply_warm_start(
            solver,
            columns,
            keys,
            warm_start,
            shift=warm_start_shift,
        )
    solver.setMaximize()
    started = time.perf_counter()
    with timed_phase("optimise/milp"):
        status = _run_solver(
            solver,
            columns=columns,
            options=options,
            horizon=horizon,
            long_horizon=lh_policy,
        )
    elapsed = time.perf_counter() - started
    _log.info(
        "plan_squad solve horizon=%s pool=%s binaries≈%s status=%s elapsed=%.2fs",
        horizon,
        count,
        int(columns.costs.size),
        status,
        elapsed,
    )
    if not options.accepts_status(status):
        raise ValueError("no plan satisfies the budget, positions, and club limits")
    log_solve_status(status, options=options)
    plan = _read(
        solver,
        pool,
        expected,
        columns,
        state,
        initial,
        horizon,
        rules,
        step_weights=weights,
    )
    integrity = inspect_horizon_picks(plan.picks, plan.chips)
    log_horizon_integrity(integrity, binary_steps=locked_steps)
    # Executable week must be a legal fifteen. Far continuous steps may fail
    # formation after 0.5-thresholding; that is look-ahead only.
    require_locked_steps(
        integrity,
        (0,),
        context="plan_squad after solve (actionable step 0)",
    )
    if lineup_reopt:
        with timed_phase("plan/lineup"):
            refreshed = reoptimize_plan_lineups(
                plan.picks,
                scenarios=scenarios,
                p_haul=p_haul,
                haul_weight=haul_weight,
                scenario_step=0,
            )
        plan = replace(plan, picks=_frame(refreshed, HorizonPickSchema))
    return replace(
        plan,
        solve_elapsed=float(elapsed),
        mip_status=status.name,
        pool_size=int(count),
    )


def plan_table(
    plan: GameweekPlan,
    model_id: str,
    season: str,
    as_of_gw: int,
    gameweeks: list[int],
) -> pl.DataFrame:
    """Attach model identity to a plan.

    Args:
        plan: Solved horizon. One entry of ``gameweeks`` per step.
        model_id: Predictor id these points came from.
        season: Season label.
        as_of_gw: Deadline the plan was built at.
        gameweeks: Target gameweek of each step, in order.

    Returns:
        A frame matching ``PlanSchema``.

    Raises:
        ValueError: ``gameweeks`` does not match the horizon.
    """
    steps = plan.picks["step"].unique().sort().to_list()
    if len(gameweeks) != (max(steps) + 1 if steps else 0):
        raise ValueError("gameweeks must have one entry per horizon step")
    chips = _chips_for_horizon(plan.chips, len(gameweeks))
    targets = pl.DataFrame(
        {
            "step": list(range(len(gameweeks))),
            "target_gw": gameweeks,
            "hit": list(plan.hits),
            "chip": chips,
        }
    )
    frame = plan.picks.join(targets, on="step").with_columns(
        pl.lit(model_id).alias("model_id"),
        pl.lit(season).alias("as_of_season"),
        pl.lit(as_of_gw).alias("as_of_gw"),
    )
    return _frame(
        frame.select(
            "model_id",
            "as_of_season",
            "as_of_gw",
            "target_gw",
            "step",
            "player_key",
            "position",
            "team_key",
            "in_xi",
            "captain",
            "vice",
            "bought",
            "sold",
            "xpts",
            "hit",
            "chip",
        ),
        PlanSchema,
    )


def ensure_plan_chip_column(frame: pl.DataFrame) -> pl.DataFrame:
    """Fill a missing ``chip`` column so legacy plan parquets still validate.

    Args:
        frame: Plan rows, possibly from before ``chip`` was on ``PlanSchema``.

    Returns:
        ``frame`` unchanged when ``chip`` is present; otherwise with a null
        ``chip`` column.
    """
    if "chip" in frame.columns:
        return frame
    return frame.with_columns(pl.lit(None).cast(pl.Utf8()).alias("chip"))


def plan_step_chip(plan: pl.DataFrame, *, step: int = 0) -> str | None:
    """Return the chip played at ``step`` on a ``PlanSchema`` frame.

    Args:
        plan: Saved plan rows (``chip`` may be absent on legacy files).
        step: Horizon step to read (default week 1 / step 0).

    Returns:
        Chip name (``freehit`` / ``wildcard`` / ``bboost`` / ``3xc``), or
        ``None`` when no chip was played or the column is missing.
    """
    if plan.is_empty() or "step" not in plan.columns:
        return None
    part = ensure_plan_chip_column(plan).filter(pl.col("step") == step)
    if part.is_empty():
        return None
    played = [value for value in part["chip"].to_list() if value is not None]
    return str(played[0]) if played else None


def covered_horizon(gameweek: int, horizon: int) -> int:
    """Weeks a plan actually covers, stopping at gameweek 38.

    Args:
        gameweek: Deadline the plan is built at.
        horizon: Weeks requested, including that deadline.

    Returns:
        The number of steps written into the plan file.
    """
    if horizon < 1 or gameweek > 38:
        return 0
    return sum(1 for offset in range(horizon) if gameweek + offset <= 38)


_POSITION_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}


def render_report(
    plan: GameweekPlan,
    season: str,
    gameweeks: list[int],
    names: Mapping[int, str] | None = None,
) -> str:
    """Describe the squad, the armbands, and the transfers for each week.

    Each gameweek is a markdown section with transfer, XI, and bench tables.
    Captain and vice are marked ``(C)`` / ``(V)`` on the Starting XI name.

    Args:
        plan: Solved horizon.
        season: Season label used as the title.
        gameweeks: Target gameweek of each step, in order.
        names: ``player_key`` to FPL ``web_name``. Missing names stay as the key.

    Returns:
        Markdown text.
    """
    lines = [f"# {season}", ""]
    for step, gameweek in enumerate(gameweeks):
        part = plan.picks.filter(pl.col("step") == step)
        lines.append(f"## GW {gameweek}")
        lines.append("")
        lines.append(f"Free transfers: {plan.free_transfers[step]}")
        lines.append(f"Hit: {plan.hits[step]:.0f}")
        chip = plan.chips[step] if step < len(plan.chips) else None
        lines.append(f"Chip: {chip or '—'}")
        lines.append("")
        lines.extend(
            _report_section(
                "Transfers out",
                part.filter(pl.col("sold")),
                names,
                armbands=False,
            )
        )
        lines.extend(
            _report_section(
                "Transfers in",
                part.filter(pl.col("bought")),
                names,
                armbands=False,
            )
        )
        lines.extend(
            _report_section(
                "Starting XI",
                part.filter(pl.col("in_xi")),
                names,
                armbands=True,
            )
        )
        lines.extend(
            _report_section(
                "Bench",
                part.filter(~pl.col("sold") & ~pl.col("in_xi")),
                names,
                armbands=False,
            )
        )
    return "\n".join(lines).rstrip() + "\n"


def _report_section(
    title: str,
    rows: pl.DataFrame,
    names: Mapping[int, str] | None,
    *,
    armbands: bool,
) -> list[str]:
    """Render one headed block: an empty note or a player table."""
    lines = [f"### {title}", ""]
    if rows.is_empty():
        lines.append("_None_")
        lines.append("")
        return lines
    lines.extend(_player_table(rows, names, armbands=armbands))
    lines.append("")
    return lines


def _player_table(
    rows: pl.DataFrame,
    names: Mapping[int, str] | None,
    *,
    armbands: bool,
) -> list[str]:
    """Build a markdown table from plan pick rows."""
    ordered = sorted(
        rows.iter_rows(named=True),
        key=lambda row: (
            _POSITION_ORDER.get(str(row["position"]), 99),
            -float(row["xpts"]),
            int(row["player_key"]),
        ),
    )
    lines = [
        "| Name | Pos | Team | Price | xPts |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for row in ordered:
        label = _label(int(row["player_key"]), names)
        if armbands:
            if bool(row["captain"]):
                label = f"{label} (C)"
            elif bool(row["vice"]):
                label = f"{label} (V)"
        lines.append(
            "| "
            + " | ".join(
                [
                    label,
                    str(row["position"]),
                    str(int(row["team_key"])),
                    f"{int(row['price']) / 10:.1f}",
                    f"{float(row['xpts']):.1f}",
                ]
            )
            + " |"
        )
    return lines


class _Columns:
    """Binary column indices and their objective coefficients."""

    def __init__(self, count: int, horizon: int) -> None:
        self.count = count
        self.horizon = horizon
        self._cost: list[float] = []
        self.squad: list[np.ndarray] = []
        self.xi: list[np.ndarray] = []
        self.captain: list[np.ndarray] = []
        self.vice: list[np.ndarray] = []
        self.buy: list[np.ndarray] = []
        self.sell: list[np.ndarray] = []
        self.triple_extra: list[np.ndarray] = []
        self.bench_score: list[np.ndarray] = []
        self.score_squad: list[np.ndarray] = []
        self.score_xi: list[np.ndarray] = []
        self.score_captain: list[np.ndarray] = []
        self.score_vice: list[np.ndarray] = []
        self.score_triple: list[np.ndarray] = []
        self.score_bench: list[np.ndarray] = []
        self.fh_squad: list[np.ndarray] = []
        self.fh_xi: list[np.ndarray] = []
        self.fh_captain: list[np.ndarray] = []
        self.fh_vice: list[np.ndarray] = []
        self.wc: list[int] = []
        self.fh: list[int] = []
        self.bb: list[int] = []
        self.tc: list[int] = []
        self.moves: dict[int, list[int]] = {}
        self.free: dict[int, dict[int, int]] = {}
        self.link: dict[int, dict[tuple[int, int], int]] = {}
        self.hit_pay: dict[int, dict[tuple[int, int], int]] = {}

    def block(self, costs: np.ndarray) -> np.ndarray:
        start = len(self._cost)
        self._cost.extend(float(value) for value in costs)
        return np.arange(start, start + costs.shape[0], dtype=np.int32)

    def flag(self, cost: float = 0.0) -> int:
        index = len(self._cost)
        self._cost.append(cost)
        return index

    @property
    def costs(self) -> np.ndarray:
        return np.asarray(self._cost, dtype=np.float64)


def _columns(
    expected: np.ndarray,
    state: SquadState,
    horizon: int,
    initial: bool,
    *,
    coarsen_steps: frozenset[int] | None = None,
    step_weights: Sequence[float] | None = None,
) -> _Columns:
    count = expected.shape[0]
    columns = _Columns(count, horizon)
    zeros = np.zeros(count, dtype=np.float64)
    coarse: frozenset[int] = coarsen_steps if coarsen_steps is not None else frozenset()
    weights = _resolve_plan_weights(state, horizon, step_weights)
    for step in range(horizon):
        weight = weights[step]
        points = np.asarray(expected[:, step], dtype=np.float64)
        bench = _scaled(points, weight * state.bench_weight)
        played = _scaled(points, weight * (1.0 - state.bench_weight))
        # Far steps drop captain/vice (and TC) objective; XI / BB remain.
        # Forced-chip steps are excluded from ``coarse`` (see coarsen_steps).
        drop_armbands = step in coarse
        armband = zeros if drop_armbands else _scaled(points, weight)
        vice = zeros if drop_armbands else _scaled(points, weight * _VICE_WEIGHT)
        bb_boost = _scaled(points, weight)
        # Structural lineup / transfer vars (objective lives on score_* / fh_*).
        columns.squad.append(columns.block(zeros))
        columns.xi.append(columns.block(zeros))
        columns.captain.append(columns.block(zeros))
        columns.vice.append(columns.block(zeros))
        columns.triple_extra.append(columns.block(zeros))
        columns.bench_score.append(columns.block(zeros))
        columns.buy.append(columns.block(zeros))
        columns.sell.append(columns.block(zeros))
        # Points when Free Hit is off (gated by 1 - fh_t).
        columns.score_squad.append(columns.block(bench))
        columns.score_xi.append(columns.block(played))
        columns.score_captain.append(columns.block(armband))
        columns.score_vice.append(columns.block(vice))
        columns.score_triple.append(columns.block(armband))
        columns.score_bench.append(columns.block(bb_boost))
        # Temporary Free Hit lineup (gated by fh_t).
        columns.fh_squad.append(columns.block(bench))
        columns.fh_xi.append(columns.block(played))
        columns.fh_captain.append(columns.block(armband))
        columns.fh_vice.append(columns.block(vice))
        columns.wc.append(columns.flag())
        columns.fh.append(columns.flag())
        columns.bb.append(columns.flag())
        columns.tc.append(columns.flag())
    for step in _priced(horizon, initial):
        columns.moves[step] = [columns.flag() for _ in range(_MAX_MOVES + 1)]
        columns.free[step] = {
            free: columns.flag() for free in range(1, state.ft_bank_cap + 1)
        }
        columns.link[step] = {}
        columns.hit_pay[step] = {}
        for free in range(1, state.ft_bank_cap + 1):
            for moves in range(_MAX_MOVES + 1):
                penalty = state.hit_cost * max(0, moves - free)
                columns.link[step][free, moves] = columns.flag()
                if penalty > 0:
                    columns.hit_pay[step][free, moves] = columns.flag(-float(penalty))
    return columns


def _constrain(
    solver: highspy.Highs,
    pool: pl.DataFrame,
    columns: _Columns,
    owned: np.ndarray,
    price: np.ndarray,
    sell_price: np.ndarray,
    state: SquadState,
    initial: bool,
    horizon: int,
    rules: ChipRules | None,
    gameweeks: list[int] | None,
    chip_windows: ChipUseWindows | None,
    *,
    coarsen_steps: frozenset[int] | None = None,
) -> None:
    positions = pool["position"].to_list()
    clubs = pool["team_key"].to_list()
    coarse: frozenset[int] = coarsen_steps if coarsen_steps is not None else frozenset()
    for step in range(horizon):
        _lineup(
            solver,
            columns.squad[step],
            columns.xi[step],
            columns.captain[step],
            columns.vice[step],
            positions,
            clubs,
            armbands=step not in coarse,
        )
    _transition(solver, columns, owned, horizon)
    _cash(solver, columns, price, sell_price, state.bank, horizon)
    _transfers(solver, columns, state, initial, horizon, rules)
    _chips(
        solver,
        columns,
        state.chips,
        rules,
        gameweeks,
        horizon,
        initial,
        price,
        sell_price,
        state.bank,
        positions,
        clubs,
        chip_windows,
        coarsen_steps=coarse,
    )


def _lineup(
    solver: highspy.Highs,
    squad: np.ndarray,
    xi: np.ndarray,
    captain: np.ndarray,
    vice: np.ndarray,
    positions: list[str],
    clubs: list[int],
    *,
    armbands: bool = True,
) -> None:
    _sum(solver, squad, _SQUAD_SIZE, _SQUAD_SIZE)
    _sum(solver, xi, _XI_SIZE, _XI_SIZE)
    if armbands:
        _sum(solver, captain, 1, 1)
        _sum(solver, vice, 1, 1)
    else:
        # Far-step coarsening: pin captain/vice off.
        for index in range(squad.shape[0]):
            _equal(solver, [int(captain[index])], [1.0], 0.0)
            _equal(solver, [int(vice[index])], [1.0], 0.0)
    for index in range(squad.shape[0]):
        _pair(solver, int(xi[index]), int(squad[index]))
        if armbands:
            _pair(solver, int(captain[index]), int(xi[index]))
            _pair(solver, int(vice[index]), int(xi[index]))
            _at_most(solver, [int(captain[index]), int(vice[index])], 1)
    for position, quota in _SQUAD.items():
        members = [i for i, name in enumerate(positions) if name == position]
        _sum(solver, [int(squad[i]) for i in members], quota, quota)
        _sum(
            solver,
            [int(xi[i]) for i in members],
            _XI_MIN[position],
            _XI_MAX[position],
        )
    for club in sorted(set(clubs)):
        chosen = [int(squad[i]) for i, key in enumerate(clubs) if key == club]
        _sum(solver, chosen, 0, _MAX_PER_CLUB)


def _chips(
    solver: highspy.Highs,
    columns: _Columns,
    inventory: ChipInventory,
    rules: ChipRules | None,
    gameweeks: list[int] | None,
    horizon: int,
    initial: bool,
    price: np.ndarray,
    sell_price: np.ndarray,
    bank: int,
    positions: list[str],
    clubs: list[int],
    chip_windows: ChipUseWindows | None,
    *,
    coarsen_steps: frozenset[int] | None = None,
) -> None:
    """Wire chip binaries, Free Hit temporary squad, and BB/TC scoring gates."""
    priced = set(_priced(horizon, initial))
    coarse: frozenset[int] = coarsen_steps if coarsen_steps is not None else frozenset()
    for step in range(horizon):
        drop_armbands = step in coarse
        for index in range(columns.count):
            extra = int(columns.triple_extra[step][index])
            bench = int(columns.bench_score[step][index])
            _pair(solver, extra, int(columns.captain[step][index]))
            _pair(solver, extra, columns.tc[step])
            _pair(solver, bench, int(columns.squad[step][index]))
            _at_most(solver, [bench, int(columns.xi[step][index])], 1)
            _pair(solver, bench, columns.bb[step])
            _gate_score(solver, columns, step, index)
            _free_hit_player(solver, columns, step, index, armbands=not drop_armbands)
        _free_hit_lineup(
            solver, columns, step, positions, clubs, armbands=not drop_armbands
        )
        _free_hit_budget(solver, columns, step, price, sell_price, bank)
        if drop_armbands:
            # TC needs a captain; coarsened steps pin captains off.
            # Forced-chip steps are excluded from ``coarse``.
            _equal(solver, [columns.tc[step]], [1.0], 0.0)
        if step not in priced:
            _equal(solver, [columns.wc[step]], [1.0], 0.0)
            _equal(solver, [columns.fh[step]], [1.0], 0.0)
        if rules is None or rules.one_chip_per_gameweek:
            _at_most(
                solver,
                [
                    columns.wc[step],
                    columns.fh[step],
                    columns.bb[step],
                    columns.tc[step],
                ],
                1,
            )
    if rules is None or gameweeks is None:
        for step in range(horizon):
            _equal(solver, [columns.wc[step]], [1.0], 0.0)
            _equal(solver, [columns.fh[step]], [1.0], 0.0)
            _equal(solver, [columns.bb[step]], [1.0], 0.0)
            _equal(solver, [columns.tc[step]], [1.0], 0.0)
        return
    chip_cols: tuple[tuple[ChipName, list[int]], ...] = (
        ("wildcard", columns.wc),
        ("freehit", columns.fh),
        ("bboost", columns.bb),
        ("3xc", columns.tc),
    )
    for chip, cols in chip_cols:
        _at_most(solver, cols, float(inventory.remaining[chip]))
    for window in rules.windows:
        steps = [
            step
            for step, gameweek in enumerate(gameweeks)
            if window.start_event <= gameweek <= window.stop_event
        ]
        if not steps:
            continue
        for chip, cols in chip_cols:
            # Only windows that grant this chip may cap it. Pre-2025/26
            # ``first_wildcard`` / ``season`` overlap in gameweeks; applying a
            # 0-cap from the wildcard-only window wrongly forbids BB/TC/FH.
            granted = sum(
                1
                for grant in rules.grants
                if grant.name == chip and grant.window == window.id
            )
            if granted == 0:
                continue
            _at_most(
                solver,
                [cols[step] for step in steps],
                float(_window_remaining(rules, inventory, chip, window)),
            )
    for step, gameweek in enumerate(gameweeks):
        for chip, cols in chip_cols:
            # Wildcard / Free Hit only affect priced transfer steps.
            if chip in {"wildcard", "freehit"} and step not in priced:
                _equal(solver, [cols[step]], [1.0], 0.0)
                continue
            if not _chip_playable(rules, inventory, chip, gameweek):
                _equal(solver, [cols[step]], [1.0], 0.0)
                continue
            if chip_windows is not None and not chip_windows.allows(chip, gameweek):
                _equal(solver, [cols[step]], [1.0], 0.0)
    if chip_windows is None:
        return
    for step, gameweek in enumerate(gameweeks):
        if gameweek in chip_windows.no_chip:
            for _chip, cols in chip_cols:
                _equal(solver, [cols[step]], [1.0], 0.0)
            continue
        for chip, cols in chip_cols:
            if not chip_windows.requires(chip, gameweek):
                continue
            if chip in {"wildcard", "freehit"} and step not in priced:
                raise ValueError(
                    f"cannot force {chip} on unpriced step (gw {gameweek})"
                )
            if not _chip_playable(rules, inventory, chip, gameweek):
                raise ValueError(
                    f"cannot force {chip} in gw {gameweek}: not playable under rules"
                )
            _equal(solver, [cols[step]], [1.0], 1.0)


def _gate_score(
    solver: highspy.Highs, columns: _Columns, step: int, index: int
) -> None:
    """Score permanent lineup only when Free Hit is off."""
    fh = columns.fh[step]
    pairs = (
        (columns.score_squad[step][index], columns.squad[step][index]),
        (columns.score_xi[step][index], columns.xi[step][index]),
        (columns.score_captain[step][index], columns.captain[step][index]),
        (columns.score_vice[step][index], columns.vice[step][index]),
        (columns.score_triple[step][index], columns.triple_extra[step][index]),
        (columns.score_bench[step][index], columns.bench_score[step][index]),
    )
    for scored, source in pairs:
        _pair(solver, int(scored), int(source))
        _at_most(solver, [int(scored), fh], 1)


def _free_hit_player(
    solver: highspy.Highs,
    columns: _Columns,
    step: int,
    index: int,
    *,
    armbands: bool = True,
) -> None:
    """Temporary Free Hit player vars; freeze permanent transfers that week."""
    fh = columns.fh[step]
    squad = int(columns.fh_squad[step][index])
    xi = int(columns.fh_xi[step][index])
    captain = int(columns.fh_captain[step][index])
    vice = int(columns.fh_vice[step][index])
    _pair(solver, squad, fh)
    _pair(solver, xi, squad)
    if armbands:
        _pair(solver, captain, xi)
        _pair(solver, vice, xi)
        _at_most(solver, [captain, vice], 1)
    else:
        _equal(solver, [captain], [1.0], 0.0)
        _equal(solver, [vice], [1.0], 0.0)
    _at_most(solver, [int(columns.buy[step][index]), fh], 1)
    _at_most(solver, [int(columns.sell[step][index]), fh], 1)


def _free_hit_lineup(
    solver: highspy.Highs,
    columns: _Columns,
    step: int,
    positions: list[str],
    clubs: list[int],
    *,
    armbands: bool = True,
) -> None:
    """Enforce a valid temporary squad/XI when Free Hit is on."""
    fh = columns.fh[step]
    squad = columns.fh_squad[step]
    xi = columns.fh_xi[step]
    captain = columns.fh_captain[step]
    vice = columns.fh_vice[step]
    _equal(
        solver,
        [*squad.tolist(), fh],
        [*([1.0] * columns.count), -float(_SQUAD_SIZE)],
        0.0,
    )
    _equal(
        solver,
        [*xi.tolist(), fh],
        [*([1.0] * columns.count), -float(_XI_SIZE)],
        0.0,
    )
    if armbands:
        _equal(solver, [*captain.tolist(), fh], [*([1.0] * columns.count), -1.0], 0.0)
        _equal(solver, [*vice.tolist(), fh], [*([1.0] * columns.count), -1.0], 0.0)
    else:
        _sum(solver, captain, 0, 0)
        _sum(solver, vice, 0, 0)
    for position, quota in _SQUAD.items():
        members = [i for i, name in enumerate(positions) if name == position]
        _equal(
            solver,
            [int(squad[i]) for i in members] + [fh],
            [1.0] * len(members) + [-float(quota)],
            0.0,
        )
        _row(
            solver,
            0.0,
            solver.getInfinity(),
            np.asarray(
                [int(xi[i]) for i in members] + [fh],
                dtype=np.int32,
            ),
            np.asarray(
                [1.0] * len(members) + [-float(_XI_MIN[position])],
                dtype=np.float64,
            ),
        )
        _row(
            solver,
            -solver.getInfinity(),
            0.0,
            np.asarray(
                [int(xi[i]) for i in members] + [fh],
                dtype=np.int32,
            ),
            np.asarray(
                [1.0] * len(members) + [-float(_XI_MAX[position])],
                dtype=np.float64,
            ),
        )
    for club in sorted(set(clubs)):
        chosen = [int(squad[i]) for i, key in enumerate(clubs) if key == club]
        _sum(solver, chosen, 0, _MAX_PER_CLUB)


def _free_hit_budget(
    solver: highspy.Highs,
    columns: _Columns,
    step: int,
    price: np.ndarray,
    sell_price: np.ndarray,
    bank: int,
) -> None:
    """Temporary Free Hit squad costs at most permanent selling value + bank.

    ``price`` / ``sell_price`` are ``(horizon, n)`` matrices; permanent squad
    value and earlier transfers use the step at which they occur.
    """
    chosen: list[int] = []
    weights: list[float] = []
    for index in range(columns.count):
        chosen.append(int(columns.fh_squad[step][index]))
        weights.append(float(price[step, index]))
        chosen.append(int(columns.squad[step][index]))
        weights.append(-float(sell_price[step, index]))
    for earlier in range(step + 1):
        for index in range(columns.count):
            chosen.append(int(columns.sell[earlier][index]))
            weights.append(-float(sell_price[earlier, index]))
            chosen.append(int(columns.buy[earlier][index]))
            weights.append(float(price[earlier, index]))
    _row(
        solver,
        -solver.getInfinity(),
        float(bank),
        np.asarray(chosen, dtype=np.int32),
        np.asarray(weights, dtype=np.float64),
    )


def _chip_gameweeks(
    season: str | None,
    gameweeks: list[int] | None,
    horizon: int,
) -> list[int] | None:
    if (season is None) != (gameweeks is None):
        raise ValueError("season and gameweeks must be provided together")
    if gameweeks is None:
        return None
    if len(gameweeks) != horizon:
        raise ValueError("gameweeks must have one entry per horizon step")
    for gameweek in gameweeks:
        if gameweek < 1 or gameweek > 38:
            raise ValueError(f"gameweek must be in 1..38, got {gameweek}")
    return gameweeks


def _window_remaining(
    rules: ChipRules,
    inventory: ChipInventory,
    chip: ChipName,
    window: ChipWindow,
) -> int:
    granted = sum(
        1 for grant in rules.grants if grant.name == chip and grant.window == window.id
    )
    played = sum(
        1
        for play in inventory.played
        if play.name == chip and window.start_event <= play.gw <= window.stop_event
    )
    return max(0, granted - played)


def _chip_playable(
    rules: ChipRules,
    inventory: ChipInventory,
    chip: ChipName,
    gameweek: int,
) -> bool:
    if inventory.remaining.get(chip, 0) <= 0:
        return False
    for window in rules.windows:
        if not (window.start_event <= gameweek <= window.stop_event):
            continue
        if _window_remaining(rules, inventory, chip, window) > 0:
            return True
    return False


def _transition(
    solver: highspy.Highs,
    columns: _Columns,
    owned: np.ndarray,
    horizon: int,
) -> None:
    for index, already in enumerate(owned):
        _equal(
            solver,
            [columns.squad[0][index], columns.buy[0][index], columns.sell[0][index]],
            [1.0, -1.0, 1.0],
            1.0 if already else 0.0,
        )
        fixed = columns.buy[0][index] if already else columns.sell[0][index]
        _equal(solver, [fixed], [1.0], 0.0)
    for step in range(1, horizon):
        for index in range(columns.count):
            _equal(
                solver,
                [
                    columns.squad[step][index],
                    columns.squad[step - 1][index],
                    columns.buy[step][index],
                    columns.sell[step][index],
                ],
                [1.0, -1.0, -1.0, 1.0],
                0.0,
            )
            _at_most(
                solver,
                [int(columns.buy[step][index]), int(columns.squad[step - 1][index])],
                1,
            )
            _pair(
                solver,
                int(columns.sell[step][index]),
                int(columns.squad[step - 1][index]),
            )


def _cash(
    solver: highspy.Highs,
    columns: _Columns,
    price: np.ndarray,
    sell_price: np.ndarray,
    bank: int,
    horizon: int,
) -> None:
    """Budget across steps; ``price`` / ``sell_price`` are ``(horizon, n)``."""
    limit = solver.getInfinity()
    for step in range(horizon):
        chosen: list[int] = []
        weights: list[float] = []
        for earlier in range(step + 1):
            for index in range(columns.count):
                chosen.append(int(columns.sell[earlier][index]))
                weights.append(float(sell_price[earlier, index]))
                chosen.append(int(columns.buy[earlier][index]))
                weights.append(-float(price[earlier, index]))
        _row(
            solver,
            -float(bank),
            limit,
            np.asarray(chosen, dtype=np.int32),
            np.asarray(weights, dtype=np.float64),
        )


def _transfers(
    solver: highspy.Highs,
    columns: _Columns,
    state: SquadState,
    initial: bool,
    horizon: int,
    rules: ChipRules | None,
) -> None:
    priced = _priced(horizon, initial)
    if not priced:
        return
    cap = state.ft_bank_cap
    keep_fh_ft = True if rules is None else rules.free_hit_keeps_free_transfers
    for step in priced:
        moves = columns.moves[step]
        _sum(solver, moves, 1, 1)
        _equal(
            solver,
            [*columns.buy[step].tolist(), *moves],
            [*([1.0] * columns.count), *[-float(k) for k in range(_MAX_MOVES + 1)]],
            0.0,
        )
        free_cols = [columns.free[step][free] for free in range(1, cap + 1)]
        _sum(solver, free_cols, 1, 1)
        if step == priced[0]:
            for free in range(1, cap + 1):
                value = 1.0 if free == state.free_transfers else 0.0
                _equal(solver, [columns.free[step][free]], [1.0], value)
        links = list(columns.link[step].values())
        _sum(solver, links, 1, 1)
        for free in range(1, cap + 1):
            for made in range(_MAX_MOVES + 1):
                link = columns.link[step][free, made]
                _and(solver, link, columns.free[step][free], moves[made])
                pay = columns.hit_pay[step].get((free, made))
                if pay is not None:
                    _hit_pay(solver, pay, link, columns.wc[step])
        if step + 1 in columns.free:
            _carry(solver, columns, step, cap, keep_fh_ft)


def _hit_pay(solver: highspy.Highs, pay: int, link: int, wildcard: int) -> None:
    """Charge the link's hit only when Wildcard is off: pay = link ∧ ¬wc."""
    _pair(solver, pay, link)
    _at_most(solver, [pay, wildcard], 1)
    _row(
        solver,
        -solver.getInfinity(),
        0,
        np.array([link, pay, wildcard], dtype=np.int32),
        np.array([1.0, -1.0, -1.0]),
    )


def _carry(
    solver: highspy.Highs,
    columns: _Columns,
    step: int,
    cap: int,
    keep_fh_ft: bool,
) -> None:
    """Bank free transfers, with Wildcard reset and Free Hit pass-through.

    Wildcard always leaves 1 FT next week. Free Hit with
    ``free_hit_keeps_free_transfers`` makes no permanent moves, so the
    ordinary zero-move carry already preserves banking; without that flag,
    Free Hit resets like Wildcard.
    """
    wc = columns.wc[step]
    fh = columns.fh[step]
    resets = [wc] if keep_fh_ft else [wc, fh]
    for nxt in range(1, cap + 1):
        incoming = [
            columns.link[step][free, made]
            for free in range(1, cap + 1)
            for made in range(_MAX_MOVES + 1)
            if _next_free(free, made, cap) == nxt
        ]
        free_next = columns.free[step + 1][nxt]
        # free - sum(incoming) ∈ [-R, R] with R driven by reset chips.
        _row(
            solver,
            -solver.getInfinity(),
            0.0,
            np.asarray([free_next, *incoming, *resets], dtype=np.int32),
            np.asarray(
                [1.0, *([-1.0] * len(incoming)), *([-1.0] * len(resets))],
                dtype=np.float64,
            ),
        )
        _row(
            solver,
            0.0,
            solver.getInfinity(),
            np.asarray([free_next, *incoming, *resets], dtype=np.int32),
            np.asarray(
                [1.0, *([-1.0] * len(incoming)), *([1.0] * len(resets))],
                dtype=np.float64,
            ),
        )
        if nxt == 1:
            for chip in resets:
                _pair(solver, chip, free_next)
        else:
            for chip in resets:
                _at_most(solver, [free_next, chip], 1)


def _and(solver: highspy.Highs, both: int, left: int, right: int) -> None:
    _pair(solver, both, left)
    _pair(solver, both, right)
    _row(
        solver,
        -solver.getInfinity(),
        1,
        np.array([left, right, both], dtype=np.int32),
        np.array([1.0, 1.0, -1.0]),
    )


def _read(
    solver: highspy.Highs,
    pool: pl.DataFrame,
    expected: np.ndarray,
    columns: _Columns,
    state: SquadState,
    initial: bool,
    horizon: int,
    rules: ChipRules | None,
    *,
    step_weights: Sequence[float] | None = None,
) -> GameweekPlan:
    values = np.asarray(solver.getSolution().col_value)
    priced = _priced(horizon, initial)
    rows: list[dict[str, object]] = []
    hits: list[float] = []
    available: list[int] = []
    chips: list[str | None] = []
    total = 0.0
    free = state.free_transfers
    keep_fh_ft = True if rules is None else rules.free_hit_keeps_free_transfers
    keys = pool["player_key"].to_list()
    weights = _resolve_plan_weights(state, horizon, step_weights)
    for step in range(horizon):
        chip = _chip_played(values, columns, step)
        chips.append(chip)
        bought = _on(values, columns.buy[step])
        if step not in priced:
            hits.append(0.0)
            available.append(0)
        else:
            moves = int(bought.sum())
            if chip in {"wildcard", "freehit"}:
                hits.append(0.0)
            else:
                hits.append(float(state.hit_cost * max(0, moves - free)))
            available.append(free)
            if chip == "wildcard" or (chip == "freehit" and not keep_fh_ft):
                free = 1
            elif chip == "freehit":
                free = _next_free(free, 0, state.ft_bank_cap)
            else:
                free = _next_free(free, moves, state.ft_bank_cap)
        weight = weights[step]
        if chip == "freehit":
            playing = _on(values, columns.fh_xi[step])
            armband = _on(values, columns.fh_captain[step])
            total += weight * float(expected[playing, step].sum())
            total += weight * float(expected[armband, step].sum())
        else:
            playing = _on(values, columns.score_xi[step])
            armband = _on(values, columns.score_captain[step])
            triple = _on(values, columns.score_triple[step])
            boosted = _on(values, columns.score_bench[step])
            total += weight * float(expected[playing, step].sum())
            total += weight * float(expected[armband, step].sum())
            total += weight * float(expected[triple, step].sum())
            total += weight * float(expected[boosted, step].sum())
        _append_step(rows, pool, expected, values, columns, keys, step, chip)
    total -= sum(hits)
    picks = _frame(pl.DataFrame(rows), HorizonPickSchema)
    return GameweekPlan(picks, tuple(hits), tuple(available), total, tuple(chips))


def _chip_played(values: np.ndarray, columns: _Columns, step: int) -> str | None:
    if values[columns.wc[step]] > 0.5:
        return "wildcard"
    if values[columns.fh[step]] > 0.5:
        return "freehit"
    if values[columns.bb[step]] > 0.5:
        return "bboost"
    if values[columns.tc[step]] > 0.5:
        return "3xc"
    return None


def _append_step(
    rows: list[dict[str, object]],
    pool: pl.DataFrame,
    expected: np.ndarray,
    values: np.ndarray,
    columns: _Columns,
    keys: list[int],
    step: int,
    chip: str | None,
) -> None:
    permanent = _on(values, columns.squad[step])
    if chip == "freehit":
        kept = _on(values, columns.fh_squad[step])
        playing = _on(values, columns.fh_xi[step])
        captain = _on(values, columns.fh_captain[step])
        vice = _on(values, columns.fh_vice[step])
        bought = kept & ~permanent
        sold = permanent & ~kept
    else:
        kept = permanent
        sold = _on(values, columns.sell[step])
        playing = _on(values, columns.xi[step])
        captain = _on(values, columns.captain[step])
        vice = _on(values, columns.vice[step])
        bought = _on(values, columns.buy[step])
    positions = pool["position"].to_list()
    clubs = pool["team_key"].to_list()
    prices = pool["price"].to_list()
    for index, key in enumerate(keys):
        if not kept[index] and not sold[index]:
            continue
        rows.append(
            {
                "step": step,
                "player_key": int(key),
                "position": positions[index],
                "team_key": int(clubs[index]),
                "price": int(prices[index]),
                "xpts": float(expected[index, step]),
                "in_xi": bool(playing[index]),
                "captain": bool(captain[index]),
                "vice": bool(vice[index]),
                "bought": bool(bought[index]),
                "sold": bool(sold[index]),
            }
        )


def _scaled(points: np.ndarray, weight: float) -> np.ndarray:
    scaled = np.asarray(points, dtype=np.float64) * weight
    return np.asarray(scaled, dtype=np.float64)


def _priced(horizon: int, initial: bool) -> list[int]:
    start = 1 if initial else 0
    return list(range(start, horizon))


def next_free_transfers(free: int, moves: int, cap: int) -> int:
    """Free transfers left next week after using ``moves`` of ``free``.

    Unused transfers bank, up to ``cap``. Spending more than ``free`` leaves
    one transfer next week. The hit for those extra moves is not applied here.

    Args:
        free: Transfers available before this gameweek.
        moves: Transfers made.
        cap: Maximum that can be banked.

    Returns:
        Transfers available at the next deadline.
    """
    return _next_free(free, moves, cap)


def _next_free(free: int, moves: int, cap: int) -> int:
    if moves <= free:
        return min(cap, free - moves + 1)
    return 1


def _owned(state: SquadState, initial: bool) -> list[OwnedPlayer]:
    owned = list(state.owned)
    if initial and owned:
        raise ValueError("an initial draft has no owned players")
    if not initial and len(owned) != _SQUAD_SIZE:
        raise ValueError("owned squad must contain 15 players")
    keys = [player.player_key for player in owned]
    if len(set(keys)) != len(keys):
        raise ValueError("owned player_key must be unique")
    return owned


def _expected(
    points: pl.DataFrame,
    index: dict[int, int],
    count: int,
    horizon: int,
) -> np.ndarray:
    """Accumulate xpts onto the candidate index; skip non-candidates."""
    expected = np.zeros((count, horizon), dtype=np.float64)
    for row in points.iter_rows(named=True):
        step = int(row["step"])
        if step >= horizon:
            raise ValueError(f"point step {step} is outside the horizon")
        key = int(row["player_key"])
        if key not in index:
            # Not priced at this deadline — cannot be selected.
            continue
        expected[index[key], step] += float(row["xpts"])
    return expected


def _raf_active(long_horizon: LongHorizonPolicy, horizon: int) -> bool:
    """True when relax-and-fix will leave far steps continuous."""
    return (
        long_horizon.relax_and_fix
        and long_horizon.should_auto_apply(horizon)
        and long_horizon.binary_steps < horizon
    )


def _integral_steps(long_horizon: LongHorizonPolicy, horizon: int) -> tuple[int, ...]:
    """Horizon steps that remain integer after the final HiGHS solve.

    Under relax-and-fix, only the leading ``binary_steps`` stay integer
    (far steps are continuous look-ahead). Without RAF, every step is
    integral.
    """
    if not _raf_active(long_horizon, horizon):
        return tuple(range(horizon))
    n_binary = min(long_horizon.binary_steps, horizon)
    return tuple(range(n_binary))


def _assert_owned_locked_fifteen(
    pool: pl.DataFrame,
    owned: list[OwnedPlayer],
    *,
    initial: bool,
) -> None:
    """Fail fast when a non-draft owned squad is not a legal fifteen."""
    if initial or not owned:
        return
    keys = [player.player_key for player in owned]
    owned_rows = pool.filter(pl.col("player_key").is_in(keys))
    missing = sorted(set(keys) - set(owned_rows["player_key"].to_list()))
    if missing:
        raise ValueError(f"owned players missing from candidate pool: {missing}")
    issue = locked_fifteen_issue(owned_rows)
    if issue is not None:
        raise ValueError(f"owned squad before plan_squad: {issue}")


def _run_solver(
    solver: highspy.Highs,
    *,
    columns: _Columns,
    options: HighsSolveOptions,
    horizon: int,
    long_horizon: LongHorizonPolicy,
) -> highspy.HighsModelStatus:
    """Run HiGHS, optionally via relax-and-fix for long horizons.

    When ``long_horizon.relax_and_fix`` applies, keep the first
    ``binary_steps`` integer, relax later steps to continuous, solve, fix
    week-1 variables to the incumbent, then re-solve with mid-horizon
    steps restored to integer (far steps stay continuous).
    """
    use_raf = _raf_active(long_horizon, horizon)
    if not use_raf:
        solver.run()
        return solver.getModelStatus()

    binary_steps = min(long_horizon.binary_steps, horizon)
    far = _column_indices_for_steps(columns, range(binary_steps, horizon))
    _set_integrality(solver, far, integer=False)
    _log.info(
        "relax-and-fix phase1 binary_steps=%s far_vars=%s",
        binary_steps,
        len(far),
    )
    solver.run()
    status = solver.getModelStatus()
    if not options.accepts_status(status):
        return status

    values = np.asarray(solver.getSolution().col_value, dtype=np.float64)
    week1 = _column_indices_for_steps(columns, range(0, 1))
    _fix_columns(solver, week1, values)
    # Seed the second solve with the relaxed incumbent (week-1 already fixed).
    solver.setSolution(
        len(values),
        np.arange(len(values), dtype=np.int32),
        values,
    )
    mid = _column_indices_for_steps(columns, range(1, binary_steps))
    if mid:
        _set_integrality(solver, mid, integer=True)
    _log.info(
        "relax-and-fix phase2 fixed_week1=%s mid_integer=%s",
        len(week1),
        len(mid),
    )
    solver.clearSolver()
    solver.run()
    return solver.getModelStatus()


def _column_indices_for_steps(
    columns: _Columns,
    steps: range,
) -> list[int]:
    """Collect all variable indices belonging to the given horizon steps."""
    indices: list[int] = []
    for step in steps:
        if step < 0 or step >= columns.horizon:
            continue
        for block in (
            columns.squad[step],
            columns.xi[step],
            columns.captain[step],
            columns.vice[step],
            columns.buy[step],
            columns.sell[step],
            columns.triple_extra[step],
            columns.bench_score[step],
            columns.score_squad[step],
            columns.score_xi[step],
            columns.score_captain[step],
            columns.score_vice[step],
            columns.score_triple[step],
            columns.score_bench[step],
            columns.fh_squad[step],
            columns.fh_xi[step],
            columns.fh_captain[step],
            columns.fh_vice[step],
        ):
            indices.extend(int(index) for index in block)
        indices.extend(
            [
                columns.wc[step],
                columns.fh[step],
                columns.bb[step],
                columns.tc[step],
            ]
        )
        if step in columns.moves:
            indices.extend(int(index) for index in columns.moves[step])
            indices.extend(int(index) for index in columns.free[step].values())
            indices.extend(int(index) for index in columns.link[step].values())
            indices.extend(int(index) for index in columns.hit_pay[step].values())
    return indices


def _set_integrality(
    solver: highspy.Highs,
    indices: list[int],
    *,
    integer: bool,
) -> None:
    """Mark ``indices`` as integer or continuous."""
    if not indices:
        return
    kind = (
        int(highspy.HighsVarType.kInteger)
        if integer
        else int(highspy.HighsVarType.kContinuous)
    )
    cols = np.asarray(indices, dtype=np.int32)
    kinds = np.full(cols.shape[0], kind, dtype=np.uint8)
    status = solver.changeColsIntegrality(cols.shape[0], cols, kinds)
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS rejected integrality change: {status}")


def _fix_columns(
    solver: highspy.Highs,
    indices: list[int],
    values: np.ndarray,
) -> None:
    """Pin columns to 0/1 from a (possibly fractional) incumbent."""
    for index in indices:
        fixed = 1.0 if float(values[index]) > 0.5 else 0.0
        status = solver.changeColBounds(int(index), fixed, fixed)
        if status != highspy.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS rejected fix of col {index}: {status}")


def shift_gameweek_plan(plan: GameweekPlan, *, steps: int = 1) -> GameweekPlan:
    """Shift a prior ``GameweekPlan`` forward for use as a MIP warm start.

    Args:
        plan: Solved plan from the previous deadline (or an earlier solve).
        steps: Leading steps to drop (1 maps last week's step 1 → this week's 0).

    Returns:
        A plan whose picks/hits/free_transfers/chips start at the new step 0.
        ``points`` and solve metadata are cleared (not meaningful after shift).

    Raises:
        ValueError: ``steps`` is negative.
    """
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")
    picks = shift_horizon_picks(plan.picks, steps=steps)
    hits = plan.hits[steps:] if steps < len(plan.hits) else ()
    free = plan.free_transfers[steps:] if steps < len(plan.free_transfers) else ()
    chips = plan.chips[steps:] if steps < len(plan.chips) else ()
    return replace(
        plan,
        picks=picks,
        hits=tuple(hits),
        free_transfers=tuple(free),
        chips=tuple(chips),
        points=0.0,
        solve_elapsed=None,
        mip_status=None,
        pool_size=None,
    )


def _apply_warm_start(
    solver: highspy.Highs,
    columns: _Columns,
    keys: list[int],
    prior: GameweekPlan,
    *,
    shift: int,
) -> None:
    """Seed HiGHS with lineup / transfer / chip vars from a prior plan."""
    if shift < 0:
        raise ValueError(f"warm_start_shift must be >= 0, got {shift}")
    picks = shift_horizon_picks(prior.picks, steps=shift)
    if picks.is_empty():
        return
    index = {int(key): pos for pos, key in enumerate(keys)}
    chips = prior.chips[shift:] if shift < len(prior.chips) else ()
    col_indices: list[int] = []
    col_values: list[float] = []
    for step in sorted(int(value) for value in picks["step"].unique().to_list()):
        if step >= columns.horizon:
            continue
        step_rows = picks.filter(pl.col("step") == step)
        for row in step_rows.iter_rows(named=True):
            key = int(row["player_key"])
            at = index.get(key)
            if at is None:
                continue
            if bool(row["sold"]):
                col_indices.append(int(columns.sell[step][at]))
                col_values.append(1.0)
                continue
            col_indices.append(int(columns.squad[step][at]))
            col_values.append(1.0)
            if bool(row["in_xi"]):
                col_indices.append(int(columns.xi[step][at]))
                col_values.append(1.0)
            if bool(row["captain"]):
                col_indices.append(int(columns.captain[step][at]))
                col_values.append(1.0)
            if bool(row["vice"]):
                col_indices.append(int(columns.vice[step][at]))
                col_values.append(1.0)
            if bool(row["bought"]):
                col_indices.append(int(columns.buy[step][at]))
                col_values.append(1.0)
        if step < len(chips):
            _append_chip_start(col_indices, col_values, columns, step, chips[step])
    if not col_indices:
        return
    status = solver.setSolution(
        len(col_indices),
        np.asarray(col_indices, dtype=np.int32),
        np.asarray(col_values, dtype=np.float64),
    )
    if status != highspy.HighsStatus.kOk:
        _log.debug("warm start rejected by HiGHS: %s", status)


def _append_chip_start(
    indices: list[int],
    values: list[float],
    columns: _Columns,
    step: int,
    chip: str | None,
) -> None:
    if chip == "wildcard":
        indices.append(columns.wc[step])
        values.append(1.0)
    elif chip == "freehit":
        indices.append(columns.fh[step])
        values.append(1.0)
    elif chip == "bboost":
        indices.append(columns.bb[step])
        values.append(1.0)
    elif chip == "3xc":
        indices.append(columns.tc[step])
        values.append(1.0)


def _resolve_plan_weights(
    state: SquadState,
    horizon: int,
    step_weights: Sequence[float] | None,
) -> tuple[float, ...]:
    """Return per-step objective weights (geometric decay or explicit)."""
    if step_weights is None:
        return geometric_step_weights(float(state.decay), horizon)
    return validate_step_weights(step_weights, horizon=horizon)


def _solver(cost: np.ndarray, options: HighsSolveOptions) -> highspy.Highs:
    solver = highspy.Highs()
    options.apply(solver)
    _add_binaries(solver, cost)
    return solver


def _equal(
    solver: highspy.Highs,
    columns: list[int],
    weights: list[float],
    value: float,
) -> None:
    _row(
        solver,
        value,
        value,
        np.asarray(columns, dtype=np.int32),
        np.asarray(weights, dtype=np.float64),
    )


def _on(values: np.ndarray, columns: np.ndarray) -> np.ndarray:
    return values[columns] > 0.5


def _label(player_key: int, names: Mapping[int, str] | None) -> str:
    if names is None:
        return str(player_key)
    name = names.get(int(player_key), "").strip()
    return name or str(player_key)


def _chips_for_horizon(chips: tuple[str | None, ...], horizon: int) -> list[str | None]:
    """Pad or trim per-step chips to the plan horizon length."""
    values = list(chips)
    if len(values) < horizon:
        values.extend([None] * (horizon - len(values)))
    return values[:horizon]


def _frame(frame: pl.DataFrame, schema: _Schema) -> pl.DataFrame:
    if schema is PlanSchema:
        frame = ensure_plan_chip_column(frame)
    checked = schema.validate(frame)
    checked = require_dataframe(checked)
    return checked
