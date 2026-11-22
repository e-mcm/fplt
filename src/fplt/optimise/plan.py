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

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import highspy
import numpy as np
import polars as pl

from fplt.optimise.chip_windows import ChipUseWindows
from fplt.optimise.prices import fpl_selling_price
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
from fplt.optimise.state import ChipInventory, OwnedPlayer, SquadState
from fplt.schemas.plan import (
    CandidateSchema,
    HorizonPickSchema,
    HorizonPointsSchema,
    PlanSchema,
)
from fplt.scoring.chips import ChipName, ChipRules, ChipWindow

_Schema = (
    type[CandidateSchema]
    | type[HorizonPickSchema]
    | type[HorizonPointsSchema]
    | type[PlanSchema]
)

_MAX_MOVES = 15
_SQUAD_SIZE = 15

# Default look-ahead for ``fplt plan`` / ``fplt decisions`` / ``fplt apply``.
# Chosen from the 2024-25 chips-off H∈{1..4} season-points sweep (H=3 best
# cross-model mean; H=4 matched H=3 actions at higher MILP cost).
DEFAULT_PLAN_HORIZON = 3


@dataclass(frozen=True)
class GameweekPlan:
    """Solved horizon. ``points`` is decayed XI plus captain, minus hits.

    ``chips`` is one entry per horizon step: ``\"wildcard\"``, ``\"freehit\"``,
    ``\"bboost\"``, ``\"3xc\"``, or ``None``.
    """

    picks: pl.DataFrame
    hits: tuple[float, ...]
    free_transfers: tuple[int, ...]
    points: float
    chips: tuple[str | None, ...] = ()


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

    Returns:
        Squad members and sales at every step.

    Raises:
        ValueError: The inputs disagree, or no squad meets the rules.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    targets = _chip_gameweeks(season, gameweeks, horizon)
    rules_season = chip_rules_season if chip_rules_season is not None else season
    if chip_windows is not None and targets is None:
        raise ValueError("chip_windows requires season and gameweeks")
    rules = ChipRules.for_season(rules_season) if rules_season is not None else None
    pool = _frame(players.sort("player_key"), CandidateSchema)
    scored = _frame(points, HorizonPointsSchema)
    owned = _owned(state, initial)
    keys = pool["player_key"].to_list()
    index = {int(key): pos for pos, key in enumerate(keys)}
    missing = [player.player_key for player in owned if player.player_key not in index]
    if missing:
        raise ValueError(f"owned players are not in the pool: {missing}")
    count = pool.height
    expected = _expected(scored, index, count, horizon)
    list_price = pool["price"].to_numpy().astype(np.float64)
    # As-of sell-on only: within-horizon steps keep these credits (no price
    # forecasts). purchase_price is sticky; market comes from candidates.
    sell_price = list_price.copy()
    owned_at = np.zeros(count, dtype=bool)
    for player in owned:
        at = index[player.player_key]
        owned_at[at] = True
        sell_price[at] = float(
            fpl_selling_price(player.purchase_price, int(list_price[at]))
        )
    columns = _columns(expected, state, horizon, initial)
    solver = _solver(columns.costs)
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
    )
    solver.setMaximize()
    solver.run()
    if solver.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise ValueError("no plan satisfies the budget, positions, and club limits")
    return _read(solver, pool, expected, columns, state, initial, horizon, rules)


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
    targets = pl.DataFrame(
        {
            "step": list(range(len(gameweeks))),
            "target_gw": gameweeks,
            "hit": list(plan.hits),
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
        ),
        PlanSchema,
    )


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


def plan_parquet(
    root: Path, model_id: str, season: str, gameweek: int, horizon: int
) -> Path:
    """Parquet path for one model's plan at one deadline and horizon.

    Args:
        root: Data directory that contains ``plans/``.
        model_id: Predictor id. This is a hash of the model spec, not the plan.
        season: Season label.
        gameweek: Deadline the plan was built at.
        horizon: Weeks the plan covers, including that deadline.

    Returns:
        ``plans/<model_id>/as_of=<season>_gwNN_hHH/part.parquet``.
    """
    stamp = f"as_of={season}_gw{gameweek:02d}_h{horizon:02d}"
    return root / "plans" / model_id / stamp / "part.parquet"


def write_plan(root: Path, frame: pl.DataFrame) -> Path:
    """Write a plan parquet. The same deadline and horizon are never replaced.

    Args:
        root: Data directory that contains ``plans/``.
        frame: Rows matching ``PlanSchema`` for one model and one deadline.

    Returns:
        The parquet path.

    Raises:
        FileExistsError: That deadline and horizon were already written.
        ValueError: The frame mixes deadlines or skips a horizon step.
    """
    checked = _frame(frame, PlanSchema)
    seasons = checked["as_of_season"].unique().to_list()
    gameweeks = checked["as_of_gw"].unique().to_list()
    models = checked["model_id"].unique().to_list()
    if len(seasons) != 1 or len(gameweeks) != 1 or len(models) != 1:
        raise ValueError("a plan file is one model and one deadline")
    steps = sorted(int(step) for step in checked["step"].unique().to_list())
    horizon = max(steps) + 1 if steps else 0
    if steps != list(range(horizon)):
        raise ValueError("a plan file needs one row set per horizon step")
    path = plan_parquet(
        root, str(models[0]), str(seasons[0]), int(gameweeks[0]), horizon
    )
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checked.write_parquet(path)
    return path


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
) -> _Columns:
    count = expected.shape[0]
    columns = _Columns(count, horizon)
    zeros = np.zeros(count, dtype=np.float64)
    for step in range(horizon):
        weight = state.decay**step
        points = np.asarray(expected[:, step], dtype=np.float64)
        bench = _scaled(points, weight * state.bench_weight)
        played = _scaled(points, weight * (1.0 - state.bench_weight))
        captain = _scaled(points, weight)
        vice = _scaled(points, weight * _VICE_WEIGHT)
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
        columns.score_captain.append(columns.block(captain))
        columns.score_vice.append(columns.block(vice))
        columns.score_triple.append(columns.block(captain))
        columns.score_bench.append(columns.block(captain))
        # Temporary Free Hit lineup (gated by fh_t).
        columns.fh_squad.append(columns.block(bench))
        columns.fh_xi.append(columns.block(played))
        columns.fh_captain.append(columns.block(captain))
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
) -> None:
    positions = pool["position"].to_list()
    clubs = pool["team_key"].to_list()
    for step in range(horizon):
        _lineup(
            solver,
            columns.squad[step],
            columns.xi[step],
            columns.captain[step],
            columns.vice[step],
            positions,
            clubs,
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
    )


def _lineup(
    solver: highspy.Highs,
    squad: np.ndarray,
    xi: np.ndarray,
    captain: np.ndarray,
    vice: np.ndarray,
    positions: list[str],
    clubs: list[int],
) -> None:
    _sum(solver, squad, _SQUAD_SIZE, _SQUAD_SIZE)
    _sum(solver, xi, _XI_SIZE, _XI_SIZE)
    _sum(solver, captain, 1, 1)
    _sum(solver, vice, 1, 1)
    for index in range(squad.shape[0]):
        _pair(solver, int(xi[index]), int(squad[index]))
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
) -> None:
    """Wire chip binaries, Free Hit temporary squad, and BB/TC scoring gates."""
    priced = set(_priced(horizon, initial))
    for step in range(horizon):
        for index in range(columns.count):
            extra = int(columns.triple_extra[step][index])
            bench = int(columns.bench_score[step][index])
            _pair(solver, extra, int(columns.captain[step][index]))
            _pair(solver, extra, columns.tc[step])
            _pair(solver, bench, int(columns.squad[step][index]))
            _at_most(solver, [bench, int(columns.xi[step][index])], 1)
            _pair(solver, bench, columns.bb[step])
            _gate_score(solver, columns, step, index)
            _free_hit_player(solver, columns, step, index)
        _free_hit_lineup(solver, columns, step, positions, clubs)
        _free_hit_budget(solver, columns, step, price, sell_price, bank)
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
    solver: highspy.Highs, columns: _Columns, step: int, index: int
) -> None:
    """Temporary Free Hit player vars; freeze permanent transfers that week."""
    fh = columns.fh[step]
    squad = int(columns.fh_squad[step][index])
    xi = int(columns.fh_xi[step][index])
    captain = int(columns.fh_captain[step][index])
    vice = int(columns.fh_vice[step][index])
    _pair(solver, squad, fh)
    _pair(solver, xi, squad)
    _pair(solver, captain, xi)
    _pair(solver, vice, xi)
    _at_most(solver, [captain, vice], 1)
    _at_most(solver, [int(columns.buy[step][index]), fh], 1)
    _at_most(solver, [int(columns.sell[step][index]), fh], 1)


def _free_hit_lineup(
    solver: highspy.Highs,
    columns: _Columns,
    step: int,
    positions: list[str],
    clubs: list[int],
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
    _equal(solver, [*captain.tolist(), fh], [*([1.0] * columns.count), -1.0], 0.0)
    _equal(solver, [*vice.tolist(), fh], [*([1.0] * columns.count), -1.0], 0.0)
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
    """Temporary Free Hit squad costs at most permanent selling value + bank."""
    chosen: list[int] = []
    weights: list[float] = []
    for index in range(columns.count):
        chosen.append(int(columns.fh_squad[step][index]))
        weights.append(float(price[index]))
        chosen.append(int(columns.squad[step][index]))
        weights.append(-float(sell_price[index]))
    for earlier in range(step + 1):
        for index in range(columns.count):
            chosen.append(int(columns.sell[earlier][index]))
            weights.append(-float(sell_price[index]))
            chosen.append(int(columns.buy[earlier][index]))
            weights.append(float(price[index]))
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
    limit = solver.getInfinity()
    for step in range(horizon):
        chosen: list[int] = []
        weights: list[float] = []
        for earlier in range(step + 1):
            for index in range(columns.count):
                chosen.append(int(columns.sell[earlier][index]))
                weights.append(float(sell_price[index]))
                chosen.append(int(columns.buy[earlier][index]))
                weights.append(-float(price[index]))
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
        weight = state.decay**step
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


def _solver(cost: np.ndarray) -> highspy.Highs:
    solver = highspy.Highs()
    solver.setOptionValue("output_flag", False)
    solver.setOptionValue("mip_rel_gap", 0)
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


def _frame(frame: pl.DataFrame, schema: _Schema) -> pl.DataFrame:
    checked = schema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked
