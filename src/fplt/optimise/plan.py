"""Multi-period squad plan: transfers, free-transfer banking, and hits.

The first period is the only one a caller should act on. Later periods exist
so a free transfer is kept when it is worth more next week. Hits are four
points (or ``SquadState.hit_cost``) for each transfer beyond the free ones,
and they are not discounted. Points are. A blank ``owned`` squad is an
initial draft and those buys are not transfers.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

import highspy
import numpy as np
import polars as pl

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
from fplt.optimise.state import OwnedPlayer, SquadState
from fplt.schemas.plan import (
    CandidateSchema,
    HorizonPickSchema,
    HorizonPointsSchema,
    PlanSchema,
)

_Schema = (
    type[CandidateSchema]
    | type[HorizonPickSchema]
    | type[HorizonPointsSchema]
    | type[PlanSchema]
)

_MAX_MOVES = 15
_SQUAD_SIZE = 15


@dataclass(frozen=True)
class GameweekPlan:
    """Solved horizon. ``points`` is decayed XI plus captain, minus hits."""

    picks: pl.DataFrame
    hits: tuple[float, ...]
    free_transfers: tuple[int, ...]
    points: float


def plan_squad(
    players: pl.DataFrame,
    points: pl.DataFrame,
    state: SquadState,
    *,
    initial: bool,
    horizon: int,
) -> GameweekPlan:
    """Solve one squad over ``horizon`` gameweeks.

    Args:
        players: Candidates matching ``CandidateSchema``.
        points: ``HorizonPointsSchema``. A missing step is zero points.
        state: Bank, free transfers, and owned selling prices.
        initial: The owned list is empty and the first week is a draft.
        horizon: Number of gameweeks, including the one to act on.

    Returns:
        Squad members and sales at every step.

    Raises:
        ValueError: The inputs disagree, or no squad meets the rules.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
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
    sell_price = pool["price"].to_numpy().astype(np.float64)
    owned_at = np.zeros(count, dtype=bool)
    for player in owned:
        at = index[player.player_key]
        owned_at[at] = True
        sell_price[at] = float(player.selling_price)
    columns = _columns(expected, state, horizon, initial)
    solver = _solver(columns.costs)
    _constrain(
        solver,
        pool,
        columns,
        owned_at,
        pool["price"].to_numpy().astype(np.float64),
        sell_price,
        state,
        initial,
        horizon,
    )
    solver.setMaximize()
    solver.run()
    if solver.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise ValueError("no plan satisfies the budget, positions, and club limits")
    return _read(solver, pool, expected, columns, state, initial, horizon)


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
        self.moves: dict[int, list[int]] = {}
        self.free: dict[int, dict[int, int]] = {}
        self.link: dict[int, dict[tuple[int, int], int]] = {}

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
    for step in range(horizon):
        weight = state.decay**step
        points = np.asarray(expected[:, step], dtype=np.float64)
        bench = _scaled(points, weight * state.bench_weight)
        columns.squad.append(columns.block(bench))
        played = 1.0 - state.bench_weight
        columns.xi.append(columns.block(_scaled(points, weight * played)))
        columns.captain.append(columns.block(_scaled(points, weight)))
        columns.vice.append(columns.block(_scaled(points, weight * _VICE_WEIGHT)))
        zeros = np.zeros(count, dtype=np.float64)
        columns.buy.append(columns.block(zeros))
        columns.sell.append(columns.block(zeros))
    for step in _priced(horizon, initial):
        columns.moves[step] = [columns.flag() for _ in range(_MAX_MOVES + 1)]
        columns.free[step] = {
            free: columns.flag() for free in range(1, state.ft_bank_cap + 1)
        }
        columns.link[step] = {}
        for free in range(1, state.ft_bank_cap + 1):
            for moves in range(_MAX_MOVES + 1):
                penalty = state.hit_cost * max(0, moves - free)
                columns.link[step][free, moves] = columns.flag(-float(penalty))
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
    _transfers(solver, columns, state, initial, horizon)


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
) -> None:
    priced = _priced(horizon, initial)
    if not priced:
        return
    cap = state.ft_bank_cap
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
                _and(
                    solver,
                    columns.link[step][free, made],
                    columns.free[step][free],
                    moves[made],
                )
        if step + 1 in columns.free:
            _carry(solver, columns, step, cap)


def _carry(solver: highspy.Highs, columns: _Columns, step: int, cap: int) -> None:
    for nxt in range(1, cap + 1):
        incoming = [
            columns.link[step][free, made]
            for free in range(1, cap + 1)
            for made in range(_MAX_MOVES + 1)
            if _next_free(free, made, cap) == nxt
        ]
        _equal(
            solver,
            [columns.free[step + 1][nxt], *incoming],
            [1.0, *([-1.0] * len(incoming))],
            0.0,
        )


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
) -> GameweekPlan:
    values = np.asarray(solver.getSolution().col_value)
    priced = _priced(horizon, initial)
    rows: list[dict[str, object]] = []
    hits: list[float] = []
    available: list[int] = []
    total = 0.0
    free = state.free_transfers
    keys = pool["player_key"].to_list()
    for step in range(horizon):
        bought = _on(values, columns.buy[step])
        if step not in priced:
            hits.append(0.0)
            available.append(0)
        else:
            moves = int(bought.sum())
            hits.append(float(state.hit_cost * max(0, moves - free)))
            available.append(free)
            free = _next_free(free, moves, state.ft_bank_cap)
        playing = _on(values, columns.xi[step])
        armband = _on(values, columns.captain[step])
        weight = state.decay**step
        total += weight * float(expected[playing, step].sum())
        total += weight * float(expected[armband, step].sum())
        _append_step(rows, pool, expected, values, columns, keys, step)
    total -= sum(hits)
    picks = _frame(pl.DataFrame(rows), HorizonPickSchema)
    return GameweekPlan(picks, tuple(hits), tuple(available), total)


def _append_step(
    rows: list[dict[str, object]],
    pool: pl.DataFrame,
    expected: np.ndarray,
    values: np.ndarray,
    columns: _Columns,
    keys: list[int],
    step: int,
) -> None:
    kept = _on(values, columns.squad[step])
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
    expected = np.zeros((count, horizon), dtype=np.float64)
    for row in points.iter_rows(named=True):
        step = int(row["step"])
        if step >= horizon:
            raise ValueError(f"point step {step} is outside the horizon")
        key = int(row["player_key"])
        if key not in index:
            raise ValueError(f"points refer to unknown player {key}")
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
