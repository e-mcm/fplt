"""Single-gameweek squad, XI, captain, and vice via HiGHS.

Player-fixture predictions are summed to a player-gameweek here. The
objective doubles the captain. Optional ``p_haul`` tilts captaincy (Wave D1
M3) without changing who is in the fifteen. The vice is the best other
starter by a tie-break small enough that it cannot change who plays or who
captains.
"""

from __future__ import annotations

from dataclasses import dataclass

import highspy
import numpy as np
import polars as pl

from fplt.schemas.squad import GameweekPointsSchema, SquadPickSchema, SquadPlayerSchema

_Schema = type[GameweekPointsSchema] | type[SquadPickSchema] | type[SquadPlayerSchema]

_BUDGET = 1000
_SQUAD = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
_XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
_XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
_XI_SIZE = 11
_MAX_PER_CLUB = 3
_VICE_WEIGHT = 1e-6
_DEFAULT_HAUL_WEIGHT = 0.05
_REQUIRED = {"player_key", "position", "team_key", "price", "xpts"}


@dataclass(frozen=True)
class SquadSelection:
    """The chosen fifteen and the expected points with the captain doubled."""

    players: pl.DataFrame
    total_xpts: float


def aggregate_gameweek(predictions: pl.DataFrame, season: str, gw: int) -> pl.DataFrame:
    """Sum player-fixture points to one row per player.

    Args:
        predictions: Columns ``player_key``, ``target_season``, ``target_gw``,
            and ``xpts``. A double gameweek contributes two rows.
        season: Season label to keep.
        gw: Gameweek to keep.

    Returns:
        A frame matching ``GameweekPointsSchema``.

    Raises:
        ValueError: A required column is missing.
    """
    missing = {"player_key", "target_season", "target_gw", "xpts"} - set(
        predictions.columns
    )
    if missing:
        raise ValueError(f"predictions is missing {sorted(missing)}")
    chosen = predictions.filter(
        (pl.col("target_season") == season) & (pl.col("target_gw") == gw)
    )
    if chosen.is_empty():
        frame = pl.DataFrame(schema={"player_key": pl.Int64(), "xpts": pl.Float64()})
    else:
        frame = chosen.group_by("player_key").agg(pl.col("xpts").sum())
    return _checked(frame, GameweekPointsSchema)


def select_squad(
    players: pl.DataFrame,
    *,
    haul_weight: float = _DEFAULT_HAUL_WEIGHT,
) -> SquadSelection:
    """Choose a squad, a starting XI, a captain, and a vice.

    Args:
        players: One row per candidate. ``SquadPlayerSchema`` columns, plus
            optional ``p_haul`` (0–1) used only for captaincy.
        haul_weight: Weight on ``p_haul`` added to the captain objective so
            near-ties prefer haul probability (M3). Ignored when ``p_haul``
            is absent. ``price`` is in tenths of a million, so 100 is £1.0m.

    Returns:
        The fifteen selected players and the XI total with the captain doubled
        (raw ``xpts``, not haul-tilted).

    Raises:
        ValueError: The pool is missing a column, repeats a player, or cannot
            meet the squad rules.
    """
    missing = _REQUIRED - set(players.columns)
    if missing:
        raise ValueError(f"players is missing {sorted(missing)}")
    pool = players.select(sorted(_REQUIRED))
    if pool["player_key"].n_unique() != pool.height:
        raise ValueError("player_key must be unique")
    checked = _checked(pool, SquadPlayerSchema).sort("player_key")
    haul = None
    if "p_haul" in players.columns:
        haul = (
            players.select("player_key", "p_haul")
            .unique(subset=["player_key"])
            .sort("player_key")
        )
        haul = checked.join(haul, on="player_key", how="left")["p_haul"].to_numpy()
        haul = np.nan_to_num(np.asarray(haul, dtype=np.float64), nan=0.0)
        haul = np.clip(haul, 0.0, 1.0)
    chosen, total = _solve(checked, haul=haul, haul_weight=haul_weight)
    return SquadSelection(players=chosen, total_xpts=total)


def _solve(
    pool: pl.DataFrame,
    *,
    haul: np.ndarray | None = None,
    haul_weight: float = _DEFAULT_HAUL_WEIGHT,
) -> tuple[pl.DataFrame, float]:
    count = pool.height
    points = pool["xpts"].to_numpy().astype(np.float64)
    captain_points = points
    if haul is not None:
        captain_points = points + float(haul_weight) * haul
    cost = np.zeros(4 * count, dtype=np.float64)
    cost[count : 2 * count] = points
    cost[2 * count : 3 * count] = captain_points
    cost[3 * count : 4 * count] = points * _VICE_WEIGHT
    solver = highspy.Highs()
    solver.setOptionValue("output_flag", False)
    _add_binaries(solver, cost)
    _constrain(solver, pool)
    solver.setMaximize()
    solver.run()
    if solver.getModelStatus() != highspy.HighsModelStatus.kOptimal:
        raise ValueError("no squad satisfies the budget, positions, and club limits")
    values = np.asarray(solver.getSolution().col_value)
    picked = values[:count] > 0.5
    starting = values[count : 2 * count] > 0.5
    captain = values[2 * count : 3 * count] > 0.5
    vice = values[3 * count : 4 * count] > 0.5
    frame = pool.with_columns(
        pl.Series("in_xi", starting),
        pl.Series("captain", captain),
        pl.Series("vice", vice),
    ).filter(pl.Series(picked))
    selected = _checked(
        frame.select(
            "player_key",
            "position",
            "team_key",
            "price",
            "xpts",
            "in_xi",
            "captain",
            "vice",
        ),
        SquadPickSchema,
    )
    total = float(points[starting].sum() + points[captain].sum())
    return selected, total


def _add_binaries(solver: highspy.Highs, cost: np.ndarray) -> None:
    count = cost.shape[0]
    zeros = np.zeros(count, dtype=np.int32)
    status = solver.addCols(
        count,
        cost,
        np.zeros(count),
        np.ones(count),
        0,
        zeros,
        np.zeros(0, dtype=np.int32),
        np.zeros(0),
    )
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS rejected the variables: {status}")
    kinds = np.full(count, int(highspy.HighsVarType.kInteger), dtype=np.uint8)
    status = solver.changeColsIntegrality(
        count, np.arange(count, dtype=np.int32), kinds
    )
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS rejected integrality: {status}")


def _constrain(solver: highspy.Highs, pool: pl.DataFrame) -> None:
    count = pool.height
    squad = np.arange(count, dtype=np.int32)
    xi = squad + count
    captain = squad + 2 * count
    vice = squad + 3 * count
    _sum(solver, squad, 15, 15)
    _sum(solver, xi, _XI_SIZE, _XI_SIZE)
    _sum(solver, captain, 1, 1)
    _sum(solver, vice, 1, 1)
    for index in range(count):
        _pair(solver, xi[index], squad[index])
        _pair(solver, captain[index], xi[index])
        _pair(solver, vice[index], xi[index])
        _at_most(solver, [int(captain[index]), int(vice[index])], 1)
    prices = pool["price"].to_numpy().astype(np.float64)
    _weighted(solver, squad, prices, -solver.getInfinity(), _BUDGET)
    positions = pool["position"].to_list()
    clubs = pool["team_key"].to_list()
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
        columns = [int(squad[i]) for i, key in enumerate(clubs) if key == club]
        _sum(solver, columns, 0, _MAX_PER_CLUB)


def _sum(
    solver: highspy.Highs,
    columns: list[int] | np.ndarray,
    lower: float,
    upper: float,
) -> None:
    cols = np.asarray(columns, dtype=np.int32)
    weights = np.ones(cols.shape[0], dtype=np.float64)
    _row(solver, lower, upper, cols, weights)


def _weighted(
    solver: highspy.Highs,
    columns: np.ndarray,
    weights: np.ndarray,
    lower: float,
    upper: float,
) -> None:
    _row(solver, lower, upper, columns.astype(np.int32), weights)


def _pair(solver: highspy.Highs, child: int, parent: int) -> None:
    _row(
        solver,
        -solver.getInfinity(),
        0,
        np.array([child, parent], dtype=np.int32),
        np.array([1.0, -1.0]),
    )


def _at_most(solver: highspy.Highs, columns: list[int], limit: float) -> None:
    _sum(solver, columns, -solver.getInfinity(), limit)


def _row(
    solver: highspy.Highs,
    lower: float,
    upper: float,
    columns: np.ndarray,
    weights: np.ndarray,
) -> None:
    status = solver.addRow(lower, upper, int(columns.shape[0]), columns, weights)
    if status != highspy.HighsStatus.kOk:
        raise RuntimeError(f"HiGHS rejected a constraint: {status}")


def _checked(frame: pl.DataFrame, schema: _Schema) -> pl.DataFrame:
    checked = schema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked
