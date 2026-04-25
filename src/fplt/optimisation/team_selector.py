"""Optimal starting XI and captain selector via integer linear programme.

Solves:
    maximise  sum_i( xpts_i * (s_i + c_i) )
    subject to:
        - Budget ≤ £100m
        - Squad of exactly 15 (2 GK, 5 DEF, 5 MID, 3 FWD)
        - Starting XI of exactly 11
        - Valid formation: 1 GK, 3-5 DEF, 2-5 MID, 1-3 FWD
        - Max 3 players per club
        - Exactly 1 captain (doubles their points)
        - s_i ≤ x_i  (can only start if in squad)
        - c_i ≤ s_i  (can only captain if starting)

Uses PuLP (already a project dependency).
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pandera.pandas as pa
import pulp
from pandera.typing import DataFrame

from fplt.schemas import PlayerPoolSchema

_BUDGET = 1000  # in tenths of a million (FPL value units: 100 = £1m)
_SQUAD_SIZE = 15
_XI_SIZE = 11
_CAPTAIN_COUNT = 1

_POSITION_SQUAD = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
_POSITION_XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
_POSITION_XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
_MAX_PER_CLUB = 3


@dataclass(frozen=True)
class SelectionResult:
    """Output of the team selector ILP."""

    squad: pd.DataFrame  # full 15-player squad
    xi: pd.DataFrame  # starting 11
    captain: pd.Series  # single-row Series for the captain
    total_xpts: float  # expected points (with captain doubling)


@pa.check_types
def select_team(players: DataFrame[PlayerPoolSchema]) -> SelectionResult:
    """Select the optimal squad, XI, and captain given predicted Xpts.

    Args:
        players: DataFrame with columns:
                     element (int, player ID)
                     name (str)
                     position (str: GK / DEF / MID / FWD)
                     team (str or int, club identifier)
                     value (int, price in FPL units — 100 = £1m)
                     xpts (float, expected points from aggregator)

    Returns:
        SelectionResult with squad, XI, captain, and total_xpts.

    Raises:
        ValueError: if the ILP is infeasible (e.g. not enough players provided).
    """
    required = {"element", "position", "team", "value", "xpts"}
    missing = required - set(players.columns)
    if missing:
        raise ValueError(f"players DataFrame missing columns: {missing}")

    players = players.reset_index(drop=True)
    n = len(players)
    idx = range(n)

    prob = pulp.LpProblem("FPL_team_selector", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("squad", idx, cat="Binary")  # in squad
    s = pulp.LpVariable.dicts("xi", idx, cat="Binary")  # in XI
    c = pulp.LpVariable.dicts("captain", idx, cat="Binary")  # captain

    # Objective: maximise Xpts (captain counts twice)
    prob += pulp.lpSum(players.loc[i, "xpts"] * (s[i] + c[i]) for i in idx)

    # Squad and XI sizes
    prob += pulp.lpSum(x[i] for i in idx) == _SQUAD_SIZE
    prob += pulp.lpSum(s[i] for i in idx) == _XI_SIZE
    prob += pulp.lpSum(c[i] for i in idx) == _CAPTAIN_COUNT

    # Hierarchical: captain ≤ XI ≤ squad
    for i in idx:
        prob += s[i] <= x[i]
        prob += c[i] <= s[i]

    # Budget
    prob += pulp.lpSum(players.loc[i, "value"] * x[i] for i in idx) <= _BUDGET

    # Squad position quotas
    for pos, count in _POSITION_SQUAD.items():
        pos_idx = players[players["position"] == pos].index.tolist()
        prob += pulp.lpSum(x[i] for i in pos_idx) == count

    # XI position bounds (formation validity)
    for pos in ("GK", "DEF", "MID", "FWD"):
        pos_idx = players[players["position"] == pos].index.tolist()
        prob += pulp.lpSum(s[i] for i in pos_idx) >= _POSITION_XI_MIN[pos]
        prob += pulp.lpSum(s[i] for i in pos_idx) <= _POSITION_XI_MAX[pos]

    # Max 3 per club
    for club, group in players.groupby("team"):
        club_idx = group.index.tolist()
        prob += pulp.lpSum(x[i] for i in club_idx) <= _MAX_PER_CLUB

    # Solve — prefer HiGHS (ARM-native); fall back to bundled CBC
    _highs = pulp.HiGHS_CMD(msg=False)
    solver = _highs if _highs.available() else pulp.PULP_CBC_CMD(msg=False)
    prob.solve(solver)

    status = pulp.LpStatus[prob.status]
    if status != "Optimal":
        raise ValueError(
            f"ILP status: {status}. Check that enough players are provided."
        )

    in_squad = [i for i in idx if pulp.value(x[i]) > 0.5]
    in_xi = [i for i in idx if pulp.value(s[i]) > 0.5]
    cap_idx = next(i for i in idx if pulp.value(c[i]) > 0.5)

    squad_df = players.loc[in_squad].copy()
    xi_df = players.loc[in_xi].copy()
    captain_row = players.loc[cap_idx]

    total_xpts = float(
        sum(players.loc[i, "xpts"] for i in in_xi)
        + players.loc[cap_idx, "xpts"]  # captain counts twice
    )

    return SelectionResult(
        squad=squad_df.reset_index(drop=True),
        xi=xi_df.reset_index(drop=True),
        captain=captain_row,
        total_xpts=total_xpts,
    )
