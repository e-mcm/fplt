"""Transfer recommendation engine — 1 or 2 player swaps.

Finds the transfer(s) that maximise net Xpts gain within the free transfer
allowance and budget constraints.

Each transfer costs 4 points if it exceeds the free transfer count (the
'hit' penalty).  The engine returns the best 1-transfer and best 2-transfer
recommendations with their net gain after the hit.

Strategy:
    For 1 transfer: enumerate all (out, in) pairs where:
        - player_out is in current squad
        - player_in is not in current squad
        - same position
        - new budget is not exceeded
        - max 3 per club still holds
    For 2 transfers: enumerate all valid pairs of (out1, in1), (out2, in2).
    Complexity is O(squad_size * pool_size) for 1 transfer and
    O((squad_size * pool_size)^2) for 2 — kept tractable by limiting pool.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import pandera.pandas as pa
from pandera.typing import DataFrame

from fplt.schemas import PlayerPoolSchema

_HIT_PENALTY = 4  # points deducted per transfer over free limit
_MAX_PER_CLUB = 3


@dataclass(frozen=True)
class Transfer:
    """A single player swap."""

    player_out: pd.Series
    player_in: pd.Series
    xpts_gain: float  # Xpts(in) - Xpts(out)
    cost_change: int  # value(in) - value(out), in FPL units


@dataclass(frozen=True)
class TransferRecommendation:
    """One or two transfer recommendations with net Xpts impact."""

    transfers: list[Transfer]
    gross_gain: float  # sum of xpts_gain across transfers
    hit_cost: float  # points penalty for going over free transfers
    net_gain: float  # gross_gain - hit_cost


@pa.check_types
def recommend_transfers(
    squad: DataFrame[PlayerPoolSchema],
    pool: DataFrame[PlayerPoolSchema],
    free_transfers: int = 1,
    n_transfers: int = 1,
) -> TransferRecommendation:
    """Find the best 1 or 2 transfers for the current squad.

    Args:
        squad: Current 15-player squad (must have element, position, team,
               value, xpts columns).
        pool: Available players not in squad (same required columns).
        free_transfers: Number of free transfers this GW (usually 1 or 2).
        n_transfers: Number of transfers to recommend (1 or 2).

    Returns:
        TransferRecommendation with the best transfer(s).
    """
    if n_transfers not in (1, 2):
        raise ValueError("n_transfers must be 1 or 2")

    squad_ids = set(squad["element"].tolist())
    # Pool excludes players already in squad
    available = pool[~pool["element"].isin(squad_ids)].copy()

    hit_cost = max(0, n_transfers - free_transfers) * _HIT_PENALTY

    if n_transfers == 1:
        best = _best_single_transfer(squad, available, hit_cost)
    else:
        best = _best_double_transfer(squad, available, hit_cost)

    return best


def _best_single_transfer(
    squad: DataFrame[PlayerPoolSchema],
    pool: DataFrame[PlayerPoolSchema],
    hit_cost: float,
) -> TransferRecommendation:
    budget = int(squad["value"].sum())
    club_counts = squad.groupby("team")["element"].count().to_dict()

    best_net = float("-inf")
    best_transfer: Transfer | None = None

    for _, out_row in squad.iterrows():
        pos = out_row["position"]
        candidates = pool[pool["position"] == pos]

        for _, in_row in candidates.iterrows():
            new_budget = budget - int(out_row["value"]) + int(in_row["value"])
            if new_budget > 1000:  # £100m cap
                continue

            # Club constraint
            club_in = in_row["team"]
            count_in_club = club_counts.get(club_in, 0)
            count_out_club = 1 if out_row["team"] == club_in else 0
            if count_in_club - count_out_club >= _MAX_PER_CLUB:
                continue

            gain = float(in_row["xpts"]) - float(out_row["xpts"])
            net = gain - hit_cost
            if net > best_net:
                best_net = net
                best_transfer = Transfer(
                    player_out=out_row,
                    player_in=in_row,
                    xpts_gain=gain,
                    cost_change=int(in_row["value"]) - int(out_row["value"]),
                )

    if best_transfer is None:
        raise ValueError("No valid single transfer found.")

    return TransferRecommendation(
        transfers=[best_transfer],
        gross_gain=best_transfer.xpts_gain,
        hit_cost=hit_cost,
        net_gain=best_net,
    )


def _best_double_transfer(
    squad: DataFrame[PlayerPoolSchema],
    pool: DataFrame[PlayerPoolSchema],
    hit_cost: float,
) -> TransferRecommendation:
    budget = int(squad["value"].sum())
    club_counts = squad.groupby("team")["element"].count().to_dict()

    best_net = float("-inf")
    best_pair: tuple[Transfer, Transfer] | None = None

    squad_rows = list(squad.iterrows())
    for i, (_, out1) in enumerate(squad_rows):
        for _, out2 in squad_rows[i + 1 :]:
            # Temporarily remove both players
            temp_budget = budget - int(out1["value"]) - int(out2["value"])
            temp_clubs = dict(club_counts)
            for out_row in (out1, out2):
                club = out_row["team"]
                temp_clubs[club] = temp_clubs.get(club, 1) - 1

            pos1, pos2 = out1["position"], out2["position"]
            cands1 = pool[pool["position"] == pos1]
            cands2 = pool[pool["position"] == pos2]

            for _, in1 in cands1.iterrows():
                for _, in2 in cands2.iterrows():
                    if in1["element"] == in2["element"]:
                        continue

                    new_budget = temp_budget + int(in1["value"]) + int(in2["value"])
                    if new_budget > 1000:
                        continue

                    # Club check for both incoming players
                    clubs_after = dict(temp_clubs)
                    valid = True
                    for in_row in (in1, in2):
                        c = in_row["team"]
                        clubs_after[c] = clubs_after.get(c, 0) + 1
                        if clubs_after[c] > _MAX_PER_CLUB:
                            valid = False
                            break
                    if not valid:
                        continue

                    gain = (
                        float(in1["xpts"])
                        - float(out1["xpts"])
                        + float(in2["xpts"])
                        - float(out2["xpts"])
                    )
                    net = gain - hit_cost
                    if net > best_net:
                        best_net = net
                        t1 = Transfer(
                            out1,
                            in1,
                            float(in1["xpts"]) - float(out1["xpts"]),
                            int(in1["value"]) - int(out1["value"]),
                        )
                        t2 = Transfer(
                            out2,
                            in2,
                            float(in2["xpts"]) - float(out2["xpts"]),
                            int(in2["value"]) - int(out2["value"]),
                        )
                        best_pair = (t1, t2)

    if best_pair is None:
        raise ValueError("No valid double transfer found.")

    t1, t2 = best_pair
    return TransferRecommendation(
        transfers=list(best_pair),
        gross_gain=t1.xpts_gain + t2.xpts_gain,
        hit_cost=hit_cost,
        net_gain=best_net,
    )
