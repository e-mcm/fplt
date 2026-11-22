"""Sample-average XI / bench / captain selection on a fixed fifteen.

Wave D1 (M2/M3): maximise mean realised points across shared scenarios with
autosub and captain/vice fallback applied per draw. Exact enumeration of
valid lineups for a locked 15 (formations × captain/vice × outfield bench
order) is the sample-average approximation for the single-GW problem.
``bench_weight`` is not used.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from itertools import combinations, permutations, product

import numpy as np
import polars as pl

from fplt.optimise.scenarios import ScenarioBundle, mean_realised_lineup

_XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
_XI_MAX = {"GK": 1, "DEF": 5, "MID": 5, "FWD": 3}
_XI_SIZE = 11
_POS = ("GK", "DEF", "MID", "FWD")
_POS_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}


@dataclass(frozen=True)
class LineupSelection:
    """XI, bench order, and armbands that maximise scenario-mean points."""

    players: pl.DataFrame
    mean_points: float


def select_lineup_scenarios(
    squad: pl.DataFrame,
    bundle: ScenarioBundle,
    *,
    chip: str | None = None,
    p_haul: dict[int, float] | None = None,
    haul_tiebreak: float = 1e-3,
) -> LineupSelection:
    """Choose XI, bench order, captain, and vice by sample-average realised pts.

    Args:
        squad: Exactly fifteen rows with ``player_key`` and ``position``.
            Optional ``xpts`` / ``team_key`` / ``price`` are passed through.
        bundle: Shared scenarios covering every squad ``player_key``.
        chip: Optional chip (``bboost`` / ``3xc``) applied in every scenario.
        p_haul: Optional haul probabilities for captaincy tie-break (M3).
        haul_tiebreak: Weight on captain ``p_haul`` added to the mean so near
            ties prefer explosive captains when ``p_haul`` is provided.

    Returns:
        The fifteen with ``in_xi``, ``captain``, ``vice``, ``bench_slot``, and
        the scenario-mean realised points.

    Raises:
        ValueError: Squad size/positions invalid, or bundle missing a player.
    """
    locked = _validate_squad(squad)
    keys = [int(value) for value in locked["player_key"].to_list()]
    positions = {
        int(row["player_key"]): str(row["position"])
        for row in locked.iter_rows(named=True)
    }
    by_pos = {
        position: [key for key in keys if positions[key] == position]
        for position in _POS
    }
    for position, members in by_pos.items():
        if not members and position == "GK":
            raise ValueError("squad needs two goalkeepers")
    index = {key: bundle.index_of(key) for key in keys}
    points = bundle.points
    minutes = bundle.minutes
    haul = p_haul or {}
    any_blank = bool(np.any(minutes <= 0))

    best_mean = -np.inf
    best_score = -np.inf
    best: tuple[tuple[int, ...], tuple[int, ...], int, int] | None = None

    for xi_keys in _iter_valid_xis(by_pos):
        xi_set = set(xi_keys)
        bench_outfield = tuple(
            key for key in keys if key not in xi_set and positions[key] != "GK"
        )
        reserve_gk = next(
            key for key in keys if key not in xi_set and positions[key] == "GK"
        )
        xi_idx = [index[key] for key in xi_keys]
        xi_pos = [positions[key] for key in xi_keys]
        outfield_orders: Sequence[tuple[int, ...]]
        if any_blank:
            outfield_orders = tuple(permutations(bench_outfield))
        else:
            outfield_orders = (bench_outfield,)
        for outfield_order in outfield_orders:
            bench_keys = outfield_order + (reserve_gk,)
            bench_idx = [index[key] for key in bench_keys]
            bench_pos = [positions[key] for key in bench_keys]
            armbands = _armband_pairs(xi_keys, any_blank=any_blank, haul=haul)
            for captain, vice in armbands:
                means = _mean_nets(
                    xi_idx=xi_idx,
                    xi_pos=xi_pos,
                    bench_idx=bench_idx,
                    bench_pos=bench_pos,
                    captain_i=index[captain],
                    vice_i=index[vice],
                    points=points,
                    minutes=minutes,
                    chip=chip,
                )
                haul_bonus = haul_tiebreak * float(haul.get(captain, 0.0))
                score = means + haul_bonus
                if score > best_score + 1e-12:
                    best_score = score
                    best_mean = means
                    best = (xi_keys, bench_keys, captain, vice)

    if best is None:
        raise ValueError("no valid XI for this squad")
    xi_keys, bench_keys, captain, vice = best
    return LineupSelection(
        players=_annotate(locked, xi_keys, bench_keys, captain, vice),
        mean_points=float(best_mean),
    )


def select_lineup_expected(
    squad: pl.DataFrame,
    *,
    haul_weight: float = 0.05,
) -> LineupSelection:
    """EV lineup with optional ``p_haul`` captaincy tilt when samples are absent.

    Builds a one-sim bundle from ``xpts`` (everyone plays 90) so the same
    enumerator applies. Captain objective uses ``xpts + haul_weight * p_haul``.

    Args:
        squad: Fifteen rows with ``player_key``, ``position``, ``xpts``, and
            optionally ``p_haul``.
        haul_weight: Weight on ``p_haul`` in the captaincy tie-break.

    Returns:
        A :class:`LineupSelection` under the degenerate one-sim bundle.
    """
    required = {"player_key", "position", "xpts"}
    missing = required - set(squad.columns)
    if missing:
        raise ValueError(f"squad missing columns: {sorted(missing)}")
    locked = _validate_squad(squad)
    keys = [int(value) for value in locked["player_key"].to_list()]
    xpts = {
        int(row["player_key"]): float(row["xpts"])
        for row in locked.iter_rows(named=True)
    }
    haul = {
        int(row["player_key"]): float(row["p_haul"])
        for row in locked.iter_rows(named=True)
        if "p_haul" in locked.columns and row["p_haul"] is not None
    }
    bundle = ScenarioBundle(
        player_keys=np.asarray(keys, dtype=np.int64),
        points=np.asarray([[xpts[key]] for key in keys], dtype=np.float64),
        minutes=np.full((len(keys), 1), 90, dtype=np.int64),
    )
    return select_lineup_scenarios(
        locked,
        bundle,
        p_haul=haul or None,
        haul_tiebreak=haul_weight,
    )


def _validate_squad(squad: pl.DataFrame) -> pl.DataFrame:
    required = {"player_key", "position"}
    missing = required - set(squad.columns)
    if missing:
        raise ValueError(f"squad missing columns: {sorted(missing)}")
    if squad["player_key"].n_unique() != squad.height:
        raise ValueError("player_key must be unique")
    if squad.height != 15:
        raise ValueError(f"squad must have 15 players, got {squad.height}")
    counts = {position: 0 for position in _POS}
    for position in squad["position"].to_list():
        if position not in counts:
            raise ValueError(f"unknown position {position!r}")
        counts[str(position)] += 1
    expected = {"GK": 2, "DEF": 5, "MID": 5, "FWD": 3}
    if counts != expected:
        raise ValueError(f"squad position counts must be {expected}, got {counts}")
    return squad.sort("player_key")


def _armband_pairs(
    xi_keys: tuple[int, ...],
    *,
    any_blank: bool,
    haul: dict[int, float],
) -> Sequence[tuple[int, int]]:
    """Captain/vice pairs to try for one XI.

    When nobody ever blanks, vice never receives the armband, so pick a
    deterministic vice (lowest ``player_key`` among non-captains) and only
    vary the captain. With blanks, enumerate every ordered pair.
    """
    if any_blank:
        return tuple(
            (captain, vice)
            for captain, vice in product(xi_keys, repeat=2)
            if captain != vice
        )
    pairs: list[tuple[int, int]] = []
    for captain in xi_keys:
        vice = min(key for key in xi_keys if key != captain)
        pairs.append((captain, vice))
    # Prefer higher-haul captains earlier only for readability; search is exact.
    if haul:
        pairs.sort(key=lambda pair: -float(haul.get(pair[0], 0.0)))
    return pairs


def _iter_valid_xis(
    by_pos: dict[str, list[int]],
) -> Sequence[tuple[int, ...]]:
    gks = by_pos["GK"]
    defs = by_pos["DEF"]
    mids = by_pos["MID"]
    fwds = by_pos["FWD"]
    formations: list[tuple[int, int, int]] = []
    for n_def in range(_XI_MIN["DEF"], _XI_MAX["DEF"] + 1):
        for n_mid in range(_XI_MIN["MID"], _XI_MAX["MID"] + 1):
            n_fwd = _XI_SIZE - 1 - n_def - n_mid
            if n_fwd < _XI_MIN["FWD"] or n_fwd > _XI_MAX["FWD"]:
                continue
            if n_def > len(defs) or n_mid > len(mids) or n_fwd > len(fwds):
                continue
            formations.append((n_def, n_mid, n_fwd))
    xis: list[tuple[int, ...]] = []
    for gk in gks:
        for n_def, n_mid, n_fwd in formations:
            for def_set in combinations(defs, n_def):
                for mid_set in combinations(mids, n_mid):
                    for fwd_set in combinations(fwds, n_fwd):
                        xi = (gk, *def_set, *mid_set, *fwd_set)
                        xis.append(tuple(sorted(xi)))
    return xis


def _mean_nets(
    *,
    xi_idx: list[int],
    xi_pos: list[str],
    bench_idx: list[int],
    bench_pos: list[str],
    captain_i: int,
    vice_i: int,
    points: np.ndarray,
    minutes: np.ndarray,
    chip: str | None,
) -> float:
    n_sims = points.shape[1]
    total = 0.0
    for sim in range(n_sims):
        total += _net_one_sim(
            xi_idx=xi_idx,
            xi_pos=xi_pos,
            bench_idx=bench_idx,
            bench_pos=bench_pos,
            captain_i=captain_i,
            vice_i=vice_i,
            pts=points[:, sim],
            mins=minutes[:, sim],
            chip=chip,
        )
    return total / float(n_sims)


def _net_one_sim(
    *,
    xi_idx: list[int],
    xi_pos: list[str],
    bench_idx: list[int],
    bench_pos: list[str],
    captain_i: int,
    vice_i: int,
    pts: np.ndarray,
    mins: np.ndarray,
    chip: str | None,
) -> float:
    order = sorted(
        range(len(xi_idx)),
        key=lambda i: (_POS_ORDER[xi_pos[i]], xi_idx[i]),
    )
    active = set(xi_idx)
    counts = {position: 0 for position in _POS}
    for pos in xi_pos:
        counts[pos] += 1
    used_bench: set[int] = set()
    for slot in order:
        starter = xi_idx[slot]
        if int(mins[starter]) > 0:
            continue
        blank_pos = xi_pos[slot]
        for bench_slot, sub in enumerate(bench_idx):
            if sub in used_bench or int(mins[sub]) <= 0:
                continue
            sub_pos = bench_pos[bench_slot]
            if not _eligible_sub(blank_pos, sub_pos, counts):
                continue
            active.discard(starter)
            active.add(sub)
            used_bench.add(sub)
            counts[blank_pos] -= 1
            counts[sub_pos] += 1
            break
    xi_sum = 0.0
    for key in active:
        if int(mins[key]) > 0:
            xi_sum += float(pts[key])
    total = xi_sum
    if int(mins[captain_i]) > 0:
        extra = 2.0 * float(pts[captain_i]) if chip == "3xc" else float(pts[captain_i])
        total += extra
    elif int(mins[vice_i]) > 0:
        if vice_i in active:
            total += float(pts[vice_i])
        else:
            total += 2.0 * float(pts[vice_i])
    if chip == "bboost":
        for sub in bench_idx:
            if sub not in used_bench and int(mins[sub]) > 0:
                total += float(pts[sub])
    return total


def _eligible_sub(
    blank_position: str, sub_position: str, counts: dict[str, int]
) -> bool:
    if blank_position == "GK":
        return sub_position == "GK"
    if sub_position == "GK":
        return False
    trial = dict(counts)
    trial[blank_position] -= 1
    trial[sub_position] += 1
    if sum(trial.values()) != _XI_SIZE:
        return False
    return all(trial[position] >= minimum for position, minimum in _XI_MIN.items())


def _annotate(
    squad: pl.DataFrame,
    xi_keys: tuple[int, ...],
    bench_keys: tuple[int, ...],
    captain: int,
    vice: int,
) -> pl.DataFrame:
    xi_set = set(xi_keys)
    slots = {key: slot for slot, key in enumerate(bench_keys, start=1)}
    return squad.with_columns(
        pl.col("player_key").is_in(list(xi_set)).alias("in_xi"),
        (pl.col("player_key") == captain).alias("captain"),
        (pl.col("player_key") == vice).alias("vice"),
        pl.col("player_key")
        .replace_strict(slots, default=None)
        .cast(pl.Int64)
        .alias("bench_slot"),
    )


def verify_lineup_mean(
    selection: LineupSelection,
    bundle: ScenarioBundle,
    *,
    chip: str | None = None,
) -> float:
    """Recompute mean via :func:`mean_realised_lineup` (slow path check)."""
    return mean_realised_lineup(selection.players, bundle, chip=chip)
