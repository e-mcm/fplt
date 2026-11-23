"""Realised FPL points from a decision-ledger gameweek.

Applies automatic substitutions and vice-captain-on-blank captaincy, then
deducts transfer hits once per gameweek. When picks include ``bench_slot``
(1–4 for the ordered bench), that order is used for autosubs; otherwise
bench order is position then ``player_key``. Selling-price rises are out of
scope: decision sims already freeze selling at purchase price.
"""

from __future__ import annotations

from dataclasses import dataclass

import polars as pl

_POS_ORDER = {"GK": 0, "DEF": 1, "MID": 2, "FWD": 3}
_XI_MIN = {"GK": 1, "DEF": 3, "MID": 2, "FWD": 1}
_XI_SIZE = 11


@dataclass(frozen=True)
class RealisedGameweek:
    """Points for one model × season × gameweek after FPL scoring rules."""

    model_id: str
    season: str
    gw: int
    points: int
    hit: float
    net_points: int


@dataclass(frozen=True)
class _Pick:
    """One kept player in a gameweek for realised scoring."""

    player_key: int
    position: str
    in_xi: bool
    captain: bool
    vice: bool
    pts: int
    minutes: int


def score_gameweek(
    picks: pl.DataFrame,
    *,
    hit: float = 0.0,
    chip: str | None = None,
) -> int:
    """Score one gameweek's kept squad with autosub and vice-on-blank.

    Args:
        picks: Non-sold players for one GW with ``player_key``, ``position``,
            ``in_xi``, ``captain``, ``vice``, ``actual_pts``, and
            ``actual_minutes``. Null points are 0; null minutes assume played.
        hit: Transfer hit cost to subtract once (non-negative).
        chip: Active chip for the gameweek (``bboost``, ``3xc``, …). When
            omitted, uses a non-null ``chip`` column on ``picks`` if present.

    Returns:
        Net realised points (XI after autosub, captain ×2 / Triple Captain
        ×3, Bench Boost bench points, minus ``hit``).

    Raises:
        ValueError: The frame is missing required columns or has no XI.
    """
    return _score_parts(picks, hit=hit, chip=_resolve_chip(picks, chip)).net


def explain_gameweek(
    picks: pl.DataFrame,
    *,
    hit: float = 0.0,
    chip: str | None = None,
    names: dict[int, str] | None = None,
) -> str:
    """Render XI / autosub / captain arithmetic for one gameweek's picks.

    Args:
        picks: Decision rows for a single model × season × gameweek.
        hit: Transfer hit cost to subtract once.
        chip: Active chip for the gameweek; see ``score_gameweek``.
        names: Optional ``player_key`` → display name map.

    Returns:
        A multi-line explanation ending in the net total.

    Raises:
        ValueError: The frame is missing required columns or has no XI.
    """
    parts = _score_parts(picks, hit=hit, chip=_resolve_chip(picks, chip))
    label = names or {}

    def _name(key: int) -> str:
        return label.get(key, str(key))

    lines = [
        "role | player | pos | mins | pts | note",
        "--- | --- | --- | ---: | ---: | ---",
    ]
    for row in parts.active:
        notes: list[str] = []
        if row.player_key in parts.autosubbed_on:
            notes.append("autosub")
        if row.captain:
            notes.append("captain")
        if row.vice:
            notes.append("vice")
        lines.append(
            f"XI | {_name(row.player_key)} | {row.position} | "
            f"{row.minutes} | {row.pts} | {', '.join(notes) or '-'}"
        )
    for row in parts.inactive_bench:
        lines.append(
            f"bench | {_name(row.player_key)} | {row.position} | "
            f"{row.minutes} | {row.pts} | -"
        )
    lines.append(f"XI sum (after autosub, mins>0): {parts.xi_sum}")
    if parts.chip == "3xc" and parts.captain_extra:
        lines.append(
            f"triple captain extra (+{parts.captain_extra}): "
            f"{_name(parts.armband_key)} ×3 armband"
        )
    elif parts.captain_extra:
        lines.append(
            f"captain extra (+{parts.captain_extra}): "
            f"{_name(parts.armband_key)} ×2 armband"
        )
    elif parts.vice_extra:
        lines.append(
            f"vice armband extra (+{parts.vice_extra}): "
            f"{_name(parts.armband_key)} (captain blanked)"
        )
    else:
        lines.append("armband extra: 0 (captain and vice blanked)")
    if parts.bench_boost:
        lines.append(f"bench boost (+{parts.bench_boost}): full bench scores")
    if parts.chip:
        lines.append(f"chip: {parts.chip}")
    lines.append(f"hit: -{int(max(hit, 0.0))}")
    lines.append(f"net: {parts.net}")
    return "\n".join(lines)


@dataclass(frozen=True)
class _ScoreParts:
    """Breakdown of one ``score_gameweek`` result for explain dumps."""

    active: tuple[_Pick, ...]
    inactive_bench: tuple[_Pick, ...]
    autosubbed_on: frozenset[int]
    xi_sum: int
    captain_extra: int
    vice_extra: int
    armband_key: int
    bench_boost: int
    chip: str | None
    net: int


def _score_parts(
    picks: pl.DataFrame, *, hit: float, chip: str | None = None
) -> _ScoreParts:
    required = {
        "player_key",
        "position",
        "in_xi",
        "captain",
        "vice",
        "actual_pts",
        "actual_minutes",
    }
    missing = required - set(picks.columns)
    if missing:
        raise ValueError(f"picks missing columns: {sorted(missing)}")
    kept = picks.filter(~pl.col("sold")) if "sold" in picks.columns else picks
    if kept.filter(pl.col("in_xi")).is_empty():
        raise ValueError("picks need at least one in_xi player")
    rows = [_pick_from_row(row) for row in kept.iter_rows(named=True)]
    starters = sorted(
        (row for row in rows if row.in_xi),
        key=lambda row: (_POS_ORDER.get(row.position, 99), row.player_key),
    )
    bench = _ordered_bench(kept, [row for row in rows if not row.in_xi])
    active_keys = {row.player_key for row in starters}
    counts = _position_counts(starters)
    used_bench: set[int] = set()
    for starter in starters:
        if starter.minutes > 0:
            continue
        for sub in bench:
            if sub.player_key in used_bench or sub.minutes <= 0:
                continue
            if not _eligible_sub(starter.position, sub.position, counts):
                continue
            active_keys.discard(starter.player_key)
            active_keys.add(sub.player_key)
            used_bench.add(sub.player_key)
            counts[starter.position] -= 1
            counts[sub.position] += 1
            break
    by_key = {row.player_key: row for row in rows}
    # Only players who appeared (minutes > 0) contribute; a blanked starter
    # with no eligible autosub scores nothing.
    active = tuple(
        by_key[key]
        for key in sorted(
            active_keys,
            key=lambda key: (
                _POS_ORDER.get(by_key[key].position, 99),
                key,
            ),
        )
        if by_key[key].minutes > 0
    )
    xi_sum = sum(row.pts for row in active)
    captain = next((row for row in rows if row.captain), None)
    vice = next((row for row in rows if row.vice), None)
    captain_extra = 0
    vice_extra = 0
    armband_key = 0
    total = xi_sum
    if captain is not None and captain.minutes > 0:
        # Triple Captain adds two extras on top of the XI copy (×3 total).
        captain_extra = 2 * captain.pts if chip == "3xc" else captain.pts
        armband_key = captain.player_key
        total += captain_extra
    elif vice is not None and vice.minutes > 0:
        armband_key = vice.player_key
        if vice.player_key in active_keys:
            vice_extra = vice.pts
        else:
            vice_extra = 2 * vice.pts
        total += vice_extra
    inactive_bench = tuple(row for row in bench if row.player_key not in used_bench)
    bench_boost = 0
    if chip == "bboost":
        # Bench Boost: every bench player with minutes scores (autosubs still
        # move blanks into the XI copy above; remaining bench is added here).
        bench_boost = sum(row.pts for row in inactive_bench if row.minutes > 0)
        total += bench_boost
    return _ScoreParts(
        active=active,
        inactive_bench=inactive_bench,
        autosubbed_on=frozenset(used_bench),
        xi_sum=xi_sum,
        captain_extra=captain_extra,
        vice_extra=vice_extra,
        armband_key=armband_key,
        bench_boost=bench_boost,
        chip=chip,
        net=int(total - max(hit, 0.0)),
    )


def summarise_decisions(frame: pl.DataFrame) -> pl.DataFrame:
    """Aggregate decision-ledger rows into model × season totals and per-GW nets.

    Args:
        frame: Rows with decision columns plus ``position`` and
            ``actual_minutes``. Sold rows are ignored for scoring; ``hit`` is
            taken once per gameweek from any kept row.

    Returns:
        ``model_id``, ``season``, ``total_points``, ``gameweeks``, and
        ``per_gw`` (sorted gw→net list) for each model × season present.
    """
    if frame.is_empty():
        return pl.DataFrame(
            schema={
                "model_id": pl.Utf8(),
                "season": pl.Utf8(),
                "total_points": pl.Int64(),
                "gameweeks": pl.Int64(),
                "per_gw": pl.List(pl.Int64()),
            }
        )
    required = {
        "model_id",
        "season",
        "gw",
        "player_key",
        "position",
        "in_xi",
        "captain",
        "vice",
        "actual_pts",
        "actual_minutes",
        "hit",
    }
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"decisions missing columns: {sorted(missing)}")
    realised: list[RealisedGameweek] = []
    keys = (
        frame.select("model_id", "season", "gw")
        .unique()
        .sort("model_id", "season", "gw")
    )
    for model_id, season, gameweek in keys.iter_rows():
        part = frame.filter(
            (pl.col("model_id") == model_id)
            & (pl.col("season") == season)
            & (pl.col("gw") == gameweek)
        )
        hit = float(part["hit"][0]) if part.height else 0.0
        chip = _resolve_chip(part, None)
        net = score_gameweek(part, hit=hit, chip=chip)
        realised.append(
            RealisedGameweek(
                model_id=str(model_id),
                season=str(season),
                gw=int(gameweek),
                points=net + int(hit),
                hit=hit,
                net_points=net,
            )
        )
    if not realised:
        return summarise_decisions(frame.clear())
    detail = pl.DataFrame(
        {
            "model_id": [row.model_id for row in realised],
            "season": [row.season for row in realised],
            "gw": [row.gw for row in realised],
            "net_points": [row.net_points for row in realised],
        }
    )
    return (
        detail.group_by("model_id", "season")
        .agg(
            pl.col("net_points").sum().alias("total_points"),
            pl.col("gw").n_unique().alias("gameweeks"),
            pl.col("net_points").sort_by("gw").alias("per_gw"),
        )
        .sort("total_points", descending=True)
    )


def model_board_label(
    model_id: str,
    *,
    model_name: str | None = None,
) -> str:
    """Stable board label: YAML name plus ``model_id`` hash when known.

    Boards must not rely on the YAML name alone — editing a released
    ModelSpec in place (e.g. pre-4d715b5 ``dc_layered_v2``) changes the
    hash while keeping the name. Prefer ``name (model_id)``.

    Args:
        model_id: Spec hash (or chip-suffixed ledger id).
        model_name: YAML ``ModelSpec.name`` when known.

    Returns:
        ``"{name} ({model_id})"`` when ``model_name`` is set, else ``model_id``.
    """
    if model_name:
        return f"{model_name} ({model_id})"
    return model_id


def format_decision_scoreboard(summary: pl.DataFrame) -> str:
    """Render a markdown table of season totals ranked by realised points.

    When ``attainable_points`` / ``share_of_attainable`` are present (from
    ``enrich_decision_summary``), those columns are appended. When
    ``model_name`` is present, the Model column shows ``name (model_id)``.
    """
    if summary.is_empty():
        return "_No decision rows to score._"
    has_share = (
        "attainable_points" in summary.columns
        and "share_of_attainable" in summary.columns
    )
    headers = ["Rank", "Model", "Season", "Points", "GWs", "Mean/GW"]
    aligns = ["---:", "---", "---", "---:", "---:", "---:"]
    if has_share:
        headers.extend(["Attainable", "Share"])
        aligns.extend(["---:", "---:"])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(aligns) + " |",
    ]
    for index, row in enumerate(summary.iter_rows(named=True), start=1):
        gameweeks = int(row["gameweeks"])
        mean = float(row["total_points"]) / gameweeks if gameweeks else 0.0
        name = row.get("model_name")
        label = model_board_label(
            str(row["model_id"]),
            model_name=str(name) if name is not None else None,
        )
        cells = [
            str(index),
            label,
            str(row["season"]),
            str(row["total_points"]),
            str(gameweeks),
            f"{mean:.1f}",
        ]
        if has_share:
            att = row["attainable_points"]
            share = row["share_of_attainable"]
            cells.append("-" if att is None else str(int(att)))
            cells.append("-" if share is None else f"{float(share):.3f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


# 2024-25 overall champion finished at ~74 pts/GW with chips. Chips-off
# bakeoffs should sit well below that; anything near 100+ is almost always
# double-counted actuals from a multi-step prediction ledger.
DEFAULT_MAX_MEAN_PPG = 90.0


def assert_sane_mean_ppg(
    summary: pl.DataFrame,
    *,
    max_mean: float = DEFAULT_MAX_MEAN_PPG,
) -> None:
    """Raise when any model × season mean points/GW looks inflated.

    Args:
        summary: Output of ``summarise_decisions``.
        max_mean: Inclusive upper bound on ``total_points / gameweeks``.

    Raises:
        ValueError: A row exceeds ``max_mean``.
    """
    if summary.is_empty() or max_mean <= 0:
        return
    for row in summary.iter_rows(named=True):
        gameweeks = int(row["gameweeks"])
        if gameweeks <= 0:
            continue
        mean = float(row["total_points"]) / gameweeks
        if mean > max_mean:
            raise ValueError(
                f"mean pts/GW {mean:.1f} for model_id={row['model_id']} "
                f"season={row['season']} exceeds max_mean={max_mean:.1f} "
                "(likely double-counted ledger actuals; "
                "run `fplt decisions --explain-gw` on a high GW)"
            )


def _resolve_chip(picks: pl.DataFrame, chip: str | None) -> str | None:
    """Prefer an explicit chip; else the first non-null ``chip`` column value."""
    if chip is not None:
        return chip
    if "chip" not in picks.columns or picks.is_empty():
        return None
    for value in picks["chip"].to_list():
        if value is not None and str(value) != "":
            return str(value)
    return None


def _ordered_bench(kept: pl.DataFrame, bench_rows: list[_Pick]) -> list[_Pick]:
    """Order the bench for autosubs: ``bench_slot`` when present, else pos/key."""
    if not bench_rows:
        return []
    if "bench_slot" not in kept.columns:
        return sorted(
            bench_rows,
            key=lambda row: (_POS_ORDER.get(row.position, 99), row.player_key),
        )
    slots = {
        _as_int(row["player_key"]): row.get("bench_slot")
        for row in kept.iter_rows(named=True)
        if not bool(row["in_xi"])
    }
    if any(slots.get(row.player_key) is None for row in bench_rows):
        return sorted(
            bench_rows,
            key=lambda row: (_POS_ORDER.get(row.position, 99), row.player_key),
        )
    return sorted(bench_rows, key=lambda row: _as_int(slots[row.player_key]))


def _pick_from_row(row: dict[str, object]) -> _Pick:
    raw_minutes = row["actual_minutes"]
    minutes = 0 if raw_minutes == 0 else _as_int(raw_minutes, default=1)
    return _Pick(
        player_key=_as_int(row["player_key"]),
        position=str(row["position"]),
        in_xi=bool(row["in_xi"]),
        captain=bool(row["captain"]),
        vice=bool(row["vice"]),
        pts=_as_int(row["actual_pts"]),
        minutes=minutes,
    )


def _as_int(value: object, default: int = 0) -> int:
    if value is None:
        return default
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        return int(value)
    if isinstance(value, str) and value != "":
        return int(value)
    return default


def _position_counts(players: list[_Pick]) -> dict[str, int]:
    counts = {"GK": 0, "DEF": 0, "MID": 0, "FWD": 0}
    for player in players:
        if player.position in counts:
            counts[player.position] += 1
    return counts


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
