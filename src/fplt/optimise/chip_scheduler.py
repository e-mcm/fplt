"""Season-level chip scheduler → ``ChipUseWindows`` for the short-horizon MILP.

Wave C5 / eval §5.4. FPL-Optimization-Tools exposes ``allowed_chip_gws`` /
``forced_chip_gws`` / ``no_chip_gws`` as settings knobs but ships no installable
season scheduler (~CLI/JSON workflow). We fill the same ``ChipUseWindows``
contract with a coarse half-season model already in-tree (polars + fixture
counts from ``chip_windows``).

**Option-value heuristic.** For each remaining chip in each active
``ChipRules`` half-window, score every gameweek from ``as_of_gw`` through
that window's end (fixture doubles/blanks, team-level expected points,
squad continuity). The option-value cut is keyed by ``ChipGrant`` (chip ×
window id) — ``deferred_best`` / ``best_score`` never mix first- and
second-half opportunities, and bare chip-name maps cannot fold grants.
Let ``H`` be the short planning horizon and ``V* = max score`` over weeks
of that window strictly after the short horizon (0 if none). A near-term
week is allowed only when its score is at least ``V*`` — play now only if
the gain beats the best expected opportunity left in the same half. Later
near-best weeks stay allowed so the MILP can wait. The unique best week is
forced when it falls inside the short horizon and clears ``V*``.
Near-horizon weeks where every remaining chip is held join ``no_chip``.

**Force-play at window end.** Remaining unused chips in a half-season
``ChipRules`` window are packed onto the last eligible gameweeks of that
window (one chip per GW, higher coarse score first) so grants cannot
expire unused. Mid-window deadlines stay optional under the option-value
cut; packing triggers when the window's ``stop_event`` falls inside the
short horizon, or when remaining chips outnumber remaining gameweeks in
the window. Chips already spent in the window are omitted.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

from fplt.optimise.chip_windows import (
    ChipUseWindows,
    gameweek_landmarks,
    team_fixture_counts,
)
from fplt.optimise.plan import DEFAULT_PLAN_HORIZON
from fplt.optimise.state import ChipInventory
from fplt.scoring.chips import ChipGrant, ChipName, ChipRules, ChipWindow

_CHIP_ORDER: tuple[ChipName, ...] = ("bboost", "3xc", "freehit", "wildcard")
_SCORE_EPS = 1e-9


@dataclass(frozen=True)
class ChipOpportunity:
    """One coarse (chip, window, gameweek) opportunity score."""

    chip: ChipName
    window: str
    gameweek: int
    score: float

    @property
    def grant(self) -> ChipGrant:
        """Typed chip×window key for opportunity maps (never bare chip name)."""
        return ChipGrant(name=self.chip, window=self.window)


def schedule_chip_windows(
    fixtures: pl.DataFrame,
    *,
    season: str,
    rules: ChipRules,
    as_of_gw: int,
    short_horizon: int = DEFAULT_PLAN_HORIZON,
    predictions: pl.DataFrame | None = None,
    player_teams: pl.DataFrame | None = None,
    owned_player_keys: Sequence[int] | None = None,
    inventory: ChipInventory | None = None,
    gameweeks: Sequence[int] | None = None,
    force_play: bool = True,
) -> ChipUseWindows:
    """Build allowed / forced / no-chip windows for the rest of the half.

    Applies the option-value cut and, when ``force_play`` is true, force-packs
    remaining unused chips onto the last eligible gameweeks of each
    ``ChipRules`` half-window so grants cannot expire.

    Args:
        fixtures: Staged fixtures (``season``, ``gw``, ``team_h_key``,
            ``team_a_key``).
        season: Fixture season label.
        rules: Chip allotment (often 2025-26 rules).
        as_of_gw: Current deadline; only ``gw >= as_of_gw`` are scheduled.
        short_horizon: Planner look-ahead ``H`` used for the option-value cut.
        predictions: Optional ``gw`` / ``player_key`` / ``xpts`` (or
            ``target_gw``) rows for team-level and captain scores.
        player_teams: Optional ``player_key`` / ``team_key`` map (from
            candidates). Required for squad continuity when owned keys are set.
        owned_player_keys: Current squad; empty / omitted uses league landmarks
            only.
        inventory: Remaining chip uses. When omitted, every grant in ``rules``
            is treated as available.
        gameweeks: Optional filter; landmarks outside are ignored.
        force_play: When true (default), pack remaining half-window chips onto
            the last eligible gameweeks. When false, only the option-value cut
            may force a unique near-horizon best.

    Returns:
        ``ChipUseWindows`` ready for ``plan_squad(..., chip_windows=...)``.

    Raises:
        ValueError: ``as_of_gw`` or ``short_horizon`` is out of range.
    """
    if as_of_gw < 1 or as_of_gw > 38:
        raise ValueError(f"as_of_gw must be in 1..38, got {as_of_gw}")
    if short_horizon < 1:
        raise ValueError("short_horizon must be at least 1")
    opportunities = score_chip_opportunities(
        fixtures,
        season=season,
        rules=rules,
        as_of_gw=as_of_gw,
        predictions=predictions,
        player_teams=player_teams,
        owned_player_keys=owned_player_keys,
        inventory=inventory,
        gameweeks=gameweeks,
    )
    return windows_from_opportunities(
        opportunities,
        as_of_gw=as_of_gw,
        short_horizon=short_horizon,
        rules=rules,
        force_play=force_play,
    )


def score_chip_opportunities(
    fixtures: pl.DataFrame,
    *,
    season: str,
    rules: ChipRules,
    as_of_gw: int,
    predictions: pl.DataFrame | None = None,
    player_teams: pl.DataFrame | None = None,
    owned_player_keys: Sequence[int] | None = None,
    inventory: ChipInventory | None = None,
    gameweeks: Sequence[int] | None = None,
) -> tuple[ChipOpportunity, ...]:
    """Score coarse chip opportunities over the remaining half-season windows.

    Args:
        fixtures: Staged fixtures for double / blank counts.
        season: Fixture season label.
        rules: Chip allotment defining half-season windows and grants.
        as_of_gw: First gameweek to consider.
        predictions: Optional player expected points.
        player_teams: Optional ``player_key`` / ``team_key`` map.
        owned_player_keys: Current squad keys for continuity terms.
        inventory: Remaining uses; omitted means all grants available.
        gameweeks: Optional deadline filter.

    Returns:
        Opportunity rows sorted by chip order then gameweek.
    """
    counts = team_fixture_counts(fixtures, season=season)
    landmarks = gameweek_landmarks(counts)
    if gameweeks is not None:
        landmarks = landmarks.filter(pl.col("gw").is_in(list(gameweeks)))
    landmark_by_gw: dict[int, tuple[int, int]] = {
        int(row["gw"]): (
            int(row["n_double_teams"]),
            int(row["n_blank_teams"]),
        )
        for row in landmarks.iter_rows(named=True)
    }
    fixtures_by_team = _fixture_count_lookup(counts)
    pred = _normalise_predictions(predictions)
    teams = _player_team_lookup(player_teams)
    owned = list(owned_player_keys or ())
    owned_teams = [teams[key] for key in owned if key in teams]
    player_xpts = _player_xpts_by_gw(pred)
    team_xpts = _team_xpts_by_gw(pred, teams)
    bag = inventory if inventory is not None else None
    scored: list[ChipOpportunity] = []
    for window in rules.windows:
        if as_of_gw > window.stop_event:
            continue
        grants = {grant.name: grant for grant in rules.grants_for_window(window.id)}
        gws = _window_gameweeks(window, as_of_gw, gameweeks)
        for chip in _CHIP_ORDER:
            grant = grants.get(chip)
            if grant is None:
                continue
            if _window_uses_left(rules, bag, chip, window) <= 0:
                continue
            for gameweek in gws:
                score = _score_chip(
                    chip,
                    gameweek,
                    landmark_by_gw=landmark_by_gw,
                    fixtures_by_team=fixtures_by_team,
                    owned_teams=owned_teams,
                    player_teams=teams,
                    player_xpts=player_xpts,
                    team_xpts=team_xpts,
                    window=window,
                )
                scored.append(
                    ChipOpportunity(
                        chip=grant.name,
                        window=grant.window,
                        gameweek=gameweek,
                        score=score,
                    )
                )
    scored.sort(key=lambda row: (_CHIP_ORDER.index(row.chip), row.window, row.gameweek))
    return tuple(scored)


def windows_from_opportunities(
    opportunities: Sequence[ChipOpportunity],
    *,
    as_of_gw: int,
    short_horizon: int = DEFAULT_PLAN_HORIZON,
    rules: ChipRules | None = None,
    force_play: bool = True,
) -> ChipUseWindows:
    """Apply the option-value cut and emit ``ChipUseWindows``.

    Args:
        opportunities: Output of ``score_chip_opportunities``.
        as_of_gw: Current deadline.
        short_horizon: Planner look-ahead ``H``.
        rules: Optional chip windows; when set and ``force_play`` is true,
            remaining chips are force-packed onto the last eligible
            gameweeks of each half.
        force_play: When false, skip half-window force-packing even if
            ``rules`` is set. Option-value unique-best forces still apply.

    Returns:
        Allowed / forced / no-chip hooks for the short-horizon MILP.
    """
    near_end = min(38, as_of_gw + short_horizon - 1)
    # Grant-keyed only: bare chip-name maps would fold first-/second-half uses.
    by_grant: dict[ChipGrant, list[ChipOpportunity]] = {}
    for row in opportunities:
        by_grant.setdefault(row.grant, []).append(row)

    allowed: dict[ChipName, set[int]] = {}
    forced_candidates: dict[ChipName, int] = {}
    # Option-value cut is per ChipGrant: second-half doubles must not empty the
    # first-half allow-list (and vice versa).
    for grant, rows in by_grant.items():
        scores = {row.gameweek: row.score for row in rows}
        if not scores:
            continue
        best_score = max(scores.values())
        later = {gw: score for gw, score in scores.items() if gw > near_end}
        deferred_best = max(later.values()) if later else 0.0
        keep: set[int] = set()
        for gameweek, score in scores.items():
            if gameweek <= near_end:
                if score + _SCORE_EPS >= deferred_best:
                    keep.add(gameweek)
            elif score + _SCORE_EPS >= best_score:
                keep.add(gameweek)
        best_gws = [
            gw for gw, score in scores.items() if abs(score - best_score) <= _SCORE_EPS
        ]
        keep.update(best_gws)
        allowed.setdefault(grant.name, set()).update(keep)
        if (
            len(best_gws) == 1
            and best_gws[0] <= near_end
            and best_score + _SCORE_EPS >= deferred_best
        ):
            forced_candidates[grant.name] = best_gws[0]

    if force_play and rules is not None:
        _apply_force_play_packing(
            rules,
            by_grant=by_grant,
            allowed=allowed,
            forced_candidates=forced_candidates,
            as_of_gw=as_of_gw,
            near_end=near_end,
        )

    forced = _resolve_forced_collisions(forced_candidates, by_grant)
    # Force-play packing may have dropped a colliding option-value force;
    # keep every surviving forced week allowed.
    for chip, gameweek in forced.items():
        allowed.setdefault(chip, set()).add(gameweek)

    no_chip: set[int] = set()
    if allowed:
        for gameweek in range(as_of_gw, near_end + 1):
            if any(gameweek in weeks for weeks in allowed.values()):
                continue
            no_chip.add(gameweek)
    # Forced weeks must never land in no_chip.
    for gameweek in forced.values():
        no_chip.discard(gameweek)

    return ChipUseWindows(
        allowed={chip: frozenset(weeks) for chip, weeks in allowed.items() if weeks},
        forced={chip: frozenset({gw}) for chip, gw in forced.items()},
        no_chip=frozenset(no_chip),
    )


def _apply_force_play_packing(
    rules: ChipRules,
    *,
    by_grant: Mapping[ChipGrant, Sequence[ChipOpportunity]],
    allowed: dict[ChipName, set[int]],
    forced_candidates: dict[ChipName, int],
    as_of_gw: int,
    near_end: int,
) -> None:
    """Pack remaining chips onto the last eligible GWs of each half-window.

    Triggers when ``stop_event`` is inside the short horizon, or when the
    number of remaining chips is at least the number of remaining
    gameweeks in the window (must play from now or a grant expires).
    Assignments overwrite option-value forces for the same chip; per-GW
    collisions are resolved later by ``_resolve_forced_collisions``.
    """
    for window in rules.windows:
        if as_of_gw > window.stop_event:
            continue
        grants_left: list[ChipGrant] = [
            ChipGrant(name=chip, window=window.id)
            for chip in _CHIP_ORDER
            if by_grant.get(ChipGrant(name=chip, window=window.id))
        ]
        if not grants_left:
            continue
        eligible = list(range(max(window.start_event, as_of_gw), window.stop_event + 1))
        if not eligible:
            continue
        stop_in_horizon = window.stop_event <= near_end
        scarce = len(eligible) <= len(grants_left)
        if not stop_in_horizon and not scarce:
            continue
        n_pack = min(len(grants_left), len(eligible))
        pack_gws = eligible[-n_pack:]
        score_at = {
            (row.chip, row.gameweek): row.score
            for grant in grants_left
            for row in by_grant.get(grant, ())
        }
        ranked: list[ChipGrant] = sorted(
            grants_left,
            key=lambda grant: (
                max(
                    (score_at.get((grant.name, gw), 0.0) for gw in pack_gws),
                    default=0.0,
                ),
                -_CHIP_ORDER.index(grant.name),
            ),
            reverse=True,
        )
        available = set(pack_gws)
        for grant in ranked[:n_pack]:
            best_gw = max(
                available,
                key=lambda gw: (
                    score_at.get((grant.name, gw), 0.0),
                    gw,  # prefer later GW on ties → last eligible
                ),
            )
            forced_candidates[grant.name] = best_gw
            allowed.setdefault(grant.name, set()).add(best_gw)
            available.remove(best_gw)


def _resolve_forced_collisions(
    forced_candidates: Mapping[ChipName, int],
    by_grant: Mapping[ChipGrant, Sequence[ChipOpportunity]],
) -> dict[ChipName, int]:
    """Keep at most one forced chip per gameweek (higher score wins)."""
    score_at = {
        (row.chip, row.gameweek): row.score
        for rows in by_grant.values()
        for row in rows
    }
    by_gw: dict[int, list[ChipName]] = {}
    for chip, gameweek in forced_candidates.items():
        by_gw.setdefault(gameweek, []).append(chip)
    chosen: dict[ChipName, int] = {}
    for gameweek, chips in by_gw.items():
        winner = max(
            chips,
            key=lambda chip: (
                score_at.get((chip, gameweek), 0.0),
                -_CHIP_ORDER.index(chip),
            ),
        )
        chosen[winner] = gameweek
    return chosen


def _score_chip(
    chip: ChipName,
    gameweek: int,
    *,
    landmark_by_gw: Mapping[int, tuple[int, int]],
    fixtures_by_team: Mapping[tuple[int, int], int],
    owned_teams: Sequence[int],
    player_teams: Mapping[int, int],
    player_xpts: Mapping[tuple[int, int], float],
    team_xpts: Mapping[tuple[int, int], float],
    window: ChipWindow,
) -> float:
    n_double, n_blank = landmark_by_gw.get(gameweek, (0, 0))
    owned_doubles = sum(
        1 for team in owned_teams if fixtures_by_team.get((gameweek, team), 0) >= 2
    )
    owned_blanks = sum(
        1 for team in owned_teams if fixtures_by_team.get((gameweek, team), 0) == 0
    )
    mean_owned = _mean_owned_xpts(gameweek, owned_teams, team_xpts, player_xpts)
    peak = _peak_player_xpts(gameweek, player_xpts)
    peak_team_double = _peak_on_double(
        gameweek, player_xpts, player_teams, fixtures_by_team, n_double
    )
    if chip == "bboost":
        squad_term = owned_doubles * mean_owned
        league_term = float(n_double) * 0.5
        return squad_term + league_term
    if chip == "3xc":
        double_bonus = 0.5 if peak_team_double else 0.0
        return peak * (1.0 + double_bonus) + 0.1 * float(n_double)
    if chip == "freehit":
        squad_term = owned_blanks * mean_owned
        league_term = float(n_blank) * 0.5
        return squad_term + league_term
    # Wildcard: restructure before the next double in this half.
    next_double = _next_double_gw(gameweek, landmark_by_gw, window)
    if next_double is None:
        return float(owned_blanks) + 0.1 * float(n_blank)
    ahead_doubles = sum(
        1 for team in owned_teams if fixtures_by_team.get((next_double, team), 0) >= 2
    )
    mismatch = max(0, len(owned_teams) - ahead_doubles) if owned_teams else 15
    gap = next_double - gameweek
    proximity = 1.0 / float(gap) if gap > 0 else 0.0
    next_n_double, _ = landmark_by_gw.get(next_double, (0, 0))
    return float(mismatch) * proximity + 0.25 * float(next_n_double)


def _next_double_gw(
    gameweek: int,
    landmark_by_gw: Mapping[int, tuple[int, int]],
    window: ChipWindow,
) -> int | None:
    candidates = [
        gw
        for gw, (n_double, _) in landmark_by_gw.items()
        if gameweek < gw <= window.stop_event and n_double > 0
    ]
    return min(candidates) if candidates else None


def _window_gameweeks(
    window: ChipWindow,
    as_of_gw: int,
    gameweeks: Sequence[int] | None,
) -> list[int]:
    start = max(window.start_event, as_of_gw)
    gws = list(range(start, window.stop_event + 1))
    if gameweeks is not None:
        allowed = set(gameweeks)
        gws = [gw for gw in gws if gw in allowed]
    return gws


def _window_uses_left(
    rules: ChipRules,
    inventory: ChipInventory | None,
    chip: ChipName,
    window: ChipWindow,
) -> int:
    """Return uses left for ``chip`` inside ``window`` (mirrors plan caps)."""
    granted = sum(
        1 for grant in rules.grants if grant.name == chip and grant.window == window.id
    )
    if inventory is None:
        return granted
    if inventory.remaining.get(chip, 0) <= 0:
        return 0
    played = sum(
        1
        for play in inventory.played
        if play.name == chip and window.start_event <= play.gw <= window.stop_event
    )
    return max(0, granted - played)


def _fixture_count_lookup(
    counts: pl.DataFrame,
) -> dict[tuple[int, int], int]:
    if counts.is_empty():
        return {}
    return {
        (int(gw), int(team)): int(n)
        for gw, team, n in counts.select("gw", "team_key", "n_fixtures").iter_rows()
    }


def _normalise_predictions(predictions: pl.DataFrame | None) -> pl.DataFrame:
    if predictions is None or predictions.is_empty():
        return pl.DataFrame(
            schema={
                "gw": pl.Int64(),
                "player_key": pl.Int64(),
                "xpts": pl.Float64(),
            }
        )
    frame = predictions
    if "gw" not in frame.columns and "target_gw" in frame.columns:
        frame = frame.rename({"target_gw": "gw"})
    needed = {"gw", "player_key", "xpts"}
    if not needed <= set(frame.columns):
        return pl.DataFrame(
            schema={
                "gw": pl.Int64(),
                "player_key": pl.Int64(),
                "xpts": pl.Float64(),
            }
        )
    return frame.select("gw", "player_key", "xpts")


def _player_team_lookup(player_teams: pl.DataFrame | None) -> dict[int, int]:
    if player_teams is None or player_teams.is_empty():
        return {}
    if not {"player_key", "team_key"} <= set(player_teams.columns):
        return {}
    return {
        int(player): int(team)
        for player, team in player_teams.select("player_key", "team_key")
        .unique(subset=["player_key"])
        .iter_rows()
    }


def _player_xpts_by_gw(
    predictions: pl.DataFrame,
) -> dict[tuple[int, int], float]:
    if predictions.is_empty():
        return {}
    return {
        (int(gw), int(player)): float(xpts)
        for gw, player, xpts in predictions.select(
            "gw", "player_key", "xpts"
        ).iter_rows()
    }


def _team_xpts_by_gw(
    predictions: pl.DataFrame, teams: Mapping[int, int]
) -> dict[tuple[int, int], float]:
    if predictions.is_empty() or not teams:
        return {}
    totals: dict[tuple[int, int], float] = {}
    for gw, player, xpts in predictions.select("gw", "player_key", "xpts").iter_rows():
        team = teams.get(int(player))
        if team is None:
            continue
        key = (int(gw), int(team))
        totals[key] = totals.get(key, 0.0) + float(xpts)
    return totals


def _mean_owned_xpts(
    gameweek: int,
    owned_teams: Sequence[int],
    team_xpts: Mapping[tuple[int, int], float],
    player_xpts: Mapping[tuple[int, int], float],
) -> float:
    if owned_teams:
        values = [team_xpts.get((gameweek, team), 0.0) for team in owned_teams]
        if values:
            return sum(values) / float(len(values))
    week = [score for (gw, _), score in player_xpts.items() if gw == gameweek]
    if not week:
        return 1.0
    return sum(week) / float(len(week))


def _peak_player_xpts(
    gameweek: int, player_xpts: Mapping[tuple[int, int], float]
) -> float:
    week = [score for (gw, _), score in player_xpts.items() if gw == gameweek]
    return max(week) if week else 0.0


def _peak_on_double(
    gameweek: int,
    player_xpts: Mapping[tuple[int, int], float],
    player_teams: Mapping[int, int],
    fixtures_by_team: Mapping[tuple[int, int], int],
    n_double_teams: int,
) -> bool:
    """Return whether the peak-xpts player's team has a double this week."""
    peak = _peak_player_xpts(gameweek, player_xpts)
    if peak <= 0.0:
        return n_double_teams > 0
    for (gw, player), score in player_xpts.items():
        if gw != gameweek or abs(score - peak) > _SCORE_EPS:
            continue
        team = player_teams.get(player)
        if team is None:
            return n_double_teams > 0
        return fixtures_by_team.get((gameweek, team), 0) >= 2
    return n_double_teams > 0
