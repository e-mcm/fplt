"""Monte Carlo composition of a scoreline, minutes, and player shares.

v1 defaults (``cs_red_void_prob=0``, ``conceded_mode="scaled"``,
``sample_other=False``):

- Clean sheet if the opponent scored zero and the player played 60 or more
  minutes.
- Goals-conceded penalty is the full-match table value times minutes/90.
- Other points are the bucket expectation with no noise.

Fidelity options (enabled on ``dc_layered_v2``) can void clean sheets at a
red-card rate, apply the full conceded penalty whenever the player appears, and
sample other points around their expectation so ``xpts_sd`` / ``p_haul`` see
that variance. Minutes are taken as drawn; hard eleven-starter constraints are
applied upstream via ``enforce_hard_starters`` before this module runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import polars as pl

from fplt.scoring.rules import Rules

_BUCKET_EDGES = (0, 60)
ConcededMode = Literal["scaled", "played"]


@dataclass(frozen=True)
class MatchSimulation:
    """One fixture, simulated ``n`` times.

    ``goals`` counts only goals given to a player. ``unattributed_home`` and
    ``unattributed_away`` are the own-goal bucket, so each side's player
    goals plus that bucket equal the sampled scoreline. ``clean_sheets`` is
    the per-sim indicator after the red-card void draw.
    """

    player_key: np.ndarray
    points: np.ndarray
    goals: np.ndarray
    assists: np.ndarray
    clean_sheets: np.ndarray
    unattributed_home: np.ndarray
    unattributed_away: np.ndarray


def simulate_match(
    scorelines: np.ndarray,
    players: pl.DataFrame,
    minutes: np.ndarray,
    other_points: np.ndarray,
    rules: Rules,
    rng: np.random.Generator,
    *,
    own_goal_weight: float = 0.05,
    p_assist: float = 0.75,
    cs_red_void_prob: float = 0.0,
    conceded_mode: ConcededMode = "scaled",
    sample_other: bool = False,
) -> MatchSimulation:
    """Simulate points for every player across the sampled scorelines.

    Args:
        scorelines: Integer home and away goals, shape ``(n, 2)``.
        players: ``player_key``, ``side`` (``home`` or ``away``), ``position``,
            ``goal_weight``, and ``assist_weight``. One row per player.
        minutes: Drawn minutes, shape ``(players, n)``. A zero draw cannot
            be given a goal or an assist.
        other_points: Expected bonus, saves, cards, and the rest for the
            buckets 0, 1–59, and 60+, shape ``(players, 3)``.
        rules: Scoring table for the fixture's season.
        rng: Generator for the allocations. Callers should build it with
            ``seed_for``.
        own_goal_weight: Weight of the unattributed bucket, in the same units
            as ``goal_weight * minutes / 90``.
        p_assist: Probability that a goal has an assist. The assister is
            someone else with minutes, weighted by ``assist_weight``.
        cs_red_void_prob: Probability that a would-be clean sheet is voided
            (red-card approximation). ``0`` keeps the v1 rule.
        conceded_mode: ``"scaled"`` multiplies the goals-conceded penalty by
            minutes/90 (v1). ``"played"`` applies the full penalty whenever
            minutes are positive.
        sample_other: When true, draw other points around the bucket mean
            (Poisson / signed Poisson) instead of plugging in the mean.

    Returns:
        Points, goals, and assists per player, plus the unattributed goals.

    Raises:
        ValueError: Shapes, sides, weights, or fidelity params are invalid.
    """
    home, away = _scorelines(scorelines)
    n = home.shape[0]
    keys, sides, positions, goal_weight, assist_weight = _players(players)
    played = _minutes(minutes, keys.shape[0], n)
    other = _other(other_points, keys.shape[0])
    if own_goal_weight < 0:
        raise ValueError(f"own_goal_weight must be non-negative, got {own_goal_weight}")
    if not 0.0 <= p_assist <= 1.0:
        raise ValueError(f"p_assist must be in [0, 1], got {p_assist}")
    if not 0.0 <= cs_red_void_prob <= 1.0:
        raise ValueError(f"cs_red_void_prob must be in [0, 1], got {cs_red_void_prob}")
    if conceded_mode not in ("scaled", "played"):
        raise ValueError(
            f"conceded_mode must be 'scaled' or 'played', got {conceded_mode!r}"
        )
    goals = np.zeros((keys.shape[0], n), dtype=np.int64)
    assists = np.zeros_like(goals)
    unattributed = np.zeros((2, n), dtype=np.int64)
    for side_index, side_goals in enumerate((home, away)):
        mask = sides == ("home" if side_index == 0 else "away")
        side_goals_out, side_assists, missed = _allocate_side(
            side_goals,
            played[mask],
            goal_weight[mask],
            assist_weight[mask],
            own_goal_weight,
            p_assist,
            rng,
        )
        goals[mask] = side_goals_out
        assists[mask] = side_assists
        unattributed[side_index] = missed
    points, clean_sheets = _points(
        rules,
        positions,
        sides,
        played,
        goals,
        assists,
        home,
        away,
        other,
        rng,
        cs_red_void_prob=cs_red_void_prob,
        conceded_mode=conceded_mode,
        sample_other=sample_other,
    )
    return MatchSimulation(
        player_key=keys,
        points=points,
        goals=goals,
        assists=assists,
        clean_sheets=clean_sheets,
        unattributed_home=unattributed[0],
        unattributed_away=unattributed[1],
    )


def _allocate_side(
    team_goals: np.ndarray,
    minutes: np.ndarray,
    goal_weight: np.ndarray,
    assist_weight: np.ndarray,
    own_goal_weight: float,
    p_assist: float,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    players = int(minutes.shape[0])
    n = int(minutes.shape[1])
    scored = np.zeros((players, n), dtype=np.int64)
    assisted = np.zeros_like(scored)
    missed = np.zeros(n, dtype=np.int64)
    if n == 0 or int(team_goals.max(initial=0)) == 0:
        return scored, assisted, missed
    playing = minutes > 0
    goal_rates = np.where(playing, goal_weight[:, None] * minutes / 90.0, 0.0)
    assist_rates = np.where(playing, assist_weight[:, None] * minutes / 90.0, 0.0)
    if players == 0:
        return scored, assisted, team_goals.copy()
    for _ in range(int(team_goals.max())):
        active = team_goals > scored.sum(axis=0) + missed
        if not np.any(active):
            break
        draw: np.ndarray = rng.random(n)
        scorer = _pick(goal_rates, draw, own_goal_weight)
        claimed = active & (scorer >= 0)
        missed += active & (scorer < 0)
        column = np.arange(n)
        np.add.at(scored, (np.maximum(scorer, 0), column), claimed)
        assist_draw: np.ndarray = rng.random(n)
        want_assist = claimed & (assist_draw < p_assist)
        blocked = assist_rates.copy()
        scored_now = scorer >= 0
        blocked[np.maximum(scorer, 0), column] = np.where(
            scored_now, 0.0, blocked[np.maximum(scorer, 0), column]
        )
        second: np.ndarray = rng.random(n)
        assister = _pick(blocked, second, 0.0)
        given = want_assist & (assister >= 0)
        np.add.at(assisted, (np.maximum(assister, 0), column), given)
    return scored, assisted, missed


def _pick(weights: np.ndarray, draw: np.ndarray, extra: float) -> np.ndarray:
    """Player index per simulation, or -1 for the extra bucket."""
    player_mass = weights.sum(axis=0)
    total = player_mass + extra
    scale = np.where(total > 0, total, 1.0)
    above = np.cumsum(weights, axis=0) > (draw * scale)
    chosen = above.argmax(axis=0)
    none = ~above.any(axis=0) | (total <= 0)
    return np.where(none, -1, chosen).astype(np.int64)


def _points(
    rules: Rules,
    positions: np.ndarray,
    sides: np.ndarray,
    minutes: np.ndarray,
    goals: np.ndarray,
    assists: np.ndarray,
    home: np.ndarray,
    away: np.ndarray,
    other_points: np.ndarray,
    rng: np.random.Generator,
    *,
    cs_red_void_prob: float,
    conceded_mode: ConcededMode,
    sample_other: bool,
) -> tuple[np.ndarray, np.ndarray]:
    goal_value = np.array([rules.goals[pos] for pos in positions], dtype=np.float64)
    sheet_value = np.array(
        [rules.clean_sheets[pos] for pos in positions], dtype=np.float64
    )
    conceded_value = np.array(
        [
            rules.goals_conceded.points if pos in rules.goals_conceded.positions else 0
            for pos in positions
        ],
        dtype=np.float64,
    )
    opponent = np.where(sides[:, None] == "home", away[None, :], home[None, :])
    every = rules.goals_conceded.every
    long = rules.appearance.long_minutes
    appearance = np.where(
        minutes >= long,
        rules.appearance.long,
        np.where(minutes > 0, rules.appearance.short, 0),
    )
    clean_sheet = (opponent == 0) & (minutes >= long)
    if cs_red_void_prob > 0.0:
        voided = rng.random(minutes.shape) < cs_red_void_prob
        clean_sheet = clean_sheet & ~voided
    if conceded_mode == "played":
        exposure = np.where(minutes > 0, 1.0, 0.0)
    else:
        exposure = np.where(minutes > 0, minutes / 90.0, 0.0)
    conceded = conceded_value[:, None] * (opponent // every) * exposure
    bucket = np.where(minutes <= 0, 0, np.where(minutes < _BUCKET_EDGES[1], 1, 2))
    other_mean = np.take_along_axis(other_points, bucket, axis=1)
    other = _draw_other(other_mean, rng) if sample_other else other_mean
    points = (
        appearance
        + goal_value[:, None] * goals
        + rules.assists * assists
        + sheet_value[:, None] * clean_sheet
        + conceded
        + other
    )
    return points, clean_sheet.astype(np.bool_)


def _draw_other(means: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Sample other points with mean equal to ``means`` (Poisson / signed)."""
    rate = np.abs(means)
    drawn = rng.poisson(rate).astype(np.float64)
    return np.where(means >= 0.0, drawn, -drawn)


def _scorelines(scorelines: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grid = np.asarray(scorelines)
    if grid.ndim != 2 or grid.shape[1] != 2:
        raise ValueError(f"scorelines must have shape (n, 2), got {grid.shape}")
    if np.any(grid < 0):
        raise ValueError("scorelines must be non-negative")
    return grid[:, 0].astype(np.int64), grid[:, 1].astype(np.int64)


def _players(
    players: pl.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    required = {"player_key", "side", "position", "goal_weight", "assist_weight"}
    missing = required - set(players.columns)
    if missing:
        raise ValueError(f"players is missing {sorted(missing)}")
    sides = players["side"].to_numpy()
    if not set(sides).issubset({"home", "away"}):
        raise ValueError("side must be 'home' or 'away'")
    goal_weight = players["goal_weight"].to_numpy().astype(np.float64)
    assist_weight = players["assist_weight"].to_numpy().astype(np.float64)
    if np.any(goal_weight < 0) or np.any(assist_weight < 0):
        raise ValueError("weights must be non-negative")
    return (
        players["player_key"].to_numpy(),
        sides,
        players["position"].to_numpy(),
        goal_weight,
        assist_weight,
    )


def _minutes(minutes: np.ndarray, players: int, sims: int) -> np.ndarray:
    drawn = np.asarray(minutes)
    if drawn.shape != (players, sims):
        raise ValueError(
            f"minutes must have shape {(players, sims)}, got {drawn.shape}"
        )
    if np.any(drawn < 0):
        raise ValueError("minutes must be non-negative")
    return drawn.astype(np.float64)


def _other(other_points: np.ndarray, players: int) -> np.ndarray:
    grid = np.asarray(other_points, dtype=np.float64)
    if grid.shape != (players, 3):
        raise ValueError(
            f"other_points must have shape {(players, 3)}, got {grid.shape}"
        )
    return grid
