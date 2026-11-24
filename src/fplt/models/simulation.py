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
that variance. Optional ``match_other`` adds scoreline-conditioned offsets
(team goals / clean sheet / player goals) on top of the minutes-bucket mean.
Minutes are taken as drawn; hard eleven-starter constraints are applied
upstream via ``enforce_hard_starters`` before this module runs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Protocol

import numpy as np
import polars as pl

from fplt.scoring.rules import Rules

_BUCKET_EDGES = (0, 60)
ConcededMode = Literal["scaled", "played"]


class MatchOtherOffsets(Protocol):
    """Optional match-conditioned additive other-points offsets."""

    def match_delta(
        self,
        *,
        positions: np.ndarray,
        minutes: np.ndarray,
        team_goals: np.ndarray,
        clean_sheet: np.ndarray,
        player_goals: np.ndarray,
    ) -> np.ndarray:
        """Return additive other points with shape ``(players, sims)``."""
        ...


@dataclass(frozen=True)
class SideAllocateUniforms:
    """Injected U(0,1) draws for one side's goal / assist loop.

    Each array has shape ``(G, n)`` with ``G`` at least ``max(team_goals)``
    and ``n`` the number of simulations. Row ``g`` is the ``g``-th goal
    iteration (scorer, assist Bernoulli, then assister). Used by L2 exact
    equivalence against ``fplt_mc._testing``; production keeps drawing from
    ``rng``.
    """

    scorer: np.ndarray
    assist_flag: np.ndarray
    assister: np.ndarray


@dataclass(frozen=True)
class MatchSimUniforms:
    """Optional injected uniforms for ``simulate_match`` logic-only tests.

    Any field left ``None`` is drawn from ``rng`` exactly as before. Shapes:

    - ``home`` / ``away``: :class:`SideAllocateUniforms` for that side.
    - ``cs_void``: ``(players, n)`` compared to ``cs_red_void_prob``.
    - ``other``: ``(players, n)`` for sequential-search Poisson when
      ``sample_other`` is true (native L2 path). When ``None``, production
      still uses ``rng.poisson``.
    """

    home: SideAllocateUniforms | None = None
    away: SideAllocateUniforms | None = None
    cs_void: np.ndarray | None = None
    other: np.ndarray | None = None


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
    match_other: MatchOtherOffsets | None = None,
    uniforms: MatchSimUniforms | None = None,
    p_penalty: float = 0.0,
    penalty_taker: np.ndarray | None = None,
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
            ``seed_for``. Ignored for any stream supplied via ``uniforms``.
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
        match_other: Optional match-conditioned offsets added after the
            minutes-bucket mean (team goals / CS / player goals).
        uniforms: Optional injected U(0,1) arrays for L2 logic tests. ``None``
            keeps the production RNG path (bit-identical to pre-PR2).
        p_penalty: Probability that a team goal is treated as a penalty
            (no assist; awarded to a playing designated taker). ``0`` keeps
            open-play allocation only (production default).
        penalty_taker: Boolean mask aligned with ``players`` rows. Ignored
            when ``p_penalty`` is 0.

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
    if not 0.0 <= p_penalty <= 1.0:
        raise ValueError(f"p_penalty must be in [0, 1], got {p_penalty}")
    if conceded_mode not in ("scaled", "played"):
        raise ValueError(
            f"conceded_mode must be 'scaled' or 'played', got {conceded_mode!r}"
        )
    taker_mask = _penalty_taker_mask(penalty_taker, keys.shape[0], p_penalty)
    goals = np.zeros((keys.shape[0], n), dtype=np.int64)
    assists = np.zeros_like(goals)
    unattributed = np.zeros((2, n), dtype=np.int64)
    for side_index, side_goals in enumerate((home, away)):
        mask = sides == ("home" if side_index == 0 else "away")
        side_uniforms = None
        if uniforms is not None:
            side_uniforms = uniforms.home if side_index == 0 else uniforms.away
        side_goals_out, side_assists, missed = _allocate_side(
            side_goals,
            played[mask],
            goal_weight[mask],
            assist_weight[mask],
            own_goal_weight,
            p_assist,
            rng,
            uniforms=side_uniforms,
            p_penalty=p_penalty,
            penalty_taker=taker_mask[mask],
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
        match_other=match_other,
        cs_void_uniforms=None if uniforms is None else uniforms.cs_void,
        other_uniforms=None if uniforms is None else uniforms.other,
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
    *,
    uniforms: SideAllocateUniforms | None = None,
    p_penalty: float = 0.0,
    penalty_taker: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Allocate goals and assists for one side across simulations.

    Args:
        team_goals: Goals scored by this side per sim, shape ``(n,)``.
        minutes: Minutes for this side's players, shape ``(players, n)``.
        goal_weight: Per-player goal involvement weights, shape ``(players,)``.
        assist_weight: Per-player assist involvement weights, shape ``(players,)``.
        own_goal_weight: Extra mass for the unattributed bucket.
        p_assist: Probability a claimed goal also gets an assist.
        rng: Source for uniforms when ``uniforms`` is ``None``.
        uniforms: Optional injected scorer / assist-flag / assister draws.
        p_penalty: Probability a goal is a penalty (taker, no assist).
        penalty_taker: Boolean mask of designated takers, shape ``(players,)``.

    Returns:
        ``(scored, assisted, missed)`` with shapes ``(players, n)``,
        ``(players, n)``, and ``(n,)``.
    """
    players = int(minutes.shape[0])
    n = int(minutes.shape[1])
    scored = np.zeros((players, n), dtype=np.int64)
    assisted = np.zeros_like(scored)
    missed = np.zeros(n, dtype=np.int64)
    max_goals = int(team_goals.max(initial=0))
    if n == 0 or max_goals == 0:
        return scored, assisted, missed
    playing = minutes > 0
    goal_rates = np.where(playing, goal_weight[:, None] * minutes / 90.0, 0.0)
    assist_rates = np.where(playing, assist_weight[:, None] * minutes / 90.0, 0.0)
    if players == 0:
        return scored, assisted, team_goals.copy()
    use_pens = (
        p_penalty > 0.0 and penalty_taker is not None and bool(np.any(penalty_taker))
    )
    taker_flags = (
        np.asarray(penalty_taker, dtype=bool)
        if use_pens
        else np.zeros(players, dtype=bool)
    )
    injected: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None
    if uniforms is not None:
        injected = (
            _require_uniform_grid(uniforms.scorer, max_goals, n, "scorer"),
            _require_uniform_grid(uniforms.assist_flag, max_goals, n, "assist_flag"),
            _require_uniform_grid(uniforms.assister, max_goals, n, "assister"),
        )
    for goal_index in range(max_goals):
        active = team_goals > scored.sum(axis=0) + missed
        if not np.any(active):
            break
        # Draw lazily so an early break consumes the same RNG stream as before.
        if injected is None:
            draw = rng.random(n)
            assist_draw = rng.random(n)
            second = rng.random(n)
        else:
            draw = injected[0][goal_index]
            assist_draw = injected[1][goal_index]
            second = injected[2][goal_index]
        pen_draw = rng.random(n) if use_pens else None
        is_pen = (
            active & (pen_draw < p_penalty)
            if use_pens and pen_draw is not None
            else np.zeros(n, dtype=bool)
        )
        column = np.arange(n)
        if np.any(is_pen):
            pen_scorer = _penalty_scorer(playing, taker_flags, is_pen)
            pen_claimed = is_pen & (pen_scorer >= 0)
            missed += is_pen & (pen_scorer < 0)
            np.add.at(scored, (np.maximum(pen_scorer, 0), column), pen_claimed)
        open_play = active & ~is_pen
        if not np.any(open_play):
            continue
        scorer = _pick(goal_rates, draw, own_goal_weight)
        claimed = open_play & (scorer >= 0)
        missed += open_play & (scorer < 0)
        np.add.at(scored, (np.maximum(scorer, 0), column), claimed)
        want_assist = claimed & (assist_draw < p_assist)
        blocked = assist_rates.copy()
        scored_now = scorer >= 0
        blocked[np.maximum(scorer, 0), column] = np.where(
            scored_now, 0.0, blocked[np.maximum(scorer, 0), column]
        )
        assister = _pick(blocked, second, 0.0)
        given = want_assist & (assister >= 0)
        np.add.at(assisted, (np.maximum(assister, 0), column), given)
    return scored, assisted, missed


def _penalty_taker_mask(
    penalty_taker: np.ndarray | None, n_players: int, p_penalty: float
) -> np.ndarray:
    if p_penalty <= 0.0 or penalty_taker is None:
        return np.zeros(n_players, dtype=bool)
    mask = np.asarray(penalty_taker, dtype=bool)
    if mask.shape != (n_players,):
        raise ValueError(
            f"penalty_taker must have shape ({n_players},), got {mask.shape}"
        )
    return mask


def _penalty_scorer(
    playing: np.ndarray, taker_flags: np.ndarray, is_pen: np.ndarray
) -> np.ndarray:
    """Index of the playing penalty taker per sim, or ``-1`` when none."""
    n = int(is_pen.shape[0])
    out = np.full(n, -1, dtype=np.int64)
    if not np.any(taker_flags):
        return out
    eligible = playing & taker_flags[:, None]
    for sim in np.flatnonzero(is_pen):
        choices = np.flatnonzero(eligible[:, sim])
        if choices.size:
            out[sim] = int(choices[0])
    return out


def _require_uniform_grid(
    values: np.ndarray, rows: int, cols: int, name: str
) -> np.ndarray:
    """Validate an injected ``(rows, cols)`` (or taller) uniform grid."""
    grid = np.asarray(values, dtype=np.float64)
    if grid.ndim != 2 or grid.shape[0] < rows or grid.shape[1] != cols:
        raise ValueError(
            f"{name} uniforms must have shape (>= {rows}, {cols}), got {grid.shape}"
        )
    return grid


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
    match_other: MatchOtherOffsets | None = None,
    cs_void_uniforms: np.ndarray | None = None,
    other_uniforms: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Score appearance, involvement, CS, conceded, and other points.

    Args:
        cs_void_uniforms: Optional ``(players, n)`` uniforms for the red-card
            CS void. ``None`` draws from ``rng``.
        other_uniforms: Optional ``(players, n)`` uniforms for sequential-search
            other-points sampling. ``None`` keeps ``rng.poisson``.
    """
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
        void_draw = (
            rng.random(minutes.shape)
            if cs_void_uniforms is None
            else np.asarray(cs_void_uniforms, dtype=np.float64)
        )
        if void_draw.shape != minutes.shape:
            raise ValueError(
                "cs_void uniforms must have shape "
                f"{minutes.shape}, got {void_draw.shape}"
            )
        voided = void_draw < cs_red_void_prob
        clean_sheet = clean_sheet & ~voided
    if conceded_mode == "played":
        exposure = np.where(minutes > 0, 1.0, 0.0)
    else:
        exposure = np.where(minutes > 0, minutes / 90.0, 0.0)
    conceded = conceded_value[:, None] * (opponent // every) * exposure
    bucket = np.where(minutes <= 0, 0, np.where(minutes < _BUCKET_EDGES[1], 1, 2))
    other_mean = np.take_along_axis(other_points, bucket, axis=1)
    if match_other is not None:
        team_goals = np.where(sides[:, None] == "home", home[None, :], away[None, :])
        other_mean = other_mean + match_other.match_delta(
            positions=positions,
            minutes=minutes,
            team_goals=team_goals,
            clean_sheet=clean_sheet,
            player_goals=goals,
        )
    if sample_other:
        other = _draw_other(other_mean, rng, uniforms=other_uniforms)
    else:
        other = other_mean
    points = (
        appearance
        + goal_value[:, None] * goals
        + rules.assists * assists
        + sheet_value[:, None] * clean_sheet
        + conceded
        + other
    )
    return points, clean_sheet.astype(np.bool_)


def _draw_other(
    means: np.ndarray,
    rng: np.random.Generator,
    *,
    uniforms: np.ndarray | None = None,
) -> np.ndarray:
    """Sample other points with mean equal to ``means`` (Poisson / signed).

    Production (``uniforms is None``) uses ``rng.poisson``. When uniforms are
    injected, use sequential-search inversion so L2 can match the native
    Philox path with the same U(0,1) array.
    """
    rate = np.abs(means)
    if uniforms is None:
        drawn = rng.poisson(rate).astype(np.float64)
    else:
        drawn = _poisson_sequential(rate, uniforms).astype(np.float64)
    return np.where(means >= 0.0, drawn, -drawn)


def _poisson_sequential(rate: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    """Inverse-CDF Poisson via sequential search (small-λ other-points path).

    Args:
        rate: Non-negative λ, any shape.
        uniforms: U(0,1) draws, same shape as ``rate``.

    Returns:
        Integer counts, same shape as ``rate``.
    """
    u = np.asarray(uniforms, dtype=np.float64)
    lam = np.asarray(rate, dtype=np.float64)
    if u.shape != lam.shape:
        raise ValueError(f"other uniforms must have shape {lam.shape}, got {u.shape}")
    out = np.zeros(lam.shape, dtype=np.int64)
    # λ == 0 → always 0; skip search (and avoid exp(0) churn on empties).
    active = lam > 0.0
    if not np.any(active):
        return out
    # Sequential search per cell; other-points λ is small so this is fine.
    flat_lam = lam.reshape(-1)
    flat_u = u.reshape(-1)
    flat_out = out.reshape(-1)
    for index in range(flat_lam.size):
        intensity = flat_lam[index]
        if intensity <= 0.0:
            continue
        threshold = flat_u[index]
        prob = np.exp(-intensity)
        cumulative = prob
        count = 0
        while threshold > cumulative:
            count += 1
            prob *= intensity / count
            cumulative += prob
        flat_out[index] = count
    return out


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
