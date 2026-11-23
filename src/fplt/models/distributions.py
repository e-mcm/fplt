"""Scoreline and minutes distributions used by the layered match model.

The scoreline grid stops at 32 goals a side. A heavier tail would move
expected goals after renormalisation, so construction refuses a truncation
that drops 1e-6 or more of the probability. Attack and defence are bounded
by 3 before rescaling, and that rescaling keeps their product, so with home
advantage 1.3 a rate can reach about 12. Thirty-two keeps that tail under
the limit. Minutes sit on a fixed set of bins so a draw is a minute count
the scoring rules can apply directly.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl

MAX_GOALS = 32
MINUTE_BINS = np.array([0, 15, 30, 45, 60, 75, 90], dtype=np.int64)
_MASS_LOST_LIMIT = 1e-6
_Side = Literal["home", "away"]


class ScorelineDist:
    """Joint distribution over home and away goals for each fixture.

    ``pmf[i, h, a]`` is the probability that fixture ``i`` finishes ``h``-``a``.
    Both axes run from 0 through ``MAX_GOALS``.
    """

    def __init__(self, fixture_ids: np.ndarray, pmf: np.ndarray) -> None:
        """Truncate ``pmf`` at ``MAX_GOALS`` and renormalise.

        Args:
            fixture_ids: One id per fixture, shape ``(F,)``.
            pmf: Non-negative joint masses, shape ``(F, G, G)`` with ``G``
                greater than ``MAX_GOALS``. Mass outside that square must be
                under 1e-6 of each fixture's total.

        Raises:
            ValueError: The shape, signs, or discarded tail is invalid.
        """
        ids = np.asarray(fixture_ids)
        grid = np.asarray(pmf, dtype=np.float64)
        if ids.ndim != 1:
            raise ValueError(f"fixture_ids must be 1-D, got shape {ids.shape}")
        if grid.ndim != 3 or grid.shape[1] != grid.shape[2]:
            raise ValueError(f"pmf must have shape (F, G, G), got {grid.shape}")
        if grid.shape[0] != ids.shape[0]:
            raise ValueError("pmf and fixture_ids have different lengths")
        if grid.shape[1] <= MAX_GOALS:
            raise ValueError(f"pmf must cover goals 0..{MAX_GOALS}")
        if np.any(grid < 0):
            raise ValueError("pmf must be non-negative")
        kept = grid[:, : MAX_GOALS + 1, : MAX_GOALS + 1]
        total = grid.sum(axis=(1, 2))
        if np.any(total <= 0):
            raise ValueError("each fixture needs positive probability mass")
        lost = (total - kept.sum(axis=(1, 2))) / total
        if np.any(lost >= _MASS_LOST_LIMIT):
            worst = float(np.max(lost))
            raise ValueError(f"truncation drops {worst:.3e} of a fixture's mass")
        self.fixture_ids = ids
        self.pmf = kept / kept.sum(axis=(1, 2), keepdims=True)

    def p_clean_sheet(self, side: _Side) -> np.ndarray:
        """Probability the named side keeps a clean sheet.

        A home clean sheet is the away team scoring 0, and the reverse.

        Args:
            side: ``home`` or ``away``.

        Returns:
            Shape ``(F,)``.
        """
        if side == "home":
            return self.pmf[:, :, 0].sum(axis=1)
        if side == "away":
            return self.pmf[:, 0, :].sum(axis=1)
        raise ValueError(f"side must be 'home' or 'away', got {side!r}")

    def marginal(self, side: _Side) -> np.ndarray:
        """Goal-count probabilities for one side.

        Args:
            side: ``home`` or ``away``.

        Returns:
            Shape ``(F, MAX_GOALS + 1)``. Column ``k`` is ``P(goals = k)``.
        """
        if side == "home":
            return self.pmf.sum(axis=2)
        if side == "away":
            return self.pmf.sum(axis=1)
        raise ValueError(f"side must be 'home' or 'away', got {side!r}")

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw scorelines for every fixture.

        Args:
            rng: Generator for this draw. Callers should build it with
                ``seed_for`` so a rerun matches.
            n: Number of draws per fixture.

        Returns:
            Integer array of shape ``(F, n, 2)`` with home goals, then away.

        Raises:
            ValueError: ``n`` is negative.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        fixtures, goals, _ = self.pmf.shape
        flat = self.pmf.reshape(fixtures, goals * goals)
        cdf = np.cumsum(flat, axis=1)
        cdf[:, -1] = 1.0
        draws = rng.random((fixtures, n))
        index = _inverse_cdf_index(cdf, draws)
        home = index // goals
        away = index % goals
        return np.stack([home, away], axis=-1).astype(np.int64)


class MinutesDist:
    """Binned minutes for each player-fixture.

    ``pmf[i, b]`` is the probability that row ``i`` plays ``bins[b]`` minutes.
    Independent ``sample`` draws do not force eleven starters; callers that
    need a hard XI constraint should pass the draws through
    ``enforce_hard_starters``.
    """

    def __init__(self, keys: pl.DataFrame, bins: np.ndarray, pmf: np.ndarray) -> None:
        """Renormalise one minutes distribution per player-fixture.

        Args:
            keys: ``player_key`` and ``fixture_id``, one row per distribution.
            bins: Minute values, shape ``(B,)``.
            pmf: Non-negative masses, shape ``(P, B)``.

        Raises:
            ValueError: The keys, shape, or masses are invalid.
        """
        if not {"player_key", "fixture_id"} <= set(keys.columns):
            raise ValueError("keys need player_key and fixture_id")
        ordered = keys.select("player_key", "fixture_id")
        grid = np.asarray(pmf, dtype=np.float64)
        support = np.asarray(bins)
        if support.ndim != 1 or support.shape[0] == 0:
            raise ValueError(f"bins must be a non-empty 1-D array, got {support.shape}")
        if grid.ndim != 2 or grid.shape[1] != support.shape[0]:
            raise ValueError(
                f"pmf must have shape (P, {support.shape[0]}), got {grid.shape}"
            )
        if grid.shape[0] != ordered.height:
            raise ValueError("pmf and keys have different lengths")
        if np.any(grid < 0) or not np.all(np.isfinite(grid)):
            raise ValueError("pmf must be finite and non-negative")
        if grid.shape[0] > 0:
            total = grid.sum(axis=1)
            if np.any(total <= 0):
                raise ValueError("each player needs positive probability mass")
            grid = grid / total[:, None]
        self.keys = ordered
        self.bins = support
        self.pmf = grid

    def sample(self, rng: np.random.Generator, n: int) -> np.ndarray:
        """Draw a minute count for every player-fixture.

        Args:
            rng: Generator for this draw. Callers should build it with
                ``seed_for`` so a rerun matches.
            n: Number of draws per player-fixture.

        Returns:
            Integer array of shape ``(P, n)``. Each value is one of ``bins``.

        Raises:
            ValueError: ``n`` is negative.
        """
        if n < 0:
            raise ValueError("n must be non-negative")
        players, _width = self.pmf.shape
        if n == 0 or players == 0:
            return np.zeros((players, n), dtype=np.int64)
        cdf = np.cumsum(self.pmf, axis=1)
        cdf[:, -1] = 1.0
        draws = rng.random((players, n))
        index = _inverse_cdf_index(cdf, draws)
        return np.asarray(self.bins, dtype=np.int64)[index]

    def start_weights(self, starter_threshold: int = 60) -> np.ndarray:
        """P(minutes ≥ ``starter_threshold``) per row, used to rank hard XI."""
        bins = np.asarray(self.bins, dtype=np.int64)
        mask = bins >= starter_threshold
        rows = int(self.pmf.shape[0])
        if not np.any(mask):
            return np.zeros(rows, dtype=np.float64)
        return np.asarray(self.pmf[:, mask].sum(axis=1), dtype=np.float64)


def enforce_hard_starters(
    minutes: np.ndarray,
    team_keys: np.ndarray,
    start_weight: np.ndarray,
    rng: np.random.Generator,
    *,
    positions: np.ndarray,
    target: int = 11,
    starter_threshold: int = 60,
    start_minutes: int = 90,
    sub_minutes: int = 15,
    gk_position: str = "GK",
    tiebreak_noise: np.ndarray | None = None,
) -> np.ndarray:
    """Force about ``target`` starters per team-fixture column with 1 GK.

    A starter is a draw with minutes ≥ ``starter_threshold``. When a team has
    more than ``target`` starters in a sim, the lowest-weight extras are
    demoted to ``sub_minutes``. When fewer, the highest-weight non-starters
    with positive ``start_weight`` are promoted to ``start_minutes``. Teams
    with fewer than ``target`` eligible players keep ``min(target, eligible)``.

    Position-aware: when at least one GK has positive ``start_weight``, exactly
    one GK is kept among the starters and the remaining slots go to outfield
    players. With no eligible GK the outfield fills up to ``target`` alone.

    Args:
        minutes: Drawn minutes, shape ``(P, n)``.
        team_keys: Team id per player row, shape ``(P,)``.
        start_weight: Ranking weight per player (e.g. P(≥60)), shape ``(P,)``.
        rng: Generator for tie-breaking among equal weights when
            ``tiebreak_noise`` is ``None``.
        positions: Position string per player row, shape ``(P,)``.
        target: Desired starter count per team (usually 11).
        starter_threshold: Minutes that count as a start.
        start_minutes: Minutes assigned when promoting a player.
        sub_minutes: Minutes assigned when demoting a player.
        gk_position: Position label treated as goalkeeper (default ``GK``).
        tiebreak_noise: Optional injected U(0,1) noise, shape ``(P,)``. When
            set, ``rng`` is unused for the order-key tie-break (L2 / native).

    Returns:
        Adjusted minutes with the same shape as ``minutes``.

    Raises:
        ValueError: Shapes disagree or ``target`` is not positive.
    """
    drawn = np.asarray(minutes, dtype=np.int64)
    teams = np.asarray(team_keys)
    weights = np.asarray(start_weight, dtype=np.float64)
    pos = np.asarray(positions)
    if drawn.ndim != 2:
        raise ValueError(f"minutes must be 2-D, got {drawn.shape}")
    if (
        teams.shape != (drawn.shape[0],)
        or weights.shape != (drawn.shape[0],)
        or pos.shape != (drawn.shape[0],)
    ):
        raise ValueError(
            "team_keys, start_weight, and positions must match minutes rows"
        )
    if target < 1:
        raise ValueError(f"target must be positive, got {target}")
    if drawn.size == 0:
        return drawn.copy()
    out = drawn.copy()
    if tiebreak_noise is None:
        noise = np.asarray(rng.random(int(drawn.shape[0])), dtype=np.float64)
    else:
        noise = np.asarray(tiebreak_noise, dtype=np.float64)
        if noise.shape != (drawn.shape[0],):
            raise ValueError(
                f"tiebreak_noise must have shape {(drawn.shape[0],)}, got {noise.shape}"
            )
    # Stable rank: higher weight first; tiny noise breaks ties.
    order_key = np.asarray(weights + 1e-9 * noise, dtype=np.float64)
    for team in np.unique(teams):
        indexes = np.flatnonzero(teams == team)
        if indexes.size == 0:
            continue
        is_gk = np.asarray([str(pos[i]) == gk_position for i in indexes])
        gk_idx = indexes[is_gk]
        out_idx = indexes[~is_gk]
        eligible_gk = gk_idx[weights[gk_idx] > 0]
        eligible_out = out_idx[weights[out_idx] > 0]
        want = min(target, int(eligible_gk.size + eligible_out.size))
        gk_want = 1 if eligible_gk.size > 0 and want >= 1 else 0
        out_want = min(want - gk_want, int(eligible_out.size))
        _adjust_starter_count(
            out,
            gk_idx,
            weights,
            order_key,
            want=gk_want,
            starter_threshold=starter_threshold,
            start_minutes=start_minutes,
            sub_minutes=sub_minutes,
        )
        _adjust_starter_count(
            out,
            out_idx,
            weights,
            order_key,
            want=out_want,
            starter_threshold=starter_threshold,
            start_minutes=start_minutes,
            sub_minutes=sub_minutes,
        )
    return out


def _inverse_cdf_index(cdf: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    """Map uniforms through a row-wise CDF (bit-identical to ``sum(cdf <= u)``).

    ``np.searchsorted(..., side="right")`` returns the same index as counting
    how many CDF entries are ``<= u``, then clipping to the last bin.

    Args:
        cdf: Shape ``(R, C)``, non-decreasing along axis 1, last column 1.
        uniforms: Shape ``(R, N)`` draws in ``[0, 1)``.

    Returns:
        Integer indexes of shape ``(R, N)``, each in ``0 .. C-1``.
    """
    rows, cells = cdf.shape
    out = np.empty(uniforms.shape, dtype=np.int64)
    for row in range(rows):
        out[row] = np.searchsorted(cdf[row], uniforms[row], side="right")
    return np.minimum(out, cells - 1)


def _adjust_starter_count(
    minutes: np.ndarray,
    indexes: np.ndarray,
    weights: np.ndarray,
    order_key: np.ndarray,
    *,
    want: int,
    starter_threshold: int,
    start_minutes: int,
    sub_minutes: int,
) -> None:
    """Promote/demote within ``indexes`` so about ``want`` are starters per sim.

    Sorts the group once by ``order_key``, then applies demote/promote masks
    over all simulation columns (same outcomes as the former per-sim loop).
    """
    if indexes.size == 0:
        return
    ranked = indexes[np.argsort(order_key[indexes])]
    is_starter = minutes[ranked] >= starter_threshold
    n_started = is_starter.sum(axis=0)
    excess = np.maximum(n_started - want, 0)
    demote = is_starter & (np.cumsum(is_starter, axis=0) <= excess)
    rows, cols = np.nonzero(demote)
    minutes[ranked[rows], cols] = sub_minutes
    need = np.maximum(want - n_started, 0)
    eligible = (weights[ranked] > 0)[:, None] & ~is_starter
    from_high = np.cumsum(eligible[::-1], axis=0)[::-1]
    promote = eligible & (from_high <= need)
    rows, cols = np.nonzero(promote)
    minutes[ranked[rows], cols] = start_minutes
