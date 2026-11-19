"""Scoreline and minutes distributions used by the layered match model.

The scoreline grid stops at 10 goals a side. A heavier tail would move
expected goals after renormalisation, so construction refuses a truncation
that drops 1e-6 or more of the probability. Minutes sit on a fixed set of
bins so a draw is a minute count the scoring rules can apply directly.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import polars as pl

MAX_GOALS = 10
MINUTE_BINS = np.array([0, 15, 30, 45, 60, 75, 90], dtype=np.int64)
_MASS_LOST_LIMIT = 1e-6
_Side = Literal["home", "away"]


class ScorelineDist:
    """Joint distribution over home and away goals for each fixture.

    ``pmf[i, h, a]`` is the probability that fixture ``i`` finishes ``h``-``a``.
    Both axes run from 0 through ``MAX_GOALS``.
    """

    def __init__(self, fixture_ids: np.ndarray, pmf: np.ndarray) -> None:
        """Truncate ``pmf`` at 10 goals and renormalise.

        Args:
            fixture_ids: One id per fixture, shape ``(F,)``.
            pmf: Non-negative joint masses, shape ``(F, G, G)`` with ``G`` at
                least 11. Mass outside 0..10 must be under 1e-6 of each
                fixture's total.

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
            Shape ``(F, 11)``. Column ``k`` is ``P(goals = k)``.
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
        index = np.sum(cdf[:, None, :] <= draws[..., None], axis=-1)
        index = np.minimum(index, goals * goals - 1)
        home = index // goals
        away = index % goals
        return np.stack([home, away], axis=-1).astype(np.int64)


class MinutesDist:
    """Binned minutes for each player-fixture.

    ``pmf[i, b]`` is the probability that row ``i`` plays ``bins[b]`` minutes.
    Draws are independent across rows: nothing stops two teammates both
    playing 90. A later version can enforce eleven starters.
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
        players, width = self.pmf.shape
        if n == 0 or players == 0:
            return np.zeros((players, n), dtype=np.int64)
        cdf = np.cumsum(self.pmf, axis=1)
        cdf[:, -1] = 1.0
        draws = rng.random((players, n))
        index = np.sum(cdf[:, None, :] <= draws[..., None], axis=-1)
        index = np.minimum(index, width - 1)
        return np.asarray(self.bins, dtype=np.int64)[index]
