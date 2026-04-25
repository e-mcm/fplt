"""Joint PMF wrapper for match score distributions.

Wraps the (n_fixtures, max_goals+1, max_goals+1) probability mass array
produced by DixonColesModel and provides clean accessors so the aggregator
always integrates over the full distribution rather than using point estimates.
"""

from __future__ import annotations

import numpy as np


class ScoreDistribution:
    """Joint probability mass function over (home goals, away goals) per fixture.

    pmf shape: (n_fixtures, max_goals+1, max_goals+1)
    pmf[i, h, a] = P(home scored h, away scored a) for fixture i
    """

    def __init__(self, pmf: np.ndarray, fixture_ids: list[int | str]) -> None:
        """
        Args:
            pmf: Array of shape (n_fixtures, max_goals+1, max_goals+1).
                 Each pmf[i] must sum to ≈ 1.0.
            fixture_ids: Ordered list of fixture identifiers matching axis 0.
        """
        if pmf.ndim != 3 or pmf.shape[1] != pmf.shape[2]:
            raise ValueError(f"pmf must be shape (n, G, G); got {pmf.shape}")
        self.pmf = pmf
        self.fixture_ids = fixture_ids

    # ------------------------------------------------------------------
    # Core accessors
    # ------------------------------------------------------------------

    def p_clean_sheet(self, side: str) -> np.ndarray:
        """P(opponent scored 0) for each fixture.

        Args:
            side: "home" (team defending at home) or "away".

        Returns:
            Array of shape (n_fixtures,).
        """
        if side == "home":
            # Home CS: P(away scored 0) = sum over all h of pmf[i, h, 0]
            return self.pmf[:, :, 0].sum(axis=1)
        else:
            # Away CS: P(home scored 0) = sum over all a of pmf[i, 0, a]
            return self.pmf[:, 0, :].sum(axis=1)

    def p_team_scored_k(self, side: str, k: int) -> np.ndarray:
        """P(team scored exactly k goals) for each fixture.

        Args:
            side: "home" or "away".
            k: Goal count.

        Returns:
            Array of shape (n_fixtures,). Returns zeros if k >= max_goals+1.
        """
        if side == "home":
            marginal = self.pmf.sum(axis=2)  # sum over away goals → shape (n, G)
        else:
            marginal = self.pmf.sum(axis=1)  # sum over home goals → shape (n, G)

        if k >= marginal.shape[1]:
            return np.zeros(len(self.fixture_ids))
        return marginal[:, k]

    def expected_goals(self, side: str) -> np.ndarray:
        """E[goals] for each fixture.

        For reference and logging only — the aggregator uses p_team_scored_k
        to integrate over the full distribution rather than this point estimate.

        Returns:
            Array of shape (n_fixtures,).
        """
        k = np.arange(self.pmf.shape[1], dtype=np.float64)
        if side == "home":
            return (self.pmf.sum(axis=2) * k).sum(axis=1)
        else:
            return (self.pmf.sum(axis=1) * k).sum(axis=1)

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def fixture_index(self, fixture_id: int | str) -> int:
        """Return the axis-0 index for a given fixture_id."""
        return self.fixture_ids.index(fixture_id)

    def __len__(self) -> int:
        return len(self.fixture_ids)

    def __repr__(self) -> str:
        g = self.pmf.shape[1] - 1
        return f"ScoreDistribution(n_fixtures={len(self)}, max_goals={g})"
