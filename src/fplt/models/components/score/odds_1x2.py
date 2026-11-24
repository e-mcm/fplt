"""Raw de-vig 1X2 scoreline baseline from bookmaker match odds.

Unlike ``odds_poisson``, this does **not** fit λ. Closing (or PIT) 1X2 prices
are de-vigged to ``(p_home, p_draw, p_away)``, then placed on a neutral
independent-Poisson shape whose H/D/A regions are rescaled to those probs so
RPS / log-loss see the market 1X2 exactly.
"""

# scipy does not ship type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from typing import Any

import numpy as np
import polars as pl
from scipy.stats import poisson

from fplt.data.pit import PITView
from fplt.models.component_params import Odds1X2Params, parse_component_params
from fplt.models.components.score.odds_lambda import (
    de_vig,
    rescale_joint_to_1x2,
)
from fplt.models.components.score.odds_poisson import join_odds, odds_frame
from fplt.models.distributions import MAX_GOALS, ScorelineDist
from fplt.models.spec import ModelSpec, register_component

_GRID = MAX_GOALS + 9
_SHAPE_HOME = 1.35
_SHAPE_AWAY = 1.1


def _neutral_shape() -> np.ndarray:
    """Independent-Poisson joint used only as within-outcome shape."""
    goals = np.arange(_GRID)
    home = poisson.pmf(goals, _SHAPE_HOME)
    away = poisson.pmf(goals, _SHAPE_AWAY)
    return np.clip(np.outer(home, away), 0, None)


_SHAPE = _neutral_shape()


class Odds1X2Score:
    """De-vig closing / PIT 1X2 into a ``ScorelineDist`` with exact margins.

    Spec params:

    - ``use_closing_proxy`` (bool, default true) — admit kickoff-stamped
      ``closing_proxy`` rows (labelled probe; not deadline PIT).
    - ``fallback_p_home`` / ``fallback_p_draw`` / ``fallback_p_away`` — 1X2
      used when a fixture has no usable odds (default flat 1/3).
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        params = parse_component_params(Odds1X2Params, spec.params)
        self._use_closing_proxy = params.use_closing_proxy
        fallback = (
            params.fallback_p_home,
            params.fallback_p_draw,
            params.fallback_p_away,
        )
        total = sum(fallback)
        if total <= 0:
            raise ValueError("fallback 1X2 probs must sum to a positive value")
        self._fallback: tuple[float, float, float] = (
            fallback[0] / total,
            fallback[1] / total,
            fallback[2] / total,
        )

    def fit(self, view: PITView) -> None:
        """No-op: market odds are read at predict time."""
        del view

    def predict(self, view: PITView, fixtures: pl.DataFrame) -> ScorelineDist:
        """Scorelines whose 1X2 margins match de-vigged bookmaker prices.

        Args:
            view: Deadline view used to read ``match_odds``.
            fixtures: Columns ``fixture_id``, ``team_h_key``, ``team_a_key``;
                ``season`` / ``kickoff_time`` help the odds join.

        Returns:
            A truncated ``ScorelineDist``.
        """
        if fixtures.is_empty():
            empty = np.zeros((0, MAX_GOALS + 1, MAX_GOALS + 1))
            return ScorelineDist(np.array([], dtype=np.int64), empty)
        odds = odds_frame(view, allow_closing_proxy=self._use_closing_proxy)
        joined = join_odds(fixtures, odds)
        joints = np.zeros((joined.height, _GRID, _GRID), dtype=np.float64)
        for index, row in enumerate(joined.iter_rows(named=True)):
            probs = _row_1x2(row, self._fallback)
            joints[index] = rescale_joint_to_1x2(_SHAPE, probs[0], probs[1], probs[2])
        return ScorelineDist(joined["fixture_id"].to_numpy(), joints)


def _row_1x2(
    row: dict[str, Any], fallback: tuple[float, float, float]
) -> tuple[float, float, float]:
    home_odds = row.get("home_odds")
    draw_odds = row.get("draw_odds")
    away_odds = row.get("away_odds")
    if home_odds is None or draw_odds is None or away_odds is None:
        return fallback
    try:
        probs = de_vig(
            np.array(
                [float(home_odds), float(draw_odds), float(away_odds)],
                dtype=np.float64,
            )
        )
    except ValueError:
        return fallback
    return float(probs[0]), float(probs[1]), float(probs[2])


register_component("odds_1x2")(Odds1X2Score)

__all__ = ["Odds1X2Score"]
