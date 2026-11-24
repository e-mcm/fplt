"""Scoreline baseline from bookmaker match odds (independent Poisson).

Reads staged ``match_odds`` through ``PITView``. Deadline-true rows
(``available_at ≤ cutoff``) are preferred. When ``use_closing_proxy`` is set
(default for ``odds_poisson_v1``), closing lines stamped at kickoff are also
admitted for the labelled closing-proxy probe — never claim deadline PIT.
"""

# scipy does not ship type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import logging

import numpy as np
import polars as pl
from scipy.stats import poisson

from fplt.data.pit import PITView
from fplt.models.component_params import OddsPoissonParams, parse_component_params
from fplt.models.components.score.odds_lambda import odds_to_lambda
from fplt.models.distributions import MAX_GOALS, ScorelineDist
from fplt.models.spec import ModelSpec, register_component

_LOG = logging.getLogger(__name__)
_ODDS_KEYS = ["season", "team_h_key", "team_a_key", "kickoff_time"]
# Match Dixon–Coles: keep 0..MAX_GOALS and a few extra bins for the tail.
_GRID = MAX_GOALS + 9


class OddsPoissonScore:
    """Map PIT ``match_odds`` into an independent-Poisson ``ScorelineDist``.

    Spec params:

    - ``use_closing_proxy`` (bool, default true) — also use
      ``odds_kind=closing_proxy`` rows whose ``available_at`` is kickoff (after
      the FPL deadline). Labelled probe only; not deadline-safe PIT.
    - ``fallback_lambda_home`` / ``fallback_lambda_away`` — rates when a
      fixture has no usable odds.
    - ``use_over_under`` (bool, default true) — fold O/U 2.5 into the λ fit
      when both sides are present.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        params = parse_component_params(OddsPoissonParams, spec.params)
        self._use_closing_proxy = params.use_closing_proxy
        self._use_ou = params.use_over_under
        self._fallback_home = params.fallback_lambda_home
        self._fallback_away = params.fallback_lambda_away

    def fit(self, view: PITView) -> None:
        """No-op: market odds are read at predict time."""
        del view

    def predict(self, view: PITView, fixtures: pl.DataFrame) -> ScorelineDist:
        """Independent-Poisson scorelines from market odds for ``fixtures``.

        Args:
            view: Deadline view (and season window) used to read ``match_odds``.
            fixtures: Columns ``fixture_id``, ``team_h_key``, ``team_a_key``;
                ``season`` when joining on teams without ``fixture_id``.

        Returns:
            A truncated ``ScorelineDist`` (ρ = 0).
        """
        if fixtures.is_empty():
            empty = np.zeros((0, MAX_GOALS + 1, MAX_GOALS + 1))
            return ScorelineDist(np.array([], dtype=np.int64), empty)
        odds = odds_frame(view, allow_closing_proxy=self._use_closing_proxy)
        joined = join_odds(fixtures, odds)
        lambda_h = np.empty(joined.height, dtype=np.float64)
        lambda_a = np.empty(joined.height, dtype=np.float64)
        for index, row in enumerate(joined.iter_rows(named=True)):
            home_odds = row.get("home_odds")
            draw_odds = row.get("draw_odds")
            away_odds = row.get("away_odds")
            if (
                home_odds is None
                or draw_odds is None
                or away_odds is None
                or float(home_odds) <= 1.0
                or float(draw_odds) <= 1.0
                or float(away_odds) <= 1.0
            ):
                lambda_h[index] = self._fallback_home
                lambda_a[index] = self._fallback_away
                continue
            over = row.get("over_25_odds") if self._use_ou else None
            under = row.get("under_25_odds") if self._use_ou else None
            if over is not None and float(over) <= 1.0:
                over = None
            if under is not None and float(under) <= 1.0:
                under = None
            if over is None or under is None:
                over, under = None, None
            fitted_home, fitted_away = odds_to_lambda(
                float(home_odds),
                float(draw_odds),
                float(away_odds),
                over_25_odds=None if over is None else float(over),
                under_25_odds=None if under is None else float(under),
            )
            lambda_h[index] = fitted_home
            lambda_a[index] = fitted_away
        ids = joined["fixture_id"].to_numpy()
        return ScorelineDist(ids, _independent_poisson_pmf(lambda_h, lambda_a))


def _independent_poisson_pmf(
    lambda_home: np.ndarray, lambda_away: np.ndarray
) -> np.ndarray:
    """Independent-Poisson joint grid past ``MAX_GOALS`` for ``ScorelineDist``."""
    goals = np.arange(_GRID)
    home = poisson.pmf(goals[None, :], lambda_home[:, None])
    away = poisson.pmf(goals[None, :], lambda_away[:, None])
    return np.clip(home[:, :, None] * away[:, None, :], 0, None)


def odds_frame(view: PITView, *, allow_closing_proxy: bool) -> pl.DataFrame:
    """PIT odds, optionally filled with closing-proxy rows for missing keys."""
    pit = view.latest("match_odds", by=_ODDS_KEYS).collect()
    if not allow_closing_proxy:
        return pit
    proxy = view.closing_proxy_match_odds().collect()
    if proxy.is_empty():
        return pit
    proxy = proxy.sort("available_at").unique(subset=_ODDS_KEYS, keep="last")
    if pit.is_empty():
        return proxy
    gap = proxy.join(pit.select(_ODDS_KEYS).unique(), on=_ODDS_KEYS, how="anti")
    if gap.is_empty():
        return pit
    _LOG.debug(
        "closing_proxy filling %s match_odds rows after cutoff=%s",
        gap.height,
        view.cutoff,
    )
    return pl.concat([pit, gap], how="diagonal_relaxed")


def join_odds(fixtures: pl.DataFrame, odds: pl.DataFrame) -> pl.DataFrame:
    """Attach odds onto fixtures by ``fixture_id``, else season+teams+date.

    Team-only joins are rejected: the same clubs meet more than once a season,
    so a bare ``(season, team_h_key, team_a_key)`` key would leak another
    fixture's line onto the target.
    """
    base = fixtures
    if "season" not in base.columns:
        base = base.with_columns(pl.lit(None).cast(pl.Utf8).alias("season"))
    odds_cols = [
        "home_odds",
        "draw_odds",
        "away_odds",
        "over_25_odds",
        "under_25_odds",
    ]
    empty_odds = [pl.lit(None).cast(pl.Float64).alias(name) for name in odds_cols]
    if odds.is_empty():
        return base.with_columns(empty_odds)
    id_keys = ["fixture_id"]
    if "season" in base.columns and "season" in odds.columns:
        id_keys = ["season", "fixture_id"]
    by_id = (
        odds.filter(pl.col("fixture_id").is_not_null())
        .sort("available_at")
        .unique(subset=id_keys, keep="last")
        .select(*id_keys, *odds_cols)
    )
    joined = base.join(by_id, on=id_keys, how="left")
    missing = joined.filter(pl.col("home_odds").is_null())
    if missing.is_empty() or "kickoff_time" not in base.columns:
        return joined
    by_date = (
        odds.with_columns(pl.col("kickoff_time").dt.date().alias("match_date"))
        .sort("available_at")
        .unique(
            subset=["season", "team_h_key", "team_a_key", "match_date"],
            keep="last",
        )
        .select(
            "season",
            "team_h_key",
            "team_a_key",
            "match_date",
            *odds_cols,
        )
    )
    filled = (
        missing.drop(odds_cols)
        .with_columns(pl.col("kickoff_time").dt.date().alias("match_date"))
        .join(
            by_date,
            on=["season", "team_h_key", "team_a_key", "match_date"],
            how="left",
        )
        .drop("match_date")
    )
    kept = joined.filter(pl.col("home_odds").is_not_null())
    if kept.is_empty():
        return filled.select(joined.columns)
    return pl.concat([kept, filled.select(joined.columns)], how="vertical_relaxed")


register_component("odds_poisson")(OddsPoissonScore)

# Private aliases kept for older call sites / tests.
_odds_frame = odds_frame
_join_odds = join_odds

__all__ = ["OddsPoissonScore", "join_odds", "odds_frame"]
