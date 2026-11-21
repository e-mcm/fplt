"""Dixon-Coles scorelines fitted from fixtures visible at a deadline.

v1 holds ξ, home advantage, and ρ fixed in the model spec and identifies only
attack and defence. v2 can jointly estimate ρ and/or home advantage, and
shrinks sparse or promoted-team ratings toward the league mean so fixed-mode
and early-season forecasts degrade less than a hard neutral prior.

When ``market_prior_strength`` > 0, ``predict`` shrinks per-match λ toward
odds-implied rates from PIT ``match_odds``. Missing odds leave pure DC.
"""

# scipy does not ship type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import numpy as np
import polars as pl
from scipy.optimize import minimize
from scipy.stats import poisson

from fplt.data.pit import PITView
from fplt.models.components.score.odds_lambda import blend_lambda, odds_to_lambda
from fplt.models.distributions import MAX_GOALS, ScorelineDist
from fplt.models.spec import ModelSpec, register_component

# 0..MAX_GOALS is kept. The extra bins are the tail ScorelineDist measures,
# so a rate near the attack/defence bound is not silently cut off.
_GRID = MAX_GOALS + 9
_NEUTRAL = 1.0
_RHO_BOUNDS = (-0.5, 0.5)
_HOME_BOUNDS = (0.8, 2.0)


def _dc_correction(
    home_goals: int, away_goals: int, lambda_h: float, lambda_a: float, rho: float
) -> float:
    """Low-score factor τ for one scoreline."""
    if home_goals == 0 and away_goals == 0:
        return 1 - lambda_h * lambda_a * rho
    if home_goals == 0 and away_goals == 1:
        return 1 + lambda_h * rho
    if home_goals == 1 and away_goals == 0:
        return 1 + lambda_a * rho
    if home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


def _neg_log_likelihood(
    params: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
    home_advantage: float,
    rho: float,
    fit_home: bool,
    fit_rho: bool,
) -> float:
    """Weighted negative log-likelihood; ρ / HA may sit in ``params``."""
    attack = params[:n_teams]
    defence = params[n_teams : 2 * n_teams]
    offset = 2 * n_teams
    if fit_home:
        home_advantage = float(params[offset])
        offset += 1
    if fit_rho:
        rho = float(params[offset])
    lambda_h = attack[home_idx] * defence[away_idx] * home_advantage
    lambda_a = attack[away_idx] * defence[home_idx]
    lambda_h = np.clip(lambda_h, 1e-6, None)
    lambda_a = np.clip(lambda_a, 1e-6, None)
    home_lp = poisson.logpmf(home_goals, lambda_h)
    away_lp = poisson.logpmf(away_goals, lambda_a)
    log_p = home_lp + away_lp
    tau = np.ones(len(home_goals))
    both_zero = (home_goals == 0) & (away_goals == 0)
    away_one = (home_goals == 0) & (away_goals == 1)
    home_one = (home_goals == 1) & (away_goals == 0)
    both_one = (home_goals == 1) & (away_goals == 1)
    tau[both_zero] = 1 - lambda_h[both_zero] * lambda_a[both_zero] * rho
    tau[away_one] = 1 + lambda_h[away_one] * rho
    tau[home_one] = 1 + lambda_a[home_one] * rho
    tau[both_one] = 1 - rho
    return -float(np.sum(weights * (log_p + np.log(np.clip(tau, 1e-10, None)))))


class MatchScoreModel:
    """Fit attack and defence, then return a ``ScorelineDist``.

    Spec params:

    - ``xi``, ``home_advantage``, ``rho`` — match NLL time decay
      ``exp(-xi × days_before_latest_kickoff)`` and (optional) starts for home
      advantage and the Dixon-Coles low-score factor. ``TrainWindow.time_decay``
      is ignored here; keep DC on ``xi`` only (do not double-decay).
    - ``fit_rho`` / ``fit_home_advantage`` — when true, estimate that scalar
      jointly with attack and defence (v2).
    - ``new_team_prior_strength`` — pseudo-matches pulling every team toward
      the league mean; effective strength is larger for promoted sides.
    - ``promoted_prior_strength`` — extra pseudo-matches for teams absent from
      earlier seasons visible at the deadline.
    - ``market_prior_strength`` — pseudo-count shrink of per-match λ toward
      PIT ``match_odds`` (0 = off / vanilla). Missing odds → pure DC.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._xi = float(spec.params.get("xi", 0.0))
        self._home = float(spec.params.get("home_advantage", 1.0))
        self._rho = float(spec.params.get("rho", 0.0))
        self._fit_rho = bool(spec.params.get("fit_rho", False))
        self._fit_home = bool(spec.params.get("fit_home_advantage", False))
        self._new_prior = float(spec.params.get("new_team_prior_strength", 0.0))
        self._promoted_prior = float(spec.params.get("promoted_prior_strength", 0.0))
        self._market_prior = float(spec.params.get("market_prior_strength", 0.0))
        if self._market_prior < 0:
            raise ValueError(
                f"market_prior_strength must be >= 0, got {self._market_prior}"
            )
        self._teams: list[int] = []
        self._attack: dict[int, float] = {}
        self._defence: dict[int, float] = {}

    def fit(self, view: PITView) -> None:
        """Estimate ratings from finished fixtures visible at ``view.as_of``."""
        played = _finished(view)
        self._teams = []
        self._attack = {}
        self._defence = {}
        if played.is_empty():
            return
        teams = sorted(
            set(played["team_h_key"].to_list()) | set(played["team_a_key"].to_list())
        )
        lookup = pl.DataFrame({"team_key": teams, "idx": list(range(len(teams)))})
        indexed = played.join(
            lookup, left_on="team_h_key", right_on="team_key", how="left"
        ).join(lookup, left_on="team_a_key", right_on="team_key", how="left")
        kick = indexed["kickoff_time"].to_numpy()
        days = (kick.max() - kick) / np.timedelta64(1, "D")
        weights = np.exp(-self._xi * days.astype(np.float64))
        n_teams = len(teams)
        start = np.ones(2 * n_teams)
        bounds: list[tuple[float, float]] = [(0.2, 3.0)] * (2 * n_teams)
        if self._fit_home:
            start = np.append(start, self._home)
            bounds.append(_HOME_BOUNDS)
        if self._fit_rho:
            start = np.append(start, self._rho)
            bounds.append(_RHO_BOUNDS)
        result = minimize(
            _neg_log_likelihood,
            start,
            args=(
                indexed["team_h_score"].to_numpy().astype(np.float64),
                indexed["team_a_score"].to_numpy().astype(np.float64),
                indexed["idx"].to_numpy(),
                indexed["idx_right"].to_numpy(),
                weights,
                n_teams,
                self._home,
                self._rho,
                self._fit_home,
                self._fit_rho,
            ),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 300, "ftol": 1e-8},
        )
        fitted = np.asarray(result.x, dtype=np.float64)
        attack = fitted[:n_teams]
        defence = fitted[n_teams : 2 * n_teams]
        offset = 2 * n_teams
        if self._fit_home:
            self._home = float(fitted[offset])
            offset += 1
        if self._fit_rho:
            self._rho = float(fitted[offset])
        scale = float(attack.mean())
        if scale <= 0:
            scale = 1.0
        attack = attack / scale
        defence = defence * scale
        counts = _match_counts(played, teams)
        promoted = _promoted_teams(view, teams)
        attack, defence = _shrink_ratings(
            attack,
            defence,
            counts,
            teams,
            promoted,
            self._new_prior,
            self._promoted_prior,
        )
        self._teams = teams
        self._attack = dict(zip(teams, attack.tolist(), strict=True))
        self._defence = dict(zip(teams, defence.tolist(), strict=True))

    def predict(self, view: PITView, fixtures: pl.DataFrame) -> ScorelineDist:
        """Joint scorelines for ``fixtures``.

        Args:
            view: Deadline the ratings belong to. Ratings come from ``fit``.
                When ``market_prior_strength`` > 0, also supplies PIT
                ``match_odds`` for the per-match λ blend.
            fixtures: Columns ``fixture_id``, ``team_h_key``, and ``team_a_key``.
                ``season`` helps the odds join when ``fixture_id`` is missing.

        Returns:
            A truncated ``ScorelineDist``.
        """
        if fixtures.is_empty():
            empty = np.zeros((0, MAX_GOALS + 1, MAX_GOALS + 1))
            return ScorelineDist(np.array([], dtype=np.int64), empty)
        ids = fixtures["fixture_id"].to_numpy()
        lambda_h, lambda_a = self._rates(
            fixtures["team_h_key"].to_list(), fixtures["team_a_key"].to_list()
        )
        if self._market_prior > 0:
            market_h, market_a = _market_rates(view, fixtures)
            blended_h = np.empty_like(lambda_h)
            blended_a = np.empty_like(lambda_a)
            for index, (dc_h, dc_a, mkt_h, mkt_a) in enumerate(
                zip(lambda_h, lambda_a, market_h, market_a, strict=True)
            ):
                market_home = None if np.isnan(mkt_h) else float(mkt_h)
                market_away = None if np.isnan(mkt_a) else float(mkt_a)
                # Blend only when both sides have market λ; else keep pure DC.
                if market_home is None or market_away is None:
                    blended_h[index] = dc_h
                    blended_a[index] = dc_a
                else:
                    blended_h[index] = blend_lambda(
                        float(dc_h), market_home, self._market_prior
                    )
                    blended_a[index] = blend_lambda(
                        float(dc_a), market_away, self._market_prior
                    )
            lambda_h, lambda_a = blended_h, blended_a
        return ScorelineDist(ids, _joint(lambda_h, lambda_a, self._rho))

    def _rates(self, home: list[int], away: list[int]) -> tuple[np.ndarray, np.ndarray]:
        attack_h = np.array([self._attack.get(team, _NEUTRAL) for team in home])
        attack_a = np.array([self._attack.get(team, _NEUTRAL) for team in away])
        defence_h = np.array([self._defence.get(team, _NEUTRAL) for team in home])
        defence_a = np.array([self._defence.get(team, _NEUTRAL) for team in away])
        return attack_h * defence_a * self._home, attack_a * defence_h


def _finished(view: PITView) -> pl.DataFrame:
    return (
        view.latest("fixtures", by=["season", "fixture_id"])
        .filter(pl.col("finished") & pl.col("team_h_score").is_not_null())
        .collect()
    )


def _match_counts(played: pl.DataFrame, teams: list[int]) -> np.ndarray:
    """Finished appearances per team, aligned with ``teams``."""
    order = {team: index for index, team in enumerate(teams)}
    counts = np.zeros(len(teams), dtype=np.float64)
    for team in played["team_h_key"].to_list():
        counts[order[int(team)]] += 1.0
    for team in played["team_a_key"].to_list():
        counts[order[int(team)]] += 1.0
    return counts


def _promoted_teams(view: PITView, teams: list[int]) -> set[int]:
    """Team keys present in the as-of season but not in any earlier season."""
    seasons = _visible_seasons(view)
    current = view.as_of.season
    earlier = [season for season in seasons if season != current]
    if not earlier:
        return set()
    prior = (
        view.latest("fixtures", by=["season", "fixture_id"])
        .filter(pl.col("season").is_in(earlier))
        .select("team_h_key", "team_a_key")
        .collect()
    )
    if prior.is_empty():
        return set(teams)
    seen = set(prior["team_h_key"].to_list()) | set(prior["team_a_key"].to_list())
    return {team for team in teams if team not in seen}


def _visible_seasons(view: PITView) -> list[str]:
    frame = (
        view.latest("fixtures", by=["season", "fixture_id"]).select("season").collect()
    )
    if frame.is_empty():
        return []
    return sorted(set(frame["season"].to_list()))


def _shrink_ratings(
    attack: np.ndarray,
    defence: np.ndarray,
    counts: np.ndarray,
    teams: list[int],
    promoted: set[int],
    new_prior: float,
    promoted_prior: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Pull sparse / promoted ratings toward the league mean (1 after rescale)."""
    if new_prior <= 0 and promoted_prior <= 0:
        return attack, defence
    strengths = np.full(len(teams), max(new_prior, 0.0), dtype=np.float64)
    if promoted_prior > 0:
        for index, team in enumerate(teams):
            if team in promoted:
                strengths[index] = max(strengths[index], promoted_prior)
    denom = counts + strengths
    shrunk_attack = np.where(
        denom > 0, (counts * attack + strengths * _NEUTRAL) / denom, attack
    )
    shrunk_defence = np.where(
        denom > 0, (counts * defence + strengths * _NEUTRAL) / denom, defence
    )
    return shrunk_attack, shrunk_defence


def _joint(lambda_h: np.ndarray, lambda_a: np.ndarray, rho: float) -> np.ndarray:
    goals = np.arange(_GRID)
    home = poisson.pmf(goals[None, :], lambda_h[:, None])
    away = poisson.pmf(goals[None, :], lambda_a[:, None])
    joint = home[:, :, None] * away[:, None, :]
    pairs = ((0, 0), (0, 1), (1, 0), (1, 1))
    for home_goals, away_goals in pairs:
        factor = [
            _dc_correction(home_goals, away_goals, float(lh), float(la), rho)
            for lh, la in zip(lambda_h, lambda_a, strict=True)
        ]
        joint[:, home_goals, away_goals] *= np.asarray(factor)
    return np.clip(joint, 0, None)


def _market_rates(
    view: PITView, fixtures: pl.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    """Odds-implied λ aligned with ``fixtures``; NaN where odds are missing."""
    n = fixtures.height
    missing = np.full(n, np.nan, dtype=np.float64)
    odds = view.latest(
        "match_odds",
        by=["season", "team_h_key", "team_a_key", "kickoff_time"],
    ).collect()
    if odds.is_empty():
        return missing.copy(), missing.copy()
    aligned = _align_odds(fixtures, odds)
    lambda_h = missing.copy()
    lambda_a = missing.copy()
    for index, row in enumerate(aligned.iter_rows(named=True)):
        home_odds = row.get("home_odds")
        draw_odds = row.get("draw_odds")
        away_odds = row.get("away_odds")
        if home_odds is None or draw_odds is None or away_odds is None:
            continue
        over = row.get("over_25_odds")
        under = row.get("under_25_odds")
        try:
            lambda_h[index], lambda_a[index] = odds_to_lambda(
                float(home_odds),
                float(draw_odds),
                float(away_odds),
                None if over is None else float(over),
                None if under is None else float(under),
            )
        except ValueError:
            continue
    return lambda_h, lambda_a


def _align_odds(fixtures: pl.DataFrame, odds: pl.DataFrame) -> pl.DataFrame:
    """Left-join odds onto fixtures; prefer ``fixture_id``, else team keys.

    FPL ``fixture_id`` values restart each season, so the id join is always
    scoped by ``season`` when that column is present on both frames.
    """
    indexed = fixtures.with_row_index("_row")
    odds_cols = [
        "home_odds",
        "draw_odds",
        "away_odds",
        "over_25_odds",
        "under_25_odds",
    ]
    empty = indexed.select("_row").with_columns(
        [pl.lit(None).cast(pl.Float64).alias(name) for name in odds_cols]
    )
    via_id = empty
    if "fixture_id" in fixtures.columns:
        id_keys = ["fixture_id"]
        if "season" in fixtures.columns and "season" in odds.columns:
            id_keys = ["season", "fixture_id"]
        with_id = (
            odds.filter(pl.col("fixture_id").is_not_null())
            .sort("available_at")
            .unique(subset=id_keys, keep="last")
        )
        if not with_id.is_empty():
            via_id = (
                indexed.select(["_row", *id_keys])
                .join(
                    with_id.select([*id_keys, *odds_cols]),
                    on=id_keys,
                    how="left",
                )
                .select(["_row", *odds_cols])
            )
    keys = ["team_h_key", "team_a_key"]
    if "season" in fixtures.columns and "season" in odds.columns:
        keys = ["season", *keys]
    team_odds = odds.sort("available_at").unique(subset=keys, keep="last")
    via_teams = (
        indexed.select(["_row", *keys])
        .join(team_odds.select([*keys, *odds_cols]), on=keys, how="left")
        .select(["_row", *odds_cols])
    )
    merged = (
        empty.select("_row")
        .join(via_id, on="_row", how="left")
        .join(via_teams, on="_row", how="left", suffix="_team")
        .sort("_row")
    )
    return merged.select(
        [
            pl.coalesce([pl.col(name), pl.col(f"{name}_team")]).alias(name)
            for name in odds_cols
        ]
    )


register_component("dixon_coles")(MatchScoreModel)
register_component("poisson_score")(MatchScoreModel)
