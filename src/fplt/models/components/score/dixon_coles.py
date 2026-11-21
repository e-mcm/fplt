"""Dixon-Coles scorelines fitted from fixtures visible at a deadline.

ξ, home advantage, and ρ are spec parameters. Attack and defence are what the
fit identifies. ρ = 0 is independent Poisson, the baseline on the component ledger.
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
from fplt.models.distributions import MAX_GOALS, ScorelineDist
from fplt.models.spec import ModelSpec, register_component

# 0..MAX_GOALS is kept. The extra bins are the tail ScorelineDist measures,
# so a rate near the attack/defence bound is not silently cut off.
_GRID = MAX_GOALS + 9
_NEUTRAL = 1.0


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
) -> float:
    """Weighted negative log-likelihood with home advantage and ρ held fixed."""
    attack = params[:n_teams]
    defence = params[n_teams:]
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
    """Fit attack and defence, then return a ``ScorelineDist``."""

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._xi = float(spec.params.get("xi", 0.0))
        self._home = float(spec.params.get("home_advantage", 1.0))
        self._rho = float(spec.params.get("rho", 0.0))
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
        bounds = [(0.2, 3.0)] * (2 * n_teams)
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
            ),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 200, "ftol": 1e-8},
        )
        fitted = np.asarray(result.x, dtype=np.float64)
        attack = fitted[:n_teams]
        defence = fitted[n_teams:]
        scale = float(attack.mean())
        attack = attack / scale
        defence = defence * scale
        self._teams = teams
        self._attack = dict(zip(teams, attack.tolist(), strict=True))
        self._defence = dict(zip(teams, defence.tolist(), strict=True))

    def predict(self, view: PITView, fixtures: pl.DataFrame) -> ScorelineDist:
        """Joint scorelines for ``fixtures``.

        Args:
            view: Deadline the ratings belong to. Ratings come from ``fit``.
            fixtures: Columns ``fixture_id``, ``team_h_key``, and ``team_a_key``.

        Returns:
            A truncated ``ScorelineDist``.
        """
        del view
        if fixtures.is_empty():
            empty = np.zeros((0, MAX_GOALS + 1, MAX_GOALS + 1))
            return ScorelineDist(np.array([], dtype=np.int64), empty)
        ids = fixtures["fixture_id"].to_numpy()
        lambda_h, lambda_a = self._rates(
            fixtures["team_h_key"].to_list(), fixtures["team_a_key"].to_list()
        )
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


register_component("dixon_coles")(MatchScoreModel)
register_component("poisson_score")(MatchScoreModel)
