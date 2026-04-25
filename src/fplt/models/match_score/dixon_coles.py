"""Dixon-Coles match score model.

Fits per-team attack (α) and defence (β) parameters plus a global home
advantage (γ) and low-score correlation correction (ρ) via maximum likelihood.

Reference: Dixon & Coles (1997) "Modelling association football scores and
inefficiencies in the football betting market", Applied Statistics 46(2).

Time decay: each match is weighted by exp(-ξ * days_since_most_recent_match),
so recent results influence the parameters more heavily.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from scipy.stats import poisson

from fplt.models.base import DistributionalModel
from fplt.models.score_distribution import ScoreDistribution


def _dc_correction(
    home_goals: int,
    away_goals: int,
    lambda_h: float,
    lambda_a: float,
    rho: float,
) -> float:
    """Dixon-Coles low-score correlation factor τ(h, a, λ_h, λ_a, ρ)."""
    if home_goals == 0 and away_goals == 0:
        return 1 - lambda_h * lambda_a * rho
    elif home_goals == 0 and away_goals == 1:
        return 1 + lambda_h * rho
    elif home_goals == 1 and away_goals == 0:
        return 1 + lambda_a * rho
    elif home_goals == 1 and away_goals == 1:
        return 1 - rho
    return 1.0


class DixonColesModel(DistributionalModel):
    """Dixon-Coles Poisson model for match score distributions.

    Learns per-team attack/defence ratings from historical fixture data.
    Produces a joint PMF over (home goals, away goals) for any pair of teams.

    Example:
        >>> from fplt.features.team_strength import build_fixture_records
        >>> from fplt.models.match_score.dixon_coles import DixonColesModel
        >>> fixtures = build_fixture_records(seasons=["2022-23", "2023-24"])
        >>> model = DixonColesModel()
        >>> model.fit(fixtures)
        >>> # Predict for a batch of upcoming fixtures
        >>> upcoming = pd.DataFrame({"team_h": [...], "team_a": [...], "kickoff_time": [...]})
        >>> dist = model.predict_distribution(upcoming)
        >>> dist.p_clean_sheet("home")   # shape (n_fixtures,)
    """

    def __init__(
        self,
        time_decay: float = 0.0065,
        max_goals: int = 10,
    ) -> None:
        """
        Args:
            time_decay: Exponential decay rate ξ. A value of 0.0065 gives
                matches 365 days ago roughly 0.09× the weight of today's match.
            max_goals: Truncation point for the PMF grid.
        """
        self.time_decay = time_decay
        self.max_goals = max_goals

        # Populated after fit()
        self._attack: dict[str | int, float] = {}
        self._defence: dict[str | int, float] = {}
        self._home_adv: float = 0.0
        self._rho: float = 0.0
        self._log_loss: float = float("nan")
        self._teams: list[str | int] = []

    # ------------------------------------------------------------------
    # FPLModel interface
    # ------------------------------------------------------------------

    @property
    def params(self) -> dict[str, object]:
        return {"time_decay": self.time_decay, "max_goals": self.max_goals}

    @property
    def metrics(self) -> dict[str, float]:
        return {"log_loss": self._log_loss}

    # ------------------------------------------------------------------
    # Fit
    # ------------------------------------------------------------------

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.DataFrame | pd.Series | None = None,
        **kwargs: object,
    ) -> DixonColesModel:
        """Fit attack/defence parameters from historical fixture scores.

        Args:
            X: DataFrame with columns team_h, team_a, team_h_score,
               team_a_score, kickoff_time (tz-aware datetime).
            y: Unused — targets are embedded in X as team_h/a_score columns.

        Returns:
            self
        """
        required = {"team_h", "team_a", "team_h_score", "team_a_score", "kickoff_time"}
        missing = required - set(X.columns)
        if missing:
            raise ValueError(f"X missing columns: {missing}")

        X = X.dropna(subset=["team_h_score", "team_a_score"]).copy()
        X["kickoff_time"] = pd.to_datetime(X["kickoff_time"], utc=True)

        # Time decay weights
        t_max = X["kickoff_time"].max()
        days_ago = (t_max - X["kickoff_time"]).dt.total_seconds() / 86400
        weights = np.exp(-self.time_decay * days_ago.to_numpy())

        self._teams = sorted(set(X["team_h"].tolist()) | set(X["team_a"].tolist()))
        n = len(self._teams)
        team_idx = {t: i for i, t in enumerate(self._teams)}

        home_idx = X["team_h"].map(team_idx).to_numpy()
        away_idx = X["team_a"].map(team_idx).to_numpy()
        home_goals = X["team_h_score"].to_numpy()
        away_goals = X["team_a_score"].to_numpy()

        # Parameters: attack[0..n-1], defence[0..n-1], home_adv, rho
        # Constraint: mean(attack) = 1 (identifiability) enforced via
        # fixing attack[0] = 1 and optimising n-1 free attack params.
        # Here we use the simpler approach of optimising all and letting
        # scipy handle it — good enough for our purposes.
        x0 = np.concatenate(
            [
                np.ones(n),  # attack (initialised to 1)
                np.ones(n),  # defence (initialised to 1)
                np.array([0.3]),  # home_adv
                np.array([-0.1]),  # rho
            ]
        )

        bounds = (
            [(0.01, 5.0)] * n  # attack
            + [(0.01, 5.0)] * n  # defence
            + [(0.5, 2.0)]  # home_adv
            + [(-0.99, 0.0)]  # rho (must keep τ > 0)
        )

        result = minimize(
            _neg_log_likelihood,
            x0,
            args=(home_goals, away_goals, home_idx, away_idx, weights, n),
            method="L-BFGS-B",
            bounds=bounds,
            options={"maxiter": 1000, "ftol": 1e-9},
        )

        attack = result.x[:n]
        defence = result.x[n : 2 * n]
        home_adv = float(result.x[2 * n])
        rho = float(result.x[2 * n + 1])

        self._attack = dict(zip(self._teams, attack.tolist()))
        self._defence = dict(zip(self._teams, defence.tolist()))
        self._home_adv = home_adv
        self._rho = rho
        self._log_loss = float(result.fun / weights.sum())

        return self

    # ------------------------------------------------------------------
    # Predict
    # ------------------------------------------------------------------

    def predict_distribution(self, X: pd.DataFrame) -> ScoreDistribution:
        """Compute joint PMF for each fixture row in X.

        Args:
            X: DataFrame with columns team_h, team_a, and optionally
               fixture_id (used as ScoreDistribution identifiers).

        Returns:
            ScoreDistribution with pmf shape (n_fixtures, max_goals+1, max_goals+1).
        """
        if not self._teams:
            raise RuntimeError("Call fit() before predict_distribution().")

        g = self.max_goals + 1
        n = len(X)
        pmf = np.zeros((n, g, g), dtype=np.float64)

        goals = np.arange(g)

        for i, (_, row) in enumerate(X.iterrows()):
            team_h = row["team_h"]
            team_a = row["team_a"]

            alpha_h = self._attack.get(team_h, 1.0)
            beta_h = self._defence.get(team_h, 1.0)
            alpha_a = self._attack.get(team_a, 1.0)
            beta_a = self._defence.get(team_a, 1.0)

            lambda_h = alpha_h * beta_a * self._home_adv
            lambda_a = alpha_a * beta_h

            p_h = poisson.pmf(goals, lambda_h)  # shape (g,)
            p_a = poisson.pmf(goals, lambda_a)  # shape (g,)

            joint = np.outer(p_h, p_a)  # shape (g, g)

            # Apply Dixon-Coles correction to low-scoring cells
            for h in range(min(2, g)):
                for a in range(min(2, g)):
                    joint[h, a] *= _dc_correction(h, a, lambda_h, lambda_a, self._rho)

            # Renormalise to correct for the correction factor
            total = joint.sum()
            if total > 0:
                joint /= total

            pmf[i] = joint

        fixture_ids: list[int | str]
        if "fixture_id" in X.columns:
            fixture_ids = X["fixture_id"].tolist()
        else:
            fixture_ids = list(range(n))

        return ScoreDistribution(pmf, fixture_ids)

    # ------------------------------------------------------------------
    # Accessors
    # ------------------------------------------------------------------

    @property
    def is_fitted(self) -> bool:
        return bool(self._teams)

    def team_ratings(self) -> pd.DataFrame:
        """Return a DataFrame of fitted attack/defence ratings per team."""
        if not self._teams:
            raise RuntimeError("Call fit() before team_ratings().")
        return (
            pd.DataFrame(
                {
                    "team": self._teams,
                    "attack": [self._attack[t] for t in self._teams],
                    "defence": [self._defence[t] for t in self._teams],
                }
            )
            .sort_values("attack", ascending=False)
            .reset_index(drop=True)
        )


# ------------------------------------------------------------------
# Likelihood helpers (module-level for speed — no closure overhead)
# ------------------------------------------------------------------


def _neg_log_likelihood(
    params: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
    home_idx: np.ndarray,
    away_idx: np.ndarray,
    weights: np.ndarray,
    n_teams: int,
) -> float:
    """Weighted negative log-likelihood for Dixon-Coles model."""
    attack = params[:n_teams]
    defence = params[n_teams : 2 * n_teams]
    home_adv = params[2 * n_teams]
    rho = params[2 * n_teams + 1]

    lambda_h = attack[home_idx] * defence[away_idx] * home_adv
    lambda_a = attack[away_idx] * defence[home_idx]

    # Clamp to avoid log(0)
    lambda_h = np.clip(lambda_h, 1e-6, None)
    lambda_a = np.clip(lambda_a, 1e-6, None)

    log_p_h = poisson.logpmf(home_goals, lambda_h)
    log_p_a = poisson.logpmf(away_goals, lambda_a)

    # Dixon-Coles τ correction (vectorised for 4 low-score cases)
    tau = np.ones(len(home_goals))
    m00 = (home_goals == 0) & (away_goals == 0)
    m01 = (home_goals == 0) & (away_goals == 1)
    m10 = (home_goals == 1) & (away_goals == 0)
    m11 = (home_goals == 1) & (away_goals == 1)

    tau[m00] = 1 - lambda_h[m00] * lambda_a[m00] * rho
    tau[m01] = 1 + lambda_h[m01] * rho
    tau[m10] = 1 + lambda_a[m10] * rho
    tau[m11] = 1 - rho

    # Guard against τ ≤ 0 which would give log(-) = nan
    tau = np.clip(tau, 1e-10, None)
    log_tau = np.log(tau)

    log_likelihood = weights * (log_p_h + log_p_a + log_tau)
    return -float(np.sum(log_likelihood))
