"""Mid-season schedule and form covariates for Dixon–Coles rates.

Predict-time adjustments only: fitted attack/defence stay the batch MLE.
Absent / zero strengths recover vanilla λ. Rest and congestion reuse the
minutes all-comp schedule helpers (PL-only when cups are empty). Form is a
current-season residual of goals for/against versus the fitted rates.
"""

from __future__ import annotations

import numpy as np
import polars as pl

from fplt.data.pit import PITView
from fplt.models.components.minutes.congestion import (
    competition_team_kickoffs,
    team_schedule_features,
)

_FORM_MULT_LO = 0.5
_FORM_MULT_HI = 2.0
_REST_NEUTRAL_WEEKS = 1.0


def schedule_rate_multipliers(
    view: PITView,
    fixtures: pl.DataFrame,
    *,
    rest_coef: float,
    congestion_coef: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Per-fixture home/away λ multipliers from rest and congestion.

    Rest uses weeks since the previous kickoff (clipped ``[0, 3]``); a week of
    rest is neutral. Congestion is prior games in seven days (clipped
    ``[0, 3]``). Both coefs are stored ≥ 0; congestion applies as a penalty::

        mult = exp(rest_coef · (rest_weeks − 1) − congestion_coef · games_last_7d)

    Args:
        view: Deadline view (fixtures + optional ``competition_fixtures``).
        fixtures: Rows with ``season``, ``fixture_id``, ``team_h_key``,
            ``team_a_key``; ``kickoff_time`` is joined from PIT when missing.
        rest_coef: Strength of rest (0 = ignore).
        congestion_coef: Strength of congestion penalty (0 = ignore).

    Returns:
        ``(home_mult, away_mult)`` aligned with ``fixtures``, ones when both
        coefs are 0 or schedule rows are missing.
    """
    n = fixtures.height
    ones = np.ones(n, dtype=np.float64)
    if rest_coef <= 0 and congestion_coef <= 0:
        return ones.copy(), ones.copy()
    enriched = _with_kickoffs(view, fixtures)
    if enriched.is_empty() or "kickoff_time" not in enriched.columns:
        return ones.copy(), ones.copy()
    calendar = (
        view.latest("fixtures", by=["season", "fixture_id"])
        .select(
            "season",
            "fixture_id",
            "team_h_key",
            "team_a_key",
            "kickoff_time",
        )
        .collect()
    )
    if calendar.is_empty():
        return ones.copy(), ones.copy()
    schedule = team_schedule_features(calendar, competition_team_kickoffs(view)).select(
        "season", "fixture_id", "team_key", "rest_days", "games_last_7d"
    )
    home = (
        enriched.select(
            "season",
            "fixture_id",
            pl.col("team_h_key").alias("team_key"),
        )
        .with_row_index("_row")
        .join(schedule, on=["season", "fixture_id", "team_key"], how="left")
        .sort("_row")
    )
    away = (
        enriched.select(
            "season",
            "fixture_id",
            pl.col("team_a_key").alias("team_key"),
        )
        .with_row_index("_row")
        .join(schedule, on=["season", "fixture_id", "team_key"], how="left")
        .sort("_row")
    )
    return (
        _side_multipliers(home, rest_coef=rest_coef, congestion_coef=congestion_coef),
        _side_multipliers(away, rest_coef=rest_coef, congestion_coef=congestion_coef),
    )


def form_rating_multipliers(
    played: pl.DataFrame,
    teams: list[int],
    attack: dict[int, float],
    defence: dict[int, float],
    *,
    home_advantage: float,
    form_window: int,
    form_strength: float,
    season: str | None,
) -> tuple[dict[int, float], dict[int, float]]:
    """Per-team attack/defence multipliers from recent goal residuals.

    For each team, take the last ``form_window`` finished matches in
    ``season`` (or all played rows when ``season`` is None). Residual is
    observed goals for/against minus the rate implied by ``attack`` /
    ``defence`` at those matches. Positive scoring residual raises attack;
    positive conceding residual raises defence (easier to score against).

    Args:
        played: Finished fixtures with scores and team keys.
        teams: Team keys in the identification.
        attack: Fitted attack by ``team_key``.
        defence: Fitted defence by ``team_key``.
        home_advantage: Home advantage scalar used for expected rates.
        form_window: Recent matches per team (≤ 0 disables).
        form_strength: Multiplier on the mean residual in log space (0 = off).
        season: Restrict form to this season when set.

    Returns:
        ``(attack_mult, defence_mult)`` maps; missing teams default to 1.0.
    """
    ones = {team: 1.0 for team in teams}
    if form_window <= 0 or form_strength <= 0 or played.is_empty():
        return ones, dict(ones)
    frame = played
    if season is not None and "season" in frame.columns:
        frame = frame.filter(pl.col("season") == season)
    if frame.is_empty():
        return ones, dict(ones)
    frame = frame.sort("kickoff_time")
    attack_mult = dict(ones)
    defence_mult = dict(ones)
    for team in teams:
        home_rows = frame.filter(pl.col("team_h_key") == team)
        away_rows = frame.filter(pl.col("team_a_key") == team)
        sides = pl.concat(
            [
                home_rows.select(
                    pl.col("kickoff_time"),
                    pl.col("team_a_key").alias("opp"),
                    pl.col("team_h_score").alias("gf"),
                    pl.col("team_a_score").alias("ga"),
                    pl.lit(True).alias("is_home"),
                ),
                away_rows.select(
                    pl.col("kickoff_time"),
                    pl.col("team_h_key").alias("opp"),
                    pl.col("team_a_score").alias("gf"),
                    pl.col("team_h_score").alias("ga"),
                    pl.lit(False).alias("is_home"),
                ),
            ]
        ).sort("kickoff_time")
        if sides.is_empty():
            continue
        recent = sides.tail(form_window)
        gf_resid: list[float] = []
        ga_resid: list[float] = []
        for row in recent.iter_rows(named=True):
            opp = int(row["opp"])
            a_self = float(attack.get(team, 1.0))
            d_self = float(defence.get(team, 1.0))
            a_opp = float(attack.get(opp, 1.0))
            d_opp = float(defence.get(opp, 1.0))
            if bool(row["is_home"]):
                exp_gf = a_self * d_opp * home_advantage
                exp_ga = a_opp * d_self
            else:
                exp_gf = a_self * d_opp
                exp_ga = a_opp * d_self * home_advantage
            gf_resid.append(float(row["gf"]) - exp_gf)
            ga_resid.append(float(row["ga"]) - exp_ga)
        mean_gf = float(np.mean(gf_resid)) if gf_resid else 0.0
        mean_ga = float(np.mean(ga_resid)) if ga_resid else 0.0
        attack_mult[team] = _clip_mult(np.exp(form_strength * mean_gf))
        defence_mult[team] = _clip_mult(np.exp(form_strength * mean_ga))
    return attack_mult, defence_mult


def _side_multipliers(
    side: pl.DataFrame, *, rest_coef: float, congestion_coef: float
) -> np.ndarray:
    rest = (
        side["rest_days"].fill_null(_REST_NEUTRAL_WEEKS).to_numpy().astype(np.float64)
    )
    games = side["games_last_7d"].fill_null(0.0).to_numpy().astype(np.float64)
    log_mult = rest_coef * (rest - _REST_NEUTRAL_WEEKS) - congestion_coef * games
    return np.exp(log_mult)


def _clip_mult(value: float) -> float:
    return float(min(max(value, _FORM_MULT_LO), _FORM_MULT_HI))


def _with_kickoffs(view: PITView, fixtures: pl.DataFrame) -> pl.DataFrame:
    """Ensure ``kickoff_time`` is present by joining PIT fixtures when needed."""
    has_kickoff = "kickoff_time" in fixtures.columns
    if has_kickoff and fixtures["kickoff_time"].null_count() == 0:
        return fixtures
    calendar = (
        view.latest("fixtures", by=["season", "fixture_id"])
        .select("season", "fixture_id", "kickoff_time")
        .collect()
    )
    if calendar.is_empty():
        return fixtures
    if "season" in fixtures.columns:
        return fixtures.join(calendar, on=["season", "fixture_id"], how="left")
    return fixtures.join(
        calendar.select("fixture_id", "kickoff_time"),
        on="fixture_id",
        how="left",
    )
