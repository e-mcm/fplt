"""Map bookmaker 1X2 (+ optional O/U 2.5) odds to independent-Poisson λ.

Shared by the odds scoreline baseline (W5) and the Dixon–Coles market prior
(W6). De-vigs decimal odds, then fits λ_home / λ_away so the independent
Poisson scoreline probabilities match the fair 1X2 (and O/U when present).
"""

# scipy does not ship type stubs.
# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from scipy.stats import poisson

_GRID = np.arange(0, 16, dtype=np.int64)
_LAMBDA_BOUNDS = (0.05, 5.0)


def de_vig(odds: np.ndarray) -> np.ndarray:
    """Convert decimal odds to a probability simplex by proportional de-vig.

    Args:
        odds: Decimal odds, each strictly greater than 1.

    Returns:
        Probabilities that sum to 1.

    Raises:
        ValueError: Odds are empty, non-finite, or not strictly greater than 1.
    """
    values = np.asarray(odds, dtype=np.float64)
    if values.size == 0:
        raise ValueError("odds must be non-empty")
    if np.any(~np.isfinite(values)) or np.any(values <= 1.0):
        raise ValueError(f"odds must be finite and > 1, got {values!r}")
    implied = 1.0 / values
    return implied / implied.sum()


def independent_poisson_markets(
    lambda_home: float, lambda_away: float
) -> tuple[float, float, float, float]:
    """1X2 and P(over 2.5) under independent Poisson scorelines.

    Args:
        lambda_home: Home expected goals.
        lambda_away: Away expected goals.

    Returns:
        ``(p_home, p_draw, p_away, p_over_25)`` on a truncated 0..15 grid,
        renormalised so the three 1X2 probs sum to 1.
    """
    home = poisson.pmf(_GRID, lambda_home)
    away = poisson.pmf(_GRID, lambda_away)
    joint = np.outer(home, away)
    total = float(joint.sum())
    if total <= 0:
        return 1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0, 0.5
    joint = joint / total
    p_home = float(np.tril(joint, k=-1).sum())
    p_draw = float(np.trace(joint))
    p_away = float(np.triu(joint, k=1).sum())
    totals = _GRID[:, None] + _GRID[None, :]
    p_over = float(joint[totals >= 3].sum())
    return p_home, p_draw, p_away, p_over


def odds_to_lambda(
    home_odds: float,
    draw_odds: float,
    away_odds: float,
    over_25_odds: float | None = None,
    under_25_odds: float | None = None,
) -> tuple[float, float]:
    """Fit independent-Poisson λ from decimal 1X2 (+ optional O/U 2.5) odds.

    Args:
        home_odds: Home win decimal odds (> 1).
        draw_odds: Draw decimal odds (> 1).
        away_odds: Away win decimal odds (> 1).
        over_25_odds: Optional over 2.5 goals decimal odds.
        under_25_odds: Optional under 2.5 goals decimal odds.

    Returns:
        ``(lambda_home, lambda_away)`` clipped to a stable positive range.

    Raises:
        ValueError: Any required odd is not finite and strictly greater than 1.
    """
    target = de_vig(np.array([home_odds, draw_odds, away_odds], dtype=np.float64))
    target_over: float | None = None
    if (
        over_25_odds is not None
        and under_25_odds is not None
        and np.isfinite(over_25_odds)
        and np.isfinite(under_25_odds)
        and over_25_odds > 1.0
        and under_25_odds > 1.0
    ):
        target_over = float(de_vig(np.array([over_25_odds, under_25_odds]))[0])

    def objective(z: np.ndarray) -> float:
        lambda_h, lambda_a = np.exp(z)
        p_home, p_draw, p_away, p_over = independent_poisson_markets(
            float(lambda_h), float(lambda_a)
        )
        err = (
            (p_home - target[0]) ** 2
            + (p_draw - target[1]) ** 2
            + (p_away - target[2]) ** 2
        )
        if target_over is not None:
            err += (p_over - target_over) ** 2
        return float(err)

    tilt = float(target[0] - target[2])
    start_h = max(1.35 + 0.45 * tilt, 0.25)
    start_a = max(1.15 - 0.45 * tilt, 0.25)
    result = minimize(
        objective,
        np.log([start_h, start_a]),
        method="L-BFGS-B",
        options={"maxiter": 200, "ftol": 1e-12},
    )
    fitted = np.exp(np.asarray(result.x, dtype=np.float64))
    low, high = _LAMBDA_BOUNDS
    return float(np.clip(fitted[0], low, high)), float(np.clip(fitted[1], low, high))


def blend_lambda(
    dc_lambda: float, market_lambda: float | None, strength: float
) -> float:
    """Shrink a DC rate toward a market rate.

    ``strength`` is a pseudo-count: 0 leaves DC unchanged; large values pull
    fully to the market. Missing market λ keeps the DC rate.

    Args:
        dc_lambda: Dixon–Coles expected goals for one side.
        market_lambda: Odds-implied rate, or ``None`` when odds are absent.
        strength: Non-negative prior strength (0 disables the blend).

    Returns:
        Blended expected goals.
    """
    if strength <= 0 or market_lambda is None:
        return dc_lambda
    weight = float(strength)
    return (float(dc_lambda) + weight * float(market_lambda)) / (1.0 + weight)
