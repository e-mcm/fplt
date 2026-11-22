"""Map bookmaker 1X2 (+ optional O/U 2.5) odds to independent-Poisson λ.

Shared by the odds scoreline baseline (W5), the Dixon–Coles market prior
(W6), and the CS-only market-λ blend. De-vigs decimal odds, then fits
λ_home / λ_away so the independent Poisson scoreline probabilities match
the fair 1X2 (and O/U when present).
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


def outcome_probs_from_joint(
    joint: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Home / draw / away probabilities from a joint scoreline grid.

    Args:
        joint: Non-negative masses with shape ``(F, G, G)`` or ``(G, G)``.

    Returns:
        ``(p_home, p_draw, p_away)`` each shape ``(F,)`` (or scalars wrapped).
    """
    grid = np.asarray(joint, dtype=np.float64)
    single = grid.ndim == 2
    if single:
        grid = grid[None, ...]
    if grid.ndim != 3 or grid.shape[1] != grid.shape[2]:
        raise ValueError(f"joint must have shape (F, G, G), got {grid.shape}")
    goals = np.arange(grid.shape[1], dtype=np.int64)
    home_gt = goals[:, None] > goals[None, :]
    home_eq = goals[:, None] == goals[None, :]
    home_lt = goals[:, None] < goals[None, :]
    p_home = grid[:, home_gt].sum(axis=1)
    p_draw = grid[:, home_eq].sum(axis=1)
    p_away = grid[:, home_lt].sum(axis=1)
    if single:
        return p_home[0:1], p_draw[0:1], p_away[0:1]
    return p_home, p_draw, p_away


def cs_probs_from_lambda(lambda_home: float, lambda_away: float) -> tuple[float, float]:
    """Independent-Poisson clean-sheet probs from expected goals.

    Home CS is ``P(away goals = 0)``; away CS is ``P(home goals = 0)``.

    Args:
        lambda_home: Home expected goals.
        lambda_away: Away expected goals.

    Returns:
        ``(p_cs_home, p_cs_away)``.
    """
    return float(poisson.pmf(0, lambda_away)), float(poisson.pmf(0, lambda_home))


def rescale_joint_cs_margins(
    joint: np.ndarray,
    p_cs_home: float | np.ndarray,
    p_cs_away: float | np.ndarray,
) -> np.ndarray:
    """Rescale zero-goal margins so clean-sheet probs match the targets.

    Home CS is the away-goals=0 slice; away CS is the home-goals=0 slice.
    Conditional shape inside each zero / positive slice is preserved. Rows
    are renormalised to sum to 1. Callers that must keep 1X2 should follow
    with :func:`rescale_joint_to_1x2`.

    Args:
        joint: Non-negative masses, shape ``(F, G, G)`` or ``(G, G)``.
        p_cs_home / p_cs_away: Target clean-sheet probs (scalars or ``(F,)``).

    Returns:
        Rescaled joint with the same shape as ``joint``.
    """
    grid = np.asarray(joint, dtype=np.float64).copy()
    single = grid.ndim == 2
    if single:
        grid = grid[None, ...]
    if grid.ndim != 3 or grid.shape[1] != grid.shape[2]:
        raise ValueError(f"joint must have shape (F, G, G), got {grid.shape}")
    n, _, _ = grid.shape
    targets_home = np.broadcast_to(np.asarray(p_cs_home, dtype=np.float64), (n,))
    targets_away = np.broadcast_to(np.asarray(p_cs_away, dtype=np.float64), (n,))
    out = grid.copy()
    for index in range(n):
        want_h = float(np.clip(targets_home[index], 0.0, 1.0))
        col0 = out[index, :, 0]
        rest_a = out[index, :, 1:]
        mass0 = float(col0.sum())
        mass1 = float(rest_a.sum())
        if want_h > 0 and mass0 > 0:
            out[index, :, 0] = col0 * (want_h / mass0)
        elif want_h > 0:
            out[index, 0, 0] = want_h
        if want_h < 1.0 and mass1 > 0:
            out[index, :, 1:] = rest_a * ((1.0 - want_h) / mass1)

        want_a = float(np.clip(targets_away[index], 0.0, 1.0))
        row0 = out[index, 0, :]
        rest_h = out[index, 1:, :]
        mass0 = float(row0.sum())
        mass1 = float(rest_h.sum())
        if want_a > 0 and mass0 > 0:
            out[index, 0, :] = row0 * (want_a / mass0)
        elif want_a > 0:
            out[index, 0, 0] = want_a
        if want_a < 1.0 and mass1 > 0:
            out[index, 1:, :] = rest_h * ((1.0 - want_a) / mass1)

    totals = out.sum(axis=(1, 2), keepdims=True)
    totals = np.where(totals > 0, totals, 1.0)
    out = out / totals
    return out[0] if single else out


def rescale_joint_to_1x2(
    joint: np.ndarray,
    p_home: float | np.ndarray,
    p_draw: float | np.ndarray,
    p_away: float | np.ndarray,
) -> np.ndarray:
    """Rescale H/D/A regions of ``joint`` so their masses match target 1X2.

    Conditional scoreline shape inside each outcome region is preserved.
    Empty regions fall back to a single representative scoreline so the
    target mass is still placed.

    Args:
        joint: Non-negative masses, shape ``(F, G, G)`` or ``(G, G)``.
        p_home / p_draw / p_away: Target outcome probs (scalars or ``(F,)``).

    Returns:
        Rescaled joint with the same shape as ``joint``, rows summing to 1.
    """
    grid = np.asarray(joint, dtype=np.float64).copy()
    single = grid.ndim == 2
    if single:
        grid = grid[None, ...]
    if grid.ndim != 3 or grid.shape[1] != grid.shape[2]:
        raise ValueError(f"joint must have shape (F, G, G), got {grid.shape}")
    n, width, _ = grid.shape
    goals = np.arange(width, dtype=np.int64)
    home_gt = goals[:, None] > goals[None, :]
    home_eq = goals[:, None] == goals[None, :]
    home_lt = goals[:, None] < goals[None, :]
    targets = (
        np.broadcast_to(np.asarray(p_home, dtype=np.float64), (n,)),
        np.broadcast_to(np.asarray(p_draw, dtype=np.float64), (n,)),
        np.broadcast_to(np.asarray(p_away, dtype=np.float64), (n,)),
    )
    masks = (home_gt, home_eq, home_lt)
    reps = ((1, 0), (0, 0), (0, 1))
    out = np.zeros_like(grid)
    for target, mask, (h_rep, a_rep) in zip(targets, masks, reps, strict=True):
        region = np.where(mask[None, ...], grid, 0.0)
        mass = region.sum(axis=(1, 2))
        for index in range(n):
            want = float(target[index])
            if want <= 0:
                continue
            if mass[index] > 0:
                out[index] += region[index] * (want / mass[index])
            else:
                out[index, h_rep, a_rep] += want
    totals = out.sum(axis=(1, 2), keepdims=True)
    totals = np.where(totals > 0, totals, 1.0)
    out = out / totals
    return out[0] if single else out
