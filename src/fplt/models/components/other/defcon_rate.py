"""Defensive-contribution rate → P(award) helpers for other-points.

Models staged action counts as a per-90 Poisson rate (shrunk toward the
position prior), then maps ``P(actions ≥ threshold | minutes)`` through the
season scoring rule. Replaces the blunt shrunk mean of realised 0/2 awards.
"""

# pyright: reportMissingTypeStubs=false, reportUnknownMemberType=false
# pyright: reportUnknownVariableType=false, reportUnknownArgumentType=false

from __future__ import annotations

from typing import Any, cast

from scipy.stats import poisson

from fplt.models.components.involvement.rates import shrunk_rate
from fplt.schemas.components import MINUTE_BUCKETS
from fplt.scoring.rules import Rules

_DEFAULT_BUCKET_MINUTES: dict[str, float] = {
    "0": 0.0,
    "1-59": 45.0,
    "60+": 75.0,
}


def parse_defcon_mode(value: object) -> str:
    """Validate ``defcon_mode`` (``mean`` or ``rate``).

    Args:
        value: Raw ModelSpec param.

    Returns:
        Normalised mode string.

    Raises:
        ValueError: Unsupported mode.
    """
    if value is None:
        return "mean"
    if not isinstance(value, str) or value not in {"mean", "rate"}:
        raise ValueError(f"defcon_mode must be 'mean' or 'rate', got {value!r}")
    return value


def parse_bucket_minutes(raw: object) -> dict[str, float]:
    """Validate minutes representatives for each ``MINUTE_BUCKETS`` label.

    Args:
        raw: Mapping of bucket → minutes, or ``None`` for defaults.

    Returns:
        Bucket → non-negative minutes.

    Raises:
        ValueError: Missing buckets or negative minutes.
    """
    if raw is None:
        return dict(_DEFAULT_BUCKET_MINUTES)
    if not isinstance(raw, dict):
        raise ValueError(
            f"defcon_bucket_minutes must be a mapping, got {type(raw).__name__}"
        )
    out: dict[str, float] = {}
    for bucket in MINUTE_BUCKETS:
        if bucket not in raw:
            raise ValueError(
                f"defcon_bucket_minutes missing bucket {bucket!r}; "
                f"need {list(MINUTE_BUCKETS)}"
            )
        minutes = float(cast(Any, raw[bucket]))
        if minutes < 0:
            raise ValueError(
                f"defcon_bucket_minutes[{bucket!r}] must be non-negative, got {minutes}"
            )
        out[bucket] = minutes
    return out


def action_count(row: dict[str, Any], position: str, rules: Rules) -> int:
    """Return the staged defensive-action count for ``position``.

    Prefers ``defensive_contribution`` when present (FPL's position-specific
    aggregate). Otherwise sums the rule's action columns.
    """
    if not rules.defensive_contributions:
        return 0
    if "defensive_contribution" in row and row["defensive_contribution"] is not None:
        return int(row["defensive_contribution"] or 0)
    for rule in rules.defensive_contributions:
        if position not in rule.positions:
            continue
        return int(sum(int(row.get(name) or 0) for name in rule.actions))
    return 0


def award_rule(rules: Rules, position: str) -> tuple[int, int] | None:
    """Return ``(threshold, points)`` for ``position``, if the season awards DC."""
    for rule in rules.defensive_contributions:
        if position in rule.positions:
            return int(rule.threshold), int(rule.points)
    return None


def p_award(*, rate_per_90: float, minutes: float, threshold: int) -> float:
    """``P(Poisson(λ) ≥ threshold)`` with ``λ = rate_per_90 × minutes / 90``.

    Args:
        rate_per_90: Expected defensive actions per 90 minutes.
        minutes: Playing time for the appearance.
        threshold: Minimum actions to award points.

    Returns:
        Probability in ``[0, 1]``. Zero minutes or non-positive rate → 0.
    """
    if minutes <= 0 or rate_per_90 <= 0 or threshold < 1:
        return 0.0
    lam = rate_per_90 * (minutes / 90.0)
    # Survival: P(X >= k) = 1 - cdf(k - 1).
    return float(poisson.sf(threshold - 1, lam))


def expected_defcon_points(
    *,
    rate_per_90: float,
    minutes: float,
    threshold: int,
    points: int,
) -> float:
    """Expected defensive-contribution points under the Poisson rate model."""
    if points == 0:
        return 0.0
    return p_award(
        rate_per_90=rate_per_90, minutes=minutes, threshold=threshold
    ) * float(points)


def fit_position_rates(
    samples: list[tuple[str, float, float]],
) -> dict[str, float]:
    """Position → actions-per-90 from ``(position, actions, nineties)`` rows."""
    totals: dict[str, list[float]] = {}
    for position, actions, nineties in samples:
        bucket = totals.setdefault(position, [0.0, 0.0])
        bucket[0] += actions
        bucket[1] += nineties
    rates: dict[str, float] = {}
    for position, (actions, nineties) in totals.items():
        rates[position] = (actions / nineties) if nineties > 0 else 0.0
    return rates


def fit_player_rates(
    samples: list[tuple[int, str, float, float]],
    position_rates: dict[str, float],
    *,
    window: int,
    shrinkage: float,
) -> dict[int, float]:
    """Player → shrunk actions-per-90 from recent appearances.

    Args:
        samples: Chronological ``(player_key, position, actions, nineties)``.
        position_rates: Position priors from ``fit_position_rates``.
        window: Max recent appearances per player.
        shrinkage: Pseudo-nineties of the position prior.

    Returns:
        Player-key → rate.
    """
    by_player: dict[int, list[tuple[str, float, float]]] = {}
    for player, position, actions, nineties in samples:
        by_player.setdefault(player, []).append((position, actions, nineties))
    out: dict[int, float] = {}
    for player, rows in by_player.items():
        recent = rows[-window:]
        position = recent[-1][0]
        prior = position_rates.get(position, 0.0)
        actions = sum(item[1] for item in recent)
        nineties = sum(item[2] for item in recent)
        out[player] = shrunk_rate(actions, nineties, prior, shrinkage)
    return out
