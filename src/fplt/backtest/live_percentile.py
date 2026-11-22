"""Map season points totals onto overall-league rank / percentile.

Evaluation north star §3: live rank is noisy; skill shows up as **percentile
across several seasons**, not one finish. Use a cached points-at-rank ladder
(``fplt.data.overall_rank``) to convert any season total — realised or
backtested — into an equivalent rank and top-fraction percentile.

**Single-season live rank is not a promote gate.** Promote-eval stays on
paired forecast / decision metrics; this module is a multi-season consistency
check only (adoption note 6 in evaluation-north-star).
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from statistics import fmean

from fplt.data.overall_rank import OverallRankSnapshot, PointsAtRank


@dataclass(frozen=True)
class RankEquivalent:
    """Season points mapped through an overall-league ladder.

    ``percentile`` is the top fraction of the field (``rank / total_players``),
    so ``0.01`` means roughly top 1% / ~100k in a 10M+ game.
    """

    season: str
    season_points: int
    rank: float
    percentile: float
    total_players: int
    captured_at: str


@dataclass(frozen=True)
class SeasonPercentile:
    """One season's equivalent percentile for a multi-season rollup."""

    season: str
    season_points: int
    percentile: float
    rank: float


def ladder_rows(snapshot: OverallRankSnapshot) -> list[PointsAtRank]:
    """Return ladder rows sorted by points descending (better ranks first).

    Args:
        snapshot: Cached overall-rank ladder.

    Returns:
        Rows ordered for interpolation (highest points first).
    """
    return sorted(snapshot.ranks, key=lambda row: (-row.points, row.observed_rank))


def equivalent_rank(season_points: int, snapshot: OverallRankSnapshot) -> float:
    """Interpolate overall rank for ``season_points`` from a ladder.

    Uses observed ranks (nearest sample to each target) and linear
    interpolation in points space between neighbouring ladder rungs.
    Points at or above the top rung map to that rung's rank; at or below the
    bottom rung map to that rung's rank (no extrapolation past the samples).

    Args:
        season_points: Season total (realised or backtested).
        snapshot: Cached points-at-rank ladder for the same season context.

    Returns:
        Estimated overall rank (lower is better).

    Raises:
        ValueError: The snapshot has no ranks.
    """
    rows = ladder_rows(snapshot)
    if not rows:
        raise ValueError("overall-rank snapshot has no ranks")
    if season_points >= rows[0].points:
        return float(rows[0].observed_rank)
    if season_points <= rows[-1].points:
        return float(rows[-1].observed_rank)
    for left, right in zip(rows, rows[1:], strict=False):
        if left.points >= season_points >= right.points:
            span = left.points - right.points
            if span == 0:
                return float(left.observed_rank)
            frac = (left.points - season_points) / span
            return left.observed_rank + frac * (
                right.observed_rank - left.observed_rank
            )
    return float(rows[-1].observed_rank)


def equivalent_percentile(
    season_points: int, snapshot: OverallRankSnapshot
) -> RankEquivalent:
    """Map ``season_points`` to rank and top-fraction percentile.

    Args:
        season_points: Season total.
        snapshot: Ladder including ``total_players``.

    Returns:
        ``RankEquivalent`` with ``percentile = rank / total_players``.

    Raises:
        ValueError: ``total_players`` is not positive, or the ladder is empty.
    """
    if snapshot.total_players <= 0:
        raise ValueError("total_players must be positive")
    rank = equivalent_rank(season_points, snapshot)
    percentile = rank / snapshot.total_players
    return RankEquivalent(
        season=snapshot.season,
        season_points=season_points,
        rank=rank,
        percentile=percentile,
        total_players=snapshot.total_players,
        captured_at=snapshot.captured_at.strftime("%Y-%m-%dT%H-%M-%SZ"),
    )


def mean_percentile(seasons: Sequence[SeasonPercentile | RankEquivalent]) -> float:
    """Arithmetic mean of top-fraction percentiles across seasons.

    This is the live-results consistency summary from evaluation-north-star
    §3. **It is not a promote gate**: a single season (or a single live rank)
    must not decide model promotion; use paired forecast/decision promote-eval
    for that, and treat this rollup as a multi-season check only.

    Args:
        seasons: Per-season percentile results (at least one).

    Returns:
        Mean of ``percentile`` values (still a top fraction; lower is better).

    Raises:
        ValueError: ``seasons`` is empty.
    """
    if not seasons:
        raise ValueError("seasons must be non-empty")
    return fmean(row.percentile for row in seasons)


def season_percentiles_from_points(
    points_by_season: Sequence[tuple[str, int]],
    snapshots: dict[str, OverallRankSnapshot],
) -> list[SeasonPercentile]:
    """Map each ``(season, points)`` through that season's ladder.

    Args:
        points_by_season: ``(season_label, season_points)`` pairs.
        snapshots: Ladders keyed by season label.

    Returns:
        One ``SeasonPercentile`` per input row, in input order.

    Raises:
        KeyError: A season has no snapshot in ``snapshots``.
    """
    out: list[SeasonPercentile] = []
    for season, points in points_by_season:
        snap = snapshots[season]
        eq = equivalent_percentile(points, snap)
        out.append(
            SeasonPercentile(
                season=season,
                season_points=points,
                percentile=eq.percentile,
                rank=eq.rank,
            )
        )
    return out
