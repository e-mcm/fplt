"""Pure post-GW decision scoring (ops-08).

Scores one checked gameweek against the plan's expected points, a
constrained hindsight optimum, captaincy regret, and transfer-hit payback.
All functions are pure: callers supply decision rows and optional ladder /
hindsight inputs. Realised scoring reuses ``score_gameweek`` (autosub,
vice-on-blank, zero-minute players).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import polars as pl

from fplt.backtest.live_percentile import equivalent_percentile
from fplt.data.overall_rank import OverallRankSnapshot
from fplt.optimise.attainable import component_regret
from fplt.optimise.realised import score_gameweek
from fplt.schemas.weekly_report import WeeklyDecisionScoreSchema

GwKind = Literal["normal", "blank", "double"]

__all__ = [
    "GameweekDecisionScore",
    "classify_gameweek",
    "hit_payback_to_date",
    "plan_expected_points",
    "score_gameweek_decisions",
    "score_gameweeks",
    "season_points_to_date",
]


@dataclass(frozen=True)
class GameweekDecisionScore:
    """Decision-quality metrics for one model × season × gameweek."""

    model_id: str
    season: str
    gw: int
    gw_kind: GwKind
    plan_ev: float
    realised_points: int
    realised_minus_plan_ev: float
    hindsight_best: int
    regret: int
    captaincy_regret: int
    hit: float
    hit_payback_to_date: float
    season_points_to_date: int
    points_at_rank: float | None
    percentile: float | None

    def to_row(self) -> dict[str, object]:
        """Return a dict matching ``WeeklyDecisionScoreSchema``."""
        return {
            "model_id": self.model_id,
            "season": self.season,
            "gw": self.gw,
            "gw_kind": self.gw_kind,
            "plan_ev": self.plan_ev,
            "realised_points": self.realised_points,
            "realised_minus_plan_ev": self.realised_minus_plan_ev,
            "hindsight_best": self.hindsight_best,
            "regret": self.regret,
            "captaincy_regret": self.captaincy_regret,
            "hit": self.hit,
            "hit_payback_to_date": self.hit_payback_to_date,
            "season_points_to_date": self.season_points_to_date,
            "points_at_rank": self.points_at_rank,
            "percentile": self.percentile,
        }


def classify_gameweek(
    *,
    n_fixtures: int | None = None,
    all_zero_minutes: bool = False,
) -> GwKind:
    """Classify a gameweek as normal, blank, or double.

    Args:
        n_fixtures: League fixtures in the gameweek. ``0`` is a full blank;
            more than 10 implies at least one double. ``None`` leaves the
            fixture count unknown.
        all_zero_minutes: True when every kept player recorded 0 minutes
            (blanked squad / BGW with no appearances).

    Returns:
        ``blank``, ``double``, or ``normal``.
    """
    if all_zero_minutes or n_fixtures == 0:
        return "blank"
    if n_fixtures is not None and n_fixtures > 10:
        return "double"
    return "normal"


def plan_expected_points(picks: pl.DataFrame) -> float:
    """Captain-doubled XI expected points minus the gameweek hit.

    Uses the plan's ``xpts`` (not actuals). Vice is ignored in the mean —
    same convention as ``fplt.optimise.explain``. Sold rows are excluded.
    An empty XI (true blank week with no picks) scores 0 before the hit.

    Args:
        picks: Decision rows for one model × season × gameweek.

    Returns:
        Net plan EV for the act-on week.

    Raises:
        ValueError: Required columns are missing.
    """
    required = {"in_xi", "captain", "xpts", "hit"}
    missing = required - set(picks.columns)
    if missing:
        raise ValueError(f"picks missing columns: {sorted(missing)}")
    kept = picks.filter(~pl.col("sold")) if "sold" in picks.columns else picks
    xi = kept.filter(pl.col("in_xi"))
    hit = float(kept["hit"][0]) if kept.height else 0.0
    hit = max(hit, 0.0)
    if xi.is_empty():
        return -hit
    base = float(xi["xpts"].sum())
    captain = xi.filter(pl.col("captain"))
    extra = float(captain["xpts"].sum()) if not captain.is_empty() else 0.0
    return base + extra - hit


def hit_payback_to_date(decisions: pl.DataFrame, *, as_of_gw: int) -> float:
    """Cumulative points from hit-week buys minus hits paid through ``as_of_gw``.

    For each gameweek ``g ≤ as_of_gw`` with ``hit > 0``, every player bought
    that week contributes their kept ``actual_pts`` from ``g`` through
    ``as_of_gw`` (rows where they are not sold). Hits are summed once per
    gameweek. Positive means the hits have paid back on raw points.

    Args:
        decisions: Multi-GW ``DecisionSchema`` rows for one model × season.
        as_of_gw: Inclusive gameweek cutoff.

    Returns:
        ``bought_points - hits`` through ``as_of_gw``.

    Raises:
        ValueError: Required columns are missing.
    """
    required = {"gw", "player_key", "bought", "sold", "actual_pts", "hit"}
    missing = required - set(decisions.columns)
    if missing:
        raise ValueError(f"decisions missing columns: {sorted(missing)}")
    window = decisions.filter(pl.col("gw") <= as_of_gw)
    if window.is_empty():
        return 0.0
    hit_gws = window.filter(pl.col("hit") > 0).select("gw", "hit").unique().sort("gw")
    hits_total = float(hit_gws["hit"].sum()) if hit_gws.height else 0.0
    buy_keys: dict[int, set[int]] = {}
    for row in window.filter(pl.col("bought") & (pl.col("hit") > 0)).iter_rows(
        named=True
    ):
        buy_keys.setdefault(int(row["gw"]), set()).add(int(row["player_key"]))
    bought_pts = 0.0
    for buy_gw, keys in buy_keys.items():
        later = window.filter(
            (pl.col("gw") >= buy_gw)
            & (pl.col("gw") <= as_of_gw)
            & pl.col("player_key").is_in(list(keys))
            & ~pl.col("sold")
        )
        if later.is_empty():
            continue
        pts = later.select(
            pl.when(pl.col("actual_pts").is_null())
            .then(0)
            .otherwise(pl.col("actual_pts"))
            .sum()
            .alias("pts")
        )
        bought_pts += float(pts["pts"][0])
    return bought_pts - hits_total


def season_points_to_date(decisions: pl.DataFrame, *, as_of_gw: int) -> int:
    """Net realised season points through ``as_of_gw`` (inclusive).

    Args:
        decisions: Decision rows for one model × season (may include later
            gameweeks; they are ignored).
        as_of_gw: Inclusive cutoff.

    Returns:
        Sum of per-GW net points after hits / autosub / vice-on-blank.
        Gameweeks with no ``in_xi`` row contribute ``-hit`` only.
    """
    window = decisions.filter(pl.col("gw") <= as_of_gw)
    if window.is_empty():
        return 0
    total = 0
    for gameweek in sorted(int(g) for g in window["gw"].unique().to_list()):
        part = window.filter(pl.col("gw") == gameweek)
        hit = float(part["hit"][0]) if part.height else 0.0
        kept = part.filter(~pl.col("sold")) if "sold" in part.columns else part
        if kept.filter(pl.col("in_xi")).is_empty():
            total -= int(max(hit, 0.0))
            continue
        total += score_gameweek(part, hit=hit)
    return total


def score_gameweek_decisions(
    decisions: pl.DataFrame,
    *,
    gw: int,
    model_id: str | None = None,
    hindsight_best: int | None = None,
    n_fixtures: int | None = None,
    rank_snapshot: OverallRankSnapshot | None = None,
    market: pl.DataFrame | None = None,
) -> GameweekDecisionScore:
    """Score one gameweek's decisions against plan EV and hindsight.

    When ``hindsight_best`` is omitted, uses the local constrained optimum
    on the owned fifteen: realised + captaincy regret + XI/bench regret
    (no further transfers). Pass an oracle GW total for transfer-inclusive
    regret under the same banking / hit / budget rules.

    Args:
        decisions: Decision rows for one model × season (at least ``gw``).
        gw: Target gameweek.
        model_id: Required when ``decisions`` covers more than one model.
        hindsight_best: Optional constrained-hindsight net points for ``gw``.
        n_fixtures: Optional fixture count for DGW/BGW classification.
        rank_snapshot: Optional overall-league ladder for points-at-rank.
        market: Optional transfer-market frame for ``component_regret``.

    Returns:
        A ``GameweekDecisionScore`` for ``gw``.

    Raises:
        ValueError: ``gw`` has no rows, multiple models without ``model_id``,
            or required columns are missing.
    """
    part = decisions.filter(pl.col("gw") == gw)
    if part.is_empty():
        raise ValueError(f"no decision rows for gw={gw}")
    models = part["model_id"].unique().to_list()
    if model_id is not None:
        part = part.filter(pl.col("model_id") == model_id)
        if part.is_empty():
            raise ValueError(f"no decision rows for model_id={model_id} gw={gw}")
    elif len(models) > 1:
        raise ValueError("multiple models present; pass model_id=")
    mid = str(part["model_id"][0])
    season = str(part["season"][0])
    model_history = decisions.filter(pl.col("model_id") == mid)
    hit = float(part["hit"][0])
    kept = part.filter(~pl.col("sold")) if "sold" in part.columns else part
    all_zero = kept.height > 0 and bool(
        (kept["actual_minutes"].fill_null(0) == 0).all()
    )
    # Empty XI is a blank week with no squad to score.
    if kept.filter(pl.col("in_xi")).is_empty():
        realised = -int(max(hit, 0.0))
        captaincy = 0
        local_hindsight = realised
        gw_kind = classify_gameweek(n_fixtures=n_fixtures, all_zero_minutes=True)
    else:
        realised = score_gameweek(part, hit=hit)
        regret_frame = component_regret(part, market=market)
        captaincy = int(regret_frame["captaincy_regret"][0])
        xi_bench = int(regret_frame["xi_bench_regret"][0])
        local_hindsight = realised + captaincy + xi_bench
        gw_kind = classify_gameweek(n_fixtures=n_fixtures, all_zero_minutes=all_zero)
    plan_ev = plan_expected_points(part)
    best = local_hindsight if hindsight_best is None else int(hindsight_best)
    regret = max(0, best - realised)
    payback = hit_payback_to_date(model_history, as_of_gw=gw)
    season_pts = season_points_to_date(model_history, as_of_gw=gw)
    rank: float | None = None
    percentile: float | None = None
    if rank_snapshot is not None:
        eq = equivalent_percentile(season_pts, rank_snapshot)
        rank = eq.rank
        percentile = eq.percentile
    return GameweekDecisionScore(
        model_id=mid,
        season=season,
        gw=gw,
        gw_kind=gw_kind,
        plan_ev=plan_ev,
        realised_points=realised,
        realised_minus_plan_ev=float(realised) - plan_ev,
        hindsight_best=best,
        regret=regret,
        captaincy_regret=captaincy,
        hit=max(hit, 0.0),
        hit_payback_to_date=payback,
        season_points_to_date=season_pts,
        points_at_rank=rank,
        percentile=percentile,
    )


def score_gameweeks(
    decisions: pl.DataFrame,
    *,
    gameweeks: list[int] | None = None,
    hindsight_by_gw: dict[int, int] | None = None,
    n_fixtures_by_gw: dict[int, int] | None = None,
    rank_snapshot: OverallRankSnapshot | None = None,
    market: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Score many gameweeks into a ``WeeklyDecisionScoreSchema`` frame.

    Args:
        decisions: Decision rows for one or more models × season.
        gameweeks: GWs to score (default: every GW present in ``decisions``).
        hindsight_by_gw: Optional oracle net points per GW (shared across
            models when set).
        n_fixtures_by_gw: Optional fixture counts for DGW/BGW tags.
        rank_snapshot: Optional ladder for points-at-rank on each row.
        market: Optional transfer-market frame for component regret.

    Returns:
        Validated weekly score rows sorted by ``model_id``, ``gw``.
    """
    if decisions.is_empty():
        return _empty_scores()
    gws = gameweeks or sorted(int(g) for g in decisions["gw"].unique().to_list())
    hindsight = hindsight_by_gw or {}
    fixtures = n_fixtures_by_gw or {}
    models = sorted(str(m) for m in decisions["model_id"].unique().to_list())
    rows: list[dict[str, object]] = []
    for mid in models:
        for gw in gws:
            part = decisions.filter((pl.col("model_id") == mid) & (pl.col("gw") == gw))
            if part.is_empty():
                continue
            rows.append(
                score_gameweek_decisions(
                    decisions,
                    gw=gw,
                    model_id=mid,
                    hindsight_best=hindsight.get(gw),
                    n_fixtures=fixtures.get(gw),
                    rank_snapshot=rank_snapshot,
                    market=market,
                ).to_row()
            )
    if not rows:
        return _empty_scores()
    frame = pl.DataFrame(rows).sort("model_id", "gw")
    return WeeklyDecisionScoreSchema.validate(frame)  # type: ignore[return-value]


def _empty_scores() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "model_id": pl.Utf8(),
            "season": pl.Utf8(),
            "gw": pl.Int64(),
            "gw_kind": pl.Utf8(),
            "plan_ev": pl.Float64(),
            "realised_points": pl.Int64(),
            "realised_minus_plan_ev": pl.Float64(),
            "hindsight_best": pl.Int64(),
            "regret": pl.Int64(),
            "captaincy_regret": pl.Int64(),
            "hit": pl.Float64(),
            "hit_payback_to_date": pl.Float64(),
            "season_points_to_date": pl.Int64(),
            "points_at_rank": pl.Float64(),
            "percentile": pl.Float64(),
        }
    )
