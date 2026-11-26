"""Pure post-GW decision scoring (ops-08)."""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl
import pytest

from fplt.backtest.live_percentile import equivalent_percentile
from fplt.data.overall_rank import OverallRankSnapshot
from fplt.optimise.decision_scoring import (
    classify_gameweek,
    hit_payback_to_date,
    plan_expected_points,
    score_gameweek_decisions,
    score_gameweeks,
)
from fplt.optimise.realised import score_gameweek

_FIXTURES = Path(__file__).parents[1] / "fixtures" / "fpl_api"


def test_plan_ev_doubles_captain_and_subtracts_hit() -> None:
    picks = _decision_gw(
        xi_pts=[2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 6],
        xi_xpts=[2.0, 3.0, 4.0, 5.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 6.0],
        captain_index=10,
        hit=4,
    )
    # XI sum 26 + captain 6 - hit 4 = 28
    assert plan_expected_points(picks) == pytest.approx(28.0)


def test_synthetic_gw_known_optimum_and_regret() -> None:
    """Wrong captain → known captaincy regret; hindsight = local optimum."""
    picks = _decision_gw(
        xi_pts=[2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 8],
        captain_index=0,  # captained the 2-pt GK; best is the 8-pt FWD
    )
    score = score_gameweek_decisions(picks, gw=1)
    realised = score_gameweek(picks, hit=0)
    assert score.realised_points == realised
    assert score.captaincy_regret == 6
    assert score.hindsight_best == realised + 6
    assert score.regret == 6
    assert score.realised_minus_plan_ev == pytest.approx(0.0)


def test_hindsight_override_sets_transfer_inclusive_regret() -> None:
    picks = _decision_gw(xi_pts=[2] * 11, captain_index=0)
    realised = score_gameweek(picks, hit=0)
    score = score_gameweek_decisions(picks, gw=1, hindsight_best=realised + 12)
    assert score.regret == 12
    assert score.hindsight_best == realised + 12


def test_zero_minute_starter_does_not_score() -> None:
    picks = _decision_gw(
        xi_pts=[2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 0],
        xi_mins=[90, 90, 90, 90, 90, 90, 90, 90, 90, 90, 0],
        captain_index=10,
        vice_index=3,
    )
    score = score_gameweek_decisions(picks, gw=1)
    # Captain blanked; vice (5) already in XI → +5 armband; no zero-min pts.
    assert score.realised_points == score_gameweek(picks, hit=0)
    assert score.gw_kind == "normal"


def test_double_gameweek_sums_fixture_points() -> None:
    """DGW actuals arrive pre-aggregated at player–GW grain (two fixtures)."""
    picks = _decision_gw(
        xi_pts=[2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 14],  # 7+7 across two fixtures
        xi_mins=[90] * 10 + [180],
        captain_index=10,
        n_fixtures=12,
    )
    score = score_gameweek_decisions(picks, gw=1, n_fixtures=12)
    assert score.gw_kind == "double"
    # XI 2+3+4+5+1*6+14 = 34 + captain 14 = 48
    assert score.realised_points == 48


def test_blank_gameweek_all_zero_minutes() -> None:
    picks = _decision_gw(
        xi_pts=[0] * 11,
        xi_mins=[0] * 11,
        captain_index=0,
        bench_mins=[0, 0, 0, 0],
    )
    score = score_gameweek_decisions(picks, gw=1, n_fixtures=0)
    assert score.gw_kind == "blank"
    assert score.realised_points == 0
    assert score.regret == 0


def test_blank_gameweek_empty_xi() -> None:
    picks = _decision_gw(xi_pts=[2] * 11, captain_index=0).with_columns(
        pl.lit(False).alias("in_xi")
    )
    score = score_gameweek_decisions(picks, gw=1, n_fixtures=0)
    assert score.gw_kind == "blank"
    assert score.realised_points == 0
    assert score.plan_ev == pytest.approx(0.0)


def test_hit_payback_to_date_tracks_bought_points_minus_hits() -> None:
    gw1 = _decision_gw(xi_pts=[2] * 11, captain_index=0, hit=4).with_columns(
        pl.when(pl.col("player_key") == 11)
        .then(pl.lit(True))
        .otherwise(pl.col("bought"))
        .alias("bought"),
        pl.when(pl.col("player_key") == 11)
        .then(pl.lit(6))
        .otherwise(pl.col("actual_pts"))
        .alias("actual_pts"),
    )
    gw2 = gw1.with_columns(
        pl.lit(2).cast(pl.Int64).alias("gw"),
        pl.lit(0.0).cast(pl.Float64).alias("hit"),
        pl.lit(False).alias("bought"),
        pl.when(pl.col("player_key") == 11)
        .then(pl.lit(5))
        .otherwise(pl.col("actual_pts"))
        .alias("actual_pts"),
    )
    decisions = pl.concat([gw1, gw2])
    # Bought FWD scored 6 + 5 = 11; hit 4 → payback +7
    assert hit_payback_to_date(decisions, as_of_gw=2) == pytest.approx(7.0)
    assert hit_payback_to_date(decisions, as_of_gw=1) == pytest.approx(2.0)


def test_points_at_rank_uses_live_percentile_ladder() -> None:
    snap = _snapshot()
    picks = _decision_gw(xi_pts=[2] * 11, captain_index=0)
    # Inflate season points by repeating the same GW rows under many GWs —
    # instead pass a high realised via hindsight path: just check wiring.
    score = score_gameweek_decisions(picks, gw=1, rank_snapshot=snap)
    assert score.points_at_rank is not None
    assert score.percentile is not None
    eq = equivalent_percentile(score.season_points_to_date, snap)
    assert score.points_at_rank == eq.rank
    assert score.percentile == eq.percentile


def test_score_gameweeks_frame_validates() -> None:
    gw1 = _decision_gw(xi_pts=[2] * 11, captain_index=0)
    gw2 = gw1.with_columns(pl.lit(2).cast(pl.Int64).alias("gw"))
    frame = score_gameweeks(pl.concat([gw1, gw2]), n_fixtures_by_gw={2: 12})
    assert frame.height == 2
    assert frame.filter(pl.col("gw") == 2)["gw_kind"][0] == "double"


def test_classify_gameweek_table() -> None:
    assert classify_gameweek(n_fixtures=10) == "normal"
    assert classify_gameweek(n_fixtures=0) == "blank"
    assert classify_gameweek(n_fixtures=12) == "double"
    assert classify_gameweek(all_zero_minutes=True) == "blank"


def test_plan_ev_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        plan_expected_points(pl.DataFrame({"gw": [1]}))


def _snapshot() -> OverallRankSnapshot:
    payload = json.loads(
        (_FIXTURES / "points_at_rank.json").read_text(encoding="utf-8")
    )
    return OverallRankSnapshot.from_payload(payload)


def _decision_gw(
    *,
    xi_pts: list[int],
    captain_index: int,
    xi_xpts: list[float] | None = None,
    xi_mins: list[int] | None = None,
    vice_index: int | None = None,
    hit: float = 0.0,
    bench_mins: list[int] | None = None,
    n_fixtures: int | None = None,
) -> pl.DataFrame:
    del n_fixtures  # callers pass n_fixtures into score_*, not the frame
    positions = [
        "GK",
        "DEF",
        "DEF",
        "DEF",
        "MID",
        "MID",
        "MID",
        "MID",
        "FWD",
        "FWD",
        "FWD",
    ]
    mins = xi_mins or [90] * 11
    xpts = xi_xpts or [float(p) for p in xi_pts]
    vice = (captain_index + 1) % 11 if vice_index is None else vice_index
    rows: list[dict[str, object]] = []
    for index, (pts, position, minutes, xp) in enumerate(
        zip(xi_pts, positions, mins, xpts, strict=True)
    ):
        rows.append(
            _row(
                key=index + 1,
                position=position,
                in_xi=True,
                captain=index == captain_index,
                vice=index == vice,
                pts=pts,
                mins=minutes,
                xpts=xp,
                hit=hit,
                team_key=(index % 7) + 1,
            )
        )
    b_mins = bench_mins or [0, 0, 0, 0]
    rows.extend(
        [
            _row(
                key=12,
                position="GK",
                in_xi=False,
                captain=False,
                vice=False,
                pts=0,
                mins=b_mins[0],
                xpts=0.0,
                hit=hit,
                team_key=8,
            ),
            _row(
                key=13,
                position="DEF",
                in_xi=False,
                captain=False,
                vice=False,
                pts=0,
                mins=b_mins[1],
                xpts=0.0,
                hit=hit,
                team_key=8,
            ),
            _row(
                key=14,
                position="DEF",
                in_xi=False,
                captain=False,
                vice=False,
                pts=0,
                mins=b_mins[2],
                xpts=0.0,
                hit=hit,
                team_key=9,
            ),
            _row(
                key=15,
                position="MID",
                in_xi=False,
                captain=False,
                vice=False,
                pts=0,
                mins=b_mins[3],
                xpts=0.0,
                hit=hit,
                team_key=10,
            ),
        ]
    )
    return pl.DataFrame(rows)


def _row(
    *,
    key: int,
    position: str,
    in_xi: bool,
    captain: bool,
    vice: bool,
    pts: int,
    mins: int,
    xpts: float,
    hit: float,
    team_key: int,
) -> dict[str, object]:
    return {
        "model_id": "policy",
        "season": "2024-25",
        "gw": 1,
        "player_key": key,
        "position": position,
        "in_xi": in_xi,
        "captain": captain,
        "vice": vice,
        "bought": False,
        "sold": False,
        "xpts": xpts,
        "actual_pts": pts,
        "actual_minutes": mins,
        "hit": float(hit),
        "free_transfers": 1,
        "team_key": team_key,
        "price": 50,
    }
