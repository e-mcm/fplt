"""Shared scenario bundles and sample-average lineup scoring."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fplt.optimise.realised import score_gameweek
from fplt.optimise.scenarios import (
    ScenarioBundle,
    default_bench_slots,
    independent_scenarios,
    mean_realised_lineup,
    realised_lineup_scores,
    scenario_bundle_from_frame,
    scenario_bundle_from_samples,
    scenario_outcomes_from_samples,
)


def test_scenario_bundle_from_frame_pivots_grid() -> None:
    frame = pl.DataFrame(
        {
            "player_key": [2, 2, 1, 1],
            "sim": [1, 0, 1, 0],
            "points": [5.0, 3.0, 4.0, 2.0],
            "minutes": [90, 0, 90, 90],
        }
    )
    bundle = scenario_bundle_from_frame(frame)
    assert bundle.player_keys.tolist() == [1, 2]
    assert bundle.n_sims == 2
    np.testing.assert_allclose(bundle.points, [[2.0, 4.0], [3.0, 5.0]])
    np.testing.assert_array_equal(bundle.minutes, [[90, 90], [0, 90]])


def test_scenario_bundle_from_samples_sums_dgw() -> None:
    """DGW fixtures that share sim indices aggregate to player–GW SAA grain."""
    samples = pl.DataFrame(
        {
            "player_key": [1, 1, 1, 1, 2, 2],
            "fixture_id": [10, 10, 11, 11, 10, 10],
            "target_season": ["2024-25"] * 6,
            "target_gw": [1] * 6,
            "sim": [0, 1, 0, 1, 0, 1],
            "points": [3.0, 4.0, 2.0, 1.0, 5.0, 6.0],
            "minutes": [90, 0, 45, 90, 90, 90],
        }
    )
    outcomes = scenario_outcomes_from_samples(samples, season="2024-25", gw=1)
    assert outcomes.height == 4  # 2 players × 2 sims
    bundle = scenario_bundle_from_samples(samples, season="2024-25", gw=1)
    assert bundle.player_keys.tolist() == [1, 2]
    # Player 1 sim0: 3+2; sim1: 4+1. Player 2: 5 / 6.
    np.testing.assert_allclose(bundle.points, [[5.0, 5.0], [5.0, 6.0]])
    np.testing.assert_array_equal(bundle.minutes, [[135, 90], [90, 90]])


def test_incomplete_scenario_grid_is_rejected() -> None:
    frame = pl.DataFrame(
        {
            "player_key": [1, 1, 2],
            "sim": [0, 1, 0],
            "points": [1.0, 2.0, 3.0],
            "minutes": [90, 90, 0],
        }
    )
    with pytest.raises(ValueError, match="full player×sim"):
        scenario_bundle_from_frame(frame)


def test_bench_slot_changes_which_sub_comes_on() -> None:
    """Regression: autosub must honour explicit bench order (M2)."""
    base = _fifteen(
        xi_mins=[90] * 10 + [0],
        xi_pts=[1] * 10 + [0],
        captain=2,
        vice=3,
    )
    # FWD 11 blanks. Two outfield bench options with different points.
    high = base.with_columns(
        pl.when(pl.col("player_key") == 13)
        .then(1)
        .when(pl.col("player_key") == 14)
        .then(2)
        .when(pl.col("player_key") == 15)
        .then(3)
        .when(pl.col("player_key") == 12)
        .then(4)
        .otherwise(None)
        .alias("bench_slot"),
        pl.when(pl.col("player_key") == 13)
        .then(9)
        .when(pl.col("player_key") == 14)
        .then(2)
        .otherwise(pl.col("actual_pts"))
        .alias("actual_pts"),
        pl.when(pl.col("player_key").is_in([13, 14]))
        .then(90)
        .otherwise(pl.col("actual_minutes"))
        .alias("actual_minutes"),
    )
    low_first = high.with_columns(
        pl.when(pl.col("player_key") == 14)
        .then(1)
        .when(pl.col("player_key") == 13)
        .then(2)
        .otherwise(pl.col("bench_slot"))
        .alias("bench_slot")
    )
    assert score_gameweek(high, hit=0) > score_gameweek(low_first, hit=0)


def test_mean_realised_lineup_averages_autosub_scenarios() -> None:
    picks = default_bench_slots(
        _fifteen(
            xi_mins=[90] * 11,
            xi_pts=[2] * 11,
            captain=2,
            vice=3,
        )
    )
    # Two sims: sim0 everyone plays; sim1 FWD 11 blanks and bench MID 13 plays.
    keys = picks.sort("player_key")["player_key"].to_list()
    n = len(keys)
    points = np.full((n, 2), 2.0)
    minutes = np.full((n, 2), 90, dtype=np.int64)
    blank = keys.index(11)
    bench_mid = keys.index(13)
    points[blank, 1] = 0.0
    minutes[blank, 1] = 0
    points[bench_mid, 1] = 8.0
    minutes[bench_mid, 1] = 90
    for key in (12, 14, 15):
        idx = keys.index(key)
        points[idx, 1] = 0.0
        minutes[idx, 1] = 0
    bundle = ScenarioBundle(
        player_keys=np.asarray(keys, dtype=np.int64),
        points=points,
        minutes=minutes,
    )
    scores = realised_lineup_scores(picks, bundle)
    assert scores.shape == (2,)
    mean = mean_realised_lineup(picks, bundle)
    assert mean == pytest.approx(float(scores.mean()))
    assert scores[1] > scores[0]  # autosub 8 replaces blanked 2


def test_independent_scenarios_respect_seed_and_shape() -> None:
    players = pl.DataFrame(
        {
            "player_key": [10, 20],
            "xpts": [4.0, 6.0],
            "p_played": [1.0, 0.5],
        }
    )
    a = independent_scenarios(players, n_sims=8, rng=np.random.default_rng(3))
    b = independent_scenarios(players, n_sims=8, rng=np.random.default_rng(3))
    np.testing.assert_array_equal(a.points, b.points)
    assert a.n_sims == 8
    assert a.player_keys.tolist() == [10, 20]
    with pytest.raises(ValueError, match="positive"):
        independent_scenarios(players, n_sims=0, rng=np.random.default_rng(1))


def _fifteen(
    *,
    xi_mins: list[int],
    xi_pts: list[int],
    captain: int,
    vice: int,
) -> pl.DataFrame:
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
        "GK",
        "MID",
        "DEF",
        "FWD",
    ]
    rows: list[dict[str, object]] = []
    for index, position in enumerate(positions, start=1):
        in_xi = index <= 11
        pts = xi_pts[index - 1] if in_xi else 0
        mins = xi_mins[index - 1] if in_xi else 0
        rows.append(
            {
                "player_key": index,
                "position": position,
                "in_xi": in_xi,
                "captain": index == captain,
                "vice": index == vice,
                "actual_pts": pts,
                "actual_minutes": mins,
            }
        )
    return pl.DataFrame(rows)
