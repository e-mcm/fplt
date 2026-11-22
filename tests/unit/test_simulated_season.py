"""Simulated-season decision eval: CRN paths and paired SEs on tiny seasons."""

from __future__ import annotations

import math

import numpy as np
import polars as pl
import pytest

from fplt.backtest.simulated_season import (
    DecisionPolicy,
    draw_truth_paths,
    format_simulated_season_report,
    mean_policy_totals,
    paired_policy_se,
    path_seed,
    policy_diff_board,
    run_simulated_season_eval,
)
from fplt.optimise.state import SquadState
from fplt.schemas.simulated_season import (
    PolicyDiffSESchema,
    SimulatedPathActualsSchema,
    SimulatedSeasonPathSchema,
)


def test_path_seed_is_stable_and_varies_by_path() -> None:
    assert path_seed(7, 0) == path_seed(7, 0)
    assert path_seed(7, 0) != path_seed(7, 1)
    assert path_seed(7, 0) != path_seed(8, 0)


def test_draw_truth_paths_are_reproducible_and_labelled() -> None:
    rates = _truth_rates()
    first = draw_truth_paths(rates, n_paths=3, seed=11)
    second = draw_truth_paths(rates, n_paths=3, seed=11)
    SimulatedPathActualsSchema.validate(first)
    assert first.equals(second)
    assert sorted(first["path_id"].unique().to_list()) == [0, 1, 2]
    # Different seed → different draws on the same rates.
    other = draw_truth_paths(rates, n_paths=3, seed=12)
    assert not first.equals(other)


def test_draw_truth_paths_rejects_non_positive_n() -> None:
    with pytest.raises(ValueError, match="n_paths"):
        draw_truth_paths(_truth_rates(), n_paths=0, seed=1)


def test_crn_policies_share_outcome_noise() -> None:
    """Two policies on the same paths must use identical actuals (CRN)."""
    candidates, rates, preds_a, preds_b = _tiny_season()
    report = run_simulated_season_eval(
        candidates,
        [
            DecisionPolicy("policy_a", preds_a),
            DecisionPolicy("policy_b", preds_b),
        ],
        _state(),
        season="2024-25",
        gameweeks=[1, 2],
        horizon=1,
        n_paths=4,
        seed=42,
        truth_rates=rates,
    )
    SimulatedSeasonPathSchema.validate(report.paths)
    assert report.n_paths == 4
    # Paired board has one season_points row for the unordered pair.
    assert report.diffs.height == 1
    PolicyDiffSESchema.validate(report.diffs)
    row = report.diffs.row(0, named=True)
    assert row["metric"] == "season_points"
    assert row["n_paths"] == 4
    assert row["se"] >= 0.0
    # Re-running with the same seed regenerates identical path totals.
    again = run_simulated_season_eval(
        candidates,
        [
            DecisionPolicy("policy_a", preds_a),
            DecisionPolicy("policy_b", preds_b),
        ],
        _state(),
        season="2024-25",
        gameweeks=[1, 2],
        horizon=1,
        n_paths=4,
        seed=42,
        truth_rates=rates,
    )
    assert report.paths.equals(again.paths)
    assert report.diffs.equals(again.diffs)


def test_paired_se_matches_manual_formula() -> None:
    paths = pl.DataFrame(
        {
            "path_id": [0, 1, 2, 0, 1, 2],
            "policy_id": ["control", "control", "control", "treat", "treat", "treat"],
            "season": ["2024-25"] * 6,
            "total_points": [10, 12, 14, 11, 15, 16],
            "gameweeks": [2] * 6,
            "attainable_points": [None] * 6,
            "share_of_attainable": [None] * 6,
        }
    )
    result = paired_policy_se(
        paths, control_id="control", treatment_id="treat", metric="season_points"
    )
    diffs = np.array([1.0, 3.0, 2.0])
    expected_mean = float(diffs.mean())
    expected_sd = float(np.std(diffs, ddof=1))
    expected_se = expected_sd / math.sqrt(3)
    row = result.row(0, named=True)
    assert row["mean_diff"] == pytest.approx(expected_mean)
    assert row["sd"] == pytest.approx(expected_sd)
    assert row["se"] == pytest.approx(expected_se)
    assert row["n_paths"] == 3


def test_predrawn_paths_and_attainable_share_se() -> None:
    candidates, rates, preds_a, preds_b = _tiny_season()
    drawn = draw_truth_paths(rates, n_paths=3, seed=99)
    report = run_simulated_season_eval(
        candidates,
        [
            DecisionPolicy("policy_a", preds_a),
            DecisionPolicy("policy_b", preds_b),
        ],
        _state(),
        season="2024-25",
        gameweeks=[1, 2],
        horizon=1,
        path_actuals=drawn,
        with_attainable=True,
    )
    assert report.with_attainable
    assert set(report.diffs["metric"].to_list()) == {
        "season_points",
        "share_of_attainable",
    }
    for row in report.paths.iter_rows(named=True):
        assert row["attainable_points"] is not None
        assert row["share_of_attainable"] is not None
        assert 0.0 <= float(row["share_of_attainable"]) <= 1.0 + 1e-9
    text = format_simulated_season_report(report)
    assert "share_of_attainable" in text
    assert "mean_diff" in text


def test_identical_policies_have_zero_diff() -> None:
    candidates, rates, preds_a, _ = _tiny_season()
    report = run_simulated_season_eval(
        candidates,
        [
            DecisionPolicy("left", preds_a),
            DecisionPolicy("right", preds_a),
        ],
        _state(),
        season="2024-25",
        gameweeks=[1],
        horizon=1,
        n_paths=5,
        seed=3,
        truth_rates=rates,
    )
    row = report.diffs.row(0, named=True)
    assert row["mean_diff"] == pytest.approx(0.0)
    assert row["se"] == pytest.approx(0.0)
    assert row["sd"] == pytest.approx(0.0)
    means = mean_policy_totals(report.paths)
    assert means["left"] == means["right"]


def test_rejects_both_or_neither_draw_source() -> None:
    candidates, rates, preds_a, preds_b = _tiny_season()
    policies = [
        DecisionPolicy("policy_a", preds_a),
        DecisionPolicy("policy_b", preds_b),
    ]
    with pytest.raises(ValueError, match="truth_rates"):
        run_simulated_season_eval(
            candidates,
            policies,
            _state(),
            season="2024-25",
            gameweeks=[1],
            horizon=1,
        )
    drawn = draw_truth_paths(rates, n_paths=1, seed=1)
    with pytest.raises(ValueError, match="not both"):
        run_simulated_season_eval(
            candidates,
            policies,
            _state(),
            season="2024-25",
            gameweeks=[1],
            horizon=1,
            truth_rates=rates,
            path_actuals=drawn,
            n_paths=1,
            seed=1,
        )


def test_policy_diff_board_explicit_pairs() -> None:
    paths = pl.DataFrame(
        {
            "path_id": [0, 0, 0],
            "policy_id": ["a", "b", "c"],
            "season": ["2024-25"] * 3,
            "total_points": [1, 2, 4],
            "gameweeks": [1] * 3,
            "attainable_points": [None] * 3,
            "share_of_attainable": [None] * 3,
        }
    )
    board = policy_diff_board(paths, pairs=[("a", "c")])
    assert board.height == 1
    assert board["control_id"][0] == "a"
    assert board["treatment_id"][0] == "c"
    assert board["mean_diff"][0] == pytest.approx(3.0)


def test_duplicate_policy_id_raises() -> None:
    candidates, rates, preds_a, _ = _tiny_season()
    with pytest.raises(ValueError, match="duplicate policy_id"):
        run_simulated_season_eval(
            candidates,
            [
                DecisionPolicy("same", preds_a),
                DecisionPolicy("same", preds_a),
            ],
            _state(),
            season="2024-25",
            gameweeks=[1],
            horizon=1,
            n_paths=1,
            seed=1,
            truth_rates=rates,
        )


def _state() -> SquadState:
    return SquadState.model_validate(
        {
            "bank": 1000,
            "free_transfers": 1,
            "ft_bank_cap": 2,
            "hit_cost": 4,
            "decay": 1.0,
            "bench_weight": 0.0,
            "owned": [],
        }
    )


def _truth_rates() -> pl.DataFrame:
    _, rates, _, _ = _tiny_season()
    return rates


def _tiny_season() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Legal 15-from-30 pool, two GWs, two prediction policies, truth rates."""
    players: list[dict[str, object]] = []
    player = 1
    for club in range(1, 7):
        for position in ("GK", "DEF", "DEF", "MID", "FWD"):
            players.append(
                {
                    "player_key": player,
                    "position": position,
                    "team_key": club,
                    "price": 40 + club,
                    "base": float(club + (2 if position == "MID" else 0)),
                }
            )
            player += 1
    pool = pl.DataFrame(players)
    candidates = pl.concat(
        [
            pool.select("player_key", "position", "team_key", "price").with_columns(
                pl.lit(gw).alias("gw")
            )
            for gw in (1, 2)
        ]
    )
    rate_rows: list[dict[str, object]] = []
    preds_a: list[dict[str, object]] = []
    preds_b: list[dict[str, object]] = []
    for row in pool.iter_rows(named=True):
        key = int(row["player_key"])
        base = float(row["base"])
        for gw in (1, 2):
            rate_rows.append(
                {
                    "gw": gw,
                    "player_key": key,
                    "rate": max(0.5, base * 0.4),
                    "minutes_rate": 90.0,
                }
            )
            preds_a.append(
                {
                    "as_of_gw": gw,
                    "target_gw": gw,
                    "player_key": key,
                    "xpts": base,
                }
            )
            # Shift value so drafts can diverge under the same actuals.
            preds_b.append(
                {
                    "as_of_gw": gw,
                    "target_gw": gw,
                    "player_key": key,
                    "xpts": base + 0.25 * (key % 5),
                }
            )
    return (
        candidates,
        pl.DataFrame(rate_rows),
        pl.DataFrame(preds_a),
        pl.DataFrame(preds_b),
    )
