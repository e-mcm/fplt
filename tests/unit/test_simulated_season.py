"""Simulated-season decision eval: CRN paths and paired SEs on tiny seasons."""

from __future__ import annotations

import math
from unittest.mock import patch

import numpy as np
import polars as pl
import pytest

from fplt.backtest.simulated_season import (
    DecisionPolicy,
    chip_arm_control_pairs,
    chip_arm_policies,
    draw_truth_paths,
    format_simulated_season_report,
    mean_policy_totals,
    paired_policy_se,
    path_actuals_from_joint_sims,
    path_seed,
    policy_diff_board,
    run_simulated_season_eval,
    truth_rates_from_actuals,
)
from fplt.optimise.attainable import OraclePin
from fplt.optimise.chip_windows import ChipUseWindows
from fplt.optimise.state import SquadState
from fplt.schemas.simulated_season import (
    JointTruthSimSchema,
    PolicyDiffSESchema,
    SimulatedPathActualsSchema,
    SimulatedSeasonPathSchema,
    TruthRateSchema,
)


def test_path_seed_is_stable_and_varies_by_path() -> None:
    assert path_seed(7, 0) == path_seed(7, 0)
    assert path_seed(7, 0) != path_seed(7, 1)
    assert path_seed(7, 0) != path_seed(8, 0)


def test_path_actuals_from_joint_sims_maps_sim_to_path() -> None:
    joint = pl.DataFrame(
        {
            "sim": [0, 0, 1, 1],
            "gw": [1, 1, 1, 1],
            "player_key": [10, 20, 10, 20],
            "points": [4.4, 1.2, 7.6, 0.1],
            "minutes": [90, 45, 90, 0],
        }
    )
    JointTruthSimSchema.validate(joint)
    paths = path_actuals_from_joint_sims(joint)
    SimulatedPathActualsSchema.validate(paths)
    assert paths.sort("path_id", "player_key")["actual_pts"].to_list() == [4, 1, 8, 0]
    assert paths["path_id"].unique().to_list() == [0, 1]
    with pytest.raises(ValueError, match="non-empty"):
        path_actuals_from_joint_sims(joint.clear())


def test_with_attainable_rejects_divergent_horizons_without_pin() -> None:
    candidates, rates, preds_a, preds_b = _tiny_season()
    policies = [
        DecisionPolicy("h3", preds_a, horizon=3),
        DecisionPolicy("h8", preds_b, horizon=8),
    ]
    with pytest.raises(ValueError, match="OraclePin"):
        run_simulated_season_eval(
            candidates,
            policies,
            _state(),
            season="2024-25",
            gameweeks=[1, 2],
            horizon=1,
            n_paths=1,
            seed=0,
            truth_rates=rates,
            with_attainable=True,
        )


def test_oracle_pin_forwarded_to_hindsight_oracle() -> None:
    """Share SE uses one pinned H/decay/chip_arm for every policy on a path."""
    candidates, rates, preds_a, preds_b = _tiny_season()
    seen: list[dict[str, object]] = []
    pin = OraclePin(horizon=5, decay=0.85, chip_arm="off", force_play=False)
    empty_decisions = pl.DataFrame(
        schema={
            "model_id": pl.Utf8(),
            "season": pl.Utf8(),
            "gw": pl.Int64(),
            "player_key": pl.Int64(),
            "position": pl.Utf8(),
            "in_xi": pl.Boolean(),
            "captain": pl.Boolean(),
            "vice": pl.Boolean(),
            "actual_pts": pl.Int64(),
            "actual_minutes": pl.Int64(),
            "hit": pl.Int64(),
        }
    )
    summary = pl.DataFrame(
        {
            "model_id": ["x"],
            "season": ["2024-25"],
            "total_points": [10],
            "gameweeks": [1],
            "per_gw": [[10]],
        }
    )

    def _oracle_spy(*args: object, **kwargs: object) -> pl.DataFrame:
        state = args[2]
        assert isinstance(state, SquadState)
        seen.append(
            {
                "horizon": kwargs["horizon"],
                "chip_arm": kwargs["chip_arm"],
                "force_play": kwargs["force_play"],
                "decay": float(state.decay),
            }
        )
        return empty_decisions

    with (
        patch(
            "fplt.backtest.simulated_season.simulate_hindsight_oracle",
            side_effect=_oracle_spy,
        ),
        patch(
            "fplt.backtest.simulated_season.simulate_decisions",
            return_value=empty_decisions,
        ),
        patch(
            "fplt.backtest.simulated_season.summarise_decisions",
            return_value=summary,
        ),
    ):
        report = run_simulated_season_eval(
            candidates,
            [
                DecisionPolicy("policy_a", preds_a, horizon=3),
                DecisionPolicy("policy_b", preds_b, horizon=8),
            ],
            _state(),
            season="2024-25",
            gameweeks=[1, 2],
            horizon=1,
            n_paths=2,
            seed=0,
            truth_rates=rates,
            with_attainable=True,
            oracle_pin=pin,
        )
    assert report.oracle_pin == pin
    assert len(seen) == 2  # one oracle call per path, not per policy
    assert all(row == seen[0] for row in seen)
    assert seen[0]["horizon"] == 5
    assert seen[0]["decay"] == pytest.approx(0.85)
    assert seen[0]["chip_arm"] == "off"
    assert seen[0]["force_play"] is False
    text = format_simulated_season_report(report)
    assert "oracle pin: H=5" in text
    assert "decay=0.85" in text


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


def test_truth_rates_from_actuals_and_chip_arm_policies() -> None:
    actuals = pl.DataFrame(
        {
            "gw": [1, 1, 2],
            "player_key": [10, 11, 10],
            "actual_pts": [4, 0, 6],
            "actual_minutes": [90, 0, 75],
        }
    )
    rates = truth_rates_from_actuals(actuals)
    TruthRateSchema.validate(rates)
    player_ten = rates.filter(pl.col("player_key") == 10).sort("gw")["rate"].to_list()
    assert player_ten == [4.0, 6.0]

    preds = pl.DataFrame(
        {
            "as_of_gw": [1],
            "target_gw": [1],
            "player_key": [10],
            "xpts": [1.0],
        }
    )
    policies = chip_arm_policies(
        preds,
        base_model_id="toy",
        arms=("off", "optimised"),
        chip_rules_season="2025-26",
    )
    assert [p.policy_id for p in policies] == ["toy", "toy::chip_optimised"]
    assert [p.chip_arm for p in policies] == ["off", "optimised"]
    assert all(p.force_play for p in policies)
    assert chip_arm_control_pairs(policies) == [("toy", "toy::chip_optimised")]
    off_policies = chip_arm_policies(
        preds,
        base_model_id="toy",
        arms=("optimised",),
        force_play=False,
    )
    assert off_policies[0].force_play is False


def test_chip_arms_require_fixtures() -> None:
    candidates, rates, preds_a, preds_b = _tiny_season()
    policies = [
        DecisionPolicy("off", preds_a, chip_arm="off"),
        DecisionPolicy(
            "rb",
            preds_b,
            chip_arm="rule_based",
            chip_rules_season="2025-26",
        ),
    ]
    with pytest.raises(ValueError, match="fixtures"):
        run_simulated_season_eval(
            candidates,
            policies,
            _state(),
            season="2024-25",
            gameweeks=[1],
            horizon=1,
            n_paths=1,
            seed=1,
            truth_rates=rates,
        )


def test_chip_arm_kwargs_reach_simulate_decisions() -> None:
    """Wiring: chip_arm / windows / fixtures are forwarded (mocked sim)."""
    candidates, rates, preds_a, _ = _tiny_season()
    fixtures = pl.DataFrame(
        {
            "season": ["2024-25"],
            "gw": [1],
            "team_h_key": [1],
            "team_a_key": [2],
        }
    )
    windows = ChipUseWindows(
        allowed={"bboost": frozenset({1})},
        forced={"bboost": frozenset({1})},
    )
    empty_decisions = pl.DataFrame(
        schema={
            "model_id": pl.Utf8(),
            "season": pl.Utf8(),
            "gw": pl.Int64(),
            "player_key": pl.Int64(),
            "position": pl.Utf8(),
            "in_xi": pl.Boolean(),
            "captain": pl.Boolean(),
            "vice": pl.Boolean(),
            "bought": pl.Boolean(),
            "sold": pl.Boolean(),
            "xpts": pl.Float64(),
            "actual_pts": pl.Int64(),
            "actual_minutes": pl.Int64(),
            "hit": pl.Float64(),
            "free_transfers": pl.Int64(),
            "chip": pl.Utf8(),
        }
    )
    empty_summary = pl.DataFrame(
        {
            "model_id": pl.Series([], dtype=pl.Utf8()),
            "season": pl.Series([], dtype=pl.Utf8()),
            "total_points": pl.Series([], dtype=pl.Int64()),
            "gameweeks": pl.Series([], dtype=pl.Int64()),
            "per_gw": pl.Series([], dtype=pl.Float64()),
        }
    )
    policies = chip_arm_policies(
        preds_a,
        base_model_id="m",
        arms=("off", "rule_based", "optimised"),
        chip_rules_season="2025-26",
        chip_windows={"rule_based": windows},
    )
    with (
        patch(
            "fplt.backtest.simulated_season.simulate_decisions",
            return_value=empty_decisions,
        ) as sim,
        patch(
            "fplt.backtest.simulated_season.summarise_decisions",
            return_value=empty_summary,
        ),
    ):
        report = run_simulated_season_eval(
            candidates,
            policies,
            _state(),
            season="2024-25",
            gameweeks=[1],
            horizon=1,
            n_paths=2,
            seed=1,
            truth_rates=rates,
            fixtures=fixtures,
        )
    assert report.n_paths == 2
    assert sim.call_count == 6  # 2 paths × 3 arms
    arms_seen = {call.kwargs["chip_arm"] for call in sim.call_args_list}
    assert arms_seen == {"off", "rule_based", "optimised"}
    rule_calls = [
        call for call in sim.call_args_list if call.kwargs["chip_arm"] == "rule_based"
    ]
    assert rule_calls[0].kwargs["chip_windows"] is windows
    assert rule_calls[0].kwargs["chip_rules_season"] == "2025-26"
    assert rule_calls[0].kwargs["fixtures"] is fixtures
    # Default pairs: each non-off arm vs chips-off control.
    assert set(
        zip(
            report.diffs["control_id"].to_list(),
            report.diffs["treatment_id"].to_list(),
            strict=True,
        )
    ) == {("m", "m::chip_rule_based"), ("m", "m::chip_optimised")}


def test_chip_arms_crn_on_toy_season() -> None:
    """End-to-end: three chip arms share CRN paths on a tiny season."""
    candidates, predictions, actuals, fixtures = _chip_toy_season()
    rates = truth_rates_from_actuals(actuals)
    drawn = draw_truth_paths(rates, n_paths=2, seed=7)
    policies = chip_arm_policies(
        predictions,
        base_model_id="toy",
        arms=("off", "rule_based", "optimised"),
        chip_rules_season="2025-26",
    )
    report = run_simulated_season_eval(
        candidates,
        policies,
        _state(),
        season="2024-25",
        gameweeks=[1, 2, 3],
        horizon=1,
        path_actuals=drawn,
        fixtures=fixtures,
    )
    SimulatedSeasonPathSchema.validate(report.paths)
    PolicyDiffSESchema.validate(report.diffs)
    assert set(report.paths["policy_id"].to_list()) == {
        "toy",
        "toy::chip_rule_based",
        "toy::chip_optimised",
    }
    assert report.diffs.height == 2
    again = run_simulated_season_eval(
        candidates,
        policies,
        _state(),
        season="2024-25",
        gameweeks=[1, 2, 3],
        horizon=1,
        path_actuals=drawn,
        fixtures=fixtures,
    )
    assert report.paths.equals(again.paths)
    assert report.diffs.equals(again.diffs)
    text = format_simulated_season_report(report)
    assert "toy::chip_optimised" in text


def test_run_chip_arm_sim_season_loads_ledger_and_delegates(tmp_path) -> None:
    """Data-dir entry point builds rates/policies and calls the eval harness."""
    from fplt.backtest.simulated_season import (
        SimulatedSeasonReport,
        run_chip_arm_sim_season,
    )

    fake_report = SimulatedSeasonReport(
        paths=_empty_season_paths_frame(),
        diffs=_empty_diffs_frame(),
        n_paths=2,
        seed=3,
        with_attainable=False,
    )
    preds = pl.DataFrame(
        {
            "as_of_gw": [1],
            "target_gw": [1],
            "player_key": [1],
            "xpts": [2.0],
        }
    )
    actuals = pl.DataFrame(
        {
            "gw": [1, 2],
            "player_key": [1, 1],
            "actual_pts": [3, 4],
            "actual_minutes": [90, 90],
        }
    )
    fixtures = pl.DataFrame(
        {
            "season": ["2025-26"],
            "gw": [1],
            "team_h_key": [1],
            "team_a_key": [2],
        }
    )
    pool = pl.DataFrame(
        {
            "player_key": [1],
            "position": ["MID"],
            "team_key": [1],
            "price": [50],
        }
    )

    class _Spec:
        model_id = "mid"

    class _Store:
        def read(self, _name: str) -> pl.DataFrame:
            return fixtures

    with (
        patch(
            "fplt.backtest.simulated_season.load_model_specs",
            return_value={"rolling_mean_v1": _Spec()},
        ),
        patch(
            "fplt.backtest.simulated_season.load_squad_state",
            return_value=_state(),
        ),
        patch("fplt.backtest.simulated_season.DataStore", return_value=_Store()),
        patch("fplt.backtest.simulated_season.PITData") as pit_cls,
        patch(
            "fplt.backtest.simulated_season.candidates_as_of",
            return_value=pool,
        ),
        patch(
            "fplt.backtest.simulated_season.read_predictions",
            return_value=pl.DataFrame(),
        ),
        patch(
            "fplt.backtest.simulated_season.prediction_rows_from_ledger",
            return_value=preds,
        ),
        patch(
            "fplt.backtest.simulated_season.actual_rows_from_ledger",
            return_value=actuals,
        ),
        patch(
            "fplt.backtest.simulated_season.run_simulated_season_eval",
            return_value=fake_report,
        ) as eval_mock,
    ):
        pit_cls.return_value.view.return_value = object()
        report = run_chip_arm_sim_season(
            tmp_path,
            "rolling_mean_v1",
            "2025-26",
            1,
            2,
            1,
            arms=("off", "optimised"),
            n_paths=2,
            seed=3,
        )
    assert report is fake_report
    kwargs = eval_mock.call_args.kwargs
    assert kwargs["season"] == "2025-26"
    assert kwargs["gameweeks"] == [1, 2]
    assert kwargs["n_paths"] == 2
    assert kwargs["seed"] == 3
    assert kwargs["fixtures"].equals(fixtures)
    policies = eval_mock.call_args.args[1]
    assert [p.chip_arm for p in policies] == ["off", "optimised"]
    assert all(p.force_play for p in policies)


def test_run_chip_arm_sim_season_forwards_force_play(tmp_path) -> None:
    """``force_play=False`` lands on every chip-arm DecisionPolicy."""
    from fplt.backtest.simulated_season import (
        SimulatedSeasonReport,
        run_chip_arm_sim_season,
    )

    fake_report = SimulatedSeasonReport(
        paths=_empty_season_paths_frame(),
        diffs=_empty_diffs_frame(),
        n_paths=1,
        seed=0,
        with_attainable=False,
    )
    preds = pl.DataFrame(
        {
            "as_of_gw": [1],
            "target_gw": [1],
            "player_key": [1],
            "xpts": [2.0],
        }
    )
    actuals = pl.DataFrame(
        {
            "gw": [1],
            "player_key": [1],
            "actual_pts": [3],
            "actual_minutes": [90],
        }
    )

    class _Spec:
        model_id = "mid"

    class _Store:
        def read(self, _name: str) -> pl.DataFrame:
            return pl.DataFrame(
                {
                    "season": ["2025-26"],
                    "gw": [1],
                    "team_h_key": [1],
                    "team_a_key": [2],
                }
            )

    with (
        patch(
            "fplt.backtest.simulated_season.load_model_specs",
            return_value={"rolling_mean_v1": _Spec()},
        ),
        patch(
            "fplt.backtest.simulated_season.load_squad_state",
            return_value=_state(),
        ),
        patch("fplt.backtest.simulated_season.DataStore", return_value=_Store()),
        patch("fplt.backtest.simulated_season.PITData") as pit_cls,
        patch(
            "fplt.backtest.simulated_season.candidates_as_of",
            return_value=pl.DataFrame(
                {
                    "player_key": [1],
                    "position": ["MID"],
                    "team_key": [1],
                    "price": [50],
                }
            ),
        ),
        patch(
            "fplt.backtest.simulated_season.read_predictions",
            return_value=pl.DataFrame(),
        ),
        patch(
            "fplt.backtest.simulated_season.prediction_rows_from_ledger",
            return_value=preds,
        ),
        patch(
            "fplt.backtest.simulated_season.actual_rows_from_ledger",
            return_value=actuals,
        ),
        patch(
            "fplt.backtest.simulated_season.run_simulated_season_eval",
            return_value=fake_report,
        ) as eval_mock,
    ):
        pit_cls.return_value.view.return_value = object()
        run_chip_arm_sim_season(
            tmp_path,
            "rolling_mean_v1",
            "2025-26",
            1,
            1,
            3,
            arms=("optimised",),
            n_paths=1,
            seed=0,
            force_play=False,
        )
    policies = eval_mock.call_args.args[1]
    assert len(policies) == 1
    assert policies[0].force_play is False
    assert policies[0].chip_arm == "optimised"


def test_run_chip_arm_sim_season_rejects_empty_actuals(tmp_path) -> None:
    from fplt.backtest.simulated_season import run_chip_arm_sim_season

    class _Spec:
        model_id = "mid"

    with (
        patch(
            "fplt.backtest.simulated_season.load_model_specs",
            return_value={"rolling_mean_v1": _Spec()},
        ),
        patch(
            "fplt.backtest.simulated_season.load_squad_state",
            return_value=_state(),
        ),
        patch("fplt.backtest.simulated_season.DataStore"),
        patch("fplt.backtest.simulated_season.PITData") as pit_cls,
        patch(
            "fplt.backtest.simulated_season.candidates_as_of",
            return_value=pl.DataFrame(
                {
                    "player_key": [1],
                    "position": ["MID"],
                    "team_key": [1],
                    "price": [50],
                }
            ),
        ),
        patch(
            "fplt.backtest.simulated_season.read_predictions",
            return_value=pl.DataFrame(),
        ),
        patch(
            "fplt.backtest.simulated_season.prediction_rows_from_ledger",
            return_value=pl.DataFrame(
                schema={
                    "as_of_gw": pl.Int64(),
                    "target_gw": pl.Int64(),
                    "player_key": pl.Int64(),
                    "xpts": pl.Float64(),
                }
            ),
        ),
        patch(
            "fplt.backtest.simulated_season.actual_rows_from_ledger",
            return_value=pl.DataFrame(
                schema={
                    "gw": pl.Int64(),
                    "player_key": pl.Int64(),
                    "actual_pts": pl.Int64(),
                    "actual_minutes": pl.Int64(),
                }
            ),
        ),
    ):
        pit_cls.return_value.view.return_value = object()
        with pytest.raises(ValueError, match="no ledger actuals"):
            run_chip_arm_sim_season(
                tmp_path,
                "rolling_mean_v1",
                "2025-26",
                1,
                2,
                1,
                arms=("off",),
                n_paths=1,
                seed=0,
            )


def test_policy_force_play_and_horizon_forwarded_to_simulate() -> None:
    """Per-policy force_play / horizon override the eval defaults (case E)."""
    candidates, rates, preds_a, preds_b = _tiny_season()
    seen: list[dict[str, object]] = []

    def _spy(*_args: object, **kwargs: object) -> pl.DataFrame:
        seen.append(
            {
                "model_id": kwargs["model_id"],
                "horizon": kwargs["horizon"],
                "force_play": kwargs["force_play"],
            }
        )
        return pl.DataFrame(
            schema={
                "model_id": pl.Utf8(),
                "season": pl.Utf8(),
                "gw": pl.Int64(),
                "player_key": pl.Int64(),
                "points": pl.Int64(),
                "hits": pl.Int64(),
                "chip": pl.Utf8(),
            }
        )

    policies = [
        DecisionPolicy(
            "control_h3_force",
            preds_a,
            chip_arm="off",
            force_play=True,
            horizon=3,
        ),
        DecisionPolicy(
            "candidate_h8_noforce",
            preds_b,
            chip_arm="off",
            force_play=False,
            horizon=8,
        ),
    ]
    with patch(
        "fplt.backtest.simulated_season.simulate_decisions",
        side_effect=_spy,
    ):
        report = run_simulated_season_eval(
            candidates,
            policies,
            _state(),
            season="2024-25",
            gameweeks=[1, 2],
            horizon=1,
            n_paths=1,
            seed=0,
            truth_rates=rates,
        )
    assert report.n_paths == 1
    by_id = {row["model_id"]: row for row in seen}
    assert by_id["control_h3_force"]["horizon"] == 3
    assert by_id["control_h3_force"]["force_play"] is True
    assert by_id["candidate_h8_noforce"]["horizon"] == 8
    assert by_id["candidate_h8_noforce"]["force_play"] is False


def test_sim_season_cli_passes_no_force_play(
    tmp_path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.backtest.simulated_season import SimulatedSeasonReport
    from fplt.cli import main
    from fplt.settings import get_settings

    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    fake = SimulatedSeasonReport(
        paths=_empty_season_paths_frame(),
        diffs=_empty_diffs_frame(),
        n_paths=1,
        seed=0,
        with_attainable=False,
    )
    with (
        patch(
            "fplt.cli.run_chip_arm_sim_season",
            return_value=fake,
        ) as run_mock,
    ):
        try:
            main(
                [
                    "sim-season",
                    "--season",
                    "2025-26",
                    "--from-gw",
                    "1",
                    "--to-gw",
                    "2",
                    "--horizon",
                    "3",
                    "--n-paths",
                    "1",
                    "--models",
                    "rolling_mean_v1",
                    "--chip-arms",
                    "optimised",
                    "--no-force-play",
                ]
            )
        finally:
            get_settings.cache_clear()
    assert run_mock.call_args.kwargs["force_play"] is False


def test_sim_season_cli_rejects_empty_arms(tmp_path, monkeypatch) -> None:
    from fplt.cli import main

    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "sim-season",
                "--season",
                "2025-26",
                "--models",
                "rolling_mean_v1",
                "--chip-arms",
            ]
        )
    # argparse error (missing chip-arms values) or our ValueError → non-zero.
    assert exc.value.code not in (0, None)


def test_sim_season_cli_missing_ledger_exits(tmp_path, monkeypatch) -> None:
    from fplt.cli import main

    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    # Minimal staged fixtures so DataStore construction is not the failure mode;
    # run_chip_arm_sim_season should fail on missing ledger actuals / deadlines.
    with (
        patch(
            "fplt.backtest.simulated_season.run_chip_arm_sim_season",
            side_effect=ValueError("no ledger actuals"),
        ),
        pytest.raises(SystemExit) as exc,
    ):
        main(
            [
                "sim-season",
                "--season",
                "2025-26",
                "--from-gw",
                "1",
                "--to-gw",
                "2",
                "--n-paths",
                "2",
                "--models",
                "rolling_mean_v1",
                "--chip-arms",
                "off",
                "optimised",
            ]
        )
    assert exc.value.code == 2


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


def _empty_season_paths_frame() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "path_id": pl.Int64(),
            "policy_id": pl.Utf8(),
            "season": pl.Utf8(),
            "total_points": pl.Int64(),
            "gameweeks": pl.Int64(),
            "attainable_points": pl.Int64(),
            "share_of_attainable": pl.Float64(),
        }
    )


def _empty_diffs_frame() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "metric": pl.Series([], dtype=pl.Utf8()),
            "control_id": pl.Series([], dtype=pl.Utf8()),
            "treatment_id": pl.Series([], dtype=pl.Utf8()),
            "n_paths": pl.Series([], dtype=pl.Int64()),
            "mean_diff": pl.Series([], dtype=pl.Float64()),
            "se": pl.Series([], dtype=pl.Float64()),
            "sd": pl.Series([], dtype=pl.Float64()),
        }
    )


def _chip_toy_season() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Three gameweeks with doubles so rule-based landmarks can fire."""
    cand_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    act_rows: list[dict[str, object]] = []
    player = 1
    for club in range(1, 7):
        for position in ("GK", "DEF", "DEF", "MID", "FWD"):
            for gw in (1, 2, 3):
                cand_rows.append(
                    {
                        "gw": gw,
                        "player_key": player,
                        "position": position,
                        "team_key": club,
                        "price": 40 + club,
                    }
                )
                xpts = float(club + gw)
                pred_rows.append(
                    {
                        "as_of_gw": gw,
                        "target_gw": gw,
                        "player_key": player,
                        "xpts": xpts,
                    }
                )
                if gw < 3:
                    pred_rows.append(
                        {
                            "as_of_gw": gw,
                            "target_gw": gw + 1,
                            "player_key": player,
                            "xpts": xpts * 0.9,
                        }
                    )
                act_rows.append(
                    {
                        "gw": gw,
                        "player_key": player,
                        "actual_pts": max(1, int(xpts)),
                        "actual_minutes": 90,
                    }
                )
            player += 1
    fixtures = pl.DataFrame(
        {
            "season": ["2024-25", "2024-25", "2024-25", "2024-25"],
            "gw": [2, 2, 3, 3],
            "team_h_key": [1, 1, 2, 3],
            "team_a_key": [2, 3, 4, 5],
        }
    )
    return (
        pl.DataFrame(cand_rows),
        pl.DataFrame(pred_rows),
        pl.DataFrame(act_rows),
        fixtures,
    )
