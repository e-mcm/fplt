"""Coverage-focused unit tests for ``fplt.optimise.decisions`` helpers."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest

from fplt.optimise.chip_windows import ChipUseWindows
from fplt.optimise.decisions import (
    _actual_rows,
    _actuals,
    _consume_chip,
    _decisions,
    _empty_decisions,
    _explain_chip_windows_note,
    _next_free_after_chip,
    _normalize_price_forecast_method,
    _prediction_rows,
    _scheduled_windows,
    _static_windows_for_arm,
    _window_points,
    actual_rows_from_ledger,
    format_chip_decision_scoreboard,
    points_from_ledger,
    prediction_rows_from_ledger,
    read_chip_windows,
    read_decisions,
    run_chip_decision_board,
    run_plan,
    simulate_decisions,
    summarise_chip_decisions,
    write_chip_windows,
    write_decisions,
)
from fplt.optimise.state import ChipInventory, SquadState
from fplt.scoring.chips import ChipRules


def _inventory(**remaining: int) -> ChipInventory:
    base = {"wildcard": 1, "freehit": 1, "bboost": 1, "3xc": 1}
    base.update(remaining)
    return ChipInventory(remaining=base, played=[])


def _rules(*, free_hit_keeps: bool = True) -> ChipRules:
    return ChipRules.model_validate(
        {
            "one_chip_per_gameweek": True,
            "free_hit_keeps_free_transfers": free_hit_keeps,
            "windows": [
                {
                    "id": "all",
                    "start_event": 1,
                    "stop_event": 38,
                    "carries_over": False,
                }
            ],
            "grants": [
                {"name": "wildcard", "window": "all"},
                {"name": "freehit", "window": "all"},
                {"name": "bboost", "window": "all"},
                {"name": "3xc", "window": "all"},
            ],
        }
    )


def _decision_row(**overrides: object) -> dict[str, object]:
    row: dict[str, object] = {
        "model_id": "m1",
        "season": "2024-25",
        "gw": 1,
        "player_key": 1,
        "position": "MID",
        "in_xi": True,
        "captain": False,
        "vice": False,
        "bought": False,
        "sold": False,
        "xpts": 4.0,
        "actual_pts": 5,
        "actual_minutes": 90,
        "hit": 0.0,
        "free_transfers": 1,
        "chip": None,
    }
    row.update(overrides)
    return row


def test_points_from_ledger_empty_and_no_match() -> None:
    empty = points_from_ledger(pl.DataFrame(), "m", "2024-25", 1, [1])
    assert empty.is_empty()
    ledger = pl.DataFrame(
        {
            "model_id": ["other"],
            "as_of_season": ["2024-25"],
            "as_of_gw": [1],
            "target_season": ["2024-25"],
            "target_gw": [1],
            "player_key": [1],
            "xpts": [2.0],
            "fixture_id": [10],
        }
    )
    assert points_from_ledger(ledger, "m", "2024-25", 1, [1]).is_empty()


def test_read_write_decisions_and_chip_windows_empty_paths(tmp_path: Path) -> None:
    assert read_decisions(tmp_path).is_empty()
    assert read_chip_windows(tmp_path).is_empty()
    path = write_decisions(tmp_path, _empty_decisions())
    assert path.name == "part.parquet"
    assert not path.exists()
    win_path = write_chip_windows(
        tmp_path, model_id="m", season="2024-25", windows_by_gw={}
    )
    assert win_path.name == "part.parquet"
    assert not win_path.exists()


def test_write_decisions_adds_chip_column_when_missing(tmp_path: Path) -> None:
    row = _decision_row()
    del row["chip"]
    frame = pl.DataFrame([row])
    path = write_decisions(tmp_path, frame)
    loaded = read_decisions(tmp_path)
    assert path.exists()
    assert "chip" in loaded.columns
    assert loaded.height == 1


def test_simulate_rejects_unknown_chip_arm_and_empty_gw_pool() -> None:
    candidates = pl.DataFrame(
        {
            "gw": [1],
            "player_key": [1],
            "position": ["MID"],
            "team_key": [1],
            "price": [50],
        }
    )
    predictions = pl.DataFrame(
        {"as_of_gw": [1], "target_gw": [1], "player_key": [1], "xpts": [1.0]}
    )
    actuals = pl.DataFrame(
        {
            "gw": [1],
            "player_key": [1],
            "actual_pts": [1],
            "actual_minutes": [90],
        }
    )
    state = SquadState.model_validate(
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
    with pytest.raises(ValueError, match="unknown chip_arm"):
        simulate_decisions(
            candidates,
            predictions,
            actuals,
            state,
            model_id="m",
            season="2024-25",
            gameweeks=[1],
            horizon=1,
            chip_arm="not_an_arm",  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="no candidates for gw"):
        simulate_decisions(
            candidates,
            predictions,
            actuals,
            state,
            model_id="m",
            season="2024-25",
            gameweeks=[2],
            horizon=1,
        )


def test_run_plan_rejects_unknown_chip_arm(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown chip_arm"):
        run_plan(
            tmp_path,
            "rolling_mean_v1",
            "2024-25",
            1,
            1,
            chip_arm="bogus",  # type: ignore[arg-type]
        )


def test_chip_board_rejects_empty_models_or_arms(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="models must be non-empty"):
        run_chip_decision_board(
            tmp_path, [], "2024-25", 1, 2, 1, simulate=False
        )
    with pytest.raises(ValueError, match="arms must be non-empty"):
        run_chip_decision_board(
            tmp_path,
            ["rolling_mean_v1"],
            "2024-25",
            1,
            2,
            1,
            arms=[],
            simulate=False,
        )


def test_summarise_and_format_empty_chip_board() -> None:
    summary = summarise_chip_decisions(
        _empty_decisions(), season="2024-25", chip_rules_season="2024-25"
    )
    assert summary.is_empty()
    assert format_chip_decision_scoreboard(summary) == (
        "_No chip-decision rows to score._"
    )


def test_prediction_and_actual_rows_empty_paths() -> None:
    assert prediction_rows_from_ledger(pl.DataFrame(), "m", "2024-25").is_empty()
    assert actual_rows_from_ledger(pl.DataFrame(), "m", "2024-25").is_empty()
    wrong = pl.DataFrame(
        {
            "model_id": ["other"],
            "as_of_season": ["2024-25"],
            "target_season": ["2024-25"],
            "as_of_gw": [1],
            "target_gw": [1],
            "player_key": [1],
            "xpts": [1.0],
            "actual_pts": [2],
            "actual_minutes": [90],
            "horizon": [1],
        }
    )
    assert _prediction_rows(wrong, "m", "2024-25").is_empty()
    assert _actual_rows(wrong, "m", "2024-25").is_empty()


def test_actual_rows_horizon_filter_without_fixture_id() -> None:
    ledger = pl.DataFrame(
        {
            "model_id": ["m", "m"],
            "target_season": ["2024-25", "2024-25"],
            "target_gw": [1, 1],
            "player_key": [1, 1],
            "actual_pts": [3, 5],
            "horizon": [1, 2],
            "as_of_gw": [1, 1],
        }
    )
    out = _actual_rows(ledger, "m", "2024-25")
    assert out.height == 1
    assert out["actual_pts"][0] == 3
    assert out["actual_minutes"][0] is None


def test_window_points_empty() -> None:
    assert _window_points(pl.DataFrame(), 1, [1, 2]).is_empty()
    preds = pl.DataFrame(
        {
            "as_of_gw": [2],
            "target_gw": [2],
            "player_key": [1],
            "xpts": [1.0],
        }
    )
    assert _window_points(preds, 1, [1]).is_empty()


def test_normalize_price_forecast_method() -> None:
    assert _normalize_price_forecast_method(None) is None
    assert _normalize_price_forecast_method("flat") is None
    assert _normalize_price_forecast_method("mean_gw_delta") == "mean_gw_delta"
    with pytest.raises(ValueError, match="unknown price forecast"):
        _normalize_price_forecast_method("magic")  # type: ignore[arg-type]


def test_next_free_after_chip_wildcard_and_freehit() -> None:
    state = SquadState.model_validate(
        {
            "bank": 0,
            "free_transfers": 2,
            "ft_bank_cap": 2,
            "hit_cost": 4,
            "decay": 1.0,
            "bench_weight": 0.0,
            "owned": [],
        }
    )
    assert _next_free_after_chip(state, chip="wildcard", moves=3, rules=None) == 1
    keep = _next_free_after_chip(
        state, chip="freehit", moves=0, rules=_rules(free_hit_keeps=True)
    )
    assert keep == 2  # banked FTs kept, no moves
    reset = _next_free_after_chip(
        state, chip="freehit", moves=0, rules=_rules(free_hit_keeps=False)
    )
    assert reset == 1


def test_consume_chip_errors() -> None:
    inv = _inventory(wildcard=0, freehit=0, bboost=0, **{"3xc": 0})
    with pytest.raises(ValueError, match="unknown chip"):
        _consume_chip(inv, "assistant_manager", 1)
    with pytest.raises(ValueError, match="no remaining"):
        _consume_chip(inv, "wildcard", 1)
    updated = _consume_chip(_inventory(), "bboost", 5)
    assert updated.remaining["bboost"] == 0
    assert updated.played[0].name == "bboost"
    assert updated.played[0].gw == 5


def test_static_windows_rule_based_requires_rules_and_fixtures() -> None:
    preds = pl.DataFrame(
        {"target_gw": [1], "player_key": [1], "xpts": [1.0], "as_of_gw": [1]}
    )
    with pytest.raises(ValueError, match="requires chip rules"):
        _static_windows_for_arm(
            "rule_based",
            fixtures=None,
            season="2024-25",
            rules=None,
            predictions=preds,
            gameweeks=[1],
            chip_windows=None,
        )
    with pytest.raises(ValueError, match="requires fixtures"):
        _static_windows_for_arm(
            "rule_based",
            fixtures=pl.DataFrame(),
            season="2024-25",
            rules=_rules(),
            predictions=preds,
            gameweeks=[1],
            chip_windows=None,
        )
    assert (
        _static_windows_for_arm(
            "optimised",
            fixtures=None,
            season="2024-25",
            rules=_rules(),
            predictions=preds,
            gameweeks=[1],
            chip_windows=None,
        )
        is None
    )
    pre = ChipUseWindows(allowed={"bboost": frozenset({1})})
    assert (
        _static_windows_for_arm(
            "rule_based",
            fixtures=None,
            season="2024-25",
            rules=None,
            predictions=preds,
            gameweeks=[1],
            chip_windows=pre,
        )
        is pre
    )


def test_scheduled_windows_requires_fixtures() -> None:
    state = SquadState.model_validate(
        {
            "bank": 0,
            "free_transfers": 1,
            "ft_bank_cap": 2,
            "hit_cost": 4,
            "decay": 1.0,
            "bench_weight": 0.0,
            "owned": [],
        }
    )
    with pytest.raises(ValueError, match="requires fixtures"):
        _scheduled_windows(
            fixtures=None,
            season="2024-25",
            rules=_rules(),
            as_of_gw=1,
            short_horizon=3,
            predictions=pl.DataFrame(
                {"target_gw": [1], "player_key": [1], "xpts": [1.0]}
            ),
            pool=pl.DataFrame({"player_key": [1], "team_key": [1]}),
            state=state,
            gameweeks=[1],
        )


def test_actuals_without_minutes_column() -> None:
    frame = pl.DataFrame(
        {"gw": [1, 1], "player_key": [1, 1], "actual_pts": [2, 3]}
    )
    out = _actuals(frame)
    assert out.height == 1
    assert out["actual_pts"][0] == 5
    assert out["actual_minutes"][0] is None


def test_explain_chip_windows_note_paths(tmp_path: Path) -> None:
    assert (
        _explain_chip_windows_note(
            tmp_path, model_id="m", season="2024-25", gameweek=1
        )
        is None
    )
    windows = ChipUseWindows(
        allowed={"bboost": frozenset({1})},
        forced={"bboost": frozenset({1})},
        no_chip=frozenset({2}),
    )
    write_chip_windows(
        tmp_path,
        model_id="m::chip_optimised",
        season="2024-25",
        windows_by_gw={1: windows},
    )
    note = _explain_chip_windows_note(
        tmp_path, model_id="m", season="2024-25", gameweek=1
    )
    assert note is not None
    assert "chip_windows" in note
    assert "bboost" in note
    assert (
        _explain_chip_windows_note(
            tmp_path, model_id="m", season="2024-25", gameweek=3
        )
        is None
    )


def test_decisions_schema_coerces_missing_chip() -> None:
    row = _decision_row()
    del row["chip"]
    checked = _decisions(pl.DataFrame([row]))
    assert checked["chip"][0] is None


def test_explain_decision_gameweek_missing_and_with_windows(tmp_path: Path) -> None:
    from fplt.optimise.decisions import explain_decision_gameweek
    from fplt.models.spec import load_model_specs

    spec = load_model_specs()["rolling_mean_v1"]
    with pytest.raises(ValueError, match="no decision rows"):
        explain_decision_gameweek(tmp_path, "rolling_mean_v1", "2024-25", 1)

    row = _decision_row(model_id=spec.model_id, player_key=1, gw=1)
    write_decisions(tmp_path, pl.DataFrame([row]))
    windows = ChipUseWindows(allowed={"bboost": frozenset({1})})
    write_chip_windows(
        tmp_path,
        model_id=f"{spec.model_id}::chip_optimised",
        season="2024-25",
        windows_by_gw={1: windows},
    )
    with (
        patch(
            "fplt.optimise.decisions._player_names",
            return_value={1: "PlayerOne"},
        ),
        patch(
            "fplt.optimise.decisions.explain_gameweek",
            return_value="body\n",
        ),
        patch("fplt.optimise.decisions.PITData"),
        patch("fplt.optimise.decisions.DataStore"),
    ):
        text = explain_decision_gameweek(
            tmp_path, "rolling_mean_v1", "2024-25", 1
        )
    assert "rolling_mean_v1" in text
    assert "chip_windows" in text
    assert "hit=" in text


def test_chip_windows_log_filled_when_provided() -> None:
    """``chip_windows_log`` records active windows for optimised arm."""
    # Minimal smoke via mock: ensure the log branch assigns when provided.
    log: dict[int, ChipUseWindows] = {}
    windows = ChipUseWindows(allowed={"3xc": frozenset({1})})
    # Drive the log assignment path without a full MILP by patching plan_squad.
    candidates = pl.DataFrame(
        {
            "gw": [1] * 15,
            "player_key": list(range(1, 16)),
            "position": ["GK", "DEF", "DEF", "DEF", "DEF", "DEF"]
            + ["MID"] * 5
            + ["FWD"] * 4,
            "team_key": [1, 1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 1, 2],
            "price": [45] * 15,
        }
    )
    predictions = pl.DataFrame(
        {
            "as_of_gw": [1] * 15,
            "target_gw": [1] * 15,
            "player_key": list(range(1, 16)),
            "xpts": [float(i) for i in range(1, 16)],
        }
    )
    actuals = pl.DataFrame(
        {
            "gw": [1] * 15,
            "player_key": list(range(1, 16)),
            "actual_pts": [2] * 15,
            "actual_minutes": [90] * 15,
        }
    )
    state = SquadState.model_validate(
        {
            "bank": 1000,
            "free_transfers": 1,
            "ft_bank_cap": 2,
            "hit_cost": 4,
            "decay": 1.0,
            "bench_weight": 0.0,
            "owned": [],
            "chips": _inventory(),
        }
    )
    picks = pl.DataFrame(
        {
            "step": [0] * 15,
            "player_key": list(range(1, 16)),
            "position": candidates["position"].to_list(),
            "team_key": candidates["team_key"].to_list(),
            "price": [45] * 15,
            "xpts": [1.0] * 15,
            "in_xi": [True] * 11 + [False] * 4,
            "captain": [True] + [False] * 14,
            "vice": [False, True] + [False] * 13,
            "bought": [True] * 15,
            "sold": [False] * 15,
        }
    )
    fake_plan = MagicMock()
    fake_plan.picks = picks
    fake_plan.chips = [None]
    fake_plan.hits = [0.0]
    fake_plan.free_transfers = [1]
    fake_plan.points = 10.0
    with patch("fplt.optimise.decisions.plan_squad", return_value=fake_plan):
        frame = simulate_decisions(
            candidates,
            predictions,
            actuals,
            state,
            model_id="m",
            season="2024-25",
            gameweeks=[1],
            horizon=1,
            chip_arm="optimised",
            chip_rules_season="2025-26",
            chip_windows=windows,
            chip_windows_log=log,
        )
    assert 1 in log
    assert frame.height == 15
