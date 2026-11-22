"""Event Brier / log scores and calibration tables."""

from __future__ import annotations

import polars as pl
import pytest

from fplt.backtest.metrics import (
    calibration_tables,
    format_calibration_markdown,
    overall_summary,
    summarise,
)


def _ledger(**overrides: object) -> pl.DataFrame:
    base: dict[str, object] = {
        "model_id": ["m", "m", "m", "m"],
        "run_kind": ["backtest"] * 4,
        "as_of_season": ["2024-25"] * 4,
        "as_of_gw": [1, 1, 1, 1],
        "target_season": ["2024-25"] * 4,
        "target_gw": [1, 1, 1, 1],
        "horizon": [1, 1, 1, 1],
        "player_key": [1, 2, 3, 4],
        "position": ["MID", "MID", "FWD", "DEF"],
        "fixture_id": [1, 1, 1, 1],
        "xpts": [8.0, 2.0, 5.0, 1.0],
        "xpts_sd": [1.0, 0.5, 1.0, 0.2],
        "p_played": [0.9, 0.2, 0.8, 0.1],
        "p_60": [0.8, 0.1, 0.7, 0.05],
        "p_goal": [0.4, 0.05, 0.3, 0.02],
        "p_assist": [0.3, 0.05, 0.2, 0.01],
        "p_cs": [0.0, 0.0, 0.0, 0.4],
        "p_haul": [0.25, 0.01, 0.1, 0.0],
        "actual_pts": [12, 0, 4, 6],
        "actual_minutes": [90, 0, 75, 90],
        "actual_goals": [1, 0, 0, 0],
        "actual_assists": [1, 0, 1, 0],
        "actual_cs": [0, 0, 0, 1],
    }
    base.update(overrides)
    return pl.DataFrame(base)


def test_summarise_emits_brier_and_log_for_each_event() -> None:
    summary = summarise(_ledger())
    mid = summary.filter(pl.col("position") == "MID").to_dicts()[0]
    # MID rows: played outcomes [1, 0], probs [0.9, 0.2] → Brier 0.025.
    assert mid["brier_played"] == pytest.approx(0.025)
    assert mid["brier_60"] is not None
    assert mid["brier_goal"] is not None
    assert mid["brier_assist"] is not None
    assert mid["brier_haul"] is not None
    assert mid["log_played"] is not None
    assert mid["log_haul"] > 0.0
    defence = summary.filter(pl.col("position") == "DEF").to_dicts()[0]
    assert defence["brier_cs"] == pytest.approx(0.36)


def test_perfect_probs_have_zero_brier_and_near_zero_log() -> None:
    ledger = _ledger(
        p_played=[1.0, 0.0, 1.0, 1.0],
        p_60=[1.0, 0.0, 1.0, 1.0],
        p_goal=[1.0, 0.0, 0.0, 0.0],
        p_assist=[1.0, 0.0, 1.0, 0.0],
        p_cs=[0.0, 0.0, 0.0, 1.0],
        p_haul=[1.0, 0.0, 0.0, 0.0],
    )
    overall = overall_summary(ledger).to_dicts()[0]
    for event in ("played", "60", "goal", "assist", "cs", "haul"):
        assert overall[f"brier_{event}"] == pytest.approx(0.0)
        assert overall[f"log_{event}"] == pytest.approx(0.0, abs=1e-12)


def test_missing_event_probs_leave_scores_null() -> None:
    ledger = _ledger(
        p_played=[None, None, None, None],
        p_goal=[None, None, None, None],
        p_assist=[None, None, None, None],
        p_cs=[None, None, None, None],
    )
    row = overall_summary(ledger).to_dicts()[0]
    assert row["brier_played"] is None
    assert row["log_goal"] is None
    assert row["brier_60"] is not None
    assert row["brier_haul"] is not None


def test_calibration_by_position_and_xpts_decile() -> None:
    n = 20
    ledger = pl.DataFrame(
        {
            "model_id": ["m"] * n,
            "run_kind": ["backtest"] * n,
            "as_of_season": ["2024-25"] * n,
            "as_of_gw": [1] * n,
            "target_season": ["2024-25"] * n,
            "target_gw": [1] * n,
            "horizon": [1] * n,
            "player_key": list(range(n)),
            "position": (["MID"] * 10) + (["FWD"] * 10),
            "fixture_id": [1] * n,
            "xpts": [float(i) for i in range(n)],
            "xpts_sd": [0.0] * n,
            "p_played": [0.5] * n,
            "p_60": [0.5] * n,
            "p_goal": [0.2] * n,
            "p_assist": [0.1] * n,
            "p_cs": [0.0] * n,
            "p_haul": [0.1] * n,
            "actual_pts": [2] * n,
            "actual_minutes": [90] * n,
            "actual_goals": [0] * n,
            "actual_assists": [0] * n,
            "actual_cs": [0] * n,
        }
    )
    cal = calibration_tables(ledger, n_deciles=10)
    assert not cal.is_empty()
    kinds = set(cal["slice_kind"].to_list())
    assert kinds == {"position", "xpts_decile"}
    positions = set(
        cal.filter(pl.col("slice_kind") == "position")["slice_value"].to_list()
    )
    assert positions == {"MID", "FWD"}
    deciles = set(
        cal.filter(
            (pl.col("slice_kind") == "xpts_decile") & (pl.col("event") == "played")
        )["slice_value"].to_list()
    )
    assert len(deciles) == 10
    md = format_calibration_markdown(cal)
    assert "xpts_decile" in md
    assert format_calibration_markdown(cal.head(0)).startswith("_No calibration")
