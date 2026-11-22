"""New OpenFPL-facing columns on staged FPL and Understat tables."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl

from fplt.data.staging import stage_event_live
from fplt.data.store import DataStore


def test_stage_event_live_keeps_ict_and_bps() -> None:
    frame = stage_event_live(
        {
            "elements": [
                {
                    "id": 10,
                    "stats": {
                        "minutes": 90,
                        "total_points": 6,
                        "goals_scored": 0,
                        "assists": 1,
                        "clean_sheets": 0,
                        "goals_conceded": 1,
                        "saves": 0,
                        "bonus": 1,
                        "yellow_cards": 0,
                        "red_cards": 0,
                        "own_goals": 0,
                        "penalties_saved": 0,
                        "penalties_missed": 0,
                        "starts": 1,
                        "influence": "12.4",
                        "creativity": "8.2",
                        "threat": "3.0",
                        "bps": "24",
                        "clearances_blocks_interceptions": 8,
                        "tackles": 3,
                        "recoveries": 5,
                        "defensive_contribution": 2,
                    },
                    "explain": [{"fixture": 99}],
                }
            ]
        },
        "2024-25",
        1,
        datetime(2024, 8, 17, 18, tzinfo=UTC),
        {10: 100},
    )
    row = frame.row(0, named=True)
    assert row["influence"] == 12.4
    assert row["creativity"] == 8.2
    assert row["threat"] == 3.0
    assert row["bps"] == 24
    assert row["clearances_blocks_interceptions"] == 8
    assert row["tackles"] == 3
    assert row["recoveries"] == 5
    assert row["defensive_contribution"] == 2


def test_stage_event_live_defaults_missing_defensive_contribution() -> None:
    frame = stage_event_live(
        {
            "elements": [
                {
                    "id": 1,
                    "stats": {"minutes": 90, "total_points": 2, "starts": 1},
                    "explain": [{"fixture": 1}],
                }
            ]
        },
        "2025-26",
        1,
        datetime(2025, 8, 16, 18, tzinfo=UTC),
        {1: 10},
    )
    assert frame["defensive_contribution"].to_list() == [0]


def test_stage_event_live_keeps_expected_xg() -> None:
    available = datetime(2024, 8, 17, 18, tzinfo=UTC)
    frame = stage_event_live(
        {
            "elements": [
                {
                    "id": 10,
                    "stats": {
                        "minutes": 90,
                        "total_points": 6,
                        "goals_scored": 1,
                        "assists": 0,
                        "starts": 1,
                        "expected_goals": "0.79",
                        "expected_assists": "0.18",
                        "expected_goal_involvements": "0.97",
                        "expected_goals_conceded": "2.24",
                    },
                    "explain": [{"fixture": 99}],
                }
            ]
        },
        "2024-25",
        1,
        available,
        {10: 100},
    )
    row = frame.row(0, named=True)
    assert row["expected_goals"] == 0.79
    assert row["expected_assists"] == 0.18
    assert row["expected_goal_involvements"] == 0.97
    assert row["expected_goals_conceded"] == 2.24
    assert row["available_at"] == available


def test_stage_event_live_null_expected_when_absent() -> None:
    frame = stage_event_live(
        {
            "elements": [
                {
                    "id": 1,
                    "stats": {"minutes": 90, "total_points": 2, "starts": 1},
                    "explain": [{"fixture": 1}],
                }
            ]
        },
        "2021-22",
        1,
        datetime(2021, 8, 14, 18, tzinfo=UTC),
        {1: 10},
    )
    assert frame["expected_goals"].to_list() == [None]
    assert frame["expected_assists"].to_list() == [None]
    assert frame["expected_goal_involvements"].to_list() == [None]
    assert frame["expected_goals_conceded"].to_list() == [None]


def test_store_fills_missing_columns_on_widen(tmp_path) -> None:
    store = DataStore(tmp_path)
    store.write(
        "player_fixture_stats",
        pl.DataFrame(
            {
                "season": ["2024-25"],
                "gw": [1],
                "player_key": [1],
                "fixture_id": [1],
                "minutes": [90],
                "total_points": [2],
                "goals_scored": [0],
                "assists": [0],
                "clean_sheets": [0],
                "goals_conceded": [0],
                "saves": [0],
                "bonus": [0],
                "yellow_cards": [0],
                "red_cards": [0],
                "own_goals": [0],
                "penalties_saved": [0],
                "penalties_missed": [0],
                "starts": [1],
                "available_at": [datetime(2024, 8, 17, 18, tzinfo=UTC)],
                "source": ["fpl_api"],
            }
        ),
    )
    got = store.read("player_fixture_stats")
    assert got["influence"].to_list() == [0.0]
    assert got["bps"].to_list() == [0]
    assert got["clearances_blocks_interceptions"].to_list() == [0]
    assert got["tackles"].to_list() == [0]
    assert got["recoveries"].to_list() == [0]
    assert got["defensive_contribution"].to_list() == [0]
    assert got["expected_goals"].to_list() == [None]
    assert got["expected_assists"].to_list() == [None]
    assert got["expected_goal_involvements"].to_list() == [None]
    assert got["expected_goals_conceded"].to_list() == [None]
