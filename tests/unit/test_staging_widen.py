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
