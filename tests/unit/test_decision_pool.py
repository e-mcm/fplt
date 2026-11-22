"""Decision-relevant player pool filter for mean-seeking boards."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from fplt.backtest.decision_pool import (
    HEADLINE_METRIC,
    POOL_FILTER_DESCRIPTION,
    filter_decision_pool,
)
from fplt.data.store import DataStore

_START = datetime(2024, 8, 1, tzinfo=UTC)
_DEADLINE_GW1 = datetime(2024, 8, 16, 17, tzinfo=UTC)
_DEADLINE_GW3 = datetime(2024, 8, 30, 17, tzinfo=UTC)
_KICKOFF_GW1 = datetime(2024, 8, 17, 14, tzinfo=UTC)
_RESULT_GW1 = _KICKOFF_GW1 + timedelta(hours=3)


def _utc(frame: pl.DataFrame) -> pl.DataFrame:
    exprs = [
        pl.col(name).cast(pl.Datetime("us", "UTC"))
        for name, dtype in frame.schema.items()
        if isinstance(dtype, pl.Datetime)
    ]
    return frame.with_columns(exprs) if exprs else frame


def _seed(store: DataStore) -> None:
    season = "2024-25"
    store.write(
        "gameweeks",
        _utc(
            pl.DataFrame(
                {
                    "season": [season, season],
                    "gw": [1, 3],
                    "deadline_time": [_DEADLINE_GW1, _DEADLINE_GW3],
                    "finished": [True, False],
                    "data_checked": [True, False],
                    "available_at": [_START, _START],
                    "source": ["fpl_api", "fpl_api"],
                }
            )
        ),
    )
    store.write(
        "player_status",
        _utc(
            pl.DataFrame(
                {
                    "season": [season] * 4,
                    "player_key": [1, 2, 3, 4],
                    "status": ["a"] * 4,
                    "news": [""] * 4,
                    # 1: premium; 2: mid-price with minutes; 3: floor fodder;
                    # 4: mid-price no minutes (excluded at GW3).
                    "price": [80, 50, 40, 50],
                    "ep_next": [5.0] * 4,
                    "chance_of_playing": [100.0] * 4,
                    "penalties_order": [None] * 4,
                    "available_at": [_START] * 4,
                    "source": ["fpl_api"] * 4,
                }
            )
        ),
    )
    store.write(
        "player_fixture_stats",
        _utc(
            pl.DataFrame(
                {
                    "season": [season, season],
                    "gw": [1, 1],
                    "player_key": [2, 3],
                    "fixture_id": [1, 1],
                    "minutes": [90, 90],
                    "total_points": [5, 2],
                    "goals_scored": [0, 0],
                    "assists": [0, 0],
                    "clean_sheets": [0, 0],
                    "goals_conceded": [1, 1],
                    "saves": [0, 0],
                    "bonus": [0, 0],
                    "yellow_cards": [0, 0],
                    "red_cards": [0, 0],
                    "own_goals": [0, 0],
                    "penalties_saved": [0, 0],
                    "penalties_missed": [0, 0],
                    "starts": [1, 1],
                    "influence": [0.0, 0.0],
                    "creativity": [0.0, 0.0],
                    "threat": [0.0, 0.0],
                    "bps": [0, 0],
                    "available_at": [_RESULT_GW1, _RESULT_GW1],
                    "source": ["fpl_api", "fpl_api"],
                }
            )
        ),
    )


def _ledger_rows() -> pl.DataFrame:
    # GW3 as_of: fodder (price 40) and unused mid (no minutes) drop out.
    rows: list[dict[str, object]] = []
    for key in (1, 2, 3, 4):
        rows.append(
            {
                "model_id": "m",
                "run_kind": "backtest",
                "as_of_season": "2024-25",
                "as_of_gw": 3,
                "target_season": "2024-25",
                "target_gw": 3,
                "horizon": 1,
                "player_key": key,
                "position": "MID",
                "fixture_id": 1,
                "xpts": 1.0,
                "xpts_sd": 0.0,
                "p_haul": None,
                "p_60": None,
                "actual_pts": 1,
                "actual_minutes": 90,
            }
        )
    return pl.DataFrame(rows)


def test_headline_metric_is_rmse() -> None:
    assert HEADLINE_METRIC == "rmse"
    assert "decision-relevant" in POOL_FILTER_DESCRIPTION


def test_filter_keeps_premium_and_minutes_cuts_fodder(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    _seed(store)
    kept = filter_decision_pool(_ledger_rows(), store)
    assert sorted(kept["player_key"].to_list()) == [1, 2]


def test_early_season_keeps_priced_players_without_minutes(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    _seed(store)
    early = _ledger_rows().with_columns(pl.lit(1).alias("as_of_gw"))
    kept = filter_decision_pool(early, store)
    # price 40 still out; mid-price without minutes kept via early-season gate.
    assert sorted(kept["player_key"].to_list()) == [1, 2, 4]
