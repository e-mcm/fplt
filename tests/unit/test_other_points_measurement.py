"""Other-points measurement: haul Brier, match residual, captain regret."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from fplt.backtest.other_points_measurement import (
    brier_score,
    captain_regret_by_gw,
    format_captain_regret_markdown,
    log_score,
    minutes_bucket,
    pearson_corr,
    run_other_points_measurement,
    team_goals_band,
)
from fplt.data.store import DataStore

_SEASON = "2024-25"
_START = datetime(2024, 8, 1, tzinfo=UTC)


def test_minutes_bucket_edges() -> None:
    assert minutes_bucket(0) == "0"
    assert minutes_bucket(45) == "1-59"
    assert minutes_bucket(60) == "60+"
    assert minutes_bucket(90) == "60+"


def test_team_goals_band() -> None:
    assert team_goals_band(0) == "0"
    assert team_goals_band(1) == "1"
    assert team_goals_band(3) == "2+"


def test_brier_and_log_empty() -> None:
    empty = np.array([], dtype=np.float64)
    assert brier_score(empty, empty) is None
    assert log_score(empty, empty) is None


def test_pearson_corr_constant_is_none() -> None:
    assert pearson_corr(np.array([1.0, 1.0]), np.array([2.0, 3.0])) is None
    correlated = pearson_corr(np.array([1.0, 2.0]), np.array([2.0, 4.0]))
    assert correlated == pytest.approx(1.0)


def test_captain_regret_by_gw_ranks_pool() -> None:
    detail = pl.DataFrame(
        {
            "season": [_SEASON, _SEASON, _SEASON],
            "gw": [3, 3, 3],
            "player_key": [10, 11, 12],
            "fixture_id": [3, 3, 3],
            "position": ["MID", "FWD", "DEF"],
            "minutes": [90, 90, 90],
            "minutes_bucket": ["60+", "60+", "60+"],
            "xpts": [8.0, 6.0, 4.0],
            "p_haul": [0.2, 0.1, 0.05],
            "actual_pts": [2, 12, 3],
            "hauled": [0.0, 1.0, 0.0],
            "actual_other": [0.0, 3.0, 0.0],
            "predicted_other": [0.5, 0.5, 0.5],
            "other_residual": [-0.5, 2.5, -0.5],
            "actual_bonus": [0.0, 3.0, 0.0],
            "actual_saves": [0.0, 0.0, 0.0],
            "actual_defcon": [0.0, 0.0, 0.0],
            "team_goals": [1, 2, 0],
            "team_goals_band": ["1", "2+", "0"],
            "clean_sheet": [0, 0, 1],
            "player_scored": [0, 1, 0],
            "in_decision_pool": [True, True, False],
        }
    )
    regret = captain_regret_by_gw(detail)
    assert regret.height == 1
    assert regret["chosen_player_key"].item() == 10
    assert regret["oracle_player_key"].item() == 11
    assert regret["regret"].item() == pytest.approx(10.0)
    assert regret["n_pool"].item() == 2


def test_format_captain_regret_markdown_empty() -> None:
    text = format_captain_regret_markdown(
        pl.DataFrame(
            schema={
                "season": pl.Utf8(),
                "gw": pl.Int64(),
                "chosen_player_key": pl.Int64(),
                "oracle_player_key": pl.Int64(),
                "chosen_xpts": pl.Float64(),
                "chosen_actual": pl.Float64(),
                "oracle_actual": pl.Float64(),
                "regret": pl.Float64(),
                "n_pool": pl.Int64(),
            }
        )
    )
    assert "No decision-pool rows" in text


def test_run_other_points_measurement_writes_board(tmp_path: Path) -> None:
    _league(DataStore(tmp_path))
    out = tmp_path / "measure"
    result = run_other_points_measurement(
        tmp_path,
        _SEASON,
        from_gw=3,
        to_gw=3,
        points_model="dc_layered_v1",
        n_sims=40,
        output_dir=out,
    )
    assert (out / "summary.json").is_file()
    assert (out / "summary.parquet").is_file()
    assert (out / "captain_regret.parquet").is_file()
    assert (out / "captain_regret.md").is_file()
    assert (out / "detail.parquet").is_file()
    metrics = set(result.summary["metric"].to_list())
    assert {"brier_haul", "log_haul", "haul_mean_p", "haul_mean_y", "haul_cal_gap"} <= (
        metrics
    )
    assert {"other_mae", "other_mean_residual", "captain_regret_mean"} <= metrics
    slices = set(result.summary["slice"].to_list())
    assert "all" in slices
    assert "decision_pool" in slices
    assert "top_xpts_decile" in slices
    assert any(s.startswith("match|team_goals=") for s in slices)
    assert result.detail.height > 0
    assert "other_residual" in result.detail.columns
    assert "p_haul" in result.detail.columns


def test_run_other_points_measurement_unknown_model(tmp_path: Path) -> None:
    _league(DataStore(tmp_path))
    with pytest.raises(ValueError, match="unknown model spec"):
        run_other_points_measurement(
            tmp_path,
            _SEASON,
            from_gw=3,
            to_gw=3,
            points_model="missing_layered",
            output_dir=tmp_path / "bad",
        )


def test_run_other_points_measurement_inverted_window(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="from_gw"):
        run_other_points_measurement(
            tmp_path,
            _SEASON,
            from_gw=5,
            to_gw=2,
            output_dir=tmp_path / "bad-window",
        )


def test_run_other_points_measurement_bad_n_sims(tmp_path: Path) -> None:
    _league(DataStore(tmp_path))
    with pytest.raises(ValueError, match="n_sims"):
        run_other_points_measurement(
            tmp_path,
            _SEASON,
            from_gw=3,
            to_gw=3,
            points_model="dc_layered_v1",
            n_sims=0,
            output_dir=tmp_path / "bad-sims",
        )


def test_match_context_residual_varies_with_goals(tmp_path: Path) -> None:
    _league(DataStore(tmp_path), bonus_on_high_scoring=True)
    result = run_other_points_measurement(
        tmp_path,
        _SEASON,
        from_gw=3,
        to_gw=3,
        points_model="dc_layered_v1",
        n_sims=40,
        output_dir=tmp_path / "ctx",
    )
    played = result.detail.filter(pl.col("minutes") > 0)
    assert played.height > 0
    high = result.summary.filter(
        (pl.col("metric") == "other_mean_actual")
        & (pl.col("slice") == "match|team_goals=2+")
    )
    zero = result.summary.filter(
        (pl.col("metric") == "other_mean_actual")
        & (pl.col("slice") == "match|team_goals=0")
    )
    # Synthetic league awards bonus only on the high-scoring side.
    if high.height and zero.height and high["n"].item() > 0 and zero["n"].item() > 0:
        assert high["value"].item() >= zero["value"].item()


def _league(
    store: DataStore,
    *,
    bonus_on_high_scoring: bool = False,
) -> DataStore:
    """Two clubs, three finished GWs plus one upcoming target."""
    kickoffs = [_START + timedelta(days=7 * gw) for gw in range(1, 5)]
    deadlines = [kick - timedelta(hours=21) for kick in kickoffs]
    available = [kick + timedelta(hours=3) for kick in kickoffs]
    available[-1] = deadlines[-1] - timedelta(days=1)
    store.write(
        "teams",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON, _SEASON],
                    "team_key": [1, 2],
                    "team_id": [1, 2],
                    "name": ["Home", "Away"],
                    "short_name": ["HOM", "AWY"],
                    "available_at": [_START, _START],
                    "source": ["fpl_api", "fpl_api"],
                }
            )
        ),
    )
    store.write(
        "players",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * 4,
                    "player_key": [10, 11, 20, 21],
                    "element_id": [10, 11, 20, 21],
                    "team_key": [1, 1, 2, 2],
                    "position": ["MID", "FWD", "MID", "FWD"],
                    "web_name": ["A", "B", "C", "D"],
                    "available_at": [_START] * 4,
                    "source": ["fpl_api"] * 4,
                }
            )
        ),
    )
    store.write(
        "gameweeks",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * 4,
                    "gw": [1, 2, 3, 4],
                    "deadline_time": deadlines,
                    "finished": [True, True, True, False],
                    "data_checked": [True, True, True, False],
                    "available_at": [_START] * 4,
                    "source": ["fpl_api"] * 4,
                }
            )
        ),
    )
    schedule_at = [deadline - timedelta(days=1) for deadline in deadlines]
    # GW3: home wins 2-0 so home side has team_goals=2+, CS=1; away team_goals=0.
    store.write(
        "fixtures",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * 8,
                    "fixture_id": [1, 1, 2, 2, 3, 3, 4, 4],
                    "gw": [1, 1, 2, 2, 3, 3, 4, 4],
                    "team_h_key": [1, 1, 2, 2, 1, 1, 1, 1],
                    "team_a_key": [2, 2, 1, 1, 2, 2, 2, 2],
                    "kickoff_time": [
                        kickoffs[0],
                        kickoffs[0],
                        kickoffs[1],
                        kickoffs[1],
                        kickoffs[2],
                        kickoffs[2],
                        kickoffs[3],
                        kickoffs[3],
                    ],
                    "team_h_score": [None, 2, None, 0, None, 2, None, None],
                    "team_a_score": [None, 1, None, 1, None, 0, None, None],
                    "finished": [False, True, False, True, False, True, False, False],
                    "available_at": [
                        schedule_at[0],
                        available[0],
                        schedule_at[1],
                        available[1],
                        schedule_at[2],
                        available[2],
                        schedule_at[3],
                        schedule_at[3],
                    ],
                    "source": ["fpl_api"] * 8,
                }
            )
        ),
    )
    store.write(
        "player_status",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * 4,
                    "player_key": [10, 11, 20, 21],
                    "status": ["a", "a", "a", "a"],
                    "news": [""] * 4,
                    "news_added": [None] * 4,
                    "price": [70, 80, 65, 90],
                    "ep_next": [4.0, 5.0, 3.0, 6.0],
                    "chance_of_playing": [100.0] * 4,
                    "chance_of_playing_this_round": [100.0] * 4,
                    "penalties_order": [None, 1, None, None],
                    "available_at": [deadlines[2]] * 4,
                    "source": ["fpl_api"] * 4,
                }
            )
        ),
    )
    gw3_bonus_a = 3 if bonus_on_high_scoring else 0
    gw3_bonus_b = 2 if bonus_on_high_scoring else 0
    appearances = {
        10: [(1, 90, 1, 1, 0), (2, 90, 1, 0, 1), (3, 90, 1, 1, gw3_bonus_a)],
        11: [(1, 90, 1, 0, 0), (2, 60, 1, 1, 0), (3, 90, 1, 0, gw3_bonus_b)],
        20: [(1, 90, 1, 0, 0), (2, 90, 1, 1, 0), (3, 90, 1, 0, 0)],
        21: [(1, 75, 1, 1, 0), (2, 90, 1, 0, 0), (3, 45, 1, 0, 0)],
    }
    stats_rows: list[dict[str, object]] = []
    for player, rows in appearances.items():
        for gw, minutes, starts, goals, bonus in rows:
            cs = 1 if gw == 3 and player in (10, 11) and minutes >= 60 else 0
            stats_rows.append(
                {
                    "season": _SEASON,
                    "gw": gw,
                    "player_key": player,
                    "fixture_id": gw,
                    "minutes": minutes,
                    "total_points": 2 + goals * 5 + bonus + cs * 1,
                    "goals_scored": goals,
                    "assists": 0,
                    "clean_sheets": cs,
                    "goals_conceded": 0 if cs else 1,
                    "saves": 0,
                    "bonus": bonus,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "own_goals": 0,
                    "penalties_saved": 0,
                    "penalties_missed": 0,
                    "starts": starts,
                    "influence": 10.0,
                    "creativity": 5.0,
                    "threat": 8.0,
                    "bps": 15 + bonus * 10,
                    "clearances_blocks_interceptions": 0,
                    "tackles": 0,
                    "recoveries": 0,
                    "defensive_contribution": 0,
                    "expected_goals": 0.0,
                    "expected_assists": 0.0,
                    "expected_goal_involvements": 0.0,
                    "expected_goals_conceded": 0.0,
                    "available_at": available[gw - 1],
                    "source": "fpl_api",
                }
            )
    store.write("player_fixture_stats", _utc(pl.DataFrame(stats_rows)))
    return store


def _utc(frame: pl.DataFrame) -> pl.DataFrame:
    columns = [
        pl.col(name).cast(pl.Datetime("us", "UTC"))
        for name, dtype in frame.schema.items()
        if isinstance(dtype, pl.Datetime)
    ]
    return frame.with_columns(columns) if columns else frame
