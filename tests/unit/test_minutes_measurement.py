"""Minutes measurement: regime split, weighted Brier, weekly misses."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from fplt.backtest.minutes_measurement import (
    brier_score,
    classify_error_regime,
    format_misses_markdown,
    fpl_flag_label,
    log_score,
    run_minutes_measurement,
    weekly_minutes_misses,
    weighted_brier,
)
from fplt.data.store import DataStore

_SEASON = "2024-25"
_START = datetime(2024, 8, 1, tzinfo=UTC)


def test_classify_error_regime_priority() -> None:
    assert classify_error_regime(fully_available=False, started=True) == "availability"
    assert classify_error_regime(fully_available=False, started=False) == "availability"
    assert classify_error_regime(fully_available=True, started=True) == "early_sub"
    assert classify_error_regime(fully_available=True, started=False) == "rotation"


def test_fpl_flag_label_bands() -> None:
    assert fpl_flag_label("a", 100.0) == "a"
    assert fpl_flag_label("a", None) == "a"
    assert fpl_flag_label("d", 75.0) == "d"
    assert fpl_flag_label("i", 0.0) == "unavailable"
    assert fpl_flag_label("a", 50.0) == "chance_lt_100"


def test_weighted_brier_emphasises_high_xpts() -> None:
    probs = np.array([0.9, 0.1])
    outcomes = np.array([0.0, 0.0])
    # First row is a bad miss; second is a good call.
    equal = weighted_brier(probs, outcomes, np.array([1.0, 1.0]))
    heavy = weighted_brier(probs, outcomes, np.array([10.0, 1.0]))
    assert equal is not None
    assert heavy is not None
    assert heavy > equal


def test_weighted_brier_none_when_no_positive_weight() -> None:
    assert weighted_brier(np.array([0.5]), np.array([1.0]), np.array([0.0])) is None


def test_weighted_brier_shape_mismatch_raises() -> None:
    with pytest.raises(ValueError, match="shape"):
        weighted_brier(np.array([0.5, 0.5]), np.array([1.0]), np.array([1.0]))


def test_brier_and_log_empty() -> None:
    empty = np.array([], dtype=np.float64)
    assert brier_score(empty, empty) is None
    assert log_score(empty, empty) is None


def test_weekly_minutes_misses_ranks_by_impact() -> None:
    detail = pl.DataFrame(
        {
            "season": [_SEASON, _SEASON, _SEASON],
            "gw": [3, 3, 3],
            "player_key": [10, 11, 12],
            "fixture_id": [3, 3, 3],
            "position": ["MID", "FWD", "DEF"],
            "regime": ["rotation", "early_sub", "availability"],
            "fpl_flag": ["a", "a", "d"],
            "returning_from_absence": [False, False, True],
            "p_start": [0.9, 0.8, 0.2],
            "started": [0.0, 1.0, 0.0],
            "xpts_weight": [5.0, 8.0, 4.0],
            "in_decision_pool": [True, True, False],
        }
    )
    misses = weekly_minutes_misses(detail, top_per_gw=2)
    assert misses.height == 2
    # |0.9-0|=0.9 × 5 = 4.5 beats |0.8-1|=0.2 × 8 = 1.6 and |0.2-0|×4=0.8
    assert misses["player_key"].to_list()[0] == 10
    assert misses["impact"].to_list()[0] == pytest.approx(4.5)


def test_format_misses_markdown_empty() -> None:
    text = format_misses_markdown(
        pl.DataFrame(
            schema={
                "season": pl.Utf8(),
                "gw": pl.Int64(),
                "player_key": pl.Int64(),
                "fixture_id": pl.Int64(),
                "position": pl.Utf8(),
                "regime": pl.Utf8(),
                "fpl_flag": pl.Utf8(),
                "returning_from_absence": pl.Boolean(),
                "p_start": pl.Float64(),
                "started": pl.Float64(),
                "abs_delta_p_start": pl.Float64(),
                "xpts_weight": pl.Float64(),
                "impact": pl.Float64(),
                "in_decision_pool": pl.Boolean(),
            }
        )
    )
    assert "No snapshot-backed rows" in text


def test_run_minutes_measurement_writes_split_and_misses(tmp_path: Path) -> None:
    _league(DataStore(tmp_path))
    out = tmp_path / "measure"
    result = run_minutes_measurement(
        tmp_path,
        _SEASON,
        from_gw=3,
        to_gw=3,
        minutes_model="empirical_minutes_v2",
        output_dir=out,
        top_misses_per_gw=10,
    )
    assert (out / "summary.json").is_file()
    assert (out / "summary.parquet").is_file()
    assert (out / "misses.parquet").is_file()
    assert (out / "misses.md").is_file()
    assert (out / "detail.parquet").is_file()
    slices = set(result.summary["slice"].to_list())
    assert "source=fpl_api" in slices
    assert "source=fpl_api|regime=availability" in slices
    assert "source=fpl_api|regime=rotation" in slices
    assert "source=fpl_api|regime=early_sub" in slices
    assert "decision_pool|weighted" in slices
    metrics = set(result.summary["metric"].to_list())
    assert {"brier_played", "log_played", "brier_60", "log_60"} <= metrics
    assert {"weighted_brier_played", "weighted_brier_60", "weighted_brier_start"} <= (
        metrics
    )
    assert result.detail.filter(pl.col("source") == "fpl_api").height > 0
    # At least one returning / flag / position slice present.
    assert any(s.startswith("source=fpl_api|flag=") for s in slices)
    assert any(s.startswith("source=fpl_api|position=") for s in slices)
    assert any(s.startswith("source=fpl_api|returning=") for s in slices)


def test_run_minutes_measurement_unknown_model(tmp_path: Path) -> None:
    _league(DataStore(tmp_path))
    with pytest.raises(ValueError, match="unknown model spec"):
        run_minutes_measurement(
            tmp_path,
            _SEASON,
            from_gw=3,
            to_gw=3,
            minutes_model="missing_minutes",
            output_dir=tmp_path / "bad",
        )


def test_run_minutes_measurement_inverted_window(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="from_gw"):
        run_minutes_measurement(
            tmp_path,
            _SEASON,
            from_gw=5,
            to_gw=2,
            output_dir=tmp_path / "bad-window",
        )


def test_availability_regime_when_flagged(tmp_path: Path) -> None:
    store = _league(DataStore(tmp_path), injure_player=10)
    del store
    result = run_minutes_measurement(
        tmp_path,
        _SEASON,
        from_gw=3,
        to_gw=3,
        minutes_model="empirical_minutes_v2",
        output_dir=tmp_path / "flagged",
    )
    flagged = result.detail.filter(pl.col("player_key") == 10)
    assert flagged.height == 1
    assert flagged["regime"].item() == "availability"
    assert flagged["fpl_flag"].item() == "unavailable"


def test_returning_from_absence_uses_prior_blank(tmp_path: Path) -> None:
    _league(DataStore(tmp_path), blank_player_gw2=11)
    result = run_minutes_measurement(
        tmp_path,
        _SEASON,
        from_gw=3,
        to_gw=3,
        minutes_model="empirical_minutes_v2",
        output_dir=tmp_path / "returning",
    )
    row = result.detail.filter(pl.col("player_key") == 11)
    assert row.height == 1
    assert row["returning_from_absence"].item() is True
    others = result.detail.filter(pl.col("player_key") != 11)
    assert others["returning_from_absence"].sum() == 0


def _league(
    store: DataStore,
    *,
    injure_player: int | None = None,
    blank_player_gw2: int | None = None,
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
                    "team_h_score": [None, 2, None, 0, None, 1, None, None],
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
    statuses = ["a", "a", "a", "a"]
    chances = [100.0, 100.0, 100.0, 100.0]
    keys = [10, 11, 20, 21]
    if injure_player is not None:
        idx = keys.index(injure_player)
        statuses[idx] = "i"
        chances[idx] = 0.0
    store.write(
        "player_status",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * 4,
                    "player_key": keys,
                    "status": statuses,
                    "news": [""] * 4,
                    "news_added": [None] * 4,
                    "price": [70, 80, 65, 90],
                    "ep_next": [4.0, 5.0, 3.0, 6.0],
                    "chance_of_playing": chances,
                    "chance_of_playing_this_round": chances,
                    "penalties_order": [None, 1, None, None],
                    "available_at": [deadlines[2]] * 4,
                    "source": ["fpl_api"] * 4,
                }
            )
        ),
    )
    appearances = {
        10: [(1, 90, 1, 1), (2, 90, 1, 0), (3, 90, 1, 1)],
        11: [(1, 90, 1, 0), (2, 60, 1, 1), (3, 90, 1, 0)],
        20: [(1, 90, 1, 0), (2, 90, 1, 1), (3, 90, 1, 0)],
        21: [(1, 75, 1, 1), (2, 90, 1, 0), (3, 45, 1, 0)],
    }
    if blank_player_gw2 is not None:
        rows = appearances[blank_player_gw2]
        appearances[blank_player_gw2] = [
            (gw, 0 if gw == 2 else minutes, 0 if gw == 2 else starts, goals)
            for gw, minutes, starts, goals in rows
        ]
    stats_rows: list[dict[str, object]] = []
    for player, rows in appearances.items():
        for gw, minutes, starts, goals in rows:
            stats_rows.append(
                {
                    "season": _SEASON,
                    "gw": gw,
                    "player_key": player,
                    "fixture_id": gw,
                    "minutes": minutes,
                    "total_points": 2 + goals * 5,
                    "goals_scored": goals,
                    "assists": 0,
                    "clean_sheets": 0,
                    "goals_conceded": 1,
                    "saves": 0,
                    "bonus": 0,
                    "yellow_cards": 0,
                    "red_cards": 0,
                    "own_goals": 0,
                    "penalties_saved": 0,
                    "penalties_missed": 0,
                    "starts": starts,
                    "influence": 10.0,
                    "creativity": 5.0,
                    "threat": 8.0,
                    "bps": 15,
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
