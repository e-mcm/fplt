"""Component eval harness writes RPS, minutes Brier, and sims rows."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from fplt.backtest.component_eval import (
    format_component_table,
    run_component_eval,
)
from fplt.data.store import DataStore

_SEASON = "2024-25"
_START = datetime(2024, 8, 1, tzinfo=UTC)


def test_run_component_eval_writes_summary_and_sims(tmp_path: Path) -> None:
    store = _league(DataStore(tmp_path))
    del store
    out = tmp_path / "eval"
    result = run_component_eval(
        tmp_path,
        _SEASON,
        from_gw=3,
        to_gw=3,
        fit_gw=3,
        sims=[20, 40],
        output_dir=out,
    )
    assert (out / "summary.json").is_file()
    assert (out / "summary.parquet").is_file()
    assert (out / "sims_sweep.parquet").is_file()
    components = set(result.summary["component"].to_list())
    assert {"score", "minutes", "involvement"} <= components
    score = result.summary.filter(
        (pl.col("component") == "score") & (pl.col("metric") == "rps")
    )
    assert score.height == 3
    assert set(score["model_name"].to_list()) == {
        "dixon_coles_v1",
        "poisson_score_v1",
        "odds_poisson_v1",
    }
    assert score.filter(pl.col("n") > 0).height >= 1
    cs = result.summary.filter(pl.col("metric") == "brier_cs_home")
    assert cs.height == 3
    assert cs.filter(pl.col("n") > 0).height >= 1
    minutes = result.summary.filter(pl.col("metric") == "brier_60")
    assert minutes.filter(pl.col("slice") == "overall")["n"].item() > 0
    assert set(result.sims["n_sims"].to_list()) == {20, 40}
    assert result.sims["n_players"].min() > 0
    table = format_component_table(result.summary)
    assert "score" in table


def test_skip_sims_leaves_sweep_empty(tmp_path: Path) -> None:
    _league(DataStore(tmp_path))
    result = run_component_eval(
        tmp_path,
        _SEASON,
        from_gw=3,
        to_gw=3,
        sims=[],
        output_dir=tmp_path / "empty-sims",
    )
    assert result.sims.is_empty()


def test_unknown_spec_raises(tmp_path: Path) -> None:
    _league(DataStore(tmp_path))
    with pytest.raises(ValueError, match="unknown model spec"):
        run_component_eval(
            tmp_path,
            _SEASON,
            from_gw=3,
            to_gw=3,
            minutes_model="missing_minutes",
            sims=[],
            output_dir=tmp_path / "bad",
        )


def test_inverted_window_raises(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="from_gw"):
        run_component_eval(
            tmp_path,
            _SEASON,
            from_gw=5,
            to_gw=2,
            sims=[],
            output_dir=tmp_path / "bad-window",
        )


def _league(store: DataStore) -> DataStore:
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
    # Schedule rows are visible at each deadline; result rows arrive after kickoff.
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
    store.write(
        "player_status",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * 4,
                    "player_key": [10, 11, 20, 21],
                    "status": ["a"] * 4,
                    "news": [""] * 4,
                    "price": [70, 80, 65, 90],
                    "ep_next": [4.0, 5.0, 3.0, 6.0],
                    "chance_of_playing": [100.0] * 4,
                    "available_at": [_START] * 4,
                    "source": ["fpl_api"] * 4,
                }
            )
        ),
    )
    stats_rows: list[dict[str, object]] = []
    appearances = {
        10: [(1, 90, 1, 1), (2, 90, 1, 0), (3, 90, 1, 1)],
        11: [(1, 90, 1, 0), (2, 60, 1, 1), (3, 90, 1, 0)],
        20: [(1, 90, 1, 0), (2, 90, 1, 1), (3, 90, 1, 0)],
        21: [(1, 75, 1, 1), (2, 90, 1, 0), (3, 90, 1, 0)],
    }
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
                    "available_at": available[gw - 1],
                    "source": "fpl_api",
                }
            )
    store.write("player_fixture_stats", _utc(pl.DataFrame(stats_rows)))
    # Closing-proxy odds for the eval GW (available_at=kickoff).
    store.write(
        "match_odds",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON],
                    "fixture_id": [3],
                    "team_h_key": [1],
                    "team_a_key": [2],
                    "kickoff_time": [kickoffs[2]],
                    "home_odds": [1.55],
                    "draw_odds": [4.10],
                    "away_odds": [5.50],
                    "over_25_odds": [1.70],
                    "under_25_odds": [2.20],
                    "odds_kind": ["closing_proxy"],
                    "available_at": [kickoffs[2]],
                    "source": ["football_data"],
                }
            )
        ),
    )
    return store


def _utc(frame: pl.DataFrame) -> pl.DataFrame:
    columns = [
        pl.col(name).cast(pl.Datetime("us", "UTC"))
        for name, dtype in frame.schema.items()
        if isinstance(dtype, pl.Datetime)
    ]
    return frame.with_columns(columns) if columns else frame
