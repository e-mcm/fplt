"""Historical backtests reuse one fit or refit at every deadline."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fplt.backtest.ledger import read_predictions
from fplt.backtest.run import run_backtest
from fplt.cli import main
from fplt.data.store import DataStore
from fplt.models.direct import LGBMDirectPredictor
from fplt.models.spec import get_model, load_model_specs
from fplt.settings import get_settings
from fplt.store import Store

_PRIOR = "2023-24"
_TEST = "2024-25"
_START = datetime(2024, 7, 1, tzinfo=UTC)
_GW1 = datetime(2024, 8, 16, 17, tzinfo=UTC)
_GW1_RESULT = datetime(2024, 8, 17, 18, tzinfo=UTC)
_GW2 = datetime(2024, 8, 23, 17, tzinfo=UTC)
_PRIOR_RESULT = datetime(2023, 8, 12, 17, tzinfo=UTC)
_ZERO = (
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "bonus",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
)


def test_lgbm_direct_is_registered_via_backtest_run() -> None:
    """Importing run_backtest must register kind lgbm_direct for CLI backtests."""
    specs = load_model_specs()
    assert get_model(specs["lgbm_direct_v1"].kind) is LGBMDirectPredictor


def test_frozen_fit_ignores_the_test_season(tmp_path: Path) -> None:
    _store(tmp_path)
    ran = run_backtest(
        tmp_path,
        _TEST,
        1,
        2,
        ["rolling_mean_v1"],
        refit="never",
        seasons_back=2,
    )
    points = _points(tmp_path, ran[0].model_id)
    assert points[1] == pytest.approx(2.0)
    assert points[2] == pytest.approx(2.0)
    assert (
        Store(tmp_path).ledger().filter(pl.col("target_gw") == 1)["actual_pts"][0] == 10
    )


def test_walk_forward_picks_up_the_test_season(tmp_path: Path) -> None:
    _store(tmp_path)
    ran = run_backtest(
        tmp_path,
        _TEST,
        1,
        2,
        ["rolling_mean_v1"],
        refit="every_gw",
        seasons_back=2,
    )
    points = _points(tmp_path, ran[0].model_id)
    assert points[1] == pytest.approx(2.0)
    assert points[2] == pytest.approx(6.0)


def test_command_fills_checked_actuals(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _store(tmp_path)
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    main(
        [
            "backtest",
            "--season",
            _TEST,
            "--from-gw",
            "1",
            "--to-gw",
            "2",
            "--models",
            "rolling_mean_v1",
            "--refit",
            "never",
            "--seasons-back",
            "2",
        ]
    )
    printed = capsys.readouterr().out
    assert (
        "rolling_mean_v1 refit=never seasons_back=2 time_decay=None model_id="
        in printed
    )
    actual = Store(tmp_path).ledger().filter(pl.col("target_gw") == 1)
    assert actual["actual_pts"][0] == 10
    get_settings.cache_clear()


def test_ep_next_on_vaastav_writes_nothing(tmp_path: Path) -> None:
    _store(tmp_path, source="vaastav")
    run_backtest(tmp_path, _TEST, 1, 2, ["ep_next_v1"], seasons_back=2)
    assert read_predictions(tmp_path).is_empty()


def test_bad_requests_are_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown model"):
        run_backtest(tmp_path, _TEST, 1, 1, ["missing"])
    with pytest.raises(ValueError, match="seasons_back"):
        run_backtest(tmp_path, _TEST, 1, 1, ["rolling_mean_v1"], seasons_back=0)
    with pytest.raises(ValueError, match="time_decay"):
        run_backtest(tmp_path, _TEST, 1, 1, ["rolling_mean_v1"], time_decay=-1.0)
    with pytest.raises(ValueError, match="gameweeks"):
        run_backtest(tmp_path, _TEST, 3, 1, ["rolling_mean_v1"])


def _points(root: Path, model_id: str) -> dict[int, float]:
    frame = read_predictions(root).filter(pl.col("model_id") == model_id)
    selected = frame.select("as_of_gw", "xpts").iter_rows()
    return {int(gw): float(points) for gw, points in selected}


def _store(root: Path, source: str = "fpl_api") -> DataStore:
    store = DataStore(root)
    store.write(
        "players",
        _utc(
            pl.DataFrame(
                {
                    "season": [_TEST],
                    "player_key": [100],
                    "element_id": [10],
                    "team_key": [1],
                    "position": ["MID"],
                    "web_name": ["A"],
                    "available_at": [_START],
                    "source": [source],
                }
            )
        ),
    )
    store.write(
        "gameweeks",
        _utc(
            pl.DataFrame(
                {
                    "season": [_TEST, _TEST],
                    "gw": [1, 2],
                    "deadline_time": [_GW1, _GW2],
                    "finished": [True, False],
                    "data_checked": [True, False],
                    "available_at": [_START, _START],
                    "source": [source, source],
                }
            )
        ),
    )
    store.write(
        "fixtures",
        _utc(
            pl.DataFrame(
                {
                    "season": [_TEST, _TEST],
                    "fixture_id": [101, 102],
                    "gw": [1, 2],
                    "team_h_key": [1, 1],
                    "team_a_key": [2, 2],
                    "kickoff_time": [_GW1, _GW2],
                    "team_h_score": [1, None],
                    "team_a_score": [0, None],
                    "finished": [True, False],
                    "available_at": [_START, _START],
                    "source": [source, source],
                }
            )
        ),
    )
    store.write(
        "player_status",
        _utc(
            pl.DataFrame(
                {
                    "season": [_TEST],
                    "player_key": [100],
                    "status": ["a"],
                    "news": [""],
                    "price": [50],
                    "ep_next": pl.Series("ep_next", [None], dtype=pl.Float64),
                    "chance_of_playing": pl.Series(
                        "chance_of_playing", [None], dtype=pl.Float64
                    ),
                    "available_at": [_START],
                    "source": [source],
                }
            )
        ),
    )
    store.write("player_fixture_stats", _utc(_appearances(source)))
    return store


def _appearances(source: str) -> pl.DataFrame:
    rows = [
        (_PRIOR, 1, 1, 2, 90, _PRIOR_RESULT),
        (_TEST, 1, 101, 10, 90, _GW1_RESULT),
    ]
    return pl.DataFrame(
        {
            "season": [row[0] for row in rows],
            "gw": [row[1] for row in rows],
            "player_key": [100, 100],
            "fixture_id": [row[2] for row in rows],
            "minutes": [row[4] for row in rows],
            "total_points": [row[3] for row in rows],
            "starts": [1, 1],
            "available_at": [row[5] for row in rows],
            "source": [source, source],
            **{name: [0, 0] for name in _ZERO},
        }
    )


def _utc(frame: pl.DataFrame) -> pl.DataFrame:
    columns = [
        pl.col(name).cast(pl.Datetime("us", "UTC"))
        for name, dtype in frame.schema.items()
        if isinstance(dtype, pl.Datetime)
    ]
    return frame.with_columns(columns) if columns else frame
