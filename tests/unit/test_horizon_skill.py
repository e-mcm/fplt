"""Skill-by-horizon curve reuses compare metrics; projections default to 10."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import polars as pl
import pytest

from fplt.backtest.component_eval import run_component_eval
from fplt.backtest.horizon_skill import (
    format_skill_curve,
    run_horizon_skill,
    skill_curve_from_ledger,
    skill_curve_from_summary,
    skill_rows_for_component_eval,
)
from fplt.backtest.horizons import DEFAULT_PROJECTION_HORIZON, DEFAULT_SKILL_STEPS
from fplt.backtest.walk_forward import horizon_for, walk_forward
from fplt.cli import main
from fplt.data.store import DataStore
from fplt.settings import get_settings
from fplt.types import GWKey

_PRIOR = "2023-24"
_TEST = "2024-25"
_START = datetime(2024, 7, 1, tzinfo=UTC)
_GW1 = datetime(2024, 8, 16, 17, tzinfo=UTC)
_GW1_RESULT = datetime(2024, 8, 17, 18, tzinfo=UTC)
_GW2 = datetime(2024, 8, 23, 17, tzinfo=UTC)
_GW2_RESULT = datetime(2024, 8, 24, 18, tzinfo=UTC)
_GW3 = datetime(2024, 8, 30, 17, tzinfo=UTC)
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


def test_projection_defaults_are_ten() -> None:
    assert DEFAULT_PROJECTION_HORIZON == 10
    assert DEFAULT_SKILL_STEPS == 10
    assert walk_forward.__defaults__ is not None
    assert walk_forward.__defaults__[0] == DEFAULT_PROJECTION_HORIZON
    assert len(horizon_for(GWKey(_TEST, 1), DEFAULT_PROJECTION_HORIZON)) == 10


def test_skill_curve_from_summary_ratios_and_prefers_walk_forward() -> None:
    summary = pl.DataFrame(
        {
            "model_name": ["m"] * 4,
            "model_id": ["id"] * 4,
            "mode": ["walk_forward", "walk_forward", "fixed", "fixed"],
            "horizon": [1, 2, 1, 2],
            "mae": [2.0, 3.0, 9.0, 9.0],
            "rmse": [4.0, 5.0, 9.0, 9.0],
            "n": [10, 8, 10, 8],
        }
    )
    curve = skill_curve_from_summary(summary)
    assert curve.height == 2
    assert curve["horizon"].to_list() == [1, 2]
    assert curve["mae"].to_list() == [2.0, 3.0]
    assert curve["rmse"].to_list() == [4.0, 5.0]
    assert curve["mae_vs_h1"].to_list() == pytest.approx([1.0, 1.5])
    assert curve["rmse_vs_h1"].to_list() == pytest.approx([1.0, 1.25])
    rows = skill_rows_for_component_eval(curve)
    assert {row["component"] for row in rows} == {"points"}
    assert {row["metric"] for row in rows} == {"mae", "rmse"}
    assert {row["slice"] for row in rows} == {"horizon=1", "horizon=2"}
    table = format_skill_curve(curve)
    assert "rmse" in table
    assert "horizon" in table


def test_skill_curve_empty_and_missing_columns() -> None:
    assert skill_curve_from_summary(pl.DataFrame()).is_empty()
    with pytest.raises(ValueError, match="missing"):
        skill_curve_from_summary(pl.DataFrame({"horizon": [1], "mae": [1.0]}))
    assert skill_curve_from_ledger(pl.DataFrame()).is_empty()


def test_run_horizon_skill_writes_curve(tmp_path: Path) -> None:
    _store(tmp_path)
    result = run_horizon_skill(
        tmp_path,
        _TEST,
        models=["rolling_mean_v1"],
        from_gw=1,
        to_gw=2,
        steps=2,
        seasons_back=2,
        output_dir=tmp_path / "skill",
    )
    assert (tmp_path / "skill" / "horizon_skill.json").is_file()
    assert (tmp_path / "skill" / "horizon_skill.parquet").is_file()
    assert not result.curve.is_empty()
    assert set(result.curve["horizon"].to_list()) >= {1}
    assert result.curve["n"].sum() > 0
    assert all(run.mode == "walk_forward" for run in result.compare.runs)


def test_eval_components_points_models_appends_horizon_rows(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = DataStore(tmp_path)
    # Minimal league so score/minutes suites can be skipped entirely.
    store.write(
        "gameweeks",
        _utc(
            pl.DataFrame(
                {
                    "season": [_TEST],
                    "gw": [1],
                    "deadline_time": [_GW1],
                    "finished": [True],
                    "data_checked": [True],
                    "available_at": [_START],
                    "source": ["fpl_api"],
                }
            )
        ),
    )
    curve = pl.DataFrame(
        {
            "model_name": ["rolling_mean_v1", "rolling_mean_v1"],
            "model_id": ["mid", "mid"],
            "horizon": [1, 2],
            "mae": [1.0, 1.5],
            "rmse": [1.2, 1.8],
            "n": [5, 4],
            "mae_vs_h1": [1.0, 1.5],
            "rmse_vs_h1": [1.0, 1.5],
        }
    )

    def fake_skill(*_a: Any, **_k: Any) -> MagicMock:
        out = MagicMock()
        out.curve = curve
        out.output_dir = tmp_path / "skill"
        return out

    monkeypatch.setattr("fplt.backtest.component_eval.run_horizon_skill", fake_skill)
    result = run_component_eval(
        tmp_path,
        _TEST,
        from_gw=1,
        to_gw=1,
        score_models=[],
        minutes_model=[],
        involvement_model=[],
        sims=[],
        points_models=["rolling_mean_v1"],
        steps=2,
        output_dir=tmp_path / "eval",
    )
    points = result.summary.filter(pl.col("component") == "points")
    assert points.height == 4
    assert set(points["slice"].to_list()) == {"horizon=1", "horizon=2"}
    assert not result.horizon_skill.is_empty()
    assert (tmp_path / "eval" / "horizon_skill.parquet").is_file()


def test_cli_horizon_skill(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _store(tmp_path)
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    main(
        [
            "horizon-skill",
            "--season",
            _TEST,
            "--models",
            "rolling_mean_v1",
            "--from-gw",
            "1",
            "--to-gw",
            "2",
            "--steps",
            "2",
            "--seasons-back",
            "2",
        ]
    )
    captured = capsys.readouterr()
    assert "horizon_skill.json" in captured.out
    get_settings.cache_clear()


def test_run_horizon_skill_rejects_bad_steps(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="steps"):
        run_horizon_skill(tmp_path, _TEST, steps=0)


def _store(root: Path) -> DataStore:
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
                    "source": ["fpl_api"],
                }
            )
        ),
    )
    store.write(
        "gameweeks",
        _utc(
            pl.DataFrame(
                {
                    "season": [_TEST, _TEST, _TEST],
                    "gw": [1, 2, 3],
                    "deadline_time": [_GW1, _GW2, _GW3],
                    "finished": [True, True, False],
                    "data_checked": [True, True, False],
                    "available_at": [_START, _START, _START],
                    "source": ["fpl_api"] * 3,
                }
            )
        ),
    )
    store.write(
        "fixtures",
        _utc(
            pl.DataFrame(
                {
                    "season": [_TEST, _TEST, _TEST],
                    "fixture_id": [101, 102, 103],
                    "gw": [1, 2, 3],
                    "team_h_key": [1, 1, 1],
                    "team_a_key": [2, 2, 2],
                    "kickoff_time": [_GW1, _GW2, _GW3],
                    "team_h_score": [1, 2, None],
                    "team_a_score": [0, 1, None],
                    "finished": [True, True, False],
                    "available_at": [_START, _START, _START],
                    "source": ["fpl_api"] * 3,
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
                    "source": ["fpl_api"],
                }
            )
        ),
    )
    rows = [
        (_PRIOR, 1, 1, 2, 90, _PRIOR_RESULT),
        (_TEST, 1, 101, 10, 90, _GW1_RESULT),
        (_TEST, 2, 102, 6, 90, _GW2_RESULT),
    ]
    store.write(
        "player_fixture_stats",
        _utc(
            pl.DataFrame(
                {
                    "season": [row[0] for row in rows],
                    "gw": [row[1] for row in rows],
                    "player_key": [100] * 3,
                    "fixture_id": [row[2] for row in rows],
                    "minutes": [row[4] for row in rows],
                    "total_points": [row[3] for row in rows],
                    "starts": [1, 1, 1],
                    "available_at": [row[5] for row in rows],
                    "source": ["fpl_api"] * 3,
                    **{name: [0, 0, 0] for name in _ZERO},
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
