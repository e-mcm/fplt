"""Predicted XI accuracy audit vs official starts."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from fplt.backtest.predicted_xi_audit import (
    DEFAULT_MIN_START_ACCURACY,
    build_audit_detail,
    format_audit_markdown,
    run_predicted_xi_audit,
    summarise_audit,
)
from fplt.data.predicted_lineups import ROLE_BENCH, ROLE_START
from fplt.data.store import DataStore
from tests.builders import build_league

_CAPTURE = datetime(2024, 8, 16, 12, tzinfo=UTC)
_KICKOFF = datetime(2024, 8, 17, 14, tzinfo=UTC)


def _lineups(
    *, roles: list[tuple[int, str]], source: str = "touchline_wire"
) -> pl.DataFrame:
    rows = {
        "season": [],
        "team_key": [],
        "fixture_id": [],
        "player_key": [],
        "element_id": [],
        "role": [],
        "confidence": [],
        "available_at": [],
        "source": [],
    }
    for player_key, role in roles:
        rows["season"].append("2024-25")
        rows["team_key"].append(3)
        rows["fixture_id"].append(1)
        rows["player_key"].append(player_key)
        rows["element_id"].append(player_key // 10)
        rows["role"].append(role)
        rows["confidence"].append(0.9)
        rows["available_at"].append(_CAPTURE)
        rows["source"].append(source)
    return pl.DataFrame(rows)


def test_audit_reports_start_precision_and_gate(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    store.write(
        "predicted_lineups",
        _lineups(roles=[(100, ROLE_START), (200, ROLE_START)]),
    )
    result = run_predicted_xi_audit(
        tmp_path,
        "2024-25",
        from_gw=1,
        to_gw=1,
        min_accuracy=0.5,
        output_dir=tmp_path / "out",
    )
    overall = result.summary.filter(
        (pl.col("metric") == "start_precision") & (pl.col("slice") == "all")
    )
    assert overall.height == 1
    assert overall["value"][0] == pytest.approx(1.0)
    assert result.passed_min_accuracy is True
    assert (tmp_path / "out" / "report.md").exists()
    assert "PASS" in format_audit_markdown(
        result.summary, passed=True, min_accuracy=DEFAULT_MIN_START_ACCURACY
    )


def test_audit_fails_gate_when_starts_wrong(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    # Player 999 never appears in fixture stats → drop from join; use 200 only
    # but flip: predict only a non-starter. Add a non-starting player via stats.
    store.write(
        "player_fixture_stats",
        pl.DataFrame(
            {
                "season": ["2024-25"],
                "gw": [1],
                "player_key": [300],
                "fixture_id": [1],
                "minutes": [0],
                "total_points": [0],
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
                "starts": [0],
                "influence": [0.0],
                "creativity": [0.0],
                "threat": [0.0],
                "bps": [0],
                "available_at": [_KICKOFF + timedelta(hours=3)],
                "source": ["fpl_api"],
            }
        ),
    )
    store.write(
        "predicted_lineups",
        _lineups(roles=[(300, ROLE_START)]),
    )
    result = run_predicted_xi_audit(
        tmp_path,
        "2024-25",
        from_gw=1,
        to_gw=1,
        min_accuracy=0.9,
        output_dir=tmp_path / "out",
    )
    overall = result.summary.filter(
        (pl.col("metric") == "start_precision") & (pl.col("slice") == "all")
    )
    assert overall["value"][0] == pytest.approx(0.0)
    assert result.passed_min_accuracy is False


def test_bench_started_rate_and_hours_slice(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    store.write(
        "predicted_lineups",
        _lineups(roles=[(100, ROLE_START), (200, ROLE_BENCH)]),
    )
    detail = build_audit_detail(store, "2024-25", from_gw=1, to_gw=1)
    assert detail.height == 1
    assert detail["hours_to_ko"][0] == pytest.approx(26.0)  # 12:00 → 14:00 next day
    summary = summarise_audit(detail, store, "2024-25", from_gw=1, to_gw=1)
    bench = summary.filter(pl.col("metric") == "bench_started_rate")
    assert bench.height == 1
    # Player 200 started in builders league.
    assert bench["value"][0] == pytest.approx(1.0)


def test_empty_lineups_yield_empty_detail(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    detail = build_audit_detail(store, "2024-25", from_gw=1, to_gw=1)
    assert detail.is_empty()


def test_bad_gw_range_refused(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="gameweeks"):
        run_predicted_xi_audit(tmp_path, "2024-25", from_gw=5, to_gw=1)
