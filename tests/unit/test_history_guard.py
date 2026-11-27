"""Training-history guard for live ops refits (#285)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from fplt.automation.history_guard import (
    HistoryGuardError,
    check_training_history,
    require_training_history,
    resolve_guard_target,
)
from fplt.data.store import DataStore
from fplt.types import GWKey

_START = datetime(2021, 8, 1, tzinfo=UTC)
_CUR_DEADLINE = datetime(2024, 9, 1, 17, 0, tzinfo=UTC)


def _stats_row(
    *,
    season: str,
    gw: int,
    available_at: datetime,
    fixture_id: int = 1,
) -> dict[str, object]:
    return {
        "season": season,
        "gw": gw,
        "player_key": 100,
        "fixture_id": fixture_id,
        "minutes": 90,
        "total_points": 2,
        "goals_scored": 0,
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
        "starts": 1,
        "influence": 1.0,
        "creativity": 1.0,
        "threat": 1.0,
        "bps": 10,
        "clearances_blocks_interceptions": 0,
        "tackles": 0,
        "recoveries": 0,
        "defensive_contribution": 0,
        "expected_goals": None,
        "expected_assists": None,
        "expected_goal_involvements": None,
        "expected_goals_conceded": None,
        "available_at": available_at,
        "source": "test",
    }


def _write_gameweeks(
    store: DataStore,
    rows: list[tuple[str, int, datetime, bool, bool]],
) -> None:
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": [r[0] for r in rows],
                "gw": [r[1] for r in rows],
                "deadline_time": [r[2] for r in rows],
                "finished": [r[3] for r in rows],
                "data_checked": [r[4] for r in rows],
                "available_at": [_START] * len(rows),
                "source": ["test"] * len(rows),
            }
        ),
    )


def _complete_store(tmp_path: Path, *, include_gw1_stats: bool = True) -> DataStore:
    """Three prior seasons + current GW1 finished with optional stats."""
    store = DataStore(tmp_path)
    prior = [
        ("2021-22", 1, _START + timedelta(days=20), True, True),
        ("2022-23", 1, _START + timedelta(days=400), True, True),
        ("2023-24", 1, _START + timedelta(days=780), True, True),
    ]
    current = [
        (
            "2024-25",
            1,
            _CUR_DEADLINE - timedelta(days=7),
            True,
            True,
        ),
        ("2024-25", 2, _CUR_DEADLINE, False, False),
    ]
    _write_gameweeks(store, [*prior, *current])
    if include_gw1_stats:
        store.write(
            "player_fixture_stats",
            pl.DataFrame(
                [
                    _stats_row(
                        season="2024-25",
                        gw=1,
                        available_at=_CUR_DEADLINE - timedelta(days=5),
                    )
                ]
            ),
        )
    return store


def test_guard_fails_on_zero_prior_seasons(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    _write_gameweeks(
        store,
        [
            ("2024-25", 1, _CUR_DEADLINE - timedelta(days=7), True, True),
            ("2024-25", 2, _CUR_DEADLINE, False, False),
        ],
    )
    store.write(
        "player_fixture_stats",
        pl.DataFrame(
            [
                _stats_row(
                    season="2024-25",
                    gw=1,
                    available_at=_CUR_DEADLINE - timedelta(days=5),
                )
            ]
        ),
    )
    result = check_training_history(
        store,
        season="2024-25",
        as_of=GWKey("2024-25", 2),
        cutoff=_CUR_DEADLINE,
        min_prior_seasons=3,
    )
    assert not result.ok
    assert result.prior_seasons == ()
    assert "need 3 prior seasons" in (result.reason or "")
    with pytest.raises(HistoryGuardError, match="prior seasons"):
        require_training_history(
            store,
            season="2024-25",
            as_of=GWKey("2024-25", 2),
            cutoff=_CUR_DEADLINE,
            min_prior_seasons=3,
        )


def test_guard_fails_on_missing_finished_gameweek(tmp_path: Path) -> None:
    store = _complete_store(tmp_path, include_gw1_stats=False)
    result = check_training_history(
        store,
        season="2024-25",
        as_of=GWKey("2024-25", 2),
        cutoff=_CUR_DEADLINE,
        min_prior_seasons=3,
    )
    assert not result.ok
    assert result.missing_finished_gws == (1,)
    assert "missing player_fixture_stats" in (result.reason or "")


def test_guard_passes_on_complete_fixture(tmp_path: Path) -> None:
    store = _complete_store(tmp_path)
    result = require_training_history(
        store,
        season="2024-25",
        as_of=GWKey("2024-25", 2),
        cutoff=_CUR_DEADLINE,
        min_prior_seasons=3,
    )
    assert result.ok
    assert result.prior_seasons == ("2021-22", "2022-23", "2023-24")
    assert result.missing_finished_gws == ()
    assert result.reason is None


def test_resolve_guard_target_prefers_upcoming_deadline(tmp_path: Path) -> None:
    store = _complete_store(tmp_path)
    season, as_of, cutoff = resolve_guard_target(
        store, _CUR_DEADLINE - timedelta(hours=1)
    )
    assert season == "2024-25"
    assert as_of == GWKey("2024-25", 2)
    assert cutoff == _CUR_DEADLINE


def test_resolve_guard_target_empty_store_raises(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    with pytest.raises(HistoryGuardError, match="no staged gameweeks"):
        resolve_guard_target(store, _CUR_DEADLINE)


def test_enforce_training_history_alerts_then_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.automation.history_guard import enforce_training_history
    from fplt.settings import Settings

    store = DataStore(tmp_path)
    titles: list[str] = []
    monkeypatch.setattr(
        "fplt.automation.history_guard.notify_recommendation",
        lambda _s, title, _m, **k: titles.append(title) or True,
    )
    with pytest.raises(HistoryGuardError, match="no staged gameweeks"):
        enforce_training_history(
            store,
            now=_CUR_DEADLINE,
            min_prior_seasons=3,
            settings=Settings(data_dir=tmp_path),
            notify=True,
            source="ops-tick",
        )
    assert titles == ["fplt history guard"]
