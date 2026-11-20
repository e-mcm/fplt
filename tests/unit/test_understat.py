"""Understat xG is staged with a clock, and later matches stay hidden."""

import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.data.understat import stage_understat
from fplt.features.registry import get_feature, materialise_feature
from fplt.types import GWKey

_WHEN = datetime(2024, 8, 1, tzinfo=UTC)
_EARLY = datetime(2024, 8, 10, tzinfo=UTC)
_LATE = datetime(2024, 8, 20, tzinfo=UTC)
_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "understat_match.json"
_PLAYERS = {501: 100, 502: 200}
_TEAMS = {10: 3, 20: 7}


def test_visible_matches_set_the_shares(tmp_path: Path) -> None:
    store = _store(tmp_path)
    matches = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    staged = stage_understat(store, matches, "2024-25", _EARLY, _PLAYERS, _TEAMS)
    assert staged.filter(pl.col("player_key").is_null()).height == 1
    stage_understat(store, [_later()], "2024-25", _LATE, _PLAYERS, _TEAMS)
    early = _shares(store, 1)
    assert early["player_key"].sort().to_list() == [100, 200]
    first = early.filter(pl.col("player_key") == 100).row(0, named=True)
    assert first["xg_per_90"] == pytest.approx(0.5)
    assert first["xa_per_90"] == pytest.approx(0.1)
    assert first["xg_share"] == pytest.approx(1.0)
    assert first["xa_share"] == pytest.approx(0.1 / 0.3)
    second = early.filter(pl.col("player_key") == 200).row(0, named=True)
    assert second["xa_per_90"] == pytest.approx(0.4)
    assert second["xg_share"] == pytest.approx(0.0)
    later = _shares(store, 2).filter(pl.col("player_key") == 100).row(0, named=True)
    assert later["xg_share"] == pytest.approx(0.5)
    assert later["xg_per_90"] == pytest.approx(0.25)


def test_an_unknown_team_is_rejected(tmp_path: Path) -> None:
    matches = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    with pytest.raises(KeyError, match="team 10"):
        stage_understat(DataStore(tmp_path), matches, "2024-25", _EARLY, _PLAYERS, {})


def test_materialise_keeps_the_first_file(tmp_path: Path) -> None:
    store = _store(tmp_path)
    matches = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    stage_understat(store, matches, "2024-25", _EARLY, _PLAYERS, _TEAMS)
    view = PITData(store).view(GWKey("2024-25", 1))
    path = materialise_feature(view, "understat_share", 1, tmp_path)
    assert path.name == "part.parquet"
    assert "v=1" in path.parts
    with pytest.raises(FileExistsError):
        materialise_feature(view, "understat_share", 1, tmp_path)
    with pytest.raises(KeyError, match="not registered"):
        get_feature("understat_share", 2)


def _shares(store: DataStore, gameweek: int):
    from fplt.features.understat import understat_share

    view = PITData(store).view(GWKey("2024-25", gameweek))
    return understat_share(view)


def _later() -> dict[str, object]:
    return {
        "id": 2,
        "team_h": 10,
        "team_a": 20,
        "h": [
            {"id": 501, "xg": 0.0, "xa": 0.0, "time": 90},
            {"id": 502, "xg": 0.5, "xa": 0.0, "time": 90},
        ],
        "a": [],
    }


def _store(root: Path) -> DataStore:
    store = DataStore(root)
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": ["2024-25", "2024-25"],
                "gw": [1, 2],
                "deadline_time": [
                    datetime(2024, 8, 16, 18, tzinfo=UTC),
                    datetime(2024, 8, 23, 18, tzinfo=UTC),
                ],
                "finished": [True, False],
                "data_checked": [True, False],
                "available_at": [_WHEN, _WHEN],
                "source": ["test", "test"],
            }
        ),
    )
    return store
