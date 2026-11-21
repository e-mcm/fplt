"""OpenFPL sample builder matches the frozen xscaler column set."""

from __future__ import annotations

from datetime import UTC, datetime

import polars as pl
import pytest

from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.external.openfpl_columns import OPENFPL_FEATURE_NAMES
from fplt.models.external.openfpl_samples import (
    assert_openfpl_features,
    build_openfpl_features,
)
from fplt.types import GWKey
from tests.builders import build_league

_AS_OF = GWKey("2024-25", 2)


def test_feature_names_match_published_scaler_count() -> None:
    assert len(OPENFPL_FEATURE_NAMES) == 228
    assert OPENFPL_FEATURE_NAMES[0] == "player fpl points 1"
    assert OPENFPL_FEATURE_NAMES[-1] == "status opponent league rank"


def test_assert_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="mismatch"):
        assert_openfpl_features({"player fpl points 1": 1.0})


def test_build_features_covers_scaler_and_uses_history(tmp_path) -> None:
    store = build_league(DataStore(tmp_path))
    # Extra finished GW so rolling windows see history before GW2.
    result = datetime(2024, 8, 18, 17, tzinfo=UTC)
    store.write(
        "player_fixture_stats",
        pl.DataFrame(
            {
                "season": ["2024-25", "2024-25"],
                "gw": [1, 1],
                "player_key": [100, 200],
                "fixture_id": [1, 1],
                "minutes": [90, 90],
                "total_points": [8, 2],
                "goals_scored": [1, 0],
                "assists": [0, 0],
                "clean_sheets": [1, 0],
                "goals_conceded": [0, 2],
                "saves": [0, 0],
                "bonus": [2, 0],
                "yellow_cards": [0, 0],
                "red_cards": [0, 0],
                "own_goals": [0, 0],
                "penalties_saved": [0, 0],
                "penalties_missed": [0, 0],
                "starts": [1, 1],
                "influence": [20.0, 5.0],
                "creativity": [10.0, 2.0],
                "threat": [15.0, 1.0],
                "bps": [25, 5],
                "available_at": [result, result],
                "source": ["fpl_api", "fpl_api"],
            }
        ),
    )
    store.write(
        "understat_player_matches",
        pl.DataFrame(
            {
                "season": ["2024-25", "2024-25"],
                "match_id": [9, 9],
                "understat_player_id": [1, 2],
                "player_key": [100, 200],
                "team_key": [3, 7],
                "xg": [0.5, 0.1],
                "xa": [0.2, 0.0],
                "shots": [3, 1],
                "key_passes": [2, 0],
                "xg_chain": [0.7, 0.1],
                "xg_buildup": [0.3, 0.0],
                "minutes": [90, 90],
                "available_at": [result, result],
                "source": ["understat", "understat"],
            }
        ),
    )
    store.write(
        "understat_team_matches",
        pl.DataFrame(
            {
                "season": ["2024-25", "2024-25"],
                "match_id": [9, 9],
                "team_key": [3, 7],
                "deep": [8.0, 3.0],
                "deep_allowed": [3.0, 8.0],
                "ppda_att": [240.0, 310.0],
                "ppda_def": [20.0, 18.0],
                "ppda_allowed_att": [310.0, 240.0],
                "ppda_allowed_def": [18.0, 20.0],
                "xg": [1.4, 0.6],
                "xga": [0.6, 1.4],
                "available_at": [result, result],
                "source": ["understat", "understat"],
            }
        ),
    )
    view = PITData(store).view(_AS_OF)
    request = pl.DataFrame(
        {
            "season": ["2024-25"],
            "gw": [2],
            "player_key": [100],
            "fixture_id": [2],
        }
    )
    features = build_openfpl_features(view, request)
    assert len(features) == 1
    row = features[0]
    assert tuple(row) == OPENFPL_FEATURE_NAMES
    assert row["player fpl points 1"] == pytest.approx(8.0)
    assert row["player influence 1"] == pytest.approx(20.0)
    assert row["player shots 1"] == pytest.approx(3.0)
    assert row["player xg 1"] == pytest.approx(0.5)
    assert row["team xg 1"] == pytest.approx(1.4)
    assert row["opponent xga 1"] == pytest.approx(1.4)
    assert row["status player availability"] == pytest.approx(1.0)
    assert row["team deep 1"] == pytest.approx(8.0)
    assert row["team deep allowed 1"] == pytest.approx(3.0)
    assert row["team ppda att 1"] == pytest.approx(240.0)
    assert row["team ppda def 1"] == pytest.approx(20.0)
    assert row["opponent deep 1"] == pytest.approx(3.0)
    assert row["opponent ppda allowed att 1"] == pytest.approx(240.0)


def test_build_features_deep_ppda_zero_without_team_table(tmp_path) -> None:
    store = build_league(DataStore(tmp_path))
    view = PITData(store).view(GWKey("2024-25", 1))
    request = pl.DataFrame(
        {
            "season": ["2024-25"],
            "gw": [1],
            "player_key": [100],
            "fixture_id": [1],
        }
    )
    row = build_openfpl_features(view, request)[0]
    assert row["team deep 1"] == 0.0
    assert row["team ppda att 1"] == 0.0


def test_build_features_empty_request(tmp_path) -> None:
    store = build_league(DataStore(tmp_path))
    view = PITData(store).view(GWKey("2024-25", 1))
    assert build_openfpl_features(view, pl.DataFrame()) == []
