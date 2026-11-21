"""football-data E0 closing odds stage into match_odds without network."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fplt.data.football_data import (
    attach_fixture_ids,
    default_team_map_path,
    e0_raw_path,
    ingest_football_data,
    load_team_names,
    parse_e0_csv,
    parse_kickoff,
    season_label_from_token,
    season_token,
    stage_match_odds,
)
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.schemas.staged import MatchOddsSchema
from fplt.types import GWKey

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "football_data"
_CSV = _FIXTURES / "E0.csv"
_MAP = _FIXTURES / "team_names.yaml"
_TEAMS = {
    "Arsenal": 3,
    "Tottenham": 6,
    "Man United": 1,
    "Fulham": 54,
    "Liverpool": 14,
    "Ipswich": 40,
}


def test_season_token_helpers() -> None:
    assert season_token("2024-25") == "2425"
    assert season_label_from_token("2425") == "2024-25"
    with pytest.raises(ValueError, match="four digits"):
        season_label_from_token("24")


def test_load_team_names_reads_yaml() -> None:
    assert load_team_names(_MAP) == _TEAMS
    assert "Arsenal" in load_team_names(default_team_map_path())


def test_parse_kickoff_london_to_utc() -> None:
    # 15:00 BST on 17 Aug 2024 → 14:00 UTC
    assert parse_kickoff("17/08/2024", "15:00") == datetime(
        2024, 8, 17, 14, 0, tzinfo=UTC
    )
    # Missing time → 15:00 local default
    assert parse_kickoff("18/08/2024", None) == datetime(2024, 8, 18, 14, 0, tzinfo=UTC)


def test_parse_e0_csv_closing_proxy(tmp_path: Path) -> None:
    frame = parse_e0_csv(_CSV, "2024-25", _TEAMS)
    assert frame.height == 3
    MatchOddsSchema.validate(frame)
    first = frame.filter(pl.col("team_h_key") == 3).row(0, named=True)
    assert first["home_odds"] == pytest.approx(1.60)
    assert first["draw_odds"] == pytest.approx(4.10)
    assert first["away_odds"] == pytest.approx(5.50)
    assert first["over_25_odds"] == pytest.approx(1.65)
    assert first["under_25_odds"] == pytest.approx(2.30)
    assert first["odds_kind"] == "closing_proxy"
    assert first["source"] == "football_data"
    assert first["available_at"] == first["kickoff_time"]
    assert first["available_at"] == datetime(2024, 8, 17, 14, 0, tzinfo=UTC)
    liverpool = frame.filter(pl.col("team_h_key") == 14).row(0, named=True)
    assert liverpool["over_25_odds"] is None
    assert liverpool["under_25_odds"] is None


def test_unknown_team_is_rejected() -> None:
    with pytest.raises(KeyError, match="Arsenal"):
        parse_e0_csv(_CSV, "2024-25", {"Tottenham": 6})


def test_stage_and_pit_hides_future_odds(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    _seed_deadlines(store)
    staged = stage_match_odds(store, _CSV, "2024-25", _TEAMS, join_fixtures=False)
    assert staged.height == 3
    assert store.read("match_odds").height == 3
    # Before first kickoff: nothing visible
    early = PITData(store).view(GWKey("2024-25", 1)).table("match_odds").collect()
    # GW1 deadline is before 17 Aug kickoffs in _seed_deadlines
    assert early.height == 0
    late = PITData(store).view(GWKey("2024-25", 2)).table("match_odds").collect()
    assert late.height == 3


def test_attach_fixture_ids_on_date_and_teams(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    kickoff = datetime(2024, 8, 17, 14, 0, tzinfo=UTC)
    store.write(
        "fixtures",
        pl.DataFrame(
            {
                "season": ["2024-25"],
                "fixture_id": [99],
                "gw": [1],
                "team_h_key": [3],
                "team_a_key": [6],
                "kickoff_time": [kickoff],
                "team_h_score": [None],
                "team_a_score": [None],
                "finished": [False],
                "available_at": [datetime(2024, 8, 1, tzinfo=UTC)],
                "source": ["fpl_api"],
            }
        ),
    )
    odds = parse_e0_csv(_CSV, "2024-25", _TEAMS)
    joined = attach_fixture_ids(odds, store.read("fixtures"))
    arsenal = joined.filter(pl.col("team_h_key") == 3).row(0, named=True)
    assert arsenal["fixture_id"] == 99
    others = joined.filter(pl.col("team_h_key") != 3)
    assert others["fixture_id"].null_count() == others.height


def test_ingest_football_data_from_raw(tmp_path: Path) -> None:
    raw = tmp_path / "raw"
    dest = e0_raw_path(raw, "2024-25")
    dest.parent.mkdir(parents=True)
    dest.write_text(_CSV.read_text(encoding="utf-8"), encoding="utf-8")
    store = DataStore(tmp_path / "data")
    assert ingest_football_data(store, raw, _TEAMS, join_fixtures=False) == 1
    assert store.read("match_odds").height == 3
    assert ingest_football_data(store, raw, _TEAMS, join_fixtures=False) == 1
    assert store.read("match_odds").height == 3
    assert ingest_football_data(store, raw, _TEAMS, seasons=["2023-24"]) == 0


def test_repo_team_map_covers_fixture_names() -> None:
    repo = load_team_names(default_team_map_path())
    for name in _TEAMS:
        assert name in repo


def _seed_deadlines(store: DataStore) -> None:
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": ["2024-25", "2024-25"],
                "gw": [1, 2],
                "deadline_time": [
                    datetime(2024, 8, 16, 17, 0, tzinfo=UTC),
                    datetime(2024, 8, 20, 17, 0, tzinfo=UTC),
                ],
                "finished": [False, False],
                "data_checked": [False, False],
                "available_at": [
                    datetime(2024, 8, 1, tzinfo=UTC),
                    datetime(2024, 8, 1, tzinfo=UTC),
                ],
                "source": ["fpl_api", "fpl_api"],
            }
        ),
    )
