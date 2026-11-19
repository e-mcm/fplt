"""Vaastav seasons land in the same staged tables, with degraded availability."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.data.vaastav import import_season
from fplt.types import GWKey

_KICKOFF = datetime(2021, 8, 14, 14, 0, tzinfo=UTC)
_KICKOFF_GW2 = datetime(2021, 8, 21, 14, 0, tzinfo=UTC)


def _write_season(
    root: Path,
    *,
    starts: list[int] | None = None,
    team: list[object] | None = None,
) -> Path:
    season = root / "2021-22"
    (season / "gws").mkdir(parents=True)
    pl.DataFrame(
        {
            "id": [1],
            "code": [100],
            "element_type": [3],
            "web_name": ["A"],
            "team": [1],
        }
    ).write_csv(season / "players_raw.csv")
    pl.DataFrame(
        {"id": [1, 2], "code": [3, 7], "name": ["A", "B"], "short_name": ["A", "B"]}
    ).write_csv(season / "teams.csv")
    pl.DataFrame(
        {
            "id": [50, 51],
            "event": [1, 2],
            "team_h": [1, 2],
            "team_a": [2, 1],
            "kickoff_time": [
                _KICKOFF.strftime("%Y-%m-%dT%H:%M:%SZ"),
                _KICKOFF_GW2.strftime("%Y-%m-%dT%H:%M:%SZ"),
            ],
            "finished": [True, False],
            "team_h_score": [2, None],
            "team_a_score": [0, None],
        }
    ).write_csv(season / "fixtures.csv")
    gw_starts = [1, 1] if starts is None else starts
    gw_team = [1, 2] if team is None else team
    pl.DataFrame(
        {
            "name": ["A", "B"],
            "element": [1, 2],
            "fixture": [50, 51],
            "GW": [1, 2],
            "was_home": [True, False],
            "team": gw_team,
            "opponent_team": [2, 1],
            "minutes": [90, 10],
            "total_points": [8, 1],
            "goals_scored": [1, 0],
            "assists": [0, 0],
            "clean_sheets": [1, 0],
            "goals_conceded": [0, 0],
            "saves": [0, 0],
            "bonus": [0, 0],
            "yellow_cards": [0, 0],
            "red_cards": [0, 0],
            "own_goals": [0, 0],
            "penalties_saved": [0, 0],
            "penalties_missed": [0, 0],
            "starts": gw_starts,
            "kickoff_time": [
                _KICKOFF.strftime("%Y-%m-%dT%H:%M:%SZ"),
                _KICKOFF_GW2.strftime("%Y-%m-%dT%H:%M:%SZ"),
            ],
            "position": ["MID", "NOPE"],
        }
    ).write_csv(season / "gws" / "merged_gw.csv")
    return season


def test_results_stay_hidden_until_after_kickoff(tmp_path: Path) -> None:
    season = _write_season(tmp_path)
    store = DataStore(tmp_path / "store")
    import_season(store, season, "2021-22")
    view = PITData(store).view(GWKey("2021-22", 1))
    assert view.table("player_fixture_stats").collect().is_empty()
    fixtures = view.latest("fixtures", by=["fixture_id"]).collect().sort("fixture_id")
    assert fixtures["team_h_score"].to_list() == [None, None]
    assert store.read("players")["player_key"].sort().to_list() == [2, 100]
    assert store.read("players")["source"].to_list() == ["vaastav", "vaastav"]
    assert store.read("players")["team_key"].sort().to_list() == [3, 7]
    assert store.read("teams")["team_key"].sort().to_list() == [3, 7]
    later = PITData(store).view(GWKey("2021-22", 2))
    visible = later.table("player_fixture_stats").collect()
    assert visible["total_points"].to_list() == [8]
    assert visible["player_key"].to_list() == [100]
    deadline = store.read("gameweeks").sort("gw")
    assert deadline["data_checked"].to_list() == [True, False]
    assert deadline["deadline_time"][0] == _KICKOFF - timedelta(hours=2)


def test_status_is_a_single_degraded_row(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "store")
    import_season(store, _write_season(tmp_path), "2021-22")
    status = store.read("player_status").sort("player_key")
    assert status["status"].to_list() == ["a", "a"]
    assert status["ep_next"].to_list() == [None, None]
    assert status["chance_of_playing"].to_list() == [None, None]
    assert status["price"].to_list() == [0, 0]
    players = store.read("players").sort("element_id")
    assert players["position"].to_list() == ["MID", "MID"]


def test_zero_starts_are_kept(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "store")
    import_season(store, _write_season(tmp_path, starts=[0, 0]), "2021-22")
    stats = store.read("player_fixture_stats").sort("gw")
    assert stats["starts"].to_list() == [0, 0]
    assert stats["minutes"].to_list() == [90, 10]


def test_seasons_before_2021_and_missing_files_are_skipped(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "store")
    early = tmp_path / "2020-21"
    (early / "gws").mkdir(parents=True)
    (early / "gws" / "merged_gw.csv").write_text("element,GW\n1,1\n", encoding="utf-8")
    import_season(store, early, "2020-21")
    import_season(store, tmp_path / "absent", "2024-25")
    assert store.read("players").is_empty()
    assert not store.path("players").exists()


def test_latin1_gameweek_file_and_missing_starts(tmp_path: Path) -> None:
    season = tmp_path / "2022-23"
    (season / "gws").mkdir(parents=True)
    text = (
        "name,element,GW,was_home,team,minutes,total_points,position,kickoff_time\n"
        "Caf\xe9,1,1,True,1,70,5,FWD,2022-08-06T14:00:00Z\n"
    )
    (season / "gws" / "merged_gw.csv").write_bytes(text.encode("latin1"))
    store = DataStore(tmp_path / "store")
    import_season(store, season, "2022-23")
    players = store.read("players")
    assert players["web_name"].to_list() == ["Café"]
    assert players["player_key"].to_list() == [1]
    stats = store.read("player_fixture_stats")
    assert stats["starts"].to_list() == [1]
    assert stats["fixture_id"].to_list() == [0]
    assert stats["available_at"][0] == datetime(2022, 8, 6, 17, 0, tzinfo=UTC)


def test_blank_team_uses_the_row_kickoff(tmp_path: Path) -> None:
    season = tmp_path / "2023-24"
    (season / "gws").mkdir(parents=True)
    pl.DataFrame(
        {
            "name": ["A"],
            "element": [4],
            "fixture": [9],
            "GW": [1],
            "was_home": [True],
            "team": ["n/a"],
            "minutes": [0],
            "position": ["GK"],
            "kickoff_time": ["2023-08-12 11:30:00"],
        }
    ).write_csv(season / "gws" / "merged_gw.csv")
    pl.DataFrame(
        {
            "id": [None],
            "code": [None],
            "name": ["Unused"],
            "short_name": [None],
        }
    ).write_csv(season / "teams.csv")
    pl.DataFrame(
        {
            "id": [9],
            "event": [None],
            "team_h": [1],
            "team_a": [2],
            "kickoff_time": [None],
            "finished": [False],
            "team_h_score": [None],
            "team_a_score": [None],
        }
    ).write_csv(season / "fixtures.csv")
    store = DataStore(tmp_path / "store")
    import_season(store, season, "2023-24")
    assert store.read("teams").is_empty()
    assert store.read("fixtures").is_empty()
    assert store.read("gameweeks").is_empty()
    stats = store.read("player_fixture_stats")
    assert stats["fixture_id"].to_list() == [9]
    assert stats["available_at"][0] == datetime(2023, 8, 12, 14, 30, tzinfo=UTC)
    assert stats["starts"].to_list() == [0]


def test_header_only_gameweek_writes_empty_tables(tmp_path: Path) -> None:
    season = tmp_path / "2024-25"
    (season / "gws").mkdir(parents=True)
    (season / "gws" / "merged_gw.csv").write_text("element,GW\n", encoding="utf-8")
    store = DataStore(tmp_path / "store")
    import_season(store, season, "2024-25")
    assert store.read("players").is_empty()
    assert store.read("player_status").is_empty()
    assert store.read("player_fixture_stats").is_empty()
