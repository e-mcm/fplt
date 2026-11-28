"""Vaastav seasons land in the same staged tables, with degraded availability."""

import hashlib
import logging
import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.data.vaastav import (
    import_available,
    import_season,
    is_complete_vaastav_capture,
    latest_season_dirs,
)
from fplt.schemas.vaastav_source import VaastavSourceFile, VaastavSourceManifest
from fplt.types import GWKey

_KICKOFF = datetime(2021, 8, 14, 14, 0, tzinfo=UTC)
_KICKOFF_GW2 = datetime(2021, 8, 21, 14, 0, tzinfo=UTC)


def _write_complete_source(capture_dir: Path, *, seasons: list[str]) -> None:
    """Write a verified ``_source.json`` covering every file under ``data/``."""
    data = capture_dir / "data"
    entries: list[VaastavSourceFile] = []
    for path in sorted(data.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(capture_dir).as_posix()
        payload = path.read_bytes()
        entries.append(
            VaastavSourceFile(
                path=rel,
                bytes=len(payload),
                sha256=hashlib.sha256(payload).hexdigest(),
            )
        )
    stamp = capture_dir.name.removeprefix("captured_at=")
    captured_at = datetime.strptime(stamp, "%Y-%m-%dT%H-%M-%SZ").replace(tzinfo=UTC)
    manifest = VaastavSourceManifest(
        repo="https://example.test/vaastav.git",
        commit_sha="a" * 40,
        captured_at=captured_at,
        seasons=seasons,
        files=entries,
    )
    (capture_dir / "_source.json").write_text(
        manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )


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
            "value": [45, 51],
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


def test_named_club_uses_the_team_code(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "store")
    import_season(store, _write_season(tmp_path, team=["A", "B"]), "2021-22")
    assert store.read("players")["team_key"].sort().to_list() == [3, 7]


def test_status_stages_per_gw_value_as_price(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "store")
    import_season(store, _write_season(tmp_path), "2021-22")
    status = store.read("player_status").sort("player_key")
    assert status["status"].to_list() == ["a", "a"]
    assert status["ep_next"].to_list() == [None, None]
    assert status["chance_of_playing"].to_list() == [None, None]
    assert status["chance_of_playing_this_round"].to_list() == [None, None]
    assert status["news_added"].to_list() == [None, None]
    assert status["penalties_order"].to_list() == [None, None]
    assert status["corners_and_indirect_freekicks_order"].to_list() == [None, None]
    assert status["direct_freekicks_order"].to_list() == [None, None]
    assert status["price"].to_list() == [51, 45]
    assert status["available_at"].to_list() == [
        _KICKOFF_GW2 - timedelta(hours=2),
        _KICKOFF - timedelta(hours=2),
    ]
    players = store.read("players").sort("element_id")
    assert players["position"].to_list() == ["MID", "MID"]
    gw1 = PITData(store).view(GWKey("2021-22", 1))
    prices = (
        gw1.latest("player_status", ["season", "player_key"])
        .collect()
        .sort("player_key")
    )
    assert prices["price"].to_list() == [45]
    assert prices["player_key"].to_list() == [100]
    gw2 = PITData(store).view(GWKey("2021-22", 2))
    later = (
        gw2.latest("player_status", ["season", "player_key"])
        .collect()
        .sort("player_key")
    )
    assert later["price"].to_list() == [51, 45]
    assert later["player_key"].to_list() == [2, 100]


def test_zero_starts_are_kept(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "store")
    import_season(store, _write_season(tmp_path, starts=[0, 0]), "2021-22")
    stats = store.read("player_fixture_stats").sort("gw")
    assert stats["starts"].to_list() == [0, 0]
    assert stats["minutes"].to_list() == [90, 10]


def test_defensive_contribution_defaults_to_zero_when_absent(
    tmp_path: Path,
) -> None:
    store = DataStore(tmp_path / "store")
    import_season(store, _write_season(tmp_path), "2021-22")
    stats = store.read("player_fixture_stats")
    assert stats["defensive_contribution"].to_list() == [0, 0]


def test_defensive_contribution_staged_when_present(tmp_path: Path) -> None:
    season = _write_season(tmp_path)
    gw = pl.read_csv(season / "gws" / "merged_gw.csv")
    gw = gw.with_columns(pl.Series("defensive_contribution", [2, 0]))
    gw.write_csv(season / "gws" / "merged_gw.csv")
    store = DataStore(tmp_path / "store")
    import_season(store, season, "2021-22")
    stats = store.read("player_fixture_stats").sort("gw")
    assert stats["defensive_contribution"].to_list() == [2, 0]


def test_expected_xg_null_when_vaastav_columns_absent(tmp_path: Path) -> None:
    store = DataStore(tmp_path / "store")
    import_season(store, _write_season(tmp_path), "2021-22")
    stats = store.read("player_fixture_stats")
    assert stats["expected_goals"].to_list() == [None, None]
    assert stats["expected_assists"].to_list() == [None, None]
    assert stats["expected_goal_involvements"].to_list() == [None, None]
    assert stats["expected_goals_conceded"].to_list() == [None, None]


def test_expected_xg_staged_from_vaastav_when_present(tmp_path: Path) -> None:
    season = _write_season(tmp_path)
    gw = pl.read_csv(season / "gws" / "merged_gw.csv")
    gw = gw.with_columns(
        pl.Series("expected_goals", [0.55, 0.0]),
        pl.Series("expected_assists", [0.12, 0.3]),
        pl.Series("expected_goal_involvements", [0.67, 0.3]),
        pl.Series("expected_goals_conceded", [1.1, 0.8]),
    )
    gw.write_csv(season / "gws" / "merged_gw.csv")
    store = DataStore(tmp_path / "store")
    import_season(store, season, "2021-22")
    stats = store.read("player_fixture_stats").sort("gw")
    assert stats["expected_goals"].to_list() == [0.55, 0.0]
    assert stats["expected_assists"].to_list() == [0.12, 0.3]
    assert stats["expected_goal_involvements"].to_list() == [0.67, 0.3]
    assert stats["expected_goals_conceded"].to_list() == [1.1, 0.8]
    assert stats["available_at"].to_list() == [
        _KICKOFF + timedelta(hours=3),
        _KICKOFF_GW2 + timedelta(hours=3),
    ]


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


def _raw_season(
    data_dir: Path,
    season: str,
    *,
    stamp: str = "2026-11-27T12-00-00Z",
    finalize: bool = True,
) -> Path:
    """Place a minimal season under ``raw/vaastav/captured_at=…/data/<season>``."""
    capture = data_dir / "raw" / "vaastav" / f"captured_at={stamp}"
    root = capture / "data" / season
    (root / "gws").mkdir(parents=True)
    (root / "gws" / "merged_gw.csv").write_text("element,GW\n", encoding="utf-8")
    if finalize:
        # Merge seasons already present under this capture into the manifest.
        seasons = sorted(
            p.name for p in (capture / "data").iterdir() if p.is_dir() and "-" in p.name
        )
        _write_complete_source(capture, seasons=seasons)
    return root


def test_import_available_skips_early_seasons(tmp_path: Path) -> None:
    _raw_season(tmp_path, "2020-21")
    _raw_season(tmp_path, "2024-25")
    # Legacy data/<season>/ must be ignored.
    legacy = tmp_path / "2023-24" / "gws"
    legacy.mkdir(parents=True)
    (legacy / "merged_gw.csv").write_text("element,GW\n1,1\n", encoding="utf-8")
    store = DataStore(tmp_path / "store")
    assert import_available(store, tmp_path) == ["2024-25"]
    assert store.read("players").is_empty()


def test_import_available_uses_latest_capture_per_season(tmp_path: Path) -> None:
    first = _raw_season(
        tmp_path, "2022-23", stamp="2025-01-01T00-00-00Z", finalize=False
    )
    (first / "marker.txt").write_text("old", encoding="utf-8")
    _write_complete_source(first.parents[1], seasons=["2022-23"])
    second = _raw_season(
        tmp_path, "2022-23", stamp="2026-06-01T00-00-00Z", finalize=False
    )
    (second / "marker.txt").write_text("new", encoding="utf-8")
    _write_complete_source(second.parents[1], seasons=["2022-23"])
    dirs = latest_season_dirs(tmp_path)
    assert dirs["2022-23"] == second


def test_raw_and_legacy_layout_stage_equivalently(tmp_path: Path) -> None:
    """Same season files → same staged tables from legacy vs raw/vaastav."""
    legacy_root = tmp_path / "legacy"
    raw_root = tmp_path / "raw_layout"
    legacy_season = _write_season(legacy_root)
    capture = raw_root / "raw" / "vaastav" / "captured_at=2026-11-01T00-00-00Z"
    raw_season = capture / "data" / "2021-22"
    raw_season.parent.mkdir(parents=True)
    shutil.copytree(legacy_season, raw_season)
    _write_complete_source(capture, seasons=["2021-22"])

    legacy_store = DataStore(tmp_path / "legacy_store")
    import_season(legacy_store, legacy_season, "2021-22")
    raw_store = DataStore(tmp_path / "raw_store")
    assert import_available(raw_store, raw_root) == ["2021-22"]

    for table in (
        "players",
        "teams",
        "fixtures",
        "gameweeks",
        "player_status",
        "player_fixture_stats",
    ):
        left = legacy_store.read(table).sort(legacy_store.read(table).columns)
        right = raw_store.read(table).sort(raw_store.read(table).columns)
        assert left.equals(right), table


def test_pit_ignores_capture_stamp_for_history(tmp_path: Path) -> None:
    """A late captured_at still exposes 2021-22 rows by row-level available_at."""
    season = _write_season(tmp_path / "src")
    capture = tmp_path / "raw" / "vaastav" / "captured_at=2026-11-27T12-00-00Z"
    dest = capture / "data" / "2021-22"
    dest.parent.mkdir(parents=True)
    shutil.copytree(season, dest)
    _write_complete_source(capture, seasons=["2021-22"])
    store = DataStore(tmp_path / "store")
    assert import_available(store, tmp_path) == ["2021-22"]
    # Before kickoff+3h: results hidden; schedule visible at season_start.
    early = PITData(store).view(GWKey("2021-22", 1))
    assert early.table("player_fixture_stats").collect().is_empty()
    assert not early.latest("fixtures", by=["fixture_id"]).collect().is_empty()
    later = PITData(store).view(GWKey("2021-22", 2))
    assert later.table("player_fixture_stats").collect()["total_points"].to_list() == [
        8
    ]


def test_incomplete_capture_ignored_by_import_and_latest(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Altered payload with ``_source.json`` present is skipped (#353)."""
    good = _raw_season(tmp_path, "2021-22", stamp="2026-01-01T00-00-00Z")
    bad_root = _raw_season(
        tmp_path, "2024-25", stamp="2026-02-01T00-00-00Z", finalize=False
    )
    capture = bad_root.parents[1]
    _write_complete_source(capture, seasons=["2024-25"])
    # Corrupt a listed file after the manifest was written.
    (bad_root / "gws" / "merged_gw.csv").write_text("tampered\n", encoding="utf-8")
    assert is_complete_vaastav_capture(good.parents[1])
    assert not is_complete_vaastav_capture(capture)
    with caplog.at_level(logging.WARNING):
        dirs = latest_season_dirs(tmp_path)
    assert dirs == {"2021-22": good}
    assert "sha256 mismatch" in caplog.text
    store = DataStore(tmp_path / "store")
    assert import_available(store, tmp_path) == ["2021-22"]


def test_missing_listed_file_skips_capture(tmp_path: Path) -> None:
    """``_source.json`` listing a missing file → incomplete (#353)."""
    root = _raw_season(
        tmp_path, "2021-22", stamp="2026-03-01T00-00-00Z", finalize=False
    )
    capture = root.parents[1]
    _write_complete_source(capture, seasons=["2021-22"])
    (root / "gws" / "merged_gw.csv").unlink()
    assert not is_complete_vaastav_capture(capture)
    assert latest_season_dirs(tmp_path) == {}


def test_missing_source_json_skips_capture(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Capture dir without ``_source.json`` is incomplete (#353)."""
    _raw_season(tmp_path, "2021-22", stamp="2026-04-01T00-00-00Z", finalize=False)
    with caplog.at_level(logging.WARNING):
        assert latest_season_dirs(tmp_path) == {}
    assert "missing _source.json" in caplog.text


def test_invalid_source_json_skips_capture(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    """Corrupt ``_source.json`` is incomplete (#353)."""
    root = _raw_season(
        tmp_path, "2021-22", stamp="2026-05-01T00-00-00Z", finalize=False
    )
    capture = root.parents[1]
    (capture / "_source.json").write_text("{not-json\n", encoding="utf-8")
    with caplog.at_level(logging.WARNING):
        assert not is_complete_vaastav_capture(capture)
    assert "invalid _source.json" in caplog.text
    assert latest_season_dirs(tmp_path) == {}
