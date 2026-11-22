"""Staged assets ingest a snapshot, and the sensor requests a run for new files."""

import gzip
import json
from pathlib import Path

import dagster as dg
import polars as pl
import pytest

from fplt.data.snapshots import SnapshotFetchError
from fplt.data.store import DataStore
from fplt.orchestration import defs
from fplt.orchestration.assets import (
    match_odds_ingest,
    raw_snapshots,
    staged_assets,
    staged_ingest,
    understat_ingest,
)
from fplt.orchestration.checks import schema_check_result
from fplt.orchestration.sensors import (
    api_football_sensor,
    match_odds_sensor,
    snapshots_sensor,
    understat_sensor,
)
from fplt.settings import get_settings

_BOOTSTRAP = {
    "teams": [{"id": 1, "code": 3, "name": "Arsenal", "short_name": "ARS"}],
    "elements": [
        {
            "id": 10,
            "code": 100,
            "team": 1,
            "element_type": 3,
            "web_name": "Saka",
            "status": "a",
            "news": "",
            "now_cost": 100,
            "ep_next": "6.0",
            "chance_of_playing_next_round": 100,
        }
    ],
    "events": [
        {
            "id": 1,
            "deadline_time": "2024-08-16T17:30:00Z",
            "finished": False,
            "data_checked": False,
        }
    ],
}
_FIXTURES = [
    {
        "id": 50,
        "event": 1,
        "team_h": 1,
        "team_a": 1,
        "kickoff_time": "2024-08-17T14:00:00Z",
        "finished": False,
        "team_h_score": None,
        "team_a_score": None,
    }
]


def _use_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()


def test_schema_check_rejects_a_short_frame() -> None:
    result = schema_check_result("players", pl.DataFrame({"season": ["2024-25"]}))
    assert result.passed is False


def test_materialize_stages_a_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_data_dir(tmp_path, monkeypatch)
    captured = (
        tmp_path
        / "raw"
        / "fpl_api"
        / "season=2024-25"
        / "captured_at=2024-08-10T00-00-00Z"
    )
    captured.mkdir(parents=True)
    for name, payload in (("bootstrap_static", _BOOTSTRAP), ("fixtures", _FIXTURES)):
        with gzip.open(captured / f"{name}.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
    result = dg.materialize([raw_snapshots, staged_ingest, *staged_assets])
    assert result.success
    players = result.output_for_node("staged_players")
    assert players["player_key"].to_list() == [100]
    assert schema_check_result("players", players).passed is True
    graph = defs.get_repository_def().asset_graph
    names = {key.path[-1] for key in graph.get_all_asset_keys()}
    assert "staged_players" in names
    get_settings.cache_clear()


def test_sensor_requests_a_run_for_new_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_data_dir(tmp_path, monkeypatch)
    target = tmp_path / "raw" / "fpl_api" / "bootstrap_static.json.gz"

    def _fetch(
        dest: Path, remote: str = "origin", branch: str = "snapshots"
    ) -> list[Path]:
        return [target]

    monkeypatch.setattr("fplt.orchestration.sensors.fetch_snapshots", _fetch)
    evaluation = snapshots_sensor.evaluate_tick(dg.build_sensor_context())
    assert evaluation.run_requests[0].run_key == target.as_posix()
    get_settings.cache_clear()


def test_sensor_skips_when_only_football_data_arrives(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_data_dir(tmp_path, monkeypatch)
    odds = (
        tmp_path
        / "raw"
        / "football_data"
        / "season=2024-25"
        / "captured_at=2025-11-01"
        / "E0.csv"
    )

    def _fetch(
        dest: Path, remote: str = "origin", branch: str = "snapshots"
    ) -> list[Path]:
        return [odds]

    monkeypatch.setattr("fplt.orchestration.sensors.fetch_snapshots", _fetch)
    evaluation = snapshots_sensor.evaluate_tick(dg.build_sensor_context())
    assert evaluation.run_requests == []
    assert evaluation.skip_message is not None
    assert "match_odds_sensor" in evaluation.skip_message
    get_settings.cache_clear()


def test_sensor_skips_when_nothing_is_new(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_data_dir(tmp_path, monkeypatch)

    def _fetch(
        dest: Path, remote: str = "origin", branch: str = "snapshots"
    ) -> list[Path]:
        return []

    monkeypatch.setattr("fplt.orchestration.sensors.fetch_snapshots", _fetch)
    evaluation = snapshots_sensor.evaluate_tick(dg.build_sensor_context())
    assert evaluation.skip_message == "no new snapshots"
    get_settings.cache_clear()


def test_sensor_skips_when_the_branch_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_data_dir(tmp_path, monkeypatch)

    def _fetch(
        dest: Path, remote: str = "origin", branch: str = "snapshots"
    ) -> list[Path]:
        raise SnapshotFetchError("missing")

    monkeypatch.setattr("fplt.orchestration.sensors.fetch_snapshots", _fetch)
    evaluation = snapshots_sensor.evaluate_tick(dg.build_sensor_context())
    assert evaluation.run_requests == []
    assert evaluation.skip_message == "missing"
    get_settings.cache_clear()


def test_understat_sensor_requests_a_run_for_new_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_data_dir(tmp_path, monkeypatch)
    target = (
        tmp_path
        / "raw"
        / "understat"
        / "season=2024-25"
        / "match_id=42"
        / "match.json.gz"
    )
    other = tmp_path / "raw" / "fpl_api" / "bootstrap_static.json.gz"

    def _fetch(
        dest: Path, remote: str = "origin", branch: str = "snapshots"
    ) -> list[Path]:
        return [other, target]

    monkeypatch.setattr("fplt.orchestration.sensors.fetch_snapshots", _fetch)
    evaluation = understat_sensor.evaluate_tick(dg.build_sensor_context())
    assert evaluation.run_requests[0].run_key == target.as_posix()
    names = {sensor.name for sensor in defs.sensors}
    assert "understat_sensor" in names
    get_settings.cache_clear()


def test_understat_sensor_skips_when_prefix_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_data_dir(tmp_path, monkeypatch)
    fpl_only = tmp_path / "raw" / "fpl_api" / "bootstrap_static.json.gz"

    def _fetch(
        dest: Path, remote: str = "origin", branch: str = "snapshots"
    ) -> list[Path]:
        return [fpl_only]

    monkeypatch.setattr("fplt.orchestration.sensors.fetch_snapshots", _fetch)
    evaluation = understat_sensor.evaluate_tick(dg.build_sensor_context())
    assert evaluation.run_requests == []
    assert evaluation.skip_message == "no new understat snapshots"
    get_settings.cache_clear()


def test_understat_sensor_uses_cursor_when_fetch_already_copied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_data_dir(tmp_path, monkeypatch)
    target = (
        tmp_path
        / "raw"
        / "understat"
        / "season=2024-25"
        / "match_id=42"
        / "match.json.gz"
    )
    target.parent.mkdir(parents=True)
    target.write_bytes(b"")

    def _fetch(
        dest: Path, remote: str = "origin", branch: str = "snapshots"
    ) -> list[Path]:
        return []

    monkeypatch.setattr("fplt.orchestration.sensors.fetch_snapshots", _fetch)
    first = understat_sensor.evaluate_tick(dg.build_sensor_context())
    assert first.run_requests[0].run_key == target.as_posix()
    again = understat_sensor.evaluate_tick(
        dg.build_sensor_context(cursor=target.as_posix())
    )
    assert again.run_requests == []
    assert again.skip_message == "no new understat snapshots"
    get_settings.cache_clear()


def test_api_football_sensor_requests_a_run_for_injury_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_data_dir(tmp_path, monkeypatch)
    injury = (
        tmp_path
        / "raw"
        / "api_football"
        / "season=2024-25"
        / "injuries"
        / "captured_at=2024-08-10T00-00-00Z"
        / "injuries.json.gz"
    )
    lineup = (
        tmp_path
        / "raw"
        / "api_football"
        / "season=2024-25"
        / "lineups"
        / "captured_at=2024-08-10T00-00-00Z"
        / "lineups.json.gz"
    )

    def _fetch(
        dest: Path, remote: str = "origin", branch: str = "snapshots"
    ) -> list[Path]:
        return [lineup, injury]

    monkeypatch.setattr("fplt.orchestration.sensors.fetch_snapshots", _fetch)
    evaluation = api_football_sensor.evaluate_tick(dg.build_sensor_context())
    assert evaluation.run_requests[0].run_key == injury.as_posix()
    names = {sensor.name for sensor in defs.sensors}
    assert "api_football_sensor" in names
    get_settings.cache_clear()


def test_api_football_sensor_skips_lineups_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_data_dir(tmp_path, monkeypatch)
    lineup = (
        tmp_path
        / "raw"
        / "api_football"
        / "season=2024-25"
        / "lineups"
        / "captured_at=2024-08-10T00-00-00Z"
        / "lineups.json.gz"
    )

    def _fetch(
        dest: Path, remote: str = "origin", branch: str = "snapshots"
    ) -> list[Path]:
        return [lineup]

    monkeypatch.setattr("fplt.orchestration.sensors.fetch_snapshots", _fetch)
    evaluation = api_football_sensor.evaluate_tick(dg.build_sensor_context())
    assert evaluation.run_requests == []
    assert evaluation.skip_message == "no new api_football snapshots"
    get_settings.cache_clear()


def test_api_football_sensor_skips_when_the_branch_is_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_data_dir(tmp_path, monkeypatch)

    def _fetch(
        dest: Path, remote: str = "origin", branch: str = "snapshots"
    ) -> list[Path]:
        raise SnapshotFetchError("missing")

    monkeypatch.setattr("fplt.orchestration.sensors.fetch_snapshots", _fetch)
    evaluation = api_football_sensor.evaluate_tick(dg.build_sensor_context())
    assert evaluation.run_requests == []
    assert evaluation.skip_message == "missing"
    get_settings.cache_clear()


def test_match_odds_sensor_requests_for_new_e0(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_data_dir(tmp_path, monkeypatch)
    csv_path = (
        tmp_path
        / "raw"
        / "football_data"
        / "season=2024-25"
        / "captured_at=2025-11-01"
        / "E0.csv"
    )
    csv_path.parent.mkdir(parents=True)
    csv_path.write_text("Div,Date\n", encoding="utf-8")
    first = match_odds_sensor.evaluate_tick(dg.build_sensor_context())
    assert first.run_requests[0].run_key == csv_path.as_posix()
    second = match_odds_sensor.evaluate_tick(
        dg.build_sensor_context(cursor=first.cursor)
    )
    assert second.run_requests == []
    assert second.skip_message == "no new football_data E0.csv"
    names = {sensor.name for sensor in defs.sensors}
    assert "match_odds_sensor" in names
    get_settings.cache_clear()


def test_understat_ingest_stages_raw_matches(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_data_dir(tmp_path, monkeypatch)
    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "understat"
    fixture = fixtures / "match_ajax.json"
    maps = fixtures / "id_maps.yaml"
    raw_match = (
        tmp_path
        / "raw"
        / "understat"
        / "season=2024-25"
        / "match_id=42"
        / "match.json.gz"
    )
    raw_match.parent.mkdir(parents=True)
    with gzip.open(raw_match, "wt", encoding="utf-8") as handle:
        json.dump(json.loads(fixture.read_text(encoding="utf-8")), handle)
    monkeypatch.setattr("fplt.orchestration.assets.default_id_maps_path", lambda: maps)
    result = dg.materialize([understat_ingest])
    assert result.success
    assert result.output_for_node("understat_ingest") == 1
    staged = result.output_for_node("understat_ingest")
    assert staged == 1
    frame = DataStore(tmp_path).read("understat_player_matches")
    assert schema_check_result("understat_player_matches", frame).passed is True
    assert frame.height == 3
    graph = defs.get_repository_def().asset_graph
    names = {key.path[-1] for key in graph.get_all_asset_keys()}
    assert "understat_ingest" in names
    get_settings.cache_clear()


def test_match_odds_ingest_stages_raw_csv(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_data_dir(tmp_path, monkeypatch)
    fixtures = Path(__file__).resolve().parents[1] / "fixtures" / "football_data"
    maps = fixtures / "team_names.yaml"
    raw_csv = tmp_path / "raw" / "football_data" / "season=2024-25" / "E0.csv"
    raw_csv.parent.mkdir(parents=True)
    raw_csv.write_text(
        (fixtures / "E0.csv").read_text(encoding="utf-8"), encoding="utf-8"
    )
    monkeypatch.setattr("fplt.orchestration.assets.default_team_map_path", lambda: maps)
    result = dg.materialize([match_odds_ingest])
    assert result.success
    assert result.output_for_node("match_odds_ingest") == 1
    frame = DataStore(tmp_path).read("match_odds")
    assert schema_check_result("match_odds", frame).passed is True
    assert frame.height == 6
    assert set(frame["odds_kind"].unique()) == {
        "closing_proxy",
        "prematch_scheduled",
    }
    graph = defs.get_repository_def().asset_graph
    names = {key.path[-1] for key in graph.get_all_asset_keys()}
    assert "match_odds_ingest" in names
    assert "staged_match_odds" in names
    get_settings.cache_clear()
