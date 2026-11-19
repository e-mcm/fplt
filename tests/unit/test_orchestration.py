"""Staged assets ingest a snapshot, and the sensor requests a run for new files."""

import gzip
import json
from pathlib import Path

import dagster as dg
import polars as pl
import pytest

from fplt.data.snapshots import SnapshotFetchError
from fplt.orchestration import defs
from fplt.orchestration.assets import raw_snapshots, staged_assets, staged_ingest
from fplt.orchestration.checks import schema_check_result
from fplt.orchestration.sensors import snapshots_sensor
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
