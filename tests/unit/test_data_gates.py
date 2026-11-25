"""ops-06 / A10 capture sanity gates and quarantine schema."""

from __future__ import annotations

import copy
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from fplt.data.gates import (
    GATE_DATA_CHECKED,
    GATE_DEADLINES_MONOTONE,
    GATE_NEXT_GW_FIXTURES,
    GATE_NULL_SHARE,
    GATE_PLAYER_COUNT,
    DataGatesConfig,
    load_data_gates_config,
    quarantine_records,
    run_bootstrap_gates,
)
from fplt.schemas.l2 import SCHEMA_VERSION
from fplt.schemas.quarantine import QuarantineRecordSchema

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "gates"
_TS = datetime(2024, 8, 20, 12, 0, 0, tzinfo=UTC)


def _load_recorded() -> tuple[dict[str, Any], list[dict[str, Any]]]:
    bootstrap = json.loads((_FIXTURES / "bootstrap_static.json").read_text())
    fixtures = json.loads((_FIXTURES / "fixtures.json").read_text())
    return bootstrap, fixtures


def _mini_config(**overrides: Any) -> DataGatesConfig:
    base: dict[str, Any] = {
        "player_count_max_delta": 2,
        "max_null_share": {"now_cost": 0.0, "status": 0.0, "team": 0.0},
        "expected_team_count": 2,
    }
    base.update(overrides)
    return DataGatesConfig.model_validate(base)


def _mini_bootstrap(
    *,
    n_players: int = 2,
    null_field: str | None = None,
    deadlines: list[str] | None = None,
    data_checked: list[bool] | None = None,
) -> dict[str, Any]:
    teams = [
        {"id": 1, "code": 3, "name": "A", "short_name": "AAA"},
        {"id": 2, "code": 7, "name": "B", "short_name": "BBB"},
    ]
    elements: list[dict[str, Any]] = []
    for i in range(n_players):
        row: dict[str, Any] = {
            "id": i + 1,
            "code": 100 + i,
            "team": (i % 2) + 1,
            "element_type": 3,
            "web_name": f"P{i}",
            "status": "a",
            "now_cost": 50,
        }
        if null_field is not None and i == 0:
            row[null_field] = None
        elements.append(row)
    stamps = deadlines or [
        "2024-08-16T17:30:00Z",
        "2024-08-23T17:30:00Z",
    ]
    checked = data_checked or [True, False]
    events = [
        {
            "id": 1,
            "deadline_time": stamps[0],
            "finished": True,
            "data_checked": checked[0],
            "is_next": False,
        },
        {
            "id": 2,
            "deadline_time": stamps[1],
            "finished": False,
            "data_checked": checked[1],
            "is_next": True,
        },
    ]
    return {"teams": teams, "elements": elements, "events": events}


def _mini_fixtures() -> list[dict[str, Any]]:
    return [
        {
            "id": 1,
            "event": 1,
            "team_h": 1,
            "team_a": 2,
            "kickoff_time": "2024-08-17T14:00:00Z",
            "finished": True,
        },
        {
            "id": 2,
            "event": 2,
            "team_h": 1,
            "team_a": 2,
            "kickoff_time": "2024-08-24T14:00:00Z",
            "finished": False,
        },
    ]


def test_load_committed_data_gates_config() -> None:
    cfg = load_data_gates_config()
    assert cfg.expected_team_count == 20
    assert cfg.player_count_max_delta == 50
    assert cfg.max_null_share["now_cost"] == 0.0


def test_quarantine_schema_validates() -> None:
    frame = pl.DataFrame(
        {
            "season": ["2024-25"],
            "source": ["fpl_api"],
            "capture_path": ["fpl_api/season=2024-25/captured_at=2024-08-20T12-00-00Z"],
            "captured_at": [_TS],
            "gate": [GATE_PLAYER_COUNT],
            "detail": ["player count blew up"],
            "previous_capture_path": [
                "fpl_api/season=2024-25/captured_at=2024-08-19T12-00-00Z"
            ],
            "available_at": [_TS],
            "schema_version": [SCHEMA_VERSION],
        }
    )
    checked = QuarantineRecordSchema.validate(frame)
    assert checked["gate"].to_list() == [GATE_PLAYER_COUNT]


def test_recorded_capture_passes_all_gates() -> None:
    bootstrap, fixtures = _load_recorded()
    previous = copy.deepcopy(bootstrap)
    report = run_bootstrap_gates(
        bootstrap, fixtures, previous=previous, config=load_data_gates_config()
    )
    assert report.passed
    assert {result.name for result in report.results} == {
        GATE_PLAYER_COUNT,
        GATE_NULL_SHARE,
        GATE_NEXT_GW_FIXTURES,
        GATE_DEADLINES_MONOTONE,
        GATE_DATA_CHECKED,
    }
    assert quarantine_records(
        report,
        season="2024-25",
        capture_path="fpl_api/season=2024-25/captured_at=2024-08-20T12-00-00Z",
        captured_at=_TS,
        available_at=_TS,
    ).is_empty()


def test_player_count_pass_and_fail() -> None:
    good = _mini_bootstrap(n_players=2)
    previous = _mini_bootstrap(n_players=2)
    cfg = _mini_config(player_count_max_delta=1)
    fixtures = _mini_fixtures()
    assert run_bootstrap_gates(good, fixtures, previous=previous, config=cfg).passed
    bad = _mini_bootstrap(n_players=4)
    report = run_bootstrap_gates(bad, fixtures, previous=previous, config=cfg)
    assert not report.passed
    failure = next(r for r in report.failures if r.name == GATE_PLAYER_COUNT)
    assert "4" in failure.detail


def test_null_share_pass_and_fail() -> None:
    fixtures = _mini_fixtures()
    cfg = _mini_config()
    good = _mini_bootstrap()
    assert run_bootstrap_gates(good, fixtures, config=cfg).passed
    bad = _mini_bootstrap(null_field="now_cost")
    report = run_bootstrap_gates(bad, fixtures, config=cfg)
    assert not report.passed
    failure = next(r for r in report.failures if r.name == GATE_NULL_SHARE)
    assert "now_cost" in failure.detail


def test_next_gw_fixtures_pass_and_fail() -> None:
    cfg = _mini_config()
    bootstrap = _mini_bootstrap()
    good = _mini_fixtures()
    assert run_bootstrap_gates(bootstrap, good, config=cfg).passed
    report = run_bootstrap_gates(bootstrap, [], config=cfg)
    assert not report.passed
    failure = next(r for r in report.failures if r.name == GATE_NEXT_GW_FIXTURES)
    assert "no fixtures" in failure.detail


def test_deadlines_monotone_pass_and_fail() -> None:
    cfg = _mini_config()
    fixtures = _mini_fixtures()
    good_report = run_bootstrap_gates(_mini_bootstrap(), fixtures, config=cfg)
    good_gate = next(
        r for r in good_report.results if r.name == GATE_DEADLINES_MONOTONE
    )
    assert good_gate.passed
    bad = _mini_bootstrap(deadlines=["2024-08-23T17:30:00Z", "2024-08-16T17:30:00Z"])
    report = run_bootstrap_gates(bad, fixtures, config=cfg)
    failure = next(r for r in report.failures if r.name == GATE_DEADLINES_MONOTONE)
    assert not failure.passed


def test_data_checked_never_regresses() -> None:
    cfg = _mini_config()
    fixtures = _mini_fixtures()
    previous = _mini_bootstrap(data_checked=[True, False])
    good = _mini_bootstrap(data_checked=[True, False])
    assert run_bootstrap_gates(good, fixtures, previous=previous, config=cfg).passed
    bad = _mini_bootstrap(data_checked=[False, False])
    report = run_bootstrap_gates(bad, fixtures, previous=previous, config=cfg)
    failure = next(r for r in report.failures if r.name == GATE_DATA_CHECKED)
    assert "true→false" in failure.detail


def test_quarantine_records_from_failures() -> None:
    cfg = _mini_config()
    bootstrap = _mini_bootstrap(null_field="status")
    report = run_bootstrap_gates(bootstrap, _mini_fixtures(), config=cfg)
    frame = quarantine_records(
        report,
        season="2024-25",
        capture_path="fpl_api/season=2024-25/captured_at=2024-08-20T12-00-00Z",
        captured_at=_TS,
        available_at=_TS,
        previous_capture_path="fpl_api/season=2024-25/captured_at=2024-08-19T12-00-00Z",
    )
    assert not frame.is_empty()
    QuarantineRecordSchema.validate(frame)
    assert GATE_NULL_SHARE in frame["gate"].to_list()


def test_first_capture_skips_comparative_gates() -> None:
    cfg = _mini_config()
    report = run_bootstrap_gates(_mini_bootstrap(), _mini_fixtures(), config=cfg)
    by_name = {result.name: result for result in report.results}
    assert by_name[GATE_PLAYER_COUNT].passed
    assert "no previous" in by_name[GATE_PLAYER_COUNT].detail
    assert by_name[GATE_DATA_CHECKED].passed
    assert "no previous" in by_name[GATE_DATA_CHECKED].detail


def test_config_rejects_bad_null_share() -> None:
    with pytest.raises(ValueError, match="max_null_share"):
        DataGatesConfig.model_validate(
            {
                "player_count_max_delta": 1,
                "max_null_share": {"now_cost": 0.0},
                "expected_team_count": 2,
            }
        )
