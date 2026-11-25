"""ops-06 staging skip: quarantined captures never reach staged/ or PIT."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from fplt.data.gates import GATE_NULL_SHARE
from fplt.data.pit import PITData
from fplt.data.quarantine import (
    append_quarantine,
    capture_relpath,
    load_quarantine,
    quarantined_capture_paths,
)
from fplt.data.raw import ingest_raw
from fplt.data.store import DataStore
from fplt.schemas.l2 import SCHEMA_VERSION
from fplt.schemas.quarantine import QuarantineRecordSchema
from fplt.types import GWKey

_GOOD_AT = datetime(2024, 8, 10, 0, 0, 0, tzinfo=UTC)
_BAD_AT = datetime(2024, 8, 18, 0, 0, 0, tzinfo=UTC)
_BOOTSTRAP: dict[str, Any] = {
    "teams": [
        {"id": 1, "code": 3, "name": "Arsenal", "short_name": "ARS"},
        {"id": 2, "code": 7, "name": "Spurs", "short_name": "TOT"},
    ],
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
        "team_a": 2,
        "kickoff_time": "2024-08-17T14:00:00Z",
        "finished": False,
        "team_h_score": None,
        "team_a_score": None,
    }
]


def _write_capture(raw_root: Path, stamp: datetime, bootstrap: dict[str, Any]) -> Path:
    label = stamp.strftime("%Y-%m-%dT%H-%M-%SZ")
    captured = raw_root / "fpl_api" / "season=2024-25" / f"captured_at={label}"
    captured.mkdir(parents=True)
    for name, payload in (("bootstrap_static", bootstrap), ("fixtures", _FIXTURES)):
        with gzip.open(captured / f"{name}.json.gz", "wt", encoding="utf-8") as handle:
            json.dump(payload, handle)
    return captured


def _quarantine_row(capture_path: str) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": ["2024-25"],
            "source": ["fpl_api"],
            "capture_path": [capture_path],
            "captured_at": [_BAD_AT],
            "gate": [GATE_NULL_SHARE],
            "detail": ["synthetic quarantine for staging-skip test"],
            "previous_capture_path": [None],
            "available_at": [_BAD_AT],
            "schema_version": [SCHEMA_VERSION],
        }
    )


def test_append_and_load_quarantine(tmp_path: Path) -> None:
    empty = load_quarantine(tmp_path)
    assert empty.is_empty()
    assert quarantined_capture_paths(tmp_path) == frozenset()
    path = "fpl_api/season=2024-25/captured_at=2024-08-18T00-00-00Z"
    merged = append_quarantine(tmp_path, _quarantine_row(path))
    QuarantineRecordSchema.validate(merged)
    assert GATE_NULL_SHARE in merged["gate"].to_list()
    assert path in quarantined_capture_paths(tmp_path)


def test_ingest_skips_quarantined_capture_pit_excludes(tmp_path: Path) -> None:
    raw_root = tmp_path / "raw"
    data_dir = tmp_path / "data"
    _write_capture(raw_root, _GOOD_AT, _BOOTSTRAP)
    bad_bootstrap = {
        **_BOOTSTRAP,
        "elements": [{**_BOOTSTRAP["elements"][0], "now_cost": 999}],
    }
    bad_dir = _write_capture(raw_root, _BAD_AT, bad_bootstrap)
    bad_rel = capture_relpath(raw_root, bad_dir)
    append_quarantine(data_dir, _quarantine_row(bad_rel))

    store = DataStore(data_dir)
    count = ingest_raw(store, raw_root)
    assert count == 1
    status = store.read("player_status")
    assert status["price"].to_list() == [100]
    assert _BAD_AT not in status["available_at"].to_list()

    view = PITData(store).view(GWKey("2024-25", 1))
    pit_status = view.latest("player_status", by=["player_key"]).collect()
    assert pit_status["price"].to_list() == [100]
    assert pit_status["available_at"].to_list() == [_GOOD_AT]
