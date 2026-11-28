"""Stage B6 PR1: content-addressed curated tables and build manifests."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fplt.data.staging import stage_bootstrap
from fplt.data.store import DataStore
from fplt.schemas.manifests import BuildManifest
from fplt.store.builds import (
    STAGING_VERSION,
    BuildExistsError,
    build_from_staged,
    build_manifest_path,
    compute_build_id,
    load_build_manifest,
    raw_cutoff,
    schema_hash,
    schema_hashes,
    table_sha_path,
    write_build_manifest,
    write_table,
)
from fplt.store.runs import hash_raw_manifest

_TS = datetime(2026, 11, 24, 12, 0, 0, tzinfo=UTC)

_BOOTSTRAP = {
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


def _stage_mini(data_dir: Path) -> None:
    store = DataStore(data_dir)
    frames = stage_bootstrap(_BOOTSTRAP, _FIXTURES, _TS, "2024-25")
    for name, frame in frames.items():
        if not frame.is_empty():
            store.write(name, frame)


def _write_raw(data_dir: Path, *, stamp: str = "2026-11-24T12-00-00Z") -> None:
    dest = (
        data_dir
        / "raw"
        / "fpl_api"
        / "season=2024-25"
        / f"captured_at={stamp}"
        / "bootstrap_static.json.gz"
    )
    dest.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(dest, "wt", encoding="utf-8") as handle:
        json.dump(_BOOTSTRAP, handle)


def test_schema_hash_stable_and_known_tables() -> None:
    hashes = schema_hashes()
    assert "players" in hashes
    assert schema_hash("players") == hashes["players"]
    assert len(hashes["players"]) == 64


def test_compute_build_id_changes_with_staging_version() -> None:
    tables = {"players": "b" * 64}
    first = compute_build_id(table_shas=tables, staging_version=1)
    second = compute_build_id(table_shas=tables, staging_version=2)
    assert first != second
    assert compute_build_id(table_shas=tables, staging_version=1) == first
    assert (
        compute_build_id(table_shas={"players": "c" * 64}, staging_version=1) != first
    )


def test_write_table_dedups_identical_bytes(tmp_path: Path) -> None:
    _stage_mini(tmp_path)
    frame = DataStore(tmp_path).read("players")
    first = write_table(tmp_path, "players", frame)
    second = write_table(tmp_path, "players", frame)
    assert first == second
    path = table_sha_path(tmp_path, "players", first.sha)
    assert path.is_file()
    assert path.stat().st_size > 0


def test_write_build_manifest_is_write_once(tmp_path: Path) -> None:
    entry_tables = {
        "players": write_table(
            tmp_path,
            "players",
            pl.DataFrame(
                {
                    "season": ["2024-25"],
                    "player_key": [1],
                    "element_id": [1],
                    "team_key": [3],
                    "position": ["MID"],
                    "web_name": ["A"],
                    "available_at": [_TS],
                    "source": ["test"],
                }
            ),
        )
    }
    manifest = BuildManifest(
        build_id="cafe" * 8,
        created_at=_TS,
        staging_version=STAGING_VERSION,
        code_sha="abc1234",
        raw_manifest_hash="c" * 64,
        raw_cutoff=_TS,
        tables=entry_tables,
    )
    path = write_build_manifest(tmp_path, manifest)
    assert path == build_manifest_path(tmp_path, manifest.build_id)
    with pytest.raises(BuildExistsError):
        write_build_manifest(tmp_path, manifest)


def test_build_from_staged_writes_manifest_and_tables(tmp_path: Path) -> None:
    _write_raw(tmp_path)
    _stage_mini(tmp_path)
    manifest = build_from_staged(
        tmp_path,
        code_sha="deadbeef",
        created_at=_TS,
        quarantined_captures=["raw/bad.json.gz"],
    )
    assert manifest.staging_version == STAGING_VERSION
    assert manifest.code_sha == "deadbeef"
    assert manifest.raw_manifest_hash == hash_raw_manifest(tmp_path)
    assert manifest.raw_cutoff == datetime(2026, 11, 24, 12, 0, 0, tzinfo=UTC)
    assert "players" in manifest.tables
    assert "fixtures" in manifest.tables
    loaded = load_build_manifest(tmp_path, manifest.build_id)
    assert loaded.build_id == manifest.build_id
    for name, entry in manifest.tables.items():
        assert table_sha_path(tmp_path, name, entry.sha).is_file()
    # Idempotent: same raw + staging → same build_id, returns existing.
    again = build_from_staged(tmp_path, code_sha="other", created_at=_TS)
    assert again.build_id == manifest.build_id
    assert again.code_sha == "deadbeef"


def test_raw_cutoff_reads_captured_at_stamps(tmp_path: Path) -> None:
    assert raw_cutoff(tmp_path) == datetime(1970, 1, 1, tzinfo=UTC)
    _write_raw(tmp_path, stamp="2026-01-01T00-00-00Z")
    _write_raw(tmp_path, stamp="2026-06-15T18-30-00Z")
    assert raw_cutoff(tmp_path) == datetime(2026, 6, 15, 18, 30, 0, tzinfo=UTC)


def test_write_table_rejects_unknown_name(tmp_path: Path) -> None:
    with pytest.raises(KeyError, match="unknown staged table"):
        write_table(tmp_path, "not_a_table", pl.DataFrame({"x": [1]}))


def test_write_table_rejects_sha_collision(tmp_path: Path) -> None:
    _stage_mini(tmp_path)
    frame = DataStore(tmp_path).read("players")
    entry = write_table(tmp_path, "players", frame)
    dest = table_sha_path(tmp_path, "players", entry.sha)
    dest.write_bytes(b"corrupted-not-parquet")
    with pytest.raises(FileExistsError, match="bytes differ"):
        write_table(tmp_path, "players", frame)


def test_load_build_manifest_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_build_manifest(tmp_path, "missing")


def test_build_from_staged_requires_tables(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no staged tables"):
        build_from_staged(tmp_path, code_sha="abc", created_at=_TS)


def test_build_from_staged_naive_created_at(tmp_path: Path) -> None:
    _write_raw(tmp_path)
    _stage_mini(tmp_path)
    naive = datetime(2026, 11, 24, 12, 0, 0)  # noqa: DTZ001 — intentional naive
    manifest = build_from_staged(tmp_path, code_sha="abc", created_at=naive)
    assert manifest.created_at.tzinfo is not None
    assert manifest.created_at == _TS
