"""Stage C catalog: rebuild idempotency, _SUCCESS visibility, missing sources."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from fplt.schemas.catalog import CatalogRunSchema
from fplt.schemas.manifests import AsOfDeadline
from fplt.store.catalog import (
    catalog_inbox_dir,
    catalog_runs_path,
    list_succeeded_run_ids,
    read_runs_catalog,
    rebuild_runs_catalog,
    write_run_inbox_row,
)
from fplt.store.runs import RunWriter, run_dir

_TS = datetime(2026, 11, 24, 12, 56, 1, tzinfo=UTC)


def _deadline() -> AsOfDeadline:
    return AsOfDeadline(season="2026-27", gw=12, deadline_utc=_TS)


def _writer(tmp_path: Path, run_id: str, **overrides: object) -> RunWriter:
    kwargs: dict[str, object] = {
        "kind": "adhoc",
        "mode": "live",
        "run_type": "predict",
        "spec_hash": "a" * 64,
        "spec_label": "demo",
        "model_id": "deadbeef",
        "as_of": [_deadline()],
        "horizon": 3,
        "code_sha": "abc1234",
        "code_dirty": False,
        "lockfile_hash": "b" * 64,
        "host": "test",
        "raw_manifest_hash": "c" * 64,
        "staged_digests": {},
        "started_at": _TS,
        "run_id": run_id,
    }
    kwargs.update(overrides)
    return RunWriter(tmp_path, **kwargs)  # type: ignore[arg-type]


def test_rebuild_empty_when_runs_missing(tmp_path: Path) -> None:
    frame = rebuild_runs_catalog(tmp_path)
    assert frame.is_empty()
    assert catalog_runs_path(tmp_path).is_file()
    assert CatalogRunSchema.validate(frame) is not None


def test_read_catalog_missing_sources_is_empty(tmp_path: Path) -> None:
    assert read_runs_catalog(tmp_path).is_empty()
    assert list_succeeded_run_ids(tmp_path) == []


def test_incomplete_run_without_success_invisible(tmp_path: Path) -> None:
    run_id = "r_20261124T125601Z_aaaaaa"
    writer = _writer(tmp_path, run_id)
    writer.__enter__()
    # Still running: manifest present, no _SUCCESS.
    assert (run_dir(tmp_path, run_id) / "manifest.json").is_file()
    assert not (run_dir(tmp_path, run_id) / "_SUCCESS").is_file()
    assert list_succeeded_run_ids(tmp_path) == []
    assert rebuild_runs_catalog(tmp_path).is_empty()
    writer.fail()
    assert not (run_dir(tmp_path, run_id) / "_SUCCESS").is_file()
    assert rebuild_runs_catalog(tmp_path).is_empty()


def test_succeeded_run_indexed_and_rebuild_idempotent(tmp_path: Path) -> None:
    run_id = "r_20261124T125601Z_bbbbbb"
    with _writer(tmp_path, run_id, spec_label="dc_layered_v5"):
        pass
    assert (run_dir(tmp_path, run_id) / "_SUCCESS").is_file()
    inbox = catalog_inbox_dir(tmp_path) / f"run={run_id}.parquet"
    assert inbox.is_file()

    first = rebuild_runs_catalog(tmp_path)
    assert first.height == 1
    assert first["run_id"].to_list() == [run_id]
    assert first["spec_label"].to_list() == ["dc_layered_v5"]
    assert not catalog_inbox_dir(tmp_path).exists()

    second = rebuild_runs_catalog(tmp_path)
    assert first.equals(second)
    assert read_runs_catalog(tmp_path).equals(first)


def test_inbox_union_before_rebuild(tmp_path: Path) -> None:
    run_a = "r_20261124T125601Z_cccccc"
    run_b = "r_20261124T125601Z_dddddd"
    with _writer(tmp_path, run_a):
        pass
    with _writer(tmp_path, run_b, spec_label="other"):
        pass
    # Compact only A, leave B in inbox.
    from fplt.store.catalog import scan_succeeded_runs

    frame_a = scan_succeeded_runs(tmp_path).filter(pl.col("run_id") == run_a)
    catalog_runs_path(tmp_path).parent.mkdir(parents=True, exist_ok=True)
    frame_a.write_parquet(catalog_runs_path(tmp_path))
    # Reset inbox to only B.
    inbox_dir = catalog_inbox_dir(tmp_path)
    if inbox_dir.exists():
        for path in inbox_dir.glob("*.parquet"):
            path.unlink()
    from fplt.store.push import load_run_manifest

    write_run_inbox_row(tmp_path, load_run_manifest(tmp_path, run_b))
    merged = read_runs_catalog(tmp_path)
    assert sorted(merged["run_id"].to_list()) == [run_a, run_b]
