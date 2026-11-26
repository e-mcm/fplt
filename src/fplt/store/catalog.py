"""Derived ``catalog/runs.parquet`` rebuildable from run manifests.

Catalog files are the only rewritten store objects (data-model §3.5). Only
runs with ``_SUCCESS`` are indexed; incomplete folders stay invisible.
Readers union the compacted file with optional ``catalog/inbox/`` rows.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import polars as pl

from fplt.schemas.catalog import CATALOG_RUN_COLUMNS, CatalogRunSchema
from fplt.schemas.manifests import RunManifest
from fplt.store.runs import run_dir

logger = logging.getLogger(__name__)

_SUCCESS = "_SUCCESS"
_MANIFEST = "manifest.json"
_RUNS_PARQUET = "runs.parquet"


def catalog_dir(data_dir: Path) -> Path:
    """Return ``data_dir/catalog/``."""
    return data_dir / "catalog"


def catalog_runs_path(data_dir: Path) -> Path:
    """Return ``data_dir/catalog/runs.parquet``."""
    return catalog_dir(data_dir) / _RUNS_PARQUET


def catalog_inbox_dir(data_dir: Path) -> Path:
    """Return ``data_dir/catalog/inbox/``."""
    return catalog_dir(data_dir) / "inbox"


def empty_runs_catalog() -> pl.DataFrame:
    """Empty ``CatalogRunSchema`` frame with typed columns."""
    frame = pl.DataFrame(schema=CATALOG_RUN_COLUMNS)
    checked = CatalogRunSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked


def manifest_to_catalog_row(manifest: RunManifest) -> dict[str, object]:
    """Flatten a succeeded ``RunManifest`` into one catalog row dict.

    Args:
        manifest: Parsed run manifest (must be ``status=succeeded`` with
            ``finished_at`` set).

    Returns:
        Column names matching ``CatalogRunSchema``.

    Raises:
        ValueError: Manifest is not a succeeded finished run.
    """
    if manifest.status != "succeeded" or manifest.finished_at is None:
        raise ValueError(
            f"catalog row requires succeeded finished run, got "
            f"status={manifest.status!r} finished_at={manifest.finished_at!r}"
        )
    primary = manifest.as_of[0]
    return {
        "run_id": manifest.run_id,
        "kind": manifest.kind,
        "mode": manifest.mode,
        "purpose": manifest.purpose,
        "experiment_id": manifest.experiment_id,
        "arm": manifest.arm,
        "run_type": manifest.run_type,
        "spec_hash": manifest.spec_hash,
        "spec_label": manifest.spec_label,
        "model_id": manifest.model_id,
        "line": manifest.line,
        "plan_horizon": manifest.plan_horizon,
        "squad_mode": manifest.squad_mode,
        "prev_run_id": manifest.prev_run_id,
        "code_sha": manifest.code_sha,
        "code_dirty": manifest.code_dirty,
        "prod_tag": manifest.prod_tag,
        "fplt_mc_version": manifest.fplt_mc_version,
        "lockfile_hash": manifest.lockfile_hash,
        "build_id": manifest.build_id,
        "as_of_season": primary.season,
        "as_of_gw": primary.gw,
        "deadline_utc": primary.deadline_utc,
        "n_as_of": len(manifest.as_of),
        "horizon": manifest.horizon,
        "checkpoint": manifest.checkpoint,
        "host": manifest.host,
        "started_at": manifest.started_at,
        "finished_at": manifest.finished_at,
        "status": manifest.status,
        "parent_run_id": manifest.parent_run_id,
    }


def write_run_inbox_row(data_dir: Path, manifest: RunManifest) -> Path:
    """Write ``catalog/inbox/run=<run_id>.parquet`` for a succeeded run.

    Args:
        data_dir: Local data root.
        manifest: Succeeded run manifest.

    Returns:
        Path of the one-row inbox parquet.
    """
    row = manifest_to_catalog_row(manifest)
    frame = CatalogRunSchema.validate(pl.DataFrame([row], schema=CATALOG_RUN_COLUMNS))
    assert isinstance(frame, pl.DataFrame)
    inbox = catalog_inbox_dir(data_dir)
    inbox.mkdir(parents=True, exist_ok=True)
    path = inbox / f"run={manifest.run_id}.parquet"
    frame.write_parquet(path)
    return path


def list_succeeded_run_ids(data_dir: Path) -> list[str]:
    """Sorted run ids whose folder contains ``_SUCCESS``.

    Args:
        data_dir: Local data root containing ``runs/``.

    Returns:
        Run ids ascending (sortable ``run_id`` order).
    """
    root = data_dir / "runs"
    if not root.is_dir():
        return []
    ids: list[str] = []
    for path in sorted(root.glob("run=*")):
        if path.is_dir() and (path / _SUCCESS).is_file():
            ids.append(path.name.removeprefix("run="))
    return ids


def _load_succeeded_manifest(data_dir: Path, run_id: str) -> RunManifest | None:
    path = run_dir(data_dir, run_id)
    if not (path / _SUCCESS).is_file():
        return None
    manifest_path = path / _MANIFEST
    if not manifest_path.is_file():
        logger.warning("run %s has _SUCCESS but no manifest; skipping", run_id)
        return None
    return RunManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))


def scan_succeeded_runs(data_dir: Path) -> pl.DataFrame:
    """Build a catalog frame by scanning local ``runs/run=*/`` folders.

    Args:
        data_dir: Local data root.

    Returns:
        ``CatalogRunSchema`` rows for every ``_SUCCESS`` run with a readable
        succeeded manifest. Missing ``runs/`` yields an empty frame.
    """
    rows: list[dict[str, object]] = []
    for run_id in list_succeeded_run_ids(data_dir):
        manifest = _load_succeeded_manifest(data_dir, run_id)
        if manifest is None:
            continue
        if manifest.status != "succeeded" or manifest.finished_at is None:
            logger.warning(
                "run %s has _SUCCESS but status=%s; skipping catalog row",
                run_id,
                manifest.status,
            )
            continue
        rows.append(manifest_to_catalog_row(manifest))
    if not rows:
        return empty_runs_catalog()
    frame = pl.DataFrame(rows, schema=CATALOG_RUN_COLUMNS)
    checked = CatalogRunSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked.sort("run_id")


def _read_inbox(data_dir: Path) -> pl.DataFrame:
    inbox = catalog_inbox_dir(data_dir)
    if not inbox.is_dir():
        return empty_runs_catalog()
    frames: list[pl.DataFrame] = []
    for path in sorted(inbox.glob("run=*.parquet")):
        if path.is_file():
            frames.append(pl.read_parquet(path))
    if not frames:
        return empty_runs_catalog()
    return pl.concat(frames, how="diagonal_relaxed")


def read_runs_catalog(data_dir: Path) -> pl.DataFrame:
    """Read compacted catalog union inbox, deduped by ``run_id`` (inbox wins).

    Args:
        data_dir: Local data root.

    Returns:
        ``CatalogRunSchema`` frame (possibly empty when neither source exists).
    """
    compacted_path = catalog_runs_path(data_dir)
    if compacted_path.is_file():
        compacted = pl.read_parquet(compacted_path)
    else:
        compacted = empty_runs_catalog()
    inbox = _read_inbox(data_dir)
    if compacted.is_empty() and inbox.is_empty():
        return empty_runs_catalog()
    if compacted.is_empty():
        merged = inbox
    elif inbox.is_empty():
        merged = compacted
    else:
        # Inbox rows replace compacted ones with the same run_id.
        merged = pl.concat([compacted, inbox], how="diagonal_relaxed").unique(
            subset=["run_id"],
            keep="last",
        )
    checked = CatalogRunSchema.validate(
        merged.select(list(CATALOG_RUN_COLUMNS)).sort("run_id")
    )
    assert isinstance(checked, pl.DataFrame)
    return checked


def rebuild_runs_catalog(data_dir: Path) -> pl.DataFrame:
    """Rescan succeeded runs, write ``catalog/runs.parquet``, clear inbox.

    Args:
        data_dir: Local data root.

    Returns:
        The compacted ``CatalogRunSchema`` frame that was written.
    """
    frame = scan_succeeded_runs(data_dir)
    out_dir = catalog_dir(data_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    path = catalog_runs_path(data_dir)
    frame.write_parquet(path)
    inbox = catalog_inbox_dir(data_dir)
    if inbox.is_dir():
        shutil.rmtree(inbox)
    logger.info(
        "catalog rebuild: wrote %s rows=%d",
        path,
        frame.height,
    )
    return frame
