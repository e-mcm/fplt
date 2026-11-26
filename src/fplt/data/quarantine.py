"""Persist ops-06 quarantine rows and expose the staging skip list.

Rows live at ``$FPLT_DATA_DIR/quarantine/part.parquet`` (not under
``staged/``). ``ingest_raw`` skips any capture path listed here until curated
builds carry ``quarantined_captures`` in the build manifest. Capture jobs
mirror the table to R2 ``status/quarantine/`` (rewritable ``status/`` prefix).
"""

from __future__ import annotations

import logging
from pathlib import Path

import polars as pl

from fplt.data.remote import Remote, RemoteError, copy_mutable
from fplt.schemas.quarantine import QuarantineRecordSchema
from fplt.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_FILE = "part.parquet"
_REMOTE_PREFIX = "status/quarantine"


def quarantine_path(data_dir: Path) -> Path:
    """Return the parquet path for quarantine records under ``data_dir``."""
    return data_dir / "quarantine" / _FILE


def load_quarantine(data_dir: Path) -> pl.DataFrame:
    """Load quarantine rows, or an empty typed frame when none exist."""
    path = quarantine_path(data_dir)
    if not path.is_file():
        return QuarantineRecordSchema.validate(pl.DataFrame(schema=_empty_schema()))
    frame = pl.read_parquet(path)
    checked = QuarantineRecordSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked


def append_quarantine(data_dir: Path, frame: pl.DataFrame) -> pl.DataFrame:
    """Validate and append quarantine rows; collapse on unique key.

    Args:
        data_dir: Local data root.
        frame: Rows conforming to ``QuarantineRecordSchema``.

    Returns:
        The full quarantine table after append.
    """
    if frame.is_empty():
        return load_quarantine(data_dir)
    incoming = QuarantineRecordSchema.validate(frame)
    assert isinstance(incoming, pl.DataFrame)
    path = quarantine_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.is_file():
        existing = load_quarantine(data_dir)
        merged = pl.concat([existing, incoming], how="vertical_relaxed")
    else:
        merged = incoming
    keys = ["capture_path", "gate", "available_at"]
    merged = merged.unique(subset=keys, keep="last").sort(keys)
    merged.write_parquet(path)
    return merged


def quarantined_capture_paths(data_dir: Path) -> frozenset[str]:
    """Return capture paths (relative to ``raw/``) currently quarantined."""
    frame = load_quarantine(data_dir)
    if frame.is_empty():
        return frozenset()
    return frozenset(frame["capture_path"].to_list())


def capture_relpath(raw_root: Path, capture_dir: Path) -> str:
    """Return a stable posix path of ``capture_dir`` relative to ``raw_root``."""
    return capture_dir.resolve().relative_to(raw_root.resolve()).as_posix()


def pull_quarantine(data_dir: Path, settings: Settings | None = None) -> bool:
    """Pull ``status/quarantine`` from the data bucket into ``data_dir/quarantine``.

    Uses a mutable rclone copy because ``status/`` is a rewritten prefix.
    No-ops when no data bucket is configured. Existing local rows are kept and
    merged with the remote table when both exist.

    Args:
        data_dir: Local data root.
        settings: Process settings. Defaults to ``get_settings()``.

    Returns:
        True when a remote pull was attempted.

    Raises:
        RemoteError: rclone failed.
    """
    cfg = get_settings() if settings is None else settings
    remote = Remote.from_settings(cfg)
    if remote is None:
        return False
    dest = data_dir / "quarantine"
    dest.mkdir(parents=True, exist_ok=True)
    local_before = load_quarantine(data_dir)
    try:
        copy_mutable(
            remote,
            f"{remote.target.rstrip('/')}/{_REMOTE_PREFIX}",
            str(dest),
        )
    except RemoteError as exc:
        # Empty remote prefix is fine on first use.
        detail = str(exc).lower()
        if "directory not found" in detail or "not found" in detail:
            logger.info("no remote quarantine yet at %s", _REMOTE_PREFIX)
            return True
        raise
    remote_frame = load_quarantine(data_dir)
    if local_before.is_empty():
        return True
    if remote_frame.is_empty():
        append_quarantine(data_dir, local_before)
        return True
    merged = pl.concat([local_before, remote_frame], how="vertical_relaxed")
    keys = ["capture_path", "gate", "available_at"]
    merged = merged.unique(subset=keys, keep="last").sort(keys)
    path = quarantine_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(path)
    return True


def push_quarantine(data_dir: Path, settings: Settings | None = None) -> bool:
    """Push local ``quarantine/`` to R2 ``status/quarantine`` (mutable).

    Args:
        data_dir: Local data root.
        settings: Process settings. Defaults to ``get_settings()``.

    Returns:
        True when a remote push was attempted.

    Raises:
        RemoteError: rclone failed.
    """
    cfg = get_settings() if settings is None else settings
    remote = Remote.from_settings(cfg)
    if remote is None:
        return False
    src = data_dir / "quarantine"
    if not src.is_dir() or not any(src.iterdir()):
        return False
    copy_mutable(
        remote,
        str(src),
        f"{remote.target.rstrip('/')}/{_REMOTE_PREFIX}",
    )
    return True


def _empty_schema() -> dict[str, pl.DataType]:
    return {
        "season": pl.Utf8(),
        "source": pl.Utf8(),
        "capture_path": pl.Utf8(),
        "captured_at": pl.Datetime(time_zone="UTC"),
        "gate": pl.Utf8(),
        "detail": pl.Utf8(),
        "previous_capture_path": pl.Utf8(),
        "available_at": pl.Datetime(time_zone="UTC"),
        "schema_version": pl.Int64(),
    }
