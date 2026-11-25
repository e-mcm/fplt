"""Persist ops-06 quarantine rows and expose the staging skip list.

Rows live at ``$FPLT_DATA_DIR/quarantine/part.parquet`` (not under
``staged/``). ``ingest_raw`` skips any capture path listed here until curated
builds carry ``quarantined_captures`` in the build manifest.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from fplt.schemas.quarantine import QuarantineRecordSchema

_FILE = "part.parquet"


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
