"""Parquet store for derived tables. Writes are validated and never edited in place."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pandera.polars as pa
import polars as pl

from fplt.schemas.staged import POLARS_SCHEMAS, STAGED_SCHEMAS, TABLE_KEYS
from fplt.types import GWKey


def _ensure_utc(frame: pl.LazyFrame) -> pl.LazyFrame:
    """Parquet round-trips drop the UTC zone. Put it back before comparisons."""
    exprs: list[pl.Expr] = []
    for name, dtype in frame.collect_schema().items():
        if isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
            exprs.append(pl.col(name).dt.replace_time_zone("UTC"))
    if not exprs:
        return frame
    return frame.with_columns(exprs)


class DataStore:
    """Read and append staged parquet under ``root/staged``."""

    def __init__(self, root: Path) -> None:
        self.root = root

    def path(self, name: str) -> Path:
        """Parquet path for a staged table."""
        return self.root / "staged" / name / "part.parquet"

    def write(self, name: str, frame: pl.DataFrame) -> pl.DataFrame:
        """Validate ``frame``, append it, and collapse exact key duplicates."""
        schema = STAGED_SCHEMAS[name]
        frame = _ensure_utc(frame.lazy()).collect()
        checked = schema.validate(frame)
        assert isinstance(checked, pl.DataFrame)
        checked = _ensure_utc(checked.lazy()).collect()
        path = self.path(name)
        path.parent.mkdir(parents=True, exist_ok=True)
        if path.exists():
            existing = _ensure_utc(pl.read_parquet(path).lazy()).collect()
            checked = pl.concat([existing, checked], how="vertical_relaxed")
        keys = TABLE_KEYS[name]
        checked = checked.unique(subset=keys, keep="last").sort(keys)
        checked.write_parquet(path)
        return checked

    def table(self, name: str) -> pl.LazyFrame:
        """Lazy scan of a staged table. Missing tables are empty and typed."""
        path = self.path(name)
        if not path.exists():
            return pl.DataFrame(schema=POLARS_SCHEMAS[name]).lazy()
        return _ensure_utc(pl.scan_parquet(path))

    def read(self, name: str) -> pl.DataFrame:
        """Collect a staged table."""
        return self.table(name).collect()

    def deadline(self, gw: GWKey) -> datetime:
        """Deadline of ``gw`` from the latest visible gameweeks row."""
        frame = (
            self.table("gameweeks")
            .filter((pl.col("season") == gw.season) & (pl.col("gw") == gw.gw))
            .sort("available_at")
            .collect()
        )
        if frame.is_empty():
            raise KeyError(f"no gameweek {gw.season} gw{gw.gw}")
        value = frame["deadline_time"][-1]
        if not isinstance(value, datetime):
            raise TypeError(f"deadline_time for {gw} is {type(value)}")
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


def validate_table(name: str, frame: pl.DataFrame) -> pl.DataFrame:
    """Validate ``frame`` against the staged schema for ``name``."""
    schema: type[pa.DataFrameModel] = STAGED_SCHEMAS[name]
    checked = schema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked
