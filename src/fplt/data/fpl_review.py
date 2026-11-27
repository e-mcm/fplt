"""Manual FPL Review CSV/JSON drops into ``benchmark_projections``.

No scraping. Premium UI exports (or a hand-built canonical file) are staged
with ``source=fpl_review`` via ``merge_benchmark_source``. See
``docs/fpl_review.md`` for terms and drop layout.
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl

from fplt.data.benchmark_projections import merge_benchmark_source
from fplt.schemas.benchmark_projections import (
    BENCHMARK_COLUMNS,
    SOURCE_FPL_REVIEW,
    BenchmarkProjectionSchema,
)
from fplt.types import GWKey
from fplt.utils.frames import require_dataframe

_LOG = logging.getLogger(__name__)

_REQUIRED = (
    "as_of_season",
    "as_of_gw",
    "target_season",
    "target_gw",
    "player_key",
    "fixture_id",
    "xpts",
)

_ALIASES: dict[str, str] = {
    "season": "as_of_season",
    "as_of": "as_of_season",
    "gw": "as_of_gw",
    "gameweek": "as_of_gw",
    "Gameweek": "as_of_gw",
    "GW": "as_of_gw",
    "target_season_name": "target_season",
    "target_gw_num": "target_gw",
    "player_id": "player_key",
    "PlayerKey": "player_key",
    "fixture": "fixture_id",
    "FixtureID": "fixture_id",
    "FixtureId": "fixture_id",
    "EV": "xpts",
    "Pts": "xpts",
    "GW_Pts": "xpts",
    "available_at_utc": "available_at",
    "captured_at": "available_at",
}


def fpl_review_raw_root(data_dir: Path) -> Path:
    """Return ``data_dir/raw/fpl_review`` (write-once drop root)."""
    return data_dir / "raw" / "fpl_review"


def parse_available_at(value: object) -> datetime:
    """Parse a UTC timestamp from a drop cell or CLI argument.

    Args:
        value: ISO-8601 string, datetime, or unix seconds.

    Returns:
        Timezone-aware UTC datetime.

    Raises:
        ValueError: The value cannot be parsed as a timestamp.
    """
    if isinstance(value, datetime):
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value), tz=UTC)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            raise ValueError("empty available_at")
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=UTC)
        return parsed.astimezone(UTC)
    raise ValueError(f"unsupported available_at type: {type(value)!r}")


def _rename_aliases(frame: pl.DataFrame) -> pl.DataFrame:
    renames = {
        column: _ALIASES[column]
        for column in frame.columns
        if column in _ALIASES and _ALIASES[column] not in frame.columns
    }
    return frame.rename(renames) if renames else frame


def _read_drop(path: Path) -> pl.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pl.read_csv(path)
    if suffix == ".json":
        payload: Any = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(payload, dict):
            mapping = cast(dict[str, Any], payload)
            rows: Any = (
                mapping.get("rows") or mapping.get("projections") or mapping.get("data")
            )
            if rows is None:
                raise ValueError(
                    f"{path}: JSON object needs a rows/projections/data array"
                )
            if not isinstance(rows, list):
                raise ValueError(f"{path}: projections payload must be a list")
            return pl.DataFrame(cast(list[dict[str, Any]], rows))
        if isinstance(payload, list):
            return pl.DataFrame(cast(list[dict[str, Any]], payload))
        raise ValueError(f"{path}: JSON root must be an object or array")
    raise ValueError(f"{path}: expected .csv or .json drop (got {suffix!r})")


def _ensure_utc(frame: pl.DataFrame) -> pl.DataFrame:
    dtype = frame.schema.get("available_at")
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
        return frame.with_columns(pl.col("available_at").dt.replace_time_zone("UTC"))
    return frame


def load_fpl_review_drop(
    path: Path,
    *,
    available_at: datetime | str | None = None,
    as_of: GWKey | None = None,
) -> pl.DataFrame:
    """Load a manual FPL Review CSV/JSON drop into ``BenchmarkProjectionSchema``.

    Required columns (after alias normalisation): ``as_of_season``,
    ``as_of_gw``, ``target_season``, ``target_gw``, ``player_key``,
    ``fixture_id``, ``xpts``. Shorthand ``season``/``gw`` fill both as_of and
    target when the longer names are absent. ``source`` is always
    ``fpl_review``.

    Args:
        path: Drop file (``.csv`` or ``.json``).
        available_at: Override / default PIT stamp when the file omits it.
        as_of: Optional deadline override applied to every row's as_of fields.

    Returns:
        Frame matching ``BenchmarkProjectionSchema``.

    Raises:
        FileNotFoundError: ``path`` does not exist.
        ValueError: Missing columns, empty file, or unparseable timestamps.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    raw = _read_drop(path)
    if raw.is_empty():
        raise ValueError(f"{path}: drop file has no rows")
    frame = _rename_aliases(raw)

    # Horizon-1 shorthand: target defaults to as_of when omitted.
    if "target_season" not in frame.columns and "as_of_season" in frame.columns:
        frame = frame.with_columns(pl.col("as_of_season").alias("target_season"))
    if "target_gw" not in frame.columns and "as_of_gw" in frame.columns:
        frame = frame.with_columns(pl.col("as_of_gw").alias("target_gw"))

    if as_of is not None:
        frame = frame.with_columns(
            pl.lit(as_of.season).alias("as_of_season"),
            pl.lit(as_of.gw).cast(pl.Int64).alias("as_of_gw"),
        )
        if "target_season" not in frame.columns:
            frame = frame.with_columns(pl.lit(as_of.season).alias("target_season"))
        if "target_gw" not in frame.columns:
            frame = frame.with_columns(
                pl.lit(as_of.gw).cast(pl.Int64).alias("target_gw")
            )

    missing = [name for name in _REQUIRED if name not in frame.columns]
    if missing:
        raise ValueError(f"{path}: missing required columns {missing}")

    stamp: datetime | None = None
    if available_at is not None:
        stamp = parse_available_at(available_at)

    if "available_at" in frame.columns:
        stamps = [
            parse_available_at(value) for value in frame["available_at"].to_list()
        ]
        frame = frame.with_columns(
            pl.Series("available_at", stamps, dtype=pl.Datetime(time_zone="UTC"))
        )
    elif stamp is not None:
        frame = frame.with_columns(
            pl.lit(stamp).cast(pl.Datetime(time_zone="UTC")).alias("available_at")
        )
    else:
        raise ValueError(
            f"{path}: set available_at in the file or pass available_at=..."
        )

    if "quality_flag" not in frame.columns:
        frame = frame.with_columns(pl.lit(None).cast(pl.Utf8).alias("quality_flag"))
    else:
        frame = frame.with_columns(pl.col("quality_flag").cast(pl.Utf8))

    frame = frame.with_columns(pl.lit(SOURCE_FPL_REVIEW).alias("source"))
    ordered = frame.select(list(BENCHMARK_COLUMNS))
    checked = BenchmarkProjectionSchema.validate(ordered)
    checked = require_dataframe(checked)
    return _ensure_utc(checked)


def ingest_fpl_review_drop(
    data_dir: Path,
    path: Path,
    *,
    available_at: datetime | str | None = None,
    as_of: GWKey | None = None,
) -> tuple[Path, int]:
    """Merge one FPL Review drop into the deadline ``benchmark_projections``.

    Args:
        data_dir: ``$FPLT_DATA_DIR``.
        path: Manual CSV/JSON drop.
        available_at: PIT stamp when the file omits ``available_at``.
        as_of: Optional deadline override.

    Returns:
        ``(parquet_path, n_rows)`` written for ``source=fpl_review``.
    """
    frame = load_fpl_review_drop(path, available_at=available_at, as_of=as_of)
    key = GWKey(str(frame["as_of_season"][0]), int(frame["as_of_gw"][0]))
    if as_of is not None and key != as_of:
        raise ValueError(f"drop as_of {key} does not match requested {as_of}")
    dest = merge_benchmark_source(data_dir, key, frame)
    _LOG.info("merged %s fpl_review rows into %s", frame.height, dest)
    return dest, frame.height


def copy_drop_to_raw(
    data_dir: Path,
    path: Path,
    *,
    as_of: GWKey,
    captured_at: datetime,
) -> Path:
    """Copy a drop into the write-once raw tree (does not overwrite).

    Layout: ``raw/fpl_review/season=…/gw=…/captured_at=…/<name>``.

    Args:
        data_dir: ``$FPLT_DATA_DIR``.
        path: Source drop file.
        as_of: Deadline the drop belongs to.
        captured_at: Save time used in the path stamp.

    Returns:
        Destination path under ``raw/fpl_review/``.

    Raises:
        FileExistsError: A file already exists at the destination.
        FileNotFoundError: ``path`` is missing.
    """
    if not path.is_file():
        raise FileNotFoundError(path)
    stamp = captured_at.astimezone(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    dest_dir = (
        fpl_review_raw_root(data_dir)
        / f"season={as_of.season}"
        / f"gw={as_of.gw}"
        / f"captured_at={stamp}"
    )
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / path.name
    if dest.exists():
        raise FileExistsError(f"raw Review drop already exists: {dest}")
    dest.write_bytes(path.read_bytes())
    return dest
