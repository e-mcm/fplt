"""Content-addressed curated builds under ``curated/``.

Stage B6 PR1. Layout from ``docs/proposals/fplt-data-model.md`` §3.2::

    curated/
      tables/<table>/sha=<sha256>.parquet
      builds/build=<build_id>.json

``build_id`` = hash(raw_manifest_hash, staging_version, schema hashes).
Unchanged table bytes reuse the same ``sha=`` file across builds.
"""

from __future__ import annotations

import hashlib
import logging
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from fplt.data.store import DataStore
from fplt.models.spec import git_sha
from fplt.schemas.manifests import BuildManifest, BuildTableEntry
from fplt.schemas.staged import POLARS_SCHEMAS, STAGED_SCHEMAS
from fplt.store.runs import hash_raw_manifest
from fplt.types import GWKey

logger = logging.getLogger(__name__)

# Bump whenever any staging transform changes output. Same discipline as
# feature versions (AGENTS.md §2). Golden hash test fails if output drifts
# without a bump — see ``tests/unit/test_staging_version_golden.py``.
STAGING_VERSION = 1

_CAPTURED_AT_RE = re.compile(
    r"captured_at=([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}-[0-9]{2}-[0-9]{2}Z)"
)
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class BuildExistsError(FileExistsError):
    """Raised when a build manifest path already exists (write-once)."""


@dataclass(frozen=True)
class CuratedBuildSource:
    """``TableSource`` over one content-addressed curated build.

    Missing tables (not present in the build manifest) return an empty typed
    frame, matching ``DataStore.table`` behaviour for absent staged files.
    """

    data_dir: Path
    build_id: str
    _tables: Mapping[str, str]

    @classmethod
    def load(cls, data_dir: Path, build_id: str) -> CuratedBuildSource:
        """Load table digests from ``curated/builds/build=<build_id>.json``.

        Args:
            data_dir: Local data root.
            build_id: Curated build id.

        Returns:
            A source ready for ``PITData`` / ``PITView``.

        Raises:
            FileNotFoundError: Manifest is missing.
        """
        manifest = load_build_manifest(data_dir, build_id)
        tables = {name: entry.sha for name, entry in manifest.tables.items()}
        return cls(data_dir=data_dir, build_id=build_id, _tables=tables)

    def table(self, name: str) -> pl.LazyFrame:
        """Lazy scan of a curated table; empty typed frame when absent."""
        wanted = POLARS_SCHEMAS[name]
        sha = self._tables.get(name)
        if sha is None:
            return pl.DataFrame(schema=wanted).lazy()
        path = table_sha_path(self.data_dir, name, sha)
        if not path.is_file():
            raise FileNotFoundError(path)
        frame = pl.scan_parquet(path)
        # Parquet round-trips drop the UTC zone; restore before PIT compare.
        exprs: list[pl.Expr] = []
        for col, dtype in frame.collect_schema().items():
            if isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
                exprs.append(pl.col(col).dt.replace_time_zone("UTC"))
        if exprs:
            frame = frame.with_columns(exprs)
        present = set(frame.collect_schema().names())
        extras = [
            pl.lit(None).cast(dtype).alias(column)
            for column, dtype in wanted.items()
            if column not in present
        ]
        if extras:
            frame = frame.with_columns(extras)
        return frame.select(list(wanted))

    def deadline(self, gw: GWKey) -> datetime:
        """Deadline of ``gw`` from the build's ``gameweeks`` table."""
        frame = (
            self.table("gameweeks")
            .filter((pl.col("season") == gw.season) & (pl.col("gw") == gw.gw))
            .sort("available_at")
            .collect()
        )
        if frame.is_empty():
            raise KeyError(
                f"no gameweek {gw.season} gw{gw.gw} in build {self.build_id}"
            )
        value = frame["deadline_time"][-1]
        if not isinstance(value, datetime):
            raise TypeError(f"deadline_time for {gw} is {type(value)}")
        if value.tzinfo is None:
            return value.replace(tzinfo=UTC)
        return value.astimezone(UTC)


def curated_root(data_dir: Path) -> Path:
    """Local curated root: ``data_dir/curated/``."""
    return data_dir / "curated"


def build_manifest_path(data_dir: Path, build_id: str) -> Path:
    """Path for ``curated/builds/build=<build_id>.json``."""
    return curated_root(data_dir) / "builds" / f"build={build_id}.json"


def table_sha_path(data_dir: Path, table: str, sha: str) -> Path:
    """Path for ``curated/tables/<table>/sha=<sha>.parquet``."""
    return curated_root(data_dir) / "tables" / table / f"sha={sha}.parquet"


def schema_hash(table: str) -> str:
    """SHA-256 of the polars column schema for a staged table name.

    Args:
        table: Key in ``POLARS_SCHEMAS`` / ``STAGED_SCHEMAS``.

    Returns:
        Hex digest of sorted ``name:dtype`` lines.

    Raises:
        KeyError: ``table`` is not a registered staged table.
    """
    columns = POLARS_SCHEMAS[table]
    lines = [f"{name}:{dtype!s}" for name, dtype in sorted(columns.items())]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def schema_hashes() -> dict[str, str]:
    """Schema digest for every registered staged table, sorted by name."""
    return {name: schema_hash(name) for name in sorted(STAGED_SCHEMAS)}


def compute_build_id(
    *,
    raw_manifest_hash: str,
    staging_version: int = STAGING_VERSION,
    schemas: dict[str, str] | None = None,
) -> str:
    """Content-addressed build id from raw + staging + schema versions.

    Args:
        raw_manifest_hash: Digest from ``hash_raw_manifest``.
        staging_version: Staging transform version constant.
        schemas: Per-table schema digests; defaults to ``schema_hashes()``.

    Returns:
        Hex SHA-256 build id.
    """
    digests = schemas if schemas is not None else schema_hashes()
    lines = [
        f"raw:{raw_manifest_hash}",
        f"staging_version:{staging_version}",
        *[f"schema:{name}:{digests[name]}" for name in sorted(digests)],
    ]
    return hashlib.sha256("\n".join(lines).encode()).hexdigest()


def raw_cutoff(data_dir: Path) -> datetime:
    """Latest ``captured_at`` stamp found under ``data_dir/raw/``.

    Falls back to Unix epoch when raw is missing or has no stamped paths.
    """
    raw_root = data_dir / "raw"
    latest = _EPOCH
    if not raw_root.is_dir():
        return latest
    for path in raw_root.rglob("*"):
        if not path.is_file():
            continue
        for part in path.parts:
            match = _CAPTURED_AT_RE.fullmatch(part)
            if match is None:
                continue
            stamp = datetime.strptime(match.group(1), "%Y-%m-%dT%H-%M-%SZ").replace(
                tzinfo=UTC
            )
            if stamp > latest:
                latest = stamp
    return latest


def hash_parquet_bytes(data: bytes) -> str:
    """SHA-256 of parquet file bytes (content address)."""
    return hashlib.sha256(data).hexdigest()


def write_table(
    data_dir: Path,
    table: str,
    frame: pl.DataFrame,
) -> BuildTableEntry:
    """Write ``frame`` once under ``curated/tables/<table>/sha=<sha>.parquet``.

    If the content-addressed file already exists, it is left untouched (dedup).

    Args:
        data_dir: Local data root.
        table: Staged table name (must be registered).
        frame: Rows to persist (``BuildTableEntry.schema_hash`` from registry).

    Returns:
        Manifest entry for this table.

    Raises:
        KeyError: ``table`` is not registered in ``STAGED_SCHEMAS``.
        FileExistsError: Destination exists but bytes differ (corruption).
    """
    if table not in STAGED_SCHEMAS:
        raise KeyError(f"unknown staged table: {table!r}")
    # Stable bytes: fixed column order from the registry.
    ordered = frame.select(list(POLARS_SCHEMAS[table]))
    # Write via a temp path then read bytes so the on-disk encoding matches
    # what readers will see (polars parquet options stay consistent).
    scratch = curated_root(data_dir) / ".tmp"
    scratch.mkdir(parents=True, exist_ok=True)
    tmp = scratch / f"{table}.{hashlib.sha1(table.encode()).hexdigest()[:8]}.parquet"
    try:
        ordered.write_parquet(tmp)
        payload = tmp.read_bytes()
    finally:
        tmp.unlink(missing_ok=True)
    sha = hash_parquet_bytes(payload)
    dest = table_sha_path(data_dir, table, sha)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        existing = dest.read_bytes()
        if existing != payload:
            raise FileExistsError(
                f"curated table collision at {dest}: existing bytes differ"
            )
    else:
        dest.write_bytes(payload)
    return BuildTableEntry(
        sha=sha,
        rows=ordered.height,
        schema_hash=schema_hash(table),
    )


def write_build_manifest(data_dir: Path, manifest: BuildManifest) -> Path:
    """Write ``curated/builds/build=<build_id>.json`` once.

    Args:
        data_dir: Local data root.
        manifest: Validated ``BuildManifest``.

    Returns:
        Absolute path written.

    Raises:
        BuildExistsError: Manifest path already exists.
    """
    path = build_manifest_path(data_dir, manifest.build_id)
    if path.exists():
        raise BuildExistsError(f"build already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def list_build_ids(data_dir: Path) -> list[str]:
    """Sorted build ids present under ``curated/builds/``."""
    root = curated_root(data_dir) / "builds"
    if not root.is_dir():
        return []
    ids: list[str] = []
    for path in sorted(root.glob("build=*.json")):
        ids.append(path.name.removeprefix("build=").removesuffix(".json"))
    return ids


def latest_build_id(data_dir: Path) -> str | None:
    """Most recently created local build id, or ``None`` when none exist.

    Orders by ``BuildManifest.created_at``, then by ``build_id`` for ties.
    """
    best_id: str | None = None
    best_at: datetime | None = None
    for build_id in list_build_ids(data_dir):
        manifest = load_build_manifest(data_dir, build_id)
        created = manifest.created_at
        if created.tzinfo is None:
            created = created.replace(tzinfo=UTC)
        if (
            best_at is None
            or created > best_at
            or (created == best_at and build_id > (best_id or ""))
        ):
            best_at = created
            best_id = build_id
    return best_id


def load_build_manifest(data_dir: Path, build_id: str) -> BuildManifest:
    """Load a curated build manifest by id.

    Args:
        data_dir: Local data root.
        build_id: Build id (filename is ``build=<build_id>.json``).

    Returns:
        Parsed ``BuildManifest``.

    Raises:
        FileNotFoundError: Manifest is missing.
    """
    path = build_manifest_path(data_dir, build_id)
    if not path.is_file():
        raise FileNotFoundError(path)
    return BuildManifest.model_validate_json(path.read_text(encoding="utf-8"))


def build_from_staged(
    data_dir: Path,
    *,
    quarantined_captures: list[str] | None = None,
    code_sha: str | None = None,
    created_at: datetime | None = None,
    staging_version: int = STAGING_VERSION,
) -> BuildManifest:
    """Snapshot every non-empty staged table into a curated build.

    Empty staged tables are skipped. ``build_id`` is computed from the raw
    manifest hash, ``staging_version``, and registered schema hashes (not from
    table bytes), so two builds from the same raw + staging version share an
    id even when later table materialisation is still in flight.

    Args:
        data_dir: Local data root with ``raw/`` and ``staged/``.
        quarantined_captures: Ops-06 skip list paths recorded on the manifest.
        code_sha: Git HEAD; defaults to ``git_sha()``.
        created_at: Manifest timestamp; defaults to now (UTC).
        staging_version: Override for tests; production uses ``STAGING_VERSION``.

    Returns:
        The written ``BuildManifest``.

    Raises:
        BuildExistsError: A manifest for this ``build_id`` already exists.
        ValueError: No non-empty staged tables to include.
    """
    store = DataStore(data_dir)
    raw_hash = hash_raw_manifest(data_dir)
    schemas = schema_hashes()
    build_id = compute_build_id(
        raw_manifest_hash=raw_hash,
        staging_version=staging_version,
        schemas=schemas,
    )
    tables: dict[str, BuildTableEntry] = {}
    for name in sorted(STAGED_SCHEMAS):
        frame = store.read(name)
        if frame.is_empty():
            continue
        tables[name] = write_table(data_dir, name, frame)
    if not tables:
        raise ValueError(f"no staged tables under {data_dir / 'staged'}")
    existing_path = build_manifest_path(data_dir, build_id)
    if existing_path.is_file():
        existing = load_build_manifest(data_dir, build_id)
        logger.info(
            "curated build already present build_id=%s path=%s",
            build_id,
            existing_path,
        )
        _push_if_configured(data_dir, build_id)
        return existing
    moment = created_at if created_at is not None else datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    else:
        moment = moment.astimezone(UTC)
    manifest = BuildManifest(
        build_id=build_id,
        created_at=moment,
        staging_version=staging_version,
        code_sha=code_sha if code_sha is not None else git_sha(),
        raw_manifest_hash=raw_hash,
        raw_cutoff=raw_cutoff(data_dir),
        tables=tables,
        quarantined_captures=list(quarantined_captures or []),
    )
    path = write_build_manifest(data_dir, manifest)
    logger.info(
        "curated build written build_id=%s tables=%s path=%s",
        build_id,
        sorted(tables),
        path,
    )
    _push_if_configured(data_dir, build_id)
    return manifest


def _push_if_configured(data_dir: Path, build_id: str) -> None:
    """Push when the data bucket is set; keep build writes offline-safe."""
    from fplt.store.push import maybe_push_build

    maybe_push_build(data_dir, build_id)
