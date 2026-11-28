"""Build a curated snapshot after capture and publish ``status/curated/latest.json``.

Producer half of "build once, pull the result" (#345). Callable without GHA via
``fplt data curate``: fetch raw → ``refresh_ops_feeds`` staging →
``build_from_staged`` → rewrite the latest pointer when ``build_id`` changes.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

from fplt.automation.ops_feeds import FeedRefreshResult, refresh_ops_feeds
from fplt.data.quarantine import quarantined_capture_paths
from fplt.data.remote import Remote, push_replace
from fplt.data.snapshots import RawSource
from fplt.schemas.manifests import BuildManifest, CuratedLatestPointer
from fplt.settings import Settings, get_settings
from fplt.store.builds import build_from_staged
from fplt.utils.timing import (
    format_kind_summary,
    timed_phase,
    timing_artifacts,
    timing_session,
    write_timing_record,
)

logger = logging.getLogger(__name__)

CURATED_LATEST_RELATIVE = "status/curated/latest.json"
_REMOTE_POINTER_KEY = "status/curated/latest.json"


@dataclass(frozen=True)
class CurateResult:
    """Outcome of one ``run_curate`` call.

    Attributes:
        feeds: Counts from the shared feed refresh.
        manifest: Curated build (new or already present).
        pointer: Local pointer after this run (``None`` when not written).
        pointer_updated: True when ``status/curated/latest.json`` was rewritten.
    """

    feeds: FeedRefreshResult
    manifest: BuildManifest
    pointer: CuratedLatestPointer | None
    pointer_updated: bool


def curated_latest_path(data_dir: Path) -> Path:
    """Return ``data_dir/status/curated/latest.json``."""
    return data_dir / CURATED_LATEST_RELATIVE


def load_curated_latest(data_dir: Path) -> CuratedLatestPointer | None:
    """Load the local latest-curated pointer, or ``None`` when missing.

    Args:
        data_dir: Local data root.

    Returns:
        Parsed ``CuratedLatestPointer``, or ``None`` when the file is absent.
    """
    path = curated_latest_path(data_dir)
    if not path.is_file():
        return None
    return CuratedLatestPointer.model_validate_json(path.read_text(encoding="utf-8"))


def write_curated_latest(
    data_dir: Path,
    pointer: CuratedLatestPointer,
) -> Path:
    """Write ``status/curated/latest.json`` locally (mutable overwrite).

    Args:
        data_dir: Local data root.
        pointer: Validated pointer document.

    Returns:
        Absolute path written.
    """
    path = curated_latest_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        pointer.model_dump_json(indent=2) + "\n",
        encoding="utf-8",
    )
    return path


def push_curated_latest(
    data_dir: Path,
    *,
    settings: Settings | None = None,
    remote: Remote | None = None,
) -> Remote | None:
    """Upload ``status/curated/latest.json`` to the data bucket (mutable).

    Args:
        data_dir: Local data root holding the pointer file.
        settings: Process settings (defaults to ``get_settings()``).
        remote: Optional pre-built remote (tests).

    Returns:
        The remote that received the file, or ``None`` when no data bucket is
        configured.

    Raises:
        FileNotFoundError: Local pointer is missing.
        ValueError: Remote kind is not the data bucket.
        RemoteError: rclone failed.
    """
    path = curated_latest_path(data_dir)
    if not path.is_file():
        raise FileNotFoundError(path)
    cfg = get_settings() if settings is None else settings
    target = remote if remote is not None else Remote.from_settings(cfg)
    if target is None:
        logger.info("curated latest push skipped: data bucket unset")
        return None
    if target.kind != "data":
        raise ValueError(
            f"curated latest must push to the data bucket; got kind={target.kind!r}"
        )
    logger.info(
        "pushing curated latest → %s/%s",
        target.target,
        _REMOTE_POINTER_KEY,
    )
    push_replace(target, path, _REMOTE_POINTER_KEY)
    return target


def pointer_from_manifest(manifest: BuildManifest) -> CuratedLatestPointer:
    """Build a ``CuratedLatestPointer`` from a ``BuildManifest``.

    Args:
        manifest: Curated build just written or already present.

    Returns:
        Pointer naming that build.
    """
    return CuratedLatestPointer(
        build_id=manifest.build_id,
        created_at=manifest.created_at,
        raw_cutoff=manifest.raw_cutoff,
        code_sha=manifest.code_sha,
    )


def run_curate(
    data_dir: Path | None = None,
    *,
    source: RawSource = "auto",
    remote: str = "origin",
    branch: str = "snapshots",
    settings: Settings | None = None,
    code_sha: str | None = None,
    push_remote: Remote | None = None,
) -> CurateResult:
    """Fetch raw, restage, build curated, and refresh the latest pointer.

    Uses ``refresh_ops_feeds`` (not a duplicated staging path). When the
    resulting ``build_id`` already matches the local pointer, the pointer is
    left untouched. Timing phases: ``remote/pull``, ``ingest``,
    ``curate/build``, ``remote/push``.

    Args:
        data_dir: Root with ``raw/`` / ``staged/`` / ``curated/``. Defaults to
            ``Settings.data_dir``.
        source: Raw fetch source: ``auto``, ``bucket``, or ``branch``.
        remote: Git remote (or local path in tests) for the snapshots branch.
        branch: Branch that holds write-once captures (default ``snapshots``).
        settings: Optional settings for bucket credentials.
        code_sha: Override git HEAD for the build manifest (tests).
        push_remote: Optional data-bucket remote for the pointer push (tests).

    Returns:
        Feed counts, build manifest, and whether the pointer was rewritten.

    Raises:
        SnapshotFetchError: Raw fetch failed.
        RemoteError: An rclone pull/push against the bucket failed.
        ValueError: No staged tables to build.
    """
    cfg = get_settings() if settings is None else settings
    root = cfg.data_dir if data_dir is None else data_dir
    previous = load_curated_latest(root)

    with timing_artifacts(root):
        with timing_session() as session:
            feeds = refresh_ops_feeds(
                root,
                source=source,
                remote=remote,
                branch=branch,
                settings=cfg,
            )
            quarantined = sorted(quarantined_capture_paths(root))
            with timed_phase("curate/build"):
                manifest = build_from_staged(
                    root,
                    quarantined_captures=quarantined,
                    code_sha=code_sha,
                )

            pointer_updated = False
            pointer: CuratedLatestPointer | None = previous
            if previous is None or previous.build_id != manifest.build_id:
                pointer = pointer_from_manifest(manifest)
                write_curated_latest(root, pointer)
                push_curated_latest(root, settings=cfg, remote=push_remote)
                pointer_updated = True
                logger.info(
                    "curated latest updated build_id=%s path=%s",
                    pointer.build_id,
                    curated_latest_path(root),
                )
            else:
                logger.info(
                    "curated latest unchanged build_id=%s",
                    manifest.build_id,
                )

            meta = {
                "build_id": manifest.build_id,
                "pointer_updated": pointer_updated,
                "snapshots_fetched": feeds.snapshots_fetched,
            }
            logger.info(format_kind_summary("curate", session, meta=meta))
            write_timing_record("curate", session, meta=meta)

    return CurateResult(
        feeds=feeds,
        manifest=manifest,
        pointer=pointer,
        pointer_updated=pointer_updated,
    )
