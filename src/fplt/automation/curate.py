"""Incremental curated builds and weekly full-rebuild parity (#345).

``fplt data curate`` (and the snapshot ``curate`` job) processes only raw
captures newer than ``status/curated/latest.json``'s ``raw_cutoff``, hydrating
``staged/`` from the previous build. ``fplt data curate --full`` restages from
all of ``raw/`` and fails (ntfy) when table shas diverge from the pointer.
"""

from __future__ import annotations

import logging
import re
import shutil
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from fplt.automation.notify import notify
from fplt.data.api_football import (
    default_id_maps_path as api_football_id_maps_path,
)
from fplt.data.api_football import (
    ingest_api_football,
)
from fplt.data.api_football import (
    load_id_maps as load_api_football_id_maps,
)
from fplt.data.quarantine import pull_quarantine, quarantined_capture_paths
from fplt.data.raw import ingest_raw, iter_snapshots
from fplt.data.remote import Remote, RemoteObject, lsjson, pull, push_replace
from fplt.data.snapshots import RawSource, fetch_raw
from fplt.data.store import DataStore
from fplt.data.understat import default_id_maps_path, ingest_understat, load_id_maps
from fplt.data.vaastav import import_available
from fplt.schemas.manifests import BuildManifest, CuratedLatestPointer
from fplt.settings import Settings, get_settings, secret_value
from fplt.store.builds import (
    build_from_staged,
    clear_staged,
    hydrate_staged_from_build,
    load_build_manifest,
    raw_cutoff,
)
from fplt.store.pull_official import OfficialPullError, pull_build
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
_CAPTURED_AT_RE = re.compile(
    r"captured_at=([0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}-[0-9]{2}-[0-9]{2}Z)"
)
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)


class CurateParityError(RuntimeError):
    """Weekly full rebuild table shas diverged from the latest pointer."""


@dataclass(frozen=True)
class CurateResult:
    """Outcome of one ``run_curate`` call.

    Attributes:
        manifest: Curated build when one was produced or compared; ``None``
            when incremental found no new raw and exited early.
        pointer: Local pointer after this run (``None`` when unchanged/absent).
        pointer_updated: True when ``status/curated/latest.json`` was rewritten.
        full: True when ``--full`` / bootstrap full rebuild ran.
        new_capture_dirs: Capture dirs pulled (incremental) or empty for full.
        skipped_no_new: True when incremental exited because nothing was new.
        parity_ok: For full rebuilds, whether table shas matched the pointer
            (``None`` when no prior pointer to compare).
    """

    manifest: BuildManifest | None
    pointer: CuratedLatestPointer | None
    pointer_updated: bool
    full: bool
    new_capture_dirs: tuple[str, ...]
    skipped_no_new: bool
    parity_ok: bool | None


def curated_latest_path(data_dir: Path) -> Path:
    """Return ``data_dir/status/curated/latest.json``."""
    return data_dir / CURATED_LATEST_RELATIVE


def load_curated_latest(data_dir: Path) -> CuratedLatestPointer | None:
    """Load the local latest-curated pointer, or ``None`` when missing."""
    path = curated_latest_path(data_dir)
    if not path.is_file():
        return None
    return CuratedLatestPointer.model_validate_json(path.read_text(encoding="utf-8"))


def write_curated_latest(data_dir: Path, pointer: CuratedLatestPointer) -> Path:
    """Write ``status/curated/latest.json`` locally (mutable overwrite)."""
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
    """Upload ``status/curated/latest.json`` to the data bucket (mutable)."""
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


def pull_curated_latest(
    data_dir: Path,
    remote: Remote,
) -> CuratedLatestPointer | None:
    """Pull ``status/curated/latest.json`` from the data bucket when present."""
    from fplt.data.remote import RemoteError

    dest = curated_latest_path(data_dir)
    dest.parent.mkdir(parents=True, exist_ok=True)
    try:
        pull(remote, "status/curated", dest.parent)
    except RemoteError as exc:
        detail = str(exc).lower()
        if "directory not found" in detail or "not found" in detail:
            return load_curated_latest(data_dir)
        raise
    return load_curated_latest(data_dir)


def pointer_for_build(
    manifest: BuildManifest,
    *,
    cutoff: datetime,
) -> CuratedLatestPointer:
    """Build a latest pointer; ``cutoff`` is the incremental watermark."""
    return CuratedLatestPointer(
        build_id=manifest.build_id,
        created_at=manifest.created_at,
        raw_cutoff=cutoff,
        code_sha=manifest.code_sha,
    )


def capture_dirs_after(
    objects: list[RemoteObject] | list[str],
    after: datetime,
) -> list[str]:
    """Unique capture-dir prefixes with ``captured_at`` strictly after ``after``.

    Drops any path with a ``.``-prefixed component (#339). Paths may be
    ``RemoteObject.path`` values relative to ``raw/`` or bare strings.

    Args:
        objects: Listed raw object paths (or ``RemoteObject`` rows).
        after: Exclusive lower bound on ``captured_at``.

    Returns:
        Sorted capture directory prefixes (posix, relative to ``raw/``).
    """
    after_utc = after if after.tzinfo is not None else after.replace(tzinfo=UTC)
    after_utc = after_utc.astimezone(UTC)
    dirs: set[str] = set()
    for item in objects:
        path = item.path if isinstance(item, RemoteObject) else item
        parts = Path(path).parts
        if any(part.startswith(".") for part in parts):
            continue
        stamp: datetime | None = None
        stamp_idx: int | None = None
        for idx, part in enumerate(parts):
            match = _CAPTURED_AT_RE.fullmatch(part)
            if match is None:
                continue
            stamp = datetime.strptime(match.group(1), "%Y-%m-%dT%H-%M-%SZ").replace(
                tzinfo=UTC
            )
            stamp_idx = idx
            break
        if stamp is None or stamp_idx is None or stamp <= after_utc:
            continue
        dirs.add("/".join(parts[: stamp_idx + 1]))
    return sorted(dirs)


def pull_capture_dirs(
    data_dir: Path,
    remote: Remote,
    capture_dirs: list[str],
) -> list[str]:
    """Pull selected ``raw/`` capture directories into ``data_dir/raw``.

    Args:
        data_dir: Local data root.
        remote: Data-bucket remote.
        capture_dirs: Prefixes relative to ``raw/`` (from ``capture_dirs_after``).

    Returns:
        The capture dirs that were requested (for logging / tests).
    """
    raw_root = data_dir / "raw"
    for rel in capture_dirs:
        dest = raw_root / rel
        dest.mkdir(parents=True, exist_ok=True)
        pull(remote, f"raw/{rel}", dest)
    return list(capture_dirs)


def _side_ingest(store: DataStore, raw_root: Path) -> None:
    """Stage vaastav / understat / api-football from whatever raw is local."""
    import_available(store, store.root)
    player_keys, team_keys = load_id_maps(default_id_maps_path())
    ingest_understat(store, raw_root, player_keys, team_keys)
    af_players, af_teams = load_api_football_id_maps(api_football_id_maps_path())
    ingest_api_football(store, raw_root, af_players, af_teams)


def _publish_pointer(
    data_dir: Path,
    manifest: BuildManifest,
    *,
    cutoff: datetime,
    settings: Settings,
    push_remote: Remote | None,
) -> tuple[CuratedLatestPointer, bool]:
    """Write + push the latest pointer; return ``(pointer, updated)``."""
    previous = load_curated_latest(data_dir)
    pointer = pointer_for_build(manifest, cutoff=cutoff)
    if (
        previous is not None
        and previous.build_id == pointer.build_id
        and previous.raw_cutoff == pointer.raw_cutoff
        and previous.code_sha == pointer.code_sha
    ):
        return previous, False
    write_curated_latest(data_dir, pointer)
    push_curated_latest(data_dir, settings=settings, remote=push_remote)
    return pointer, True


def _table_shas(manifest: BuildManifest) -> dict[str, str]:
    return {name: entry.sha for name, entry in sorted(manifest.tables.items())}


def _compare_parity(
    full: BuildManifest,
    previous: CuratedLatestPointer,
    data_dir: Path,
) -> bool:
    """Return True when ``full`` table shas match the pointer's build."""
    try:
        prior = load_build_manifest(data_dir, previous.build_id)
    except FileNotFoundError:
        return False
    return _table_shas(full) == _table_shas(prior)


def run_curate(
    data_dir: Path | None = None,
    *,
    full: bool = False,
    source: RawSource = "auto",
    remote: str = "origin",
    branch: str = "snapshots",
    settings: Settings | None = None,
    code_sha: str | None = None,
    push_remote: Remote | None = None,
    data_remote: Remote | None = None,
    notify_parity: bool = True,
) -> CurateResult:
    """Build a curated snapshot incrementally or via full rebuild.

    Incremental (default): read / pull ``status/curated/latest.json``, hydrate
    ``staged/`` from that build, pull only raw capture dirs newer than
    ``raw_cutoff``, stage those captures, build, and rewrite the pointer.
    Missing pointer bootstraps via the full path. No new captures → exit
    without changing the pointer.

    Full (``full=True``): pull all raw, stage from scratch, build, and compare
    table shas to the previous pointer (ntfy + raise on mismatch).

    Args:
        data_dir: Local data root. Defaults to ``Settings.data_dir``.
        full: Force weekly full-rebuild parity path.
        source: Raw fetch mode for the full / bootstrap path.
        remote: Git remote for snapshots branch fallback.
        branch: Snapshots branch name.
        settings: Process settings.
        code_sha: Override git HEAD on the build manifest (tests).
        push_remote: Optional remote for pointer / build push (tests).
        data_remote: Optional data-bucket remote for listing/pulling (tests).
        notify_parity: When True, ntfy on full-rebuild sha mismatch.

    Returns:
        ``CurateResult`` summarizing the run.

    Raises:
        CurateParityError: Full rebuild table shas diverged from the pointer.
        OfficialPullError: Previous curated build missing from the bucket.
        RemoteError: rclone failed.
        ValueError: No staged tables to build.
    """
    cfg = get_settings() if settings is None else settings
    root = cfg.data_dir if data_dir is None else data_dir
    bucket = data_remote if data_remote is not None else Remote.from_settings(cfg)

    with timing_artifacts(root):
        with timing_session() as session:
            if full:
                result = _run_full(
                    root,
                    cfg=cfg,
                    source=source,
                    remote=remote,
                    branch=branch,
                    code_sha=code_sha,
                    push_remote=push_remote,
                    data_remote=bucket,
                    notify_parity=notify_parity,
                )
            else:
                result = _run_incremental(
                    root,
                    cfg=cfg,
                    source=source,
                    remote=remote,
                    branch=branch,
                    code_sha=code_sha,
                    push_remote=push_remote,
                    data_remote=bucket,
                    notify_parity=notify_parity,
                )
            meta = {
                "full": result.full,
                "pointer_updated": result.pointer_updated,
                "skipped_no_new": result.skipped_no_new,
                "new_captures": len(result.new_capture_dirs),
                "build_id": result.manifest.build_id if result.manifest else None,
                "parity_ok": result.parity_ok,
            }
            logger.info(format_kind_summary("curate", session, meta=meta))
            write_timing_record("curate", session, meta=meta)
    return result


def _run_incremental(
    root: Path,
    *,
    cfg: Settings,
    source: RawSource,
    remote: str,
    branch: str,
    code_sha: str | None,
    push_remote: Remote | None,
    data_remote: Remote | None,
    notify_parity: bool,
) -> CurateResult:
    previous = load_curated_latest(root)
    if previous is None and data_remote is not None:
        with timed_phase("remote/pull"):
            previous = pull_curated_latest(root, data_remote)

    if previous is None:
        logger.info("curate: no latest pointer; bootstrapping full rebuild")
        return _run_full(
            root,
            cfg=cfg,
            source=source,
            remote=remote,
            branch=branch,
            code_sha=code_sha,
            push_remote=push_remote,
            data_remote=data_remote,
            notify_parity=notify_parity,
            bootstrap=True,
        )

    if data_remote is None:
        raise ValueError(
            "incremental curate requires a data bucket remote "
            "(FPLT_DATA_BUCKET + R2 credentials)"
        )

    with timed_phase("remote/pull"):
        pull_quarantine(root, cfg)
        pull_build(root, data_remote, previous.build_id)
        objects = lsjson(data_remote, "raw")
        new_dirs = capture_dirs_after(objects, previous.raw_cutoff)
        if not new_dirs:
            logger.info(
                "curate: no raw captures after raw_cutoff=%s; unchanged",
                previous.raw_cutoff.isoformat(),
            )
            return CurateResult(
                manifest=None,
                pointer=previous,
                pointer_updated=False,
                full=False,
                new_capture_dirs=(),
                skipped_no_new=True,
                parity_ok=None,
            )
        pull_capture_dirs(root, data_remote, new_dirs)

    clear_staged(root)
    with timed_phase("curate/hydrate"):
        hydrate_staged_from_build(root, previous.build_id)

    store = DataStore(root)
    raw_root = root / "raw"
    snaps = [
        snap
        for snap in iter_snapshots(raw_root)
        if snap.captured_at > previous.raw_cutoff
    ]
    ingest_raw(
        store,
        raw_root,
        snapshots=snaps,
        event_live_after=previous.raw_cutoff,
    )
    _side_ingest(store, raw_root)

    quarantined = sorted(quarantined_capture_paths(root))
    with timed_phase("curate/build"):
        manifest = build_from_staged(
            root,
            quarantined_captures=quarantined,
            code_sha=code_sha,
        )
    cutoff = max(previous.raw_cutoff, raw_cutoff(root))
    pointer, updated = _publish_pointer(
        root,
        manifest,
        cutoff=cutoff,
        settings=cfg,
        push_remote=push_remote if push_remote is not None else data_remote,
    )
    return CurateResult(
        manifest=manifest,
        pointer=pointer,
        pointer_updated=updated,
        full=False,
        new_capture_dirs=tuple(new_dirs),
        skipped_no_new=False,
        parity_ok=None,
    )


def _run_full(
    root: Path,
    *,
    cfg: Settings,
    source: RawSource,
    remote: str,
    branch: str,
    code_sha: str | None,
    push_remote: Remote | None,
    data_remote: Remote | None,
    notify_parity: bool,
    bootstrap: bool = False,
) -> CurateResult:
    previous = load_curated_latest(root)
    if previous is None and data_remote is not None and not bootstrap:
        with timed_phase("remote/pull"):
            previous = pull_curated_latest(root, data_remote)
            if previous is not None:
                try:
                    pull_build(root, data_remote, previous.build_id)
                except OfficialPullError:
                    logger.warning(
                        "curate full: prior build %s missing; parity skipped",
                        previous.build_id,
                    )
                    previous = None

    with timed_phase("remote/pull"):
        pull_quarantine(root, cfg)
        if data_remote is not None and source in ("auto", "bucket"):
            # Full rebuild is the only path that copies all of raw (#345).
            raw_dest = root / "raw"
            if raw_dest.exists():
                shutil.rmtree(raw_dest)
            raw_dest.mkdir(parents=True, exist_ok=True)
            pull(data_remote, "raw", raw_dest)
        else:
            fetch_raw(
                root / "raw",
                cfg,
                source=source,
                remote=remote,
                branch=branch,
            )

    clear_staged(root)
    store = DataStore(root)
    raw_root = root / "raw"
    ingest_raw(store, raw_root)
    _side_ingest(store, raw_root)

    quarantined = sorted(quarantined_capture_paths(root))
    with timed_phase("curate/build"):
        manifest = build_from_staged(
            root,
            quarantined_captures=quarantined,
            code_sha=code_sha,
        )

    parity_ok: bool | None = None
    if previous is not None:
        parity_ok = _compare_parity(manifest, previous, root)
        if not parity_ok:
            msg = (
                f"curate full parity failed: build_id={manifest.build_id} "
                f"vs pointer={previous.build_id}"
            )
            logger.error(msg)
            if notify_parity:
                notify(
                    secret_value(cfg.ntfy_url),
                    "fplt curate parity",
                    msg,
                    runbook_anchor="data-curate",
                )
            raise CurateParityError(msg)

    cutoff = raw_cutoff(root)
    if cutoff == _EPOCH and previous is not None:
        cutoff = previous.raw_cutoff
    pointer, updated = _publish_pointer(
        root,
        manifest,
        cutoff=cutoff,
        settings=cfg,
        push_remote=push_remote if push_remote is not None else data_remote,
    )
    return CurateResult(
        manifest=manifest,
        pointer=pointer,
        pointer_updated=updated,
        full=True,
        new_capture_dirs=(),
        skipped_no_new=False,
        parity_ok=parity_ok,
    )
