"""Push finished runs and curated builds to R2.

Runs (B4): official → ``fplt-data``, else ``fplt-lab``. Only runs with
``_SUCCESS`` are pushable. Curated builds (B6): always ``fplt-data/curated/``.
Routing follows ``docs/proposals/fplt-data-model.md`` §5 /
``r2-data-bucket-plan.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fplt.data.remote import Remote, push
from fplt.schemas.manifests import RunKind, RunManifest
from fplt.settings import Settings, get_settings
from fplt.store.builds import (
    build_manifest_path,
    load_build_manifest,
    table_sha_path,
)
from fplt.store.runs import run_dir

logger = logging.getLogger(__name__)

_SUCCESS = "_SUCCESS"
_MANIFEST = "manifest.json"


class IncompleteRunError(ValueError):
    """Raised when a run has no ``_SUCCESS`` marker yet."""


def load_run_manifest(data_dir: Path, run_id: str) -> RunManifest:
    """Load ``manifest.json`` for a local run.

    Args:
        data_dir: Local data root.
        run_id: Run id (folder is ``runs/run=<run_id>/``).

    Returns:
        The parsed ``RunManifest``.

    Raises:
        FileNotFoundError: The run folder or manifest is missing.
    """
    path = run_dir(data_dir, run_id) / _MANIFEST
    if not path.is_file():
        raise FileNotFoundError(path)
    return RunManifest.model_validate_json(path.read_text(encoding="utf-8"))


def bucket_kind_for_run(kind: RunKind) -> str:
    """Return ``\"data\"`` for official runs, else ``\"lab\"``."""
    return "data" if kind == "official" else "lab"


def push_run(
    data_dir: Path,
    run_id: str,
    *,
    settings: Settings | None = None,
    remote: Remote | None = None,
) -> Remote:
    """Upload a finished run to the bucket for its kind.

    Official runs go to the data bucket; adhoc, experiment, benchmark, and
    legacy runs go to the lab bucket. The local folder must contain
    ``_SUCCESS``.

    Args:
        data_dir: Local data root holding ``runs/``.
        run_id: Run to push.
        settings: Process settings (defaults to ``get_settings()``). Ignored
            when ``remote`` is provided.
        remote: Optional pre-built remote (tests). Must match the kind's
            bucket (``data`` vs ``lab``).

    Returns:
        The remote that received the run.

    Raises:
        IncompleteRunError: ``_SUCCESS`` is missing.
        FileNotFoundError: The run or manifest is missing.
        ValueError: The destination bucket is unset, or ``remote.kind`` does
            not match the run kind.
        RemoteError: rclone failed (including immutable overwrite).
    """
    path = run_dir(data_dir, run_id)
    if not (path / _SUCCESS).is_file():
        raise IncompleteRunError(
            f"run {run_id} has no _SUCCESS; refusing to push incomplete run"
        )
    manifest = load_run_manifest(data_dir, run_id)
    wanted = bucket_kind_for_run(manifest.kind)
    target = remote if remote is not None else _remote_for(wanted, settings)
    if target.kind != wanted:
        raise ValueError(
            f"run {run_id} kind={manifest.kind!r} requires the {wanted} bucket; "
            f"got remote kind={target.kind!r}"
        )
    prefix = f"runs/run={run_id}"
    logger.info(
        "pushing run_id=%s kind=%s → %s/%s",
        run_id,
        manifest.kind,
        target.target,
        prefix,
    )
    push(target, path, prefix)
    return target


def maybe_push_run(
    data_dir: Path,
    run_id: str,
    *,
    settings: Settings | None = None,
) -> Remote | None:
    """Push a finished run when its destination bucket is configured.

    No-op (returns ``None``) when the data/lab bucket for the run's kind is
    unset, so local dual-write stays offline-friendly.

    Args:
        data_dir: Local data root.
        run_id: Run to push.
        settings: Process settings; defaults to ``get_settings()``.

    Returns:
        The remote used, or ``None`` when the bucket is not configured.

    Raises:
        IncompleteRunError: ``_SUCCESS`` is missing.
        RemoteError: rclone failed when a bucket *is* configured.
    """
    manifest = load_run_manifest(data_dir, run_id)
    wanted = bucket_kind_for_run(manifest.kind)
    cfg = settings if settings is not None else get_settings()
    bucket_name = cfg.data_bucket if wanted == "data" else cfg.lab_bucket
    if not bucket_name:
        logger.debug(
            "skip push run_id=%s: %s bucket unset",
            run_id,
            wanted,
        )
        return None
    return push_run(data_dir, run_id, settings=cfg)


def push_build(
    data_dir: Path,
    build_id: str,
    *,
    settings: Settings | None = None,
    remote: Remote | None = None,
) -> Remote:
    """Upload one curated build (manifest + table shas) to ``fplt-data``.

    Pushes each referenced ``curated/tables/<table>/sha=…`` file and the
    ``curated/builds/build=<build_id>.json`` manifest. Content-addressed
    table files already present remotely are left untouched (``--immutable``).

    Args:
        data_dir: Local data root holding ``curated/``.
        build_id: Build to push.
        settings: Process settings (defaults to ``get_settings()``). Ignored
            when ``remote`` is provided.
        remote: Optional pre-built data-bucket remote (tests).

    Returns:
        The remote that received the build.

    Raises:
        FileNotFoundError: Manifest or a referenced table file is missing.
        ValueError: The data bucket is unset, or ``remote.kind`` is not
            ``\"data\"``.
        RemoteError: rclone failed (including immutable overwrite).
    """
    manifest = load_build_manifest(data_dir, build_id)
    target = remote if remote is not None else _remote_for("data", settings)
    if target.kind != "data":
        raise ValueError(
            f"curated builds require the data bucket; got remote kind={target.kind!r}"
        )
    for name, entry in sorted(manifest.tables.items()):
        local = table_sha_path(data_dir, name, entry.sha)
        if not local.is_file():
            raise FileNotFoundError(local)
        # Push the single-file parent so the remote path is
        # curated/tables/<name>/sha=<sha>.parquet.
        push(target, local.parent, f"curated/tables/{name}")
    builds_dir = build_manifest_path(data_dir, build_id).parent
    logger.info(
        "pushing curated build_id=%s tables=%s → %s/curated/",
        build_id,
        sorted(manifest.tables),
        target.target,
    )
    push(target, builds_dir, "curated/builds")
    return target


def maybe_push_build(
    data_dir: Path,
    build_id: str,
    *,
    settings: Settings | None = None,
) -> Remote | None:
    """Push a curated build when ``FPLT_DATA_BUCKET`` is configured.

    No-op (returns ``None``) when the data bucket is unset.

    Args:
        data_dir: Local data root.
        build_id: Build to push.
        settings: Process settings; defaults to ``get_settings()``.

    Returns:
        The remote used, or ``None`` when the data bucket is not configured.
    """
    cfg = settings if settings is not None else get_settings()
    if not cfg.data_bucket:
        logger.debug("skip push build_id=%s: data bucket unset", build_id)
        return None
    return push_build(data_dir, build_id, settings=cfg)


def push_official(
    data_dir: Path,
    *,
    season: str,
    gw: int,
    checkpoint: str = "T-90m",
    settings: Settings | None = None,
    remote: Remote | None = None,
) -> Remote:
    """Upload one official checkpoint folder to the data bucket.

    Args:
        data_dir: Local data root holding ``official/``.
        season: Season key.
        gw: Gameweek number.
        checkpoint: Checkpoint label (default ``T-90m``).
        settings: Process settings (defaults to ``get_settings()``). Ignored
            when ``remote`` is provided.
        remote: Optional pre-built data-bucket remote (tests).

    Returns:
        The remote that received the checkpoint.

    Raises:
        FileNotFoundError: The local checkpoint directory is missing.
        ValueError: The data bucket is unset, or ``remote.kind`` is not
            ``\"data\"``.
        RemoteError: rclone failed.
    """
    from fplt.store.official import official_checkpoint_dir

    local = official_checkpoint_dir(
        data_dir, season=season, gw=gw, checkpoint=checkpoint
    )
    if not local.is_dir():
        raise FileNotFoundError(local)
    target = remote if remote is not None else _remote_for("data", settings)
    if target.kind != "data":
        raise ValueError(
            "official pointers require the data bucket; "
            f"got remote kind={target.kind!r}"
        )
    prefix = f"official/season={season}/gw={gw}/checkpoint={checkpoint}"
    logger.info(
        "pushing official season=%s gw=%s checkpoint=%s → %s/%s",
        season,
        gw,
        checkpoint,
        target.target,
        prefix,
    )
    push(target, local, prefix)
    return target


def maybe_push_official(
    data_dir: Path,
    *,
    season: str,
    gw: int,
    checkpoint: str = "T-90m",
    settings: Settings | None = None,
) -> Remote | None:
    """Push an official checkpoint when ``FPLT_DATA_BUCKET`` is configured.

    No-op (returns ``None``) when the data bucket is unset.

    Args:
        data_dir: Local data root.
        season: Season key.
        gw: Gameweek number.
        checkpoint: Checkpoint label.
        settings: Process settings; defaults to ``get_settings()``.

    Returns:
        The remote used, or ``None`` when the data bucket is not configured.
    """
    cfg = settings if settings is not None else get_settings()
    if not cfg.data_bucket:
        logger.debug(
            "skip push official season=%s gw=%s: data bucket unset", season, gw
        )
        return None
    return push_official(
        data_dir,
        season=season,
        gw=gw,
        checkpoint=checkpoint,
        settings=cfg,
    )


def _remote_for(wanted: str, settings: Settings | None) -> Remote:
    cfg = settings if settings is not None else get_settings()
    bucket_name = cfg.data_bucket if wanted == "data" else cfg.lab_bucket
    if not bucket_name:
        raise ValueError(
            f"{wanted} bucket is not configured "
            f"(set FPLT_{'DATA' if wanted == 'data' else 'LAB'}_BUCKET)"
        )
    remote = Remote.from_settings(cfg, bucket=bucket_name)
    if remote is None:
        raise ValueError(f"could not build remote for bucket {bucket_name!r}")
    return remote
