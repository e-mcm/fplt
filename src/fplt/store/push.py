"""Push finished runs to R2 (official → ``fplt-data``, else ``fplt-lab``).

Stage B4 PR2. Only runs with ``_SUCCESS`` are pushable. Routing follows
``docs/proposals/fplt-data-model.md`` §5 / ``r2-data-bucket-plan.md``.
"""

from __future__ import annotations

import logging
from pathlib import Path

from fplt.data.remote import Remote, push
from fplt.schemas.manifests import RunKind, RunManifest
from fplt.settings import Settings, get_settings
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

    Official runs go to the data bucket; adhoc, experiment, and benchmark
    runs go to the lab bucket. The local folder must contain ``_SUCCESS``.

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
