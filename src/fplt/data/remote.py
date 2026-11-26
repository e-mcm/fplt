"""rclone transfers against the fplt R2 data and lab buckets.

Credentials travel only in the process environment (``RCLONE_CONFIG_FPLT_*``),
never on the argv. Library and feature code still read a local
``$FPLT_DATA_DIR``; this module is the edge that pulls and pushes.

Each successful ``pull`` / ``push`` / ``check_one_way`` records wall-clock
seconds (and bytes / object counts when known) into the active timing
session when one is bound, and writes a ``kind=remote`` JSON artifact under
``$FPLT_DATA_DIR/timing/`` for ``fplt perf-report``.
"""

from __future__ import annotations

import logging
import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fplt.settings import Settings, get_settings
from fplt.utils.timing import (
    TimingSession,
    active_timing_dir,
    timed_phase,
    write_timing_record,
)

logger = logging.getLogger(__name__)

_RCLONE_INSTALL_HINT = (
    "rclone is not installed; install with "
    "`brew install rclone` or `sudo apt-get install -y rclone`"
)

_RemoteOp = Literal["pull", "push", "check"]

PUSHABLE: dict[str, frozenset[str]] = {
    "data": frozenset(
        {"raw", "curated", "specs", "runs", "official", "catalog", "status"}
    ),
    "lab": frozenset({"runs", "experiments", "batches", "specs", "catalog"}),
}


class RemoteError(RuntimeError):
    """An rclone transfer against the bucket failed."""


@dataclass(frozen=True)
class Remote:
    """An rclone target plus the env that configures it (credentials never in argv)."""

    target: str  # "fplt:fplt-data", or a local dir in tests
    kind: Literal["data", "lab"]
    env: dict[str, str]

    @classmethod
    def from_settings(cls, s: Settings, bucket: str | None = None) -> Remote | None:
        """Build the R2 remote from settings; ``None`` when no bucket is configured.

        Args:
            s: Process settings with R2 credentials and bucket names.
            bucket: Bucket name to target. Defaults to ``s.data_bucket``.
                Must equal ``s.data_bucket`` or ``s.lab_bucket``.

        Returns:
            A configured remote, or ``None`` when the chosen bucket is unset.

        Raises:
            ValueError: ``bucket`` is set but matches neither configured name.
        """
        name = s.data_bucket if bucket is None else bucket
        if not name:
            return None
        if name == s.data_bucket:
            kind: Literal["data", "lab"] = "data"
        elif name == s.lab_bucket:
            kind = "lab"
        else:
            raise ValueError(
                f"unknown bucket {name!r}; expected "
                f"data_bucket={s.data_bucket!r} or lab_bucket={s.lab_bucket!r}"
            )
        return cls(target=f"fplt:{name}", kind=kind, env=_rclone_env(s))


def _rclone_env(settings: Settings) -> dict[str, str]:
    """Map settings onto ``RCLONE_CONFIG_FPLT_*`` env vars for the ``fplt`` remote."""
    env: dict[str, str] = {
        "RCLONE_CONFIG_FPLT_TYPE": "s3",
        "RCLONE_CONFIG_FPLT_PROVIDER": "Cloudflare",
        "RCLONE_CONFIG_FPLT_NO_CHECK_BUCKET": "true",
    }
    if settings.r2_account_id:
        env["RCLONE_CONFIG_FPLT_ENDPOINT"] = (
            f"https://{settings.r2_account_id}.r2.cloudflarestorage.com"
        )
    if settings.r2_access_key_id:
        env["RCLONE_CONFIG_FPLT_ACCESS_KEY_ID"] = settings.r2_access_key_id
    if settings.r2_secret_access_key is not None:
        env["RCLONE_CONFIG_FPLT_SECRET_ACCESS_KEY"] = (
            settings.r2_secret_access_key.get_secret_value()
        )
    return env


def pull(remote: Remote, prefix: str, dest: Path) -> list[Path]:
    """Copy new objects under ``prefix`` into ``dest``; never overwrite local files.

    Args:
        remote: rclone target (R2 or a local directory in tests).
        prefix: Object prefix under the bucket (e.g. ``raw``).
        dest: Local directory that receives the prefix contents.

    Returns:
        Newly written local paths (diff of the file set before and after).

    Raises:
        RemoteError: rclone failed, including a local/remote content mismatch.
    """
    dest.mkdir(parents=True, exist_ok=True)
    before = _file_set(dest)
    source = f"{remote.target.rstrip('/')}/{prefix.strip('/')}"
    start = time.perf_counter()
    with timed_phase("remote/pull"):
        _rclone(
            [
                "copy",
                source,
                str(dest),
                "--immutable",
                "--fast-list",
                "--transfers",
                "16",
            ],
            env=remote.env,
        )
    elapsed = time.perf_counter() - start
    after = _file_set(dest)
    written = sorted(after - before)
    nbytes = _total_size(written)
    _emit_remote_timing(
        "pull",
        remote=remote,
        prefix=prefix,
        elapsed=elapsed,
        n_objects=len(written),
        nbytes=nbytes,
    )
    return written


def push(remote: Remote, src: Path, prefix: str) -> None:
    """Upload new files under ``src`` to ``prefix``; refuse non-whitelisted prefixes.

    Args:
        remote: rclone target (R2 or a local directory in tests).
        src: Local directory whose contents are copied.
        prefix: Destination prefix under the bucket (e.g. ``raw`` or ``runs/…``).

    Raises:
        ValueError: The prefix root is not in ``PUSHABLE`` for ``remote.kind``.
        RemoteError: rclone failed, including an immutable overwrite attempt.
    """
    root = prefix.strip("/").split("/", 1)[0]
    allowed = PUSHABLE[remote.kind]
    if root not in allowed:
        raise ValueError(
            f"prefix {prefix!r} is not pushable to the {remote.kind} bucket; "
            f"allowed roots: {sorted(allowed)}"
        )
    dest = f"{remote.target.rstrip('/')}/{prefix.strip('/')}"
    files = sorted(_file_set(src))
    nbytes = _total_size(files)
    start = time.perf_counter()
    with timed_phase("remote/push"):
        _rclone(
            ["copy", str(src), dest, "--immutable", "--fast-list"],
            env=remote.env,
        )
    elapsed = time.perf_counter() - start
    _emit_remote_timing(
        "push",
        remote=remote,
        prefix=prefix,
        elapsed=elapsed,
        n_objects=len(files),
        nbytes=nbytes,
    )


def copy_mutable(remote: Remote, source: str, dest: str) -> None:
    """rclone ``copy`` without ``--immutable`` for rewritten prefixes (``status/``).

    Args:
        remote: rclone target (credentials via ``remote.env``).
        source: Local path or ``fplt:bucket/prefix`` source.
        dest: Local path or ``fplt:bucket/prefix`` destination.

    Raises:
        RemoteError: rclone failed.
    """
    _rclone(["copy", source, dest, "--fast-list"], env=remote.env)


def push_replace(remote: Remote, src: Path, object_key: str) -> None:
    """Upload one file with overwrite allowed (mutable ``status/`` / ``catalog/``).

    Uses ``rclone copyto`` without ``--immutable`` so ops can rewrite
    ``status/status.json`` each tick. Only ``status`` and ``catalog`` roots are
    accepted on the data bucket.

    Args:
        remote: rclone target (R2 or a local directory in tests).
        src: Local file to upload.
        object_key: Destination key under the bucket (e.g. ``status/status.json``).

    Raises:
        ValueError: The key root is not a mutable rewrite prefix, or ``src`` is
            not a file.
        RemoteError: rclone failed.
    """
    if not src.is_file():
        raise ValueError(f"push_replace requires a file; got {src}")
    key = object_key.strip("/")
    root = key.split("/", 1)[0]
    mutable = frozenset({"status", "catalog"})
    if remote.kind != "data" or root not in mutable:
        raise ValueError(
            f"object key {object_key!r} is not a mutable rewrite on the "
            f"{remote.kind} bucket; allowed roots: {sorted(mutable)} on data"
        )
    if root not in PUSHABLE[remote.kind]:
        raise ValueError(
            f"prefix root {root!r} is not pushable to the {remote.kind} bucket"
        )
    dest = f"{remote.target.rstrip('/')}/{key}"
    nbytes = src.stat().st_size
    start = time.perf_counter()
    with timed_phase("remote/push"):
        _rclone(
            ["copyto", str(src), dest, "--fast-list"],
            env=remote.env,
        )
    elapsed = time.perf_counter() - start
    _emit_remote_timing(
        "push",
        remote=remote,
        prefix=key,
        elapsed=elapsed,
        n_objects=1,
        nbytes=nbytes,
    )


def check_one_way(src: Path, remote: Remote, prefix: str) -> None:
    """Assert every file under ``src`` exists under ``prefix`` with the same size.

    Uses ``rclone check --one-way`` (source → dest). Extra objects under the
    prefix are ignored — the dual-write parity gate only requires the branch tip
    to be present in the bucket.

    Args:
        src: Local directory to treat as the source of truth for the check.
        remote: rclone target (R2 or a local directory in tests).
        prefix: Object prefix under the bucket (e.g. ``raw``).

    Raises:
        RemoteError: rclone reported missing or different files, or failed.
    """
    dest = f"{remote.target.rstrip('/')}/{prefix.strip('/')}"
    files = sorted(_file_set(src))
    nbytes = _total_size(files)
    start = time.perf_counter()
    with timed_phase("remote/check"):
        _rclone(
            ["check", "--one-way", "--fast-list", str(src), dest],
            env=remote.env,
        )
    elapsed = time.perf_counter() - start
    _emit_remote_timing(
        "check",
        remote=remote,
        prefix=prefix,
        elapsed=elapsed,
        n_objects=len(files),
        nbytes=nbytes,
    )


def _file_set(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {path for path in root.rglob("*") if path.is_file()}


def _total_size(paths: list[Path]) -> int:
    """Sum on-disk sizes; missing paths contribute 0."""
    total = 0
    for path in paths:
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def _timing_data_dir() -> Path | None:
    """Prefer the active timing ContextVar; else ``Settings.data_dir``.

    Under pytest, skip the implicit ``./data`` default unless the test set
    ``FPLT_DATA_DIR`` (or installed ``timing_artifacts``), so unit suites do
    not litter the checkout.
    """
    active = active_timing_dir()
    if active is not None:
        return active
    if os.environ.get("PYTEST_CURRENT_TEST") and "FPLT_DATA_DIR" not in os.environ:
        return None
    return get_settings().data_dir


def _emit_remote_timing(
    op: _RemoteOp,
    *,
    remote: Remote,
    prefix: str,
    elapsed: float,
    n_objects: int,
    nbytes: int,
) -> None:
    """Log and persist a ``kind=remote`` timing artifact for one rclone call.

    ``n_objects`` / ``nbytes`` are newly written files for pull, and the local
    source tree for push/check (rclone may skip identical remote objects).
    """
    # Floor so near-instant mocks still produce a positive phase for reports.
    seconds = elapsed if elapsed > 0 else 1e-6
    phase = f"remote/{op}"
    meta = {
        "op": op,
        "prefix": prefix.strip("/"),
        "bucket_kind": remote.kind,
        "target": remote.target,
        "n_objects": n_objects,
        "bytes": nbytes,
    }
    logger.info(
        "remote %s prefix=%s kind=%s objects=%d bytes=%d elapsed=%.3fs",
        op,
        meta["prefix"],
        remote.kind,
        n_objects,
        nbytes,
        seconds,
    )
    session = TimingSession()
    session.add(phase, seconds)
    write_timing_record("remote", session, meta=meta, data_dir=_timing_data_dir())


def _rclone(args: list[str], *, env: dict[str, str]) -> None:
    """Run ``rclone`` with ``args``; credentials stay in ``env``, never argv.

    Raises:
        RemoteError: The binary is missing or exits non-zero.
    """
    merged = {**os.environ, **env}
    try:
        subprocess.run(
            ["rclone", *args],
            check=True,
            capture_output=True,
            text=True,
            env=merged,
        )
    except FileNotFoundError as exc:
        raise RemoteError(_RCLONE_INSTALL_HINT) from exc
    except subprocess.CalledProcessError as exc:
        detail = (exc.stderr or exc.stdout or "").strip()
        raise RemoteError(detail or f"rclone {' '.join(args)} failed") from exc
