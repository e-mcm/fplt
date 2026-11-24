"""Copy gzipped snapshots into the local data dir (bucket or git branch).

When ``Settings.data_bucket`` is set, ``fetch_raw`` pulls from R2 via
``fplt.data.remote``. Otherwise it clones the orphan ``snapshots`` branch.
Nothing here rewrites a file that is already on disk.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Literal

from fplt.data.remote import Remote, pull
from fplt.settings import Settings, get_settings

RawSource = Literal["auto", "bucket", "branch"]


class SnapshotFetchError(RuntimeError):
    """The snapshots branch could not be read."""


def fetch_raw(
    dest: Path,
    settings: Settings | None = None,
    *,
    source: RawSource = "auto",
    remote: str = "origin",
    branch: str = "snapshots",
) -> list[Path]:
    """Copy new raw captures into ``dest`` from the bucket or the snapshots branch.

    ``source="auto"`` (default) uses the R2 data bucket when
    ``settings.data_bucket`` is set, and the git ``snapshots`` branch otherwise.
    ``source="bucket"`` requires a configured data bucket; ``source="branch"``
    always uses git.

    Args:
        dest: Directory that corresponds to ``$FPLT_DATA_DIR/raw``.
        settings: Process settings (R2 credentials / bucket). Defaults to
            ``get_settings()``.
        source: ``auto``, ``bucket``, or ``branch``.
        remote: Git remote, or a local path in tests (branch path only).
        branch: Branch that holds write-once snapshots (branch path only).

    Returns:
        Paths that were newly written. Files already present are left untouched.

    Raises:
        SnapshotFetchError: The branch could not be cloned, or ``source`` is
            ``bucket`` but no data bucket is configured.
        RemoteError: An rclone pull against the bucket failed.
        ValueError: ``source`` is not one of ``auto`` / ``bucket`` / ``branch``.
    """
    cfg = get_settings() if settings is None else settings
    if source not in ("auto", "bucket", "branch"):
        raise ValueError(
            f"source must be 'auto', 'bucket', or 'branch'; got {source!r}"
        )
    use_bucket = source == "bucket" or (source == "auto" and bool(cfg.data_bucket))
    if source == "bucket" and not cfg.data_bucket:
        raise SnapshotFetchError(
            "source=bucket requires FPLT_DATA_BUCKET (settings.data_bucket) to be set"
        )
    if use_bucket:
        r2 = Remote.from_settings(cfg)
        if r2 is None:
            raise SnapshotFetchError(
                "data bucket is configured but Remote.from_settings returned None"
            )
        return pull(r2, "raw", dest)
    return fetch_snapshots(dest, remote=remote, branch=branch)


def fetch_snapshots(
    dest: Path, remote: str = "origin", branch: str = "snapshots"
) -> list[Path]:
    """Copy new snapshot files from ``branch`` into ``dest``.

    Copies write-once ``*.json.gz`` captures and football-data ``E0.csv``
    files. Existing destinations are left untouched.

    Args:
        dest: Directory that corresponds to ``$FPLT_DATA_DIR/raw``.
        remote: Git remote, or a local path in tests.
        branch: Branch that holds the write-once snapshots.

    Returns:
        Paths that were newly written. Files already present are left untouched.

    Raises:
        SnapshotFetchError: The branch could not be cloned.
    """
    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / "repo"
        try:
            _git(
                [
                    "clone",
                    "--branch",
                    branch,
                    "--single-branch",
                    "--depth",
                    "1",
                    remote,
                    str(clone),
                ]
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise SnapshotFetchError(detail or f"could not fetch {branch}") from exc
        source = clone / "data" / "raw"
        if not source.exists():
            return []
        return _copy_new(source, dest)


def is_football_data_raw(path: Path) -> bool:
    """Return whether ``path`` sits under a ``football_data`` raw prefix."""
    return "football_data" in path.parts


def _git(args: list[str]) -> None:
    subprocess.run(
        ["git", *args],
        check=True,
        capture_output=True,
        text=True,
        env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
    )


def _copy_new(source: Path, dest: Path) -> list[Path]:
    copied: list[Path] = []
    candidates = list(source.rglob("*.json.gz"))
    football = source / "football_data"
    if football.exists():
        candidates.extend(football.rglob("E0.csv"))
    for path in sorted(candidates):
        if not path.is_file():
            continue
        target = dest / path.relative_to(source)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        copied.append(target)
    return copied


__all__ = [
    "RawSource",
    "SnapshotFetchError",
    "fetch_raw",
    "fetch_snapshots",
    "is_football_data_raw",
]
