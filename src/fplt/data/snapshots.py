"""Copy gzipped snapshots from the orphan ``snapshots`` branch into the data dir.

The GitHub Action commits new captures onto that branch. Nothing here rewrites a
file that is already on disk.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


class SnapshotFetchError(RuntimeError):
    """The snapshots branch could not be read."""


def fetch_snapshots(
    dest: Path, remote: str = "origin", branch: str = "snapshots"
) -> list[Path]:
    """Copy new ``*.json.gz`` files from ``branch`` into ``dest``.

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
    for path in sorted(source.rglob("*.json.gz")):
        if not path.is_file():
            continue
        target = dest / path.relative_to(source)
        if target.exists():
            continue
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, target)
        copied.append(target)
    return copied
