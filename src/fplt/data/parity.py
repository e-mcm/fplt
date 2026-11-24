"""Branch tip vs R2 raw parity during dual-write.

Clones the orphan ``snapshots`` branch tip and runs a one-way
``rclone check`` against ``fplt-data/raw``. Used by ``fplt data parity`` and
the weekly GHA job while both sinks are live (rollout A6 / bucket-plan PR 5).
"""

from __future__ import annotations

import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path

from fplt.data.remote import Remote, RemoteError, check_one_way
from fplt.settings import Settings, get_settings

logger = logging.getLogger(__name__)


class ParityError(RuntimeError):
    """The snapshots branch tip and the data bucket disagree, or the check failed."""


@dataclass(frozen=True)
class ParityResult:
    """Outcome of a successful one-way branch → bucket raw check.

    Attributes:
        source_files: Number of files under the branch tip's ``data/raw``.
        remote_target: rclone target that was checked (bucket or local path).
        branch: Git branch that was cloned.
    """

    source_files: int
    remote_target: str
    branch: str


def check_raw_parity(
    settings: Settings | None = None,
    *,
    remote: str = "origin",
    branch: str = "snapshots",
    prefix: str = "raw",
) -> ParityResult:
    """Clone ``branch`` tip and one-way-check its ``data/raw`` against the bucket.

    Args:
        settings: Process settings (R2 credentials / data bucket). Defaults to
            ``get_settings()``.
        remote: Git remote, or a local path in tests.
        branch: Branch that holds write-once captures (default ``snapshots``).
        prefix: Bucket prefix to compare (default ``raw``).

    Returns:
        A ``ParityResult`` when every branch file is present in the bucket.

    Raises:
        ParityError: No data bucket is configured, the branch could not be
            cloned, or ``rclone check`` found missing/different keys.
    """
    cfg = get_settings() if settings is None else settings
    r2 = Remote.from_settings(cfg)
    if r2 is None:
        raise ParityError(
            "data bucket is not configured; set FPLT_DATA_BUCKET to run parity"
        )
    with tempfile.TemporaryDirectory() as tmp:
        clone = Path(tmp) / "repo"
        try:
            subprocess.run(
                [
                    "git",
                    "clone",
                    "--branch",
                    branch,
                    "--single-branch",
                    "--depth",
                    "1",
                    remote,
                    str(clone),
                ],
                check=True,
                capture_output=True,
                text=True,
                env={**os.environ, "GIT_TERMINAL_PROMPT": "0"},
            )
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "").strip()
            raise ParityError(detail or f"could not fetch {branch}") from exc
        source = clone / "data" / "raw"
        if not source.exists():
            source.mkdir(parents=True, exist_ok=True)
        source_files = sum(1 for path in source.rglob("*") if path.is_file())
        logger.info(
            "raw parity: checking %d files from %s → %s/%s",
            source_files,
            branch,
            r2.target,
            prefix,
        )
        try:
            check_one_way(source, r2, prefix)
        except RemoteError as exc:
            raise ParityError(str(exc)) from exc
    return ParityResult(
        source_files=source_files,
        remote_target=r2.target,
        branch=branch,
    )


__all__ = [
    "ParityError",
    "ParityResult",
    "check_raw_parity",
]
