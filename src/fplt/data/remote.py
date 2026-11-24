"""rclone transfers against the fplt R2 data and lab buckets.

Credentials travel only in the process environment (``RCLONE_CONFIG_FPLT_*``),
never on the argv. Library and feature code still read a local
``$FPLT_DATA_DIR``; this module is the edge that pulls and pushes.
"""

from __future__ import annotations

import os
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from fplt.settings import Settings

_RCLONE_INSTALL_HINT = (
    "rclone is not installed; install with "
    "`brew install rclone` or `sudo apt-get install -y rclone`"
)

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
    after = _file_set(dest)
    return sorted(after - before)


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
    _rclone(
        ["copy", str(src), dest, "--immutable", "--fast-list"],
        env=remote.env,
    )


def _file_set(root: Path) -> set[Path]:
    if not root.exists():
        return set()
    return {path for path in root.rglob("*") if path.is_file()}


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
