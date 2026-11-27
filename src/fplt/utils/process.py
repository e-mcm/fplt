"""Subprocess helpers with resolved executables (bandit S607)."""

from __future__ import annotations

import shutil
import subprocess
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any


def resolve_executable(name: str) -> str:
    """Resolve ``name`` to an absolute path via ``PATH``.

    Args:
        name: Executable basename (e.g. ``git``, ``rclone``).

    Returns:
        Absolute path string.

    Raises:
        FileNotFoundError: ``name`` is not on ``PATH``.
    """
    path = shutil.which(name)
    if path is None:
        raise FileNotFoundError(f"executable not found on PATH: {name}")
    return path


def run_command(
    argv: Sequence[str],
    *,
    cwd: str | Path | None = None,
    check: bool = False,
    capture_output: bool = False,
    text: bool | None = None,
    env: Mapping[str, str] | None = None,
    **kwargs: Any,
) -> subprocess.CompletedProcess[Any]:
    """Run ``argv`` after resolving ``argv[0]`` to a full path.

    Args:
        argv: Command and arguments; ``argv[0]`` may be a basename.
        cwd: Working directory.
        check: Raise on non-zero exit.
        capture_output: Capture stdout/stderr.
        text: Decode streams as text when true.
        env: Optional environment mapping.
        **kwargs: Forwarded to ``subprocess.run``.

    Returns:
        Completed process result.

    Raises:
        FileNotFoundError: ``argv[0]`` cannot be resolved.
        ValueError: ``argv`` is empty.
    """
    if not argv:
        raise ValueError("argv must be non-empty")
    resolved = [resolve_executable(str(argv[0])), *[str(a) for a in argv[1:]]]
    return subprocess.run(  # noqa: S603 - argv[0] resolved; remaining args caller-owned
        resolved,
        cwd=cwd,
        check=check,
        capture_output=capture_output,
        text=text,
        env=None if env is None else dict(env),
        **kwargs,
    )
