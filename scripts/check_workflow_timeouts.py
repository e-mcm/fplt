#!/usr/bin/env python3
"""Fail if any GitHub Actions job lacks ``timeout-minutes`` (ops-11).

GitHub's default job timeout is 6 hours. Every job in ``.github/workflows/``
must set an explicit ``timeout-minutes`` so a hung step cannot burn the
full quota. Sized at roughly 2–3× expected p95 wall time; ops jobs must
also exceed the sum of HiGHS ``time_limit`` budgets used in a tick.

Usage::

    uv run python scripts/check_workflow_timeouts.py
    uv run python scripts/check_workflow_timeouts.py --workflows-dir path/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_WORKFLOWS = _REPO_ROOT / ".github" / "workflows"


def iter_workflow_jobs(
    workflows_dir: Path,
) -> list[tuple[Path, str, dict[str, Any]]]:
    """Yield ``(path, job_id, job_body)`` for every job under ``workflows_dir``.

    Args:
        workflows_dir: Directory containing ``*.yml`` / ``*.yaml`` workflows.

    Returns:
        Flat list of jobs. Missing or empty ``jobs:`` yields an empty list
        for that file (still counted as scanned).
    """
    paths = sorted(workflows_dir.glob("*.yml")) + sorted(workflows_dir.glob("*.yaml"))
    out: list[tuple[Path, str, dict[str, Any]]] = []
    for path in paths:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            continue
        doc = cast(dict[str, Any], raw)
        jobs_raw = doc.get("jobs")
        if not isinstance(jobs_raw, dict):
            continue
        jobs = cast(dict[str, Any], jobs_raw)
        for job_id, body in jobs.items():
            if isinstance(body, dict):
                out.append((path, str(job_id), cast(dict[str, Any], body)))
    return out


def missing_timeouts(
    workflows_dir: Path,
) -> list[str]:
    """Return human-readable violations for jobs without ``timeout-minutes``.

    Args:
        workflows_dir: Directory of workflow YAML files.

    Returns:
        One string per missing timeout, e.g. ``ci.yml::lint``. Empty when
        every job sets a positive integer ``timeout-minutes``.
    """
    violations: list[str] = []
    for path, job_id, body in iter_workflow_jobs(workflows_dir):
        timeout = body.get("timeout-minutes")
        rel = path.name
        if timeout is None:
            violations.append(f"{rel}::{job_id} missing timeout-minutes")
            continue
        if not isinstance(timeout, int) or isinstance(timeout, bool):
            violations.append(
                f"{rel}::{job_id} timeout-minutes must be a positive int, "
                f"got {timeout!r}"
            )
            continue
        if timeout <= 0:
            violations.append(
                f"{rel}::{job_id} timeout-minutes must be > 0, got {timeout}"
            )
    return violations


def main(argv: list[str] | None = None) -> int:
    """Scan workflows and exit non-zero when any job lacks a timeout.

    Args:
        argv: CLI args (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` when all jobs have a positive ``timeout-minutes``, else ``1``.
    """
    parser = argparse.ArgumentParser(
        description=("Fail if any GitHub Actions job lacks timeout-minutes (ops-11).")
    )
    parser.add_argument(
        "--workflows-dir",
        type=Path,
        default=_DEFAULT_WORKFLOWS,
        help="Directory of workflow YAML files (default: .github/workflows)",
    )
    args = parser.parse_args(argv)
    workflows_dir = cast(Path, args.workflows_dir)
    if not workflows_dir.is_dir():
        print(f"workflows dir not found: {workflows_dir}", file=sys.stderr)
        return 1
    violations = missing_timeouts(workflows_dir)
    if violations:
        print(
            "ops-11: every workflow job must set timeout-minutes:",
            file=sys.stderr,
        )
        for line in violations:
            print(f"  {line}", file=sys.stderr)
        return 1
    n_jobs = len(iter_workflow_jobs(workflows_dir))
    print(f"ops-11: ok ({n_jobs} jobs have timeout-minutes)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
