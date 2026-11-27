#!/usr/bin/env python3
"""Fail if a GHA job reads ``secrets.R2_`` without ``environment: r2`` (#278).

R2 credentials are GitHub **environment** secrets on ``r2``. A job that
references ``secrets.R2_*`` without ``environment: r2`` always sees them as
empty and historically soft-skipped uploads. This lint keeps that footgun
out of new workflows.

Usage::

    uv run python scripts/check_workflow_r2_environment.py
    uv run python scripts/check_workflow_r2_environment.py --workflows-dir path/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, cast

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_WORKFLOWS = _REPO_ROOT / ".github" / "workflows"
_R2_SECRET_MARKER = "secrets.R2_"
_REQUIRED_ENVIRONMENT = "r2"


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


def job_environment_name(body: dict[str, Any]) -> str | None:
    """Return the job's ``environment`` name, if any.

    Args:
        body: Parsed job mapping from a workflow YAML file.

    Returns:
        The environment name string, or ``None`` when unset / unparseable.
    """
    env = body.get("environment")
    if isinstance(env, str):
        return env
    if isinstance(env, dict):
        mapping = cast(dict[str, object], env)
        raw_name = mapping.get("name")
        if raw_name is None:
            return None
        return str(raw_name)
    return None


def job_references_r2_secrets(body: dict[str, Any]) -> bool:
    """Return True when the job YAML mentions ``secrets.R2_``.

    Args:
        body: Parsed job mapping from a workflow YAML file.

    Returns:
        Whether any step / env / with value references an ``R2_`` secret.
    """
    dumped = yaml.safe_dump(body, default_flow_style=False, sort_keys=False)
    return _R2_SECRET_MARKER in dumped


def jobs_missing_r2_environment(workflows_dir: Path) -> list[str]:
    """Return violations for jobs that read R2 secrets without ``environment: r2``.

    Args:
        workflows_dir: Directory of workflow YAML files.

    Returns:
        One string per violation, e.g.
        ``snapshot.yml::capture references secrets.R2_ without environment: r2``.
    """
    violations: list[str] = []
    for path, job_id, body in iter_workflow_jobs(workflows_dir):
        if not job_references_r2_secrets(body):
            continue
        if job_environment_name(body) == _REQUIRED_ENVIRONMENT:
            continue
        violations.append(
            f"{path.name}::{job_id} references secrets.R2_ "
            f"without environment: {_REQUIRED_ENVIRONMENT}"
        )
    return violations


def main(argv: list[str] | None = None) -> int:
    """Scan workflows and exit non-zero when R2 jobs lack ``environment: r2``.

    Args:
        argv: CLI args (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` when every R2-secret job sets ``environment: r2``, else ``1``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fail if any GHA job references secrets.R2_ without environment: r2."
        )
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
    violations = jobs_missing_r2_environment(workflows_dir)
    if violations:
        print(
            "r2-env: every job that reads secrets.R2_ must set environment: r2:",
            file=sys.stderr,
        )
        for line in violations:
            print(f"  {line}", file=sys.stderr)
        return 1
    n_jobs = sum(
        1
        for _, _, body in iter_workflow_jobs(workflows_dir)
        if job_references_r2_secrets(body)
    )
    print(f"r2-env: ok ({n_jobs} jobs with secrets.R2_ use environment: r2)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
