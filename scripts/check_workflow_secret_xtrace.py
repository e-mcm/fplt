#!/usr/bin/env python3
"""Fail if secret-handling GHA steps enable shell xtrace (``set -x``).

Public Actions logs must never echo derived secret values. ``set -x`` /
``set -o xtrace`` prints every command with expansions, which defeats
GitHub's secret masking for derived forms. Steps that reference
``secrets.*`` or known credential env vars must not enable xtrace.

Usage::

    uv run python scripts/check_workflow_secret_xtrace.py
    uv run python scripts/check_workflow_secret_xtrace.py --workflows-dir path/
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, cast

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]

# ``set -euxo``, ``set -x``, ``set -e -x``, or ``set -o xtrace``.
_XTRACE_RE = re.compile(
    r"(?:^|\n)\s*set\s+(?:(?:-[a-zA-Z]*\s+)*-[a-zA-Z]*x[a-zA-Z]*\b|-o\s+xtrace\b)",
    re.IGNORECASE,
)
_SECRETS_EXPR_RE = re.compile(r"\$\{\{\s*secrets\.")
_SECRET_ENV_RE = re.compile(
    r"\b(?:"
    r"R2_SECRET_ACCESS_KEY|R2_ACCESS_KEY_ID|R2_ACCOUNT_ID|"
    r"FPLT_R2_SECRET_ACCESS_KEY|FPLT_R2_ACCESS_KEY_ID|FPLT_R2_ACCOUNT_ID|"
    r"FPLT_FPL_API_TOKEN|FPLT_API_FOOTBALL_KEY|API_FOOTBALL_KEY|APISPORTS_KEY|"
    r"FPLT_NTFY_URL|NTFY_URL|FPLT_ENTRY_ID|"
    r"HC_PING_URL_[A-Z0-9_]+|"
    r"GH_TOKEN|GITHUB_TOKEN"
    r")\b"
)


def _step_handles_secrets(step: dict[str, Any]) -> bool:
    """Return True when the step references secrets or credential env names."""
    chunks: list[str] = []
    run = step.get("run")
    if isinstance(run, str):
        chunks.append(run)
    for key in ("env", "with"):
        block = step.get(key)
        if isinstance(block, dict):
            for env_key, value in cast(dict[str, Any], block).items():
                chunks.append(str(env_key))
                if isinstance(value, str):
                    chunks.append(value)
                elif value is not None:
                    chunks.append(str(value))
    text = "\n".join(chunks)
    return bool(_SECRETS_EXPR_RE.search(text) or _SECRET_ENV_RE.search(text))


def _step_enables_xtrace(step: dict[str, Any]) -> bool:
    """Return True when the step's ``run`` script enables shell xtrace."""
    run = step.get("run")
    if not isinstance(run, str):
        return False
    return _XTRACE_RE.search(run) is not None


def _run_steps_from_jobs(
    path: Path, jobs: dict[str, Any]
) -> list[tuple[str, int, dict[str, Any]]]:
    out: list[tuple[str, int, dict[str, Any]]] = []
    for job_id, body in jobs.items():
        if not isinstance(body, dict):
            continue
        job = cast(dict[str, Any], body)
        steps_raw = job.get("steps")
        if not isinstance(steps_raw, list):
            continue
        steps = cast(list[Any], steps_raw)
        for idx, step in enumerate(steps):
            if isinstance(step, dict) and "run" in step:
                out.append((f"{path.name}::{job_id}", idx, cast(dict[str, Any], step)))
    return out


def iter_workflow_run_steps(
    workflows_dir: Path,
) -> list[tuple[str, int, dict[str, Any]]]:
    """Yield ``(location, step_index, step)`` for every ``run:`` step.

    Args:
        workflows_dir: Directory containing ``*.yml`` / ``*.yaml`` workflows.

    Returns:
        Flat list of run steps.
    """
    out: list[tuple[str, int, dict[str, Any]]] = []
    paths = sorted(workflows_dir.glob("*.yml")) + sorted(workflows_dir.glob("*.yaml"))
    for path in paths:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            continue
        doc = cast(dict[str, Any], raw)
        jobs_raw = doc.get("jobs")
        if isinstance(jobs_raw, dict):
            out.extend(_run_steps_from_jobs(path, cast(dict[str, Any], jobs_raw)))
    return out


def iter_action_run_steps(
    actions_dir: Path,
) -> list[tuple[str, int, dict[str, Any]]]:
    """Yield run steps from composite actions under ``actions_dir``."""
    out: list[tuple[str, int, dict[str, Any]]] = []
    if not actions_dir.is_dir():
        return out
    for action_yml in sorted(actions_dir.glob("*/action.yml")) + sorted(
        actions_dir.glob("*/action.yaml")
    ):
        raw = yaml.safe_load(action_yml.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            continue
        doc = cast(dict[str, Any], raw)
        runs_raw = doc.get("runs")
        if not isinstance(runs_raw, dict):
            continue
        runs = cast(dict[str, Any], runs_raw)
        steps_raw = runs.get("steps")
        if not isinstance(steps_raw, list):
            continue
        steps = cast(list[Any], steps_raw)
        loc = f"{action_yml.parent.name}/{action_yml.name}"
        for idx, step in enumerate(steps):
            if isinstance(step, dict) and "run" in step:
                out.append((loc, idx, cast(dict[str, Any], step)))
    return out


def xtrace_secret_violations(
    *,
    workflows_dir: Path,
    actions_dir: Path | None = None,
) -> list[str]:
    """Return violations for secret-handling steps that enable xtrace.

    Args:
        workflows_dir: ``.github/workflows`` (or a test fixture dir).
        actions_dir: Optional ``.github/actions`` directory.

    Returns:
        One string per violation. Empty when clean.
    """
    steps = list(iter_workflow_run_steps(workflows_dir))
    if actions_dir is not None:
        steps.extend(iter_action_run_steps(actions_dir))
    violations: list[str] = []
    for loc, idx, step in steps:
        if not _step_handles_secrets(step):
            continue
        if not _step_enables_xtrace(step):
            continue
        name = step.get("name")
        label = f" step={name!r}" if isinstance(name, str) else f" step[{idx}]"
        violations.append(f"{loc}{label} enables set -x while handling secrets")
    return violations


def main(argv: list[str] | None = None) -> int:
    """CLI entry: exit 1 when any secret-handling step enables xtrace."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workflows-dir",
        type=Path,
        default=_REPO_ROOT / ".github" / "workflows",
        help="Workflows directory to scan",
    )
    parser.add_argument(
        "--actions-dir",
        type=Path,
        default=_REPO_ROOT / ".github" / "actions",
        help="Composite actions directory (ignored when missing)",
    )
    parser.add_argument(
        "--skip-actions",
        action="store_true",
        help="Do not scan composite actions (tests)",
    )
    args = parser.parse_args(argv)
    actions = None if args.skip_actions else args.actions_dir
    violations = xtrace_secret_violations(
        workflows_dir=args.workflows_dir,
        actions_dir=actions,
    )
    if violations:
        print("secret-handling steps must not enable set -x / xtrace:", file=sys.stderr)
        for line in violations:
            print(f"  {line}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
