#!/usr/bin/env python3
"""Fail if CI cancels main runs or scheduled workflows lack concurrency (#305).

``ci.yml`` must never cancel superseded ``main`` (or other non-PR) runs —
only pull_request runs may use ``cancel-in-progress``. Every workflow with a
``schedule:`` trigger must declare a ``concurrency.group`` so cron + dispatch
cannot overlap and race.

Usage::

    uv run python scripts/check_workflow_concurrency.py
    uv run python scripts/check_workflow_concurrency.py --workflows-dir path/
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path
from typing import Any, cast

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_WORKFLOWS = _REPO_ROOT / ".github" / "workflows"
_CI_NAME = "ci.yml"
# Expression that cancels only when the event is a pull_request.
_PR_ONLY_CANCEL = re.compile(
    r"github\.event_name\s*==\s*['\"]pull_request['\"]",
    re.IGNORECASE,
)


def load_workflow(path: Path) -> dict[str, Any] | None:
    """Parse a workflow YAML file into a mapping, or ``None`` if unusable.

    Args:
        path: Path to a ``*.yml`` / ``*.yaml`` workflow.

    Returns:
        The document mapping, or ``None`` when the file is not a mapping.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return None
    return cast(dict[str, Any], raw)


def workflow_on(doc: dict[str, Any]) -> dict[str, Any]:
    """Return the workflow ``on:`` mapping (PyYAML may key it as ``True``).

    Args:
        doc: Parsed workflow document.

    Returns:
        The trigger mapping, or an empty dict when missing / not a mapping.
    """
    # PyYAML 1.1 may parse the key ``on`` as boolean ``True``.
    alt = cast(dict[object, Any], doc).get(True)
    value: Any = doc.get("on", alt)
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    return {}


def has_schedule(doc: dict[str, Any]) -> bool:
    """Return whether the workflow declares a ``schedule`` trigger.

    Args:
        doc: Parsed workflow document.

    Returns:
        True when ``on.schedule`` is present.
    """
    return "schedule" in workflow_on(doc)


def ci_cancels_main_runs(doc: dict[str, Any]) -> str | None:
    """Return a violation when ``ci.yml`` can cancel non-PR (e.g. main) runs.

    Args:
        doc: Parsed ``ci.yml`` document.

    Returns:
        A human-readable violation, or ``None`` when main runs are safe.
    """
    conc_raw = doc.get("concurrency")
    if not isinstance(conc_raw, dict):
        return "ci.yml missing concurrency: (main runs can pile up / cancel)"
    conc = cast(dict[str, Any], conc_raw)
    cancel: Any = conc.get("cancel-in-progress")
    if cancel is True:
        return (
            "ci.yml concurrency.cancel-in-progress is true (cancels main / non-PR runs)"
        )
    if cancel is False or cancel is None:
        return None
    if isinstance(cancel, str):
        if _PR_ONLY_CANCEL.search(cancel):
            return None
        return (
            f"ci.yml concurrency.cancel-in-progress={cancel!r} "
            "must only cancel pull_request runs"
        )
    return (
        "ci.yml concurrency.cancel-in-progress has unexpected type "
        f"{type(cancel).__name__}"
    )


def scheduled_concurrency_violations(workflows_dir: Path) -> list[str]:
    """Return violations for scheduled workflows without a concurrency group.

    Also flags ``cancel-in-progress: true`` on scheduled workflows — overlapping
    cron + dispatch must queue, not cancel.

    Args:
        workflows_dir: Directory of workflow YAML files.

    Returns:
        One string per violation (filename-prefixed).
    """
    paths = sorted(workflows_dir.glob("*.yml")) + sorted(workflows_dir.glob("*.yaml"))
    violations: list[str] = []
    for path in paths:
        doc = load_workflow(path)
        if doc is None or not has_schedule(doc):
            continue
        conc_raw = doc.get("concurrency")
        if not isinstance(conc_raw, dict):
            violations.append(
                f"{path.name}: scheduled workflow missing concurrency.group"
            )
            continue
        conc = cast(dict[str, Any], conc_raw)
        if not conc.get("group"):
            violations.append(
                f"{path.name}: scheduled workflow missing concurrency.group"
            )
            continue
        if conc.get("cancel-in-progress") is True:
            violations.append(
                f"{path.name}: scheduled workflow must not set cancel-in-progress: true"
            )
    return violations


def concurrency_violations(workflows_dir: Path) -> list[str]:
    """Return all #305 concurrency lint violations under ``workflows_dir``.

    Args:
        workflows_dir: Directory of workflow YAML files.

    Returns:
        Combined list of ci.yml + scheduled-workflow violations.
    """
    violations: list[str] = []
    ci_path = workflows_dir / _CI_NAME
    if ci_path.is_file():
        doc = load_workflow(ci_path)
        if doc is None:
            violations.append("ci.yml is not a YAML mapping")
        else:
            msg = ci_cancels_main_runs(doc)
            if msg is not None:
                violations.append(msg)
    violations.extend(scheduled_concurrency_violations(workflows_dir))
    return violations


def main(argv: list[str] | None = None) -> int:
    """Scan workflows and exit non-zero on concurrency policy violations.

    Args:
        argv: CLI args (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` when policy holds, else ``1``.
    """
    parser = argparse.ArgumentParser(
        description=(
            "Fail if ci.yml cancels main runs or a scheduled workflow "
            "lacks concurrency (#305)."
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
    violations = concurrency_violations(workflows_dir)
    if violations:
        print("concurrency (#305): policy violations:", file=sys.stderr)
        for line in violations:
            print(f"  {line}", file=sys.stderr)
        return 1
    print("concurrency (#305): ok")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
