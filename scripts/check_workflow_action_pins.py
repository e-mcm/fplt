#!/usr/bin/env python3
"""Fail if any third-party GitHub Action is not SHA-pinned (#309).

Mutable tags (``@v4``, ``@v1.2.3``) can move. Every ``uses:`` of a
third-party action under ``.github/workflows/`` and
``.github/actions/`` must pin a 40-character commit SHA, with a version
comment for humans (``# v4.4.0``). Local composite actions
(``./.github/actions/...``) are exempt.

Usage::

    uv run python scripts/check_workflow_action_pins.py
    uv run python scripts/check_workflow_action_pins.py --root path/
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_ROOT = _REPO_ROOT / ".github"

# List or mapping step: ``- uses: owner/name@<40-hex>`` (+ optional ``#``).
_USES_RE = re.compile(r"^\s*-?\s*uses:\s*(?P<ref>\S+)(?P<trail>\s*(?:#.*)?)?\s*$")
_SHA_RE = re.compile(r"^[0-9a-f]{40}$")
_VERSION_COMMENT_RE = re.compile(r"#\s*v?\d")


def iter_action_yamls(root: Path) -> list[Path]:
    """Return workflow and composite-action YAML paths under ``root``.

    Args:
        root: Typically ``.github`` (contains ``workflows/`` and ``actions/``).

    Returns:
        Sorted list of ``*.yml`` / ``*.yaml`` files.
    """
    paths: list[Path] = []
    for sub in ("workflows", "actions"):
        d = root / sub
        if not d.is_dir():
            continue
        paths.extend(sorted(d.rglob("*.yml")))
        paths.extend(sorted(d.rglob("*.yaml")))
    return paths


def unpinned_actions(root: Path) -> list[str]:
    """Return human-readable violations for unpinned third-party actions.

    Args:
        root: Directory containing ``workflows/`` and/or ``actions/``.

    Returns:
        One string per violation. Empty when every third-party ``uses:``
        is SHA-pinned with a version comment.
    """
    violations: list[str] = []
    for path in iter_action_yamls(root):
        rel = path.relative_to(root) if path.is_relative_to(root) else path
        for line_no, line in enumerate(
            path.read_text(encoding="utf-8").splitlines(), 1
        ):
            match = _USES_RE.match(line)
            if match is None:
                continue
            ref = match.group("ref")
            trail = match.group("trail") or ""
            if ref.startswith("./") or ref.startswith(".\\"):
                continue
            if "@" not in ref:
                violations.append(f"{rel}:{line_no}: missing @ref in {ref!r}")
                continue
            _action, at = ref.rsplit("@", 1)
            if not _SHA_RE.fullmatch(at):
                violations.append(
                    f"{rel}:{line_no}: unpinned action {ref!r} "
                    "(need 40-char commit SHA)"
                )
                continue
            if not _VERSION_COMMENT_RE.search(trail):
                violations.append(
                    f"{rel}:{line_no}: SHA-pinned {ref!r} missing version "
                    "comment (e.g. # v4.4.0)"
                )
    return violations


def main(argv: list[str] | None = None) -> int:
    """Scan workflows/composite actions and exit non-zero on unpinned uses.

    Args:
        argv: CLI args (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` when every third-party action is SHA-pinned with a version
        comment, else ``1``.
    """
    parser = argparse.ArgumentParser(
        description=("Fail if any third-party GitHub Action is not SHA-pinned (#309).")
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=_DEFAULT_ROOT,
        help="Directory with workflows/ and actions/ (default: .github)",
    )
    args = parser.parse_args(argv)
    root = args.root
    if not root.is_dir():
        print(f"root dir not found: {root}", file=sys.stderr)
        return 1
    violations = unpinned_actions(root)
    if violations:
        print(
            "#309: every third-party action must be SHA-pinned with a version comment:",
            file=sys.stderr,
        )
        for line in violations:
            print(f"  {line}", file=sys.stderr)
        return 1
    n_files = len(iter_action_yamls(root))
    print(f"#309: ok (action pins clean across {n_files} YAML files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
