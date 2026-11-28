"""Refuse raw trees whose relative path components start with ``.``.

Used by ``push-raw`` and snapshots-branch publish so a working cache such as
``.vaastav_cache`` can never reach write-once ``raw/`` (#339).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def find_dot_component_paths(root: Path) -> list[str]:
    """Return relative posix paths under ``root`` with a ``.``-prefixed part.

    Args:
        root: Directory to walk (files and directories).

    Returns:
        Sorted relative paths (posix) that contain a component starting with
        ``.``. Empty when the tree is clean or ``root`` is missing.
    """
    if not root.is_dir():
        return []
    root = root.resolve()
    bad: list[str] = []
    for path in sorted(root.rglob("*")):
        rel = path.relative_to(root)
        if any(part.startswith(".") for part in rel.parts):
            bad.append(rel.as_posix())
    return bad


def reject_raw_dot_paths(root: Path) -> list[str]:
    """Return offending relative paths (empty means ok to upload/publish).

    Args:
        root: Local raw capture directory about to be pushed or copied.

    Returns:
        Paths that would violate the no-dot-component rule.
    """
    return find_dot_component_paths(root)


def main(argv: list[str] | None = None) -> int:
    """CLI: exit non-zero when ``root`` contains ``.``-prefixed components."""
    parser = argparse.ArgumentParser(
        prog="reject_raw_dot_paths",
        description="Fail if a raw tree contains path components starting with '.'",
    )
    parser.add_argument("root", type=Path, help="Local raw capture directory.")
    args = parser.parse_args(argv)
    bad = reject_raw_dot_paths(args.root)
    if not bad:
        return 0
    sample = bad[0]
    print(
        f"::error::raw source rejects path component starting with '.': {sample}",
        file=sys.stderr,
    )
    if len(bad) > 1:
        print(
            f"::error::raw source has {len(bad)} paths with '.'-prefixed components",
            file=sys.stderr,
        )
    return 1


if __name__ == "__main__":
    sys.exit(main())
