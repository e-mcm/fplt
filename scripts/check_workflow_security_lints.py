#!/usr/bin/env python3
"""Run actionlint + zizmor on ``.github/workflows`` (#309).

Downloads pinned release binaries into ``.cache/workflow-lints/`` (gitignored)
when missing, verifies SHA-256, then:

* ``actionlint`` — syntax / expression lint (must exit 0)
* ``zizmor --min-severity medium --format json`` — fail if any non-ignored
  finding remains (zizmor itself only non-zeros on ``error``-class findings)

Usage::

    uv run python scripts/check_workflow_security_lints.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import tarfile
import tempfile
import urllib.request
from pathlib import Path
from typing import Any, cast

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_WORKFLOWS = _REPO_ROOT / ".github" / "workflows"
_CACHE = _REPO_ROOT / ".cache" / "workflow-lints"

_ACTIONLINT_VER = "1.7.12"
_ACTIONLINT_URL = (
    "https://github.com/rhysd/actionlint/releases/download/"
    f"v{_ACTIONLINT_VER}/actionlint_{_ACTIONLINT_VER}_linux_amd64.tar.gz"
)
_ACTIONLINT_SHA256 = "8aca8db96f1b94770f1b0d72b6dddcb1ebb8123cb3712530b08cc387b349a3d8"

_ZIZMOR_VER = "1.33.1"
_ZIZMOR_URL = (
    "https://github.com/zizmorcore/zizmor/releases/download/"
    f"v{_ZIZMOR_VER}/zizmor-x86_64-unknown-linux-gnu.tar.gz"
)
_ZIZMOR_SHA256 = "3e6a301c7a820a2501915f5a9de8c0486da82bdec7965f068477538a98743d43"


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _ensure_binary(
    *,
    name: str,
    url: str,
    sha256: str,
    member: str,
    cache_dir: Path,
) -> Path:
    """Download and extract ``member`` from a tar.gz if not already cached.

    Args:
        name: Cache filename for the binary.
        url: Release tarball URL.
        sha256: Expected hex digest of the tarball.
        member: Archive member path of the executable.
        cache_dir: Directory for tarball + extracted binary.

    Returns:
        Path to the executable.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    binary = cache_dir / name
    if binary.is_file() and os.access(binary, os.X_OK):
        return binary
    tarball = cache_dir / f"{name}.tar.gz"
    if not tarball.is_file() or _sha256(tarball) != sha256:
        print(f"downloading {url}", file=sys.stderr)
        with urllib.request.urlopen(url) as resp, tarball.open("wb") as out:
            out.write(resp.read())
    digest = _sha256(tarball)
    if digest != sha256:
        raise SystemExit(f"checksum mismatch for {name}: got {digest}, want {sha256}")
    with tarfile.open(tarball, "r:gz") as tar:
        try:
            source = tar.extractfile(member)
        except KeyError:
            # Some archives nest the binary; search by basename.
            source = None
            for info in tar.getmembers():
                if Path(info.name).name == member and info.isfile():
                    source = tar.extractfile(info)
                    break
        if source is None:
            raise SystemExit(f"member {member!r} not found in {tarball}")
        with source, binary.open("wb") as out:
            out.write(source.read())
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    return binary


def run_actionlint(binary: Path, workflows_dir: Path) -> int:
    """Run actionlint on ``workflows_dir``; return the process exit code.

    Args:
        binary: Path to the actionlint executable.
        workflows_dir: Directory of workflow YAML files.

    Returns:
        Exit code from actionlint (0 = clean).
    """
    print("actionlint: scanning", workflows_dir, flush=True)
    proc = subprocess.run(  # noqa: S603 - binary path resolved by caller
        [str(binary), "-color"],
        cwd=str(workflows_dir.parent.parent),
        check=False,
    )
    return int(proc.returncode)


def run_zizmor(binary: Path, workflows_dir: Path) -> int:
    """Run zizmor and fail on any non-ignored medium-or-higher finding.

    Args:
        binary: Path to the zizmor executable.
        workflows_dir: Directory of workflow YAML files.

    Returns:
        ``0`` when clean, ``1`` when findings remain.
    """
    print("zizmor: scanning", workflows_dir, flush=True)
    with tempfile.NamedTemporaryFile(mode="w+", suffix=".json", delete=False) as tmp:
        out_path = Path(tmp.name)
    try:
        proc = subprocess.run(  # noqa: S603 - binary path resolved by caller
            [
                str(binary),
                str(workflows_dir),
                "--min-severity",
                "medium",
                "--format",
                "json",
            ],
            cwd=str(_REPO_ROOT),
            check=False,
            capture_output=True,
            text=True,
        )
        out_path.write_text(proc.stdout, encoding="utf-8")
        if proc.stderr:
            # Keep progress noise on stderr for operators.
            sys.stderr.write(proc.stderr)
        try:
            parsed: object = json.loads(proc.stdout or "[]")
        except json.JSONDecodeError:
            print("zizmor: failed to parse JSON output", file=sys.stderr)
            print(proc.stdout[:2000], file=sys.stderr)
            return 1
        if not isinstance(parsed, list):
            print("zizmor: expected a JSON list of findings", file=sys.stderr)
            return 1
        active: list[dict[str, Any]] = []
        for entry in cast(list[object], parsed):
            if not isinstance(entry, dict):
                continue
            finding = cast(dict[str, Any], entry)
            if finding.get("ignored"):
                continue
            active.append(finding)
        if active:
            print(
                f"zizmor: {len(active)} finding(s) at medium+ severity:",
                file=sys.stderr,
            )
            for finding in active:
                print(f"  {json.dumps(finding, sort_keys=True)[:240]}", file=sys.stderr)
            return 1
        print("zizmor: ok (no medium+ findings)")
        return 0
    finally:
        out_path.unlink(missing_ok=True)


def main(argv: list[str] | None = None) -> int:
    """Download pinned linters if needed and scan workflows.

    Args:
        argv: CLI args (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` when actionlint and zizmor are clean, else ``1``.
    """
    parser = argparse.ArgumentParser(
        description="Run actionlint + zizmor on .github/workflows (#309)."
    )
    parser.add_argument(
        "--workflows-dir",
        type=Path,
        default=_DEFAULT_WORKFLOWS,
        help="Directory of workflow YAML files",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_CACHE,
        help="Directory for pinned linter binaries",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Use actionlint/zizmor from PATH only (no download)",
    )
    args = parser.parse_args(argv)
    workflows_dir: Path = args.workflows_dir
    if not workflows_dir.is_dir():
        print(f"workflows dir not found: {workflows_dir}", file=sys.stderr)
        return 1

    if args.skip_download:
        from fplt.utils.process import resolve_executable

        actionlint = Path(resolve_executable("actionlint"))
        zizmor = Path(resolve_executable("zizmor"))
    else:
        actionlint = _ensure_binary(
            name="actionlint",
            url=_ACTIONLINT_URL,
            sha256=_ACTIONLINT_SHA256,
            member="actionlint",
            cache_dir=args.cache_dir,
        )
        zizmor = _ensure_binary(
            name="zizmor",
            url=_ZIZMOR_URL,
            sha256=_ZIZMOR_SHA256,
            member="zizmor",
            cache_dir=args.cache_dir,
        )

    rc = run_actionlint(actionlint, workflows_dir)
    if rc != 0:
        return 1
    return run_zizmor(zizmor, workflows_dir)


if __name__ == "__main__":
    raise SystemExit(main())
