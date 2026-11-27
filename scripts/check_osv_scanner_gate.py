#!/usr/bin/env python3
"""Assert osv-scanner fails on a known-vulnerable lockfile fixture (#309).

Downloads a pinned ``osv-scanner`` binary (same pattern as workflow security
lints), scans ``tests/fixtures/security/vulnerable-requirements.txt``, and
exits non-zero unless at least one vulnerability is reported. Used as a
regression gate so the audit tooling cannot silently stop failing.

Usage::

    uv run python scripts/check_osv_scanner_gate.py
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import stat
import subprocess
import sys
import urllib.request
from pathlib import Path
from typing import Any, cast

_REPO_ROOT = Path(__file__).resolve().parents[1]
_CACHE = _REPO_ROOT / ".cache" / "workflow-lints"
_FIXTURE_DIR = _REPO_ROOT / "tests" / "fixtures" / "security" / "vulnerable"

_OSV_VER = "2.7.1"
_OSV_URL = (
    "https://github.com/google/osv-scanner/releases/download/"
    f"v{_OSV_VER}/osv-scanner_linux_amd64"
)
_OSV_SHA256 = "fb7dab19a8bd9bbe5c3189f09d4b7ee1820c9553a23bac202648a56efd2f6ee8"


def _sha256(path: Path) -> str:
    """Return the hex SHA-256 of ``path``."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def ensure_osv_scanner(cache_dir: Path) -> Path:
    """Return a path to a local osv-scanner binary, downloading if needed.

    Args:
        cache_dir: Directory for the cached binary.

    Returns:
        Executable path.
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    binary = cache_dir / "osv-scanner"
    if binary.is_file() and os.access(binary, os.X_OK):
        return binary
    print(f"downloading {_OSV_URL}", file=sys.stderr)
    with urllib.request.urlopen(_OSV_URL) as resp, binary.open("wb") as out:
        out.write(resp.read())
    binary.chmod(binary.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP | stat.S_IXOTH)
    digest = _sha256(binary)
    if digest != _OSV_SHA256:
        raise SystemExit(f"osv-scanner checksum mismatch: {digest}")
    return binary


def scan_fixture(binary: Path, fixture_dir: Path) -> list[dict[str, Any]]:
    """Run osv-scanner on ``fixture_dir`` and return vulnerability objects.

    Args:
        binary: Path to osv-scanner.
        fixture_dir: Directory containing a known-vulnerable ``requirements.txt``.

    Returns:
        Flattened list of vulnerability dicts from the JSON report.
    """
    # Bypass the fixture-local osv-scanner.toml PackageOverrides (ignore=true)
    # so repo-wide scans stay clean while this gate still sees the vulns.
    root_config = _REPO_ROOT / "osv-scanner.toml"
    proc = subprocess.run(  # noqa: S603 - binary path resolved by caller
        [
            str(binary),
            "scan",
            "source",
            str(fixture_dir),
            "--config",
            str(root_config),
            "--format",
            "json",
            "--verbosity",
            "error",
        ],
        cwd=str(_REPO_ROOT),
        check=False,
        capture_output=True,
        text=True,
    )
    if not proc.stdout.strip():
        print(proc.stderr[:2000], file=sys.stderr)
        raise SystemExit("osv-scanner produced no JSON output")
    payload: object = json.loads(proc.stdout)
    if not isinstance(payload, dict):
        raise SystemExit("osv-scanner JSON root must be an object")
    data = cast(dict[str, Any], payload)
    vulns: list[dict[str, Any]] = []
    results_obj: object = data.get("results") or []
    if not isinstance(results_obj, list):
        return vulns
    for result_entry in cast(list[object], results_obj):
        if not isinstance(result_entry, dict):
            continue
        result = cast(dict[str, Any], result_entry)
        packages_obj: object = result.get("packages") or []
        if not isinstance(packages_obj, list):
            continue
        for pkg_entry in cast(list[object], packages_obj):
            if not isinstance(pkg_entry, dict):
                continue
            pkg = cast(dict[str, Any], pkg_entry)
            vuln_obj: object = pkg.get("vulnerabilities") or []
            if not isinstance(vuln_obj, list):
                continue
            for vuln_entry in cast(list[object], vuln_obj):
                if isinstance(vuln_entry, dict):
                    vulns.append(cast(dict[str, Any], vuln_entry))
    return vulns


def main(argv: list[str] | None = None) -> int:
    """Scan the vulnerable fixture and fail if no vulns are reported.

    Args:
        argv: CLI args (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` when the fixture yields ≥1 vulnerability (gate works),
        ``1`` when the scanner is silent (broken gate).
    """
    parser = argparse.ArgumentParser(
        description="Assert osv-scanner detects the known-vulnerable fixture (#309)."
    )
    parser.add_argument(
        "--fixture-dir",
        type=Path,
        default=_FIXTURE_DIR,
        help="Directory with vulnerable requirements.txt fixture",
    )
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_CACHE,
        help="Directory for the osv-scanner binary",
    )
    args = parser.parse_args(argv)
    fixture_dir: Path = args.fixture_dir
    if not (fixture_dir / "requirements.txt").is_file():
        print(f"fixture not found: {fixture_dir}/requirements.txt", file=sys.stderr)
        return 1
    binary = ensure_osv_scanner(args.cache_dir)
    vulns = scan_fixture(binary, fixture_dir)
    if not vulns:
        print(
            "osv-scanner gate broken: fixture reported zero vulnerabilities",
            file=sys.stderr,
        )
        return 1
    print(f"osv-scanner gate ok ({len(vulns)} vulns on fixture)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
