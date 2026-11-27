#!/usr/bin/env python3
"""Rank pytest case durations from a JUnit XML report (CI profiling).

``fplt.utils.timing`` / ``fplt perf-report`` cover walk-forward and eval
phases, not the pytest suite. This script fills that gap: run (or parse)
a CI-shaped pytest invocation and print the slowest cases so speedups
can target real offenders without skipping tests or cutting coverage.

Default invocation mirrors ``.github/workflows/ci.yml`` ``test-linux``::

    uv run pytest -n auto --cov=fplt --cov-branch --cov-report=xml

Usage::

    make pytest-profile
    uv run python scripts/pytest_profile.py --top 25
    uv run python scripts/pytest_profile.py --json -o /tmp/pytest-durations.json
    uv run python scripts/pytest_profile.py --parse-only /tmp/pytest-junit.xml
    uv run python scripts/pytest_profile.py --pytest-args '-m slow --no-cov'
"""

from __future__ import annotations

import argparse
import json
import sys
import xml.etree.ElementTree as ET
from dataclasses import asdict, dataclass
from pathlib import Path

from fplt.utils.process import run_command

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_JUNIT = _REPO_ROOT / ".pytest_cache" / "pytest-profile-junit.xml"
_CI_PYTEST_ARGS = ("-n", "auto", "--cov=fplt", "--cov-branch", "--cov-report=xml")


@dataclass(frozen=True, slots=True)
class CaseDuration:
    """One pytest case wall time from JUnit XML."""

    nodeid: str
    seconds: float
    classname: str
    name: str
    file: str | None = None

    @property
    def suite(self) -> str:
        """Best-effort suite label (file path or classname)."""
        if self.file:
            return self.file
        return self.classname


def _nodeid(classname: str, name: str, file_attr: str | None) -> str:
    """Build a pytest-like node id from JUnit attributes."""
    if file_attr:
        # pytest junit ``file`` is relative (e.g. tests/unit/foo.py).
        path = file_attr.replace("\\", "/")
        if "::" in name:
            return f"{path}::{name}"
        return f"{path}::{name}"
    if classname:
        return f"{classname}::{name}"
    return name


def parse_junit_durations(junit_path: Path) -> list[CaseDuration]:
    """Parse per-case durations from a pytest ``--junitxml`` report.

    Args:
        junit_path: Path to a JUnit XML file written by pytest.

    Returns:
        Case durations sorted descending by ``seconds``.

    Raises:
        FileNotFoundError: If ``junit_path`` does not exist.
        ET.ParseError: If the XML is malformed.
        ValueError: If the root element is not a ``testsuites`` / ``testsuite``.
    """
    if not junit_path.is_file():
        raise FileNotFoundError(f"JUnit report not found: {junit_path}")
    tree = ET.parse(junit_path)
    root = tree.getroot()
    if root.tag not in {"testsuites", "testsuite"}:
        raise ValueError(
            f"Expected testsuites/testsuite root, got {root.tag!r} in {junit_path}"
        )
    cases: list[CaseDuration] = []
    for case in root.iter("testcase"):
        name = case.attrib.get("name", "")
        classname = case.attrib.get("classname", "")
        file_attr = case.attrib.get("file")
        raw_time = case.attrib.get("time", "0")
        try:
            seconds = float(raw_time)
        except ValueError:
            seconds = 0.0
        cases.append(
            CaseDuration(
                nodeid=_nodeid(classname, name, file_attr),
                seconds=seconds,
                classname=classname,
                name=name,
                file=file_attr,
            )
        )
    cases.sort(key=lambda c: c.seconds, reverse=True)
    return cases


def aggregate_by_file(cases: list[CaseDuration]) -> list[tuple[str, float, int]]:
    """Sum durations by source file (or classname when file is absent).

    Args:
        cases: Parsed case durations.

    Returns:
        ``(suite, total_seconds, n_cases)`` rows, slowest first.
    """
    totals: dict[str, list[float]] = {}
    for case in cases:
        key = case.suite
        totals.setdefault(key, []).append(case.seconds)
    rows = [(suite, sum(times), len(times)) for suite, times in totals.items()]
    rows.sort(key=lambda r: r[1], reverse=True)
    return rows


def format_duration_table(
    cases: list[CaseDuration],
    *,
    top: int,
    min_seconds: float = 0.0,
) -> str:
    """Render a plain-text ranking of the slowest cases.

    Args:
        cases: Durations sorted descending (as from ``parse_junit_durations``).
        top: Maximum rows to include.
        min_seconds: Drop cases strictly below this threshold.

    Returns:
        Multi-line table suitable for terminal or markdown fences.
    """
    filtered = [c for c in cases if c.seconds >= min_seconds][:top]
    if not filtered:
        return "(no cases at or above threshold)"
    width = max(len(c.nodeid) for c in filtered)
    width = min(max(width, 12), 120)
    lines = [f"{'seconds':>10}  nodeid", f"{'-' * 10}  {'-' * width}"]
    for case in filtered:
        node = (
            case.nodeid
            if len(case.nodeid) <= width
            else f"…{case.nodeid[-(width - 1) :]}"
        )
        lines.append(f"{case.seconds:10.3f}  {node}")
    return "\n".join(lines)


def format_file_table(
    rows: list[tuple[str, float, int]],
    *,
    top: int,
) -> str:
    """Render a plain-text ranking of the slowest files/suites."""
    sliced = rows[:top]
    if not sliced:
        return "(no suites)"
    width = max(len(r[0]) for r in sliced)
    width = min(max(width, 12), 100)
    lines = [
        f"{'seconds':>10}  {'n':>5}  suite",
        f"{'-' * 10}  {'-' * 5}  {'-' * width}",
    ]
    for suite, total, n in sliced:
        label = suite if len(suite) <= width else f"…{suite[-(width - 1) :]}"
        lines.append(f"{total:10.3f}  {n:5d}  {label}")
    return "\n".join(lines)


def run_pytest_profile(
    *,
    junit_path: Path,
    pytest_args: list[str],
    repo_root: Path = _REPO_ROOT,
) -> int:
    """Run pytest with ``--junitxml`` and return its exit code.

    Args:
        junit_path: Destination for the JUnit report.
        pytest_args: Extra args after ``pytest`` (marker/cov flags, paths).
        repo_root: Working directory for the subprocess.

    Returns:
        Pytest process return code.
    """
    junit_path.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        "uv",
        "run",
        "pytest",
        f"--junitxml={junit_path}",
        "--durations=25",
        *pytest_args,
    ]
    print("+", " ".join(cmd), flush=True)
    completed = run_command(cmd, cwd=repo_root, check=False)
    return int(completed.returncode)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--top",
        type=int,
        default=25,
        help="Number of slowest cases / files to print (default 25)",
    )
    parser.add_argument(
        "--min-seconds",
        type=float,
        default=0.5,
        help="Hide cases faster than this many seconds (default 0.5)",
    )
    parser.add_argument(
        "--junit",
        type=Path,
        default=_DEFAULT_JUNIT,
        help=f"JUnit XML path (default {_DEFAULT_JUNIT})",
    )
    parser.add_argument(
        "--parse-only",
        type=Path,
        default=None,
        metavar="JUNIT_XML",
        help="Skip running pytest; rank an existing JUnit report",
    )
    parser.add_argument(
        "--pytest-args",
        default="",
        help=(
            "Extra pytest argv string. Default mirrors CI: "
            f"{' '.join(_CI_PYTEST_ARGS)!r}"
        ),
    )
    parser.add_argument(
        "--no-default-ci-args",
        action="store_true",
        help="Do not prepend the CI default -m/--cov flags",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Emit machine-readable JSON summary to stdout (after tables)",
    )
    parser.add_argument(
        "-o",
        "--output",
        type=Path,
        default=None,
        help="Also write the JSON summary to this path",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """CLI entry: run or parse, then print / write duration rankings."""
    args = _build_parser().parse_args(argv)
    junit_path = args.junit
    pytest_rc = 0

    if args.parse_only is not None:
        junit_path = args.parse_only
    else:
        extra: list[str] = (
            list(args.pytest_args.split()) if args.pytest_args.strip() else []
        )
        base: list[str] = [] if args.no_default_ci_args else list(_CI_PYTEST_ARGS)
        pytest_args = base + extra
        pytest_rc = run_pytest_profile(junit_path=junit_path, pytest_args=pytest_args)

    cases = parse_junit_durations(junit_path)
    files = aggregate_by_file(cases)
    total = sum(c.seconds for c in cases)
    print()
    print(f"cases={len(cases)}  sum_case_seconds={total:.1f}  junit={junit_path}")
    print()
    print(f"Top {args.top} slowest cases (≥ {args.min_seconds:g}s):")
    print(format_duration_table(cases, top=args.top, min_seconds=args.min_seconds))
    print()
    print(f"Top {args.top} files by summed case time:")
    print(format_file_table(files, top=args.top))

    payload = {
        "junit": str(junit_path),
        "n_cases": len(cases),
        "sum_case_seconds": total,
        "top_cases": [asdict(c) for c in cases[: args.top]],
        "top_files": [
            {"suite": s, "seconds": t, "n_cases": n} for s, t, n in files[: args.top]
        ],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"\nWrote {args.output}")
    if args.json:
        print()
        print(json.dumps(payload, indent=2))

    # Surface pytest failures, but still emit the ranking when XML exists.
    return pytest_rc


if __name__ == "__main__":
    sys.exit(main())
