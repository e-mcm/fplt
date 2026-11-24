"""Thin wrapper around ``fplt perf-report`` for Makefile / ad-hoc use.

Example::

    uv run python scripts/perf_report.py --limit 5
    make perf-report ARGS='--kind walk_forward --json'
"""

from __future__ import annotations

import sys

from fplt.cli import main


def run(argv: list[str] | None = None) -> None:
    """Forward argv to ``fplt perf-report``."""
    args = ["perf-report", *(argv if argv is not None else sys.argv[1:])]
    main(args)


if __name__ == "__main__":
    run()
