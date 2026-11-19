"""Command line entry point for fplt."""

from __future__ import annotations

import argparse
import sys


def _not_implemented(name: str) -> int:
    print(f"fplt {name} is not implemented yet", file=sys.stderr)
    return 2


def main(argv: list[str] | None = None) -> None:
    """Parse arguments and dispatch a subcommand."""
    parser = argparse.ArgumentParser(prog="fplt")
    sub = parser.add_subparsers(dest="command")
    sub.add_parser("backtest", help="Walk-forward backtest a model")
    sub.add_parser("predict", help="Predict the next gameweeks")
    sub.add_parser("plan", help="Recommend a squad and transfers")
    args = parser.parse_args(argv)
    if args.command is None:
        return
    raise SystemExit(_not_implemented(args.command))


if __name__ == "__main__":
    main()
