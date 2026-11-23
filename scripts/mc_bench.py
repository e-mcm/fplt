#!/usr/bin/env python3
"""Per-step MC timing harness (native-mc-build-spec §1.1).

PR1 stub: prints the measurement protocol and exits 0. Later PRs wire the
real fixture timing once kernels land. Safe to run on any machine; does not
compile C++.
"""

from __future__ import annotations

import argparse
import sys


def main(argv: list[str] | None = None) -> int:
    """Print the §1.1 benchmark protocol."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--n-sims", type=int, default=10_000, help="Simulations per fixture"
    )
    parser.add_argument(
        "--backend",
        choices=("python", "native"),
        default="python",
        help="MC backend to time (native requires fplt-mc wheel)",
    )
    args = parser.parse_args(argv)

    print("mc-bench (PR1 stub)")
    print(f"  n_sims={args.n_sims} backend={args.backend}")
    print("  Protocol: 40 players, 33×33 scoreline grid, dc_layered_v3 fidelity.")
    print("  Steps to time: hard_starters, allocate_side, points, minutes, scoreline.")
    print("  Full harness lands with K1+; rerun on the cloud VM before committing.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
