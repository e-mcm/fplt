"""
data_setup.py — Download FPL data from vaastav/Fantasy-Premier-League into this directory.

Usage:
    python data/data_setup.py

Requires git to be installed and on PATH.
"""

import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/vaastav/Fantasy-Premier-League.git"
DATA_DIR = Path(__file__).parent.resolve()


def run(cmd: list[str]) -> None:
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=DATA_DIR)
    if result.returncode != 0:
        sys.exit(result.returncode)


def main() -> None:
    print(f"Downloading FPL data into {DATA_DIR} ...")
    run(["git", "init"])
    run(["git", "remote", "add", "origin", REPO_URL])
    run(["git", "sparse-checkout", "set", "data"])
    run(["git", "pull", "--depth=1", "origin", "main"])

    # Move contents of data/ up one level (repo puts it in data/data/)
    nested = DATA_DIR / "data"
    if nested.is_dir():
        for item in nested.iterdir():
            item.rename(DATA_DIR / item.name)
        nested.rmdir()

    print("\nDone.")


if __name__ == "__main__":
    main()
