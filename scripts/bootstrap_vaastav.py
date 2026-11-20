"""Place vaastav historical season directories under ``$FPLT_DATA_DIR``.

``import_available`` expects season folders named ``YYYY-YY`` (with
``gws/merged_gw.csv``) directly under the data directory. This script clones
or refreshes [vaastav/Fantasy-Premier-League](https://github.com/vaastav/Fantasy-Premier-League)
and copies those seasons into place.

Requires ``git`` on ``PATH``. Network access is only needed when ``--source``
is omitted.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import subprocess
import sys
from pathlib import Path

REPO_URL = "https://github.com/vaastav/Fantasy-Premier-League.git"
MIN_SEASON = "2021-22"
_CACHE_NAME = ".vaastav"
_log = logging.getLogger(__name__)


def default_data_dir() -> Path:
    """Return ``$FPLT_DATA_DIR`` when set, otherwise ``./data``."""
    raw = os.environ.get("FPLT_DATA_DIR")
    return Path(raw) if raw else Path("./data")


def is_season_dir(path: Path, min_season: str = MIN_SEASON) -> bool:
    """True when ``path`` looks like a vaastav season ``import_available`` can use."""
    if not path.is_dir():
        return False
    if path.name < min_season:
        return False
    return (path / "gws" / "merged_gw.csv").is_file()


def list_seasons(source: Path, min_season: str = MIN_SEASON) -> list[Path]:
    """Return season directories under ``source`` that meet the minimum season.

    Args:
        source: Directory whose children are season folders (vaastav ``data/``).
        min_season: Earliest season label to keep (inclusive), e.g. ``"2021-22"``.

    Returns:
        Season directories sorted by name.
    """
    if not source.is_dir():
        return []
    return sorted(path for path in source.iterdir() if is_season_dir(path, min_season))


def place_seasons(
    source: Path,
    data_dir: Path,
    *,
    min_season: str = MIN_SEASON,
    force: bool = False,
) -> list[str]:
    """Copy qualifying season directories from ``source`` into ``data_dir``.

    Existing season directories are left alone unless ``force`` is true.

    Args:
        source: Directory containing vaastav season folders.
        data_dir: Destination root (``$FPLT_DATA_DIR``).
        min_season: Earliest season label to copy.
        force: Replace existing destination season directories when true.

    Returns:
        Season labels that were copied (or refreshed).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    placed: list[str] = []
    for season_dir in list_seasons(source, min_season):
        dest = data_dir / season_dir.name
        if dest.exists() and not force:
            _log.info("skip existing %s", dest)
            continue
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(season_dir, dest)
        placed.append(season_dir.name)
        _log.info("placed %s", dest)
    return placed


def _run_git(args: list[str], cwd: Path) -> None:
    result = subprocess.run(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")


def fetch_vaastav(cache_dir: Path, repo_url: str = REPO_URL) -> Path:
    """Clone or update a sparse checkout of vaastav under ``cache_dir``.

    Args:
        cache_dir: Local cache directory for the sparse clone.
        repo_url: Git remote for the Fantasy-Premier-League repository.

    Returns:
        Path to the repository's ``data/`` directory containing season folders.

    Raises:
        RuntimeError: If a git command fails.
    """
    data = cache_dir / "data"
    if (cache_dir / ".git").is_dir():
        _run_git(["pull", "--ff-only", "origin", "main"], cwd=cache_dir)
        return data
    parent = cache_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
    # Clone into a sibling path, then rename, so a failed clone leaves no
    # half-written tree at the final cache path.
    staging = parent / f"{cache_dir.name}.partial"
    if staging.exists():
        shutil.rmtree(staging)
    _run_git(
        [
            "clone",
            "--depth=1",
            "--filter=blob:none",
            "--sparse",
            repo_url,
            staging.name,
        ],
        cwd=parent,
    )
    _run_git(["sparse-checkout", "set", "data"], cwd=staging)
    if cache_dir.exists():
        shutil.rmtree(cache_dir)
    staging.rename(cache_dir)
    return data


def bootstrap(
    data_dir: Path,
    *,
    source: Path | None = None,
    min_season: str = MIN_SEASON,
    force: bool = False,
    repo_url: str = REPO_URL,
) -> list[str]:
    """Place vaastav season directories under ``data_dir`` for ``import_available``.

    When ``source`` is omitted, clones or refreshes a sparse checkout under
    ``data_dir/.vaastav`` and copies seasons from there.

    Args:
        data_dir: Destination root (``$FPLT_DATA_DIR``).
        source: Optional local vaastav ``data/`` directory to copy from.
        min_season: Earliest season label to place.
        force: Replace existing season directories when true.
        repo_url: Remote used when ``source`` is omitted.

    Returns:
        Season labels that were newly copied or refreshed.
    """
    if source is None:
        source = fetch_vaastav(data_dir / _CACHE_NAME, repo_url=repo_url)
    return place_seasons(source, data_dir, min_season=min_season, force=force)


def main(argv: list[str] | None = None) -> int:
    """Parse CLI args and place vaastav seasons under the data directory."""
    parser = argparse.ArgumentParser(
        prog="bootstrap_vaastav",
        description=(
            "Place vaastav season directories where import_available expects them "
            "under $FPLT_DATA_DIR."
        ),
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=None,
        help="Destination root (default: $FPLT_DATA_DIR or ./data).",
    )
    parser.add_argument(
        "--source",
        type=Path,
        default=None,
        help="Local vaastav data/ directory; skip the git clone when set.",
    )
    parser.add_argument(
        "--min-season",
        default=MIN_SEASON,
        help=f"Earliest season to copy (default: {MIN_SEASON}).",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing season directories under the data directory.",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    data_dir = args.data_dir if args.data_dir is not None else default_data_dir()
    try:
        placed = bootstrap(
            data_dir,
            source=args.source,
            min_season=args.min_season,
            force=args.force,
        )
    except RuntimeError as exc:
        _log.error("%s", exc)
        return 1
    if placed:
        print(f"placed {', '.join(placed)} under {data_dir.resolve()}")
    else:
        print(
            f"no seasons copied under {data_dir.resolve()} "
            "(already present, or none matched --min-season)"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
