"""Capture vaastav seasons into ``$FPLT_DATA_DIR/raw/vaastav/captured_at=…``.

Mirrors upstream ``data/`` trees write-once and writes ``_source.json``
(``VaastavSourceManifest``). Unchanged seasons are skipped by comparing
sha256 digests against prior captures. Requires ``git`` on ``PATH``.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shutil
import sys
from datetime import UTC, datetime
from pathlib import Path

from fplt.data.capture_health import record_capture_outcome
from fplt.data.vaastav import is_complete_vaastav_capture, sha256_file
from fplt.schemas.vaastav_source import VaastavSourceFile, VaastavSourceManifest
from fplt.utils.process import run_command

REPO_URL = "https://github.com/vaastav/Fantasy-Premier-League.git"
_SOURCE = "_source.json"
_SEASON_RE = re.compile(r"^\d{4}-\d{2}$")
_CAP_RE = re.compile(r"^captured_at=(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)$")
_log = logging.getLogger(__name__)


def default_raw_dir() -> Path:
    """Return ``$FPLT_DATA_DIR/raw`` when set, otherwise ``./data/raw``."""
    root = (
        Path(os.environ["FPLT_DATA_DIR"])
        if os.environ.get("FPLT_DATA_DIR")
        else Path("./data")
    )
    return root / "raw"


def default_vaastav_cache_dir(raw_root: Path) -> Path:
    """Return a sparse-clone cache path that is never under ``raw_root``.

    Defaults to a sibling of ``raw_root`` named ``vaastav_cache`` so GHA
    ``--out $RUNNER_TEMP/raw`` keeps the git checkout outside the upload tree.
    """
    return raw_root.resolve().parent / "vaastav_cache"


def stamp_captured_at(moment: datetime) -> str:
    """Format ``moment`` as a ``captured_at`` directory stamp."""
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def is_season_dir(path: Path) -> bool:
    """True when ``path`` is a season folder with ``gws/merged_gw.csv``."""
    return (
        path.is_dir()
        and _SEASON_RE.match(path.name) is not None
        and (path / "gws" / "merged_gw.csv").is_file()
    )


def list_season_dirs(data_dir: Path) -> list[Path]:
    """Return sorted season directories under vaastav ``data/``."""
    if not data_dir.is_dir():
        return []
    return sorted(p for p in data_dir.iterdir() if is_season_dir(p))


def _run_git(args: list[str], cwd: Path) -> str:
    result = run_command(
        ["git", *args],
        cwd=cwd,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        detail = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(f"git {' '.join(args)} failed: {detail}")
    return (result.stdout or "").strip()


def resolve_commit_sha(repo: Path) -> str:
    """Return lowercase ``HEAD`` sha for ``repo``."""
    return _run_git(["rev-parse", "HEAD"], cwd=repo).lower()


def fetch_vaastav_repo(cache_dir: Path, *, repo_url: str = REPO_URL) -> Path:
    """Clone or update a sparse vaastav checkout under ``cache_dir``."""
    if (cache_dir / ".git").is_dir():
        _run_git(["pull", "--ff-only", "origin", "main"], cwd=cache_dir)
        return cache_dir
    parent = cache_dir.parent
    parent.mkdir(parents=True, exist_ok=True)
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
    return cache_dir


def iter_capture_dirs(raw_root: Path) -> list[Path]:
    """Return ``raw_root/vaastav/captured_at=*`` dirs sorted by stamp."""
    base = raw_root / "vaastav"
    if not base.is_dir():
        return []
    found: list[tuple[datetime, Path]] = []
    for path in base.iterdir():
        match = _CAP_RE.match(path.name)
        if match and path.is_dir():
            stamp = datetime.strptime(match.group(1), "%Y-%m-%dT%H-%M-%SZ").replace(
                tzinfo=UTC
            )
            found.append((stamp, path))
    return [p for _, p in sorted(found)]


def load_prior_file_hashes(raw_root: Path) -> dict[str, str]:
    """Merge ``path → sha256`` from prior complete manifests (latest wins).

    Incomplete captures (missing ``_source.json`` or a listed file / digest
    mismatch) are skipped so the next run re-captures those seasons (#353).
    """
    hashes: dict[str, str] = {}
    for capture in iter_capture_dirs(raw_root):
        if not is_complete_vaastav_capture(capture):
            continue
        manifest_path = capture / _SOURCE
        manifest = VaastavSourceManifest.model_validate(
            json.loads(manifest_path.read_text(encoding="utf-8"))
        )
        for entry in manifest.files:
            hashes[entry.path] = entry.sha256
    return hashes


def file_entries_for(
    data_dir: Path, seasons: set[str] | None = None
) -> list[VaastavSourceFile]:
    """Build ``VaastavSourceFile`` rows (paths relative to the capture root)."""
    entries: list[VaastavSourceFile] = []
    for path in sorted(data_dir.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(data_dir)
        if rel.parts and _SEASON_RE.match(rel.parts[0]):
            if seasons is not None and rel.parts[0] not in seasons:
                continue
        entries.append(
            VaastavSourceFile(
                path=(Path("data") / rel).as_posix(),
                bytes=path.stat().st_size,
                sha256=sha256_file(path),
            )
        )
    return entries


def _top_level_changed(data_dir: Path, prior: dict[str, str]) -> bool:
    for path in sorted(data_dir.iterdir()):
        if path.is_file():
            if prior.get(f"data/{path.name}") != sha256_file(path):
                return True
        elif path.is_dir() and not is_season_dir(path):
            for nested in sorted(path.rglob("*")):
                if nested.is_file():
                    key = f"data/{nested.relative_to(data_dir).as_posix()}"
                    if prior.get(key) != sha256_file(nested):
                        return True
    return False


def seasons_needing_capture(
    data_dir: Path, prior: dict[str, str], *, full: bool = False
) -> list[str]:
    """Return seasons whose files differ from ``prior`` (or all when ``full``)."""
    all_seasons = [p.name for p in list_season_dirs(data_dir)]
    changed = [
        season
        for season in all_seasons
        if any(
            prior.get(e.path) != e.sha256 for e in file_entries_for(data_dir, {season})
        )
    ]
    top_changed = _top_level_changed(data_dir, prior)
    if not changed and not top_changed and prior:
        return []
    if full or not prior:
        return all_seasons
    return changed


def write_capture(
    raw_root: Path,
    data_dir: Path,
    *,
    seasons: list[str],
    repo: str,
    commit_sha: str,
    captured_at: datetime,
    include_top_level: bool = True,
) -> Path:
    """Write one write-once ``captured_at=…`` tree and ``_source.json``."""
    if not seasons and not include_top_level:
        raise ValueError("seasons must be non-empty when include_top_level is false")
    dest = raw_root / "vaastav" / f"captured_at={stamp_captured_at(captured_at)}"
    if dest.exists():
        raise FileExistsError(f"capture already exists: {dest}")
    dest.mkdir(parents=True, exist_ok=False)
    try:
        for season in seasons:
            src = data_dir / season
            if not is_season_dir(src):
                raise FileNotFoundError(f"missing season directory: {src}")
            shutil.copytree(src, dest / "data" / season)
        if include_top_level:
            for path in sorted(data_dir.iterdir()):
                if path.is_file():
                    (dest / "data").mkdir(parents=True, exist_ok=True)
                    shutil.copy2(path, dest / "data" / path.name)
                elif path.is_dir() and not is_season_dir(path):
                    shutil.copytree(path, dest / "data" / path.name)
        entries = file_entries_for(
            dest / "data", None if include_top_level else set(seasons)
        )
        if not entries:
            raise ValueError("capture would contain no files")
        manifest = VaastavSourceManifest(
            repo=repo,
            commit_sha=commit_sha,
            captured_at=captured_at.astimezone(UTC),
            seasons=list(seasons),
            files=entries,
        )
        (dest / _SOURCE).write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
    except Exception:
        if dest.exists():
            shutil.rmtree(dest)
        raise
    _log.info(
        "wrote vaastav capture %s seasons=%s files=%d",
        dest.name,
        ",".join(seasons) or "-",
        len(entries),
    )
    return dest


def capture_vaastav(
    raw_root: Path,
    *,
    repo_path: Path | None = None,
    cache_dir: Path | None = None,
    repo_url: str = REPO_URL,
    full: bool = False,
    captured_at: datetime | None = None,
) -> Path | None:
    """Capture changed vaastav seasons into ``raw_root/vaastav/``.

    Returns the new capture directory, or ``None`` when nothing changed.
    """
    if repo_path is None:
        cache = (
            cache_dir if cache_dir is not None else default_vaastav_cache_dir(raw_root)
        )
        raw_resolved = raw_root.resolve()
        cache_resolved = cache.resolve()
        if cache_resolved == raw_resolved or raw_resolved in cache_resolved.parents:
            raise ValueError(
                f"vaastav cache_dir {cache} must not be under raw root {raw_root}"
            )
        repo = fetch_vaastav_repo(cache, repo_url=repo_url)
        repo_label = repo_url
    else:
        repo = repo_path
        repo_label = str(repo_path)
    data_dir = repo / "data"
    if not data_dir.is_dir():
        raise FileNotFoundError(f"vaastav data/ missing under {repo}")
    prior = load_prior_file_hashes(raw_root)
    seasons = seasons_needing_capture(data_dir, prior, full=full)
    top_changed = _top_level_changed(data_dir, prior)
    if not seasons and not top_changed:
        _log.info("vaastav capture: nothing new")
        return None
    return write_capture(
        raw_root,
        data_dir,
        seasons=seasons,
        repo=repo_label,
        commit_sha=resolve_commit_sha(repo),
        captured_at=captured_at if captured_at is not None else datetime.now(UTC),
        include_top_level=full or not prior or top_changed,
    )


def main(argv: list[str] | None = None) -> int:
    """Parse CLI args and capture vaastav into the raw layout."""
    parser = argparse.ArgumentParser(
        prog="bootstrap_vaastav",
        description="Capture vaastav into $FPLT_DATA_DIR/raw/vaastav/captured_at=…",
    )
    parser.add_argument(
        "--out", type=Path, default=None, help="Raw root (default: …/raw)."
    )
    parser.add_argument("--repo", type=Path, default=None, help="Local git checkout.")
    parser.add_argument(
        "--cache-dir", type=Path, default=None, help="Sparse-clone cache."
    )
    parser.add_argument("--full", action="store_true", help="Include every season.")
    parser.add_argument(
        "--captured-at", default=None, help="UTC stamp YYYY-MM-DDTHH-MM-SSZ."
    )
    parser.add_argument("--repo-url", default=REPO_URL, help="Upstream git remote.")
    parser.add_argument(
        "--status-dir",
        type=Path,
        default=None,
        help="Local status/ mirror for unavailable counters (default: …/status).",
    )
    parser.add_argument(
        "--unavailable-threshold",
        type=int,
        default=None,
        help="Consecutive unavailable runs before exit non-zero (default: config).",
    )
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    out = args.out if args.out is not None else default_raw_dir()
    status_dir = (
        args.status_dir
        if args.status_dir is not None
        else (
            Path(os.environ["FPLT_DATA_DIR"]) / "status"
            if os.environ.get("FPLT_DATA_DIR")
            else Path("./data/status")
        )
    )
    captured_at: datetime | None = None
    if args.captured_at is not None:
        try:
            captured_at = datetime.strptime(
                args.captured_at, "%Y-%m-%dT%H-%M-%SZ"
            ).replace(tzinfo=UTC)
        except ValueError:
            _log.error(
                "invalid --captured-at %r (want YYYY-MM-DDTHH-MM-SSZ)",
                args.captured_at,
            )
            return 1
    moment = datetime.now(UTC)
    try:
        written = capture_vaastav(
            out,
            repo_path=args.repo,
            cache_dir=args.cache_dir,
            repo_url=args.repo_url,
            full=bool(args.full),
            captured_at=captured_at,
        )
    except RuntimeError as exc:
        # Upstream git unreachable / clone-pull failure (#338).
        outcome = record_capture_outcome(
            source="vaastav",
            status_dir=status_dir,
            available=False,
            reason=str(exc),
            threshold=args.unavailable_threshold,
            now=moment,
        )
        _log.error("%s", exc)
        return int(outcome.exit_code)
    except (FileNotFoundError, FileExistsError, ValueError) as exc:
        _log.error("%s", exc)
        return 1
    outcome = record_capture_outcome(
        source="vaastav",
        status_dir=status_dir,
        available=True,
        threshold=args.unavailable_threshold,
        now=moment,
    )
    if written is None:
        print(f"nothing new under {out.resolve()}")
        return int(outcome.exit_code)
    print(f"wrote {written.resolve()}")
    return int(outcome.exit_code)


if __name__ == "__main__":
    sys.exit(main())
