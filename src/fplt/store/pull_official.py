"""Pull official pointers, runs, and curated UI tables (weekly-ui §3.2 / #280)."""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fplt.data.remote import Remote, RemoteError, pull, r2_credentials_complete
from fplt.schemas.manifests import OfficialPointer
from fplt.settings import Settings, get_settings
from fplt.store.builds import (
    build_manifest_path,
    load_build_manifest,
    table_sha_path,
)
from fplt.store.push import load_run_manifest
from fplt.store.runs import run_dir

logger = logging.getLogger(__name__)

_DEFAULT_CHECKPOINT = "T-90m"
_CHECKPOINT_ALIASES = frozenset({"manual", "latest"})
_MANUAL_PREFIX = "manual-"
_UI_TABLES = frozenset({"players", "teams", "fixtures", "gameweeks", "player_status"})
_GW_RE = re.compile(r"^gw=(\d+)$")
_CRED_MSG = (
    "fplt data pull --official requires FPLT_DATA_BUCKET and "
    "FPLT_R2_ACCOUNT_ID, FPLT_R2_ACCESS_KEY_ID, FPLT_R2_SECRET_ACCESS_KEY"
)


class OfficialPullError(RuntimeError):
    """Official pointer resolution or pull failed."""


@dataclass(frozen=True)
class OfficialPullResult:
    """Summary of one ``pull_official`` invocation."""

    season: str
    gw: int
    checkpoint: str
    line: str | None
    predict_run_ids: tuple[str, ...]
    plan_run_ids: tuple[str, ...]
    build_ids: tuple[str, ...]
    n_written: int


def require_data_remote(settings: Settings | None = None) -> Remote:
    """Build the data remote; raise naming ``FPLT_R2_*`` when incomplete."""
    cfg = settings if settings is not None else get_settings()
    if not cfg.data_bucket or not r2_credentials_complete(cfg):
        raise RemoteError(_CRED_MSG)
    remote = Remote.from_settings(cfg)
    if remote is None:
        raise RemoteError(_CRED_MSG)
    return remote


def list_official_availability(official_root: Path, *, season: str) -> list[str]:
    """Return sorted ``gw=N checkpoint=C`` labels under one season."""
    season_dir = official_root / f"season={season}"
    found: set[str] = set()
    if season_dir.is_dir():
        for gw_dir in season_dir.glob("gw=*"):
            m = _GW_RE.match(gw_dir.name)
            if not m or not gw_dir.is_dir():
                continue
            for cp in gw_dir.glob("checkpoint=*"):
                if cp.is_dir():
                    cp_id = cp.name.removeprefix("checkpoint=")
                    found.add(f"gw={m.group(1)} checkpoint={cp_id}")
    return sorted(found, key=lambda s: (int(s.split()[0][3:]), s))


def resolve_latest_gw(
    official_root: Path,
    *,
    season: str,
    checkpoint: str = _DEFAULT_CHECKPOINT,
    primary_line: str | None = None,
) -> int:
    """Highest GW with a primary-line pointer at ``checkpoint``.

    ``checkpoint`` may be a concrete id (``T-90m``, ``manual-…``) or an alias
    (``manual`` / ``latest``); aliases resolve via ``resolve_checkpoint_target``.
    """
    if checkpoint in _CHECKPOINT_ALIASES:
        gw, _ = resolve_checkpoint_target(
            official_root,
            season=season,
            checkpoint=checkpoint,
            primary_line=primary_line,
        )
        return gw
    line = primary_line or _primary_line_name()
    best: int | None = None
    season_dir = official_root / f"season={season}"
    if season_dir.is_dir():
        for gw_dir in season_dir.glob("gw=*"):
            m = _GW_RE.match(gw_dir.name)
            if not m or not gw_dir.is_dir():
                continue
            gw = int(m.group(1))
            if (gw_dir / f"checkpoint={checkpoint}" / f"line={line}.json").is_file():
                best = gw if best is None else max(best, gw)
    if best is not None:
        return best
    avail = list_official_availability(official_root, season=season)
    raise OfficialPullError(
        f"no official pointer for season={season} checkpoint={checkpoint} "
        f"line={line}; available: {', '.join(avail) if avail else '(none)'}"
    )


def resolve_checkpoint_target(
    official_root: Path,
    *,
    season: str,
    checkpoint: str,
    gw: int | None = None,
    primary_line: str | None = None,
) -> tuple[int, str]:
    """Resolve ``checkpoint`` alias / concrete id to ``(gw, concrete_id)``.

    Aliases (#364):
    - ``manual``: newest ``manual-*`` on the highest GW that has one.
    - ``latest``: newest pointer of any checkpoint (by ``written_at``, then
      checkpoint id) for the primary line.

    A concrete checkpoint id with ``gw=None`` picks the highest GW that has
    that checkpoint (same as ``resolve_latest_gw``).
    """
    line = primary_line or _primary_line_name()
    if checkpoint == "manual":
        return _resolve_manual_alias(official_root, season=season, line=line, gw=gw)
    if checkpoint == "latest":
        return _resolve_latest_alias(official_root, season=season, line=line, gw=gw)
    if gw is not None:
        path = (
            official_root
            / f"season={season}"
            / f"gw={gw}"
            / f"checkpoint={checkpoint}"
            / f"line={line}.json"
        )
        if path.is_file():
            return gw, checkpoint
        avail = list_official_availability(official_root, season=season)
        raise OfficialPullError(
            f"no official pointer for season={season} gw={gw} "
            f"checkpoint={checkpoint} line={line}; available: "
            f"{', '.join(avail) if avail else '(none)'}"
        )
    resolved_gw = resolve_latest_gw(
        official_root, season=season, checkpoint=checkpoint, primary_line=line
    )
    return resolved_gw, checkpoint


def _resolve_manual_alias(
    official_root: Path, *, season: str, line: str, gw: int | None
) -> tuple[int, str]:
    """Newest ``manual-*`` checkpoint for ``line`` (optionally pinned GW)."""
    candidates = _primary_pointer_candidates(
        official_root, season=season, line=line, gw=gw
    )
    manuals = [
        item for item in candidates if item.checkpoint.startswith(_MANUAL_PREFIX)
    ]
    if not manuals:
        avail = list_official_availability(official_root, season=season)
        raise OfficialPullError(
            f"no manual-* official pointer for season={season} line={line}"
            + (f" gw={gw}" if gw is not None else "")
            + f"; available: {', '.join(avail) if avail else '(none)'}"
        )
    # Prefer highest GW, then lexicographic stamp (second-resolution ids).
    best = max(manuals, key=lambda item: (item.gw, item.checkpoint, item.written_at))
    return best.gw, best.checkpoint


def _resolve_latest_alias(
    official_root: Path, *, season: str, line: str, gw: int | None
) -> tuple[int, str]:
    """Newest pointer of any checkpoint for ``line`` (by written_at)."""
    candidates = _primary_pointer_candidates(
        official_root, season=season, line=line, gw=gw
    )
    if not candidates:
        avail = list_official_availability(official_root, season=season)
        raise OfficialPullError(
            f"no official pointer for season={season} line={line}"
            + (f" gw={gw}" if gw is not None else "")
            + f"; available: {', '.join(avail) if avail else '(none)'}"
        )
    best = max(candidates, key=lambda item: (item.written_at, item.gw, item.checkpoint))
    return best.gw, best.checkpoint


@dataclass(frozen=True)
class _PointerCandidate:
    """One primary-line pointer under ``official/`` for alias resolution."""

    gw: int
    checkpoint: str
    written_at: datetime


def _primary_pointer_candidates(
    official_root: Path, *, season: str, line: str, gw: int | None
) -> list[_PointerCandidate]:
    """List primary-line pointers, optionally filtered to one GW."""
    season_dir = official_root / f"season={season}"
    if not season_dir.is_dir():
        return []
    out: list[_PointerCandidate] = []
    for gw_dir in season_dir.glob("gw=*"):
        m = _GW_RE.match(gw_dir.name)
        if not m or not gw_dir.is_dir():
            continue
        gw_n = int(m.group(1))
        if gw is not None and gw_n != gw:
            continue
        for cp_dir in gw_dir.glob("checkpoint=*"):
            if not cp_dir.is_dir():
                continue
            path = cp_dir / f"line={line}.json"
            if not path.is_file():
                continue
            try:
                pointer = OfficialPointer.model_validate_json(
                    path.read_text(encoding="utf-8")
                )
            except ValueError:
                continue
            out.append(
                _PointerCandidate(
                    gw=gw_n,
                    checkpoint=cp_dir.name.removeprefix("checkpoint="),
                    written_at=pointer.written_at,
                )
            )
    return out


def _primary_line_name() -> str:
    """Lazy load of the production primary line (avoids import cycle with ops)."""
    from fplt.automation.production import load_production_ops

    return load_production_ops().primary_line().line


def mirror_official_index(data_dir: Path, remote: Remote) -> Path:
    """Pull ``official/`` into ``data_dir`` (small JSON index)."""
    dest = data_dir / "official"
    try:
        pull(remote, "official", dest)
    except RemoteError as exc:
        if "directory not found" in str(exc).lower():
            raise OfficialPullError(
                "no official/ prefix in the data bucket yet; available: (none)"
            ) from exc
        raise
    return dest


def infer_official_season(official_root: Path, season: str | None) -> str:
    """Return ``season`` or the max ``season=*`` under ``official_root``."""
    if season is not None:
        return season
    seasons = sorted(
        p.name.removeprefix("season=")
        for p in official_root.glob("season=*")
        if p.is_dir()
    )
    if not seasons:
        raise OfficialPullError("no official/season=* directories found; pass --season")
    return seasons[-1]


def pull_official(
    data_dir: Path,
    remote: Remote,
    *,
    season: str,
    gw: int,
    checkpoint: str = _DEFAULT_CHECKPOINT,
    line: str | None = None,
) -> OfficialPullResult:
    """Pull one GW's official pointers, runs, and curated UI tables."""
    prefix = f"official/season={season}/gw={gw}"
    gw_dir = data_dir / "official" / f"season={season}" / f"gw={gw}"
    written = _safe_pull(remote, prefix, gw_dir)
    pointers = _load_pointers(gw_dir, checkpoint=checkpoint, line=line)
    if not pointers:
        raise OfficialPullError(
            f"no official line pointers under {prefix}/checkpoint={checkpoint}"
            + (f"/line={line}" if line else "")
        )
    predict_ids: list[str] = []
    plan_ids: list[str] = []
    for pointer in pointers:
        for run_id in (pointer.predict_run_id, pointer.plan_run_id):
            written += _pull_run(data_dir, remote, run_id)
        predict_ids.append(pointer.predict_run_id)
        plan_ids.append(pointer.plan_run_id)
    build_ids: list[str] = []
    for predict_id in dict.fromkeys(predict_ids):
        build_id = load_run_manifest(data_dir, predict_id).build_id
        if build_id:
            build_ids.append(build_id)
            written += pull_build(data_dir, remote, build_id, tables=_UI_TABLES)
    result = OfficialPullResult(
        season=season,
        gw=gw,
        checkpoint=checkpoint,
        line=line,
        predict_run_ids=tuple(dict.fromkeys(predict_ids)),
        plan_run_ids=tuple(dict.fromkeys(plan_ids)),
        build_ids=tuple(dict.fromkeys(build_ids)),
        n_written=written,
    )
    logger.info(
        "pulled official season=%s gw=%s predict=%s plan=%s new=%d",
        season,
        gw,
        result.predict_run_ids,
        result.plan_run_ids,
        written,
    )
    return result


def _safe_pull(remote: Remote, prefix: str, dest: Path) -> int:
    try:
        return len(pull(remote, prefix, dest))
    except RemoteError as exc:
        if "directory not found" in str(exc).lower():
            return 0
        raise


def _pull_run(data_dir: Path, remote: Remote, run_id: str) -> int:
    dest = run_dir(data_dir, run_id)
    try:
        n = len(pull(remote, f"runs/run={run_id}", dest))
    except RemoteError as exc:
        if "directory not found" in str(exc).lower():
            raise OfficialPullError(
                f"run {run_id} missing in the data bucket "
                "(referenced by an official pointer)"
            ) from exc
        raise
    if not (dest / "_SUCCESS").is_file():
        raise OfficialPullError(
            f"run {run_id} missing in the data bucket "
            "(referenced by an official pointer)"
        )
    return n


def pull_build(
    data_dir: Path,
    remote: Remote,
    build_id: str,
    *,
    tables: frozenset[str] | None = None,
) -> int:
    """Pull one curated build manifest and its table sha files.

    Args:
        data_dir: Local data root.
        remote: Data-bucket remote.
        build_id: Curated build id.
        tables: Optional allow-list of table names. ``None`` pulls every table
            named in the manifest (incremental curate / #345). Official UI
            pulls pass the UI subset.

    Returns:
        Count of newly written local objects.

    Raises:
        OfficialPullError: Manifest or a required table sha is missing remotely.
    """
    builds_dest = build_manifest_path(data_dir, build_id).parent
    builds_dest.mkdir(parents=True, exist_ok=True)
    n = _safe_pull(remote, f"curated/builds/build={build_id}.json", builds_dest)
    if not build_manifest_path(data_dir, build_id).is_file():
        raise OfficialPullError(f"curated build {build_id} missing in the data bucket")
    for table, entry in load_build_manifest(data_dir, build_id).tables.items():
        if tables is not None and table not in tables:
            continue
        dest = table_sha_path(data_dir, table, entry.sha).parent
        dest.mkdir(parents=True, exist_ok=True)
        path = table_sha_path(data_dir, table, entry.sha)
        before = path.is_file()
        _safe_pull(remote, f"curated/tables/{table}/sha={entry.sha}.parquet", dest)
        if not path.is_file():
            raise OfficialPullError(
                f"curated table {table} sha={entry.sha} missing for build {build_id}"
            )
        if not before:
            n += 1
    return n


def _load_pointers(
    gw_dir: Path, *, checkpoint: str, line: str | None
) -> list[OfficialPointer]:
    cp_dir = gw_dir / f"checkpoint={checkpoint}"
    if not cp_dir.is_dir():
        return []
    paths = (
        [cp_dir / f"line={line}.json"]
        if line is not None
        else sorted(cp_dir.glob("line=*.json"))
    )
    return [
        OfficialPointer.model_validate_json(p.read_text(encoding="utf-8"))
        for p in paths
        if p.is_file()
    ]
