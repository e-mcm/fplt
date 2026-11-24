"""Wrap pre-run local outputs as ``kind=legacy`` runs (Stage B10).

Scans ``predictions/``, ``plans/``, ``ledger/``, ``artifacts/``, and common
eval output trees under ``$FPLT_DATA_DIR``, then writes immutable run folders
via :class:`~fplt.store.runs.RunWriter`. Manifests are best-effort
(``code_sha`` / ``spec_hash`` / ``lockfile_hash`` = ``\"unknown\"``) with
``run_id = legacy-<model_id>-<as_of>``. Idempotent: existing run ids are
skipped. Push to ``fplt-lab`` is opt-in (``push=True``).
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from fplt.backtest.ledger import align_ledger_columns, ledger_path
from fplt.data.store import DataStore
from fplt.schemas.manifests import AsOfDeadline, RunMode, RunType
from fplt.store.runs import RunExistsError, RunWriter, run_dir
from fplt.types import GWKey

logger = logging.getLogger(__name__)

UNKNOWN = "unknown"
_EPOCH = datetime(1970, 1, 1, tzinfo=UTC)
_AS_OF_RE = re.compile(
    r"^as_of=(?P<season>\d{4}-\d{2})_gw(?P<gw>\d{2})(?:_h(?P<horizon>\d{2}))?$"
)
_EVAL_ROOTS = (
    "compare",
    "promote_eval",
    "bakeoff",
    "component_eval",
    "horizon_skill",
    "benchmark_projections",
    "archive_compare",
)


@dataclass(frozen=True)
class ParsedAsOf:
    """Parsed ``as_of=…`` directory stamp."""

    season: str
    gw: int
    horizon: int | None
    stamp: str


@dataclass
class LegacyImportResult:
    """Counts from one ``import_legacy`` pass."""

    imported: list[str] = field(default_factory=list[str])
    skipped: list[str] = field(default_factory=list[str])

    @property
    def n_imported(self) -> int:
        """Number of newly written runs."""
        return len(self.imported)

    @property
    def n_skipped(self) -> int:
        """Number of run ids that already existed."""
        return len(self.skipped)


def parse_as_of_dir(name: str) -> ParsedAsOf | None:
    """Parse ``as_of=<season>_gwNN`` or ``…_hHH`` directory names.

    Args:
        name: Directory basename (with or without a trailing slash).

    Returns:
        Parsed fields, or ``None`` when the name does not match.
    """
    match = _AS_OF_RE.match(name.strip().rstrip("/"))
    if match is None:
        return None
    horizon_raw = match.group("horizon")
    stamp = name.removeprefix("as_of=")
    return ParsedAsOf(
        season=match.group("season"),
        gw=int(match.group("gw")),
        horizon=int(horizon_raw) if horizon_raw is not None else None,
        stamp=stamp,
    )


def legacy_run_id(model_id: str, as_of_stamp: str) -> str:
    """Build ``legacy-<model_id>-<as_of>`` (as_of without the ``as_of=`` prefix)."""
    stamp = as_of_stamp.removeprefix("as_of=")
    return f"legacy-{model_id}-{stamp}"


def import_legacy(
    data_dir: Path,
    *,
    dry_run: bool = False,
    push: bool = False,
) -> LegacyImportResult:
    """Import local migration trees as ``kind=legacy`` runs.

    Args:
        data_dir: Local data root (``$FPLT_DATA_DIR``).
        dry_run: Discover and count only; write nothing.
        push: After each successful import, call ``maybe_push_run`` (lab).

    Returns:
        Imported and skipped ``run_id`` lists.
    """
    result = LegacyImportResult()
    for item in _discover(data_dir):
        if run_dir(data_dir, item.run_id).exists():
            result.skipped.append(item.run_id)
            continue
        if dry_run:
            result.imported.append(item.run_id)
            continue
        try:
            _write_item(data_dir, item)
        except RunExistsError:
            result.skipped.append(item.run_id)
            continue
        result.imported.append(item.run_id)
        if push:
            from fplt.store.push import maybe_push_run

            maybe_push_run(data_dir, item.run_id)
    logger.info(
        "legacy import: imported=%d skipped=%d dry_run=%s",
        result.n_imported,
        result.n_skipped,
        dry_run,
    )
    return result


@dataclass(frozen=True)
class _LegacyItem:
    """One candidate legacy run to materialise."""

    run_id: str
    model_id: str
    run_type: RunType
    mode: RunMode
    as_of: AsOfDeadline
    horizon: int
    plan_horizon: int | None
    purpose: str
    source_tag: str
    parquet_dest: str | None
    parquet_source: Path | None
    copy_files: tuple[tuple[str, Path], ...]
    code_sha: str


def _discover(data_dir: Path) -> list[_LegacyItem]:
    items: list[_LegacyItem] = []
    seen: set[str] = set()

    def _add(item: _LegacyItem) -> None:
        if item.run_id in seen:
            return
        seen.add(item.run_id)
        items.append(item)

    for item in _from_predictions(data_dir):
        _add(item)
    for item in _from_plans(data_dir):
        _add(item)
    for item in _from_ledger(data_dir, seen):
        _add(item)
    for item in _from_artifacts(data_dir, seen):
        _add(item)
    for item in _from_eval_dirs(data_dir):
        _add(item)
    return items


def _from_predictions(data_dir: Path) -> list[_LegacyItem]:
    root = data_dir / "predictions"
    if not root.is_dir():
        return []
    items: list[_LegacyItem] = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        model_id = model_dir.name
        for as_dir in sorted(model_dir.glob("as_of=*")):
            if not as_dir.is_dir():
                continue
            parsed = parse_as_of_dir(as_dir.name)
            part = as_dir / "part.parquet"
            if parsed is None or not part.is_file():
                continue
            copies = _artifact_copies(data_dir, model_id, parsed.stamp)
            code_sha = _code_sha_sidecar(as_dir) or _code_sha_sidecar(
                data_dir / "artifacts" / model_id / as_dir.name
            )
            items.append(
                _item_predict(
                    data_dir,
                    model_id=model_id,
                    parsed=parsed,
                    part=part,
                    copies=copies,
                    code_sha=code_sha,
                    source_tag="predictions",
                    purpose=f"legacy predict {model_id} {parsed.stamp}",
                )
            )
    return items


def _from_plans(data_dir: Path) -> list[_LegacyItem]:
    root = data_dir / "plans"
    if not root.is_dir():
        return []
    items: list[_LegacyItem] = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        if model_dir.name == "superseded":
            continue
        model_id = model_dir.name
        for as_dir in sorted(model_dir.glob("as_of=*")):
            if not as_dir.is_dir():
                continue
            parsed = parse_as_of_dir(as_dir.name)
            part = as_dir / "part.parquet"
            if parsed is None or parsed.horizon is None or not part.is_file():
                continue
            horizon = parsed.horizon
            items.append(
                _LegacyItem(
                    run_id=legacy_run_id(model_id, parsed.stamp),
                    model_id=model_id,
                    run_type="plan",
                    mode="live",
                    as_of=_deadline(data_dir, parsed.season, parsed.gw, part),
                    horizon=horizon,
                    plan_horizon=horizon,
                    purpose=f"legacy plan {model_id} {parsed.stamp}",
                    source_tag="plans",
                    parquet_dest="plans/plan.parquet",
                    parquet_source=part,
                    copy_files=(),
                    code_sha=UNKNOWN,
                )
            )
    return items


def _from_ledger(data_dir: Path, seen: set[str]) -> list[_LegacyItem]:
    path = ledger_path(data_dir)
    if not path.is_file():
        return []
    frame = align_ledger_columns(pl.read_parquet(path))
    if frame.is_empty():
        return []
    items: list[_LegacyItem] = []
    keys = (
        frame.select(["model_id", "as_of_season", "as_of_gw", "run_kind", "horizon"])
        .unique()
        .sort(["model_id", "as_of_season", "as_of_gw"])
    )
    for row in keys.iter_rows(named=True):
        model_id = str(row["model_id"])
        season = str(row["as_of_season"])
        gw = int(row["as_of_gw"])
        stamp = f"{season}_gw{gw:02d}"
        run_id = legacy_run_id(model_id, stamp)
        if run_id in seen:
            continue
        mode: RunMode = "backtest" if str(row["run_kind"]) == "backtest" else "live"
        horizon = max(int(row["horizon"]), 1)
        parsed = ParsedAsOf(season=season, gw=gw, horizon=None, stamp=stamp)
        items.append(
            _item_predict(
                data_dir,
                model_id=model_id,
                parsed=parsed,
                part=path,
                copies=_artifact_copies(data_dir, model_id, stamp),
                code_sha=UNKNOWN,
                source_tag="ledger",
                purpose=f"legacy ledger {model_id} {stamp}",
                mode=mode,
                horizon=horizon,
            )
        )
    return items


def _from_artifacts(data_dir: Path, seen: set[str]) -> list[_LegacyItem]:
    root = data_dir / "artifacts"
    if not root.is_dir():
        return []
    items: list[_LegacyItem] = []
    for model_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        model_id = model_dir.name
        for as_dir in sorted(model_dir.glob("as_of=*")):
            if not as_dir.is_dir():
                continue
            parsed = parse_as_of_dir(as_dir.name)
            if parsed is None:
                continue
            run_id = legacy_run_id(model_id, parsed.stamp)
            if run_id in seen:
                continue
            copies = tuple(
                (f"artifacts/{path.name}", path)
                for path in sorted(as_dir.iterdir())
                if path.is_file()
            )
            if not copies:
                continue
            items.append(
                _LegacyItem(
                    run_id=run_id,
                    model_id=model_id,
                    run_type="predict",
                    mode="live",
                    as_of=_deadline(data_dir, parsed.season, parsed.gw, copies[0][1]),
                    horizon=1,
                    plan_horizon=None,
                    purpose=f"legacy artifacts {model_id} {parsed.stamp}",
                    source_tag="artifacts",
                    parquet_dest=None,
                    parquet_source=None,
                    copy_files=copies,
                    code_sha=_code_sha_sidecar(as_dir),
                )
            )
    return items


def _from_eval_dirs(data_dir: Path) -> list[_LegacyItem]:
    items: list[_LegacyItem] = []
    for root_name in _EVAL_ROOTS:
        root = data_dir / root_name
        if not root.is_dir():
            continue
        if root_name == "benchmark_projections":
            for as_dir in sorted(root.glob("as_of=*")):
                part = as_dir / "part.parquet"
                parsed = parse_as_of_dir(as_dir.name)
                if parsed is None or not part.is_file():
                    continue
                model_id = "benchmark_projections"
                items.append(
                    _item_predict(
                        data_dir,
                        model_id=model_id,
                        parsed=parsed,
                        part=part,
                        copies=(),
                        code_sha=UNKNOWN,
                        source_tag="benchmark_projections",
                        purpose=f"legacy benchmark {parsed.stamp}",
                        mode="live",
                        run_type="eval",
                    )
                )
            continue
        for leaf in _eval_leaves(root):
            rel = leaf.relative_to(data_dir).as_posix()
            stamp = rel.replace("/", "-")
            files = tuple(
                (f"eval/{path.relative_to(leaf).as_posix()}", path)
                for path in sorted(leaf.rglob("*"))
                if path.is_file()
            )
            if not files:
                continue
            season, gw = _season_gw_from_name(leaf.name)
            items.append(
                _LegacyItem(
                    run_id=legacy_run_id("eval", stamp),
                    model_id=f"eval-{root_name}",
                    run_type="eval",
                    mode="backtest",
                    as_of=_deadline(data_dir, season, gw, files[0][1]),
                    horizon=1,
                    plan_horizon=None,
                    purpose=f"legacy eval {rel}",
                    source_tag=root_name,
                    parquet_dest=None,
                    parquet_source=None,
                    copy_files=files,
                    code_sha=UNKNOWN,
                )
            )
    return items


def _item_predict(
    data_dir: Path,
    *,
    model_id: str,
    parsed: ParsedAsOf,
    part: Path,
    copies: tuple[tuple[str, Path], ...],
    code_sha: str,
    source_tag: str,
    purpose: str,
    mode: RunMode = "live",
    horizon: int | None = None,
    run_type: RunType = "predict",
) -> _LegacyItem:
    if horizon is not None:
        resolved_horizon = horizon
    elif source_tag == "ledger":
        resolved_horizon = 1
    else:
        resolved_horizon = _horizon_from_frame(pl.read_parquet(part))
    return _LegacyItem(
        run_id=legacy_run_id(model_id, parsed.stamp),
        model_id=model_id,
        run_type=run_type,
        mode=mode,
        as_of=_deadline(data_dir, parsed.season, parsed.gw, part),
        horizon=resolved_horizon,
        plan_horizon=None,
        purpose=purpose,
        source_tag=source_tag,
        parquet_dest="predictions/player_fixture.parquet",
        parquet_source=part,
        copy_files=copies,
        code_sha=code_sha,
    )


def _write_item(data_dir: Path, item: _LegacyItem) -> None:
    frame: pl.DataFrame | None = None
    if item.parquet_dest is not None and item.parquet_source is not None:
        raw = pl.read_parquet(item.parquet_source)
        if item.source_tag == "ledger":
            frame = align_ledger_columns(raw).filter(
                (pl.col("model_id") == item.model_id)
                & (pl.col("as_of_season") == item.as_of.season)
                & (pl.col("as_of_gw") == item.as_of.gw)
            )
        else:
            frame = raw
    with RunWriter(
        data_dir,
        kind="legacy",
        mode=item.mode,
        run_type=item.run_type,
        spec_hash=UNKNOWN,
        spec_label=f"legacy:{item.source_tag}",
        model_id=item.model_id,
        as_of=[item.as_of],
        horizon=item.horizon,
        plan_horizon=item.plan_horizon,
        purpose=item.purpose,
        run_id=item.run_id,
        code_sha=item.code_sha,
        code_dirty=False,
        lockfile_hash=UNKNOWN,
        host="legacy-import",
        raw_manifest_hash=UNKNOWN,
        staged_digests={},
        build_id=None,
        tags={"legacy_source": item.source_tag},
    ) as writer:
        if item.parquet_dest is not None and frame is not None:
            writer.write_parquet(item.parquet_dest, frame)
        for relative, source in item.copy_files:
            writer.copy_file(relative, source)


def _artifact_copies(
    data_dir: Path, model_id: str, stamp: str
) -> tuple[tuple[str, Path], ...]:
    folder = data_dir / "artifacts" / model_id / f"as_of={stamp}"
    if not folder.is_dir():
        return ()
    return tuple(
        (f"artifacts/{path.name}", path)
        for path in sorted(folder.iterdir())
        if path.is_file()
    )


def _code_sha_sidecar(folder: Path) -> str:
    path = folder / "git_sha.txt"
    if not path.is_file():
        path = folder / "code_sha.txt"
    if not path.is_file():
        return UNKNOWN
    value = path.read_text(encoding="utf-8").strip()
    return value or UNKNOWN


def _deadline(data_dir: Path, season: str, gw: int, source: Path) -> AsOfDeadline:
    try:
        deadline = DataStore(data_dir).deadline(GWKey(season, gw))
    except Exception:
        mtime = source.stat().st_mtime if source.exists() else 0.0
        deadline = datetime.fromtimestamp(mtime, tz=UTC) if mtime > 0 else _EPOCH
    return AsOfDeadline(season=season, gw=gw, deadline_utc=deadline)


def _horizon_from_frame(frame: pl.DataFrame) -> int:
    if "horizon" in frame.columns and frame.height > 0:
        return max(int(frame["horizon"].max()), 1)  # type: ignore[arg-type]
    if (
        "target_gw" in frame.columns
        and "as_of_gw" in frame.columns
        and frame.height > 0
    ):
        delta = (frame["target_gw"] - frame["as_of_gw"]).max()
        return max(int(delta) + 1, 1)  # type: ignore[arg-type]
    return 1


def _eval_leaves(root: Path) -> list[Path]:
    """Directories under ``root`` that contain files and no child dirs with files.

    Prefer deepest folders that hold outputs (e.g. ``compare/stamp/``,
    ``promote_eval/name/stamp/``).
    """
    leaves: list[Path] = []
    for path in sorted(root.rglob("*")):
        if not path.is_dir():
            continue
        has_files = any(child.is_file() for child in path.iterdir())
        has_subdir_files = any(
            child.is_dir() and any(g.is_file() for g in child.rglob("*"))
            for child in path.iterdir()
        )
        if has_files and not has_subdir_files:
            leaves.append(path)
    return leaves


def _season_gw_from_name(name: str) -> tuple[str, int]:
    match = re.search(r"(\d{4}-\d{2})_gw(\d{2})", name)
    if match is None:
        return "1970-71", 1
    return match.group(1), int(match.group(2))
