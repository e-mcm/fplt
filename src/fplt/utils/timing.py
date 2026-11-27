"""Wall-clock phase timings shared by walk-forward, plan, and eval code.

Records named durations into an active session, formats them for ``logging``,
and optionally persists a JSON artifact agents can aggregate into a hotspot
report. The artifact data root is ContextVar-scoped via ``timing_artifacts`` /
``@binds_timing_artifacts`` so helpers need not take a ``timing_dir`` argument.
Nested ``timed_phase`` calls (for example ``fit/score`` under ``fit``)
accumulate separately so summaries can show top-level shares plus optional
sub-breakdowns.

JSON shape (``schema_version`` 1)::

    {
      "schema_version": 1,
      "kind": "walk_forward",
      "recorded_at": "2026-11-23T12:00:00+00:00",
      "meta": {"model": "dc_layered_v3", "n_gw": 8},
      "phases": {"fit": 27.1, "predict": 297.4, "fit/score": 2.3},
      "total_s": 324.5
    }
"""

from __future__ import annotations

import functools
import inspect
import json
import logging
import re
import time
from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

from fplt.schemas.timing import TIMING_RECORD_SCHEMA_VERSION, TimingRecord

logger = logging.getLogger(__name__)

TIMING_SCHEMA_VERSION = TIMING_RECORD_SCHEMA_VERSION

# Phases that make up a gameweek / walk-forward wall-clock total and share line.
TOP_LEVEL: tuple[str, ...] = ("setup", "fit", "predict", "write")

# Kind → exclusive top-level phase names used for ``total_s`` and share lines.
KIND_TOP_LEVEL: Mapping[str, tuple[str, ...]] = {
    "walk_forward": TOP_LEVEL,
    "plan": (
        "plan/setup",
        "plan/chips",
        "optimise/milp",
        "plan/lineup",
        "plan/write",
    ),
    "decisions": ("plan/chips", "optimise/milp", "plan/lineup"),
    "eval_board": ("board/core", "board/openfpl"),
    "ingest": ("ingest",),
    "remote": ("remote/pull", "remote/push", "remote/check"),
}

# Nested names shared with walk-forward / plan; excluded from ops ``total_s``.
SHARED_SUBPHASES: frozenset[str] = frozenset(
    {
        "fit/score",
        "fit/minutes",
        "fit/involvement",
        "fit/other",
        "predict/setup",
        "predict/fixtures",
        "plan/setup",
        "plan/lineup",
        "plan/chips",
        "plan/write",
        "optimise/milp",
    }
)

OPS_TIMING_KINDS: frozenset[str] = frozenset({"ops_tick", "ops_catchup"})

# Write-once relative paths under a run folder / official checkpoint.
RUN_TIMING_RELATIVE = "logs/timing.json"
OPS_TIMING_FILENAME = "ops_timing.json"


class TimingExistsError(FileExistsError):
    """Raised when a write-once timing path already exists."""


_SAFE_META = re.compile(r"[^A-Za-z0-9._-]+")


@dataclass
class TimingSession:
    """Mutable bag of named wall-clock seconds."""

    seconds: dict[str, float] = field(default_factory=dict[str, float])

    def add(self, name: str, elapsed: float) -> None:
        """Add ``elapsed`` seconds under ``name`` (ignore non-positive)."""
        if elapsed <= 0:
            return
        self.seconds[name] = self.seconds.get(name, 0.0) + elapsed

    def merge(self, other: TimingSession) -> None:
        """Fold another session's durations into this one."""
        for name, value in other.seconds.items():
            self.add(name, value)

    def get(self, name: str) -> float:
        """Seconds recorded for ``name``, or ``0.0``."""
        return float(self.seconds.get(name, 0.0))

    def total(self, top_level: Sequence[str] | None = None) -> float:
        """Sum of top-level phases (default: walk-forward ``TOP_LEVEL``)."""
        names = TOP_LEVEL if top_level is None else top_level
        if not names:
            # Leaf phases only (no ``parent/child``) — used by eval_board.
            return sum(value for name, value in self.seconds.items() if "/" not in name)
        return sum(self.get(name) for name in names)

    def nested(self) -> Mapping[str, float]:
        """Non-top-level phases with positive duration, sorted by name."""
        return {
            name: value
            for name, value in sorted(self.seconds.items())
            if name not in TOP_LEVEL and value > 0
        }

    def phases(self) -> dict[str, float]:
        """All positive phase durations, sorted by name."""
        return {
            name: float(value)
            for name, value in sorted(self.seconds.items())
            if value > 0
        }


_active: ContextVar[TimingSession | None] = ContextVar(
    "fplt_backtest_timing", default=None
)
_artifact_root: ContextVar[Path | None] = ContextVar(
    "fplt_timing_artifacts", default=None
)
_clock: ContextVar[Callable[[], float] | None] = ContextVar(
    "fplt_timing_clock", default=None
)


def _now() -> float:
    """Return the active clock (``perf_counter`` unless a test clock is set)."""
    clock = _clock.get()
    return time.perf_counter() if clock is None else clock()


@contextmanager
def timing_clock(clock: Callable[[], float]) -> Iterator[None]:
    """Install a deterministic clock for ``timed_phase`` (tests / fake clocks).

    Args:
        clock: Zero-arg callable returning monotonic seconds.
    """
    token = _clock.set(clock)
    try:
        yield
    finally:
        _clock.reset(token)


def active_timing_session() -> TimingSession | None:
    """Return the ContextVar-scoped ``TimingSession``, if any."""
    return _active.get()


@contextmanager
def timing_session(session: TimingSession | None = None) -> Iterator[TimingSession]:
    """Install ``session`` (or a fresh one) as the active timing sink.

    Args:
        session: Existing bag to accumulate into. When omitted, a new
            ``TimingSession`` is created.

    Yields:
        The active ``TimingSession``.
    """
    active = TimingSession() if session is None else session
    token = _active.set(active)
    try:
        yield active
    finally:
        _active.reset(token)


@contextmanager
def timed_phase(name: str) -> Iterator[None]:
    """Accumulate wall time for ``name`` into the active session, if any.

    When no session is active the block still runs; nothing is recorded.
    Nested phases (for example ``fit/score`` inside ``fit``) are stored as
    separate keys so parents keep the full inclusive duration.

    Args:
        name: Phase label (top-level or nested ``parent/child``).
    """
    start = _now()
    try:
        yield
    finally:
        elapsed = _now() - start
        bucket = _active.get()
        if bucket is not None:
            bucket.add(name, elapsed)


def active_timing_dir() -> Path | None:
    """Return the ContextVar-scoped data root for timing JSON, if any."""
    return _artifact_root.get()


@contextmanager
def timing_artifacts(data_dir: Path) -> Iterator[Path]:
    """Install ``data_dir`` as the default root for ``write_timing_record``.

    Nested entry points push/pop via ContextVar so library helpers (for
    example ``walk_forward``) need not take a ``timing_dir`` argument.

    Args:
        data_dir: Install root (same as ``Settings.data_dir``). Artifacts
            land under ``data_dir/timing/``.

    Yields:
        The normalised ``data_dir`` path.
    """
    root = Path(data_dir)
    token = _artifact_root.set(root)
    try:
        yield root
    finally:
        _artifact_root.reset(token)


def binds_timing_artifacts[**P, R](
    param: str = "data_dir",
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator: wrap a call in ``timing_artifacts`` from ``param``.

    Reads the wrapped function's ``param`` (default ``data_dir``) and
    installs it for the duration of the call. When the bound value is
    ``None``, the wrap is a no-op.

    Example::

        @binds_timing_artifacts()
        def run_backtest(data_dir: Path, ...) -> list[ModelSpec]:
            ...
    """

    def decorator(fn: Callable[P, R]) -> Callable[P, R]:
        signature = inspect.signature(fn)
        if param not in signature.parameters:
            raise TypeError(f"{fn.__qualname__} has no parameter {param!r}")

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            bound = signature.bind_partial(*args, **kwargs)
            bound.apply_defaults()
            raw = bound.arguments.get(param)
            if raw is None:
                return fn(*args, **kwargs)
            with timing_artifacts(Path(raw)):
                return fn(*args, **kwargs)

        return wrapper

    return decorator


def timing_root(data_dir: Path) -> Path:
    """Return ``data_dir/timing`` where JSON timing artifacts are stored."""
    return Path(data_dir) / "timing"


def run_timing_path(run_dir: Path) -> Path:
    """Return ``run_dir/logs/timing.json`` (data model §3.3)."""
    return Path(run_dir) / RUN_TIMING_RELATIVE


def ops_timing_path(checkpoint_dir: Path) -> Path:
    """Return ``checkpoint_dir/ops_timing.json`` under an official checkpoint."""
    return Path(checkpoint_dir) / OPS_TIMING_FILENAME


def ops_phase_total(phases: Mapping[str, float]) -> float:
    """Sum ops top-level phases, excluding shared nested sub-phase names."""
    return sum(
        seconds
        for name, seconds in phases.items()
        if name not in SHARED_SUBPHASES and seconds > 0
    )


def build_timing_record(
    kind: str,
    session: TimingSession,
    meta: Mapping[str, Any] | None = None,
    *,
    recorded_at: datetime | None = None,
) -> dict[str, Any]:
    """Build a schema_version=1 timing dict agents can parse.

    Args:
        kind: Surface label (``walk_forward``, ``plan``, ``decisions``, …).
        session: Accumulated phase durations.
        meta: Optional run identity (model, season, n_gw, …).
        recorded_at: UTC timestamp; defaults to now.

    Returns:
        A JSON-serialisable timing record validated by ``TimingRecord``.
    """
    top = KIND_TOP_LEVEL.get(kind)
    stamp = recorded_at if recorded_at is not None else datetime.now(UTC)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    stamp = stamp.astimezone(UTC)
    phases = session.phases()
    if kind in OPS_TIMING_KINDS:
        total = ops_phase_total(phases)
    elif top is None:
        total = sum(phases.values())
    else:
        total = session.total(top)
    record = TimingRecord(
        schema_version=TIMING_SCHEMA_VERSION,
        kind=kind,
        recorded_at=stamp,
        meta=dict(meta or {}),
        phases=phases,
        total_s=round(total, 6),
    )
    return record.model_dump_json_doc()


def write_timing_record(
    kind: str,
    session: TimingSession,
    meta: Mapping[str, Any] | None = None,
    *,
    data_dir: Path | None = None,
    path: Path | None = None,
) -> Path | None:
    """Persist a timing JSON (never overwrite).

    Empty sessions are skipped (returns ``None``). When ``path`` is set, writes
    that exact file once (run-folder ``logs/timing.json`` or official
    ``ops_timing.json``). Otherwise writes under ``…/timing/`` with a
    kind/model/timestamp filename. When ``data_dir`` is omitted for the
    legacy layout, uses the active ``timing_artifacts`` ContextVar root; if
    neither root nor ``path`` is set, returns ``None`` without writing.

    Args:
        kind: Surface label.
        session: Accumulated phases.
        meta: Optional run identity.
        data_dir: Install root override. Defaults to ``active_timing_dir()``.
        path: Exact write-once destination. Mutually exclusive with relying on
            a ``data_dir`` / ContextVar root for the legacy ``timing/`` layout.

    Returns:
        Path written, or ``None`` when nothing was recorded / no root set.

    Raises:
        TimingExistsError: ``path`` already exists (write-once).
        ValueError: Both ``path`` and ``data_dir`` are set.
    """
    if path is not None and data_dir is not None:
        raise ValueError("pass path or data_dir, not both")
    if not session.phases():
        return None
    record = build_timing_record(kind, session, meta)
    if path is not None:
        dest = Path(path)
        if dest.exists():
            raise TimingExistsError(f"timing path already exists: {dest}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            json.dumps(record, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        logger.info(
            "timing artifact kind=%s path=%s total=%.1fs phases=%s",
            kind,
            dest,
            float(record["total_s"]),
            len(record["phases"]),
        )
        return dest
    root_dir = data_dir if data_dir is not None else active_timing_dir()
    if root_dir is None:
        return None
    root = timing_root(root_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    model = meta.get("model") if meta else None
    slug = _SAFE_META.sub("-", str(model))[:48] if model else "run"
    dest = root / f"{kind}_{slug}_{stamp}.json"
    dest.write_text(
        json.dumps(record, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    logger.info(
        "timing artifact kind=%s path=%s total=%.1fs phases=%s",
        kind,
        dest,
        float(record["total_s"]),
        len(record["phases"]),
    )
    return dest


def format_gw_breakdown(session: TimingSession, *, digits: int = 1) -> str:
    """Compact ``fit=1.2s predict=3.4s write=0.1s`` for a single gameweek.

    Args:
        session: Per-gameweek timings.
        digits: Decimal places for seconds.

    Returns:
        Space-separated ``name=…s`` pairs for top-level phases. ``setup`` is
        omitted when it recorded no time; ``fit`` / ``predict`` / ``write``
        always appear.
    """
    parts: list[str] = []
    for name in TOP_LEVEL:
        value = session.get(name)
        if name == "setup" and value <= 0:
            continue
        parts.append(f"{name}={value:.{digits}f}s")
    return " ".join(parts)


def format_run_summary(
    model_name: str,
    session: TimingSession,
    *,
    n_gw: int,
    digits: int = 1,
) -> str:
    """One log line: run totals and shares for fit vs predict (+ subphases).

    Args:
        model_name: Spec name for the walk-forward run.
        session: Aggregated timings across gameweeks.
        n_gw: Number of as-of deadlines walked.
        digits: Decimal places for seconds.

    Returns:
        A single ``walk_forward timing …`` line suitable for ``logger.info``.
    """
    total = session.total()
    bits = [
        f"walk_forward timing model={model_name}",
        f"gws={n_gw}",
        f"total={total:.{digits}f}s",
    ]
    for name in TOP_LEVEL:
        value = session.get(name)
        if name == "setup" and value <= 0:
            continue
        share = 0.0 if total <= 0 else 100.0 * value / total
        bits.append(f"{name}={value:.{digits}f}s ({share:.0f}%)")
    nested = session.nested()
    if nested:
        nested_bits = " ".join(
            f"{name}={value:.{digits}f}s" for name, value in nested.items()
        )
        bits.append(f"| {nested_bits}")
    return " ".join(bits)


def format_kind_summary(
    kind: str,
    session: TimingSession,
    *,
    digits: int = 1,
    meta: Mapping[str, Any] | None = None,
) -> str:
    """One log line for a non-walk-forward timing surface.

    Args:
        kind: Surface label.
        session: Accumulated phases.
        digits: Decimal places for seconds.
        meta: Optional identity fields appended as ``k=v``.

    Returns:
        A ``timing kind=… total=…`` line suitable for ``logger.info``.
    """
    top = KIND_TOP_LEVEL.get(kind)
    total = session.total(top) if top is not None else session.total(())
    bits = [f"timing kind={kind}", f"total={total:.{digits}f}s"]
    if meta:
        for key in ("model", "season", "n_gw", "horizon"):
            if key in meta and meta[key] is not None:
                bits.append(f"{key}={meta[key]}")
    phases = session.phases()
    ranked = sorted(phases.items(), key=lambda item: item[1], reverse=True)
    for name, value in ranked:
        share = 0.0 if total <= 0 else 100.0 * value / total
        bits.append(f"{name}={value:.{digits}f}s ({share:.0f}%)")
    return " ".join(bits)


@dataclass(frozen=True)
class Hotspot:
    """One ranked phase across one or more timing records."""

    phase: str
    seconds: float
    share_pct: float
    category: str


def _format_remote_meta(meta: Mapping[str, Any]) -> str:
    """Compact ``op=pull prefix=raw objects=3 bytes=1200`` for hotspot briefs."""
    bits: list[str] = []
    for key in ("op", "prefix", "bucket_kind", "n_objects", "bytes"):
        if key not in meta or meta[key] is None:
            continue
        label = "objects" if key == "n_objects" else key
        bits.append(f"{label}={meta[key]}")
    return " ".join(bits)


def classify_phase(phase: str) -> str:
    """Map a phase name to a coarse hotspot category.

    Categories feed the perf-report skill / port-to-cpp pipeline:

    - ``python_loop`` — tight numeric / MC loops (C++ candidates)
    - ``solver`` — HiGHS / MILP wall time
    - ``fit`` — model fitting (usually polars / scipy, not rewrite-first)
    - ``io`` — ingest / write / setup I/O
    - ``other`` — everything else
    """
    lower = phase.lower()
    if (
        "fixture" in lower
        or lower.endswith("/mc")
        or "simulate" in lower
        or lower == "predict"
    ):
        return "python_loop"
    if "milp" in lower or "solver" in lower or lower.startswith("optimise"):
        return "solver"
    if lower == "fit" or lower.startswith("fit/"):
        return "fit"
    if (
        lower in {"setup", "write", "ingest"}
        or lower.endswith("/setup")
        or lower.endswith("/write")
        or lower.startswith("ingest")
        or lower.startswith("remote/")
        or lower == "remote"
    ):
        return "io"
    if lower.startswith("plan/chips") or "chip" in lower:
        return "other"
    if lower.startswith("model/"):
        return "other"
    return "other"


def _discover_timing_paths(data_dir: Path) -> list[Path]:
    """Collect legacy ``timing/*.json``, run ``logs/timing.json``, and ops files."""
    root = Path(data_dir)
    found: list[Path] = []
    legacy = timing_root(root)
    if legacy.is_dir():
        found.extend(legacy.glob("*.json"))
    runs = root / "runs"
    if runs.is_dir():
        found.extend(runs.glob("run=*/logs/timing.json"))
    official = root / "official"
    if official.is_dir():
        found.extend(official.glob("season=*/gw=*/checkpoint=*/ops_timing.json"))
    return found


def _load_timing_file(path: Path) -> dict[str, Any] | None:
    """Parse one timing JSON file; return ``None`` when unreadable."""
    try:
        raw: object = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        logger.warning("skip unreadable timing artifact path=%s", path)
        return None
    if not isinstance(raw, dict):
        return None
    record = cast(dict[str, Any], raw)
    record = dict(record)
    record["_path"] = str(path)
    return record


def load_timing_records(
    data_dir: Path,
    *,
    kind: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load timing JSON artifacts newest-first.

    Reads ``data_dir/timing/*.json`` plus run-folder ``logs/timing.json`` and
    official ``ops_timing.json`` so ``fplt perf-report`` aggregates walk-forward
    and ops records together.

    Args:
        data_dir: Install root containing ``timing/`` and/or ``runs/`` /
            ``official/``.
        kind: Optional kind filter.
        limit: Max records to return (after filtering).

    Returns:
        Parsed records, newest ``recorded_at`` / mtime first.
    """
    paths = sorted(
        _discover_timing_paths(data_dir),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    records: list[dict[str, Any]] = []
    for path in paths:
        record = _load_timing_file(path)
        if record is None:
            continue
        if kind is not None and record.get("kind") != kind:
            continue
        records.append(record)
        if limit is not None and len(records) >= limit:
            break
    return records


def rank_hotspots(
    records: Sequence[Mapping[str, Any]],
    *,
    min_share_pct: float = 0.0,
) -> list[Hotspot]:
    """Aggregate phase seconds across ``records`` and rank by absolute time.

    Nested phases (names containing ``/``) and leaf phases are ranked
    independently; callers usually care about the largest absolute buckets.
    Share is versus the sum of aggregated phase seconds (not parent-exclusive),
    so nested + parent can exceed 100% — prefer nested leaves for rewrite
    targeting when both appear.

    Args:
        records: Timing dicts with a ``phases`` map.
        min_share_pct: Drop phases below this share of the aggregated sum.

    Returns:
        Hotspots sorted by seconds descending.
    """
    totals: dict[str, float] = {}
    for record in records:
        phases_obj: object = record.get("phases")
        if not isinstance(phases_obj, dict):
            continue
        phases_map = cast(dict[str, Any], phases_obj)
        for name, raw_value in phases_map.items():
            if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
                continue
            seconds = float(raw_value)
            if seconds <= 0:
                continue
            totals[str(name)] = totals.get(str(name), 0.0) + seconds
    grand = sum(totals.values())
    ranked = [
        Hotspot(
            phase=name,
            seconds=seconds,
            share_pct=0.0 if grand <= 0 else 100.0 * seconds / grand,
            category=classify_phase(name),
        )
        for name, seconds in totals.items()
    ]
    ranked.sort(key=lambda item: item.seconds, reverse=True)
    if min_share_pct > 0:
        ranked = [item for item in ranked if item.share_pct >= min_share_pct]
    return ranked


def format_hotspot_report(
    records: Sequence[Mapping[str, Any]],
    *,
    digits: int = 1,
    limit: int = 20,
) -> str:
    """Render a plaintext hotspot brief from timing records.

    Args:
        records: Newest-first timing dicts.
        digits: Decimal places for seconds.
        limit: Max hotspot rows.

    Returns:
        Multi-line report suitable for stdout or a skill brief.
    """
    if not records:
        return "No timing artifacts found."
    lines = [
        f"perf report: {len(records)} timing record(s)",
    ]
    for record in records[:5]:
        kind = record.get("kind", "?")
        total_obj = record.get("total_s", 0.0)
        total = float(total_obj) if isinstance(total_obj, (int, float)) else 0.0
        meta_obj: object = record.get("meta")
        model = ""
        remote_bits = ""
        if isinstance(meta_obj, dict):
            meta_map = cast(dict[str, Any], meta_obj)
            raw_model: object = meta_map.get("model", "")
            model = str(raw_model) if raw_model is not None else ""
            if kind == "remote":
                remote_bits = _format_remote_meta(meta_map)
        path_obj: object = record.get("_path", "")
        path = str(path_obj or "")
        lines.append(
            f"  - kind={kind} total={total:.{digits}f}s"
            + (f" model={model}" if model else "")
            + (f" {remote_bits}" if remote_bits else "")
            + (f" path={path}" if path else "")
        )
    hotspots = rank_hotspots(records)
    lines.append("hotspots (absolute + % of aggregated phase seconds):")
    if not hotspots:
        lines.append("  (none)")
        return "\n".join(lines)
    for index, spot in enumerate(hotspots[:limit], start=1):
        lines.append(
            f"  {index}. {spot.phase}={spot.seconds:.{digits}f}s "
            f"({spot.share_pct:.0f}%) [{spot.category}]"
        )
    return "\n".join(lines)
