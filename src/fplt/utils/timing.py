"""Wall-clock phase timings shared by walk-forward, plan, and eval code.

Records named durations into an active session, formats them for ``logging``,
and optionally persists a JSON artifact agents can aggregate into a hotspot
report. Nested ``timed_phase`` calls (for example ``fit/score`` under ``fit``)
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

import json
import logging
import re
import time
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

TIMING_SCHEMA_VERSION = 1

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
}

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
    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        bucket = _active.get()
        if bucket is not None:
            bucket.add(name, elapsed)


def timing_root(data_dir: Path) -> Path:
    """Return ``data_dir/timing`` where JSON timing artifacts are stored."""
    return Path(data_dir) / "timing"


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
        A JSON-serialisable timing record.
    """
    top = KIND_TOP_LEVEL.get(kind)
    stamp = recorded_at if recorded_at is not None else datetime.now(UTC)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    phases = session.phases()
    if top is None:
        total = sum(phases.values())
    else:
        total = session.total(top)
    return {
        "schema_version": TIMING_SCHEMA_VERSION,
        "kind": kind,
        "recorded_at": stamp.astimezone(UTC).isoformat().replace("+00:00", "Z"),
        "meta": dict(meta or {}),
        "phases": phases,
        "total_s": round(total, 6),
    }


def write_timing_record(
    data_dir: Path,
    kind: str,
    session: TimingSession,
    meta: Mapping[str, Any] | None = None,
) -> Path | None:
    """Persist a timing JSON under ``data_dir/timing/`` (never overwrite).

    Empty sessions are skipped (returns ``None``). Filenames include kind,
    optional model slug, and a UTC timestamp so repeats do not collide.

    Args:
        data_dir: Install root (same as ``Settings.data_dir``).
        kind: Surface label.
        session: Accumulated phases.
        meta: Optional run identity.

    Returns:
        Path written, or ``None`` when nothing was recorded.
    """
    if not session.phases():
        return None
    record = build_timing_record(kind, session, meta)
    root = timing_root(data_dir)
    root.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
    model = meta.get("model") if meta else None
    slug = _SAFE_META.sub("-", str(model))[:48] if model else "run"
    path = root / f"{kind}_{slug}_{stamp}.json"
    payload = json.dumps(record, indent=2, sort_keys=True) + "\n"
    path.write_text(payload, encoding="utf-8")
    logger.info(
        "timing artifact kind=%s path=%s total=%.1fs phases=%s",
        kind,
        path,
        float(record["total_s"]),
        len(record["phases"]),
    )
    return path


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
    ):
        return "io"
    if lower.startswith("plan/chips") or "chip" in lower:
        return "other"
    if lower.startswith("model/"):
        return "other"
    return "other"


def load_timing_records(
    data_dir: Path,
    *,
    kind: str | None = None,
    limit: int | None = None,
) -> list[dict[str, Any]]:
    """Load timing JSON artifacts newest-first.

    Args:
        data_dir: Install root containing ``timing/``.
        kind: Optional kind filter.
        limit: Max records to return (after filtering).

    Returns:
        Parsed records, newest ``recorded_at`` / mtime first.
    """
    root = timing_root(data_dir)
    if not root.exists():
        return []
    paths = sorted(
        root.glob("*.json"),
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    records: list[dict[str, Any]] = []
    for path in paths:
        try:
            raw: object = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            logger.warning("skip unreadable timing artifact path=%s", path)
            continue
        if not isinstance(raw, dict):
            continue
        record = cast(dict[str, Any], raw)
        if kind is not None and record.get("kind") != kind:
            continue
        record = dict(record)
        record["_path"] = str(path)
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
        if isinstance(meta_obj, dict):
            meta_map = cast(dict[str, Any], meta_obj)
            raw_model: object = meta_map.get("model", "")
            model = str(raw_model) if raw_model is not None else ""
        path_obj: object = record.get("_path", "")
        path = str(path_obj or "")
        lines.append(
            f"  - kind={kind} total={total:.{digits}f}s"
            + (f" model={model}" if model else "")
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
