"""Wall-clock phase timings shared by walk-forward and model code.

Records named durations into an active session and formats them for the
existing ``logging`` style. This is not a metrics store — times are logged
only. Nested ``timed_phase`` calls (for example ``fit/score`` under ``fit``)
accumulate separately so summaries can show top-level shares plus optional
sub-breakdowns.
"""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field

# Phases that make up a gameweek / run wall-clock total and share line.
TOP_LEVEL: tuple[str, ...] = ("setup", "fit", "predict", "write")


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

    def total(self) -> float:
        """Sum of top-level phases (setup + fit + predict + write)."""
        return sum(self.get(name) for name in TOP_LEVEL)

    def nested(self) -> Mapping[str, float]:
        """Non-top-level phases with positive duration, sorted by name."""
        return {
            name: value
            for name, value in sorted(self.seconds.items())
            if name not in TOP_LEVEL and value > 0
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
