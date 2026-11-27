"""Ops timing session: fake clock, setup, partial vs official writes."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fplt.automation.ops_timing import (
    append_github_step_summary,
    finalize_ops_timing,
    format_ops_step_summary,
    record_job_setup,
)
from fplt.utils.timing import (
    TimingExistsError,
    TimingSession,
    ops_phase_total,
    ops_timing_path,
    timed_phase,
    timing_clock,
    timing_session,
)


class _FakeClock:
    """Monotonic clock that advances on each read or via ``advance``."""

    def __init__(self, start: float = 100.0) -> None:
        self.value = start

    def __call__(self) -> float:
        return self.value

    def advance(self, seconds: float) -> None:
        self.value += seconds


def test_fake_clock_nested_subphases_sum_le_parent() -> None:
    clock = _FakeClock()
    session = TimingSession()
    with timing_clock(clock), timing_session(session):
        clock.advance(0)  # start parent
        with timed_phase("fit/demo"):
            clock.advance(1.0)
            with timed_phase("fit/score"):
                clock.advance(2.0)
            clock.advance(1.0)
            with timed_phase("fit/minutes"):
                clock.advance(1.5)
            clock.advance(0.5)
    assert session.get("fit/demo") == pytest.approx(6.0)
    nested = session.get("fit/score") + session.get("fit/minutes")
    assert nested == pytest.approx(3.5)
    assert nested <= session.get("fit/demo")
    total = ops_phase_total(session.phases())
    assert total == pytest.approx(6.0)
    assert session.get("fit/demo") <= total + 1e-9


def test_total_s_ge_top_level_sum_and_skipped_phase_absent() -> None:
    clock = _FakeClock()
    session = TimingSession()
    with timing_clock(clock), timing_session(session):
        with timed_phase("ingest"):
            clock.advance(3.0)
        with timed_phase("remote/push"):
            clock.advance(2.0)
        # zero-length phase is absent (add ignores non-positive)
        with timed_phase("remote/pull"):
            pass
    assert "remote/pull" not in session.phases()
    record_job_setup(session, environ={}, now_epoch=0.0)
    assert "setup" not in session.phases()
    total = ops_phase_total(session.phases())
    assert total == pytest.approx(5.0)
    assert total >= session.get("ingest") + session.get("remote/push") - 1e-9


def test_record_job_setup_from_env() -> None:
    session = TimingSession()
    record_job_setup(
        session,
        environ={"JOB_STARTED_AT": "1000"},
        now_epoch=1005.5,
    )
    assert session.get("setup") == pytest.approx(5.5)


def test_failed_tick_writes_local_not_ops_timing(tmp_path: Path) -> None:
    session = TimingSession()
    session.add("ingest", 2.0)
    session.add("fit/score", 1.0)
    checkpoint = tmp_path / "official" / "season=2025-26" / "gw=1" / "checkpoint=T-90m"
    local = finalize_ops_timing(
        "ops_tick",
        session,
        tmp_path,
        meta={"models": ["demo"]},
        checkpoint_dir=checkpoint,
        succeeded=False,
    )
    assert local is not None
    assert local.parent == tmp_path / "timing"
    assert not ops_timing_path(checkpoint).exists()


def test_success_writes_ops_timing_once(tmp_path: Path) -> None:
    session = TimingSession()
    session.add("setup", 1.0)
    session.add("ingest", 4.0)
    checkpoint = tmp_path / "official" / "season=2025-26" / "gw=2" / "checkpoint=T-90m"
    local = finalize_ops_timing(
        "ops_tick",
        session,
        tmp_path,
        meta={"checkpoint": "T-90m"},
        checkpoint_dir=checkpoint,
        succeeded=True,
    )
    assert local is not None
    ops_path = ops_timing_path(checkpoint)
    assert ops_path.is_file()
    payload = json.loads(ops_path.read_text(encoding="utf-8"))
    assert payload["kind"] == "ops_tick"
    assert payload["total_s"] == pytest.approx(5.0)
    with pytest.raises(TimingExistsError):
        finalize_ops_timing(
            "ops_tick",
            session,
            tmp_path,
            checkpoint_dir=checkpoint,
            succeeded=True,
        )


def test_github_step_summary_appends_table(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    record = {
        "phases": {"ingest": 3.0, "fit/score": 1.0, "setup": 2.0},
        "total_s": 5.0,
    }
    written = append_github_step_summary(record)
    assert written == summary
    text = summary.read_text(encoding="utf-8")
    assert "### Ops timing" in text
    assert "`ingest`" in text
    assert "`setup`" in text
    assert "fit/score" not in text  # nested excluded from top-level table
    assert "**total_s**" in text
    assert format_ops_step_summary(record).startswith("### Ops timing")
