"""Ops-tick / ops-catchup helpers on top of ``fplt.utils.timing``.

One timing session per ops run. ``JOB_STARTED_AT`` (epoch seconds, exported by
the GHA workflow's first step) becomes the ``setup`` phase. Nested library
phases (``remote/pull``, ``ingest``, ``plan/…``) accumulate into the same
session. Official ``ops_timing.json`` is write-once and only written after a
successful checkpoint finalize — failures keep a local ``timing/`` artifact.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

from fplt.automation.bench import ops_budget_status
from fplt.utils.timing import (
    SHARED_SUBPHASES,
    TimingSession,
    build_timing_record,
    ops_phase_total,
    ops_timing_path,
    write_timing_record,
)

logger = logging.getLogger(__name__)

_JOB_STARTED_AT = "JOB_STARTED_AT"
_STEP_SUMMARY = "GITHUB_STEP_SUMMARY"
_OPS_TIMEOUT_MINUTES = "OPS_TIMEOUT_MINUTES"


def record_job_setup(
    session: TimingSession,
    *,
    environ: Mapping[str, str] | None = None,
    now_epoch: float | None = None,
) -> None:
    """Record ``setup`` from ``JOB_STARTED_AT`` when present (skip if unset).

    Args:
        session: Active ops timing bag.
        environ: Environment mapping; defaults to ``os.environ``.
        now_epoch: Override for ``time.time()`` (tests).
    """
    env = os.environ if environ is None else environ
    raw = env.get(_JOB_STARTED_AT)
    if raw is None or raw.strip() == "":
        return
    try:
        started = float(raw)
    except ValueError:
        logger.warning("ignoring invalid %s=%r", _JOB_STARTED_AT, raw)
        return
    end = time.time() if now_epoch is None else now_epoch
    session.add("setup", end - started)


def format_ops_step_summary(
    record: Mapping[str, Any],
    *,
    timeout_minutes: int | None = None,
) -> str:
    """Render a markdown table of ops top-level phases for Actions summaries.

    When ``total_s`` exceeds the configured share of the workflow timeout
    (``configs/ops/bench.yaml`` / ``OPS_TIMEOUT_MINUTES``), appends a
    **WARNING** line so operators see budget pressure before a deadline run
    times out (#312).
    """
    phases_obj: object = record.get("phases", {})
    phases: dict[str, float] = {}
    if isinstance(phases_obj, dict):
        raw_phases = cast(dict[str, Any], phases_obj)
        for name, value in raw_phases.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            seconds = float(value)
            if seconds <= 0:
                continue
            phases[str(name)] = seconds
    top = {
        name: seconds
        for name, seconds in phases.items()
        if name not in SHARED_SUBPHASES
    }
    ranked = sorted(top.items(), key=lambda item: item[1], reverse=True)
    lines = [
        "### Ops timing",
        "",
        "| Phase | Seconds |",
        "| --- | ---: |",
    ]
    for name, seconds in ranked:
        lines.append(f"| `{name}` | {seconds:.1f} |")
    total = float(record.get("total_s", ops_phase_total(phases)) or 0.0)
    lines.append(f"| **total_s** | {total:.1f} |")
    lines.append("")
    minutes = timeout_minutes
    if minutes is None:
        raw_timeout = os.environ.get(_OPS_TIMEOUT_MINUTES)
        if raw_timeout is not None and raw_timeout.strip() != "":
            try:
                minutes = int(raw_timeout)
            except ValueError:
                logger.warning(
                    "ignoring invalid %s=%r",
                    _OPS_TIMEOUT_MINUTES,
                    raw_timeout,
                )
                minutes = None
    status = ops_budget_status(total, timeout_minutes=minutes)
    warning = status.warning_line
    if warning is not None:
        lines.append(warning)
        lines.append("")
        logger.warning("%s", warning.replace("**", ""))
    return "\n".join(lines)


def append_github_step_summary(record: Mapping[str, Any]) -> Path | None:
    """Append the ops timing table to ``$GITHUB_STEP_SUMMARY`` when set."""
    summary = os.environ.get(_STEP_SUMMARY)
    if not summary:
        return None
    path = Path(summary)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(format_ops_step_summary(record))
    return path


def finalize_ops_timing(
    kind: str,
    session: TimingSession,
    data_dir: Path,
    *,
    meta: Mapping[str, Any] | None = None,
    checkpoint_dir: Path | None = None,
    succeeded: bool,
) -> Path | None:
    """Persist ops timing; official ``ops_timing.json`` only on success.

    Always writes a local ``data_dir/timing/`` artifact when the session has
    phases. When ``succeeded`` and ``checkpoint_dir`` is set, also writes
    write-once ``ops_timing.json`` under that official checkpoint folder.

    Args:
        kind: ``ops_tick`` or ``ops_catchup``.
        session: Accumulated phases.
        data_dir: Local data root for the legacy ``timing/`` layout.
        meta: Optional identity (season, gw, run ids, …).
        checkpoint_dir: Official checkpoint directory for ``ops_timing.json``.
        succeeded: When False, never write ``ops_timing.json``.

    Returns:
        Path of the local timing artifact, or ``None`` when empty.
    """
    payload_meta = dict(meta or {})
    local = write_timing_record(kind, session, meta=payload_meta, data_dir=data_dir)
    if local is None:
        return None
    record = build_timing_record(kind, session, payload_meta)
    append_github_step_summary(record)
    if succeeded and checkpoint_dir is not None:
        write_timing_record(
            kind,
            session,
            meta=payload_meta,
            path=ops_timing_path(checkpoint_dir),
        )
    return local
