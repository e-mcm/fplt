"""Capture-workflow wiring for ops-06 / A10 sanity gates.

Runs bootstrap gates on new captures, appends quarantine rows, and alerts
once per new failure type (gate name not already in the quarantine table).
"""

from __future__ import annotations

import logging
import shutil
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from urllib.request import Request

import polars as pl

from fplt.automation.notify import notify
from fplt.data.gates import (
    CaptureGateReport,
    load_data_gates_config,
    quarantine_records,
    run_bootstrap_gates,
)
from fplt.data.quarantine import (
    append_quarantine,
    capture_relpath,
    load_quarantine,
    pull_quarantine,
    push_quarantine,
    quarantined_capture_paths,
)
from fplt.data.raw import RawSnapshot, iter_snapshots, read_json
from fplt.data.remote import Remote, RemoteError, pull
from fplt.settings import Settings, get_settings

logger = logging.getLogger(__name__)

_Post = Callable[[Request], object]


@dataclass(frozen=True)
class CaptureGateAction:
    """Result of gating one bootstrap capture."""

    capture_path: str
    report: CaptureGateReport
    new_failure_types: tuple[str, ...]


@dataclass(frozen=True)
class CheckCapturesResult:
    """Summary of ``check_new_captures``."""

    actions: tuple[CaptureGateAction, ...]
    alerted: tuple[str, ...]

    @property
    def failed(self) -> tuple[CaptureGateAction, ...]:
        """Actions whose gate report failed."""
        return tuple(action for action in self.actions if not action.report.passed)


def parse_since(value: str, *, now: datetime | None = None) -> datetime:
    """Parse ``--since`` as an ISO stamp or a relative duration (``7d``, ``24h``).

    Args:
        value: ISO-8601 datetime/date, or ``<n>d`` / ``<n>h`` / ``<n>m``.
        now: Clock for relative durations. Defaults to ``datetime.now(UTC)``.

    Returns:
        Timezone-aware UTC lower bound.

    Raises:
        ValueError: ``value`` is not a supported form.
    """
    text = value.strip()
    moment = datetime.now(UTC) if now is None else now.astimezone(UTC)
    if len(text) >= 2 and text[-1] in "dhm" and text[:-1].isdigit():
        amount = int(text[:-1])
        unit = text[-1]
        delta = {
            "d": timedelta(days=amount),
            "h": timedelta(hours=amount),
            "m": timedelta(minutes=amount),
        }[unit]
        return moment - delta
    normalized = text.replace("Z", "+00:00")
    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError as exc:
        raise ValueError(
            f"invalid --since {value!r}; use ISO datetime/date or Nd/Nh/Nm"
        ) from exc
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def format_gates_status(data_dir: Path, *, since: datetime) -> str:
    """Explain quarantine rows with ``available_at >= since``.

    Args:
        data_dir: Local data root holding ``quarantine/``.
        since: Inclusive lower bound on ``available_at``.

    Returns:
        Human-readable multi-line status for ``fplt data gates --since``.
    """
    bound = since.astimezone(UTC)
    frame = load_quarantine(data_dir)
    if frame.is_empty():
        return f"quarantine since {bound.isoformat()}: 0 row(s); none on disk\n"
    stamps = pl.col("available_at").dt.replace_time_zone("UTC")
    recent = frame.filter(stamps >= bound).sort(
        ["available_at", "gate", "capture_path"]
    )
    lines = [
        f"quarantine since {bound.isoformat()}: {recent.height} row(s)",
    ]
    if recent.is_empty():
        lines.append("  (no rows in window)")
        return "\n".join(lines) + "\n"
    for row in recent.iter_rows(named=True):
        stamp = cast(datetime, row["available_at"]).astimezone(UTC).isoformat()
        lines.append(f"  {stamp}  gate={row['gate']}  capture={row['capture_path']}")
        lines.append(f"    {row['detail']}")
    types = sorted(set(recent["gate"].to_list()))
    lines.append(f"failure types in window: {', '.join(types)}")
    return "\n".join(lines) + "\n"


def previous_good_snapshot(
    raw_root: Path,
    *,
    season: str,
    before: datetime,
    data_dir: Path,
) -> RawSnapshot | None:
    """Newest non-quarantined bootstrap capture for ``season`` before ``before``."""
    skipped = quarantined_capture_paths(data_dir)
    candidates = [
        snap
        for snap in iter_snapshots(raw_root)
        if snap.season == season
        and snap.captured_at < before
        and capture_relpath(raw_root, snap.capture_dir) not in skipped
    ]
    return candidates[-1] if candidates else None


def evaluate_capture(
    data_dir: Path,
    raw_root: Path,
    snap: RawSnapshot,
    *,
    available_at: datetime | None = None,
    notify_url: str | None = None,
    post: _Post | None = None,
) -> CaptureGateAction:
    """Gate one capture; quarantine failures; alert only new failure types.

    Args:
        data_dir: Local data root for quarantine persistence.
        raw_root: Directory corresponding to ``data/raw``.
        snap: Bootstrap (+ fixtures) capture to evaluate.
        available_at: Quarantine write stamp. Defaults to now (UTC).
        notify_url: Optional ntfy topic for new failure types.
        post: Optional HTTP fake for tests.

    Returns:
        Gate outcome plus any newly alerted failure types.
    """
    if available_at is None:
        written = datetime.now(UTC)
    else:
        written = available_at.astimezone(UTC)
    capture_path = capture_relpath(raw_root, snap.capture_dir)
    bootstrap = cast(dict[str, Any], read_json(snap.bootstrap_path))
    fixtures = cast(
        list[dict[str, Any]],
        read_json(snap.fixtures_path) if snap.fixtures_path else [],
    )
    previous = previous_good_snapshot(
        raw_root,
        season=snap.season,
        before=snap.captured_at,
        data_dir=data_dir,
    )
    previous_payload = (
        cast(dict[str, Any], read_json(previous.bootstrap_path))
        if previous is not None
        else None
    )
    previous_path = None
    if previous is not None:
        previous_path = capture_relpath(raw_root, previous.capture_dir)
    report = run_bootstrap_gates(
        bootstrap,
        fixtures,
        previous=previous_payload,
        config=load_data_gates_config(),
    )
    if report.passed:
        logger.info("capture gates passed for %s", capture_path)
        return CaptureGateAction(capture_path, report, ())

    existing = load_quarantine(data_dir)
    known: set[str] = set()
    if not existing.is_empty():
        known = set(cast(list[str], existing["gate"].to_list()))
    new_types = tuple(
        sorted(failure.name for failure in report.failures if failure.name not in known)
    )
    frame = quarantine_records(
        report,
        season=snap.season,
        capture_path=capture_path,
        captured_at=snap.captured_at,
        available_at=written,
        previous_capture_path=previous_path,
    )
    append_quarantine(data_dir, frame)
    if new_types and notify_url:
        detail_lines = [
            f"{failure.name}: {failure.detail}"
            for failure in report.failures
            if failure.name in new_types
        ]
        message = (
            f"capture={capture_path}\n"
            f"new failure types: {', '.join(new_types)}\n"
            + "\n".join(detail_lines)
            + "\ninspect: fplt data gates --since 7d"
        )
        notify(notify_url, "fplt capture quarantined", message, post=post)
        logger.warning(
            "quarantined %s; alerted new types=%s", capture_path, ",".join(new_types)
        )
    else:
        logger.warning("quarantined %s; no new failure types to alert", capture_path)
    return CaptureGateAction(capture_path, report, new_types)


def check_new_captures(
    data_dir: Path | None = None,
    *,
    new_raw: Path | None = None,
    settings: Settings | None = None,
    notify_alerts: bool = False,
    sync_remote: bool = True,
    available_at: datetime | None = None,
    post: _Post | None = None,
) -> CheckCapturesResult:
    """Pull quarantine, gate new bootstrap captures, push quarantine, alert.

    Args:
        data_dir: Local data root. Defaults to ``Settings.data_dir``.
        new_raw: Directory of newly written raw captures (GHA temp). When set,
            those files are copied write-once into ``data_dir/raw`` first.
        settings: Process settings (bucket + ntfy). Defaults to ``get_settings()``.
        notify_alerts: When True, POST ntfy for newly seen failure types.
        sync_remote: When True and a data bucket is configured, pull/push
            ``status/quarantine`` around the check.
        available_at: Quarantine write stamp. Defaults to now (UTC).
        post: Optional HTTP fake for tests.

    Returns:
        Per-capture actions and the union of alerted failure types.
    """
    cfg = get_settings() if settings is None else settings
    root = cfg.data_dir if data_dir is None else data_dir
    raw_root = root / "raw"
    raw_root.mkdir(parents=True, exist_ok=True)

    if sync_remote:
        pull_quarantine(root, cfg)
        _pull_fpl_api_history(raw_root, cfg)

    gated: list[RawSnapshot] = []
    if new_raw is not None:
        gated = _merge_new_raw(new_raw, raw_root)
    if not gated:
        # No explicit new tree: gate only captures not already quarantined that
        # are the latest per season (capture workflow passes --new-raw).
        skipped = quarantined_capture_paths(root)
        by_season: dict[str, RawSnapshot] = {}
        for snap in iter_snapshots(raw_root):
            rel = capture_relpath(raw_root, snap.capture_dir)
            if rel in skipped:
                continue
            by_season[snap.season] = snap
        gated = list(by_season.values())

    notify_url = cfg.ntfy_url if notify_alerts else None
    actions: list[CaptureGateAction] = []
    alerted: list[str] = []
    for snap in gated:
        action = evaluate_capture(
            root,
            raw_root,
            snap,
            available_at=available_at,
            notify_url=notify_url,
            post=post,
        )
        actions.append(action)
        alerted.extend(action.new_failure_types)

    if sync_remote:
        push_quarantine(root, cfg)

    return CheckCapturesResult(tuple(actions), tuple(sorted(set(alerted))))


def _pull_fpl_api_history(raw_root: Path, settings: Settings) -> None:
    """Pull existing ``raw/fpl_api`` for previous-good comparisons (write-once)."""
    remote = Remote.from_settings(settings)
    if remote is None:
        return
    dest = raw_root / "fpl_api"
    dest.mkdir(parents=True, exist_ok=True)
    try:
        pull(remote, "raw/fpl_api", dest)
    except RemoteError as exc:
        detail = str(exc).lower()
        if "directory not found" in detail or "not found" in detail:
            logger.info("no remote fpl_api raw yet; comparative gates may skip")
            return
        raise


def _merge_new_raw(new_raw: Path, raw_root: Path) -> list[RawSnapshot]:
    """Copy new capture files write-once into ``raw_root``; return those snaps."""
    if not new_raw.is_dir():
        return []
    before = {
        capture_relpath(raw_root, snap.capture_dir) for snap in iter_snapshots(raw_root)
    }
    for src in new_raw.rglob("*"):
        if not src.is_file():
            continue
        dest = raw_root / src.relative_to(new_raw)
        if dest.exists():
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dest)
    after = iter_snapshots(raw_root)
    return [
        snap
        for snap in after
        if capture_relpath(raw_root, snap.capture_dir) not in before
    ]
