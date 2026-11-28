"""Separate healthy \"nothing new\" captures from source-unavailable runs.

Capture jobs must not look green when the upstream is down: heartbeat pings
and exit 0 would hide an outage. This helper records a consecutive-unavailable
counter under ``status/<source>_unavailable.json`` (rewritable ``status/``
prefix), emits a GitHub ``::warning::``, suppresses the heartbeat, and exits
non-zero after ``unavailable_threshold`` consecutive failures.
"""

from __future__ import annotations

import json
import logging
import os
import re
import sys
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

logger = logging.getLogger(__name__)

DEFAULT_THRESHOLD = 3
_THRESHOLD_ENV = "FPLT_CAPTURE_UNAVAILABLE_THRESHOLD"
_THRESHOLD_LINE = re.compile(r"^unavailable_threshold\s*:\s*(\d+)\s*(?:#.*)?$")


@dataclass(frozen=True)
class CaptureHealthResult:
    """Decision for one capture attempt after updating unavailable state."""

    source: str
    available: bool
    consecutive_unavailable: int
    threshold: int
    ping_heartbeat: bool
    exit_code: int
    status_path: Path
    message: str


def default_config_path() -> Path:
    """Return the repo ``configs/ops/capture_health.yaml`` path."""
    root = Path(__file__).resolve().parents[3]
    return root / "configs" / "ops" / "capture_health.yaml"


def load_unavailable_threshold(path: Path | None = None) -> int:
    """Load ``unavailable_threshold`` from YAML, env, or the default.

    Precedence: ``FPLT_CAPTURE_UNAVAILABLE_THRESHOLD`` env, then ``path``
    (default ``configs/ops/capture_health.yaml``), then ``DEFAULT_THRESHOLD``.

    Args:
        path: Optional YAML path. Missing file falls through to the default.

    Returns:
        Positive integer threshold of consecutive unavailable runs.
    """
    raw = os.environ.get(_THRESHOLD_ENV, "").strip()
    if raw:
        value = int(raw)
        if value < 1:
            raise ValueError(f"{_THRESHOLD_ENV} must be >= 1, got {value}")
        return value
    cfg = path if path is not None else default_config_path()
    if cfg.is_file():
        for line in cfg.read_text(encoding="utf-8").splitlines():
            match = _THRESHOLD_LINE.match(line.strip())
            if match is not None:
                value = int(match.group(1))
                if value < 1:
                    raise ValueError(
                        f"unavailable_threshold must be >= 1 in {cfg}, got {value}"
                    )
                return value
    return DEFAULT_THRESHOLD


def status_filename(source: str) -> str:
    """Return ``<source>_unavailable.json`` for a capture source key."""
    key = source.strip().lower().replace(" ", "_")
    if not key or "/" in key or ".." in key:
        raise ValueError(f"invalid capture source name: {source!r}")
    return f"{key}_unavailable.json"


def status_path(status_dir: Path, source: str) -> Path:
    """Return the local path for ``source``'s unavailable counter file."""
    return status_dir / status_filename(source)


def load_unavailable_state(path: Path) -> dict[str, Any] | None:
    """Load a status JSON object, or ``None`` when the file is missing.

    Args:
        path: Path to ``status/<source>_unavailable.json``.

    Returns:
        Parsed object, or ``None`` if ``path`` does not exist.

    Raises:
        ValueError: The file exists but is not a JSON object.
    """
    if not path.is_file():
        return None
    payload: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"status file must be a JSON object: {path}")
    return cast(dict[str, Any], payload)


def _write_state(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def emit_github_warning(message: str) -> None:
    """Emit a GitHub Actions warning annotation (plain text locally)."""
    # Annotations must be a single line; collapse newlines. Write to stdout
    # (not logging) so Actions parses the ``::warning::`` token.
    clean = " ".join(message.split())
    sys.stdout.write(f"::warning::{clean}\n")
    sys.stdout.flush()


def set_github_output(name: str, value: str) -> None:
    """Append ``name=value`` to ``GITHUB_OUTPUT`` when running in Actions."""
    target = os.environ.get("GITHUB_OUTPUT")
    if not target:
        return
    with Path(target).open("a", encoding="utf-8") as handle:
        handle.write(f"{name}={value}\n")


def apply_github_outputs(result: CaptureHealthResult) -> None:
    """Write ``source_ok`` / ``ping_heartbeat`` for workflow heartbeat gating."""
    ok = "true" if result.available else "false"
    ping = "true" if result.ping_heartbeat else "false"
    set_github_output("source_ok", ok)
    set_github_output("ping_heartbeat", ping)


def record_capture_outcome(
    *,
    source: str,
    status_dir: Path,
    available: bool,
    reason: str = "",
    threshold: int | None = None,
    now: datetime | None = None,
    emit_warning: bool = True,
) -> CaptureHealthResult:
    """Update unavailable state and return exit / heartbeat decisions.

    Args:
        source: Capture source key (``understat``, ``api_football``, …).
        status_dir: Local directory mirroring R2 ``status/``.
        available: True when the source responded usefully (including
            nothing-new). False when the source was unreachable / erroring.
        reason: Human-readable cause when ``available`` is false.
        threshold: Consecutive-unavailable limit before non-zero exit.
            Defaults to :func:`load_unavailable_threshold`.
        now: Clock for ``updated_at`` (UTC). Defaults to wall clock.
        emit_warning: When true, print ``::warning::`` on unavailable.

    Returns:
        :class:`CaptureHealthResult` with exit code and heartbeat flag.
    """
    limit = threshold if threshold is not None else load_unavailable_threshold()
    if limit < 1:
        raise ValueError(f"threshold must be >= 1, got {limit}")
    moment = now if now is not None else datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    else:
        moment = moment.astimezone(UTC)
    path = status_path(status_dir, source)
    stamp = moment.strftime("%Y-%m-%dT%H:%M:%SZ")

    if available:
        payload = {
            "source": source,
            "consecutive_unavailable": 0,
            "threshold": limit,
            "last_reason": "",
            "updated_at": stamp,
            "available": True,
        }
        _write_state(path, payload)
        result = CaptureHealthResult(
            source=source,
            available=True,
            consecutive_unavailable=0,
            threshold=limit,
            ping_heartbeat=True,
            exit_code=0,
            status_path=path,
            message="source available",
        )
        apply_github_outputs(result)
        return result

    prior = load_unavailable_state(path)
    previous = 0
    if prior is not None:
        raw_count = prior.get("consecutive_unavailable", 0)
        previous = int(raw_count) if raw_count is not None else 0
    consecutive = previous + 1
    detail = reason.strip() or "source unavailable"
    payload = {
        "source": source,
        "consecutive_unavailable": consecutive,
        "threshold": limit,
        "last_reason": detail,
        "updated_at": stamp,
        "available": False,
    }
    _write_state(path, payload)
    message = f"{source} unavailable ({consecutive}/{limit} consecutive): {detail}"
    if emit_warning:
        emit_github_warning(message)
    exit_code = 1 if consecutive >= limit else 0
    if exit_code != 0:
        message = (
            f"{message}; threshold reached — failing the job "
            "(see docs/ops-runbook.md#source-unavailable)"
        )
        logger.error("%s", message)
    result = CaptureHealthResult(
        source=source,
        available=False,
        consecutive_unavailable=consecutive,
        threshold=limit,
        ping_heartbeat=False,
        exit_code=exit_code,
        status_path=path,
        message=message,
    )
    apply_github_outputs(result)
    return result
