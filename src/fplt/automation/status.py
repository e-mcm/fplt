"""Build and publish ``status.json`` for ops health (ops-10).

``fplt status --json`` and every ops tick call ``build_status``. The private
document is written under ``$FPLT_DATA_DIR/status/status.json`` and pushed to
``fplt-data/status/status.json`` with a mutable overwrite. ``public_status``
strips private fields for a later website / widget whitelist.
"""

from __future__ import annotations

import json
import logging
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from fplt.auth import status as auth_status
from fplt.automation.production import load_production_ops
from fplt.data.api_football import iter_raw_injuries
from fplt.data.football_data import iter_raw_e0
from fplt.data.quarantine import load_quarantine
from fplt.data.raw import iter_snapshots
from fplt.data.remote import Remote, RemoteError, push_replace
from fplt.data.store import DataStore
from fplt.data.understat import iter_raw_dates, iter_raw_teams
from fplt.models.spec import load_model_specs
from fplt.schemas.status import (
    HEARTBEAT_NAMES,
    PRIVATE_STATUS_FIELDS,
    STATUS_SCHEMA_VERSION,
    STATUS_SOURCES,
    CaptureSourceStatus,
    HeartbeatStamp,
    LastErrorStatus,
    NextDeadlineStatus,
    PlanDeadlineStatus,
    QuarantineStatusItem,
    StatusDocument,
    TokenExpiryStatus,
)
from fplt.settings import Settings, get_settings
from fplt.utils.process import run_command

logger = logging.getLogger(__name__)

_STATUS_DIR = "status"
_STATUS_FILE = "status.json"
_PUBLIC_FILE = "status.public.json"
_LAST_ERROR_FILE = "last_error.json"
_AS_OF_RE = re.compile(
    r"^as_of=(?P<season>\d{4}-\d{2})_gw(?P<gw>\d{2})_h(?P<horizon>\d{2})$"
)
_CAPTURE_STAMP_RE = re.compile(
    r"captured_at=(\d{4}-\d{2}-\d{2}(?:T\d{2}-\d{2}-\d{2}Z)?)"
)
_SOURCE_TO_HEARTBEAT = {
    "fpl_api": "snapshot",
    "understat": "understat",
    "api_football": "api_football",
    "football_data": "football_data",
}


def status_dir(data_dir: Path) -> Path:
    """Return ``data_dir/status``."""
    return data_dir / _STATUS_DIR


def status_path(data_dir: Path) -> Path:
    """Return the private ``status.json`` path under ``data_dir``."""
    return status_dir(data_dir) / _STATUS_FILE


def public_status_path(data_dir: Path) -> Path:
    """Return the public whitelist ``status.public.json`` path."""
    return status_dir(data_dir) / _PUBLIC_FILE


def last_error_path(data_dir: Path) -> Path:
    """Return the path of the persisted last-error sidecar."""
    return status_dir(data_dir) / _LAST_ERROR_FILE


def build_status(
    data_dir: Path,
    *,
    now: datetime | None = None,
    models: list[str] | None = None,
    horizon: int | None = None,
    settings: Settings | None = None,
    clear_error: bool = False,
    repo_root: Path | None = None,
) -> StatusDocument:
    """Assemble a ``StatusDocument`` from the local data directory.

    Missing sources, plans, sibling-ops fields (checkpoints, prod tag), and
    auth report ``null`` with a ``reason`` rather than raising.

    Args:
        data_dir: Local data root (``raw/``, ``plans/``, ``staged/``, …).
        now: Clock for next-deadline / token expiry; defaults to UTC now.
        models: Production model spec names. Defaults to
            ``load_production_ops().models``.
        horizon: Plan horizon used when looking up plans. Defaults to
            ``load_production_ops().plan_horizon``.
        settings: Unused today; reserved for credentialed lookups.
        clear_error: When True, delete any persisted last-error sidecar before
            reading (successful ops tick).
        repo_root: Git checkout used to resolve the ``prod`` tag.

    Returns:
        A validated ``StatusDocument``.
    """
    del settings  # reserved for future remote lookups
    moment = _aware(datetime.now(UTC) if now is None else now)
    ops = load_production_ops()
    chosen = list(ops.models) if models is None else list(models)
    steps = ops.plan_horizon if horizon is None else horizon

    if clear_error:
        clear_last_error(data_dir)

    captures = _last_captures(data_dir / "raw")
    plans = _last_plans(data_dir, chosen, steps)
    next_deadline = _next_deadline(data_dir, moment)
    token = _token_expiry(moment)
    quarantines = _quarantines(data_dir)
    prod_sha, prod_reason = _prod_sha(repo_root)
    error, error_reason = _last_error(data_dir)
    heartbeats = _heartbeats(captures, moment)

    return StatusDocument(
        schema_version=STATUS_SCHEMA_VERSION,
        generated_at=moment,
        last_captures=captures,
        last_plans=plans,
        next_deadline=next_deadline,
        token_expiry=token,
        quarantines=quarantines,
        prod_sha=prod_sha,
        prod_sha_reason=prod_reason,
        last_error=error,
        last_error_reason=error_reason,
        heartbeats=heartbeats,
    )


def public_status(doc: StatusDocument) -> dict[str, Any]:
    """Return a JSON-ready dict with private fields removed.

    Args:
        doc: Full private status document.

    Returns:
        Whitelisted fields only (no token expiry, no last error / stack text).
    """
    payload = doc.model_dump_json_doc()
    for key in PRIVATE_STATUS_FIELDS:
        payload.pop(key, None)
    payload.pop("last_error_reason", None)
    return payload


def write_status(
    data_dir: Path,
    doc: StatusDocument,
    *,
    write_public: bool = True,
) -> Path:
    """Write private (and optional public) status JSON under ``data_dir/status``.

    Args:
        data_dir: Local data root.
        doc: Document to persist.
        write_public: Also write ``status.public.json``.

    Returns:
        Path of the private ``status.json``.
    """
    folder = status_dir(data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = status_path(data_dir)
    path.write_text(
        json.dumps(doc.model_dump_json_doc(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    if write_public:
        public_status_path(data_dir).write_text(
            json.dumps(public_status(doc), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
    return path


def push_status(
    data_dir: Path,
    *,
    settings: Settings | None = None,
    remote: Remote | None = None,
) -> Remote | None:
    """Upload ``status/status.json`` to the data bucket (mutable overwrite).

    Args:
        data_dir: Local data root holding ``status/status.json``.
        settings: Process settings (defaults to ``get_settings()``).
        remote: Optional pre-built remote (tests).

    Returns:
        The remote that received the file, or ``None`` when no data bucket is
        configured.

    Raises:
        FileNotFoundError: Local ``status.json`` is missing.
        RemoteError: rclone failed.
    """
    path = status_path(data_dir)
    if not path.is_file():
        raise FileNotFoundError(path)
    cfg = get_settings() if settings is None else settings
    target = remote if remote is not None else Remote.from_settings(cfg)
    if target is None:
        logger.info("status push skipped: data bucket unset")
        return None
    if target.kind != "data":
        raise ValueError(
            f"status.json must push to the data bucket; got kind={target.kind!r}"
        )
    logger.info("pushing status.json → %s/status/status.json", target.target)
    push_replace(target, path, "status/status.json")
    return target


def publish_ops_status(
    data_dir: Path,
    *,
    now: datetime | None = None,
    models: list[str] | None = None,
    horizon: int | None = None,
    settings: Settings | None = None,
    clear_error: bool = False,
) -> StatusDocument:
    """Build, write, and best-effort push status for an ops tick.

    Args:
        data_dir: Local data root.
        now: Clock for the document.
        models: Production models.
        horizon: Plan horizon.
        settings: Settings for the R2 push.
        clear_error: Clear the last-error sidecar (successful tick).

    Returns:
        The document that was written.
    """
    cfg = get_settings() if settings is None else settings
    doc = build_status(
        data_dir,
        now=now,
        models=models,
        horizon=horizon,
        settings=cfg,
        clear_error=clear_error,
    )
    write_status(data_dir, doc)
    try:
        push_status(data_dir, settings=cfg)
    except (RemoteError, ValueError, FileNotFoundError):
        logger.exception("status push failed; local status.json retained")
    return doc


def record_last_error(
    data_dir: Path,
    exc: BaseException,
    *,
    source: str = "ops-tick",
    now: datetime | None = None,
) -> Path:
    """Persist a last-error sidecar for the next status build.

    Args:
        data_dir: Local data root.
        exc: Exception to summarise (message only; never secrets).
        source: Component that failed (e.g. ``ops-tick``).
        now: Failure timestamp.

    Returns:
        Path of the written sidecar.
    """
    moment = _aware(datetime.now(UTC) if now is None else now)
    payload = {
        "at": moment.isoformat(),
        "message": f"{type(exc).__name__}: {exc}",
        "source": source,
    }
    path = last_error_path(data_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def clear_last_error(data_dir: Path) -> None:
    """Remove the last-error sidecar when present."""
    path = last_error_path(data_dir)
    if path.is_file():
        path.unlink()


def format_status_human(doc: StatusDocument) -> str:
    """Render a short human-readable status block.

    Args:
        doc: Status document.

    Returns:
        Multi-line text ending with a newline.
    """
    lines = [
        f"fplt status  generated_at={doc.generated_at.isoformat()}",
        f"  prod_sha: {doc.prod_sha or doc.prod_sha_reason}",
    ]
    for capture in doc.last_captures:
        stamp = (
            capture.captured_at.isoformat()
            if capture.captured_at is not None
            else capture.reason
        )
        lines.append(f"  capture[{capture.source}]: {stamp}")
    for plan in doc.last_plans:
        if plan.written_at is not None:
            lines.append(
                f"  plan[{plan.model}]: {plan.season} gw{plan.gw} "
                f"h{plan.horizon} @ {plan.written_at.isoformat()}"
            )
        else:
            lines.append(f"  plan[{plan.model}]: {plan.reason}")
    nd = doc.next_deadline
    if nd.deadline_utc is not None:
        lines.append(
            f"  next_deadline: {nd.season} gw{nd.gw} @ {nd.deadline_utc.isoformat()}"
        )
        if nd.reason:
            lines.append(f"    checkpoints: {nd.reason}")
    else:
        lines.append(f"  next_deadline: {nd.reason}")
    token = doc.token_expiry
    if token.expires_at is not None:
        lines.append(
            f"  token_expiry: {token.expires_at.isoformat()} expired={token.expired}"
        )
    else:
        lines.append(f"  token_expiry: {token.reason}")
    lines.append(f"  quarantines: {len(doc.quarantines)}")
    if doc.last_error is not None:
        lines.append(f"  last_error: {doc.last_error.message}")
    else:
        lines.append(f"  last_error: {doc.last_error_reason}")
    return "\n".join(lines) + "\n"


def _last_captures(raw_root: Path) -> list[CaptureSourceStatus]:
    out: list[CaptureSourceStatus] = []
    for source in STATUS_SOURCES:
        at, path, reason = _latest_capture(raw_root, source)
        out.append(
            CaptureSourceStatus(
                source=source,
                captured_at=at,
                path=path,
                reason=reason,
            )
        )
    return out


def _latest_capture(
    raw_root: Path, source: str
) -> tuple[datetime | None, str | None, str | None]:
    if not raw_root.exists():
        return None, None, f"raw root missing: no {source} captures"
    if source == "fpl_api":
        snaps = iter_snapshots(raw_root)
        if not snaps:
            return None, None, "no fpl_api bootstrap captures"
        last = snaps[-1]
        rel = last.capture_dir.relative_to(raw_root).as_posix()
        return last.captured_at, rel, None
    if source == "understat":
        candidates: list[tuple[datetime, Path]] = []
        for path, _season in iter_raw_teams(raw_root):
            stamp = _stamp_from_path(path)
            if stamp is not None:
                candidates.append((stamp, path))
        for path, _season in iter_raw_dates(raw_root):
            stamp = _stamp_from_path(path)
            if stamp is not None:
                candidates.append((stamp, path))
        if not candidates:
            return None, None, "no understat captured_at snapshots"
        stamp, path = max(candidates, key=lambda item: item[0])
        return stamp, path.relative_to(raw_root).as_posix(), None
    if source == "api_football":
        rows = iter_raw_injuries(raw_root)
        stamped: list[tuple[datetime, Path]] = []
        for path, _season in rows:
            stamp = _stamp_from_path(path)
            if stamp is not None:
                stamped.append((stamp, path))
        if not stamped:
            return None, None, "no api_football injury captures"
        stamp, path = max(stamped, key=lambda item: item[0])
        return stamp, path.relative_to(raw_root).as_posix(), None
    if source == "football_data":
        rows = iter_raw_e0(raw_root)
        stamped: list[tuple[datetime, Path]] = []
        for path, _season in rows:
            stamp = _stamp_from_path(path)
            if stamp is not None:
                stamped.append((stamp, path))
            else:
                # Legacy season=*/E0.csv — fall back to mtime.
                stamped.append((_mtime_utc(path), path))
        if not stamped:
            return None, None, "no football_data E0 captures"
        stamp, path = max(stamped, key=lambda item: item[0])
        return stamp, path.relative_to(raw_root).as_posix(), None
    return None, None, f"unknown source {source!r}"


def _last_plans(
    data_dir: Path, models: list[str], horizon: int
) -> list[PlanDeadlineStatus]:
    specs = load_model_specs()
    out: list[PlanDeadlineStatus] = []
    for name in models:
        if name not in specs:
            out.append(
                PlanDeadlineStatus(
                    model=name,
                    reason=f"model spec {name!r} not registered",
                )
            )
            continue
        model_id = specs[name].model_id
        found = _plans_for_model(data_dir, model_id, horizon)
        if not found:
            out.append(
                PlanDeadlineStatus(
                    model=name,
                    model_id=model_id,
                    horizon=horizon,
                    reason="no plan parquet for production model/horizon",
                )
            )
            continue
        # Latest deadline for this model at the ops horizon.
        latest = max(found, key=lambda item: (item[0], item[1], item[3]))
        season, gw, weeks, written_at, rel = latest
        out.append(
            PlanDeadlineStatus(
                model=name,
                model_id=model_id,
                season=season,
                gw=gw,
                horizon=weeks,
                written_at=written_at,
                path=rel,
            )
        )
    return out


def _plans_for_model(
    data_dir: Path, model_id: str, horizon: int
) -> list[tuple[str, int, int, datetime, str]]:
    base = data_dir / "plans" / model_id
    if not base.is_dir():
        return []
    found: list[tuple[str, int, int, datetime, str]] = []
    for parquet in base.glob("as_of=*/part.parquet"):
        match = _AS_OF_RE.match(parquet.parent.name)
        if match is None:
            continue
        weeks = int(match.group("horizon"))
        if weeks != horizon:
            continue
        season = match.group("season")
        gw = int(match.group("gw"))
        rel = parquet.relative_to(data_dir).as_posix()
        found.append((season, gw, weeks, _mtime_utc(parquet), rel))
    return found


def _next_deadline(data_dir: Path, now: datetime) -> NextDeadlineStatus:
    staged = data_dir / "staged"
    if not staged.exists():
        return NextDeadlineStatus(reason="staged gameweeks missing")
    try:
        frame = DataStore(data_dir).read("gameweeks")
    except Exception as exc:  # noqa: BLE001 — status must tolerate empty dirs
        return NextDeadlineStatus(reason=f"gameweeks unreadable: {exc}")
    if frame.is_empty():
        return NextDeadlineStatus(reason="no gameweeks rows")
    unchecked = frame.filter(~pl.col("data_checked"))
    upcoming: list[tuple[datetime, str, int]] = []
    for row in unchecked.iter_rows(named=True):
        deadline = _aware(row["deadline_time"])
        if deadline > now:
            upcoming.append((deadline, str(row["season"]), int(row["gw"])))
    if not upcoming:
        return NextDeadlineStatus(reason="no upcoming unchecked deadline")
    upcoming.sort(key=lambda item: item[0])
    deadline, season, gw = upcoming[0]
    done, due, reason = _checkpoint_state(data_dir, now, deadline, season=season, gw=gw)
    return NextDeadlineStatus(
        season=season,
        gw=gw,
        deadline_utc=deadline,
        checkpoints_done=done,
        checkpoints_due=due,
        reason=reason,
    )


def _checkpoint_state(
    data_dir: Path,
    now: datetime,
    deadline: datetime,
    *,
    season: str,
    gw: int,
) -> tuple[list[str] | None, list[str] | None, str | None]:
    """Resolve ops-02 checkpoint done/due lists for the next deadline."""
    try:
        from fplt.automation.checkpoints import (
            due_checkpoints,
            resolve_checkpoint_state,
        )
        from fplt.types import GWKey
    except ImportError:
        return (
            None,
            None,
            "ops-02 due_checkpoints not available",
        )
    try:
        key = GWKey(season, gw)
        state = resolve_checkpoint_state(data_dir, key, deadline)
        due = due_checkpoints(now, deadline, state.done)
        done = [str(item) for item in state.done]
        due_ids = [cp.id for cp in due]
        return done, due_ids, None
    except Exception as exc:  # noqa: BLE001 — tolerate sibling-ops probes
        return None, None, f"ops-02 checkpoint probe failed: {exc}"


def _token_expiry(now: datetime) -> TokenExpiryStatus:
    info = auth_status(now=now)
    token = info.token
    if token is None:
        return TokenExpiryStatus(reason="token missing; run fplt auth set")
    if token.expires_at is None:
        return TokenExpiryStatus(reason="token expiry claim missing")
    return TokenExpiryStatus(
        expires_at=token.expires_at,
        expired=token.expired,
    )


def _quarantines(data_dir: Path) -> list[QuarantineStatusItem]:
    frame = load_quarantine(data_dir)
    if frame.is_empty():
        return []
    items: list[QuarantineStatusItem] = []
    for row in frame.iter_rows(named=True):
        items.append(
            QuarantineStatusItem(
                source=str(row["source"]),
                capture_path=str(row["capture_path"]),
                gate=str(row["gate"]),
                detail=str(row["detail"]),
                captured_at=_aware(row["captured_at"]),
                available_at=_aware(row["available_at"]),
            )
        )
    return items


def _prod_sha(repo_root: Path | None) -> tuple[str | None, str | None]:
    cwd = Path.cwd() if repo_root is None else repo_root
    try:
        completed = run_command(
            ["git", "rev-parse", "prod"],
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
    except (OSError, subprocess.CalledProcessError):
        return None, "prod tag not found (ops-04)"
    sha = completed.stdout.strip()
    if not sha:
        return None, "prod tag empty (ops-04)"
    return sha, None


def _last_error(
    data_dir: Path,
) -> tuple[LastErrorStatus | None, str | None]:
    path = last_error_path(data_dir)
    if not path.is_file():
        return None, "no recorded error"
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        return (
            LastErrorStatus(
                at=_aware(datetime.fromisoformat(str(raw["at"]))),
                message=str(raw["message"]),
                source=raw.get("source"),
            ),
            None,
        )
    except (OSError, KeyError, TypeError, ValueError) as exc:
        return None, f"last_error unreadable: {exc}"


def _heartbeats(
    captures: list[CaptureSourceStatus], now: datetime
) -> list[HeartbeatStamp]:
    by_source = {item.source: item for item in captures}
    out: list[HeartbeatStamp] = []
    for name in HEARTBEAT_NAMES:
        if name == "ops_tick":
            out.append(HeartbeatStamp(name=name, last_at=now))
            continue
        source = next(
            (src for src, hb in _SOURCE_TO_HEARTBEAT.items() if hb == name),
            None,
        )
        capture = by_source.get(source) if source else None
        if capture is None:
            out.append(HeartbeatStamp(name=name, reason="unknown heartbeat"))
            continue
        if capture.captured_at is None:
            out.append(
                HeartbeatStamp(
                    name=name,
                    reason=capture.reason or "no capture yet",
                )
            )
        else:
            out.append(HeartbeatStamp(name=name, last_at=capture.captured_at))
    return out


def _stamp_from_path(path: Path) -> datetime | None:
    match = _CAPTURE_STAMP_RE.search(path.as_posix())
    if match is None:
        return None
    raw = match.group(1)
    if "T" in raw:
        return datetime.strptime(raw, "%Y-%m-%dT%H-%M-%SZ").replace(tzinfo=UTC)
    return datetime.strptime(raw, "%Y-%m-%d").replace(tzinfo=UTC)


def _mtime_utc(path: Path) -> datetime:
    return datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)
