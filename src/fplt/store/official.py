"""Write-once official pointers and applied / submitted / intervention records.

Paths (``fplt-data-model.md`` §3.5):
``official/season=<s>/gw=<gw>/checkpoint=<cp>/line=<line>.json``,
``applied.json``, ``submitted.json``, ``intervention.json``.

Pointers require both predict and plan ``_SUCCESS``. ops-02 / ops-04 wire
callers; this module is the storage contract.
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from fplt.schemas.manifests import (
    AppliedRecord,
    InterventionRecord,
    LineRole,
    OfficialPointer,
    SquadMode,
    SubmittedRecord,
)
from fplt.store.runs import run_dir

logger = logging.getLogger(__name__)

_SUCCESS = "_SUCCESS"
_DEFAULT_CHECKPOINT = "T-90m"


class OfficialExistsError(FileExistsError):
    """Raised when an official path already exists (write-once)."""


@dataclass(frozen=True)
class LineRunPair:
    """Succeeded predict + plan run ids for one official line."""

    line: str
    role: LineRole
    squad_mode: SquadMode
    predict_run_id: str
    plan_run_id: str


def official_checkpoint_dir(
    data_dir: Path, *, season: str, gw: int, checkpoint: str = _DEFAULT_CHECKPOINT
) -> Path:
    """Directory for one season/GW/checkpoint under ``official/``."""
    return (
        data_dir
        / "official"
        / f"season={season}"
        / f"gw={gw}"
        / f"checkpoint={checkpoint}"
    )


def pointer_path(
    data_dir: Path,
    *,
    season: str,
    gw: int,
    line: str,
    checkpoint: str = _DEFAULT_CHECKPOINT,
) -> Path:
    """Path for one per-line official pointer JSON."""
    return (
        official_checkpoint_dir(data_dir, season=season, gw=gw, checkpoint=checkpoint)
        / f"line={line}.json"
    )


def require_run_succeeded(data_dir: Path, run_id: str) -> Path:
    """Return the run directory if ``_SUCCESS`` exists.

    Raises:
        FileNotFoundError: The run folder or ``_SUCCESS`` marker is missing.
    """
    path = run_dir(data_dir, run_id)
    if not path.is_dir():
        raise FileNotFoundError(f"run directory missing: {path}")
    if not (path / _SUCCESS).is_file():
        raise FileNotFoundError(f"run {run_id} has no _SUCCESS")
    return path


def write_official_pointer(
    data_dir: Path,
    *,
    season: str,
    gw: int,
    line: str,
    pointer: OfficialPointer,
    checkpoint: str = _DEFAULT_CHECKPOINT,
) -> Path:
    """Write one line pointer once. Requires both runs to have succeeded."""
    require_run_succeeded(data_dir, pointer.predict_run_id)
    require_run_succeeded(data_dir, pointer.plan_run_id)
    return _write_model_once(
        pointer_path(data_dir, season=season, gw=gw, line=line, checkpoint=checkpoint),
        pointer,
    )


def write_applied_record(
    data_dir: Path,
    *,
    season: str,
    gw: int,
    record: AppliedRecord,
    checkpoint: str = _DEFAULT_CHECKPOINT,
) -> Path:
    """Write ``applied.json`` once for the checkpoint."""
    require_run_succeeded(data_dir, record.plan_run_id)
    dest = (
        official_checkpoint_dir(data_dir, season=season, gw=gw, checkpoint=checkpoint)
        / "applied.json"
    )
    return _write_model_once(dest, record)


def write_submitted_record(
    data_dir: Path,
    *,
    season: str,
    gw: int,
    record: SubmittedRecord,
    checkpoint: str = _DEFAULT_CHECKPOINT,
) -> Path:
    """Write ``submitted.json`` once (post-apply FPL readback)."""
    dest = (
        official_checkpoint_dir(data_dir, season=season, gw=gw, checkpoint=checkpoint)
        / "submitted.json"
    )
    return _write_model_once(dest, record)


def write_intervention_record(
    data_dir: Path,
    *,
    season: str,
    gw: int,
    record: InterventionRecord,
    checkpoint: str = _DEFAULT_CHECKPOINT,
) -> Path:
    """Write ``intervention.json`` once beside applied / submitted."""
    dest = (
        official_checkpoint_dir(data_dir, season=season, gw=gw, checkpoint=checkpoint)
        / "intervention.json"
    )
    return _write_model_once(dest, record)


def finalize_official_checkpoint(
    data_dir: Path,
    *,
    season: str,
    gw: int,
    pairs: Sequence[LineRunPair],
    checkpoint: str = _DEFAULT_CHECKPOINT,
    written_at: datetime | None = None,
    apply_primary: bool = True,
) -> list[Path]:
    """Write per-line pointers; optionally ``applied.json`` for the primary.

    Args:
        data_dir: Local data root.
        season: Season key (e.g. ``2026-27``).
        gw: Gameweek number.
        pairs: One entry per line with succeeded predict/plan run ids.
        checkpoint: Ops checkpoint label (default ``T-90m``).
        written_at: Pointer timestamp; defaults to now (UTC).
        apply_primary: When True, also write ``applied.json`` for the primary.

    Returns:
        Paths written (pointers, then applied when requested).

    Raises:
        ValueError: ``pairs`` empty, or not exactly one primary when applying.
        FileNotFoundError: A named run lacks ``_SUCCESS``.
        OfficialExistsError: A destination path already exists.
    """
    if not pairs:
        raise ValueError("finalize_official_checkpoint requires at least one line")
    moment = written_at if written_at is not None else datetime.now(UTC)
    moment = (
        moment.replace(tzinfo=UTC) if moment.tzinfo is None else moment.astimezone(UTC)
    )

    written: list[Path] = []
    primaries = [pair for pair in pairs if pair.role == "primary"]
    for pair in pairs:
        written.append(
            write_official_pointer(
                data_dir,
                season=season,
                gw=gw,
                line=pair.line,
                pointer=OfficialPointer(
                    predict_run_id=pair.predict_run_id,
                    plan_run_id=pair.plan_run_id,
                    role=pair.role,
                    squad_mode=pair.squad_mode,
                    written_at=moment,
                ),
                checkpoint=checkpoint,
            )
        )

    if apply_primary:
        if len(primaries) != 1:
            raise ValueError(
                f"apply_primary needs exactly one primary pair, got {len(primaries)}"
            )
        primary = primaries[0]
        written.append(
            write_applied_record(
                data_dir,
                season=season,
                gw=gw,
                record=AppliedRecord(
                    line=primary.line,
                    plan_run_id=primary.plan_run_id,
                    submitted_at=moment,
                ),
                checkpoint=checkpoint,
            )
        )
    logger.info(
        "official checkpoint written season=%s gw=%s checkpoint=%s lines=%s",
        season,
        gw,
        checkpoint,
        [pair.line for pair in pairs],
    )
    return written


def _write_model_once(dest: Path, model: BaseModel) -> Path:
    if dest.exists():
        raise OfficialExistsError(f"official path already exists: {dest}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(model.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return dest
