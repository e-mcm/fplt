"""Run-timing panel shaping for the This-week page (#291 PR3).

Pure helpers: callers load ``ops_timing.json`` / remote timing records (or pass
parsed dicts) and these functions build tables for the stacked bar, sub-phase
expander, R2 transfer summary, and cross-gameweek trend. No Streamlit / I/O.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

import polars as pl

from fplt.schemas.run_timing import (
    RUN_TIMING_PHASE_COLUMNS,
    RUN_TIMING_TRANSFER_COLUMNS,
    RUN_TIMING_TREND_COLUMNS,
    RunTimingPhaseSchema,
    RunTimingTransferSchema,
    RunTimingTrendSchema,
)
from fplt.schemas.timing import TimingRecord
from fplt.utils.timing import (
    OPS_TIMING_FILENAME,
    SHARED_SUBPHASES,
    ops_phase_total,
)

_MIB = 1024.0 * 1024.0


@dataclass(frozen=True)
class RunTimingPanel:
    """Shaped tables for the This-week Run timing panel."""

    present: bool
    kind: str | None
    total_s: float
    top_phases: pl.DataFrame
    sub_phases: pl.DataFrame
    transfers: pl.DataFrame
    trend: pl.DataFrame


def parse_timing_record(raw: Mapping[str, Any] | None) -> TimingRecord | None:
    """Validate a timing JSON object; return ``None`` when missing/invalid."""
    if raw is None:
        return None
    try:
        return TimingRecord.model_validate(dict(raw))
    except (ValueError, TypeError):
        return None


def load_ops_timing_file(path: Path) -> TimingRecord | None:
    """Load and validate one ``ops_timing.json`` path (``None`` if absent)."""
    if not path.is_file():
        return None
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        return None
    try:
        payload: object = json.loads(text)
    except (OSError, ValueError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    return parse_timing_record(cast(dict[str, Any], payload))


def top_level_phases(record: TimingRecord) -> pl.DataFrame:
    """Return top-level ops phases (excludes shared nested sub-phases).

    Args:
        record: Validated timing artifact.

    Returns:
        ``RunTimingPhaseSchema`` rows sorted by seconds descending.
    """
    total = (
        float(record.total_s) if record.total_s > 0 else ops_phase_total(record.phases)
    )
    rows: list[dict[str, object]] = []
    for name, seconds in record.phases.items():
        if name in SHARED_SUBPHASES or seconds <= 0:
            continue
        share = 0.0 if total <= 0 else 100.0 * seconds / total
        rows.append({"phase": name, "seconds": float(seconds), "share_pct": share})
    if not rows:
        return pl.DataFrame(schema=RUN_TIMING_PHASE_COLUMNS)
    frame = pl.DataFrame(rows, schema=RUN_TIMING_PHASE_COLUMNS).sort(
        "seconds", descending=True
    )
    return RunTimingPhaseSchema.validate(frame)  # type: ignore[return-value]


def sub_phases_table(record: TimingRecord) -> pl.DataFrame:
    """Return nested sub-phases (fit components, plan/MILP, …).

    Args:
        record: Validated timing artifact.

    Returns:
        ``RunTimingPhaseSchema`` rows for ``SHARED_SUBPHASES`` present in the
        record, sorted by seconds descending.
    """
    total = (
        float(record.total_s) if record.total_s > 0 else ops_phase_total(record.phases)
    )
    rows: list[dict[str, object]] = []
    for name, seconds in record.phases.items():
        if name not in SHARED_SUBPHASES or seconds <= 0:
            continue
        share = 0.0 if total <= 0 else 100.0 * seconds / total
        rows.append({"phase": name, "seconds": float(seconds), "share_pct": share})
    if not rows:
        return pl.DataFrame(schema=RUN_TIMING_PHASE_COLUMNS)
    frame = pl.DataFrame(rows, schema=RUN_TIMING_PHASE_COLUMNS).sort(
        "seconds", descending=True
    )
    return RunTimingPhaseSchema.validate(frame)  # type: ignore[return-value]


def transfer_table(
    remote_records: Sequence[Mapping[str, Any]],
    *,
    phase_seconds: Mapping[str, float] | None = None,
) -> pl.DataFrame:
    """Build R2 transfer rows from ``kind=remote`` timing artifacts.

    Args:
        remote_records: Parsed remote timing dicts (``meta`` + ``phases``).
        phase_seconds: Optional ops-record phases to fill seconds when a remote
            artifact is missing elapsed time for ``remote/<op>``.

    Returns:
        ``RunTimingTransferSchema`` frame (may be empty).
    """
    phases = dict(phase_seconds or {})
    rows: list[dict[str, object]] = []
    for record in remote_records:
        meta_obj: object = record.get("meta", {})
        if not isinstance(meta_obj, dict):
            continue
        meta = cast(dict[str, Any], meta_obj)
        op_raw = meta.get("op")
        if not isinstance(op_raw, str) or not op_raw:
            continue
        op = op_raw
        phase_name = f"remote/{op}"
        phases_obj: object = record.get("phases", {})
        seconds = 0.0
        if isinstance(phases_obj, dict):
            raw_phases = cast(dict[str, Any], phases_obj)
            value = raw_phases.get(phase_name, raw_phases.get(op))
            if isinstance(value, (int, float)) and not isinstance(value, bool):
                seconds = float(value)
        if seconds <= 0 and phase_name in phases:
            seconds = float(phases[phase_name])
        if seconds <= 0:
            continue
        n_objects = _optional_int(meta.get("n_objects"))
        nbytes = _optional_int(meta.get("bytes"))
        prefix_raw = meta.get("prefix")
        prefix = str(prefix_raw) if prefix_raw is not None else None
        throughput: float | None = None
        if nbytes is not None and seconds > 0:
            throughput = (nbytes / _MIB) / seconds
        rows.append(
            {
                "op": op,
                "prefix": prefix,
                "n_objects": n_objects,
                "bytes": nbytes,
                "seconds": seconds,
                "throughput_mib_s": throughput,
            }
        )
    if not rows and phases:
        for name, seconds in phases.items():
            if not name.startswith("remote/") or seconds <= 0:
                continue
            rows.append(
                {
                    "op": name.removeprefix("remote/"),
                    "prefix": None,
                    "n_objects": None,
                    "bytes": None,
                    "seconds": float(seconds),
                    "throughput_mib_s": None,
                }
            )
    if not rows:
        return pl.DataFrame(schema=RUN_TIMING_TRANSFER_COLUMNS)
    frame = pl.DataFrame(rows, schema=RUN_TIMING_TRANSFER_COLUMNS)
    return RunTimingTransferSchema.validate(frame)  # type: ignore[return-value]


def timing_trend(
    ops_records: Sequence[tuple[str, int, str, TimingRecord]],
) -> pl.DataFrame:
    """Build a per-gameweek trend of ``total_s`` (and top-phase columns).

    Args:
        ops_records: ``(season, gw, checkpoint, record)`` tuples.

    Returns:
        ``RunTimingTrendSchema`` frame sorted by season then gw. Extra dynamic
        columns name top-level phases (seconds) when present.
    """
    if not ops_records:
        return pl.DataFrame(schema=RUN_TIMING_TREND_COLUMNS)
    phase_names: set[str] = set()
    base_rows: list[dict[str, object]] = []
    for season, gw, checkpoint, record in ops_records:
        top = {
            name: float(seconds)
            for name, seconds in record.phases.items()
            if name not in SHARED_SUBPHASES and seconds > 0
        }
        phase_names.update(top)
        row: dict[str, object] = {
            "season": season,
            "gw": int(gw),
            "checkpoint": checkpoint,
            "total_s": float(record.total_s),
        }
        row.update(top)
        base_rows.append(row)
    # Stable schema: base columns + sorted phase columns.
    schema: dict[str, pl.DataType] = dict(RUN_TIMING_TREND_COLUMNS)
    for name in sorted(phase_names):
        schema[name] = pl.Float64()
    normalised: list[dict[str, object]] = []
    for row in base_rows:
        filled = {key: row.get(key) for key in schema}
        normalised.append(filled)
    frame = pl.DataFrame(normalised, schema=schema).sort(["season", "gw", "checkpoint"])
    return RunTimingTrendSchema.validate(frame)  # type: ignore[return-value]


def discover_ops_timing(
    data_dir: Path,
    *,
    season: str | None = None,
    checkpoint: str | None = None,
) -> list[tuple[str, int, str, TimingRecord]]:
    """Scan ``official/…/ops_timing.json`` files under ``data_dir``.

    Args:
        data_dir: Local ``$FPLT_DATA_DIR`` root.
        season: Optional season filter.
        checkpoint: Optional checkpoint filter.

    Returns:
        Sorted ``(season, gw, checkpoint, record)`` list.
    """
    root = Path(data_dir) / "official"
    if not root.is_dir():
        return []
    found: list[tuple[str, int, str, TimingRecord]] = []
    for path in sorted(root.glob("season=*/gw=*/checkpoint=*/" + OPS_TIMING_FILENAME)):
        parts = path.relative_to(root).parts
        if len(parts) < 4:
            continue
        season_part, gw_part, cp_part, _name = parts[0], parts[1], parts[2], parts[3]
        if not season_part.startswith("season="):
            continue
        if not gw_part.startswith("gw="):
            continue
        if not cp_part.startswith("checkpoint="):
            continue
        season_val = season_part.removeprefix("season=")
        checkpoint_val = cp_part.removeprefix("checkpoint=")
        if season is not None and season_val != season:
            continue
        if checkpoint is not None and checkpoint_val != checkpoint:
            continue
        try:
            gw_val = int(gw_part.removeprefix("gw="))
        except ValueError:
            continue
        record = load_ops_timing_file(path)
        if record is None:
            continue
        found.append((season_val, gw_val, checkpoint_val, record))
    found.sort(key=lambda item: (item[0], item[1], item[2]))
    return found


def run_timing_panel(
    *,
    ops_record: TimingRecord | None,
    remote_records: Sequence[Mapping[str, Any]] = (),
    trend_records: Sequence[tuple[str, int, str, TimingRecord]] = (),
) -> RunTimingPanel:
    """Assemble the Run timing panel tables for one selected checkpoint.

    Args:
        ops_record: ``ops_timing.json`` for the selected checkpoint, or ``None``.
        remote_records: Local ``kind=remote`` timing artifacts for transfer rows.
        trend_records: Other pulled ops timings (same season/checkpoint typical).

    Returns:
        ``RunTimingPanel``; ``present`` is False when ``ops_record`` is missing.
    """
    empty_phases = pl.DataFrame(schema=RUN_TIMING_PHASE_COLUMNS)
    empty_transfers = pl.DataFrame(schema=RUN_TIMING_TRANSFER_COLUMNS)
    if ops_record is None:
        return RunTimingPanel(
            present=False,
            kind=None,
            total_s=0.0,
            top_phases=empty_phases,
            sub_phases=empty_phases,
            transfers=empty_transfers,
            trend=timing_trend(trend_records),
        )
    return RunTimingPanel(
        present=True,
        kind=ops_record.kind,
        total_s=float(ops_record.total_s),
        top_phases=top_level_phases(ops_record),
        sub_phases=sub_phases_table(ops_record),
        transfers=transfer_table(remote_records, phase_seconds=ops_record.phases),
        trend=timing_trend(trend_records),
    )


def _optional_int(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value)
