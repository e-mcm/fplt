"""Overview-page shaping: runs catalog, plan summary, quarantine rollup.

Pure polars helpers used by the Streamlit overview. No I/O and no Streamlit
imports — the UI adapter loads data, then these functions shape it.
"""

from __future__ import annotations

from collections.abc import Sequence

import polars as pl

from fplt.schemas.manifests import OfficialPointer, RunManifest


def runs_catalog_frame(manifests: Sequence[RunManifest]) -> pl.DataFrame:
    """Build a catalog table from run manifests (newest first).

    Args:
        manifests: Parsed ``RunManifest`` values (any status).

    Returns:
        One row per manifest with the overview columns. Empty when none.
    """
    if not manifests:
        return pl.DataFrame(schema=_runs_schema())
    rows: list[dict[str, object]] = []
    for manifest in manifests:
        first = manifest.as_of[0]
        rows.append(
            {
                "run_id": manifest.run_id,
                "kind": manifest.kind,
                "run_type": manifest.run_type,
                "status": manifest.status,
                "spec_label": manifest.spec_label,
                "model_id": manifest.model_id,
                "line": manifest.line,
                "season": first.season,
                "gw": first.gw,
                "horizon": manifest.horizon,
                "checkpoint": manifest.checkpoint,
                "host": manifest.host,
                "started_at": manifest.started_at,
                "finished_at": manifest.finished_at,
            }
        )
    frame = pl.DataFrame(rows, schema=_runs_schema())
    return frame.sort("run_id", descending=True)


def plan_summary_frame(plan: pl.DataFrame) -> pl.DataFrame:
    """Summarise step-0 transfers / XI / captain from a ``PlanSchema`` frame.

    Args:
        plan: Rows matching ``PlanSchema`` (one model, one deadline).

    Returns:
        A single-row summary, or empty when ``plan`` has no step-0 rows.
    """
    if plan.is_empty() or "step" not in plan.columns:
        return pl.DataFrame(schema=_plan_summary_schema())
    step0 = plan.filter(pl.col("step") == 0)
    if step0.is_empty():
        return pl.DataFrame(schema=_plan_summary_schema())
    season = str(step0["as_of_season"][0])
    gw = int(step0["as_of_gw"][0])
    model_id = str(step0["model_id"][0])
    bought = step0.filter(pl.col("bought"))["player_key"].to_list()
    sold = step0.filter(pl.col("sold"))["player_key"].to_list()
    captains = step0.filter(pl.col("captain"))["player_key"].to_list()
    vices = step0.filter(pl.col("vice"))["player_key"].to_list()
    hit_raw = step0.select(pl.col("hit").max()).item()
    hit = float(hit_raw) if isinstance(hit_raw, (int, float)) else 0.0
    return pl.DataFrame(
        {
            "as_of_season": [season],
            "as_of_gw": [gw],
            "model_id": [model_id],
            "transfers_in": [len(bought)],
            "transfers_out": [len(sold)],
            "hit": [hit],
            "captain_key": [int(captains[0]) if captains else None],
            "vice_key": [int(vices[0]) if vices else None],
            "xi_count": [int(step0.filter(pl.col("in_xi")).height)],
        },
        schema=_plan_summary_schema(),
    )


def quarantine_summary_frame(quarantine: pl.DataFrame) -> pl.DataFrame:
    """Roll quarantine rows up by ``source`` and ``gate``.

    Args:
        quarantine: Rows matching ``QuarantineRecordSchema``.

    Returns:
        Counts and latest ``available_at`` per source/gate.
    """
    if quarantine.is_empty():
        return pl.DataFrame(schema=_quarantine_schema())
    return (
        quarantine.group_by(["source", "gate"])
        .agg(
            pl.len().alias("n_rows"),
            pl.col("available_at").max().alias("latest_available_at"),
        )
        .select(["source", "gate", "n_rows", "latest_available_at"])
        .sort(["source", "gate"])
    )


def official_pointers_frame(
    rows: Sequence[tuple[str, int, str, str, OfficialPointer]],
) -> pl.DataFrame:
    """Build an official-line table from parsed pointers.

    Args:
        rows: ``(season, gw, checkpoint, line, OfficialPointer)`` tuples.

    Returns:
        One row per pointer, sorted by season / gw / checkpoint / line.
    """
    if not rows:
        return pl.DataFrame(schema=_official_schema())
    records: list[dict[str, object]] = []
    for season, gw, checkpoint, line, pointer in rows:
        records.append(
            {
                "season": season,
                "gw": gw,
                "checkpoint": checkpoint,
                "line": line,
                "role": pointer.role,
                "squad_mode": pointer.squad_mode,
                "predict_run_id": pointer.predict_run_id,
                "plan_run_id": pointer.plan_run_id,
                "written_at": pointer.written_at,
            }
        )
    return pl.DataFrame(records, schema=_official_schema()).sort(
        ["season", "gw", "checkpoint", "line"]
    )


def _runs_schema() -> dict[str, pl.DataType]:
    return {
        "run_id": pl.Utf8(),
        "kind": pl.Utf8(),
        "run_type": pl.Utf8(),
        "status": pl.Utf8(),
        "spec_label": pl.Utf8(),
        "model_id": pl.Utf8(),
        "line": pl.Utf8(),
        "season": pl.Utf8(),
        "gw": pl.Int64(),
        "horizon": pl.Int64(),
        "checkpoint": pl.Utf8(),
        "host": pl.Utf8(),
        "started_at": pl.Datetime(time_zone="UTC"),
        "finished_at": pl.Datetime(time_zone="UTC"),
    }


def _plan_summary_schema() -> dict[str, pl.DataType]:
    return {
        "as_of_season": pl.Utf8(),
        "as_of_gw": pl.Int64(),
        "model_id": pl.Utf8(),
        "transfers_in": pl.Int64(),
        "transfers_out": pl.Int64(),
        "hit": pl.Float64(),
        "captain_key": pl.Int64(),
        "vice_key": pl.Int64(),
        "xi_count": pl.Int64(),
    }


def _quarantine_schema() -> dict[str, pl.DataType]:
    return {
        "source": pl.Utf8(),
        "gate": pl.Utf8(),
        "n_rows": pl.UInt32(),
        "latest_available_at": pl.Datetime(time_zone="UTC"),
    }


def _official_schema() -> dict[str, pl.DataType]:
    return {
        "season": pl.Utf8(),
        "gw": pl.Int64(),
        "checkpoint": pl.Utf8(),
        "line": pl.Utf8(),
        "role": pl.Utf8(),
        "squad_mode": pl.Utf8(),
        "predict_run_id": pl.Utf8(),
        "plan_run_id": pl.Utf8(),
        "written_at": pl.Datetime(time_zone="UTC"),
    }


__all__ = [
    "official_pointers_frame",
    "plan_summary_frame",
    "quarantine_summary_frame",
    "runs_catalog_frame",
]
