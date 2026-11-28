"""Find plan runs under ``runs/`` (data-model cut-over, step 10).

Plans used to be looked up at ``plans/<model_id>/as_of=…/part.parquet``. That
path does not change with code, purpose or squad, so a stale or ad hoc file
could stand in for a fresh one. Readers now resolve the newest succeeded
``run_type=plan`` run for ``(spec_label, season, gw, plan_horizon)`` from the
local run catalog instead.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from fplt.store.catalog import scan_succeeded_runs
from fplt.store.runs import run_dir
from fplt.types import GWKey

PLAN_TABLE = "plans/plan.parquet"
REPORT_FILE = "report.md"


def plan_runs(
    data_dir: Path,
    *,
    spec_label: str,
    key: GWKey,
    horizon: int,
    catalog: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Return succeeded plan runs for one spec, deadline and horizon, oldest first.

    Args:
        data_dir: Local data root that contains ``runs/``.
        spec_label: YAML spec name (``RunManifest.spec_label``).
        key: Deadline planned from.
        horizon: Covered plan weeks (``plan_horizon``; falls back to
            ``horizon`` for manifests without it).
        catalog: Optional preloaded ``CatalogRunSchema`` frame (tests / batch).

    Returns:
        Matching catalog rows sorted by ``finished_at`` (possibly empty).
    """
    frame = scan_succeeded_runs(data_dir) if catalog is None else catalog
    if frame.is_empty() or horizon < 1:
        return frame.head(0)
    horizon_match = (
        pl.when(pl.col("plan_horizon").is_not_null())
        .then(pl.col("plan_horizon") == horizon)
        .otherwise(pl.col("horizon") == horizon)
    )
    return frame.filter(
        (pl.col("run_type") == "plan")
        & (pl.col("spec_label") == spec_label)
        & (pl.col("as_of_season") == key.season)
        & (pl.col("as_of_gw") == key.gw)
        & horizon_match
    ).sort("finished_at", "run_id")


def latest_plan_run_id(
    data_dir: Path,
    *,
    spec_label: str,
    key: GWKey,
    horizon: int,
    catalog: pl.DataFrame | None = None,
) -> str | None:
    """Return the newest matching plan run id, or ``None`` when there is none."""
    rows = plan_runs(
        data_dir, spec_label=spec_label, key=key, horizon=horizon, catalog=catalog
    )
    if rows.is_empty():
        return None
    return str(rows.get_column("run_id")[-1])


def read_latest_plan(
    data_dir: Path,
    *,
    spec_label: str,
    key: GWKey,
    horizon: int,
) -> pl.DataFrame:
    """Read the ``PlanSchema`` table of the newest matching plan run.

    Args:
        data_dir: Local data root.
        spec_label: YAML spec name.
        key: Deadline planned from.
        horizon: Covered plan weeks.

    Returns:
        The plan rows.

    Raises:
        FileNotFoundError: No succeeded plan run matches.
    """
    run_id = latest_plan_run_id(
        data_dir, spec_label=spec_label, key=key, horizon=horizon
    )
    if run_id is None:
        raise FileNotFoundError(
            f"no plan run for {spec_label} {key.season} gw{key.gw} h{horizon} "
            f"under {data_dir / 'runs'}"
        )
    return pl.read_parquet(run_dir(data_dir, run_id) / PLAN_TABLE)


def plan_reports(data_dir: Path) -> str:
    """Concatenate ``report.md`` from every succeeded plan run, oldest first.

    Args:
        data_dir: Local data root.

    Returns:
        The markdown, or an empty string when no plan run has a report.
    """
    frame = scan_succeeded_runs(data_dir)
    if frame.is_empty():
        return ""
    plans = frame.filter(pl.col("run_type") == "plan").sort("finished_at", "run_id")
    parts: list[str] = []
    for run_id in plans.get_column("run_id").to_list():
        path = run_dir(data_dir, str(run_id)) / REPORT_FILE
        if path.exists():
            parts.append(path.read_text(encoding="utf-8"))
    return "\n".join(parts)


__all__ = [
    "PLAN_TABLE",
    "REPORT_FILE",
    "latest_plan_run_id",
    "plan_reports",
    "plan_runs",
    "read_latest_plan",
]
