"""Remote run-done index for catch-up cold-cache skips (#371 / #381).

Local ``runs/`` is the fast path. When ``FPLT_DATA_BUCKET`` is set, catch-up
also consults the durable R2 catalog (``catalog/runs.parquet`` plus
``catalog/inbox/``) on the lab and/or data remotes. Catch-up for past GWs is
**predict-only** (#381), keyed by
``(spec_label, season, gw, horizon, run_type=predict)``. Upcoming-deadline
plans still use ``run_type=plan`` / ``plan_horizon`` via ``catalog_has_plans``.
Ops fingerprint ``runs/by_fp/`` remains the skip-if-done path for official
ticks (#346).
"""

from __future__ import annotations

import logging
import tempfile
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from fplt.data.remote import Remote, RemoteError, pull
from fplt.settings import Settings, get_settings
from fplt.store.catalog import empty_runs_catalog, read_runs_catalog
from fplt.types import GWKey

logger = logging.getLogger(__name__)


def catalog_has_plans(
    catalog: pl.DataFrame,
    models: Sequence[str],
    key: GWKey,
    horizon: int,
) -> bool:
    """Return whether every model has a succeeded plan row for ``key``.

    Args:
        catalog: ``CatalogRunSchema`` rows (possibly empty).
        models: YAML spec names (matched on ``spec_label``).
        key: Deadline to look up.
        horizon: Requested plan horizon (matched on ``plan_horizon``, falling
            back to ``horizon`` when ``plan_horizon`` is null).

    Returns:
        True when each model has at least one ``run_type=plan`` row for the
        season/gw/horizon key.
    """
    if not models or catalog.is_empty() or horizon < 1:
        return False
    horizon_match = (
        pl.when(pl.col("plan_horizon").is_not_null())
        .then(pl.col("plan_horizon") == horizon)
        .otherwise(pl.col("horizon") == horizon)
    )
    plans = catalog.filter(
        (pl.col("run_type") == "plan")
        & (pl.col("as_of_season") == key.season)
        & (pl.col("as_of_gw") == key.gw)
        & horizon_match
    )
    if plans.is_empty():
        return False
    labels = set(plans.get_column("spec_label").to_list())
    return all(name in labels for name in models)


def catalog_has_predicts(
    catalog: pl.DataFrame,
    models: Sequence[str],
    key: GWKey,
    horizon: int,
) -> bool:
    """Return whether every model has a succeeded predict row for ``key``.

    Used by catch-up done checks (#381): past checked GWs need ledger /
    outcome coverage, not a plan against the live squad.

    Args:
        catalog: ``CatalogRunSchema`` rows (possibly empty).
        models: YAML spec names (matched on ``spec_label``).
        key: Deadline to look up.
        horizon: Walk-forward / predict horizon (matched on ``horizon``).

    Returns:
        True when each model has at least one ``run_type=predict`` row for the
        season/gw/horizon key.
    """
    if not models or catalog.is_empty() or horizon < 1:
        return False
    predicts = catalog.filter(
        (pl.col("run_type") == "predict")
        & (pl.col("as_of_season") == key.season)
        & (pl.col("as_of_gw") == key.gw)
        & (pl.col("horizon") == horizon)
    )
    if predicts.is_empty():
        return False
    labels = set(predicts.get_column("spec_label").to_list())
    return all(name in labels for name in models)


def plan_index_remotes(
    settings: Settings | None = None,
    *,
    remotes: Sequence[Remote] | None = None,
) -> list[Remote]:
    """Remotes that may hold a durable plan catalog.

    Prefers the lab bucket (catch-up adhoc plans) then the data bucket
    (official plans). Returns ``remotes`` unchanged when the caller injects
    them (tests).

    Args:
        settings: Process settings; defaults to ``get_settings()``.
        remotes: Optional pre-built remotes (tests).

    Returns:
        Zero or more configured remotes. Empty when no data bucket is set and
        no remotes were injected.
    """
    if remotes is not None:
        return list(remotes)
    cfg = get_settings() if settings is None else settings
    if not cfg.data_bucket:
        return []
    out: list[Remote] = []
    seen: set[str] = set()
    for bucket in (cfg.lab_bucket, cfg.data_bucket):
        if not bucket:
            continue
        target = Remote.from_settings(cfg, bucket=bucket)
        if target is None or target.target in seen:
            continue
        seen.add(target.target)
        out.append(target)
    return out


def fetch_remote_plan_catalog(
    *,
    settings: Settings | None = None,
    remotes: Sequence[Remote] | None = None,
    cache_dir: Path | None = None,
) -> pl.DataFrame:
    """Pull ``catalog/`` from configured remotes and return the unioned frame.

    Soft-fails per remote (logs and continues) so a missing lab catalog does
    not block a data-bucket hit or local-only behaviour.

    Args:
        settings: Process settings; defaults to ``get_settings()``.
        remotes: Optional pre-built remotes (tests).
        cache_dir: Parent for per-remote pull dirs; defaults to a fresh temp
            directory that is not cleaned up by this call (OS temp).

    Returns:
        ``CatalogRunSchema`` rows, possibly empty.
    """
    targets = plan_index_remotes(settings, remotes=remotes)
    if not targets:
        return empty_runs_catalog()
    root = (
        cache_dir
        if cache_dir is not None
        else Path(tempfile.mkdtemp(prefix="fplt-plan-idx-"))
    )
    frames: list[pl.DataFrame] = []
    for remote in targets:
        dest = root / remote.kind
        catalog_dest = dest / "catalog"
        try:
            pull(remote, "catalog", catalog_dest)
        except RemoteError as exc:
            logger.info(
                "plan_index: catalog pull failed for %s (%s); skipping",
                remote.target,
                exc,
            )
            continue
        frame = read_runs_catalog(dest)
        if not frame.is_empty():
            frames.append(frame)
            logger.info(
                "plan_index: loaded %s plan-index rows from %s",
                frame.filter(pl.col("run_type") == "plan").height,
                remote.target,
            )
    if not frames:
        return empty_runs_catalog()
    if len(frames) == 1:
        return frames[0]
    return (
        pl.concat(frames, how="diagonal_relaxed")
        .unique(subset=["run_id"], keep="last")
        .sort("run_id")
    )


def resolve_plan_catalog(
    *,
    settings: Settings | None = None,
    remotes: Sequence[Remote] | None = None,
    remote_catalog: pl.DataFrame | None = None,
    cache_dir: Path | None = None,
) -> pl.DataFrame:
    """Return a plan-index catalog when remote consultation is enabled.

    Args:
        settings: Process settings; defaults to ``get_settings()``.
        remotes: Optional pre-built remotes (tests).
        remote_catalog: Preloaded catalog (tests / callers that already
            fetched once for a batch of gameweeks).
        cache_dir: Optional pull cache for ``fetch_remote_plan_catalog``.

    Returns:
        Catalog rows when a data bucket is configured (or remotes /
        ``remote_catalog`` were injected); otherwise an empty frame.
    """
    if remote_catalog is not None:
        cfg = get_settings() if settings is None else settings
        if remotes is not None or cfg.data_bucket:
            return remote_catalog
        return empty_runs_catalog()
    return fetch_remote_plan_catalog(
        settings=settings, remotes=remotes, cache_dir=cache_dir
    )
