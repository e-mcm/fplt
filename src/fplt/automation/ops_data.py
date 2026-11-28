"""Ops data path: curated latest, required-tables pull, throttle, freshness.

Ops never pulls ``raw/`` (#346). Ticks read ``status/curated/latest.json`` and
the build manifest, skip when ``runs/by_fp/`` already has the fingerprint, and
otherwise pull only ``required_tables(spec)`` into ``staged/``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path

from fplt.automation.curate import (
    load_curated_latest,
    pull_curated_latest,
)
from fplt.automation.notify import notify_recommendation
from fplt.automation.production import OfficialLine, ProductionOps
from fplt.data.remote import Remote, RemoteError, lsf
from fplt.models.spec import ModelSpec, load_model_specs
from fplt.schemas.manifests import BuildManifest, CuratedLatestPointer
from fplt.settings import Settings, get_settings
from fplt.store.builds import (
    build_manifest_path,
    hydrate_staged_from_build,
    load_build_manifest,
)
from fplt.store.fingerprint import (
    fp_index_exists,
    required_tables,
    run_fingerprint,
)
from fplt.store.pull_official import pull_build
from fplt.types import GWKey

logger = logging.getLogger(__name__)

_THROTTLE_RELATIVE = "status/ops/last_run_by_line.json"


@dataclass(frozen=True)
class OpsBuildContext:
    """Curated build loaded for one ops tick.

    Attributes:
        pointer: ``status/curated/latest.json`` contents.
        build: Parsed build manifest (table shas).
        remote_by_fp: Remote ``runs/by_fp/`` listing when available.
    """

    pointer: CuratedLatestPointer
    build: BuildManifest
    remote_by_fp: frozenset[str]


@dataclass(frozen=True)
class LineFpDecision:
    """Skip / run decision for one official line.

    Attributes:
        line: Official line name.
        spec_name: ModelSpec name.
        predict_fp: Predict-run fingerprint.
        plan_fp: Plan-run fingerprint.
        skip: True when both fps already exist in ``by_fp``.
        predict_run_id: Existing predict run when skipped.
        plan_run_id: Existing plan run when skipped.
    """

    line: str
    spec_name: str
    predict_fp: str
    plan_fp: str
    skip: bool
    predict_run_id: str | None = None
    plan_run_id: str | None = None


class StaleBuildError(RuntimeError):
    """Refresh checkpoint refused a build older than ``max_build_age``."""


def load_ops_build_context(
    data_dir: Path,
    *,
    settings: Settings | None = None,
    remote: Remote | None = None,
) -> OpsBuildContext | None:
    """Pull curated latest + build manifest (+ optional by_fp listing).

    Never pulls ``raw/``. Returns ``None`` when no curated pointer exists.

    Args:
        data_dir: Local data root.
        settings: Process settings for the data-bucket remote.
        remote: Optional pre-built data remote (tests).

    Returns:
        Context, or ``None`` when curated latest is missing.
    """
    cfg = get_settings() if settings is None else settings
    target = remote
    if target is None:
        target = Remote.from_settings(cfg, bucket=cfg.data_bucket)
    pointer = load_curated_latest(data_dir)
    if target is not None:
        pulled = pull_curated_latest(data_dir, target)
        if pulled is not None:
            pointer = pulled
    if pointer is None:
        logger.info("ops_data: no status/curated/latest.json")
        return None

    _ensure_build_manifest(data_dir, pointer.build_id, remote=target)
    build = load_build_manifest(data_dir, pointer.build_id)
    remote_by_fp: frozenset[str] = frozenset()
    if target is not None:
        try:
            remote_by_fp = frozenset(lsf(target, "runs/by_fp", recursive=False))
        except RemoteError as exc:
            logger.info("ops_data: by_fp lsf failed (%s); using local index", exc)
    return OpsBuildContext(pointer=pointer, build=build, remote_by_fp=remote_by_fp)


def pull_required_tables(
    data_dir: Path,
    build_id: str,
    tables: set[str],
    *,
    settings: Settings | None = None,
    remote: Remote | None = None,
) -> list[str]:
    """Pull ``tables`` for ``build_id`` and hydrate them into ``staged/``.

    Never pulls ``raw/``. Only the named curated table sha files are fetched.

    Args:
        data_dir: Local data root.
        build_id: Curated build id.
        tables: Staged table names to pull (``required_tables(spec)``).
        settings: Process settings for the data-bucket remote.
        remote: Optional pre-built data remote (tests).

    Returns:
        Sorted table names hydrated into ``staged/``.

    Raises:
        OfficialPullError: Manifest or a required table sha is missing remotely.
        FileNotFoundError: Local hydrate source is missing after pull.
    """
    cfg = get_settings() if settings is None else settings
    target = remote
    if target is None:
        target = Remote.from_settings(cfg, bucket=cfg.data_bucket)
    if target is not None:
        # Manifest first so we can intersect with tables actually in the build
        # (empty staged tables are omitted from builds — #345 / #346).
        pull_build(data_dir, target, build_id, tables=frozenset())
    elif not build_manifest_path(data_dir, build_id).is_file():
        raise FileNotFoundError(build_manifest_path(data_dir, build_id))
    build = load_build_manifest(data_dir, build_id)
    allow = frozenset(set(tables) & set(build.tables))
    if target is not None and allow:
        pull_build(data_dir, target, build_id, tables=allow)
    if not allow:
        return []
    return hydrate_staged_from_build(data_dir, build_id, tables=allow)


def decisions_for_slate(
    ops: ProductionOps,
    build: BuildManifest,
    key: GWKey,
    *,
    data_dir: Path,
    prod_tag: str | None = None,
    remote_by_fp: frozenset[str] | set[str] | None = None,
    specs: dict[str, ModelSpec] | None = None,
) -> list[LineFpDecision]:
    """Compute predict/plan fingerprints for every official line.

    Args:
        ops: Production slate.
        build: Curated build for table shas.
        key: Target gameweek.
        data_dir: Local data root for ``by_fp`` lookups.
        prod_tag: Moving ops tag when known.
        remote_by_fp: Optional remote listing of ``runs/by_fp/``.
        specs: Optional preloaded ModelSpecs.

    Returns:
        One decision per official line.
    """
    loaded = load_model_specs() if specs is None else specs
    listing = set(remote_by_fp) if remote_by_fp is not None else None
    out: list[LineFpDecision] = []
    for line in ops.lines:
        spec = loaded[line.spec]
        predict_fp = run_fingerprint(
            spec=spec,
            build=build,
            target_gw=key,
            run_type="predict",
            prod_tag=prod_tag,
        )
        plan_fp = run_fingerprint(
            spec=spec,
            build=build,
            target_gw=key,
            run_type="plan",
            prod_tag=prod_tag,
        )
        pred_done = fp_index_exists(data_dir, predict_fp, remote_listing=listing)
        plan_done = fp_index_exists(data_dir, plan_fp, remote_listing=listing)
        pred_id = _local_run_id(data_dir, predict_fp) if pred_done else None
        plan_id = _local_run_id(data_dir, plan_fp) if plan_done else None
        out.append(
            LineFpDecision(
                line=line.line,
                spec_name=line.spec,
                predict_fp=predict_fp,
                plan_fp=plan_fp,
                skip=pred_done and plan_done,
                predict_run_id=pred_id,
                plan_run_id=plan_id,
            )
        )
    return out


def tables_for_slate(
    ops: ProductionOps, specs: dict[str, ModelSpec] | None = None
) -> set[str]:
    """Union of ``required_tables`` across every distinct slate spec."""
    loaded = load_model_specs() if specs is None else specs
    tables: set[str] = set()
    for name in ops.predict_groups():
        tables |= required_tables(loaded[name])
    return tables


def build_is_fresh(
    pointer: CuratedLatestPointer,
    *,
    now: datetime,
    max_age: timedelta,
) -> bool:
    """Return whether the curated build is newer than ``max_age``."""
    created = pointer.created_at
    if created.tzinfo is None:
        created = created.replace(tzinfo=UTC)
    moment = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    return (moment - created) <= max_age


def enforce_build_freshness(
    pointer: CuratedLatestPointer,
    *,
    now: datetime,
    ops: ProductionOps,
    settings: Settings,
    notify: bool,
    checkpoint: str,
) -> None:
    """Refuse stale builds at refresh checkpoints (T−90m / T−30m).

    Raises:
        StaleBuildError: Build older than ``ops.max_build_age_minutes``.
    """
    max_age = timedelta(minutes=ops.max_build_age_minutes)
    if build_is_fresh(pointer, now=now, max_age=max_age):
        return
    age = now.astimezone(UTC) - (
        pointer.created_at
        if pointer.created_at.tzinfo is not None
        else pointer.created_at.replace(tzinfo=UTC)
    )
    message = (
        f"curated build {pointer.build_id} is stale for {checkpoint}: "
        f"age={age} > max_build_age={max_age}"
    )
    logger.error("ops_data: %s", message)
    if notify:
        notify_recommendation(
            settings,
            f"fplt stale build {checkpoint}",
            message,
            priority="5",
        )
    raise StaleBuildError(message)


def throttle_allows(
    data_dir: Path,
    line: str,
    *,
    now: datetime,
    min_interval: timedelta,
    checkpoint: bool,
) -> bool:
    """Return whether a non-checkpoint run is allowed for ``line``.

    Checkpoints always return ``True``. Outside checkpoints, at most one run
    per line per ``min_interval`` (#346).
    """
    if checkpoint:
        return True
    stamp = _load_last_run(data_dir).get(line)
    if stamp is None:
        return True
    last = datetime.fromisoformat(stamp)
    if last.tzinfo is None:
        last = last.replace(tzinfo=UTC)
    moment = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    return (moment - last) >= min_interval


def record_line_run(data_dir: Path, line: str, *, now: datetime) -> None:
    """Record that ``line`` produced a run at ``now`` (throttle clock)."""
    path = data_dir / _THROTTLE_RELATIVE
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _load_last_run(data_dir)
    moment = now if now.tzinfo is not None else now.replace(tzinfo=UTC)
    payload[line] = moment.astimezone(UTC).isoformat()
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def push_fp_index(
    data_dir: Path,
    fp: str,
    *,
    settings: Settings | None = None,
    remote: Remote | None = None,
) -> None:
    """Push ``runs/by_fp/`` (including ``fp=<fp>.json``) when configured."""
    from fplt.data.remote import push
    from fplt.store.fingerprint import by_fp_path

    path = by_fp_path(data_dir, fp)
    if not path.is_file():
        return
    cfg = get_settings() if settings is None else settings
    target = remote if remote is not None else Remote.from_settings(cfg)
    if target is None:
        return
    # Immutable directory push; existing index entries are never overwritten.
    push(target, path.parent, "runs/by_fp")


def _ensure_build_manifest(
    data_dir: Path, build_id: str, *, remote: Remote | None
) -> None:
    path = build_manifest_path(data_dir, build_id)
    if path.is_file():
        return
    if remote is None:
        raise FileNotFoundError(path)
    pull_build(data_dir, remote, build_id, tables=frozenset())


def _local_run_id(data_dir: Path, fp: str) -> str | None:
    from fplt.store.fingerprint import load_fp_index

    entry = load_fp_index(data_dir, fp)
    return None if entry is None else entry.run_id


def _load_last_run(data_dir: Path) -> dict[str, str]:
    from typing import cast

    path = data_dir / _THROTTLE_RELATIVE
    if not path.is_file():
        return {}
    raw: object = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        return {}
    typed = cast(dict[object, object], raw)
    return {str(key): str(value) for key, value in typed.items()}


def line_tables(
    line: OfficialLine, specs: dict[str, ModelSpec] | None = None
) -> set[str]:
    """``required_tables`` for one official line's ModelSpec."""
    loaded = load_model_specs() if specs is None else specs
    return required_tables(loaded[line.spec])
