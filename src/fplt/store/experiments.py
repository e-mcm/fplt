"""Write-once experiment / batch storage under ``experiments/`` and ``batches/``.

Mints ``experiment_id``, freezes gates at submit, appends event files for
status, and looks up arm runs by the exact reuse key from
``docs/proposals/fplt-data-model.md`` §3.8. No bucket locks here — the owner
adds ``experiments/`` / ``batches/`` locks later.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from fplt.schemas.experiment_results import (
    ArmMetricsSchema,
    PairedMetricsSchema,
    PerUnitDiffSchema,
)
from fplt.schemas.manifests import (
    AsOfDeadline,
    BatchManifest,
    ExperimentArm,
    ExperimentEvent,
    ExperimentEventKind,
    ExperimentManifest,
    ExperimentStatusName,
    ExperimentVerdict,
    ExperimentWindow,
    RunManifest,
)
from fplt.store.runs import detect_host, mint_run_id

logger = logging.getLogger(__name__)

_MANIFEST = "manifest.json"
_SUCCESS = "_SUCCESS"
_EVENTS = "events"
_RESULTS = "results"


@dataclass(frozen=True)
class ArmReuseKey:
    """Exact identity for catalog arm reuse.

    Match requires every field equal: ``(spec_hash, code_sha, lockfile_hash,
    build_id, deadlines, horizon, seeds)``.
    """

    spec_hash: str
    code_sha: str
    lockfile_hash: str
    build_id: str | None
    deadlines: tuple[tuple[str, int, str], ...]
    horizon: int
    seeds: tuple[tuple[str, int], ...]


@dataclass(frozen=True)
class FoldedStatus:
    """Derived experiment status from the event set."""

    status: ExperimentStatusName
    arms_succeeded: int
    arms_total: int
    label: str


def mint_experiment_id(
    *, when: datetime | None = None, suffix: str | None = None
) -> str:
    """Mint a time-sortable experiment id: ``exp_<stamp>_<6 hex>``."""
    return _mint_id("exp", when=when, suffix=suffix)


def mint_batch_id(*, when: datetime | None = None, suffix: str | None = None) -> str:
    """Mint a time-sortable batch id: ``batch_<stamp>_<6 hex>``."""
    return _mint_id("batch", when=when, suffix=suffix)


def experiment_dir(data_dir: Path, experiment_id: str) -> Path:
    """Local path ``data_dir/experiments/exp=<experiment_id>/``."""
    return data_dir / "experiments" / f"exp={experiment_id}"


def batch_dir(data_dir: Path, batch_id: str) -> Path:
    """Local path ``data_dir/batches/batch=<batch_id>/``."""
    return data_dir / "batches" / f"batch={batch_id}"


def arm_reuse_key(
    *,
    spec_hash: str,
    code_sha: str,
    lockfile_hash: str,
    build_id: str | None,
    deadlines: Sequence[AsOfDeadline] | Sequence[tuple[str, int, datetime]],
    horizon: int,
    seeds: dict[str, int],
) -> ArmReuseKey:
    """Build the exact arm-reuse key from manifest fields.

    Args:
        spec_hash: Resolved ModelSpec / PipelineSpec hash.
        code_sha: Git HEAD that produced the arm.
        lockfile_hash: ``uv.lock`` digest.
        build_id: Curated build id, or ``None`` when none pinned.
        deadlines: ``AsOfDeadline`` rows or ``(season, gw, deadline_utc)``.
        horizon: Prediction / plan horizon.
        seeds: Named RNG seeds.

    Returns:
        Frozen ``ArmReuseKey``.
    """
    normalized: list[tuple[str, int, str]] = []
    for item in deadlines:
        if isinstance(item, AsOfDeadline):
            season, gw, deadline = item.season, item.gw, item.deadline_utc
        else:
            season, gw, deadline = item
        stamp = _aware(deadline).isoformat().replace("+00:00", "Z")
        normalized.append((season, gw, stamp))
    return ArmReuseKey(
        spec_hash=spec_hash,
        code_sha=code_sha,
        lockfile_hash=lockfile_hash,
        build_id=build_id,
        deadlines=tuple(sorted(normalized)),
        horizon=horizon,
        seeds=tuple(sorted(seeds.items())),
    )


def arm_reuse_key_from_manifest(manifest: RunManifest) -> ArmReuseKey:
    """Derive the reuse key from a succeeded run manifest."""
    return arm_reuse_key(
        spec_hash=manifest.spec_hash,
        code_sha=manifest.code_sha,
        lockfile_hash=manifest.lockfile_hash,
        build_id=manifest.build_id,
        deadlines=manifest.as_of,
        horizon=manifest.horizon,
        seeds=dict(manifest.seeds),
    )


def find_reusable_arm(data_dir: Path, key: ArmReuseKey) -> str | None:
    """Return a succeeded ``run_id`` whose reuse key equals ``key``, or ``None``.

    Exact match only — any field difference (including ``build_id is None`` vs
    a concrete id, or a single seed) is a miss.
    """
    root = data_dir / "runs"
    if not root.is_dir():
        return None
    for path in sorted(root.glob("run=*")):
        if not path.is_dir() or not (path / _SUCCESS).is_file():
            continue
        manifest_path = path / _MANIFEST
        if not manifest_path.is_file():
            continue
        try:
            manifest = RunManifest.model_validate_json(
                manifest_path.read_text(encoding="utf-8")
            )
        except Exception:
            logger.debug("skip unreadable run manifest %s", manifest_path)
            continue
        if manifest.status != "succeeded":
            continue
        if arm_reuse_key_from_manifest(manifest) == key:
            return manifest.run_id
    return None


def fold_experiment_status(
    events: Sequence[ExperimentEvent],
    *,
    n_arms: int,
) -> FoldedStatus:
    """Fold write-once events into planned / running / analysed / failed / aborted.

    Folding is set-based: duplicate ``(event, arm)`` pairs are ignored, and
    arrival order does not matter (out-of-order events still fold correctly).

    Args:
        events: Event payloads (order ignored).
        n_arms: Declared arm count on the manifest.

    Returns:
        ``FoldedStatus`` with a human ``label`` such as ``running (2/3 arms)``.
    """
    if n_arms < 1:
        raise ValueError("n_arms must be >= 1")
    unique: set[tuple[ExperimentEventKind, str | None]] = set()
    for event in events:
        unique.add((event.event, event.arm))

    kinds = {kind for kind, _ in unique}
    succeeded = {arm for kind, arm in unique if kind == "arm_succeeded" and arm}
    failed = {arm for kind, arm in unique if kind == "arm_failed" and arm}
    # A later succeed wins over a failed duplicate for the same arm.
    failed -= succeeded
    k = len(succeeded)

    if "aborted" in kinds:
        return FoldedStatus("aborted", k, n_arms, "aborted")
    if "analysed" in kinds:
        return FoldedStatus("analysed", k, n_arms, "analysed")
    if not unique:
        return FoldedStatus("planned", 0, n_arms, "planned")

    terminal = succeeded | failed
    if len(terminal) >= n_arms and failed:
        return FoldedStatus("failed", k, n_arms, "failed")
    if "submitted" in kinds or any(kind.startswith("arm_") for kind in kinds):
        return FoldedStatus(
            "running",
            k,
            n_arms,
            f"running ({k}/{n_arms} arms)",
        )
    return FoldedStatus("planned", 0, n_arms, "planned")


def hash_eval_spec_payload(payload: dict[str, Any]) -> str:
    """SHA-256 of a canonical EvalSpec / gates JSON payload."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(raw.encode()).hexdigest()


def write_experiment_manifest(data_dir: Path, manifest: ExperimentManifest) -> Path:
    """Write ``manifest.json`` once under ``experiments/exp=<id>/``.

    Raises:
        FileExistsError: The experiment directory already exists.
    """
    root = experiment_dir(data_dir, manifest.experiment_id)
    if root.exists():
        raise FileExistsError(f"experiment already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    (root / _EVENTS).mkdir()
    (root / _RESULTS).mkdir()
    path = root / _MANIFEST
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def append_experiment_event(
    data_dir: Path,
    experiment_id: str,
    event: ExperimentEvent,
) -> Path:
    """Append one write-once event file; never overwrites an existing name."""
    root = experiment_dir(data_dir, experiment_id)
    if not root.is_dir():
        raise FileNotFoundError(f"experiment not found: {experiment_id}")
    stamp = _aware(event.at).strftime("%Y%m%dT%H%M%S%fZ")
    arm_bit = f"_{event.arm}" if event.arm else ""
    name = f"{stamp}_{event.event}{arm_bit}.json"
    path = root / _EVENTS / name
    if path.exists():
        # Extremely unlikely collision on same microsecond; add entropy.
        name = f"{stamp}_{event.event}{arm_bit}_{secrets.token_hex(2)}.json"
        path = root / _EVENTS / name
    path.write_text(event.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def load_experiment_manifest(data_dir: Path, experiment_id: str) -> ExperimentManifest:
    """Load and validate an experiment manifest."""
    path = experiment_dir(data_dir, experiment_id) / _MANIFEST
    return ExperimentManifest.model_validate_json(path.read_text(encoding="utf-8"))


def load_experiment_events(data_dir: Path, experiment_id: str) -> list[ExperimentEvent]:
    """Load all event files for an experiment (filesystem order)."""
    folder = experiment_dir(data_dir, experiment_id) / _EVENTS
    if not folder.is_dir():
        return []
    events: list[ExperimentEvent] = []
    for path in sorted(folder.glob("*.json")):
        events.append(
            ExperimentEvent.model_validate_json(path.read_text(encoding="utf-8"))
        )
    return events


def experiment_status(data_dir: Path, experiment_id: str) -> FoldedStatus:
    """Fold on-disk events into the current experiment status."""
    manifest = load_experiment_manifest(data_dir, experiment_id)
    events = load_experiment_events(data_dir, experiment_id)
    return fold_experiment_status(events, n_arms=len(manifest.arms))


def write_batch_manifest(data_dir: Path, manifest: BatchManifest) -> Path:
    """Write ``batches/batch=<id>/manifest.json`` once."""
    root = batch_dir(data_dir, manifest.batch_id)
    if root.exists():
        raise FileExistsError(f"batch already exists: {root}")
    root.mkdir(parents=True, exist_ok=False)
    (root / _EVENTS).mkdir()
    path = root / _MANIFEST
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return path


def write_experiment_results(
    data_dir: Path,
    experiment_id: str,
    *,
    arm_metrics: pl.DataFrame | None = None,
    paired: pl.DataFrame | None = None,
    per_unit: pl.DataFrame | None = None,
    verdict: ExperimentVerdict | None = None,
    report_md: str | None = None,
) -> Path:
    """Write standard ``results/`` tables (write-once per filename).

    Args:
        data_dir: Local data root.
        experiment_id: Target experiment.
        arm_metrics: Optional ``ArmMetricsSchema`` rows.
        paired: Optional ``PairedMetricsSchema`` rows.
        per_unit: Optional ``PerUnitDiffSchema`` rows.
        verdict: Optional computed verdict JSON.
        report_md: Optional ``report.md`` body.

    Returns:
        The experiment ``results/`` directory.
    """
    root = experiment_dir(data_dir, experiment_id) / _RESULTS
    if not root.is_dir():
        raise FileNotFoundError(f"experiment results dir missing: {experiment_id}")
    if arm_metrics is not None:
        checked = ArmMetricsSchema.validate(arm_metrics)
        _write_parquet_once(root / "arm_metrics.parquet", checked)
    if paired is not None:
        checked = PairedMetricsSchema.validate(paired)
        _write_parquet_once(root / "paired.parquet", checked)
    if per_unit is not None:
        checked = PerUnitDiffSchema.validate(per_unit)
        _write_parquet_once(root / "per_unit.parquet", checked)
    if verdict is not None:
        path = root / "verdict.json"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(verdict.model_dump_json(indent=2) + "\n", encoding="utf-8")
    if report_md is not None:
        path = root / "report.md"
        if path.exists():
            raise FileExistsError(path)
        path.write_text(report_md, encoding="utf-8")
    return root


def submit_experiment(
    data_dir: Path,
    *,
    flavour: str,
    question: str,
    arms: Sequence[ExperimentArm],
    windows: Sequence[ExperimentWindow],
    horizon: int,
    code_sha: str,
    lockfile_hash: str,
    gates: dict[str, Any],
    metrics: Sequence[str] | None = None,
    seeds: dict[str, int] | None = None,
    build_id: str | None = None,
    as_of: Sequence[AsOfDeadline] | None = None,
    eval_spec: str | None = None,
    eval_spec_hash: str | None = None,
    batch_id: str | None = None,
    submitted_by: str = "unknown",
    host: str | None = None,
    experiment_id: str | None = None,
    submitted_at: datetime | None = None,
) -> ExperimentManifest:
    """Mint an experiment, write the frozen manifest, and emit ``submitted``.

    Gates and ``eval_spec_hash`` are snapshotted here; later YAML edits do not
    change the on-disk contract.
    """
    when = _aware(submitted_at if submitted_at is not None else datetime.now(UTC))
    exp_id = (
        experiment_id if experiment_id is not None else mint_experiment_id(when=when)
    )
    manifest = ExperimentManifest(
        experiment_id=exp_id,
        flavour=flavour,  # type: ignore[arg-type]
        question=question,
        batch_id=batch_id,
        eval_spec=eval_spec,
        eval_spec_hash=eval_spec_hash,
        arms=list(arms),
        build_id=build_id,
        windows=list(windows),
        as_of=list(as_of or []),
        horizon=horizon,
        seeds=dict(seeds or {}),
        metrics=list(metrics or []),
        gates=dict(gates),
        code_sha=code_sha,
        lockfile_hash=lockfile_hash,
        submitted_by=submitted_by,
        host=host if host is not None else detect_host(),
        submitted_at=when,
    )
    write_experiment_manifest(data_dir, manifest)
    append_experiment_event(
        data_dir,
        exp_id,
        ExperimentEvent(event="submitted", at=when),
    )
    return manifest


def bind_arm_run(
    data_dir: Path,
    experiment_id: str,
    *,
    arm: str,
    run_id: str,
    reused: bool,
    at: datetime | None = None,
) -> Path:
    """Record ``arm_succeeded`` and a write-once ``arms/<arm>.json`` pointer.

    The experiment manifest stays immutable; per-arm pointer files carry the
    resolved ``run_id`` without rewriting ``manifest.json``.
    """
    when = _aware(at if at is not None else datetime.now(UTC))
    root = experiment_dir(data_dir, experiment_id)
    arms_dir = root / "arms"
    arms_dir.mkdir(exist_ok=True)
    pointer = arms_dir / f"{arm}.json"
    if pointer.exists():
        raise FileExistsError(pointer)
    pointer.write_text(
        json.dumps(
            {"arm": arm, "run_id": run_id, "reused": reused},
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    append_experiment_event(
        data_dir,
        experiment_id,
        ExperimentEvent(
            event="arm_succeeded",
            at=when,
            arm=arm,
            run_id=run_id,
            detail="reused" if reused else "fresh",
        ),
    )
    return pointer


def resolve_or_create_arm_run(
    data_dir: Path,
    *,
    key: ArmReuseKey,
    kind: str = "experiment",
    mode: str = "backtest",
    run_type: str = "predict",
    spec_label: str,
    model_id: str,
    experiment_id: str | None = None,
    arm: str | None = None,
    purpose: str = "",
    host: str | None = None,
) -> tuple[str, bool]:
    """Return ``(run_id, reused)`` for an arm, creating a succeeded shell run if needed.

    A shell run carries the reuse key in its manifest so later experiments can
    hit the catalog. Prediction tables may be empty until a later producer
    dual-writes the ledger into the run.
    """
    from fplt.utils.logging import log_context

    with log_context(experiment_id=experiment_id, spec=spec_label):
        existing = find_reusable_arm(data_dir, key)
        if existing is not None:
            logger.info(
                "experiment arm reused run_id=%s experiment_id=%s arm=%s",
                existing,
                experiment_id,
                arm,
            )
            return existing, True

        from fplt.models.predictions import empty_predictions
        from fplt.store.runs import RunWriter

        as_of = [
            AsOfDeadline(
                season=season,
                gw=gw,
                deadline_utc=datetime.fromisoformat(stamp.replace("Z", "+00:00")),
            )
            for season, gw, stamp in key.deadlines
        ]
        if not as_of:
            raise ValueError("arm reuse key requires at least one deadline")
        run_id = mint_run_id()
        with RunWriter(
            data_dir,
            kind=kind,  # type: ignore[arg-type]
            mode=mode,  # type: ignore[arg-type]
            run_type=run_type,  # type: ignore[arg-type]
            spec_hash=key.spec_hash,
            spec_label=spec_label,
            model_id=model_id,
            as_of=as_of,
            horizon=key.horizon,
            purpose=purpose,
            seeds=dict(key.seeds),
            run_id=run_id,
            code_sha=key.code_sha,
            code_dirty=False,
            lockfile_hash=key.lockfile_hash,
            build_id=key.build_id,
            auto_build_id=False,
            host=host,
            experiment_id=experiment_id,
            arm=arm,
            raw_manifest_hash="0" * 64,
            staged_digests={},
        ) as writer:
            writer.write_parquet(
                "predictions/player_fixture.parquet",
                empty_predictions(),
            )
        return writer.run_id, False


def frozen_gates(manifest: ExperimentManifest) -> dict[str, Any]:
    """Return the gates dict frozen at submit (copy)."""
    return dict(manifest.gates)


def _mint_id(prefix: str, *, when: datetime | None, suffix: str | None) -> str:
    moment = _aware(when if when is not None else datetime.now(UTC))
    stamp = moment.strftime("%Y%m%dT%H%M%SZ")
    entropy = suffix if suffix is not None else secrets.token_hex(3)
    if len(entropy) != 6 or any(c not in "0123456789abcdef" for c in entropy):
        raise ValueError(f"{prefix} suffix must be 6 lowercase hex chars")
    return f"{prefix}_{stamp}_{entropy}"


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


def _write_parquet_once(path: Path, frame: pl.DataFrame) -> None:
    if path.exists():
        raise FileExistsError(path)
    frame.write_parquet(path)


def list_experiment_ids(data_dir: Path) -> list[str]:
    """Sorted experiment ids present under ``experiments/``."""
    root = data_dir / "experiments"
    if not root.is_dir():
        return []
    return sorted(
        path.name.removeprefix("exp=")
        for path in root.glob("exp=*")
        if path.is_dir() and (path / _MANIFEST).is_file()
    )


__all__ = [
    "ArmReuseKey",
    "FoldedStatus",
    "append_experiment_event",
    "arm_reuse_key",
    "arm_reuse_key_from_manifest",
    "batch_dir",
    "bind_arm_run",
    "experiment_dir",
    "experiment_status",
    "find_reusable_arm",
    "fold_experiment_status",
    "frozen_gates",
    "hash_eval_spec_payload",
    "list_experiment_ids",
    "load_experiment_events",
    "load_experiment_manifest",
    "mint_batch_id",
    "mint_experiment_id",
    "resolve_or_create_arm_run",
    "submit_experiment",
    "write_batch_manifest",
    "write_experiment_manifest",
    "write_experiment_results",
]
