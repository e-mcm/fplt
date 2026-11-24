"""Write-once analytic runs under ``runs/run=<run_id>/``.

``RunWriter`` mints a sortable ``run_id``, writes ``manifest.json`` first with
``status=running``, accepts parquet tables, then finishes with the final
manifest and ``_SUCCESS`` (or ``status=failed`` without the marker). Until
curated builds exist (B6), manifests record ``raw_manifest_hash`` plus the
staged-table digests from ``hash_staged_data``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Self

import polars as pl

from fplt.backtest.produce_ledger import (
    hash_staged_data,
    resolved_spec_sha256,
    uv_lock_sha256,
)
from fplt.models.spec import ModelSpec, git_sha
from fplt.schemas.manifests import (
    AsOfDeadline,
    RunKind,
    RunManifest,
    RunMode,
    RunStatus,
    RunType,
)
from fplt.types import GWKey

logger = logging.getLogger(__name__)

_SUCCESS = "_SUCCESS"
_MANIFEST = "manifest.json"
_RUN_ID_RE_PREFIX = "r_"


class RunExistsError(FileExistsError):
    """Raised when a run directory already exists (write-once)."""


def mint_run_id(*, when: datetime | None = None, suffix: str | None = None) -> str:
    """Mint a time-sortable globally unique run id.

    Format: ``r_<YYYYMMDDTHHMMSSZ>_<6 hex>`` (UTC timestamp + random entropy).

    Args:
        when: Timestamp embedded in the id. Defaults to now (UTC).
        suffix: Optional 6-character hex. Drawn randomly when omitted.

    Returns:
        A new ``run_id`` string.
    """
    moment = when if when is not None else datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    else:
        moment = moment.astimezone(UTC)
    stamp = moment.strftime("%Y%m%dT%H%M%SZ")
    entropy = suffix if suffix is not None else secrets.token_hex(3)
    if len(entropy) != 6 or any(c not in "0123456789abcdef" for c in entropy):
        raise ValueError(
            f"run_id suffix must be 6 lowercase hex chars, got {entropy!r}"
        )
    return f"{_RUN_ID_RE_PREFIX}{stamp}_{entropy}"


def run_dir(data_dir: Path, run_id: str) -> Path:
    """Local path for one run folder: ``data_dir/runs/run=<run_id>/``."""
    return data_dir / "runs" / f"run={run_id}"


def hash_raw_manifest(data_dir: Path) -> str:
    """SHA-256 of the sorted ``(relpath, size)`` list under ``data_dir/raw/``.

    Matches the build-manifest convention in ``fplt-data-model.md`` §3.2:
    raw is write-once, so path plus size identifies each capture. An empty or
    missing ``raw/`` hashes the empty list.
    """
    raw_root = data_dir / "raw"
    lines: list[str] = []
    if raw_root.is_dir():
        for path in sorted(p for p in raw_root.rglob("*") if p.is_file()):
            rel = path.relative_to(raw_root).as_posix()
            lines.append(f"{rel}\t{path.stat().st_size}")
    payload = "\n".join(lines).encode()
    return hashlib.sha256(payload).hexdigest()


def detect_host() -> str:
    """Host label for the run manifest (``FPLT_HOST`` or ``unknown``)."""
    return os.environ.get("FPLT_HOST", "").strip() or "unknown"


def fplt_mc_version() -> str | None:
    """Installed ``fplt-mc`` version, or ``None`` when the package is absent."""
    try:
        return version("fplt-mc")
    except Exception:
        return None


def _git_is_dirty(*, cwd: Path | None = None) -> bool:
    """Return whether the working tree has uncommitted changes."""
    try:
        completed = subprocess.run(
            ["git", "status", "--porcelain"],
            check=True,
            capture_output=True,
            text=True,
            cwd=cwd,
        )
    except (OSError, subprocess.CalledProcessError):
        return False
    return bool(completed.stdout.strip())


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


class RunWriter:
    """Context manager that writes one immutable run folder.

    On enter: refuse an existing ``run=<id>/``, create the directory, write
    ``manifest.json`` with ``status=running``. On clean exit: rewrite the
    manifest as ``succeeded`` and touch ``_SUCCESS``. On exception: rewrite as
    ``failed`` without ``_SUCCESS``. Tables are write-once inside the folder.
    """

    def __init__(
        self,
        data_dir: Path,
        *,
        kind: RunKind,
        mode: RunMode,
        run_type: RunType,
        spec_hash: str,
        spec_label: str,
        model_id: str,
        as_of: list[AsOfDeadline],
        horizon: int,
        purpose: str = "",
        plan_horizon: int | None = None,
        parent_run_id: str | None = None,
        seeds: dict[str, int] | None = None,
        config_hashes: dict[str, str] | None = None,
        tags: dict[str, str] | None = None,
        run_id: str | None = None,
        code_sha: str | None = None,
        code_dirty: bool | None = None,
        lockfile_hash: str | None = None,
        host: str | None = None,
        raw_manifest_hash: str | None = None,
        staged_digests: dict[str, str] | None = None,
        started_at: datetime | None = None,
    ) -> None:
        """Build a writer; the folder is created only on ``__enter__``.

        Args:
            data_dir: Local data root (``$FPLT_DATA_DIR``).
            kind: Official / adhoc / experiment / benchmark.
            mode: Live or backtest.
            run_type: Predict, plan, or eval.
            spec_hash: ``resolved_spec_sha256`` until B8.
            spec_label: Human label (usually the ModelSpec name).
            model_id: Existing model id (compatibility).
            as_of: Deadlines this run covers.
            horizon: Prediction / plan horizon length.
            purpose: Free-text purpose.
            plan_horizon: Required when ``run_type="plan"``.
            parent_run_id: Predict run a plan reads, when known.
            seeds: Recorded RNG seeds.
            config_hashes: Extra config digests (scoring, ops, …).
            tags: Arbitrary string tags.
            run_id: Optional fixed id (tests). Minted when omitted.
            code_sha: Git HEAD; defaults to ``git_sha()``.
            code_dirty: Working-tree dirty flag; defaults to ``_git_is_dirty()``.
            lockfile_hash: ``uv.lock`` digest; defaults to ``uv_lock_sha256()``.
            host: Manifest host label; defaults to ``detect_host()``.
            raw_manifest_hash: Override; defaults to ``hash_raw_manifest``.
            staged_digests: Override; defaults to ``hash_staged_data``.
            started_at: Manifest start time; defaults to now (UTC).
        """
        self.data_dir = data_dir
        self.run_id = run_id if run_id is not None else mint_run_id()
        self._kind: RunKind = kind
        self._mode: RunMode = mode
        self._run_type: RunType = run_type
        self._spec_hash = spec_hash
        self._spec_label = spec_label
        self._model_id = model_id
        self._as_of = list(as_of)
        self._horizon = horizon
        self._purpose = purpose
        self._plan_horizon = plan_horizon
        self._parent_run_id = parent_run_id
        self._seeds = dict(seeds or {})
        self._config_hashes = dict(config_hashes or {})
        self._tags = dict(tags or {})
        self._code_sha = code_sha if code_sha is not None else git_sha()
        self._code_dirty = code_dirty if code_dirty is not None else _git_is_dirty()
        self._lockfile_hash = (
            lockfile_hash if lockfile_hash is not None else uv_lock_sha256()
        )
        self._host = host if host is not None else detect_host()
        self._raw_manifest_hash = (
            raw_manifest_hash
            if raw_manifest_hash is not None
            else hash_raw_manifest(data_dir)
        )
        self._staged_digests = (
            dict(staged_digests)
            if staged_digests is not None
            else hash_staged_data(data_dir)
        )
        self._started_at = _aware(
            started_at if started_at is not None else datetime.now(UTC)
        )
        self._fplt_mc_version = fplt_mc_version()
        self._manifest: RunManifest | None = None
        self._finalized = False
        self._entered = False

    @property
    def path(self) -> Path:
        """Directory for this run."""
        return run_dir(self.data_dir, self.run_id)

    @property
    def manifest(self) -> RunManifest:
        """Current on-disk manifest (running or final)."""
        if self._manifest is None:
            raise RuntimeError("RunWriter has not been entered")
        return self._manifest

    def __enter__(self) -> Self:
        """Create the run folder and write the running manifest."""
        if self.path.exists():
            raise RunExistsError(f"run already exists: {self.path}")
        self.path.mkdir(parents=True, exist_ok=False)
        self._manifest = self._build_manifest(status="running", finished_at=None)
        self._write_manifest(self._manifest)
        self._entered = True
        logger.info(
            "run started run_id=%s kind=%s run_type=%s path=%s",
            self.run_id,
            self._kind,
            self._run_type,
            self.path,
        )
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object,
    ) -> None:
        """Finalize as succeeded or failed; never suppress the exception."""
        del tb
        if not self._entered or self._finalized:
            return None
        if exc_type is not None:
            self._finalize("failed")
        else:
            self._finalize("succeeded")
        return None

    def write_parquet(self, relative: str, frame: pl.DataFrame) -> Path:
        """Write ``frame`` once under the run directory.

        Args:
            relative: Path relative to the run folder (e.g.
                ``predictions/player_fixture.parquet``).
            frame: Table to persist.

        Returns:
            Absolute path written.

        Raises:
            RuntimeError: The writer is not open or already finalized.
            FileExistsError: That relative path was already written.
            ValueError: ``relative`` escapes the run directory.
        """
        self._require_open()
        dest = self._resolve(relative)
        if dest.exists():
            raise FileExistsError(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(dest)
        return dest

    def write_json(self, relative: str, payload: object) -> Path:
        """Write a JSON document once under the run directory.

        Args:
            relative: Path relative to the run folder.
            payload: JSON-serialisable value.

        Returns:
            Absolute path written.
        """
        self._require_open()
        dest = self._resolve(relative)
        if dest.exists():
            raise FileExistsError(dest)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_text(
            json.dumps(payload, sort_keys=True, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return dest

    def succeed(self) -> RunManifest:
        """Mark the run succeeded and write ``_SUCCESS`` (idempotent)."""
        return self._finalize("succeeded")

    def fail(self) -> RunManifest:
        """Mark the run failed without writing ``_SUCCESS`` (idempotent)."""
        return self._finalize("failed")

    def _finalize(self, status: RunStatus) -> RunManifest:
        if self._finalized:
            assert self._manifest is not None
            return self._manifest
        self._require_open()
        finished = datetime.now(UTC)
        self._manifest = self._build_manifest(status=status, finished_at=finished)
        self._write_manifest(self._manifest)
        if status == "succeeded":
            (self.path / _SUCCESS).write_text("", encoding="utf-8")
        self._finalized = True
        logger.info(
            "run finished run_id=%s status=%s",
            self.run_id,
            status,
        )
        return self._manifest

    def _build_manifest(
        self,
        *,
        status: RunStatus,
        finished_at: datetime | None,
    ) -> RunManifest:
        return RunManifest(
            run_id=self.run_id,
            kind=self._kind,
            mode=self._mode,
            purpose=self._purpose,
            run_type=self._run_type,
            spec_hash=self._spec_hash,
            spec_label=self._spec_label,
            model_id=self._model_id,
            config_hashes=self._config_hashes,
            plan_horizon=self._plan_horizon,
            code_sha=self._code_sha,
            code_dirty=self._code_dirty,
            fplt_mc_version=self._fplt_mc_version,
            lockfile_hash=self._lockfile_hash,
            build_id=None,
            raw_manifest_hash=self._raw_manifest_hash,
            staged_digests=self._staged_digests,
            as_of=self._as_of,
            horizon=self._horizon,
            seeds=self._seeds,
            host=self._host,
            started_at=self._started_at,
            finished_at=finished_at,
            status=status,
            parent_run_id=self._parent_run_id,
            tags=self._tags,
        )

    def _write_manifest(self, manifest: RunManifest) -> None:
        path = self.path / _MANIFEST
        path.write_text(
            manifest.model_dump_json(indent=2) + "\n",
            encoding="utf-8",
        )

    def _resolve(self, relative: str) -> Path:
        cleaned = relative.strip().lstrip("/")
        if not cleaned or cleaned == _MANIFEST or cleaned == _SUCCESS:
            raise ValueError(f"reserved or empty run path: {relative!r}")
        dest = (self.path / cleaned).resolve()
        if not dest.is_relative_to(self.path.resolve()):
            raise ValueError(f"path escapes run directory: {relative!r}")
        return dest

    def _require_open(self) -> None:
        if not self._entered:
            raise RuntimeError("RunWriter must be entered before writing")
        if self._finalized:
            raise RuntimeError(f"run {self.run_id} is already finalized")


def _seed_from_spec(spec: ModelSpec) -> dict[str, int]:
    raw = spec.params.get("seed", 0)
    try:
        return {"base": int(raw)}
    except (TypeError, ValueError):
        return {"base": 0}


def write_predict_run(
    data_dir: Path,
    *,
    spec: ModelSpec,
    as_of: GWKey,
    deadline_utc: datetime,
    horizon: int,
    frame: pl.DataFrame,
    kind: RunKind = "adhoc",
    purpose: str = "",
) -> RunManifest:
    """Dual-write a live predict table into a new run folder.

    Legacy ``predictions/<model_id>/…`` paths stay the caller's responsibility.
    Manifest uses ``resolved_spec_sha256`` as ``spec_hash`` until B8.

    Args:
        data_dir: Local data root.
        spec: Model that produced ``frame``.
        as_of: Deadline predicted from.
        deadline_utc: That deadline's UTC timestamp.
        horizon: Horizon length passed to predict.
        frame: ``PredictionSchema`` rows.
        kind: Run kind (default adhoc for local CLI).
        purpose: Free-text purpose.

    Returns:
        The succeeded ``RunManifest``.
    """
    deadline = AsOfDeadline(
        season=as_of.season, gw=as_of.gw, deadline_utc=_aware(deadline_utc)
    )
    with RunWriter(
        data_dir,
        kind=kind,
        mode="live",
        run_type="predict",
        spec_hash=resolved_spec_sha256(spec),
        spec_label=spec.name,
        model_id=spec.model_id,
        as_of=[deadline],
        horizon=horizon,
        purpose=purpose or f"predict {spec.name} {as_of.season} gw{as_of.gw}",
        seeds=_seed_from_spec(spec),
    ) as writer:
        writer.write_parquet("predictions/player_fixture.parquet", frame)
    return writer.manifest


def write_plan_run(
    data_dir: Path,
    *,
    spec: ModelSpec,
    as_of: GWKey,
    deadline_utc: datetime,
    horizon: int,
    frame: pl.DataFrame,
    kind: RunKind = "adhoc",
    purpose: str = "",
    parent_run_id: str | None = None,
) -> RunManifest:
    """Dual-write a plan table into a new run folder.

    Args:
        data_dir: Local data root.
        spec: Model whose predictions fed the plan.
        as_of: Deadline planned from.
        deadline_utc: That deadline's UTC timestamp.
        horizon: Plan horizon (also ``plan_horizon`` on the manifest).
        frame: ``PlanSchema`` rows.
        kind: Run kind (default adhoc).
        purpose: Free-text purpose.
        parent_run_id: Optional predict run this plan reads.

    Returns:
        The succeeded ``RunManifest``.
    """
    deadline = AsOfDeadline(
        season=as_of.season, gw=as_of.gw, deadline_utc=_aware(deadline_utc)
    )
    with RunWriter(
        data_dir,
        kind=kind,
        mode="live",
        run_type="plan",
        spec_hash=resolved_spec_sha256(spec),
        spec_label=spec.name,
        model_id=spec.model_id,
        as_of=[deadline],
        horizon=horizon,
        plan_horizon=horizon,
        parent_run_id=parent_run_id,
        purpose=purpose or f"plan {spec.name} {as_of.season} gw{as_of.gw} h{horizon}",
        seeds=_seed_from_spec(spec),
    ) as writer:
        writer.write_parquet("plans/plan.parquet", frame)
    return writer.manifest


def write_backtest_run(
    data_dir: Path,
    *,
    spec: ModelSpec,
    as_ofs: list[tuple[GWKey, datetime]],
    horizon: int,
    frame: pl.DataFrame,
    kind: RunKind = "adhoc",
    purpose: str = "",
) -> RunManifest:
    """Dual-write a backtest prediction window into a new run folder.

    Args:
        data_dir: Local data root.
        spec: Model that produced ``frame``.
        as_ofs: ``(GWKey, deadline_utc)`` pairs covered by the window.
        horizon: Walk-forward steps.
        frame: Ledger-shaped prediction rows from ``walk_forward``.
        kind: Run kind (default adhoc).
        purpose: Free-text purpose.

    Returns:
        The succeeded ``RunManifest``.

    Raises:
        ValueError: ``as_ofs`` is empty.
    """
    if not as_ofs:
        raise ValueError("backtest runs require at least one as_of deadline")
    deadlines = [
        AsOfDeadline(season=key.season, gw=key.gw, deadline_utc=_aware(deadline))
        for key, deadline in as_ofs
    ]
    first, last = as_ofs[0][0], as_ofs[-1][0]
    with RunWriter(
        data_dir,
        kind=kind,
        mode="backtest",
        run_type="predict",
        spec_hash=resolved_spec_sha256(spec),
        spec_label=spec.name,
        model_id=spec.model_id,
        as_of=deadlines,
        horizon=horizon,
        purpose=purpose
        or (f"backtest {spec.name} {first.season} gw{first.gw}-{last.gw}"),
        seeds=_seed_from_spec(spec),
    ) as writer:
        writer.write_parquet("predictions/player_fixture.parquet", frame)
    return writer.manifest
