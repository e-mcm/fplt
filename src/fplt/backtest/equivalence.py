"""Equivalence levers for promote-eval (``kind: environment`` / ``refactor``).

Control is the shipped lockfile or git ref; treatment is the candidate. Arms
share one ModelSpec. Environment differs only in ``uv.lock``; refactor differs
only in the git ref. Each arm produces in a worktree under ``uv sync --frozen``
via subprocess. Verdict follows feature-eval-pipeline §3.2: identical →
``keep``; different-not-worse → ``inconclusive``; worse (interval entirely bad)
→ ``drop``.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import shutil
import subprocess
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal, cast

import polars as pl
from polars.testing import assert_frame_equal
from pydantic import BaseModel, ConfigDict, Field

from fplt.backtest.intervals import (
    equivalence_drop_from_intervals,
    intervals_for_pair,
)
from fplt.backtest.produce_ledger import produce_ledger, uv_lock_sha256
from fplt.models.spec import ModelSpec, load_model_specs

logger = logging.getLogger(__name__)

EquivalenceKind = Literal["environment", "refactor"]
Verdict = Literal["keep", "tune", "drop", "inconclusive"]

_KEY_COLS = (
    "as_of_season",
    "as_of_gw",
    "target_season",
    "target_gw",
    "player_key",
    "fixture_id",
)
_XPTS_COLS = ("xpts", "xpts_sd")
_PROB_COLS = ("p_haul", "p_60")
_COMPARE_COLS = (*_KEY_COLS, *_XPTS_COLS, *_PROB_COLS)
_PACKAGE_BLOCK = re.compile(
    r"\[\[package\]\]\s*\nname = \"([^\"]+)\"\s*\nversion = \"([^\"]+)\"",
    re.MULTILINE,
)


class EquivalenceTolerance(BaseModel):
    """Absolute tolerances for prediction identity (§3.2)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    xpts_abs: float = Field(default=1.0e-6, ge=0.0)
    prob_abs: float = Field(default=1.0e-9, ge=0.0)


class EquivalenceSpec(BaseModel):
    """Equivalence block on an EvalSpec."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    tolerance: EquivalenceTolerance = Field(default_factory=EquivalenceTolerance)
    compare: Literal["predictions"] = "predictions"


@dataclass(frozen=True)
class LockfileDiff:
    """Direct and transitive package version changes between two lockfiles."""

    control_sha256: str
    treatment_sha256: str
    direct_changed: tuple[str, ...]
    transitive_changed: tuple[str, ...]
    missing_packages: tuple[str, ...]
    identical: bool


@dataclass
class EquivalenceDiff:
    """Row-level prediction comparison for §3.2 step 1–2."""

    identical: bool
    n_rows: int
    n_beyond_tolerance: int
    max_abs_diff: float | None
    top_offenders: list[dict[str, Any]] = field(default_factory=list[dict[str, Any]])


@dataclass
class ArmProduceResult:
    """One arm's produced prediction frame plus provenance."""

    role: str
    ref: str
    git_sha: str
    lock_sha256: str
    frame: pl.DataFrame
    cache_hit: bool
    worktree: Path
    wall_clock_s: float


@dataclass
class EquivalenceRunResult:
    """Artifacts from producing and comparing both equivalence arms."""

    control: ArmProduceResult
    treatment: ArmProduceResult
    diff: EquivalenceDiff
    interval_drop: bool
    intervals: list[dict[str, Any]]
    verdict: Verdict
    lock_diff: LockfileDiff | None = None


def is_equivalence_kind(kind: str) -> bool:
    """Return whether ``kind`` is an equivalence lever."""
    return kind in {"environment", "refactor"}


def lockfile_package_versions(lock_path: Path) -> dict[str, str]:
    """Parse ``name → version`` from a ``uv.lock`` file."""
    if not lock_path.is_file():
        raise FileNotFoundError(lock_path)
    text = lock_path.read_text(encoding="utf-8")
    return {name: version for name, version in _PACKAGE_BLOCK.findall(text)}


def direct_dependency_names(pyproject_path: Path) -> set[str]:
    """Return normalised direct dependency names from ``pyproject.toml``."""
    raw = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))
    project_obj: object = raw.get("project")
    if not isinstance(project_obj, dict):
        return set()
    project = cast(dict[str, Any], project_obj)
    deps_obj: object = project.get("dependencies")
    if not isinstance(deps_obj, list):
        return set()
    names: set[str] = set()
    for entry_obj in cast(list[object], deps_obj):
        if not isinstance(entry_obj, str):
            continue
        name = re.split(r"[<>=!~;\[]", entry_obj, maxsplit=1)[0].strip().lower()
        if name:
            names.add(name)
    return names


def diff_lockfiles(
    control_lock: Path,
    treatment_lock: Path,
    *,
    pyproject: Path,
    declared_packages: list[str],
) -> LockfileDiff:
    """Diff two lockfiles; flag undeclared direct dependency changes.

    Args:
        control_lock: Control ``uv.lock``.
        treatment_lock: Treatment ``uv.lock``.
        pyproject: Project ``pyproject.toml`` (direct dependency list).
        declared_packages: ``change.packages`` from the EvalSpec.

    Returns:
        ``LockfileDiff`` including ``missing_packages`` (direct changes absent
        from ``declared_packages``).
    """
    control_sha = (
        hashlib.sha256(control_lock.read_bytes()).hexdigest()
        if control_lock.is_file()
        else "missing"
    )
    treatment_sha = (
        hashlib.sha256(treatment_lock.read_bytes()).hexdigest()
        if treatment_lock.is_file()
        else "missing"
    )
    left = lockfile_package_versions(control_lock) if control_lock.is_file() else {}
    right = (
        lockfile_package_versions(treatment_lock) if treatment_lock.is_file() else {}
    )
    direct = {name.lower() for name in direct_dependency_names(pyproject)}
    declared = {name.lower() for name in declared_packages}
    all_names = sorted(set(left) | set(right))
    direct_changed: list[str] = []
    transitive_changed: list[str] = []
    for name in all_names:
        if left.get(name) == right.get(name):
            continue
        label = f"{name}: {left.get(name, 'absent')} → {right.get(name, 'absent')}"
        if name.lower() in direct:
            direct_changed.append(label)
        else:
            transitive_changed.append(label)
    missing = tuple(
        sorted(
            {
                entry.split(":", 1)[0].lower()
                for entry in direct_changed
                if entry.split(":", 1)[0].lower() not in declared
            }
        )
    )
    return LockfileDiff(
        control_sha256=control_sha,
        treatment_sha256=treatment_sha,
        direct_changed=tuple(direct_changed),
        transitive_changed=tuple(transitive_changed),
        missing_packages=missing,
        identical=control_sha == treatment_sha,
    )


def resolve_git_sha(ref: str, *, cwd: Path | None = None) -> str:
    """Resolve a git ref to a full SHA."""
    completed = subprocess.run(
        ["git", "rev-parse", "--verify", ref],
        check=True,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    return completed.stdout.strip()


def add_worktree(repo: Path, ref: str, path: Path) -> Path:
    """Create a detached worktree at ``ref`` under ``path``.

    Args:
        repo: Repository root.
        ref: Git ref (branch, tag, or SHA).
        path: Destination directory (must not exist).

    Returns:
        The worktree path.
    """
    if path.exists():
        raise FileExistsError(f"worktree path already exists: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["git", "worktree", "add", "--detach", str(path), ref],
        check=True,
        capture_output=True,
        text=True,
        cwd=repo,
    )
    return path


def remove_worktree(repo: Path, path: Path) -> None:
    """Remove a worktree created by ``add_worktree``."""
    if not path.exists():
        return
    try:
        subprocess.run(
            ["git", "worktree", "remove", "--force", str(path)],
            check=True,
            capture_output=True,
            text=True,
            cwd=repo,
        )
    except subprocess.CalledProcessError:
        shutil.rmtree(path, ignore_errors=True)
        subprocess.run(
            ["git", "worktree", "prune"],
            check=False,
            capture_output=True,
            text=True,
            cwd=repo,
        )


def sync_frozen(worktree: Path) -> None:
    """Run ``uv sync --frozen`` inside ``worktree``."""
    logger.info("uv sync --frozen in %s", worktree)
    subprocess.run(
        ["uv", "sync", "--frozen"],
        check=True,
        capture_output=True,
        text=True,
        cwd=worktree,
    )


def produce_ledger_in_worktree(
    worktree: Path,
    *,
    data_dir: Path,
    model_name: str,
    season: str,
    mode: str,
    test_from_gw: int,
    test_to_gw: int,
    fit_gw: int | None,
    steps: int,
    seasons_back: int | None,
    out_parquet: Path,
    use_cache: bool,
    models_dir: Path | None = None,
) -> dict[str, Any]:
    """Produce one arm inside ``worktree`` via ``uv run --frozen`` subprocess.

    The worktree interpreter loads this host's ``produce_ledger`` helper script
    so older refs still share the cache protocol, while dependencies come from
    the worktree lockfile.

    Args:
        worktree: Synced git worktree root.
        data_dir: Shared staged data / ledger cache root.
        model_name: ModelSpec name.
        season: Evaluation season.
        mode: ``walk_forward`` or ``fixed``.
        test_from_gw: First scored gameweek.
        test_to_gw: Last scored gameweek.
        fit_gw: Fixed-mode fit gameweek.
        steps: Horizon steps.
        seasons_back: Train window override.
        out_parquet: Where to write the prediction parquet.
        use_cache: Whether the content-addressed cache is enabled.
        models_dir: Optional models directory override (host path).

    Returns:
        JSON-serialisable metadata (rows, cache_hit, lock sha, digest).

    Raises:
        RuntimeError: Subprocess failed.
    """
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "data_dir": str(data_dir),
        "model_name": model_name,
        "season": season,
        "mode": mode,
        "test_from_gw": test_from_gw,
        "test_to_gw": test_to_gw,
        "fit_gw": fit_gw,
        "steps": steps,
        "seasons_back": seasons_back,
        "out_parquet": str(out_parquet),
        "use_cache": use_cache,
        "models_dir": str(models_dir) if models_dir is not None else None,
        "lock_root": str(worktree),
    }
    request_path = out_parquet.with_suffix(".request.json")
    request_path.write_text(json.dumps(payload) + "\n", encoding="utf-8")
    helper = Path(__file__).resolve().parent / "_produce_arm_main.py"
    cmd = [
        "uv",
        "run",
        "--directory",
        str(worktree),
        "--frozen",
        "python",
        str(helper),
        str(request_path),
    ]
    logger.info("producing arm in worktree %s model=%s", worktree, model_name)
    completed = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if completed.returncode != 0:
        raise RuntimeError(
            "produce-ledger subprocess failed "
            f"(exit {completed.returncode}): {completed.stderr or completed.stdout}"
        )
    meta_path = out_parquet.with_suffix(".meta.json")
    if meta_path.is_file():
        return json.loads(meta_path.read_text(encoding="utf-8"))
    # Fallback: parse last JSON line from stdout.
    lines = [line for line in completed.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("produce-ledger subprocess returned no metadata")
    return json.loads(lines[-1])


def produce_arm_local(
    *,
    data_dir: Path,
    spec: ModelSpec,
    season: str,
    mode: Literal["walk_forward", "fixed"],
    test_from_gw: int,
    test_to_gw: int,
    fit_gw: int | None,
    steps: int,
    seasons_back: int | None,
    out_parquet: Path,
    use_cache: bool,
    lock_root: Path | None,
) -> dict[str, Any]:
    """Produce one arm in-process (tests / helper entrypoint).

    Args:
        data_dir: Staged data root.
        spec: Resolved ModelSpec.
        season: Evaluation season.
        mode: Compare mode.
        test_from_gw: First scored gameweek.
        test_to_gw: Last scored gameweek.
        fit_gw: Fixed fit gameweek.
        steps: Horizon.
        seasons_back: Train-window override.
        out_parquet: Output parquet path.
        use_cache: Content-addressed cache flag.
        lock_root: Optional root for ``uv.lock`` hashing.

    Returns:
        Metadata dict written beside the parquet.
    """
    produced = produce_ledger(
        data_dir,
        spec,
        season,
        mode=mode,
        test_from_gw=test_from_gw,
        test_to_gw=test_to_gw,
        fit_gw=fit_gw,
        steps=steps,
        seasons_back=seasons_back,
        use_cache=use_cache,
        lock_root=lock_root,
    )
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    produced.frame.write_parquet(out_parquet)
    meta = {
        "rows": produced.frame.height,
        "cache_hit": produced.cache_hit,
        "digest": produced.key.digest(),
        "uv_lock_sha256": produced.key.uv_lock_sha256,
        "resolved_spec_sha256": produced.key.resolved_spec_sha256,
    }
    out_parquet.with_suffix(".meta.json").write_text(
        json.dumps(meta, indent=2) + "\n", encoding="utf-8"
    )
    return meta


def compare_prediction_frames(
    control: pl.DataFrame,
    treatment: pl.DataFrame,
    tolerance: EquivalenceTolerance,
    *,
    top_n: int = 10,
) -> EquivalenceDiff:
    """Compare prediction ledgers per §3.2 step 1.

    Sorts by key columns and uses ``assert_frame_equal`` (``rtol=0``). When that
    fails, rows within ``tolerance`` still count as identical; offenders beyond
    tolerance are listed.

    Args:
        control: Control prediction rows.
        treatment: Treatment prediction rows.
        tolerance: Absolute tolerances for xPts and probability columns.
        top_n: Max offenders to return.

    Returns:
        ``EquivalenceDiff`` summarising identity.
    """
    left = _prep_predictions(control)
    right = _prep_predictions(treatment)
    if left.height != right.height:
        return EquivalenceDiff(
            identical=False,
            n_rows=min(left.height, right.height),
            n_beyond_tolerance=abs(left.height - right.height),
            max_abs_diff=None,
            top_offenders=[
                {
                    "reason": "row_count_mismatch",
                    "control_rows": left.height,
                    "treatment_rows": right.height,
                }
            ],
        )
    if left.height == 0:
        return EquivalenceDiff(
            identical=True,
            n_rows=0,
            n_beyond_tolerance=0,
            max_abs_diff=None,
            top_offenders=[],
        )
    try:
        assert_frame_equal(left, right, rel_tol=0.0, abs_tol=0.0, check_dtypes=False)
        return EquivalenceDiff(
            identical=True,
            n_rows=left.height,
            n_beyond_tolerance=0,
            max_abs_diff=0.0,
            top_offenders=[],
        )
    except AssertionError:
        pass

    joined = left.join(right, on=list(_KEY_COLS), how="inner", suffix="_t")
    beyond = 0
    max_diff = 0.0
    offenders: list[dict[str, Any]] = []
    for col in (*_XPTS_COLS, *_PROB_COLS):
        tol = tolerance.xpts_abs if col in _XPTS_COLS else tolerance.prob_abs
        right_col = f"{col}_t"
        if col not in joined.columns or right_col not in joined.columns:
            continue
        delta = (joined[col] - joined[right_col]).abs()
        joined = joined.with_columns(delta.alias(f"abs_{col}"))
        col_max = float(delta.max()) if delta.len() else 0.0  # type: ignore[arg-type]
        max_diff = max(max_diff, col_max)
        mask = delta > tol
        n_bad = int(mask.sum())  # type: ignore[arg-type]
        beyond = max(beyond, n_bad)
        if n_bad:
            bad = (
                joined.filter(mask)
                .select(
                    *list(_KEY_COLS),
                    pl.col(col).alias("control"),
                    pl.col(right_col).alias("treatment"),
                    pl.col(f"abs_{col}").alias("abs_diff"),
                    pl.lit(col).alias("column"),
                )
                .sort("abs_diff", descending=True)
                .head(top_n)
            )
            offenders.extend(bad.to_dicts())

    offenders_sorted = sorted(
        offenders, key=lambda row: float(row.get("abs_diff") or 0.0), reverse=True
    )[:top_n]
    identical = beyond == 0
    return EquivalenceDiff(
        identical=identical,
        n_rows=joined.height,
        n_beyond_tolerance=beyond,
        max_abs_diff=max_diff if joined.height else None,
        top_offenders=offenders_sorted,
    )


def equivalence_verdict(
    diff: EquivalenceDiff,
    *,
    interval_drop: bool,
    force_inconclusive: bool = False,
) -> Verdict:
    """Apply the §3.2 identical / not-worse / worse rule.

    Args:
        diff: Prediction identity report.
        interval_drop: True when any primary-metric interval is entirely bad.
        force_inconclusive: Dirty tree / allow_low_coverage override.

    Returns:
        ``keep``, ``inconclusive``, or ``drop``.
    """
    if force_inconclusive:
        return "inconclusive"
    if diff.identical:
        return "keep"
    if interval_drop:
        return "drop"
    return "inconclusive"


def interval_drop_for_frames(
    control: pl.DataFrame,
    treatment: pl.DataFrame,
    *,
    control_id: str,
    treatment_id: str,
    actuals: pl.DataFrame | None,
) -> tuple[bool, list[dict[str, Any]]]:
    """Compute whether any primary interval is entirely on the bad side.

    Args:
        control: Control predictions (with ``model_id``).
        treatment: Treatment predictions.
        control_id: Control ``model_id`` label used after tagging.
        treatment_id: Treatment ``model_id`` label.
        actuals: Optional frame with actuals join keys and ``actual_pts``.

    Returns:
        ``(drop, interval_dicts)``. Missing actuals → no drop.
    """
    if actuals is None or actuals.is_empty():
        return False, []
    tagged_control = control.with_columns(pl.lit(control_id).alias("model_id"))
    tagged_treatment = treatment.with_columns(pl.lit(treatment_id).alias("model_id"))
    scored = _attach_actuals(
        pl.concat([tagged_control, tagged_treatment], how="vertical_relaxed"),
        actuals,
    )
    if scored.is_empty() or scored["actual_pts"].null_count() == scored.height:
        return False, []
    paired = intervals_for_pair(
        scored,
        control_id=control_id,
        treatment_id=treatment_id,
        metrics=("mae", "rmse", "spearman"),
    )
    from fplt.backtest.intervals import interval_to_dict

    return (
        equivalence_drop_from_intervals(paired),
        [interval_to_dict(item) for item in paired],
    )


def load_arm_spec(
    model_name: str,
    *,
    models_dir: Path | None,
    worktree: Path | None = None,
) -> ModelSpec:
    """Load a ModelSpec from the host or a worktree configs tree."""
    directory = models_dir
    if directory is None and worktree is not None:
        candidate = worktree / "configs" / "models"
        if candidate.is_dir():
            directory = candidate
    specs = load_model_specs(directory)
    if model_name not in specs:
        raise KeyError(f"ModelSpec {model_name!r} not found")
    return specs[model_name]


def _prep_predictions(frame: pl.DataFrame) -> pl.DataFrame:
    missing = [col for col in _COMPARE_COLS if col not in frame.columns]
    if missing:
        raise ValueError(f"prediction frame missing columns: {missing}")
    return frame.select(list(_COMPARE_COLS)).sort(list(_KEY_COLS))


def _attach_actuals(preds: pl.DataFrame, actuals: pl.DataFrame) -> pl.DataFrame:
    keys = ["target_season", "target_gw", "player_key", "fixture_id"]
    keep = [col for col in (*keys, "actual_pts", "position") if col in actuals.columns]
    slim = actuals.select(keep).unique(subset=keys, keep="last")
    joined = preds.join(slim, on=keys, how="left")
    if "run_kind" not in joined.columns:
        joined = joined.with_columns(pl.lit("equivalence").alias("run_kind"))
    if "horizon" not in joined.columns:
        joined = joined.with_columns(pl.lit(1).alias("horizon"))
    if "position" not in joined.columns:
        joined = joined.with_columns(pl.lit("MID").alias("position"))
    if "actual_minutes" not in joined.columns:
        joined = joined.with_columns(
            pl.lit(None).cast(pl.Int64).alias("actual_minutes")
        )
    return joined


def lock_sha_at(root: Path) -> str:
    """SHA-256 of ``uv.lock`` under ``root``."""
    return uv_lock_sha256(root=root)


__all__ = [
    "ArmProduceResult",
    "EquivalenceDiff",
    "EquivalenceRunResult",
    "EquivalenceSpec",
    "EquivalenceTolerance",
    "LockfileDiff",
    "add_worktree",
    "compare_prediction_frames",
    "diff_lockfiles",
    "direct_dependency_names",
    "equivalence_verdict",
    "interval_drop_for_frames",
    "is_equivalence_kind",
    "load_arm_spec",
    "lock_sha_at",
    "lockfile_package_versions",
    "produce_arm_local",
    "produce_ledger_in_worktree",
    "remove_worktree",
    "resolve_git_sha",
    "sync_frozen",
]
