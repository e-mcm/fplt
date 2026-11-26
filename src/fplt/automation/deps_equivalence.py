"""Runtime dependency PRs → environment promote-eval (ops-12).

Renovate labels runtime lockfile PRs ``deps:runtime`` / ``equivalence``.
The label-triggered workflow builds a ``kind: environment`` EvalSpec from the
base vs head ``uv.lock`` diff on the fixed board
(``configs/evals/runtime_deps_equivalence.yaml``), runs promote-eval, and
posts the verdict. ``fplt-mc`` is never an auto-bump target.

Prod freeze (ops-04): when ``fplt.ops.freeze.in_prod_freeze`` exists it is
honoured; until then :func:`check_prod_freeze` soft-skips.
"""

from __future__ import annotations

import importlib
import logging
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

import yaml

from fplt.backtest.equivalence import (
    diff_lockfiles,
    lockfile_package_versions,
)
from fplt.backtest.promote_eval import (
    EvalSpec,
    PromoteEvalResult,
    execute_promote_eval,
    load_eval_spec,
)

logger = logging.getLogger(__name__)

RUNTIME_DEPS_EVAL = "runtime_deps_equivalence"
FPLT_MC = "fplt-mc"
EQUIVALENCE_LABELS = frozenset({"deps:runtime", "equivalence"})

FreezeState = Literal["frozen", "open", "soft_skip"]
GateVerdict = Literal["pass", "fail", "skip"]


@dataclass(frozen=True)
class FreezeCheck:
    """Result of the ops-04 freeze-window hook.

    Attributes:
        state: ``frozen`` blocks merge; ``open`` allows it; ``soft_skip`` means
            ops-04 is not installed yet.
        reason: Human-readable explanation for the PR comment / logs.
    """

    state: FreezeState
    reason: str


@dataclass(frozen=True)
class LockDiffPackages:
    """Direct runtime packages that differ between two lockfiles.

    Attributes:
        packages: Normalised direct dependency names to declare on the EvalSpec.
        direct_changed: Human-readable ``name: old → new`` lines.
        transitive_changed: Transitive-only version changes (listed, not declared).
        fplt_mc_changed: True when ``fplt-mc`` versions differ (must block).
    """

    packages: tuple[str, ...]
    direct_changed: tuple[str, ...]
    transitive_changed: tuple[str, ...]
    fplt_mc_changed: bool


@dataclass(frozen=True)
class DepsEquivalencePrep:
    """Prepared EvalSpec path plus lock contrast for a runtime deps PR.

    Attributes:
        eval_name: Stem written under ``evals_dir``.
        evals_dir: Directory containing the rewritten EvalSpec YAML.
        packages: Declared ``change.packages``.
        control_ref: Git ref for the control lockfile arm.
        treatment_ref: Git ref for the treatment lockfile arm.
        lock_diff: Package contrast used to fill the EvalSpec.
    """

    eval_name: str
    evals_dir: Path
    packages: tuple[str, ...]
    control_ref: str
    treatment_ref: str
    lock_diff: LockDiffPackages


@dataclass(frozen=True)
class DepsEquivalenceGate:
    """Merge-gate outcome for a runtime dependency PR.

    Attributes:
        gate: ``pass`` / ``fail`` / ``skip``.
        verdict: Promote-eval verdict when execute ran, else ``None``.
        summary: Short text for the PR comment body.
        recommendation: Full ``recommendation.md`` when available.
        freeze: Freeze-window check result.
        packages: Declared direct packages when prep succeeded.
        control_ref: Control ref when prep succeeded.
        treatment_ref: Treatment ref when prep succeeded.
    """

    gate: GateVerdict
    verdict: str | None
    summary: str
    recommendation: str | None
    freeze: FreezeCheck
    packages: tuple[str, ...] = ()
    control_ref: str | None = None
    treatment_ref: str | None = None


def check_prod_freeze() -> FreezeCheck:
    """Respect the ops-04 freeze window when present; otherwise soft-skip.

    Looks for ``fplt.ops.freeze.in_prod_freeze``. When that helper is missing,
    returns ``soft_skip`` so Renovate / equivalence can still land before
    ops-04. When present and true, returns ``frozen`` (merge must wait).

    Returns:
        A :class:`FreezeCheck` describing the freeze state.
    """
    try:
        module = importlib.import_module("fplt.ops.freeze")
    except ModuleNotFoundError:
        return FreezeCheck(
            state="soft_skip",
            reason=(
                "ops-04 freeze helper absent (fplt.ops.freeze); "
                "soft-skip until prod freeze lands"
            ),
        )
    hook = getattr(module, "in_prod_freeze", None)
    if hook is None or not callable(hook):
        return FreezeCheck(
            state="soft_skip",
            reason=(
                "fplt.ops.freeze imported but in_prod_freeze is missing; soft-skip"
            ),
        )
    raw: object = hook()
    frozen: bool
    detail: str
    if isinstance(raw, bool):
        frozen = raw
        detail = "in_prod_freeze() returned True" if frozen else "outside freeze"
    elif isinstance(raw, tuple):
        items = cast(tuple[object, ...], raw)
        head: object | None = items[0] if len(items) >= 1 else None
        tail: object | None = items[1] if len(items) >= 2 else None
        if head is None:
            frozen = False
            detail = "in_prod_freeze() returned empty tuple"
        else:
            frozen = bool(head)
            detail = str(tail) if tail is not None else str(head)
    else:
        frozen = bool(raw)
        detail = repr(raw)
    if frozen:
        return FreezeCheck(state="frozen", reason=f"prod freeze active: {detail}")
    return FreezeCheck(state="open", reason=f"outside prod freeze: {detail}")


def changed_direct_packages(
    control_lock: Path,
    treatment_lock: Path,
    *,
    pyproject: Path,
) -> LockDiffPackages:
    """List direct dependency names that differ between two ``uv.lock`` files.

    Args:
        control_lock: Lockfile on the control ref (usually ``main``).
        treatment_lock: Lockfile on the PR head.
        pyproject: ``pyproject.toml`` used to classify direct vs transitive.

    Returns:
        :class:`LockDiffPackages` including an ``fplt_mc_changed`` flag.

    Raises:
        FileNotFoundError: A required path is missing.
    """
    if not control_lock.is_file():
        raise FileNotFoundError(control_lock)
    if not treatment_lock.is_file():
        raise FileNotFoundError(treatment_lock)
    if not pyproject.is_file():
        raise FileNotFoundError(pyproject)

    left = lockfile_package_versions(control_lock)
    right = lockfile_package_versions(treatment_lock)
    probe = diff_lockfiles(
        control_lock,
        treatment_lock,
        pyproject=pyproject,
        declared_packages=[],
    )
    packages = tuple(
        sorted({entry.split(":", 1)[0].lower() for entry in probe.direct_changed})
    )
    fplt_mc_changed = left.get(FPLT_MC) != right.get(FPLT_MC)
    return LockDiffPackages(
        packages=packages,
        direct_changed=probe.direct_changed,
        transitive_changed=probe.transitive_changed,
        fplt_mc_changed=fplt_mc_changed,
    )


def write_runtime_deps_eval(
    *,
    board: EvalSpec,
    packages: list[str],
    control_ref: str,
    treatment_ref: str,
    evals_dir: Path,
    eval_name: str = RUNTIME_DEPS_EVAL,
    diff_notes: list[str] | None = None,
) -> Path:
    """Write a ``kind: environment`` EvalSpec for a runtime lockfile PR.

    Args:
        board: Fixed-board defaults loaded from
            ``configs/evals/runtime_deps_equivalence.yaml``.
        packages: Direct dependency names that changed (``change.packages``).
        control_ref: Control git ref (base / ``main``).
        treatment_ref: Treatment git ref (PR head SHA).
        evals_dir: Directory to write ``<eval_name>.yaml`` into.
        eval_name: EvalSpec stem (must match ``name`` in the YAML).
        diff_notes: Optional ``change.diff`` lines; defaults to package list.

    Returns:
        Path of the written YAML.

    Raises:
        ValueError: ``packages`` is empty or contains ``fplt-mc``.
    """
    if not packages:
        raise ValueError("runtime deps equivalence requires at least one package")
    normalised = [name.lower() for name in packages]
    if FPLT_MC in normalised:
        raise ValueError(
            "fplt-mc must not appear on a Renovate runtime PR; "
            "bump it only via mc-release"
        )
    notes = diff_notes or [f"uv.lock: {name}" for name in normalised]
    payload = board.model_dump(mode="python")
    payload["name"] = eval_name
    payload["change"] = {
        **payload.get("change", {}),
        "lever": "dependency_upgrade",
        "packages": normalised,
        "control_ref": control_ref,
        "treatment_ref": treatment_ref,
        "diff": notes,
        "tables": [],
        "features": [],
        "allow_multi_lever": False,
    }
    # Same ModelSpec both arms — environment lever.
    payload["control"] = board.control
    payload["treatment"] = board.treatment
    evals_dir.mkdir(parents=True, exist_ok=True)
    path = evals_dir / f"{eval_name}.yaml"
    path.write_text(
        yaml.safe_dump(payload, sort_keys=False),
        encoding="utf-8",
    )
    return path


def prepare_runtime_deps_eval(
    *,
    repo_root: Path,
    control_ref: str,
    treatment_ref: str,
    control_lock: Path,
    treatment_lock: Path,
    board_name: str = RUNTIME_DEPS_EVAL,
    evals_dir: Path | None = None,
) -> DepsEquivalencePrep:
    """Build the rewritten EvalSpec for a runtime dependency PR.

    Args:
        repo_root: Repository root (holds ``pyproject.toml``).
        control_ref: Control git ref string stored on the EvalSpec.
        treatment_ref: Treatment git ref string stored on the EvalSpec.
        control_lock: Path to the control ``uv.lock`` contents.
        treatment_lock: Path to the treatment ``uv.lock`` contents.
        board_name: Fixed-board EvalSpec stem under ``configs/evals/``.
        evals_dir: Where to write the rewritten YAML. Defaults to a new
            temporary directory.

    Returns:
        :class:`DepsEquivalencePrep` ready for ``execute_promote_eval``.

    Raises:
        ValueError: Lockfiles identical, no direct changes, or ``fplt-mc`` moved.
        FileNotFoundError: Board or lock paths missing.
    """
    pyproject = repo_root / "pyproject.toml"
    lock_diff = changed_direct_packages(
        control_lock,
        treatment_lock,
        pyproject=pyproject,
    )
    if lock_diff.fplt_mc_changed:
        raise ValueError(
            "fplt-mc changed between refs; refuse Renovate equivalence "
            "(bump only via mc-release)"
        )
    if not lock_diff.packages and not lock_diff.transitive_changed:
        raise ValueError("uv.lock is identical between control and treatment")
    if not lock_diff.packages:
        raise ValueError(
            "only transitive lock changes detected; declare a direct "
            "dependency bump or skip equivalence"
        )
    board = load_eval_spec(board_name)
    if evals_dir is not None:
        out_dir = evals_dir
    else:
        out_dir = Path(tempfile.mkdtemp(prefix="deps-eq-"))
    write_runtime_deps_eval(
        board=board,
        packages=list(lock_diff.packages),
        control_ref=control_ref,
        treatment_ref=treatment_ref,
        evals_dir=out_dir,
        eval_name=board_name,
        diff_notes=list(lock_diff.direct_changed),
    )
    return DepsEquivalencePrep(
        eval_name=board_name,
        evals_dir=out_dir,
        packages=lock_diff.packages,
        control_ref=control_ref,
        treatment_ref=treatment_ref,
        lock_diff=lock_diff,
    )


def gate_from_verdict(verdict: str) -> GateVerdict:
    """Map a promote-eval equivalence verdict onto the merge gate.

    ``keep`` and ``inconclusive`` pass (human reviews inconclusive).
    ``drop`` fails. Anything else is treated as fail-closed.

    Args:
        verdict: Promote-eval verdict string.

    Returns:
        ``pass`` or ``fail``.
    """
    if verdict in {"keep", "inconclusive"}:
        return "pass"
    return "fail"


def format_pr_comment(
    gate: DepsEquivalenceGate,
    *,
    packages: tuple[str, ...] | list[str],
    control_ref: str,
    treatment_ref: str,
) -> str:
    """Render the PR comment body for the equivalence workflow.

    Args:
        gate: Gate outcome including freeze and optional recommendation.
        packages: Declared package names.
        control_ref: Control ref.
        treatment_ref: Treatment ref.

    Returns:
        Markdown suitable for ``gh pr comment``.
    """
    lines = [
        "## Runtime deps equivalence (ops-12)",
        "",
        f"- **gate:** `{gate.gate}`",
        f"- **verdict:** `{gate.verdict or 'n/a'}`",
        f"- **packages:** {', '.join(packages) if packages else '(none)'}",
        f"- **control_ref:** `{control_ref}`",
        f"- **treatment_ref:** `{treatment_ref}`",
        f"- **freeze:** `{gate.freeze.state}` — {gate.freeze.reason}",
        "",
        gate.summary,
    ]
    if gate.recommendation:
        lines.extend(
            [
                "",
                "<details><summary>recommendation.md</summary>",
                "",
                "```markdown",
                gate.recommendation.rstrip(),
                "```",
                "",
                "</details>",
            ]
        )
    return "\n".join(lines) + "\n"


def run_deps_equivalence(
    *,
    repo_root: Path,
    control_ref: str,
    treatment_ref: str,
    data_dir: Path,
    output_dir: Path,
    control_lock: Path | None = None,
    treatment_lock: Path | None = None,
    live: bool = False,
    execute: bool = True,
) -> DepsEquivalenceGate:
    """Prepare and optionally execute the runtime-deps equivalence board.

    Args:
        repo_root: Repository root.
        control_ref: Control git ref (base SHA or ``main``).
        treatment_ref: Treatment git ref (head SHA).
        data_dir: ``FPLT_DATA_DIR`` for promote-eval.
        output_dir: Where promote-eval writes artifacts.
        control_lock: Override control lockfile path (tests).
        treatment_lock: Override treatment lockfile path (tests).
        live: When false, skip execute and soft-skip the gate (scaffolding).
        execute: When true and ``live``, run ``execute_promote_eval``.

    Returns:
        :class:`DepsEquivalenceGate` for the workflow exit code / comment.
    """
    freeze = check_prod_freeze()
    if freeze.state == "frozen":
        return DepsEquivalenceGate(
            gate="fail",
            verdict=None,
            summary=f"Blocked by prod freeze window. {freeze.reason}",
            recommendation=None,
            freeze=freeze,
            control_ref=control_ref,
            treatment_ref=treatment_ref,
        )

    ctrl_lock = control_lock or (repo_root / "uv.lock")
    treat_lock = treatment_lock or (repo_root / "uv.lock")
    evals_dir = output_dir / "evals"
    try:
        prep = prepare_runtime_deps_eval(
            repo_root=repo_root,
            control_ref=control_ref,
            treatment_ref=treatment_ref,
            control_lock=ctrl_lock,
            treatment_lock=treat_lock,
            evals_dir=evals_dir,
        )
    except ValueError as exc:
        return DepsEquivalenceGate(
            gate="fail",
            verdict=None,
            summary=f"Prep failed: {exc}",
            recommendation=None,
            freeze=freeze,
            control_ref=control_ref,
            treatment_ref=treatment_ref,
        )

    if not live or not execute:
        return DepsEquivalenceGate(
            gate="skip",
            verdict=None,
            summary=(
                "Prepared EvalSpec "
                f"({', '.join(prep.packages)}); execute skipped "
                "(set DEPS_EQUIVALENCE_LIVE=true to run promote-eval)."
            ),
            recommendation=None,
            freeze=freeze,
            packages=prep.packages,
            control_ref=prep.control_ref,
            treatment_ref=prep.treatment_ref,
        )

    spec = load_eval_spec(prep.eval_name, directory=prep.evals_dir)
    result: PromoteEvalResult = execute_promote_eval(
        spec,
        data_dir=data_dir,
        dry_run=False,
        output_dir=output_dir / "promote",
        argv=["deps-equivalence", "--execute"],
    )
    recommendation = result.recommendation
    gate = gate_from_verdict(result.verdict)
    summary = (
        f"Equivalence verdict `{result.verdict}` → gate `{gate}` "
        f"(exit {result.exit_code})."
    )
    return DepsEquivalenceGate(
        gate=gate,
        verdict=result.verdict,
        summary=summary,
        recommendation=recommendation,
        freeze=freeze,
        packages=prep.packages,
        control_ref=prep.control_ref,
        treatment_ref=prep.treatment_ref,
    )


def materialise_lock_from_git(
    repo_root: Path,
    ref: str,
    dest: Path,
    *,
    path: str = "uv.lock",
) -> Path:
    """Write ``path`` at ``ref`` to ``dest`` via ``git show``.

    Args:
        repo_root: Git repository root.
        ref: Git ref to read.
        dest: Destination file path.
        path: Repo-relative path to show.

    Returns:
        ``dest``.

    Raises:
        RuntimeError: ``git show`` failed.
    """
    import subprocess

    completed = subprocess.run(
        ["git", "show", f"{ref}:{path}"],
        check=False,
        capture_output=True,
        text=True,
        cwd=repo_root,
    )
    if completed.returncode != 0:
        raise RuntimeError(f"git show {ref}:{path} failed: {completed.stderr.strip()}")
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(completed.stdout, encoding="utf-8")
    return dest


def board_defaults_payload() -> dict[str, Any]:
    """Return the fixed-board EvalSpec as a plain dict (tests / docs)."""
    return load_eval_spec(RUNTIME_DEPS_EVAL).model_dump(mode="python")


__all__ = [
    "EQUIVALENCE_LABELS",
    "FPLT_MC",
    "RUNTIME_DEPS_EVAL",
    "DepsEquivalenceGate",
    "DepsEquivalencePrep",
    "FreezeCheck",
    "LockDiffPackages",
    "board_defaults_payload",
    "changed_direct_packages",
    "check_prod_freeze",
    "format_pr_comment",
    "gate_from_verdict",
    "materialise_lock_from_git",
    "prepare_runtime_deps_eval",
    "run_deps_equivalence",
    "write_runtime_deps_eval",
]
