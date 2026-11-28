"""CI governance for production model and ops config changes (ops-09).

Fails a PR that retargets the live slate in ``configs/ops/production.yaml``
(``models`` / ``plan_horizon`` / ``pipeline`` / ``lines``) or a ModelSpec
listed in it unless the PR body links a promote-eval result
(``promote-eval-run: <run_id>``) whose verdict is passing (``keep``).
Additive ops keys (e.g. ``min_prior_seasons``) do not require a promote-eval.
Production ModelSpecs are immutable: an in-place edit of a listed YAML
always fails — bump the name.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import yaml

from fplt.backtest.results_registry import (
    default_results_registry_path,
    load_results_registry,
)
from fplt.utils.process import run_command

PRODUCTION_RELPATH = "configs/ops/production.yaml"
MODELS_PREFIX = "configs/models/"
PROMOTE_EVAL_RUN_RE = re.compile(
    r"promote-eval-run:\s*([A-Za-z0-9._/-]+)",
    re.IGNORECASE,
)
PASSING_VERDICTS = frozenset({"keep"})
VerdictLookup = Callable[[str], str | None]


@dataclass(frozen=True)
class GovernanceResult:
    """Outcome of one production-governance check."""

    passed: bool
    reasons: tuple[str, ...] = ()
    promote_eval_runs: tuple[str, ...] = ()
    in_place_edits: tuple[str, ...] = ()
    gated_paths: tuple[str, ...] = ()

    @property
    def failures(self) -> tuple[str, ...]:
        """Human-readable failure reasons (empty when passed)."""
        return () if self.passed else self.reasons


@dataclass(frozen=True)
class _DiffSnapshot:
    """Base/head file texts relevant to production governance."""

    changed_paths: tuple[str, ...]
    base_production: str | None
    head_production: str | None
    base_specs: Mapping[str, str]
    head_specs: Mapping[str, str]


@dataclass
class _GateState:
    """Mutable collector while evaluating governance rules."""

    reasons: list[str] = field(default_factory=list[str])
    in_place: list[str] = field(default_factory=list[str])
    gated: list[str] = field(default_factory=list[str])


def model_spec_relpath(name: str) -> str:
    """Return the repo-relative ModelSpec path for a production model name."""
    return f"{MODELS_PREFIX}{name}.yaml"


def parse_production_models(yaml_text: str) -> list[str]:
    """Extract the ``models:`` list from a production ops YAML document.

    Args:
        yaml_text: Contents of ``configs/ops/production.yaml``.

    Returns:
        Non-empty model name list.

    Raises:
        ValueError: YAML is not a mapping or ``models`` is missing/invalid.
    """
    loaded = yaml.safe_load(yaml_text)
    if not isinstance(loaded, dict):
        raise ValueError("production ops YAML must contain a mapping")
    raw = cast(dict[str, Any], loaded)
    models_obj = raw.get("models")
    if not isinstance(models_obj, list):
        raise ValueError("production ops must set models: [name, ...]")
    names = [str(item).strip() for item in cast(list[object], models_obj)]
    if not names or any(not name for name in names):
        raise ValueError("models must be a non-empty list of non-empty names")
    return names


def extract_promote_eval_runs(pr_body: str) -> list[str]:
    """Return ``promote-eval-run:`` ids linked in a PR description."""
    return PROMOTE_EVAL_RUN_RE.findall(pr_body or "")


def resolve_promote_eval_verdict(
    run_id: str,
    *,
    data_dir: Path | None = None,
    registry_path: Path | None = None,
) -> str | None:
    """Look up a promote-eval verdict for ``run_id``.

    Search order: committed results registry entry id, then
    ``configs/evals/results/<run_id>/summary.json``, then under
    ``$data_dir/promote_eval/**/summary.json`` / ``runs/run=<run_id>/``.

    Args:
        run_id: Linked promote-eval / registry id from the PR body.
        data_dir: Optional ``$FPLT_DATA_DIR`` mirror for live artifacts.
        registry_path: Override path to ``registry.yaml``.

    Returns:
        Lowercased verdict string, or ``None`` when not found.
    """
    registry_verdict = _registry_verdict(run_id, registry_path=registry_path)
    if registry_verdict is not None:
        return registry_verdict

    committed = (
        Path(__file__).resolve().parents[3]
        / "configs"
        / "evals"
        / "results"
        / run_id
        / "summary.json"
    )
    from_file = _verdict_from_summary(committed)
    if from_file is not None:
        return from_file

    if data_dir is not None:
        run_summary = data_dir / "runs" / f"run={run_id}" / "summary.json"
        from_run = _verdict_from_summary(run_summary)
        if from_run is not None:
            return from_run
        promote_root = data_dir / "promote_eval"
        if promote_root.is_dir():
            for path in sorted(promote_root.rglob("summary.json")):
                if run_id in path.parts:
                    found = _verdict_from_summary(path)
                    if found is not None:
                        return found
    return None


def check_production_governance(
    *,
    changed_paths: Sequence[str],
    base_production_yaml: str | None,
    head_production_yaml: str | None,
    base_model_specs: Mapping[str, str],
    head_model_specs: Mapping[str, str],
    pr_body: str,
    verdict_for_run: VerdictLookup | None = None,
    data_dir: Path | None = None,
) -> GovernanceResult:
    """Evaluate production governance for one PR diff.

    Args:
        changed_paths: Repo-relative paths touched by the PR.
        base_production_yaml: Base-branch ``production.yaml`` text, or None.
        head_production_yaml: Head-branch ``production.yaml`` text, or None.
        base_model_specs: Base contents keyed by ModelSpec relpath.
        head_model_specs: Head contents keyed by ModelSpec relpath.
        pr_body: Pull-request description (may be empty).
        verdict_for_run: Optional override for promote-eval verdict lookup.
        data_dir: Optional data root for default verdict resolution.

    Returns:
        ``GovernanceResult`` with pass/fail and reasons.
    """
    changed = {path.replace("\\", "/") for path in changed_paths}
    state = _GateState()
    base_models = _models_or_empty(base_production_yaml)
    head_models = _models_or_empty(head_production_yaml)
    listed_at_base = set(base_models)
    listed_at_head = set(head_models)

    _collect_in_place_edits(
        state,
        changed=changed,
        listed_at_base=listed_at_base,
        base_model_specs=base_model_specs,
        head_model_specs=head_model_specs,
    )
    _collect_gated_paths(
        state,
        changed=changed,
        listed_at_base=listed_at_base,
        listed_at_head=listed_at_head,
        base_production_yaml=base_production_yaml,
        head_production_yaml=head_production_yaml,
    )

    if not state.gated and not state.in_place:
        return GovernanceResult(passed=True)

    if state.in_place:
        state.reasons.append(
            "in-place edit of production ModelSpec(s) "
            f"{', '.join(state.in_place)}; create a new spec name instead"
        )

    runs = extract_promote_eval_runs(pr_body)

    def _default_lookup(run_id: str) -> str | None:
        return resolve_promote_eval_verdict(run_id, data_dir=data_dir)

    lookup: VerdictLookup = (
        verdict_for_run if verdict_for_run is not None else _default_lookup
    )
    if state.gated and not state.in_place:
        _require_passing_promote_eval(state, runs=runs, lookup=lookup)

    return GovernanceResult(
        passed=not state.reasons,
        reasons=tuple(state.reasons),
        promote_eval_runs=tuple(runs),
        in_place_edits=tuple(state.in_place),
        gated_paths=tuple(state.gated),
    )


def check_git_production_governance(
    *,
    repo: Path,
    base_ref: str,
    head_ref: str = "HEAD",
    pr_body: str = "",
    verdict_for_run: VerdictLookup | None = None,
    data_dir: Path | None = None,
) -> GovernanceResult:
    """Run governance against a git base…head range.

    Args:
        repo: Repository root.
        base_ref: Merge-base / PR base SHA or ref.
        head_ref: PR head SHA or ref (default ``HEAD``).
        pr_body: Pull-request description.
        verdict_for_run: Optional verdict lookup override.
        data_dir: Optional data root for default verdict resolution.

    Returns:
        ``GovernanceResult`` for the range.
    """
    snap = _diff_snapshot(repo, base_ref=base_ref, head_ref=head_ref)
    return check_production_governance(
        changed_paths=snap.changed_paths,
        base_production_yaml=snap.base_production,
        head_production_yaml=snap.head_production,
        base_model_specs=snap.base_specs,
        head_model_specs=snap.head_specs,
        pr_body=pr_body,
        verdict_for_run=verdict_for_run,
        data_dir=data_dir,
    )


def _collect_in_place_edits(
    state: _GateState,
    *,
    changed: set[str],
    listed_at_base: set[str],
    base_model_specs: Mapping[str, str],
    head_model_specs: Mapping[str, str],
) -> None:
    for name in sorted(listed_at_base):
        rel = model_spec_relpath(name)
        if rel not in changed:
            continue
        base_text = base_model_specs.get(rel)
        head_text = head_model_specs.get(rel)
        if base_text is None:
            continue
        if head_text is None or head_text != base_text:
            state.in_place.append(rel)
            if rel not in state.gated:
                state.gated.append(rel)


def _collect_gated_paths(
    state: _GateState,
    *,
    changed: set[str],
    listed_at_base: set[str],
    listed_at_head: set[str],
    base_production_yaml: str | None,
    head_production_yaml: str | None,
) -> None:
    if PRODUCTION_RELPATH in changed and _live_slate_changed(
        base_production_yaml, head_production_yaml
    ):
        state.gated.append(PRODUCTION_RELPATH)
    for name in sorted(listed_at_base | listed_at_head):
        rel = model_spec_relpath(name)
        if rel in changed and rel not in state.gated:
            state.gated.append(rel)


_LIVE_SLATE_KEYS = frozenset({"models", "plan_horizon", "pipeline", "lines"})


def _live_slate_changed(base_yaml: str | None, head_yaml: str | None) -> bool:
    """True when decision-slate keys differ between base and head production YAML.

    ``shadow_predict`` line additions/edits (#389) do not retarget what we play,
    so they alone do not require a promote-eval gate.
    """
    if base_yaml is None and head_yaml is None:
        return False
    if base_yaml is None or head_yaml is None:
        return True
    base_raw = yaml.safe_load(base_yaml)
    head_raw = yaml.safe_load(head_yaml)
    if not isinstance(base_raw, dict) or not isinstance(head_raw, dict):
        return True
    base_map = cast(dict[str, Any], base_raw)
    head_map = cast(dict[str, Any], head_raw)
    for key in _LIVE_SLATE_KEYS:
        if key == "lines":
            base_lines = _decision_lines_payload(base_map.get("lines"))
            head_lines = _decision_lines_payload(head_map.get("lines"))
            if base_lines != head_lines:
                return True
            continue
        if base_map.get(key) != head_map.get(key):
            return True
    return False


def _decision_lines_payload(raw: object) -> object:
    """Return ``lines`` with ``shadow_predict`` entries stripped for gating."""
    if not isinstance(raw, list):
        return raw
    kept: list[Any] = []
    for item in cast(list[Any], raw):
        if isinstance(item, dict):
            entry = cast(dict[str, Any], item)
            if entry.get("role") == "shadow_predict":
                continue
            kept.append(entry)
        else:
            kept.append(item)
    return kept


def _require_passing_promote_eval(
    state: _GateState,
    *,
    runs: Sequence[str],
    lookup: VerdictLookup,
) -> None:
    if not runs:
        state.reasons.append(
            "production config or listed ModelSpec changed without a "
            "promote-eval link; add `promote-eval-run: <run_id>` with a "
            "passing (keep) verdict"
        )
        return
    passing = False
    details: list[str] = []
    for run_id in runs:
        verdict = lookup(run_id)
        if verdict is None:
            details.append(f"{run_id}=missing")
            continue
        normalised = verdict.strip().lower()
        details.append(f"{run_id}={normalised}")
        if normalised in PASSING_VERDICTS:
            passing = True
    if not passing:
        state.reasons.append(
            "promote-eval link present but no passing (keep) verdict "
            f"({', '.join(details)})"
        )


def _models_or_empty(yaml_text: str | None) -> list[str]:
    if yaml_text is None:
        return []
    return parse_production_models(yaml_text)


def _diff_snapshot(repo: Path, *, base_ref: str, head_ref: str) -> _DiffSnapshot:
    changed = _git_diff_names(repo, base_ref, head_ref)
    interesting = [PRODUCTION_RELPATH]
    for path in changed:
        if path.startswith(MODELS_PREFIX) and path.endswith(".yaml"):
            interesting.append(path)
    base_models_text = _git_show(repo, base_ref, PRODUCTION_RELPATH)
    head_models_text = _git_show(repo, head_ref, PRODUCTION_RELPATH)
    names: set[str] = set()
    if base_models_text is not None:
        names.update(parse_production_models(base_models_text))
    if head_models_text is not None:
        names.update(parse_production_models(head_models_text))
    for name in names:
        interesting.append(model_spec_relpath(name))
    base_specs: dict[str, str] = {}
    head_specs: dict[str, str] = {}
    for rel in sorted(set(interesting)):
        if rel == PRODUCTION_RELPATH:
            continue
        base_text = _git_show(repo, base_ref, rel)
        head_text = _git_show(repo, head_ref, rel)
        if base_text is not None:
            base_specs[rel] = base_text
        if head_text is not None:
            head_specs[rel] = head_text
    return _DiffSnapshot(
        changed_paths=tuple(changed),
        base_production=base_models_text,
        head_production=head_models_text,
        base_specs=base_specs,
        head_specs=head_specs,
    )


def _git_diff_names(repo: Path, base_ref: str, head_ref: str) -> list[str]:
    result = run_command(
        ["git", "diff", "--name-only", f"{base_ref}...{head_ref}"],
        cwd=repo,
        check=True,
        capture_output=True,
        text=True,
    )
    return [line.strip() for line in result.stdout.splitlines() if line.strip()]


def _git_show(repo: Path, ref: str, relpath: str) -> str | None:
    result = run_command(
        ["git", "show", f"{ref}:{relpath}"],
        cwd=repo,
        check=False,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        return None
    return result.stdout


def _registry_verdict(
    run_id: str,
    *,
    registry_path: Path | None,
) -> str | None:
    path = default_results_registry_path() if registry_path is None else registry_path
    if not path.is_file():
        return None
    try:
        registry = load_results_registry(path, require_audits=False)
    except (OSError, ValueError, FileNotFoundError, yaml.YAMLError):
        # Corrupt / unreadable registry must not break PR governance lookup.
        return None
    for entry in registry.entries:
        if entry.id == run_id:
            return entry.verdict
    return None


def _verdict_from_summary(path: Path) -> str | None:
    if not path.is_file():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(payload, dict):
        return None
    verdict = cast(dict[str, Any], payload).get("verdict")
    if not isinstance(verdict, str) or not verdict.strip():
        return None
    return verdict.strip().lower()


__all__ = [
    "MODELS_PREFIX",
    "PASSING_VERDICTS",
    "PRODUCTION_RELPATH",
    "GovernanceResult",
    "check_git_production_governance",
    "check_production_governance",
    "extract_promote_eval_runs",
    "model_spec_relpath",
    "parse_production_models",
    "resolve_promote_eval_verdict",
]
