"""Promote / roll back the moving ``prod`` tag (ops-04).

Gates (all must pass before the tag moves):

1. CI on the candidate commit is green.
2. Not inside a pre-deadline freeze window.
3. Smoke plan / production load succeeds.

Immutable history tags look like ``prod-YYYYMMDD-<shortsha>``. Rollback is the
same path with ``ref=<previous immutable tag>``.
"""

from __future__ import annotations

import logging
import os
import subprocess
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import polars as pl

from fplt.automation.freeze import (
    FreezeWindow,
    active_freeze,
    format_freeze_block,
    freeze_windows,
)
from fplt.automation.production import load_production_ops
from fplt.models.spec import git_sha, load_model_specs

logger = logging.getLogger(__name__)

CiStatus = Literal["success", "failure", "pending", "unknown"]
CiStatusFn = Callable[[str], CiStatus]
PROD_TAG = "prod"
_PROD_TAG_ENV = "FPLT_PROD_TAG"


class ReleaseGateError(RuntimeError):
    """A release gate refused to move ``prod`` (clear operator message)."""


@dataclass(frozen=True)
class ReleaseCandidate:
    """Resolved commit that would become ``prod``.

    Attributes:
        ref: Caller-supplied ref (branch, tag, or sha).
        sha: Full commit sha.
        short_sha: First 7 hex chars of ``sha``.
    """

    ref: str
    sha: str
    short_sha: str


@dataclass(frozen=True)
class SmokeResult:
    """Outcome of the release smoke checks.

    Attributes:
        model: Production ModelSpec name that loaded.
        plan_horizon: Production plan horizon.
        detail: Short human summary.
    """

    model: str
    plan_horizon: int
    detail: str


@dataclass(frozen=True)
class ReleaseCheckResult:
    """All gates passed for a candidate.

    Attributes:
        candidate: Resolved commit.
        smoke: Smoke outcome.
        freeze: ``None`` when outside every freeze window.
        history_tag: Immutable tag that would be created
            (``prod-YYYYMMDD-<shortsha>``).
    """

    candidate: ReleaseCandidate
    smoke: SmokeResult
    freeze: FreezeWindow | None
    history_tag: str


def resolve_candidate(
    ref: str,
    *,
    cwd: Path | None = None,
    rev_parse: Callable[[str], str] | None = None,
) -> ReleaseCandidate:
    """Resolve ``ref`` to a full commit sha.

    Args:
        ref: Branch, tag, or sha.
        cwd: Git working tree (default: process cwd).
        rev_parse: Optional injector; defaults to ``git rev-parse``.

    Returns:
        Resolved ``ReleaseCandidate``.

    Raises:
        ReleaseGateError: Git cannot resolve ``ref``.
    """

    def _default_parse(value: str) -> str:
        return _git_rev_parse(value, cwd=cwd)

    parser: Callable[[str], str] = rev_parse or _default_parse
    try:
        sha = parser(ref).strip()
    except (OSError, subprocess.CalledProcessError, ValueError) as exc:
        raise ReleaseGateError(f"cannot resolve ref {ref!r}: {exc}") from exc
    if len(sha) < 7:
        raise ReleaseGateError(f"cannot resolve ref {ref!r} to a commit sha")
    short = sha[:7]
    return ReleaseCandidate(ref=ref, sha=sha, short_sha=short)


def assert_ci_green(sha: str, *, status_fn: CiStatusFn) -> None:
    """Refuse when CI on ``sha`` is not green.

    Args:
        sha: Full commit sha.
        status_fn: Returns ``success`` / ``failure`` / ``pending`` / ``unknown``.

    Raises:
        ReleaseGateError: Status is anything other than ``success``.
    """
    status = status_fn(sha)
    if status == "success":
        return
    if status == "failure":
        raise ReleaseGateError(f"refusing release: CI is red for {sha}")
    if status == "pending":
        raise ReleaseGateError(f"refusing release: CI still pending for {sha}")
    raise ReleaseGateError(
        f"refusing release: CI status {status!r} for {sha} (need success)"
    )


def assert_outside_freeze(
    now: datetime,
    gameweeks: pl.DataFrame,
    fixtures: pl.DataFrame,
) -> None:
    """Refuse when ``now`` falls inside any freeze window.

    Args:
        now: Clock reading.
        gameweeks: Staged gameweeks.
        fixtures: Staged fixtures.

    Raises:
        ReleaseGateError: Inside a freeze window.
    """
    window = active_freeze(now, freeze_windows(gameweeks, fixtures))
    if window is not None:
        raise ReleaseGateError(format_freeze_block(window, now=now))


def run_smoke_plan(
    *,
    environ: Mapping[str, str] | None = None,
    specs_dir: Path | None = None,
) -> SmokeResult:
    """Dry-run smoke: production ops + ModelSpec must load cleanly.

    A full predict→plan against live data is too heavy for the gate; this
    checks that the production pipeline the ops jobs will run is importable
    and the ModelSpec resolves (including native-mc params).

    Args:
        environ: Env for ``load_production_ops`` (default ``os.environ``).
        specs_dir: Optional ModelSpec directory override.

    Returns:
        ``SmokeResult`` on success.

    Raises:
        ReleaseGateError: Production config or ModelSpec fails to load.
    """
    try:
        ops = load_production_ops(environ=environ)
        if specs_dir is not None:
            specs = load_model_specs(specs_dir)
        else:
            specs = load_model_specs()
        name = ops.models[0]
        if name not in specs:
            raise ReleaseGateError(
                f"refusing release: smoke plan failed — production model "
                f"{name!r} is not in the ModelSpec registry"
            )
        spec = specs[name]
        _ = spec.model_id
    except ReleaseGateError:
        raise
    except Exception as exc:  # noqa: BLE001 — surface any load failure clearly
        raise ReleaseGateError(f"refusing release: smoke plan failed — {exc}") from exc
    detail = f"loaded production model {name!r} (horizon={ops.plan_horizon})"
    logger.info("release smoke ok: %s", detail)
    return SmokeResult(model=name, plan_horizon=ops.plan_horizon, detail=detail)


def history_tag_name(sha: str, *, when: datetime | None = None) -> str:
    """Immutable tag name ``prod-YYYYMMDD-<shortsha>``.

    Args:
        sha: Full or short commit sha.
        when: Date stamp (default: now UTC).

    Returns:
        Tag name.
    """
    moment = when if when is not None else datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    else:
        moment = moment.astimezone(UTC)
    return f"prod-{moment.strftime('%Y%m%d')}-{sha[:7]}"


def evaluate_release_gates(
    ref: str,
    *,
    now: datetime,
    gameweeks: pl.DataFrame,
    fixtures: pl.DataFrame,
    status_fn: CiStatusFn,
    cwd: Path | None = None,
    rev_parse: Callable[[str], str] | None = None,
    environ: Mapping[str, str] | None = None,
    specs_dir: Path | None = None,
) -> ReleaseCheckResult:
    """Run CI, freeze, and smoke gates for ``ref``.

    Args:
        ref: Candidate branch / tag / sha.
        now: Clock for the freeze check.
        gameweeks: Staged gameweeks (may be empty → no freeze).
        fixtures: Staged fixtures.
        status_fn: CI status lookup.
        cwd: Git cwd for resolving ``ref``.
        rev_parse: Optional git resolver injector.
        environ: Env for smoke / production load.
        specs_dir: Optional ModelSpec directory.

    Returns:
        ``ReleaseCheckResult`` when every gate passes.

    Raises:
        ReleaseGateError: Any gate fails (message says which).
    """
    candidate = resolve_candidate(ref, cwd=cwd, rev_parse=rev_parse)
    assert_ci_green(candidate.sha, status_fn=status_fn)
    assert_outside_freeze(now, gameweeks, fixtures)
    smoke = run_smoke_plan(environ=environ, specs_dir=specs_dir)
    tag = history_tag_name(candidate.sha, when=now)
    return ReleaseCheckResult(
        candidate=candidate,
        smoke=smoke,
        freeze=None,
        history_tag=tag,
    )


def promote_prod_tags(
    candidate: ReleaseCandidate,
    *,
    history_tag: str,
    cwd: Path | None = None,
    dry_run: bool = False,
    runner: Callable[[Sequence[str]], None] | None = None,
) -> list[str]:
    """Create the immutable history tag and move the moving ``prod`` tag.

    Args:
        candidate: Resolved commit.
        history_tag: ``prod-YYYYMMDD-<shortsha>``.
        cwd: Git working tree.
        dry_run: When true, return the commands without running them.
        runner: Optional injector for ``git`` commands.

    Returns:
        The git command lines that were (or would be) run.
    """
    snap_msg = f"prod snapshot {candidate.sha}"
    move_msg = f"prod -> {candidate.sha}"
    commands = [
        ["git", "tag", "-a", history_tag, "-m", snap_msg, candidate.sha],
        ["git", "tag", "-f", "-a", PROD_TAG, "-m", move_msg, candidate.sha],
        ["git", "push", "origin", history_tag],
        ["git", "push", "origin", f"refs/tags/{PROD_TAG}", "--force"],
    ]
    rendered = [" ".join(cmd) for cmd in commands]
    if dry_run:
        return rendered

    def _default_run(cmd: Sequence[str]) -> None:
        _run_git(cmd, cwd=cwd)

    run: Callable[[Sequence[str]], None] = runner or _default_run
    for cmd in commands:
        logger.info("release: %s", " ".join(cmd))
        run(cmd)
    return rendered


def resolve_prod_tag_label(*, environ: Mapping[str, str] | None = None) -> str | None:
    """Return the ops ``prod`` tag label from the environment, if set.

    Ops workflows set ``FPLT_PROD_TAG=prod`` after checking out the moving tag
    so plan manifests can record it.

    Args:
        environ: Env mapping (default ``os.environ``).

    Returns:
        Stripped tag label, or ``None`` when unset / empty.
    """
    env = os.environ if environ is None else environ
    raw = env.get(_PROD_TAG_ENV)
    if raw is None:
        return None
    text = raw.strip()
    return text or None


def current_code_sha() -> str:
    """Return ``git_sha()`` for plan metadata / sidecars."""
    return git_sha()


def _git_rev_parse(ref: str, *, cwd: Path | None) -> str:
    completed = subprocess.run(
        ["git", "rev-parse", f"{ref}^{{commit}}"],
        check=True,
        capture_output=True,
        text=True,
        cwd=cwd,
    )
    return completed.stdout.strip()


def _run_git(cmd: Sequence[str], *, cwd: Path | None) -> None:
    subprocess.run(list(cmd), check=True, cwd=cwd)
