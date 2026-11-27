"""Submit experiment / batch / comparison manifests from EvalSpecs or model names.

Storage CLI surface for data-model step 7: ``fplt exp submit --batch`` freezes
gates, mints a batch, and resolves arms with catalog reuse so a shared control
is created once across the batch. ``fplt compare --spec`` mints a
``comparison``-flavour experiment (no verdict). Does not run promote-eval or
walk-forward compute — that stays on ``promote-eval`` / legacy ``compare``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import TYPE_CHECKING

from fplt.backtest.produce_ledger import uv_lock_sha256
from fplt.models.spec import ModelSpec, git_sha, load_model_specs
from fplt.models.spec_identity import spec_hash as model_spec_hash
from fplt.schemas.manifests import (
    AsOfDeadline,
    BatchManifest,
    ExperimentArm,
    ExperimentEvent,
    ExperimentWindow,
)
from fplt.store.builds import latest_build_id
from fplt.store.experiments import (
    append_experiment_event,
    arm_reuse_key,
    bind_arm_run,
    hash_eval_spec_payload,
    mint_batch_id,
    resolve_or_create_arm_run,
    submit_experiment,
    write_batch_manifest,
)
from fplt.store.runs import detect_host
from fplt.utils.logging import log_context

if TYPE_CHECKING:
    from fplt.backtest.promote_eval import EvalSpec


@dataclass(frozen=True)
class ArmBind:
    """One resolved arm pointer after submit."""

    experiment_id: str
    arm: str
    run_id: str
    reused: bool
    spec_label: str


@dataclass(frozen=True)
class BatchSubmitResult:
    """Outcome of ``submit_batch``."""

    batch_id: str
    experiment_ids: list[str]
    arms: list[ArmBind]
    fresh_runs: int
    reused_runs: int


@dataclass(frozen=True)
class ComparisonSubmitResult:
    """Outcome of ``submit_comparison``."""

    experiment_id: str
    arms: list[ArmBind]
    fresh_runs: int
    reused_runs: int
    build_id: str | None


@dataclass
class _ArmPlan:
    """Internal arm declaration before resolve."""

    name: str
    role: str
    spec: ModelSpec
    digest: str


def load_eval_ref(ref: str, *, evals_dir: Path | None = None) -> EvalSpec:
    """Load an EvalSpec from a stem or a ``.yaml`` path.

    Args:
        ref: Eval name (``configs/evals/<ref>.yaml``) or filesystem path.
        evals_dir: Override evals folder when ``ref`` is a stem.

    Returns:
        Validated ``EvalSpec``.

    Raises:
        FileNotFoundError: Path or stem does not exist.
        ValueError: YAML fails validation.
    """
    from fplt.backtest.promote_eval import load_eval_spec

    path = Path(ref)
    if path.suffix in {".yaml", ".yml"} and path.is_file():
        return load_eval_spec(path.stem, directory=path.parent)
    return load_eval_spec(ref, directory=evals_dir)


def default_evals_dir() -> Path:
    """Return ``configs/evals`` at the repo root."""
    from fplt.backtest.promote_eval import default_evals_dir as _default

    return _default()


def submit_batch(
    data_dir: Path,
    specs: Sequence[EvalSpec],
    *,
    batch_label: str,
    models_dir: Path | None = None,
    build_id: str | None = None,
    code_sha: str | None = None,
    lockfile_hash: str | None = None,
    submitted_by: str = "exp-submit",
    host: str | None = None,
    submitted_at: datetime | None = None,
    batch_id: str | None = None,
) -> BatchSubmitResult:
    """Mint a batch and submit each EvalSpec as an ``ab`` experiment.

    Shared arm reuse keys across the batch resolve to one run (exact catalog
    match). Gates and ``eval_spec_hash`` are frozen per experiment at submit.

    Args:
        data_dir: Local data root.
        specs: EvalSpecs to submit (at least one).
        batch_label: Human batch label (e.g. ``nightly``).
        models_dir: Override ``configs/models``.
        build_id: Pin curated build; default ``latest_build_id`` or ``None``.
        code_sha: Git HEAD; default ``git_sha()``.
        lockfile_hash: ``uv.lock`` digest; default ``uv_lock_sha256()``.
        submitted_by: Manifest author label.
        host: Host label; default ``detect_host()``.
        submitted_at: Submit timestamp (UTC).
        batch_id: Optional explicit batch id (tests).

    Returns:
        ``BatchSubmitResult`` with experiment ids and arm binds.

    Raises:
        ValueError: Empty ``specs`` or unknown ModelSpec names.
    """
    if not specs:
        raise ValueError("submit_batch requires at least one EvalSpec")
    when = _aware(submitted_at if submitted_at is not None else datetime.now(UTC))
    bid = batch_id if batch_id is not None else mint_batch_id(when=when)
    code = code_sha if code_sha is not None else git_sha()
    lock = lockfile_hash if lockfile_hash is not None else uv_lock_sha256()
    pinned_build = build_id if build_id is not None else latest_build_id(data_dir)
    host_label = host if host is not None else detect_host()
    model_specs = load_model_specs(models_dir)

    experiment_ids: list[str] = []
    binds: list[ArmBind] = []
    fresh = 0
    reused = 0

    for spec in specs:
        arms_plan = _ab_arm_plans(spec, model_specs)
        windows = [
            ExperimentWindow(
                role="primary",
                seasons=list(spec.window.seasons),
                from_gw=spec.window.from_gw,
                to_gw=spec.window.to_gw,
            )
        ]
        deadlines = _window_deadlines(
            spec.window.seasons, spec.window.from_gw, spec.window.to_gw
        )
        horizon = max(spec.suites.steps, 1)
        seeds = _crn_seeds(arms_plan)
        eval_hash = hash_eval_spec_payload(spec.model_dump(mode="json"))
        gates = spec.gates.model_dump(mode="json")
        experiment_arms = [
            ExperimentArm(
                arm=plan.name,
                role=plan.role,  # type: ignore[arg-type]
                spec_hash=plan.digest,
                spec_label=plan.spec.name,
            )
            for plan in arms_plan
        ]
        manifest = submit_experiment(
            data_dir,
            flavour="ab",
            question=spec.question,
            arms=experiment_arms,
            windows=windows,
            horizon=horizon,
            code_sha=code,
            lockfile_hash=lock,
            gates=gates,
            metrics=list(spec.gates.keep_if.prediction_any),
            seeds=seeds,
            build_id=pinned_build,
            as_of=deadlines,
            eval_spec=spec.name,
            eval_spec_hash=eval_hash,
            batch_id=bid,
            submitted_by=submitted_by,
            host=host_label,
            submitted_at=when,
        )
        exp_id = manifest.experiment_id
        experiment_ids.append(exp_id)
        first = deadlines[0] if deadlines else None
        with log_context(
            experiment_id=exp_id,
            season=None if first is None else first.season,
            gw=None if first is None else first.gw,
        ):
            for plan in arms_plan:
                arm_seeds = {"base": int(plan.spec.seed)}
                key = arm_reuse_key(
                    spec_hash=plan.digest,
                    code_sha=code,
                    lockfile_hash=lock,
                    build_id=pinned_build,
                    deadlines=deadlines,
                    horizon=horizon,
                    seeds=arm_seeds,
                )
                append_experiment_event(
                    data_dir,
                    exp_id,
                    ExperimentEvent(event="arm_started", at=when, arm=plan.name),
                )
                run_id, was_reused = resolve_or_create_arm_run(
                    data_dir,
                    key=key,
                    spec_label=plan.spec.name,
                    model_id=plan.spec.model_id,
                    experiment_id=exp_id,
                    arm=plan.name,
                    purpose=f"batch {batch_label} {spec.name} {plan.name}",
                    host=host_label,
                )
                bind_arm_run(
                    data_dir,
                    exp_id,
                    arm=plan.name,
                    run_id=run_id,
                    reused=was_reused,
                    at=when,
                )
                binds.append(
                    ArmBind(
                        experiment_id=exp_id,
                        arm=plan.name,
                        run_id=run_id,
                        reused=was_reused,
                        spec_label=plan.spec.name,
                    )
                )
                if was_reused:
                    reused += 1
                else:
                    fresh += 1

    write_batch_manifest(
        data_dir,
        BatchManifest(
            batch_id=bid,
            label=batch_label.strip() or "batch",
            experiment_ids=experiment_ids,
            submitted_at=when,
            submitted_by=submitted_by,
            host=host_label,
        ),
    )
    return BatchSubmitResult(
        batch_id=bid,
        experiment_ids=experiment_ids,
        arms=binds,
        fresh_runs=fresh,
        reused_runs=reused,
    )


def submit_comparison(
    data_dir: Path,
    spec_names: Sequence[str],
    *,
    horizon: int,
    question: str | None = None,
    season: str | None = None,
    gw: int | None = None,
    models_dir: Path | None = None,
    build_id: str | None = None,
    code_sha: str | None = None,
    lockfile_hash: str | None = None,
    submitted_by: str = "compare",
    host: str | None = None,
    submitted_at: datetime | None = None,
    experiment_id: str | None = None,
) -> ComparisonSubmitResult:
    """Mint a ``comparison``-flavour experiment for named ModelSpecs.

    Pins ``build_id`` (latest when omitted) and a single ``as_of`` deadline.
    No gates / verdict — differences are inspected later.

    Args:
        data_dir: Local data root.
        spec_names: At least two ModelSpec names (first = control role).
        horizon: Prediction horizon.
        question: One-line question; default names the specs and horizon.
        season: Season for the as-of deadline (default ``2024-25``).
        gw: Gameweek for the as-of deadline (default ``1``).
        models_dir: Override ``configs/models``.
        build_id: Pin curated build; default latest or ``None``.
        code_sha: Git HEAD; default ``git_sha()``.
        lockfile_hash: ``uv.lock`` digest.
        submitted_by: Manifest author label.
        host: Host label.
        submitted_at: Submit timestamp.
        experiment_id: Optional explicit id (tests).

    Returns:
        ``ComparisonSubmitResult``.

    Raises:
        ValueError: Fewer than two specs or unknown names.
    """
    names = [n.strip() for n in spec_names if n and n.strip()]
    if len(names) < 2:
        raise ValueError("comparison requires at least two --spec names")
    if len(names) != len(set(names)):
        raise ValueError("comparison --spec names must be unique")
    if horizon < 1:
        raise ValueError("horizon must be >= 1")

    when = _aware(submitted_at if submitted_at is not None else datetime.now(UTC))
    code = code_sha if code_sha is not None else git_sha()
    lock = lockfile_hash if lockfile_hash is not None else uv_lock_sha256()
    pinned_build = build_id if build_id is not None else latest_build_id(data_dir)
    host_label = host if host is not None else detect_host()
    model_specs = load_model_specs(models_dir)

    plans: list[_ArmPlan] = []
    for index, name in enumerate(names):
        model = _require_model(name, model_specs)
        if index == 0:
            role = "control"
        elif index == 1:
            role = "treatment"
        else:
            role = "baseline"
        plans.append(
            _ArmPlan(
                name=name,
                role=role,
                spec=model,
                digest=model_spec_hash(model),
            )
        )

    as_season = season if season is not None else "2024-25"
    as_gw = gw if gw is not None else 1
    deadlines = [
        AsOfDeadline(
            season=as_season,
            gw=as_gw,
            deadline_utc=synthetic_deadline(as_season, as_gw),
        )
    ]
    windows = [
        ExperimentWindow(
            role="primary",
            seasons=[as_season],
            from_gw=as_gw,
            to_gw=as_gw,
        )
    ]
    q = (
        question.strip()
        if question and question.strip()
        else f"H={horizon} predicted points: {' vs '.join(names)}"
    )
    seeds = {"base": int(plans[0].spec.seed)}
    experiment_arms = [
        ExperimentArm(
            arm=plan.name,
            role=plan.role,  # type: ignore[arg-type]
            spec_hash=plan.digest,
            spec_label=plan.spec.name,
        )
        for plan in plans
    ]
    manifest = submit_experiment(
        data_dir,
        flavour="comparison",
        question=q,
        arms=experiment_arms,
        windows=windows,
        horizon=horizon,
        code_sha=code,
        lockfile_hash=lock,
        gates={},
        metrics=[],
        seeds=seeds,
        build_id=pinned_build,
        as_of=deadlines,
        submitted_by=submitted_by,
        host=host_label,
        submitted_at=when,
        experiment_id=experiment_id,
    )
    binds: list[ArmBind] = []
    fresh = 0
    reused = 0
    with log_context(
        experiment_id=manifest.experiment_id,
        season=as_season,
        gw=as_gw,
    ):
        for plan in plans:
            key = arm_reuse_key(
                spec_hash=plan.digest,
                code_sha=code,
                lockfile_hash=lock,
                build_id=pinned_build,
                deadlines=deadlines,
                horizon=horizon,
                seeds={"base": int(plan.spec.seed)},
            )
            append_experiment_event(
                data_dir,
                manifest.experiment_id,
                ExperimentEvent(event="arm_started", at=when, arm=plan.name),
            )
            run_id, was_reused = resolve_or_create_arm_run(
                data_dir,
                key=key,
                spec_label=plan.spec.name,
                model_id=plan.spec.model_id,
                experiment_id=manifest.experiment_id,
                arm=plan.name,
                purpose=f"comparison H={horizon} {plan.name}",
                host=host_label,
            )
            bind_arm_run(
                data_dir,
                manifest.experiment_id,
                arm=plan.name,
                run_id=run_id,
                reused=was_reused,
                at=when,
            )
            binds.append(
                ArmBind(
                    experiment_id=manifest.experiment_id,
                    arm=plan.name,
                    run_id=run_id,
                    reused=was_reused,
                    spec_label=plan.spec.name,
                )
            )
            if was_reused:
                reused += 1
            else:
                fresh += 1
    return ComparisonSubmitResult(
        experiment_id=manifest.experiment_id,
        arms=binds,
        fresh_runs=fresh,
        reused_runs=reused,
        build_id=pinned_build,
    )


def synthetic_deadline(season: str, gw: int) -> datetime:
    """Stable UTC deadline derived from season / GW for arm-reuse keys."""
    year = int(season.split("-", 1)[0])
    return datetime(year, 8, 1, 11, 0, 0, tzinfo=UTC) + timedelta(weeks=gw - 1)


def _ab_arm_plans(spec: EvalSpec, model_specs: dict[str, ModelSpec]) -> list[_ArmPlan]:
    rows: list[tuple[str, str, str]] = [
        ("control", "control", spec.control),
        ("treatment", "treatment", spec.treatment),
    ]
    for extra in spec.arms:
        rows.append((extra, "baseline", extra))
    plans: list[_ArmPlan] = []
    for arm_name, role, model_name in rows:
        model = _require_model(model_name, model_specs)
        plans.append(
            _ArmPlan(
                name=arm_name,
                role=role,
                spec=model,
                digest=model_spec_hash(model),
            )
        )
    return plans


def _require_model(name: str, model_specs: dict[str, ModelSpec]) -> ModelSpec:
    try:
        return model_specs[name]
    except KeyError as exc:
        raise ValueError(f"unknown ModelSpec {name!r}") from exc


def _window_deadlines(
    seasons: Sequence[str], from_gw: int, to_gw: int
) -> list[AsOfDeadline]:
    first = seasons[0]
    last = seasons[-1]
    return [
        AsOfDeadline(
            season=first, gw=from_gw, deadline_utc=synthetic_deadline(first, from_gw)
        ),
        AsOfDeadline(
            season=last, gw=to_gw, deadline_utc=synthetic_deadline(last, to_gw)
        ),
    ]


def _crn_seeds(plans: Sequence[_ArmPlan]) -> dict[str, int]:
    return {"eval": int(plans[0].spec.seed) if plans else 0}


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = [
    "ArmBind",
    "BatchSubmitResult",
    "ComparisonSubmitResult",
    "default_evals_dir",
    "load_eval_ref",
    "submit_batch",
    "submit_comparison",
    "synthetic_deadline",
]
