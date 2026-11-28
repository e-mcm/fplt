"""Component A/B experiment runner (toolkit §2.4 / ``fplt exp component``).

Fits a base layered pipeline once per deadline, caches component outputs
(§2.2), applies arm patches (§2.1), reuses unchanged upstream fits, then
scores via the Measure interface (§2.3) at the component stage and the
player-fixture (prediction) stage. Results land under ``experiments/`` for
``fplt-lab``.
"""

from __future__ import annotations

import json
import logging
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl

from fplt.backtest.forecast_scores import compute_forecast_scores
from fplt.backtest.measure import (
    empty_measure_metrics,
    get_measure,
    validate_measure_metrics,
)
from fplt.backtest.minutes_measurement import brier_score, log_score
from fplt.backtest.produce_ledger import uv_lock_sha256
from fplt.data.outcomes import build_player_fixture_outcomes
from fplt.data.pit import PITData, PITView
from fplt.data.store import DataStore
from fplt.models.component_cache import (
    LAYERED_COMPONENT_SLOTS,
    component_hashes_for_spec,
    fit_or_load_component,
    reusable_slots,
)
from fplt.models.composite import CompositePredictor
from fplt.models.distributions import MinutesDist
from fplt.models.pipeline_patch import ResolvedArm, apply_patch
from fplt.models.pipeline_spec import (
    PipelineSpec,
    live_pipeline_defaults,
    pipeline_hash,
    resolve_pipeline_model,
)
from fplt.models.spec import ModelSpec, get_component, git_sha, load_model_specs
from fplt.models.universe import player_fixtures
from fplt.schemas.component_eval import COMPONENT_EVAL_COLUMNS
from fplt.schemas.experiment_results import ArmMetricsSchema
from fplt.schemas.manifests import (
    ExperimentArm,
    ExperimentEvent,
    ExperimentWindow,
)
from fplt.store.experiments import (
    append_experiment_event,
    submit_experiment,
    write_experiment_results,
)
from fplt.store.points_pmf import points_pmf_tables_from_sample
from fplt.store.runs import detect_host
from fplt.types import GWKey
from fplt.utils.frames import require_dataframe
from fplt.utils.timing import timed_phase, timing_session

logger = logging.getLogger(__name__)

_WINDOW_RE = re.compile(
    r"^(?P<season>\d{4}-\d{2})(?::(?P<from_gw>\d{1,2})-(?P<to_gw>\d{1,2}))?$"
)
_ARM_KV_RE = re.compile(r"^([A-Za-z_][\w.]*)=(.+)$")
CONTROL_ARM = "control"


@dataclass(frozen=True)
class ParsedWindow:
    """Season + inclusive GW range for a component experiment."""

    season: str
    from_gw: int
    to_gw: int

    def as_ofs(self) -> list[GWKey]:
        """Return ``GWKey`` deadlines covering ``from_gw``…``to_gw``."""
        return [GWKey(self.season, gw) for gw in range(self.from_gw, self.to_gw + 1)]

    def label(self) -> str:
        """Stable window label for Measure / arm_metrics rows."""
        if self.from_gw == 1 and self.to_gw == 38:
            return self.season
        return f"{self.season}:{self.from_gw}-{self.to_gw}"


@dataclass
class CacheStats:
    """Per-slot fit_or_load hit / miss counts across the run."""

    hits: dict[str, int] = field(default_factory=dict[str, int])
    misses: dict[str, int] = field(default_factory=dict[str, int])

    def record(self, slot: str, *, hit: bool) -> None:
        """Increment hit or miss for ``slot``."""
        bucket = self.hits if hit else self.misses
        bucket[slot] = bucket.get(slot, 0) + 1


@dataclass(frozen=True)
class ComponentExperimentResult:
    """Outcome of ``run_component_experiment``."""

    experiment_id: str
    arm_metrics: pl.DataFrame
    measure_metrics: pl.DataFrame
    cache: CacheStats
    phases: Mapping[str, float]
    window: ParsedWindow
    arms: tuple[str, ...]


def parse_component_window(raw: str) -> ParsedWindow:
    """Parse ``season`` or ``season:gw-gw`` into a ``ParsedWindow``.

    Args:
        raw: CLI ``--window`` value (e.g. ``2024-25`` or ``2024-25:1-10``).

    Returns:
        Parsed season and inclusive GW bounds (default ``1``…``38``).

    Raises:
        ValueError: Malformed window or ``from_gw`` > ``to_gw``.
    """
    cleaned = raw.strip()
    match = _WINDOW_RE.match(cleaned)
    if match is None:
        raise ValueError(f"window must be 'YYYY-YY' or 'YYYY-YY:gw-gw', got {raw!r}")
    season = match.group("season")
    from_raw, to_raw = match.group("from_gw"), match.group("to_gw")
    from_gw = 1 if from_raw is None else int(from_raw)
    to_gw = 38 if to_raw is None else int(to_raw)
    if not 1 <= from_gw <= 38 or not 1 <= to_gw <= 38:
        raise ValueError(f"gameweeks must be in 1..38, got {from_gw}-{to_gw}")
    if from_gw > to_gw:
        raise ValueError(f"from_gw ({from_gw}) must be <= to_gw ({to_gw})")
    return ParsedWindow(season=season, from_gw=from_gw, to_gw=to_gw)


def parse_arm_patch(raw: str, *, component: str) -> dict[str, Any]:
    """Parse one ``--arm`` value into a flat pipeline patch.

    Accepted forms:
    - JSON object (``{"model.components.minutes.params.logistic_blend": 0.8}``)
    - dotted ``path=value`` (JSON-decoded value when possible)
    - registered ModelSpec name → ``model.components.<component>: <name>``
    - path to a YAML/JSON file containing a mapping

    Args:
        raw: CLI arm string.
        component: Slot being experimented on (e.g. ``minutes``).

    Returns:
        Flat dotted-path patch for ``apply_patch``.

    Raises:
        ValueError: Empty / unparseable arm.
        FileNotFoundError: Path form points at a missing file.
    """
    text = raw.strip()
    if not text:
        raise ValueError("arm patch must be non-empty")
    path = Path(text)
    if path.suffix in {".yaml", ".yml", ".json"} and path.is_file():
        payload = path.read_text(encoding="utf-8")
        if path.suffix == ".json":
            loaded = json.loads(payload)
        else:
            import yaml

            loaded = yaml.safe_load(payload)
        if not isinstance(loaded, Mapping):
            raise ValueError(f"arm file {path} must contain a mapping")
        mapping = cast(Mapping[Any, Any], loaded)
        return {str(key): value for key, value in mapping.items()}
    if text.startswith(("{", "[")):
        loaded = json.loads(text)
        if not isinstance(loaded, Mapping):
            raise ValueError("JSON arm must be an object")
        mapping = cast(Mapping[Any, Any], loaded)
        return {str(key): value for key, value in mapping.items()}
    kv = _ARM_KV_RE.match(text)
    if kv is not None:
        key, value_raw = kv.group(1), kv.group(2)
        try:
            value: Any = json.loads(value_raw)
        except json.JSONDecodeError:
            value = value_raw
        return {key: value}
    specs = load_model_specs()
    if text in specs:
        return {f"model.components.{component}": text}
    raise ValueError(
        f"unrecognised arm {raw!r}; pass JSON, path=value, "
        f"a registered ModelSpec name, or a YAML/JSON file"
    )


def load_base_pipeline(
    base: str, *, specs: dict[str, ModelSpec] | None = None
) -> PipelineSpec:
    """Resolve ``--base`` to a ``PipelineSpec``.

    ``production`` loads the live ops pipeline. A registered ModelSpec name is
    wrapped with ``live_pipeline_defaults`` (python MC for experiment VMs).

    Args:
        base: ``production`` or a ModelSpec / pipeline label.
        specs: Optional ModelSpec registry override.

    Returns:
        The base pipeline to patch.

    Raises:
        KeyError: Named model is not registered.
        ValueError: Blank base.
    """
    name = base.strip()
    if not name:
        raise ValueError("base pipeline must be non-empty")
    if name == "production":
        from fplt.automation.production import load_production_pipeline

        return load_production_pipeline()
    loaded = specs if specs is not None else load_model_specs()
    if name not in loaded:
        raise KeyError(f"unknown base pipeline / ModelSpec {name!r}")
    model = loaded[name]
    backend = str(model.params.get("mc_backend", "python"))
    if backend not in {"python", "native"}:
        backend = "python"
    return live_pipeline_defaults(
        model=name,
        name=name,
        mc_backend=backend,  # type: ignore[arg-type]
    )


def model_for_pipeline(pipeline: PipelineSpec) -> ModelSpec:
    """Resolve the layered ``ModelSpec``, honouring ``env.mc_backend``."""
    model = resolve_pipeline_model(pipeline)
    desired = pipeline.env.mc_backend
    current = str(model.params.get("mc_backend", "python"))
    if current == desired:
        return model
    return model.model_copy(update={"params": {**model.params, "mc_backend": desired}})


def measure_to_arm_metrics(measure: pl.DataFrame) -> pl.DataFrame:
    """Project Measure rows onto ``ArmMetricsSchema`` (preferred slices).

    Keeps ``overall`` / ``decision_pool`` / ``source=fpl_api`` for components
    and ``all`` for prediction; drops the ``slice`` column.
    """
    if measure.is_empty():
        return require_dataframe(
            ArmMetricsSchema.validate(
                pl.DataFrame(
                    schema={
                        "arm": pl.Utf8(),
                        "window": pl.Utf8(),
                        "metric": pl.Utf8(),
                        "value": pl.Float64(),
                        "se": pl.Float64(),
                        "n": pl.Int64(),
                        "stage": pl.Utf8(),
                        "schema_version": pl.Int64(),
                    }
                )
            )
        )
    preferred = measure.filter(
        (
            (pl.col("stage") == "components")
            & pl.col("slice").is_in(["overall", "decision_pool", "source=fpl_api"])
        )
        | ((pl.col("stage") == "prediction") & (pl.col("slice") == "all"))
    )
    if preferred.is_empty():
        preferred = measure
    # One row per arm×window×metric×stage (prefer overall over source=…).
    ranked = preferred.with_columns(
        pl.when(pl.col("slice") == "overall")
        .then(0)
        .when(pl.col("slice") == "decision_pool")
        .then(1)
        .when(pl.col("slice") == "all")
        .then(0)
        .otherwise(2)
        .alias("_rank")
    ).sort("_rank")
    deduped = ranked.unique(
        subset=["arm", "window", "metric", "stage"], keep="first"
    ).drop(["slice", "_rank"])
    return require_dataframe(ArmMetricsSchema.validate(deduped))


def run_component_experiment(
    data_dir: Path,
    *,
    base: str | PipelineSpec,
    component: str,
    arms: Sequence[str | Mapping[str, Any]],
    window: str | ParsedWindow,
    code_sha: str | None = None,
    lockfile_hash: str | None = None,
    build_id: str = "local",
    experiment_id: str | None = None,
    submitted_by: str = "exp-component",
    host: str | None = None,
    n_sims_override: int | None = None,
) -> ComponentExperimentResult:
    """Run a component A/B with cached upstream fits and Measure scoring.

    Args:
        data_dir: Local data root (``$FPLT_DATA_DIR`` / lab mirror).
        base: ``production``, a ModelSpec name, or an inline ``PipelineSpec``.
        component: Slot under test (``minutes``, ``score``, …).
        arms: Treatment arm patches (control is always added). Each entry is a
            CLI string (see ``parse_arm_patch``) or a flat mapping.
        window: ``season[:gw-gw]`` string or a ``ParsedWindow``.
        code_sha: Git HEAD; default ``git_sha()``.
        lockfile_hash: ``uv.lock`` digest; default ``uv_lock_sha256()``.
        build_id: Curated build id stamped into cache keys.
        experiment_id: Optional explicit experiment id (tests).
        submitted_by: Manifest author label.
        host: Host label; default ``detect_host()``.
        n_sims_override: Optional cheap sim count (tests / screening).

    Returns:
        Experiment id, Measure + arm_metrics tables, and cache stats.

    Raises:
        ValueError: Unknown component slot, empty arms, or bad window.
        KeyError: Base / arm ModelSpec missing from the registry.
    """
    if component not in LAYERED_COMPONENT_SLOTS:
        raise ValueError(
            f"component must be one of {LAYERED_COMPONENT_SLOTS}, got {component!r}"
        )
    parsed = (
        window if isinstance(window, ParsedWindow) else parse_component_window(window)
    )
    base_pipeline = base if isinstance(base, PipelineSpec) else load_base_pipeline(base)
    specs = load_model_specs()
    resolved_arms = _resolve_arms(
        base_pipeline,
        component=component,
        arms=arms,
        specs=specs,
        n_sims_override=n_sims_override,
    )
    sha = code_sha if code_sha is not None else git_sha()
    lock_hash = lockfile_hash if lockfile_hash is not None else uv_lock_sha256()
    host_label = host if host is not None else detect_host()
    window_label = parsed.label()

    store = DataStore(data_dir)
    pit = PITData(store)
    outcomes = build_player_fixture_outcomes(store)
    cache = CacheStats()
    component_summaries: list[pl.DataFrame] = []
    forecast_scores: list[pl.DataFrame] = []

    with timing_session() as session:
        with timed_phase("exp_component"):
            for as_of in parsed.as_ofs():
                view = pit.view(as_of)
                control_spec = resolved_arms[0].model
                control_fitted = _fit_layered_cached(
                    data_dir,
                    spec=control_spec,
                    view=view,
                    as_of=as_of,
                    code_sha=sha,
                    lockfile_hash=lock_hash,
                    build_id=build_id,
                    cache=cache,
                )
                control_model = CompositePredictor(control_spec)
                control_model.fit(view, precomputed=control_fitted)
                _score_arm(
                    arm_label=CONTROL_ARM,
                    model=control_model,
                    view=view,
                    as_of=as_of,
                    store=store,
                    outcomes=outcomes,
                    component=component,
                    component_summaries=component_summaries,
                    forecast_scores=forecast_scores,
                )
                for arm in resolved_arms[1:]:
                    reuse = reusable_slots(control_spec, arm.model)
                    arm_fitted = _fit_layered_cached(
                        data_dir,
                        spec=arm.model,
                        view=view,
                        as_of=as_of,
                        code_sha=sha,
                        lockfile_hash=lock_hash,
                        build_id=build_id,
                        cache=cache,
                        precomputed_prefer=control_fitted,
                        reusable=reuse,
                    )
                    treatment = CompositePredictor(arm.model)
                    treatment.fit(view, precomputed=arm_fitted)
                    _score_arm(
                        arm_label=arm.label,
                        model=treatment,
                        view=view,
                        as_of=as_of,
                        store=store,
                        outcomes=outcomes,
                        component=component,
                        component_summaries=component_summaries,
                        forecast_scores=forecast_scores,
                    )

        phases = dict(session.phases())

    measure_rows = _score_measures(
        component_summaries=component_summaries,
        forecast_scores=forecast_scores,
        window_label=window_label,
    )
    arm_metrics = measure_to_arm_metrics(measure_rows)

    exp_arms = [
        ExperimentArm(
            arm=CONTROL_ARM,
            role="control",
            spec_hash=pipeline_hash(resolved_arms[0].pipeline, specs=specs),
            spec_label=resolved_arms[0].label,
        )
    ]
    for arm in resolved_arms[1:]:
        exp_arms.append(
            ExperimentArm(
                arm=arm.label,
                role="treatment",
                spec_hash=arm.pipeline_hash,
                spec_label=arm.label,
            )
        )
    manifest = submit_experiment(
        data_dir,
        flavour="comparison",
        question=(
            f"Component A/B on {component} over {window_label} "
            f"(base={_base_label(base_pipeline)})"
        ),
        arms=exp_arms,
        windows=[
            ExperimentWindow(
                role="primary",
                seasons=[parsed.season],
                from_gw=parsed.from_gw,
                to_gw=parsed.to_gw,
            )
        ],
        horizon=1,
        code_sha=sha,
        lockfile_hash=lock_hash,
        gates={},
        metrics=["brier_played", "brier_60", "mean_rps", "mean_log_score"],
        build_id=build_id,
        submitted_by=submitted_by,
        host=host_label,
        experiment_id=experiment_id,
    )
    write_experiment_results(
        data_dir,
        manifest.experiment_id,
        arm_metrics=arm_metrics if not arm_metrics.is_empty() else None,
    )
    append_experiment_event(
        data_dir,
        manifest.experiment_id,
        ExperimentEvent(event="analysed", at=datetime.now(UTC)),
    )
    logger.info(
        "component experiment %s arms=%s window=%s cache_hits=%s",
        manifest.experiment_id,
        [a.arm for a in exp_arms],
        window_label,
        cache.hits,
    )
    return ComponentExperimentResult(
        experiment_id=manifest.experiment_id,
        arm_metrics=arm_metrics,
        measure_metrics=measure_rows,
        cache=cache,
        phases=phases,
        window=parsed,
        arms=tuple(a.arm for a in exp_arms),
    )


def run_component_experiment_bench(
    data_dir: Path,
    *,
    base: str = "dc_layered_v4",
    component: str = "minutes",
    blend: float = 0.85,
    window: str = "2024-25:1-1",
    n_sims: int = 40,
) -> dict[str, float]:
    """Timed minutes A/B for the nightly bench (toolkit §2.4 ≤10 min/season).

    Args:
        data_dir: Local data root with staged builder / fixture data.
        base: Base ModelSpec name.
        component: Slot under test.
        blend: Treatment ``logistic_blend`` override.
        window: Screening window (default one GW).
        n_sims: Reduced sim count for the bench arm.

    Returns:
        ``{"exp_component": seconds}`` for ``tracked_phases``.
    """
    patch = {f"model.components.{component}.params.logistic_blend": blend}
    result = run_component_experiment(
        data_dir,
        base=base,
        component=component,
        arms=[patch],
        window=window,
        n_sims_override=n_sims,
        build_id="bench",
        submitted_by="bench",
    )
    seconds = float(result.phases.get("exp_component", 0.0))
    return {"exp_component": seconds}


@dataclass(frozen=True)
class _ArmModel:
    """Resolved arm: patched pipeline + layered model + label."""

    pipeline: PipelineSpec
    model: ModelSpec
    pipeline_hash: str
    label: str
    patch: dict[str, Any]


def _base_label(pipeline: PipelineSpec) -> str:
    return pipeline.name or pipeline.model_name


def _resolve_arms(
    base: PipelineSpec,
    *,
    component: str,
    arms: Sequence[str | Mapping[str, Any]],
    specs: dict[str, ModelSpec],
    n_sims_override: int | None,
) -> list[_ArmModel]:
    if not arms:
        raise ValueError("at least one treatment --arm is required")
    control_pipe = base
    control_model = _maybe_override_sims(
        model_for_pipeline(control_pipe), n_sims_override
    )
    if n_sims_override is not None:
        control_pipe = _pipeline_with_model(control_pipe, control_model)
    out: list[_ArmModel] = [
        _ArmModel(
            pipeline=control_pipe,
            model=control_model,
            pipeline_hash=pipeline_hash(control_pipe, specs=specs),
            label=CONTROL_ARM,
            patch={},
        )
    ]
    seen: set[str] = {CONTROL_ARM}
    for index, raw in enumerate(arms, start=1):
        patch = (
            dict(raw)
            if isinstance(raw, Mapping)
            else parse_arm_patch(str(raw), component=component)
        )
        resolved: ResolvedArm = apply_patch(
            base, patch, specs=specs, base_label=_base_label(base)
        )
        model = _maybe_override_sims(
            model_for_pipeline(resolved.pipeline), n_sims_override
        )
        pipe = (
            _pipeline_with_model(resolved.pipeline, model)
            if n_sims_override is not None
            else resolved.pipeline
        )
        label = resolved.label
        if label in seen or label == CONTROL_ARM:
            label = f"arm{index}"
        seen.add(label)
        out.append(
            _ArmModel(
                pipeline=pipe,
                model=model,
                pipeline_hash=pipeline_hash(pipe, specs=specs),
                label=label,
                patch=dict(resolved.patch),
            )
        )
    return out


def _maybe_override_sims(model: ModelSpec, n_sims: int | None) -> ModelSpec:
    if n_sims is None:
        return model
    return model.model_copy(update={"params": {**model.params, "n_sims": int(n_sims)}})


def _pipeline_with_model(pipeline: PipelineSpec, model: ModelSpec) -> PipelineSpec:
    return pipeline.model_copy(update={"model": model})


def _fit_layered_cached(
    data_dir: Path,
    *,
    spec: ModelSpec,
    view: PITView,
    as_of: GWKey,
    code_sha: str,
    lockfile_hash: str,
    build_id: str,
    cache: CacheStats,
    precomputed_prefer: Mapping[str, Any] | None = None,
    reusable: frozenset[str] | None = None,
) -> dict[str, Any]:
    """Fit or load every layered slot; reuse preferred fits for reusable slots."""
    hashes = component_hashes_for_spec(spec)
    fitted: dict[str, Any] = {}
    prefer = dict(precomputed_prefer or {})
    reuse: frozenset[str] = reusable if reusable is not None else frozenset()
    for slot in LAYERED_COMPONENT_SLOTS:
        if slot in reuse and slot in prefer:
            # Still go through the cache so a second experiment run hits disk.
            def _reuse(cached: Any = prefer[slot]) -> Any:
                return cached

            component, _key, hit = fit_or_load_component(
                data_dir,
                slot=slot,
                component_spec=spec.components[slot],
                component_hashes=hashes,
                code_sha=code_sha,
                lockfile_hash=lockfile_hash,
                build_id=build_id,
                deadline=as_of,
                fit=_reuse,
            )
            cache.record(slot, hit=hit)
            fitted[slot] = component
            continue

        def _fit(slot_name: str = slot) -> Any:
            model = get_component(spec.components[slot_name].kind)(
                spec.components[slot_name]
            )
            model.fit(view)
            return model

        component, _key, hit = fit_or_load_component(
            data_dir,
            slot=slot,
            component_spec=spec.components[slot],
            component_hashes=hashes,
            code_sha=code_sha,
            lockfile_hash=lockfile_hash,
            build_id=build_id,
            deadline=as_of,
            fit=_fit,
        )
        cache.record(slot, hit=hit)
        fitted[slot] = component
    return fitted


def _score_arm(
    *,
    arm_label: str,
    model: CompositePredictor,
    view: PITView,
    as_of: GWKey,
    store: DataStore,
    outcomes: pl.DataFrame,
    component: str,
    component_summaries: list[pl.DataFrame],
    forecast_scores: list[pl.DataFrame],
) -> None:
    """Append component-stage and prediction-stage rows for one arm × deadline."""
    if component == "minutes":
        detail = _minutes_detail_rows(model.minutes_model, view, store, as_of)
        if not detail.is_empty():
            component_summaries.append(_component_eval_from_detail(detail, arm_label))
    else:
        component_summaries.append(pl.DataFrame(schema=COMPONENT_EVAL_COLUMNS))

    preds = model.predict(view, [as_of])
    if preds.is_empty():
        return
    players = (
        view.latest("players", by=["season", "player_key"])
        .select("player_key", "team_key", "position")
        .unique(subset=["player_key"], keep="last")
        .collect()
    )
    deadline = _deadline_utc(store, as_of)
    n_sims = int(model.spec.params.get("n_sims", 40))

    def _sample(keys: Sequence[GWKey]) -> pl.DataFrame:
        return model.sample(view, list(keys), n_sims)

    try:
        pmf, _gw = points_pmf_tables_from_sample(
            _sample,
            horizon_keys=[as_of],
            players=players.select("player_key", "team_key"),
            run_id=f"exp_{arm_label}",
            as_of=as_of,
            deadline_utc=deadline,
        )
    except ValueError:
        return
    gw_outcomes = outcomes.filter(
        (pl.col("season") == as_of.season) & (pl.col("gw") == as_of.gw)
    )
    if gw_outcomes.is_empty() or pmf.is_empty():
        return
    scores = _forecast_score_rows(
        model_id=arm_label,
        as_of=as_of,
        points_pmf=pmf,
        predictions=preds,
        outcomes=gw_outcomes,
        players=players,
    )
    if not scores.is_empty():
        forecast_scores.append(scores)


def _score_measures(
    *,
    component_summaries: Sequence[pl.DataFrame],
    forecast_scores: Sequence[pl.DataFrame],
    window_label: str,
) -> pl.DataFrame:
    """Run Measure adapters and concatenate validated metrics."""
    frames: list[pl.DataFrame] = []
    non_empty_components = [
        frame for frame in component_summaries if not frame.is_empty()
    ]
    if non_empty_components:
        summary = pl.concat(non_empty_components, how="vertical_relaxed")
        summary = _aggregate_component_eval(summary)
        if not summary.is_empty():
            frames.append(
                get_measure("component_eval").score(
                    "components",
                    summary,
                    pl.DataFrame(),
                    window=window_label,
                )
            )
    if forecast_scores:
        scores = pl.concat(list(forecast_scores), how="vertical_relaxed")
        if not scores.is_empty():
            frames.append(
                get_measure("forecast_scores").score(
                    "prediction",
                    scores,
                    pl.DataFrame(),
                    window=window_label,
                    slices=["all"],
                )
            )
    if not frames:
        return empty_measure_metrics()
    return validate_measure_metrics(pl.concat(frames, how="vertical_relaxed"))


def _minutes_detail_rows(
    minutes_model: Any,
    view: PITView,
    store: DataStore,
    as_of: GWKey,
) -> pl.DataFrame:
    """Build a minutes-measurement detail frame for one fitted deadline."""
    asked = player_fixtures(view, [as_of])
    if asked.is_empty():
        return pl.DataFrame()
    dist: MinutesDist = minutes_model.predict(
        view, asked.select("player_key", "fixture_id")
    )
    probs = _minutes_event_probs(dist)
    actuals = (
        store.read("player_fixture_stats")
        .filter((pl.col("season") == as_of.season) & (pl.col("gw") == as_of.gw))
        .sort("available_at")
        .unique(subset=["player_key", "fixture_id"], keep="last")
        .select(
            "player_key",
            "fixture_id",
            "source",
            (pl.col("minutes") > 0).cast(pl.Float64).alias("played"),
            (pl.col("minutes") >= 60).cast(pl.Float64).alias("is_60"),
            (pl.col("starts") > 0).cast(pl.Float64).alias("started"),
        )
    )
    joined = probs.join(actuals, on=["player_key", "fixture_id"], how="inner")
    if joined.is_empty():
        return pl.DataFrame()
    position_frame = (
        view.latest("players", by=["season", "player_key"])
        .select("player_key", "position")
        .unique(subset=["player_key"], keep="last")
        .collect()
    )
    positions = {
        int(row["player_key"]): str(row["position"])
        for row in position_frame.iter_rows(named=True)
    }
    rows: list[dict[str, object]] = []
    for row in joined.iter_rows(named=True):
        player = int(row["player_key"])
        started = float(row["started"]) >= 1.0
        rows.append(
            {
                "season": as_of.season,
                "gw": as_of.gw,
                "player_key": player,
                "fixture_id": int(row["fixture_id"]),
                "position": positions.get(player, "UNK"),
                "source": str(row["source"]),
                "regime": "available_started" if started else "available_benched",
                "fpl_flag": "a",
                "returning_from_absence": False,
                "p_played": float(row["p_played"]),
                "p_60": float(row["p_60"]),
                "p_start": float(row["p_60"]),
                "played": float(row["played"]),
                "is_60": float(row["is_60"]),
                "started": float(started),
                "xpts_weight": float(row["p_60"]),
                "status": "a",
                "in_decision_pool": True,
            }
        )
    return pl.DataFrame(rows)


def _minutes_event_probs(dist: MinutesDist) -> pl.DataFrame:
    bins = np.asarray(dist.bins)
    pmf = np.asarray(dist.pmf, dtype=np.float64)
    zero = np.where(bins == 0)[0]
    p_zero = pmf[:, int(zero[0])] if len(zero) else np.zeros(pmf.shape[0])
    p_60 = (
        pmf[:, bins >= 60].sum(axis=1) if np.any(bins >= 60) else np.zeros(pmf.shape[0])
    )
    return dist.keys.with_columns(
        pl.Series("p_played", 1.0 - p_zero),
        pl.Series("p_60", p_60),
    )


def _component_eval_from_detail(detail: pl.DataFrame, arm: str) -> pl.DataFrame:
    """One-deadline component_eval rows (Brier + log) for Measure."""
    if detail.is_empty():
        return pl.DataFrame(schema=COMPONENT_EVAL_COLUMNS)
    rows: list[dict[str, object]] = []
    for slice_name, part in (
        ("overall", detail),
        ("source=fpl_api", detail.filter(pl.col("source") == "fpl_api")),
        ("decision_pool", detail.filter(pl.col("in_decision_pool"))),
    ):
        if part.is_empty() and slice_name != "overall":
            continue
        for metric, p_col, y_col in (
            ("brier_played", "p_played", "played"),
            ("brier_60", "p_60", "is_60"),
            ("log_played", "p_played", "played"),
            ("log_60", "p_60", "is_60"),
        ):
            if part.is_empty():
                value: float | None = None
                n = 0
            elif metric.startswith("brier"):
                value = brier_score(part[p_col].to_numpy(), part[y_col].to_numpy())
                n = part.height
            else:
                value = log_score(part[p_col].to_numpy(), part[y_col].to_numpy())
                n = part.height
            rows.append(
                {
                    "component": "minutes",
                    "metric": metric,
                    "model_name": arm,
                    "slice": slice_name,
                    "value": value,
                    "n": n,
                }
            )
    return pl.DataFrame(rows, schema=COMPONENT_EVAL_COLUMNS)


def _aggregate_component_eval(summary: pl.DataFrame) -> pl.DataFrame:
    """Mean component_eval values across deadlines (weighted by ``n``)."""
    if summary.is_empty():
        return summary
    rows: list[dict[str, object]] = []
    for keys, group in summary.group_by(["component", "metric", "model_name", "slice"]):
        component, metric, model_name, slice_name = keys  # type: ignore[misc]
        n = int(group["n"].sum())
        if n <= 0 or group["value"].null_count() == group.height:
            value = None
        else:
            weights = group["n"].to_numpy().astype(np.float64)
            vals = group["value"].to_numpy().astype(np.float64)
            mask = np.isfinite(vals) & (weights > 0)
            value = (
                float(np.average(vals[mask], weights=weights[mask]))
                if mask.any()
                else None
            )
        rows.append(
            {
                "component": str(component),
                "metric": str(metric),
                "model_name": str(model_name),
                "slice": str(slice_name),
                "value": value,
                "n": n,
            }
        )
    return pl.DataFrame(rows, schema=COMPONENT_EVAL_COLUMNS)


def _forecast_score_rows(
    *,
    model_id: str,
    as_of: GWKey,
    points_pmf: pl.DataFrame,
    predictions: pl.DataFrame,
    outcomes: pl.DataFrame,
    players: pl.DataFrame,
) -> pl.DataFrame:
    """Build a ``ForecastScoresSchema`` frame for the Measure adapter."""
    if "price" not in players.columns:
        meta = players.select("player_key", "position").with_columns(
            pl.lit(60).cast(pl.Int64).alias("price")
        )
    else:
        meta = players.select("player_key", "position", "price")
    pred = predictions.with_columns(pl.lit(model_id).alias("model_id"))
    return compute_forecast_scores(
        model_id=model_id,
        key=as_of,
        points_pmf=points_pmf,
        predictions=pred,
        outcomes=outcomes,
        meta=meta,
    )


def _deadline_utc(store: DataStore, as_of: GWKey) -> datetime:
    gws = (
        store.read("gameweeks")
        .filter((pl.col("season") == as_of.season) & (pl.col("gw") == as_of.gw))
        .sort("available_at")
    )
    if gws.is_empty():
        return datetime(2024, 8, 16, 17, 0, tzinfo=UTC)
    value = gws["deadline_time"][0]
    if isinstance(value, datetime):
        return value if value.tzinfo is not None else value.replace(tzinfo=UTC)
    return datetime(2024, 8, 16, 17, 0, tzinfo=UTC)


__all__ = [
    "CONTROL_ARM",
    "CacheStats",
    "ComponentExperimentResult",
    "ParsedWindow",
    "load_base_pipeline",
    "measure_to_arm_metrics",
    "model_for_pipeline",
    "parse_arm_patch",
    "parse_component_window",
    "run_component_experiment",
    "run_component_experiment_bench",
]
