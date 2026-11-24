"""PipelineSpec: data / model / planner / policy / env, hashed like ModelSpec.

Experiments and official lines identify an arm by the full pipeline, not the
model alone (``docs/proposals/fplt-experiments.md`` §3 / §9 step 1). Planner
and policy knobs that previously lived only in CLI flags and code defaults
are typed here so they can be hashed and patched.

``pipeline_hash`` is sha256 of the fully resolved payload: section defaults
filled, ``name`` excluded, and ``model`` expanded to the nameless resolved
ModelSpec payload (typed component defaults included). ModelSpec
``spec_hash`` remains for component / model caching.
"""

from __future__ import annotations

import hashlib
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

from fplt.models.spec import ModelSpec, load_model_specs
from fplt.models.spec_identity import canonical_spec_json, resolved_spec_payload
from fplt.optimise.candidate_prune import DEFAULT_TARGET_POOL
from fplt.optimise.decisions import (
    DEFAULT_LIVE_CHIP_ARM,
    DEFAULT_LIVE_PRICE_FORECAST_METHOD,
)
from fplt.optimise.highs_options import DEFAULT_HIGHS_OPTIONS
from fplt.optimise.plan import DEFAULT_PLAN_HORIZON

CaptureCheckpoint = Literal["T-90m"]
PlannerObjective = Literal["mean"]
ChipScheduler = Literal["season_v1"]
ChipArmName = Literal["off", "rule_based", "optimised"]
CaptainPolicy = Literal["max_xpts"]
PriceForecastName = Literal["flat", "mean_gw_delta"]
McBackend = Literal["python", "native"]


class DataSources(BaseModel):
    """Which optional data feeds the pipeline may consume."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    odds_capture: bool = False
    fplcache: bool = False
    api_football_injuries: bool = True


class DataSpec(BaseModel):
    """Data stage of a pipeline: sources and official capture checkpoint."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    sources: DataSources = Field(default_factory=DataSources)
    capture_checkpoint: CaptureCheckpoint = "T-90m"


class PlannerPool(BaseModel):
    """Candidate-pool size for the transfer MILP."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    target: int = Field(default=DEFAULT_TARGET_POOL, ge=15)


class PlannerSpec(BaseModel):
    """Planner stage: horizon, decay, pool, and HiGHS solve knobs."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    horizon: int = Field(default=DEFAULT_PLAN_HORIZON, ge=1)
    decay: float = Field(default=0.9, gt=0.0, le=1.0)
    pool: PlannerPool = Field(default_factory=PlannerPool)
    mip_gap: float = Field(default=DEFAULT_HIGHS_OPTIONS.mip_rel_gap, ge=0.0)
    time_limit_s: float = Field(default=600.0, gt=0.0)
    objective: PlannerObjective = "mean"


class PolicyChips(BaseModel):
    """Chip scheduler identity and live arm / force-play flags."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    scheduler: ChipScheduler = "season_v1"
    arm: ChipArmName = DEFAULT_LIVE_CHIP_ARM
    force_play: bool = True


class PolicySpec(BaseModel):
    """Decision-policy stage: hits, chips, captain, lineup, and prices."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    max_hits: int = Field(default=1, ge=0)
    hit_min_gain: float = Field(default=4.0, ge=0.0)
    chips: PolicyChips = Field(default_factory=PolicyChips)
    captain: CaptainPolicy = "max_xpts"
    lineup_reopt: bool = True
    price_forecast: PriceForecastName = DEFAULT_LIVE_PRICE_FORECAST_METHOD
    bench_weight: float = Field(default=0.0, ge=0.0)


class EnvSpec(BaseModel):
    """Runtime environment that can change results (e.g. MC backend)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    mc_backend: McBackend = "native"


class PipelineSpec(BaseModel):
    """Fully typed pipeline: data, model, planner, policy, and env.

    ``model`` is a registered ModelSpec name. Inline ModelSpec objects are
    accepted for hashing / tests; YAML production lines use names.
    ``name`` is a human label excluded from ``pipeline_hash``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str | None = None
    data: DataSpec = Field(default_factory=DataSpec)
    model: str | ModelSpec
    planner: PlannerSpec = Field(default_factory=PlannerSpec)
    policy: PolicySpec = Field(default_factory=PolicySpec)
    env: EnvSpec = Field(default_factory=EnvSpec)

    @field_validator("model", mode="before")
    @classmethod
    def _model_not_blank(cls, value: object) -> object:
        if isinstance(value, str):
            cleaned = value.strip()
            if not cleaned:
                raise ValueError("model must be a non-empty ModelSpec name")
            return cleaned
        return value

    @property
    def model_name(self) -> str:
        """Registered ModelSpec name, or the inline spec's ``name``."""
        if isinstance(self.model, str):
            return self.model
        return self.model.name


def live_pipeline_defaults(
    *,
    model: str,
    horizon: int | None = None,
    mc_backend: McBackend | None = None,
    name: str | None = None,
) -> PipelineSpec:
    """Build a PipelineSpec that matches today's live ops behaviour.

    Section defaults mirror ``run_plan`` / automation / ``squad_state.yaml`` /
    ``DEFAULT_HIGHS_OPTIONS`` / ``DEFAULT_TARGET_POOL``. Callers that only
    know the production model name and horizon get a fully typed pipeline.

    Args:
        model: Registered ModelSpec name (e.g. ``dc_layered_v4_native``).
        horizon: Planner horizon. Defaults to ``DEFAULT_PLAN_HORIZON``.
        mc_backend: Optional env override. Defaults to ``native`` for the
            live production stack; pass ``python`` for the parent model.
        name: Optional human label (excluded from the hash).

    Returns:
        A ``PipelineSpec`` with live defaults filled in.
    """
    backend: McBackend = "native" if mc_backend is None else mc_backend
    steps = DEFAULT_PLAN_HORIZON if horizon is None else horizon
    return PipelineSpec(
        name=name,
        data=DataSpec(),
        model=model,
        planner=PlannerSpec(horizon=steps),
        policy=PolicySpec(),
        env=EnvSpec(mc_backend=backend),
    )


def resolve_pipeline_model(
    pipeline: PipelineSpec,
    *,
    specs: dict[str, ModelSpec] | None = None,
) -> ModelSpec:
    """Resolve ``pipeline.model`` to a loaded ``ModelSpec``.

    Args:
        pipeline: Pipeline whose model field is a name or inline spec.
        specs: Optional name → ModelSpec map. Defaults to ``load_model_specs()``.

    Returns:
        The resolved ``ModelSpec``.

    Raises:
        KeyError: Named model is not in the registry.
    """
    if isinstance(pipeline.model, ModelSpec):
        return pipeline.model
    loaded = specs if specs is not None else load_model_specs()
    try:
        return loaded[pipeline.model]
    except KeyError:
        raise KeyError(f"no ModelSpec registered for {pipeline.model!r}") from None


def resolved_pipeline_payload(
    pipeline: PipelineSpec,
    *,
    specs: dict[str, ModelSpec] | None = None,
) -> dict[str, Any]:
    """Return the nameless, defaults-filled payload used for ``pipeline_hash``.

    The ``model`` field is replaced by the resolved ModelSpec payload (same
    shape as ``resolved_spec_payload``), so two names that resolve to one
    model hash to the same pipeline when other sections match.

    Args:
        pipeline: Spec to resolve.
        specs: Optional ModelSpec registry for name lookup.

    Returns:
        A JSON-serialisable mapping suitable for hashing.
    """
    model = resolve_pipeline_model(pipeline, specs=specs)
    dump = pipeline.model_dump(mode="json")
    dump.pop("name", None)
    dump["model"] = resolved_spec_payload(model)
    return dump


def pipeline_hash(
    pipeline: PipelineSpec,
    *,
    specs: dict[str, ModelSpec] | None = None,
) -> str:
    """Sha256 of the fully resolved, name-excluded PipelineSpec payload.

    Args:
        pipeline: Pipeline to identify.
        specs: Optional ModelSpec registry for name lookup.

    Returns:
        64-character lowercase hex digest.
    """
    raw = canonical_spec_json(resolved_pipeline_payload(pipeline, specs=specs))
    return hashlib.sha256(raw.encode()).hexdigest()
