"""Load the live production model list, plan horizon, and PipelineSpec.

Committed defaults live in ``configs/ops/production.yaml``. Operators can
override without editing Python via ``FPLT_PRODUCTION_MODELS`` (comma-
separated spec names) and ``FPLT_PLAN_HORIZON``. Env wins over YAML.

Stage B11 expresses production as a typed ``PipelineSpec`` (data / model /
planner / policy / env). ``models`` and ``plan_horizon`` stay as the short
form automation reads; when ``pipeline:`` is present they must match.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fplt.models.pipeline_spec import PipelineSpec, live_pipeline_defaults

_DEFAULT_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "ops" / "production.yaml"
)
_MODELS_ENV = "FPLT_PRODUCTION_MODELS"
_HORIZON_ENV = "FPLT_PLAN_HORIZON"


class ProductionOps(BaseModel):
    """Live models, planner horizon, and the primary PipelineSpec."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    models: list[str] = Field(min_length=1)
    plan_horizon: int = Field(ge=1)
    pipeline: PipelineSpec

    @field_validator("models")
    @classmethod
    def _non_empty_names(cls, value: list[str]) -> list[str]:
        cleaned = [name.strip() for name in value]
        if not cleaned or any(not name for name in cleaned):
            raise ValueError("models must be a non-empty list of non-empty names")
        return cleaned

    @model_validator(mode="after")
    def _pipeline_matches_short_form(self) -> ProductionOps:
        if self.pipeline.model_name != self.models[0]:
            raise ValueError(
                f"pipeline.model {self.pipeline.model_name!r} must equal "
                f"models[0] {self.models[0]!r}"
            )
        if self.pipeline.planner.horizon != self.plan_horizon:
            raise ValueError(
                f"pipeline.planner.horizon {self.pipeline.planner.horizon} "
                f"must equal plan_horizon {self.plan_horizon}"
            )
        return self


def default_production_path() -> Path:
    """Return the repo-root path to ``configs/ops/production.yaml``."""
    return _DEFAULT_PATH


def load_production_ops(
    path: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> ProductionOps:
    """Load production ops from YAML, then apply env overrides.

    Args:
        path: Config file. Defaults to ``configs/ops/production.yaml``.
        environ: Environment mapping. Defaults to ``os.environ``.

    Returns:
        Validated ``ProductionOps`` (``models``, ``plan_horizon``,
        ``pipeline``).

    Raises:
        FileNotFoundError: The YAML path does not exist.
        ValueError: The file is empty, not a mapping, or fails validation /
            env parsing.
    """
    source = default_production_path() if path is None else path
    if not source.is_file():
        raise FileNotFoundError(f"production ops config not found: {source}")
    loaded = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{source.name} must contain a mapping")
    raw = cast(dict[str, Any], loaded)
    ops = _ops_from_mapping(raw)

    env = os.environ if environ is None else environ
    models = _models_from_env(env.get(_MODELS_ENV))
    horizon = _horizon_from_env(env.get(_HORIZON_ENV))
    if models is None and horizon is None:
        return ops
    new_models = models if models is not None else list(ops.models)
    new_horizon = horizon if horizon is not None else ops.plan_horizon
    pipeline = ops.pipeline.model_copy(
        update={
            "model": new_models[0],
            "planner": ops.pipeline.planner.model_copy(update={"horizon": new_horizon}),
        }
    )
    return ProductionOps(
        models=new_models,
        plan_horizon=new_horizon,
        pipeline=pipeline,
    )


def load_production_pipeline(
    path: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> PipelineSpec:
    """Return the primary production ``PipelineSpec`` (after env overrides).

    Args:
        path: Config file. Defaults to ``configs/ops/production.yaml``.
        environ: Environment mapping. Defaults to ``os.environ``.

    Returns:
        The validated primary pipeline.
    """
    return load_production_ops(path, environ=environ).pipeline


def _ops_from_mapping(raw: dict[str, Any]) -> ProductionOps:
    models_obj = raw.get("models")
    horizon_obj = raw.get("plan_horizon")
    if not isinstance(models_obj, list):
        raise ValueError("production ops must set models: [name, ...]")
    models = [str(item) for item in cast(list[object], models_obj)]
    if not models:
        raise ValueError("models must be a non-empty list of non-empty names")
    if horizon_obj is None:
        raise ValueError("production ops must set plan_horizon")
    horizon = int(horizon_obj)

    pipeline_obj = raw.get("pipeline")
    if pipeline_obj is None:
        pipeline = live_pipeline_defaults(model=models[0], horizon=horizon)
    elif isinstance(pipeline_obj, dict):
        pipeline = PipelineSpec.model_validate(cast(dict[str, Any], pipeline_obj))
    else:
        raise ValueError("pipeline must be a mapping when set")

    return ProductionOps(models=models, plan_horizon=horizon, pipeline=pipeline)


def _models_from_env(raw: str | None) -> list[str] | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        raise ValueError(f"{_MODELS_ENV} is set but empty")
    names = [part.strip() for part in text.split(",")]
    if not names or any(not name for name in names):
        raise ValueError(
            f"{_MODELS_ENV} must be a comma-separated list of non-empty names"
        )
    return names


def _horizon_from_env(raw: str | None) -> int | None:
    if raw is None:
        return None
    text = raw.strip()
    if not text:
        raise ValueError(f"{_HORIZON_ENV} is set but empty")
    try:
        value = int(text)
    except ValueError as exc:
        raise ValueError(f"{_HORIZON_ENV} must be an integer, got {raw!r}") from exc
    if value < 1:
        raise ValueError(f"{_HORIZON_ENV} must be >= 1, got {value}")
    return value
