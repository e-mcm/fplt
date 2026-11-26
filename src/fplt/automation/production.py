"""Load the live production model list, plan horizon, and PipelineSpec.

Committed defaults live in ``configs/ops/production.yaml``. Operators can
override without editing Python via ``FPLT_PRODUCTION_MODELS`` (comma-
separated spec names) and ``FPLT_PLAN_HORIZON``. Env wins over YAML.

Stage B11 expresses production as a typed ``PipelineSpec`` (data / model /
planner / policy / env). ``models`` and ``plan_horizon`` stay as the short
form automation reads; when ``pipeline:`` is present they must match.

Stage B12 adds ``lines:`` — named official lines (primary + candidates) that
share one predict per distinct spec and one plan per line
(``fplt-data-model.md`` §4.1 / step 6).
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from fplt.models.pipeline_spec import PipelineSpec, live_pipeline_defaults
from fplt.schemas.manifests import LineRole, SquadMode

_DEFAULT_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "ops" / "production.yaml"
)
_MODELS_ENV = "FPLT_PRODUCTION_MODELS"
_HORIZON_ENV = "FPLT_PLAN_HORIZON"


class OfficialLine(BaseModel):
    """One official production line: model spec, plan horizon, and role.

    ``squad_mode`` defaults to ``real`` for the primary line and ``virtual``
    for candidates (data model §4.1). Changing a line's spec or horizon means
    a new ``line`` name.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    line: str = Field(min_length=1)
    spec: str = Field(min_length=1)
    horizon: int = Field(ge=1)
    role: LineRole
    squad_mode: SquadMode = "virtual"

    @field_validator("line", "spec")
    @classmethod
    def _strip_non_empty(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("line and spec must be non-empty")
        return cleaned

    @model_validator(mode="before")
    @classmethod
    def _default_squad_mode(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        raw = cast(dict[str, Any], data)
        role = raw.get("role")
        mode = raw.get("squad_mode")
        if mode is None:
            raw = {**raw, "squad_mode": "real" if role == "primary" else "virtual"}
        elif role == "primary" and mode != "real":
            raise ValueError("primary line must use squad_mode=real")
        return raw


class ProductionOps(BaseModel):
    """Live models, planner horizon, PipelineSpec, and official lines."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    models: list[str] = Field(min_length=1)
    plan_horizon: int = Field(ge=1)
    pipeline: PipelineSpec
    lines: list[OfficialLine] = Field(min_length=1)

    @field_validator("models")
    @classmethod
    def _non_empty_names(cls, value: list[str]) -> list[str]:
        cleaned = [name.strip() for name in value]
        if not cleaned or any(not name for name in cleaned):
            raise ValueError("models must be a non-empty list of non-empty names")
        return cleaned

    @model_validator(mode="after")
    def _pipeline_and_lines(self) -> ProductionOps:
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
        names = [entry.line for entry in self.lines]
        if len(names) != len(set(names)):
            raise ValueError(f"line names must be unique, got {names}")
        primaries = [entry for entry in self.lines if entry.role == "primary"]
        if len(primaries) != 1:
            raise ValueError(
                f"lines must contain exactly one primary, got {len(primaries)}"
            )
        primary = primaries[0]
        if primary.spec != self.models[0]:
            raise ValueError(
                f"primary line spec {primary.spec!r} must equal "
                f"models[0] {self.models[0]!r}"
            )
        if primary.horizon != self.plan_horizon:
            raise ValueError(
                f"primary line horizon {primary.horizon} must equal "
                f"plan_horizon {self.plan_horizon}"
            )
        return self

    def primary_line(self) -> OfficialLine:
        """Return the single primary official line."""
        for entry in self.lines:
            if entry.role == "primary":
                return entry
        raise RuntimeError("ProductionOps invariant broken: no primary line")

    def predict_groups(self) -> dict[str, int]:
        """Map each distinct ModelSpec name to the longest horizon its lines need.

        Official ops run one predict per key at the returned horizon, then one
        plan per line (data model §4.1).
        """
        groups: dict[str, int] = {}
        for entry in self.lines:
            current = groups.get(entry.spec, 0)
            if entry.horizon > current:
                groups[entry.spec] = entry.horizon
        return groups


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
        ``pipeline``, ``lines``).

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
    # Emergency env override collapses to a single primary line so we never
    # silently multi-arm live (AGENTS / design §8.5).
    new_models = models if models is not None else list(ops.models)
    new_horizon = horizon if horizon is not None else ops.plan_horizon
    pipeline = ops.pipeline.model_copy(
        update={
            "model": new_models[0],
            "planner": ops.pipeline.planner.model_copy(update={"horizon": new_horizon}),
        }
    )
    lines = [
        OfficialLine(
            line="env_override",
            spec=new_models[0],
            horizon=new_horizon,
            role="primary",
            squad_mode="real",
        )
    ]
    return ProductionOps(
        models=new_models,
        plan_horizon=new_horizon,
        pipeline=pipeline,
        lines=lines,
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

    lines = _lines_from_mapping(raw, models=models, horizon=horizon)
    return ProductionOps(
        models=models, plan_horizon=horizon, pipeline=pipeline, lines=lines
    )


def _lines_from_mapping(
    raw: dict[str, Any], *, models: list[str], horizon: int
) -> list[OfficialLine]:
    lines_obj = raw.get("lines")
    if lines_obj is None:
        return [
            OfficialLine(
                line="primary",
                spec=models[0],
                horizon=horizon,
                role="primary",
                squad_mode="real",
            )
        ]
    if not isinstance(lines_obj, list):
        raise ValueError("lines must be a list when set")
    if not lines_obj:
        raise ValueError("lines must be a non-empty list when set")
    parsed: list[OfficialLine] = []
    for item in cast(list[object], lines_obj):
        if not isinstance(item, dict):
            raise ValueError("each lines entry must be a mapping")
        parsed.append(OfficialLine.model_validate(cast(dict[str, Any], item)))
    return parsed


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
