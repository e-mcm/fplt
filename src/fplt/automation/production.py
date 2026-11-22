"""Load the live production model list and plan horizon for automation.

Committed defaults live in ``configs/ops/production.yaml``. Operators can
override without editing Python via ``FPLT_PRODUCTION_MODELS`` (comma-
separated spec names) and ``FPLT_PLAN_HORIZON``. Env wins over YAML.
"""

from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

_DEFAULT_PATH = (
    Path(__file__).resolve().parents[3] / "configs" / "ops" / "production.yaml"
)
_MODELS_ENV = "FPLT_PRODUCTION_MODELS"
_HORIZON_ENV = "FPLT_PLAN_HORIZON"


class ProductionOps(BaseModel):
    """Live models and planner horizon used by catch-up / pre-deadline."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    models: list[str] = Field(min_length=1)
    plan_horizon: int = Field(ge=1)

    @field_validator("models")
    @classmethod
    def _non_empty_names(cls, value: list[str]) -> list[str]:
        cleaned = [name.strip() for name in value]
        if not cleaned or any(not name for name in cleaned):
            raise ValueError("models must be a non-empty list of non-empty names")
        return cleaned


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
        Validated ``ProductionOps`` (``models``, ``plan_horizon``).

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
    ops = ProductionOps.model_validate(loaded)

    env = os.environ if environ is None else environ
    models = _models_from_env(env.get(_MODELS_ENV))
    horizon = _horizon_from_env(env.get(_HORIZON_ENV))
    if models is None and horizon is None:
        return ops
    return ProductionOps(
        models=models if models is not None else list(ops.models),
        plan_horizon=horizon if horizon is not None else ops.plan_horizon,
    )


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
