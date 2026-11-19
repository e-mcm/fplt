"""Model identity: a YAML-loadable spec and a class registry."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import version
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class TrainWindow(BaseModel):
    """How much history a fit is allowed to see."""

    model_config = ConfigDict(frozen=True)

    seasons_back: int = 4
    time_decay: float | None = None


class ModelSpec(BaseModel):
    """Everything that distinguishes one fitted model from another."""

    model_config = ConfigDict(frozen=True)

    name: str
    kind: str
    params: dict[str, Any] = Field(default_factory=dict)
    components: dict[str, ModelSpec] = Field(default_factory=dict)
    features: list[str] = Field(default_factory=list)
    train_window: TrainWindow = Field(default_factory=TrainWindow)
    refit: Literal["every_gw", "every_n_gw", "never"] = "every_gw"
    refit_every: int = 1
    seed: int = 0

    @property
    def model_id(self) -> str:
        """Short hash of this spec plus the installed package version.

        Changing a parameter, component, feature version, window, or package
        version changes the id. Artifacts are addressed by this id.
        """
        payload = {
            "spec": self.model_dump(mode="json"),
            "version": _package_version(),
        }
        raw = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(raw.encode()).hexdigest()[:16]


_REGISTRY: dict[str, type[Any]] = {}


def register_model(kind: str) -> Any:
    """Register a predictor class under ``kind`` for YAML specs."""

    def decorate(cls: type[Any]) -> type[Any]:
        if kind in _REGISTRY and _REGISTRY[kind] is not cls:
            raise ValueError(f"model kind {kind!r} is already registered")
        _REGISTRY[kind] = cls
        return cls

    return decorate


def get_model(kind: str) -> type[Any]:
    """Return the class registered for ``kind``."""
    try:
        return _REGISTRY[kind]
    except KeyError:
        raise KeyError(f"no model registered for kind {kind!r}") from None


def registered_kinds() -> list[str]:
    """Sorted registry keys. The contract suite parametrises over these."""
    return sorted(_REGISTRY)


def _package_version() -> str:
    try:
        return version("fplt")
    except Exception:
        return "0.0.0"
