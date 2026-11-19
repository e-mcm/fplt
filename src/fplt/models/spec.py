"""Model identity: a YAML-loadable spec and a class registry."""

from __future__ import annotations

import hashlib
import json
import subprocess
from importlib.metadata import version
from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field

from fplt.types import GWKey


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


_COMPONENTS: dict[str, type[Any]] = {}


def register_component(kind: str) -> Any:
    """Register a component class under ``kind``.

    Components are not points predictors, so they stay out of the contract
    suite that walks ``registered_kinds``.
    """

    def decorate(cls: type[Any]) -> type[Any]:
        if kind in _COMPONENTS and _COMPONENTS[kind] is not cls:
            raise ValueError(f"component kind {kind!r} is already registered")
        _COMPONENTS[kind] = cls
        return cls

    return decorate


def get_component(kind: str) -> type[Any]:
    """Return the component class registered for ``kind``."""
    try:
        return _COMPONENTS[kind]
    except KeyError:
        raise KeyError(f"no component registered for kind {kind!r}") from None


def registered_components() -> list[str]:
    """Sorted component registry keys."""
    return sorted(_COMPONENTS)


def load_model_specs(directory: Path | None = None) -> dict[str, ModelSpec]:
    """Load every YAML model spec from ``directory``.

    Args:
        directory: Folder of ``*.yaml`` files. Defaults to ``configs/models``.

    Returns:
        Specs keyed by ``name``, in file order.

    Raises:
        FileNotFoundError: ``directory`` does not exist.
        ValueError: A file is empty, not a mapping, or repeats a name.
    """
    folder = directory if directory is not None else _default_specs_dir()
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    loaded: dict[str, ModelSpec] = {}
    for path in sorted(folder.glob("*.yaml")):
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(raw, dict):
            raise ValueError(f"{path.name} must contain a mapping")
        spec = ModelSpec.model_validate(raw)
        if spec.name in loaded:
            raise ValueError(f"duplicate model spec name {spec.name!r}")
        loaded[spec.name] = spec
    return loaded


def artifact_dir(data_dir: Path, spec: ModelSpec, as_of: GWKey) -> Path:
    """Directory for one fitted spec: ``artifacts/<model_id>/as_of=...``."""
    stamp = f"as_of={as_of.season}_gw{as_of.gw:02d}"
    return data_dir / "artifacts" / spec.model_id / stamp


def write_identity(dest: Path, spec: ModelSpec) -> None:
    """Write ``spec.json`` and ``git_sha.txt`` into ``dest``.

    Args:
        dest: Artifact directory for one spec and gameweek.
        spec: The spec being recorded.

    Raises:
        FileExistsError: Either identity file is already present.
    """
    dest.mkdir(parents=True, exist_ok=True)
    spec_path = dest / "spec.json"
    sha_path = dest / "git_sha.txt"
    if spec_path.exists() or sha_path.exists():
        raise FileExistsError(f"artifact already exists at {dest}")
    payload = json.dumps(spec.model_dump(mode="json"), sort_keys=True, indent=2)
    spec_path.write_text(payload + "\n", encoding="utf-8")
    sha_path.write_text(git_sha() + "\n", encoding="utf-8")


def save_spec(data_dir: Path, spec: ModelSpec, as_of: GWKey) -> Path:
    """Write ``spec.json`` and ``git_sha.txt`` for this spec and gameweek.

    Args:
        data_dir: Root that contains ``artifacts/``. Usually ``$FPLT_DATA_DIR``.
        spec: The spec being recorded.
        as_of: Gameweek the artifact belongs to.

    Returns:
        The directory that was written.

    Raises:
        FileExistsError: Either identity file is already present.
    """
    dest = artifact_dir(data_dir, spec, as_of)
    write_identity(dest, spec)
    return dest


def git_sha() -> str:
    """Current ``HEAD`` commit, or ``unknown`` when git cannot answer."""
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return "unknown"
    return completed.stdout.strip() or "unknown"


def _default_specs_dir() -> Path:
    return Path(__file__).resolve().parents[3] / "configs" / "models"


def _package_version() -> str:
    try:
        return version("fplt")
    except Exception:
        return "0.0.0"
