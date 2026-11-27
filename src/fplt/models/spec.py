"""Model identity: a YAML-loadable spec and a class registry."""

from __future__ import annotations

import copy
import hashlib
import json
import os
import subprocess
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any, Literal, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field

from fplt.types import GWKey

_EXTENDS_KEYS = frozenset({"name", "extends", "overrides"})


class TrainWindow(BaseModel):
    """How much history a fit is allowed to see.

    ``seasons_back`` is enforced by walk-forward / PIT season filtering.
    ``time_decay`` is an optional calendar rate λ (inverse days): trainers that
    honour it weight rows by ``exp(-λ × days_before_as_of)``. Default ``None``
    keeps equal weights (backwards compatible). Dixon–Coles uses its own
    ``xi`` instead — do not also set ``time_decay`` on a DC score spec unless
    you intentionally want a second decay path (currently ignored by DC).
    """

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

    Files may set ``extends: <parent_name>`` with an ``overrides`` mapping.
    The loader deep-merges the override dict onto a copy of the parent raw
    YAML (one parent only; no chains), then validates with ``ModelSpec``.
    ``extends`` / ``overrides`` are loader keys, not ``ModelSpec`` fields.

    Args:
        directory: Folder of ``*.yaml`` files. Defaults to ``configs/models``.

    Returns:
        Specs keyed by ``name``, in file order.

    Raises:
        FileNotFoundError: ``directory`` does not exist.
        ValueError: A file is empty, not a mapping, repeats a name, has a
            missing ``extends`` target, extends a fork (chain deeper than
            one), or mixes ``extends`` with unexpected top-level keys.
    """
    folder = directory if directory is not None else _default_specs_dir()
    if not folder.is_dir():
        raise FileNotFoundError(folder)
    raw_by_path: list[tuple[Path, dict[str, Any]]] = []
    raw_by_name: dict[str, dict[str, Any]] = {}
    for path in sorted(folder.glob("*.yaml")):
        # Identity lock lives beside ModelSpec YAMLs; it is not a spec.
        if path.name == "registry.lock.yaml" or path.name.endswith(".lock.yaml"):
            continue
        loaded_raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        if not isinstance(loaded_raw, dict):
            raise ValueError(f"{path.name} must contain a mapping")
        raw = cast(dict[str, Any], loaded_raw)
        name = raw.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError(f"{path.name} must set a non-empty string name")
        if name in raw_by_name:
            raise ValueError(f"duplicate model spec name {name!r}")
        raw_by_name[name] = raw
        raw_by_path.append((path, raw))

    loaded: dict[str, ModelSpec] = {}
    for path, raw in raw_by_path:
        resolved = _resolve_extends(raw, raw_by_name, path.name)
        spec = ModelSpec.model_validate(resolved)
        loaded[spec.name] = spec
    return _expand_forecast_combine_members(loaded)


def _expand_forecast_combine_members(
    loaded: dict[str, ModelSpec],
) -> dict[str, ModelSpec]:
    """Fold ``params.members`` name lists into ``components`` for blends.

    Ensemble YAMLs list member ModelSpec names under ``params.members``. After
    every file is validated, those names are resolved into ``components`` so the
    ensemble ``model_id`` includes the full member specs. Explicit ``components``
    on a ``forecast_combine`` spec are left alone when ``members`` is absent.

    Args:
        loaded: Specs keyed by name.

    Returns:
        The same mapping with forecast_combine arms expanded.

    Raises:
        ValueError: Unknown member, self-reference, nested forecast_combine, or
            empty ``members``.
    """
    expanded: dict[str, ModelSpec] = {}
    for name, spec in loaded.items():
        if spec.kind != "forecast_combine":
            expanded[name] = spec
            continue
        members_raw = spec.params.get("members")
        if members_raw is None:
            if not spec.components:
                raise ValueError(
                    f"{name}: forecast_combine needs params.members or components"
                )
            expanded[name] = spec
            continue
        if not isinstance(members_raw, list) or not members_raw:
            raise ValueError(f"{name}: params.members must be a non-empty list")
        components: dict[str, ModelSpec] = {}
        for member_obj in cast(list[object], members_raw):
            if not isinstance(member_obj, str) or not member_obj:
                raise ValueError(f"{name}: members entries must be non-empty strings")
            member_name = member_obj
            if member_name == name:
                raise ValueError(f"{name}: cannot list itself in params.members")
            try:
                member = loaded[member_name]
            except KeyError:
                raise ValueError(
                    f"{name}: params.members names unknown ModelSpec {member_name!r}"
                ) from None
            if member.kind == "forecast_combine":
                raise ValueError(
                    f"{name}: member {member_name!r} cannot be forecast_combine"
                )
            components[member_name] = member
        # Drop members from params so identity is carried by components only.
        params = {key: value for key, value in spec.params.items() if key != "members"}
        expanded[name] = spec.model_copy(
            update={"components": components, "params": params}
        )
    return expanded


def deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    """Deep-merge ``overrides`` onto a copy of ``base``.

    Nested dicts merge recursively. Non-dict values (including lists) in
    ``overrides`` replace the base value. A nested mapping that sets a
    different ``kind`` than the base replaces that node wholesale so component
    params do not leak across kinds. Used by the ModelSpec ``extends`` loader;
    not OmegaConf.

    Args:
        base: Parent mapping (not mutated).
        overrides: Values to layer on top.

    Returns:
        A new merged dict.
    """
    return _deep_merge(copy.deepcopy(base), overrides)


def _deep_merge(base: dict[str, Any], overrides: dict[str, Any]) -> dict[str, Any]:
    for key, value in overrides.items():
        existing = base.get(key)
        if isinstance(existing, dict) and isinstance(value, dict):
            existing_map = cast(dict[str, Any], existing)
            value_map = cast(dict[str, Any], value)
            if "kind" in value_map and existing_map.get("kind") != value_map.get(
                "kind"
            ):
                # Replacing a ModelSpec-shaped node (different ``kind``) must not
                # deep-merge params: parent keys like ``xg_blend`` would otherwise
                # leak onto ``involvement_shares`` and break ``extra="forbid"``.
                base[key] = copy.deepcopy(value_map)
            else:
                _deep_merge(existing_map, value_map)
        else:
            base[key] = copy.deepcopy(value)
    return base


def _resolve_extends(
    raw: dict[str, Any],
    raw_by_name: dict[str, dict[str, Any]],
    source: str,
) -> dict[str, Any]:
    """Resolve one-level ``extends`` / ``overrides`` into a ModelSpec dict.

    Args:
        raw: YAML mapping for one file.
        raw_by_name: All raw specs in the directory, keyed by ``name``.
        source: Filename used in error messages.

    Returns:
        A mapping suitable for ``ModelSpec.model_validate``.

    Raises:
        ValueError: Missing parent, chain deeper than one, ``overrides``
            without ``extends``, or unexpected top-level keys on a fork.
    """
    if "extends" not in raw:
        if "overrides" in raw:
            raise ValueError(f"{source}: 'overrides' requires 'extends'")
        return raw

    extra = set(raw) - _EXTENDS_KEYS
    if extra:
        keys = ", ".join(sorted(repr(k) for k in extra))
        raise ValueError(
            f"{source}: extends forks may only set name/extends/overrides;"
            f" unexpected keys: {keys}"
        )

    parent_name = raw["extends"]
    if not isinstance(parent_name, str) or not parent_name:
        raise ValueError(f"{source}: extends must be a non-empty parent name")
    try:
        parent = raw_by_name[parent_name]
    except KeyError:
        raise ValueError(
            f"{source}: extends target {parent_name!r} not found"
        ) from None
    if "extends" in parent:
        raise ValueError(
            f"{source}: extends chain deeper than one"
            f" ({raw['name']!r} -> {parent_name!r})"
        )

    overrides_raw = raw.get("overrides")
    if overrides_raw is None:
        overrides_raw = {}
    if not isinstance(overrides_raw, dict):
        raise ValueError(f"{source}: overrides must be a mapping")
    overrides = cast(dict[str, Any], overrides_raw)

    merged = deep_merge(parent, overrides)
    merged.pop("extends", None)
    merged.pop("overrides", None)
    merged["name"] = raw["name"]
    return merged


def artifact_dir(data_dir: Path, spec: ModelSpec, as_of: GWKey) -> Path:
    """Directory for one fitted spec: ``artifacts/<model_id>/as_of=...``."""
    stamp = f"as_of={as_of.season}_gw{as_of.gw:02d}"
    return data_dir / "artifacts" / spec.model_id / stamp


def write_identity(dest: Path, spec: ModelSpec) -> None:
    """Write ``spec.json``, ``git_sha.txt``, and optional MC provenance.

    When the spec (or ``FPLT_MC_BACKEND``) selects the native Monte Carlo
    backend, also writes ``mc_provenance.json`` with ``fplt_mc.build_info()``
    and the env override, matching audit.json provenance fields.

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
    backend = str(spec.params.get("mc_backend", "python")).strip().lower()
    env_override = os.environ.get("FPLT_MC_BACKEND", "").strip()
    if backend == "native" or env_override.lower() == "native":
        from fplt.models.simulation_native import native_provenance

        prov_path = dest / "mc_provenance.json"
        prov_path.write_text(
            json.dumps(native_provenance(), sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )


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
    except PackageNotFoundError:
        return "0.0.0"
