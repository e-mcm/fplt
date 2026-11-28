"""Patch engine for PipelineSpec experiment arms.

Arms are flat dotted-path diffs of a base ``PipelineSpec``
(``docs/proposals/fplt-experiments.md`` §9 step 2 /
``model-improvement-toolkit.md`` §2.1): scalar overrides, component
substitution by registered ModelSpec name, grid expansion, and auto-labels.

Validity: the flat leaf-diff of resolved payloads equals the effect of the
declared patch (no hidden changes). Arm identity is ``pipeline_hash`` of the
resolved arm.
"""

from __future__ import annotations

import copy
import itertools
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, cast

from fplt.models.pipeline_spec import (
    PipelineSpec,
    pipeline_hash,
    resolve_pipeline_model,
    resolved_pipeline_payload,
)
from fplt.models.spec import ModelSpec, load_model_specs
from fplt.models.spec_identity import resolved_spec_payload


class PatchError(ValueError):
    """Base error for PipelineSpec patch failures."""


class UnknownPatchPathError(PatchError):
    """A dotted path does not exist on the base pipeline."""


class PatchDiffError(PatchError):
    """Resolved arm differs from the base by more (or less) than the patch."""


@dataclass(frozen=True)
class ResolvedArm:
    """One experiment arm after applying a patch to a base PipelineSpec.

    Attributes:
        patch: Flat dotted-path overrides that produced this arm.
        pipeline: Resolved ``PipelineSpec`` (model may be inline after patches).
        pipeline_hash: Arm identity (``pipeline_hash`` of ``pipeline``).
        label: Auto-label derived from the patch (or an explicit override).
    """

    patch: dict[str, Any]
    pipeline: PipelineSpec
    pipeline_hash: str
    label: str


def apply_patch(
    base: PipelineSpec,
    patch: Mapping[str, Any],
    *,
    specs: dict[str, ModelSpec] | None = None,
    label: str | None = None,
    base_label: str | None = None,
) -> ResolvedArm:
    """Apply a flat dotted-path patch to ``base`` and validate the resolved diff.

    Args:
        base: Pipeline to patch.
        patch: Dotted paths → values. Component slots accept a registered
            ModelSpec name, an inline mapping, or a ``ModelSpec``.
        specs: Optional ModelSpec registry (defaults to ``load_model_specs()``).
        label: Optional explicit arm label. Defaults to ``auto_label``.
        base_label: Name used in the auto-label. Defaults to ``base.name`` or
            ``base.model_name``.

    Returns:
        The resolved arm (pipeline, hash, label, normalised patch).

    Raises:
        UnknownPatchPathError: A path is missing on the base.
        PatchDiffError: Applying the patch changed more (or less) than declared.
        KeyError: Component substitution names an unknown ModelSpec.
        ValidationError: The patched mapping is not a valid ``PipelineSpec``.
    """
    loaded = specs if specs is not None else load_model_specs()
    normalised = _normalise_patch(patch)
    arm_pipeline = _build_arm_pipeline(base, normalised, specs=loaded)
    validate_resolved_diff(base, normalised, arm_pipeline, specs=loaded)
    digest = pipeline_hash(arm_pipeline, specs=loaded)
    arm_label = (
        label
        if label is not None
        else auto_label(_base_label(base, base_label), normalised)
    )
    return ResolvedArm(
        patch=dict(normalised),
        pipeline=arm_pipeline,
        pipeline_hash=digest,
        label=arm_label,
    )


def expand_grid(
    base: PipelineSpec,
    grid: Mapping[str, Sequence[Any]],
    *,
    specs: dict[str, ModelSpec] | None = None,
    base_label: str | None = None,
) -> list[ResolvedArm]:
    """Expand a path → values grid into resolved arms (cartesian product).

    Path key order does not affect each arm's ``pipeline_hash``; the returned
    list is ordered by sorted path names then ``itertools.product`` order.

    Args:
        base: Pipeline to patch.
        grid: Dotted path → sequence of values. An empty mapping yields one
            control arm (empty patch).
        specs: Optional ModelSpec registry.
        base_label: Name used in auto-labels.

    Returns:
        One ``ResolvedArm`` per grid combination.
    """
    if not grid:
        return [
            apply_patch(base, {}, specs=specs, base_label=base_label),
        ]
    paths = sorted(grid.keys())
    value_lists = [list(grid[path]) for path in paths]
    arms: list[ResolvedArm] = []
    for combo in itertools.product(*value_lists):
        patch = {path: value for path, value in zip(paths, combo, strict=True)}
        arms.append(apply_patch(base, patch, specs=specs, base_label=base_label))
    return arms


def validate_resolved_diff(
    base: PipelineSpec,
    patch: Mapping[str, Any],
    arm: PipelineSpec,
    *,
    specs: dict[str, ModelSpec] | None = None,
) -> None:
    """Refuse an arm whose resolved leaf-diff is not exactly the patch effect.

    Args:
        base: Unpatched pipeline.
        patch: Declared flat patch.
        arm: Candidate resolved pipeline.
        specs: Optional ModelSpec registry.

    Raises:
        PatchDiffError: ``flat_diff(resolved(base), resolved(arm))`` does not
            equal the expected diff of applying ``patch`` to the resolved base.
        UnknownPatchPathError: ``patch`` names a path missing on the base.
    """
    loaded = specs if specs is not None else load_model_specs()
    normalised = _normalise_patch(patch)
    base_payload = resolved_pipeline_payload(base, specs=loaded)
    arm_payload = resolved_pipeline_payload(arm, specs=loaded)
    actual = flat_diff(base_payload, arm_payload)
    expected = expected_resolved_diff(base, normalised, specs=loaded)
    if actual != expected:
        raise PatchDiffError(
            "resolved pipeline diff does not equal the declared patch: "
            f"actual={actual!r} expected={expected!r}"
        )


def expected_resolved_diff(
    base: PipelineSpec,
    patch: Mapping[str, Any],
    *,
    specs: dict[str, ModelSpec] | None = None,
) -> dict[str, Any]:
    """Return the flat leaf-diff expected from applying ``patch`` to ``base``.

    Args:
        base: Unpatched pipeline.
        patch: Flat dotted-path overrides.
        specs: Optional ModelSpec registry.

    Returns:
        Mapping of dotted paths to new values (added/changed leaves only).

    Raises:
        UnknownPatchPathError: A path is missing on the resolved base.
        KeyError: Component substitution names an unknown ModelSpec.
    """
    loaded = specs if specs is not None else load_model_specs()
    normalised = _normalise_patch(patch)
    before = resolved_pipeline_payload(base, specs=loaded)
    after = copy.deepcopy(before)
    for path, value in normalised.items():
        resolved_value = _resolve_value_for_payload(path, value, specs=loaded)
        _set_path(after, path.split("."), resolved_value)
    return flat_diff(before, after)


def flat_diff(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    """Flat leaf-diff of two nested mappings (added and changed keys only).

    Args:
        before: Baseline nested mapping.
        after: Updated nested mapping.

    Returns:
        Dotted path → value in ``after`` for leaves that differ or were added.
        Removals are represented as ``path → None`` when a leaf disappears.
    """
    before_flat = _flatten(before)
    after_flat = _flatten(after)
    out: dict[str, Any] = {}
    for path, value in after_flat.items():
        if before_flat.get(path, _MISSING) != value:
            out[path] = value
    for path in before_flat:
        if path not in after_flat:
            out[path] = None
    return dict(sorted(out.items()))


def auto_label(base_name: str, patch: Mapping[str, Any]) -> str:
    """Build a truthful auto-label from a base name and flat patch.

    Empty patches return ``base_name`` alone. Otherwise
    ``{base}[{short_path}={value},…]`` with paths sorted. Component param
    paths shorten ``model.components.<slot>.params.<key>`` to
    ``<slot>.<key>`` (data-model §3.7).

    Args:
        base_name: Human label for the unpatched pipeline.
        patch: Flat dotted-path overrides.

    Returns:
        Auto-label string.
    """
    normalised = _normalise_patch(patch)
    if not normalised:
        return base_name
    parts = [
        f"{_shorten_path(path)}={_format_label_value(value)}"
        for path, value in normalised.items()
    ]
    return f"{base_name}[{','.join(parts)}]"


def resolved_flat_diff(
    base: PipelineSpec,
    other: PipelineSpec,
    *,
    specs: dict[str, ModelSpec] | None = None,
) -> dict[str, Any]:
    """Flat leaf-diff of two pipelines' resolved payloads.

    Args:
        base: Baseline pipeline.
        other: Comparison pipeline.
        specs: Optional ModelSpec registry.

    Returns:
        Dotted path → value diff (see ``flat_diff``).
    """
    loaded = specs if specs is not None else load_model_specs()
    return flat_diff(
        resolved_pipeline_payload(base, specs=loaded),
        resolved_pipeline_payload(other, specs=loaded),
    )


_MISSING = object()


def _normalise_patch(patch: Mapping[str, Any]) -> dict[str, Any]:
    """Copy and sort-stable normalise a patch mapping."""
    out: dict[str, Any] = {}
    for path_raw, value in patch.items():
        path = path_raw.strip()
        if not path:
            raise PatchError(f"patch paths must be non-empty strings, got {path_raw!r}")
        if ".." in path or path.startswith(".") or path.endswith("."):
            raise UnknownPatchPathError(f"invalid patch path {path!r}")
        if path in out:
            raise PatchError(f"duplicate patch path {path!r}")
        out[path] = value
    return {key: out[key] for key in sorted(out)}


def _base_label(base: PipelineSpec, override: str | None) -> str:
    if override is not None:
        return override
    if base.name:
        return base.name
    return base.model_name


def _build_arm_pipeline(
    base: PipelineSpec,
    patch: Mapping[str, Any],
    *,
    specs: dict[str, ModelSpec],
) -> PipelineSpec:
    """Mutate a dump of ``base`` by ``patch`` and validate a ``PipelineSpec``."""
    working = _working_dump(base, specs=specs)
    for path, value in patch.items():
        prepared = _resolve_value_for_dump(path, value, specs=specs)
        _set_path(working, path.split("."), prepared)
    model_obj = cast(dict[str, Any], working["model"])
    working = {
        **working,
        "model": ModelSpec.model_validate(model_obj),
    }
    return PipelineSpec.model_validate(working)


def _working_dump(
    base: PipelineSpec,
    *,
    specs: dict[str, ModelSpec],
) -> dict[str, Any]:
    """JSON-mode dump with ``model`` expanded to a ModelSpec mapping."""
    dump = base.model_dump(mode="json")
    model = resolve_pipeline_model(base, specs=specs)
    dump["model"] = model.model_dump(mode="json")
    return dump


def _resolve_value_for_dump(
    path: str,
    value: Any,
    *,
    specs: dict[str, ModelSpec],
) -> Any:
    """Resolve a patch value for insertion into a PipelineSpec dump."""
    if _is_component_or_model_path(path) and isinstance(value, str):
        return _lookup_model(value, specs=specs).model_dump(mode="json")
    if isinstance(value, ModelSpec):
        return value.model_dump(mode="json")
    return copy.deepcopy(value)


def _resolve_value_for_payload(
    path: str,
    value: Any,
    *,
    specs: dict[str, ModelSpec],
) -> Any:
    """Resolve a patch value for insertion into a resolved payload tree."""
    if _is_component_or_model_path(path) and isinstance(value, str):
        return resolved_spec_payload(_lookup_model(value, specs=specs))
    if isinstance(value, ModelSpec):
        return resolved_spec_payload(value)
    if _is_component_or_model_path(path) and isinstance(value, dict):
        return resolved_spec_payload(
            ModelSpec.model_validate(cast(dict[str, Any], value))
        )
    return copy.deepcopy(value)


def _is_component_or_model_path(path: str) -> bool:
    """True when ``path`` addresses the whole model or a component slot."""
    if path == "model":
        return True
    parts = path.split(".")
    return len(parts) == 3 and parts[0] == "model" and parts[1] == "components"


def _lookup_model(name: str, *, specs: dict[str, ModelSpec]) -> ModelSpec:
    cleaned = name.strip()
    if not cleaned:
        raise KeyError("component substitution requires a non-empty ModelSpec name")
    try:
        return specs[cleaned]
    except KeyError:
        raise KeyError(f"no ModelSpec registered for {cleaned!r}") from None


def _set_path(
    tree: dict[str, Any],
    parts: list[str],
    value: Any,
) -> None:
    """Set a dotted path on ``tree``, requiring the full path to exist.

    Args:
        tree: Nested mapping to mutate.
        parts: Path segments (non-empty).
        value: Value to assign at the leaf.

    Raises:
        UnknownPatchPathError: A segment is missing or an intermediate value
            is not a mapping.
    """
    current: dict[str, Any] = tree
    for index, part in enumerate(parts[:-1]):
        if part not in current:
            raise UnknownPatchPathError(".".join(parts[: index + 1]))
        next_value = current[part]
        if not isinstance(next_value, dict):
            raise UnknownPatchPathError(
                f"{'.'.join(parts[: index + 1])} is not a mapping"
            )
        current = cast(dict[str, Any], next_value)
    leaf = parts[-1]
    if leaf not in current:
        raise UnknownPatchPathError(".".join(parts))
    current[leaf] = value


def _flatten(obj: Any, prefix: str = "") -> dict[str, Any]:
    """Flatten a nested mapping to dotted leaf paths."""
    if isinstance(obj, dict):
        out: dict[str, Any] = {}
        for key_obj, value in cast(dict[str, Any], obj).items():
            key = str(key_obj)
            path = f"{prefix}.{key}" if prefix else key
            out.update(_flatten(value, path))
        return out
    if isinstance(obj, list):
        items = cast(list[Any], obj)
        return {prefix: list(items)}
    return {prefix: obj}


def _shorten_path(path: str) -> str:
    """Shorten model component param paths for auto-labels."""
    prefix = "model.components."
    if not path.startswith(prefix):
        return path
    rest = path[len(prefix) :]
    parts = rest.split(".")
    if len(parts) >= 3 and parts[1] == "params":
        return f"{parts[0]}.{'.'.join(parts[2:])}"
    return rest


def _format_label_value(value: Any) -> str:
    """Compact value rendering for auto-labels."""
    if isinstance(value, ModelSpec):
        return value.name
    if isinstance(value, dict):
        mapping = cast(dict[str, Any], value)
        name_obj = mapping.get("name")
        if isinstance(name_obj, str) and name_obj:
            return name_obj
    if isinstance(value, float):
        return f"{value:.12g}"
    return str(cast(object, value))
