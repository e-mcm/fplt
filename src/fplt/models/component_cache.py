"""Component output cache + dependency graph for layered experiment arms.

Toolkit §2.2 / ``fplt-experiments.md`` §4 and §9 step 4: a component's fitted
state is addressed by ``(component spec hash, upstream component hashes,
code_sha, lockfile_hash, build_id, deadline)``. Arms that share an upstream
slot reuse the cached fit instead of refitting (e.g. a minutes arm keeps the
control's scoreline / involvement / other). Cross-slot dependencies are
explicit so a changed upstream invalidates downstream keys.
"""

from __future__ import annotations

import hashlib
import json
import logging
import pickle
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from fplt.models.spec import ModelSpec
from fplt.models.spec_identity import canonical_spec_json, spec_hash
from fplt.types import GWKey

logger = logging.getLogger(__name__)

# Layered ``dc_layered`` slots. Involvement team-normalises internally (no
# upstream component). Lineup multipliers (planned model-03) would list
# ``minutes`` here so their cache keys include the minutes hash.
LAYERED_COMPONENT_SLOTS: tuple[str, ...] = (
    "score",
    "minutes",
    "involvement",
    "other",
)

LAYERED_COMPONENT_UPSTREAM: dict[str, tuple[str, ...]] = {
    "score": (),
    "minutes": (),
    "involvement": (),
    "other": (),
}


class ComponentCacheError(ValueError):
    """Base error for component-cache failures."""


class UnknownComponentSlotError(ComponentCacheError):
    """A slot name is not in the declared dependency graph."""


def component_upstream(
    slot: str,
    *,
    graph: Mapping[str, Sequence[str]] | None = None,
) -> tuple[str, ...]:
    """Return the declared upstream slots for ``slot``.

    Args:
        slot: Component slot name (e.g. ``minutes``).
        graph: Optional override of ``LAYERED_COMPONENT_UPSTREAM``.

    Returns:
        Upstream slot names in declaration order.

    Raises:
        UnknownComponentSlotError: ``slot`` is missing from the graph.
    """
    deps = LAYERED_COMPONENT_UPSTREAM if graph is None else graph
    if slot not in deps:
        raise UnknownComponentSlotError(f"unknown component slot {slot!r}")
    return tuple(deps[slot])


def transitive_dependents(
    changed: str,
    *,
    graph: Mapping[str, Sequence[str]] | None = None,
) -> frozenset[str]:
    """Return slots whose cache keys include ``changed`` (transitively).

    Args:
        changed: Slot whose spec or output changed.
        graph: Optional override of ``LAYERED_COMPONENT_UPSTREAM``.

    Returns:
        Downstream slots that must be invalidated (never includes ``changed``).

    Raises:
        UnknownComponentSlotError: ``changed`` is missing from the graph.
    """
    deps = dict(LAYERED_COMPONENT_UPSTREAM if graph is None else graph)
    if changed not in deps:
        raise UnknownComponentSlotError(f"unknown component slot {changed!r}")
    reverse: dict[str, set[str]] = {name: set() for name in deps}
    for slot, upstream in deps.items():
        for parent in upstream:
            if parent not in reverse:
                raise UnknownComponentSlotError(
                    f"upstream slot {parent!r} of {slot!r} is not in the graph"
                )
            reverse[parent].add(slot)
    found: set[str] = set()
    stack = list(reverse[changed])
    while stack:
        slot = stack.pop()
        if slot in found:
            continue
        found.add(slot)
        stack.extend(reverse[slot])
    return frozenset(found)


def upstream_component_hashes(
    slot: str,
    component_hashes: Mapping[str, str],
    *,
    graph: Mapping[str, Sequence[str]] | None = None,
) -> dict[str, str]:
    """Map each upstream slot of ``slot`` to its component spec hash.

    Args:
        slot: Slot whose upstream hashes are needed for the cache key.
        component_hashes: ``slot → spec_hash`` for every relevant component.
        graph: Optional dependency-graph override.

    Returns:
        Sorted-key mapping of upstream slot → hash (empty when independent).

    Raises:
        UnknownComponentSlotError: ``slot`` is unknown.
        KeyError: An upstream slot is missing from ``component_hashes``.
    """
    upstream = component_upstream(slot, graph=graph)
    missing = [name for name in upstream if name not in component_hashes]
    if missing:
        raise KeyError(
            f"missing upstream component hashes for {slot!r}: {sorted(missing)}"
        )
    return {name: component_hashes[name] for name in sorted(upstream)}


def component_cache_key(
    *,
    component_spec_hash: str,
    upstream_hashes: Mapping[str, str],
    code_sha: str,
    lockfile_hash: str,
    build_id: str,
    deadline: GWKey,
) -> str:
    """Sha256 cache key for one fitted component at one deadline.

    Payload is ``(component spec hash, upstream hashes, code_sha,
    lockfile_hash, build_id, deadline)`` per experiments §4 / toolkit §2.2.

    Args:
        component_spec_hash: ``spec_hash`` of this slot's ``ModelSpec``.
        upstream_hashes: Upstream slot → ``spec_hash`` (may be empty).
        code_sha: Git HEAD of the checkout that produced the fit.
        lockfile_hash: Digest of ``uv.lock``.
        build_id: Curated build id the PITView was made from.
        deadline: ``as_of`` gameweek the fit is for.

    Returns:
        64-character lowercase hex digest.
    """
    payload = {
        "component_spec_hash": component_spec_hash,
        "upstream": {key: upstream_hashes[key] for key in sorted(upstream_hashes)},
        "code_sha": code_sha,
        "lockfile_hash": lockfile_hash,
        "build_id": build_id,
        "deadline": {"season": deadline.season, "gw": deadline.gw},
    }
    raw = canonical_spec_json(payload)
    return hashlib.sha256(raw.encode()).hexdigest()


def component_hashes_for_spec(spec: ModelSpec) -> dict[str, str]:
    """Return ``slot → spec_hash`` for every component on a layered ``ModelSpec``.

    Args:
        spec: Layered model with a ``components`` mapping.

    Returns:
        Hash per component slot.

    Raises:
        KeyError: A layered slot is missing from ``spec.components``.
    """
    missing = [name for name in LAYERED_COMPONENT_SLOTS if name not in spec.components]
    if missing:
        raise KeyError(f"layered spec is missing components {missing}")
    return {slot: spec_hash(spec.components[slot]) for slot in LAYERED_COMPONENT_SLOTS}


def reusable_slots(
    control: ModelSpec,
    arm: ModelSpec,
    *,
    graph: Mapping[str, Sequence[str]] | None = None,
) -> frozenset[str]:
    """Slots whose fitted state the arm can reuse from the control.

    A slot is reusable when its own ``spec_hash`` matches the control and every
    upstream slot is also reusable (so upstream hashes in the cache key match).

    Args:
        control: Control (base) layered model.
        arm: Treatment arm layered model.
        graph: Optional dependency-graph override.

    Returns:
        Slot names the arm may inject via ``CompositePredictor.fit(precomputed=…)``.
    """
    control_hashes = component_hashes_for_spec(control)
    arm_hashes = component_hashes_for_spec(arm)
    deps = LAYERED_COMPONENT_UPSTREAM if graph is None else graph
    memo: dict[str, bool] = {}

    def can_reuse(slot: str) -> bool:
        if slot in memo:
            return memo[slot]
        if control_hashes[slot] != arm_hashes[slot]:
            memo[slot] = False
            return False
        upstream = tuple(deps.get(slot, ()))
        ok = all(can_reuse(parent) for parent in upstream)
        memo[slot] = ok
        return ok

    return frozenset(slot for slot in LAYERED_COMPONENT_SLOTS if can_reuse(slot))


def component_cache_dir(data_dir: Path, cache_key: str) -> Path:
    """Directory for one cache entry: ``component_cache/key=<digest>/``."""
    if len(cache_key) != 64 or any(c not in "0123456789abcdef" for c in cache_key):
        raise ValueError(
            f"component cache key must be 64 lowercase hex chars, got {cache_key!r}"
        )
    return data_dir / "component_cache" / f"key={cache_key}"


def save_component(
    data_dir: Path,
    cache_key: str,
    component: Any,
    *,
    meta: Mapping[str, Any] | None = None,
) -> Path:
    """Write a fitted component under ``component_cache/key=<digest>/`` (write-once).

    Args:
        data_dir: Local data root (``$FPLT_DATA_DIR``).
        cache_key: Digest from ``component_cache_key``.
        component: Fitted component object (pickled).
        meta: Optional JSON-serialisable provenance written beside the state.

    Returns:
        The cache entry directory.

    Raises:
        FileExistsError: The entry already exists with different bytes, or a
            colliding ``state.pkl`` is present.
    """
    dest = component_cache_dir(data_dir, cache_key)
    dest.mkdir(parents=True, exist_ok=True)
    state_path = dest / "state.pkl"
    payload = pickle.dumps(component)
    if state_path.exists():
        if state_path.read_bytes() == payload:
            return dest
        raise FileExistsError(f"component cache entry already exists at {dest}")
    state_path.write_bytes(payload)
    if meta is not None:
        meta_path = dest / "meta.json"
        if not meta_path.exists():
            meta_path.write_text(
                json.dumps(dict(meta), sort_keys=True, indent=2) + "\n",
                encoding="utf-8",
            )
    return dest


def load_component(data_dir: Path, cache_key: str) -> Any | None:
    """Load a fitted component for ``cache_key``, or ``None`` on a miss.

    Args:
        data_dir: Local data root.
        cache_key: Digest from ``component_cache_key``.

    Returns:
        The unpickled component, or ``None`` when the entry is absent.
    """
    state_path = component_cache_dir(data_dir, cache_key) / "state.pkl"
    if not state_path.exists():
        return None
    return pickle.loads(state_path.read_bytes())


def fit_or_load_component(
    data_dir: Path,
    *,
    slot: str,
    component_spec: ModelSpec,
    component_hashes: Mapping[str, str],
    code_sha: str,
    lockfile_hash: str,
    build_id: str,
    deadline: GWKey,
    fit: Any,
    graph: Mapping[str, Sequence[str]] | None = None,
) -> tuple[Any, str, bool]:
    """Return a fitted component from cache or by calling ``fit``.

    Args:
        data_dir: Local data root.
        slot: Component slot name.
        component_spec: Spec for this slot (used only for the hash / meta).
        component_hashes: Full ``slot → spec_hash`` map for upstream lookup.
        code_sha: Git HEAD.
        lockfile_hash: ``uv.lock`` digest.
        build_id: Curated build id.
        deadline: Fit deadline.
        fit: Zero-arg callable that returns a fitted component on a miss.
        graph: Optional dependency-graph override.

    Returns:
        ``(component, cache_key, cache_hit)``.
    """
    digest = spec_hash(component_spec)
    key = component_cache_key(
        component_spec_hash=digest,
        upstream_hashes=upstream_component_hashes(slot, component_hashes, graph=graph),
        code_sha=code_sha,
        lockfile_hash=lockfile_hash,
        build_id=build_id,
        deadline=deadline,
    )
    cached = load_component(data_dir, key)
    if cached is not None:
        logger.info("component cache hit slot=%s key=%s", slot, key[:12])
        return cached, key, True
    logger.info("component cache miss slot=%s key=%s", slot, key[:12])
    fitted = fit()
    save_component(
        data_dir,
        key,
        fitted,
        meta={
            "slot": slot,
            "component_spec_hash": digest,
            "build_id": build_id,
            "deadline": {"season": deadline.season, "gw": deadline.gw},
        },
    )
    return fitted, key, False
