"""Spec identity: ``spec_hash``, registry lock, and ``specs/sha=`` store.

``spec_hash`` is the sha256 of the fully resolved ModelSpec: ``extends`` /
``overrides`` already applied by the loader, every typed component default
filled in, ``name`` excluded (recursively), canonical JSON. Names stay human
labels; the lock file pins released names to hashes so an in-place edit fails
CI (data-model §3.7 / rollout B8).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, cast

import yaml

from fplt.models.component_params import PARAMS_BY_KIND, resolve_component_params
from fplt.models.spec import ModelSpec, load_model_specs

_LOCK_SCHEMA_VERSION = 1


def resolved_spec_payload(spec: ModelSpec) -> dict[str, Any]:
    """Return the nameless, defaults-filled payload used for ``spec_hash``.

    Component kinds in ``PARAMS_BY_KIND`` get params validated so every default
    is explicit. Nested component ``name`` fields are dropped the same way as
    the top-level name. Top-level predictor params (e.g. ``dc_layered``
    ``n_sims``) stay as written until those kinds gain typed params models.

    Args:
        spec: A loaded (already extends-resolved) ``ModelSpec``.

    Returns:
        A JSON-serialisable mapping suitable for hashing or the specs store.
    """
    dump: dict[str, Any] = spec.model_dump(mode="json")
    dump.pop("name", None)
    kind = str(dump["kind"])
    if kind in PARAMS_BY_KIND:
        raw_params_obj: object = dump.get("params", {})
        if not isinstance(raw_params_obj, dict):
            raise TypeError(f"params for kind {kind!r} must be a mapping")
        dump["params"] = resolve_component_params(
            kind, cast(dict[str, Any], raw_params_obj)
        )
    components_obj: object = dump.get("components", {})
    if not isinstance(components_obj, dict):
        raise TypeError("components must be a mapping")
    components: dict[str, Any] = {}
    for slot_obj, child in cast(dict[str, Any], components_obj).items():
        child_spec = ModelSpec.model_validate(child)
        components[str(slot_obj)] = resolved_spec_payload(child_spec)
    dump["components"] = components
    return dump


def canonical_spec_json(payload: dict[str, Any]) -> str:
    """Canonical JSON for hashing: sorted keys, compact separators."""
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def spec_hash(spec: ModelSpec) -> str:
    """Sha256 of the fully resolved, name-excluded ModelSpec payload.

    Args:
        spec: Model to identify.

    Returns:
        64-character lowercase hex digest.
    """
    raw = canonical_spec_json(resolved_spec_payload(spec))
    return hashlib.sha256(raw.encode()).hexdigest()


def specs_dir(data_dir: Path) -> Path:
    """Content-addressed resolved-spec root: ``data_dir/specs/``."""
    return data_dir / "specs"


def spec_store_path(data_dir: Path, digest: str) -> Path:
    """Path for one resolved spec: ``data_dir/specs/sha=<digest>.json``."""
    if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
        raise ValueError(f"spec_hash must be 64 lowercase hex chars, got {digest!r}")
    return specs_dir(data_dir) / f"sha={digest}.json"


def write_resolved_spec(data_dir: Path, spec: ModelSpec) -> Path:
    """Write the resolved payload under ``specs/sha=<hash>.json`` (write-once).

    When the file already exists with identical bytes, returns that path. A
    colliding path with different content raises ``FileExistsError``.

    Args:
        data_dir: Local data root (``$FPLT_DATA_DIR``).
        spec: Spec whose resolved payload is stored.

    Returns:
        The path that was written or already matched.

    Raises:
        FileExistsError: Path exists with different content.
        ValueError: Hash / payload inconsistency (should not happen).
    """
    digest = spec_hash(spec)
    payload = resolved_spec_payload(spec)
    text = canonical_spec_json(payload) + "\n"
    path = spec_store_path(data_dir, digest)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        existing = path.read_text(encoding="utf-8")
        if existing != text:
            raise FileExistsError(
                f"specs store collision at {path}: content differs from {digest}"
            )
        return path
    path.write_text(text, encoding="utf-8")
    return path


def load_resolved_spec(data_dir: Path, digest: str) -> dict[str, Any]:
    """Load a previously stored resolved-spec payload.

    Args:
        data_dir: Local data root.
        digest: ``spec_hash`` key.

    Returns:
        The JSON object written by ``write_resolved_spec``.

    Raises:
        FileNotFoundError: No file for ``digest``.
        ValueError: File is not a JSON object.
    """
    path = spec_store_path(data_dir, digest)
    loaded = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return cast(dict[str, Any], loaded)


def duplicate_hash_groups(name_to_hash: dict[str, str]) -> dict[str, list[str]]:
    """Map each shared ``spec_hash`` to the sorted names that resolve to it.

    Args:
        name_to_hash: Registry ``name → spec_hash``.

    Returns:
        Only groups with two or more names, keyed by hash.
    """
    buckets: dict[str, list[str]] = {}
    for name, digest in name_to_hash.items():
        buckets.setdefault(digest, []).append(name)
    return {
        digest: sorted(names) for digest, names in buckets.items() if len(names) > 1
    }


def registry_hashes(
    specs: dict[str, ModelSpec] | None = None,
) -> dict[str, str]:
    """Compute ``name → spec_hash`` for every registered ModelSpec.

    Args:
        specs: Specs keyed by name. Defaults to ``load_model_specs()``.

    Returns:
        Sorted-by-name mapping of name to digest.
    """
    loaded = specs if specs is not None else load_model_specs()
    return {name: spec_hash(spec) for name, spec in sorted(loaded.items())}


def default_registry_lock_path() -> Path:
    """Committed lock path: ``configs/models/registry.lock.yaml``."""
    root = Path(__file__).resolve().parents[3]
    return root / "configs" / "models" / "registry.lock.yaml"


def load_registry_lock(path: Path | None = None) -> dict[str, str]:
    """Load the committed ``name → spec_hash`` lock file.

    Args:
        path: Lock file. Defaults to ``configs/models/registry.lock.yaml``.

    Returns:
        The ``hashes`` mapping.

    Raises:
        FileNotFoundError: Lock file missing.
        ValueError: Empty, wrong shape, or bad digests.
    """
    lock_path = path if path is not None else default_registry_lock_path()
    raw_obj = yaml.safe_load(lock_path.read_text(encoding="utf-8"))
    if not isinstance(raw_obj, dict):
        raise ValueError(f"{lock_path.name} must contain a mapping")
    raw = cast(dict[str, Any], raw_obj)
    hashes_obj = raw.get("hashes")
    if not isinstance(hashes_obj, dict) or not hashes_obj:
        raise ValueError(f"{lock_path.name} must set a non-empty hashes mapping")
    out: dict[str, str] = {}
    for name_obj, digest_obj in cast(dict[Any, Any], hashes_obj).items():
        if not isinstance(name_obj, str) or not name_obj:
            raise ValueError(f"{lock_path.name}: hash keys must be non-empty strings")
        if not isinstance(digest_obj, str) or len(digest_obj) != 64:
            raise ValueError(
                f"{lock_path.name}: hashes[{name_obj!r}] must be a 64-char sha256"
            )
        if any(c not in "0123456789abcdef" for c in digest_obj):
            raise ValueError(
                f"{lock_path.name}: hashes[{name_obj!r}] must be lowercase hex"
            )
        out[name_obj] = digest_obj
    return out


def render_registry_lock(name_to_hash: dict[str, str]) -> str:
    """Render a lock file body for ``name_to_hash``.

    Args:
        name_to_hash: Mapping to pin.

    Returns:
        YAML text including a short header comment.
    """
    payload = {
        "schema_version": _LOCK_SCHEMA_VERSION,
        "hashes": {name: name_to_hash[name] for name in sorted(name_to_hash)},
    }
    body = yaml.safe_dump(payload, sort_keys=False, default_flow_style=False)
    header = (
        "# ModelSpec identity lock (rollout B8 / data-model §3.7).\n"
        "# name → spec_hash of the fully resolved spec (defaults filled, name\n"
        "# excluded). CI fails if a released name's hash changes — pick a new\n"
        "# name instead of editing in place. Regenerate via\n"
        "# fplt.models.spec_identity.write_registry_lock().\n"
    )
    return header + body


def write_registry_lock(
    path: Path | None = None,
    *,
    specs: dict[str, ModelSpec] | None = None,
) -> Path:
    """Write (overwrite) the committed registry lock from current specs.

    Args:
        path: Destination. Defaults to ``configs/models/registry.lock.yaml``.
        specs: Specs to hash. Defaults to ``load_model_specs()``.

    Returns:
        The path written.
    """
    lock_path = path if path is not None else default_registry_lock_path()
    lock_path.write_text(render_registry_lock(registry_hashes(specs)), encoding="utf-8")
    return lock_path


def check_registry_lock(
    *,
    specs: dict[str, ModelSpec] | None = None,
    lock_path: Path | None = None,
) -> None:
    """Fail if any registered name's ``spec_hash`` drifted from the lock.

    Also fails when the live registry and lock disagree on the set of names
    (added or removed YAML without updating the lock).

    Args:
        specs: Live specs. Defaults to ``load_model_specs()``.
        lock_path: Lock file path.

    Raises:
        AssertionError: Drift, missing names, or extra lock entries.
    """
    live = registry_hashes(specs)
    pinned = load_registry_lock(lock_path)
    missing = sorted(set(live) - set(pinned))
    extra = sorted(set(pinned) - set(live))
    drifted = sorted(
        name for name, digest in live.items() if pinned.get(name) != digest
    )
    problems: list[str] = []
    if missing:
        problems.append(f"names missing from lock: {missing}")
    if extra:
        problems.append(f"lock names with no YAML: {extra}")
    if drifted:
        problems.append(f"spec_hash drifted for: {drifted}")
    if problems:
        raise AssertionError(
            "; ".join(problems) + ". Update configs/models/registry.lock.yaml via "
            "write_registry_lock() in the same PR as the intentional change."
        )
