"""Stage B8: spec_hash, registry.lock pin, specs/sha= store."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from fplt.models.spec import ModelSpec, load_model_specs
from fplt.models.spec_identity import (
    check_registry_lock,
    duplicate_hash_groups,
    load_registry_lock,
    load_resolved_spec,
    registry_hashes,
    resolved_spec_payload,
    spec_hash,
    spec_store_path,
    write_registry_lock,
    write_resolved_spec,
)

# Known aliases: same resolved model under different released names.
# Update deliberately when an alias is retired or a new alias appears.
_KNOWN_ALIAS_GROUPS: frozenset[frozenset[str]] = frozenset(
    {
        frozenset({"dc_layered_cs_v1", "dc_layered_defcon_rate_v1", "dc_layered_v4"}),
        frozenset({"dc_layered_nocs_v1", "dc_layered_v3"}),
    }
)


def test_spec_hash_excludes_name_and_fills_component_defaults() -> None:
    sparse = ModelSpec(
        name="a",
        kind="involvement_xg",
        params={"window": 10, "xg_blend": 0.7},
    )
    renamed = sparse.model_copy(update={"name": "b"})
    assert spec_hash(sparse) == spec_hash(renamed)
    payload = resolved_spec_payload(sparse)
    assert "name" not in payload
    assert payload["params"]["shrinkage"] == 5.0
    assert payload["params"]["use_penalties_order"] is True
    assert "name" not in payload["components"]


def test_nested_component_names_excluded_from_hash() -> None:
    left = ModelSpec(
        name="layered_a",
        kind="dc_layered",
        params={"n_sims": 100},
        components={
            "involvement": ModelSpec(
                name="inv_a",
                kind="involvement_xg",
                params={"xg_blend": 0.5},
            )
        },
    )
    right = left.model_copy(
        update={
            "name": "layered_b",
            "components": {
                "involvement": ModelSpec(
                    name="inv_b",
                    kind="involvement_xg",
                    params={"xg_blend": 0.5},
                )
            },
        }
    )
    assert spec_hash(left) == spec_hash(right)
    nested = resolved_spec_payload(left)["components"]["involvement"]
    assert "name" not in nested
    assert nested["params"]["window"] == 10


def test_specs_store_is_content_addressed_and_write_once(tmp_path: Path) -> None:
    spec = ModelSpec(name="demo", kind="involvement_shares", params={"window": 8})
    digest = spec_hash(spec)
    path = write_resolved_spec(tmp_path, spec)
    assert path == spec_store_path(tmp_path, digest)
    assert path.is_file()
    loaded = load_resolved_spec(tmp_path, digest)
    assert loaded == resolved_spec_payload(spec)
    assert "name" not in loaded
    # Identical rewrite is fine.
    assert write_resolved_spec(tmp_path, spec) == path
    # Collision with different content refuses.
    path.write_text('{"tampered": true}\n', encoding="utf-8")
    with pytest.raises(FileExistsError, match="collision"):
        write_resolved_spec(tmp_path, spec)


def test_registry_lock_pins_every_registered_name() -> None:
    check_registry_lock()
    live = registry_hashes()
    pinned = load_registry_lock()
    assert live == pinned
    assert len(live) == len(load_model_specs())


def test_duplicate_hashes_are_flagged_as_known_aliases() -> None:
    groups = duplicate_hash_groups(registry_hashes())
    found = frozenset(frozenset(names) for names in groups.values())
    assert found == _KNOWN_ALIAS_GROUPS


def test_check_registry_lock_detects_drift(tmp_path: Path) -> None:
    specs = {
        "keep": ModelSpec(name="keep", kind="involvement_shares", params={}),
        "change_me": ModelSpec(
            name="change_me", kind="involvement_shares", params={"window": 3}
        ),
    }
    lock_path = tmp_path / "registry.lock.yaml"
    write_registry_lock(lock_path, specs=specs)
    drifted = {
        "keep": specs["keep"],
        "change_me": ModelSpec(
            name="change_me", kind="involvement_shares", params={"window": 9}
        ),
    }
    with pytest.raises(AssertionError, match="drifted"):
        check_registry_lock(specs=drifted, lock_path=lock_path)


def test_load_model_specs_skips_registry_lock(tmp_path: Path) -> None:
    (tmp_path / "rolling.yaml").write_text(
        "name: rolling_mean_v1\nkind: rolling_mean\nparams:\n  window: 5\n",
        encoding="utf-8",
    )
    lock = {
        "schema_version": 1,
        "hashes": {"rolling_mean_v1": "a" * 64},
    }
    (tmp_path / "registry.lock.yaml").write_text(yaml.safe_dump(lock), encoding="utf-8")
    loaded = load_model_specs(tmp_path)
    assert list(loaded) == ["rolling_mean_v1"]


def test_write_predict_run_persists_specs_store(tmp_path: Path) -> None:
    from datetime import UTC, datetime

    from fplt.models.predictions import empty_predictions
    from fplt.store.runs import write_predict_run
    from fplt.types import GWKey

    spec = load_model_specs()["rolling_mean_v1"]
    digest = spec_hash(spec)
    write_predict_run(
        tmp_path,
        spec=spec,
        as_of=GWKey(season="2024-25", gw=1),
        deadline_utc=datetime(2024, 8, 16, 17, 30, tzinfo=UTC),
        horizon=1,
        frame=empty_predictions(),
    )
    assert spec_store_path(tmp_path, digest).is_file()
    assert load_resolved_spec(tmp_path, digest)["kind"] == "rolling_mean"
