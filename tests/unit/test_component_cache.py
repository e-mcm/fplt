"""Toolkit 5 / experiments step 4: component output cache + composite hook."""

from __future__ import annotations

from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest

import fplt.models.composite  # noqa: F401 — registers dc_layered
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.component_cache import (
    LAYERED_COMPONENT_SLOTS,
    LAYERED_COMPONENT_UPSTREAM,
    UnknownComponentSlotError,
    component_cache_dir,
    component_cache_key,
    component_hashes_for_spec,
    component_upstream,
    fit_or_load_component,
    load_component,
    reusable_slots,
    save_component,
    transitive_dependents,
    upstream_component_hashes,
)
from fplt.models.composite import CompositePredictor
from fplt.models.spec import ModelSpec, load_model_specs
from fplt.models.spec_identity import spec_hash
from fplt.schemas.manifests import RunManifest
from fplt.schemas.predictions import PredictionSchema
from fplt.types import GWKey
from tests.builders import build_league
from tests.unit.test_data_model_schemas import _run_kwargs

_AS_OF = GWKey("2024-25", 1)
_HORIZON = [GWKey("2024-25", 1)]
_KEYS = ["target_season", "target_gw", "player_key", "fixture_id"]


def _layered_small() -> ModelSpec:
    """dc_layered_v4 with a cheap sim count for unit tests."""
    base = load_model_specs()["dc_layered_v4"]
    return base.model_copy(
        update={"params": {**base.params, "n_sims": 40, "mc_backend": "python"}}
    )


def _minutes_arm(base: ModelSpec, blend: float) -> ModelSpec:
    minutes = base.components["minutes"].model_copy(
        update={
            "params": {
                **base.components["minutes"].params,
                "logistic_blend": blend,
            }
        }
    )
    return base.model_copy(
        update={"components": {**base.components, "minutes": minutes}}
    )


def _sorted(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.sort(_KEYS)


def test_run_manifest_accepts_component_run_type() -> None:
    """RunType gains ``component`` (#405 / AGENTS.md §7)."""
    manifest = RunManifest.model_validate(
        _run_kwargs(run_type="component", tags={"component": "minutes"})
    )
    assert manifest.run_type == "component"
    assert manifest.tags["component"] == "minutes"


def test_dependency_graph_is_explicit() -> None:
    """Layered slots are declared; involvement has no upstream component."""
    assert set(LAYERED_COMPONENT_UPSTREAM) == set(LAYERED_COMPONENT_SLOTS)
    assert LAYERED_COMPONENT_UPSTREAM["involvement"] == ()
    assert LAYERED_COMPONENT_UPSTREAM["minutes"] == ()
    # Planned lineup multipliers that read minutes must list minutes upstream.
    with_lineup = {
        **LAYERED_COMPONENT_UPSTREAM,
        "lineup_multipliers": ("minutes",),
    }
    assert transitive_dependents("minutes", graph=with_lineup) == frozenset(
        {"lineup_multipliers"}
    )
    assert transitive_dependents("score") == frozenset()


def test_upstream_change_invalidates_downstream_cache_key() -> None:
    """A changed upstream hash changes the downstream cache key."""
    deadline = _AS_OF
    common = {
        "code_sha": "abc",
        "lockfile_hash": "d" * 64,
        "build_id": "build_1",
        "deadline": deadline,
    }
    graph = {
        "score": (),
        "minutes": (),
        "involvement": (),
        "other": ("minutes",),
    }
    hashes_a = {
        "score": "1" * 64,
        "minutes": "2" * 64,
        "involvement": "3" * 64,
        "other": "4" * 64,
    }
    hashes_b = {**hashes_a, "minutes": "9" * 64}
    key_a = component_cache_key(
        component_spec_hash=hashes_a["other"],
        upstream_hashes=upstream_component_hashes("other", hashes_a, graph=graph),
        **common,
    )
    key_b = component_cache_key(
        component_spec_hash=hashes_b["other"],
        upstream_hashes=upstream_component_hashes("other", hashes_b, graph=graph),
        **common,
    )
    assert key_a != key_b
    # Unrelated score key is stable when only minutes changes.
    score_a = component_cache_key(
        component_spec_hash=hashes_a["score"],
        upstream_hashes=upstream_component_hashes("score", hashes_a, graph=graph),
        **common,
    )
    score_b = component_cache_key(
        component_spec_hash=hashes_b["score"],
        upstream_hashes=upstream_component_hashes("score", hashes_b, graph=graph),
        **common,
    )
    assert score_a == score_b
    assert "other" in transitive_dependents("minutes", graph=graph)


def test_reusable_slots_skips_changed_minutes() -> None:
    """A minutes arm reuses score / involvement / other from the control."""
    control = _layered_small()
    arm = _minutes_arm(control, blend=0.9)
    assert control.components["minutes"].params.get("logistic_blend") != 0.9
    assert reusable_slots(control, arm) == frozenset({"score", "involvement", "other"})
    assert reusable_slots(control, control) == frozenset(LAYERED_COMPONENT_SLOTS)


def test_cache_round_trip_and_fit_or_load(tmp_path: Path) -> None:
    """save/load and fit_or_load_component hit after the first miss."""
    control = _layered_small()
    hashes = component_hashes_for_spec(control)
    calls = {"n": 0}

    def _fit() -> dict[str, str]:
        calls["n"] += 1
        return {"fitted": "score"}

    first, key, hit = fit_or_load_component(
        tmp_path,
        slot="score",
        component_spec=control.components["score"],
        component_hashes=hashes,
        code_sha="deadbeef",
        lockfile_hash="e" * 64,
        build_id="build_x",
        deadline=_AS_OF,
        fit=_fit,
    )
    assert hit is False
    assert calls["n"] == 1
    assert first == {"fitted": "score"}
    second, key2, hit2 = fit_or_load_component(
        tmp_path,
        slot="score",
        component_spec=control.components["score"],
        component_hashes=hashes,
        code_sha="deadbeef",
        lockfile_hash="e" * 64,
        build_id="build_x",
        deadline=_AS_OF,
        fit=_fit,
    )
    assert hit2 is True
    assert key2 == key
    assert second == first
    assert calls["n"] == 1
    assert load_component(tmp_path, key) == first
    with pytest.raises(FileExistsError):
        save_component(tmp_path, key, {"fitted": "other"})


def test_arm_reusing_cached_components_is_bit_identical(tmp_path: Path) -> None:
    """Cached unchanged components + refit minutes == full arm refit."""
    store = build_league(DataStore(tmp_path / "league"))
    view = PITData(store).view(_AS_OF)
    control_spec = _layered_small()
    arm_spec = _minutes_arm(control_spec, blend=0.85)

    control = CompositePredictor(control_spec)
    control.fit(view)

    full_arm = CompositePredictor(arm_spec)
    full_arm.fit(view)
    full_pred = _sorted(full_arm.predict(view, _HORIZON))

    reuse = reusable_slots(control_spec, arm_spec)
    assert reuse == frozenset({"score", "involvement", "other"})
    precomputed = {
        "score": control.score_model,
        "involvement": control.involvement_model,
        "other": control._other,
    }
    hooked = CompositePredictor(arm_spec)
    hooked.fit(view, precomputed=precomputed)
    hooked_pred = _sorted(hooked.predict(view, _HORIZON))
    assert hooked_pred.equals(full_pred)


def test_composite_hook_passes_schema_and_leakage(tmp_path: Path) -> None:
    """Contract suite behaviours (schema + leakage) hold with the hook."""
    store = build_league(DataStore(tmp_path / "league"))
    view = PITData(store).view(_AS_OF)
    spec = _layered_small()
    baseline = CompositePredictor(spec)
    baseline.fit(view)
    expected = _sorted(baseline.predict(view, _HORIZON))
    checked = PredictionSchema.validate(expected)
    assert isinstance(checked, pl.DataFrame)

    hooked = CompositePredictor(spec)
    hooked.fit(
        view,
        precomputed={
            "score": baseline.score_model,
            "minutes": baseline.minutes_model,
            "involvement": baseline.involvement_model,
            "other": baseline._other,
        },
    )
    after_hook = _sorted(hooked.predict(view, _HORIZON))
    assert after_hook.equals(expected)

    # Leakage: bump post-cutoff rows; predictions must not move.
    cutoff = store.deadline(_AS_OF)
    later = cutoff + timedelta(days=1)
    status = store.read("player_status").with_columns(
        pl.lit(later).cast(pl.Datetime("us", "UTC")).alias("available_at"),
        (pl.col("ep_next") + 5).alias("ep_next"),
    )
    store.write("player_status", status)
    stats = store.read("player_fixture_stats")
    bumped = stats.with_columns(
        pl.when(pl.col("available_at") > cutoff)
        .then(pl.col("total_points") + 1)
        .otherwise(pl.col("total_points"))
        .alias("total_points")
    )
    store.write("player_fixture_stats", bumped)
    view2 = PITData(store).view(_AS_OF)
    leaked = CompositePredictor(spec)
    leaked.fit(
        view2,
        precomputed={
            "score": baseline.score_model,
            "minutes": baseline.minutes_model,
            "involvement": baseline.involvement_model,
            "other": baseline._other,
        },
    )
    assert _sorted(leaked.predict(view2, _HORIZON)).equals(expected)


def test_unknown_slot_and_missing_upstream_raise() -> None:
    """Unknown slots and missing upstream hashes fail loudly."""
    with pytest.raises(UnknownComponentSlotError):
        component_upstream("not_a_slot")
    with pytest.raises(UnknownComponentSlotError):
        transitive_dependents("not_a_slot")
    with pytest.raises(UnknownComponentSlotError, match="not in the graph"):
        transitive_dependents(
            "score",
            graph={"score": (), "other": ("missing_parent",)},
        )
    with pytest.raises(KeyError, match="missing upstream"):
        upstream_component_hashes(
            "other",
            {"other": "a" * 64},
            graph={"other": ("minutes",), "minutes": ()},
        )
    assert component_upstream("score") == ()


def test_reusable_slots_memoises_shared_upstream() -> None:
    """Shared upstream is memoised; a broken upstream blocks dependents."""
    control = _layered_small()
    # Same hashes as control, but a graph where involvement + other share minutes.
    graph = {
        "score": (),
        "minutes": (),
        "involvement": ("minutes",),
        "other": ("minutes",),
    }
    assert reusable_slots(control, control, graph=graph) == frozenset(
        LAYERED_COMPONENT_SLOTS
    )
    arm = _minutes_arm(control, blend=0.99)
    assert reusable_slots(control, arm, graph=graph) == frozenset({"score"})


def test_transitive_dependents_skips_already_seen() -> None:
    """Diamond dependency: a shared grandchild is only collected once."""
    graph = {
        "score": (),
        "minutes": (),
        "involvement": ("score",),
        "other": ("score",),
        # Both parents push this child onto the stack; the second visit is skipped.
        "lineup": ("involvement", "other"),
    }
    assert transitive_dependents("score", graph=graph) == frozenset(
        {"involvement", "other", "lineup"}
    )


def test_cache_dir_rejects_bad_key_and_idempotent_save(tmp_path: Path) -> None:
    """Invalid keys raise; identical re-save is a no-op; meta is write-once."""
    with pytest.raises(ValueError, match="64 lowercase hex"):
        component_cache_dir(tmp_path, "not-a-digest")
    key = "a" * 64
    dest = save_component(tmp_path, key, {"x": 1}, meta={"slot": "score"})
    assert (dest / "meta.json").exists()
    # Identical payload returns the same path without raising.
    again = save_component(tmp_path, key, {"x": 1}, meta={"slot": "ignored"})
    assert again == dest
    assert load_component(tmp_path, key) == {"x": 1}
    # Miss path.
    assert load_component(tmp_path, "b" * 64) is None


def test_precomputed_unknown_slot_raises(tmp_path: Path) -> None:
    """CompositePredictor.fit rejects unknown precomputed slots."""
    store = build_league(DataStore(tmp_path / "league"))
    view = PITData(store).view(_AS_OF)
    model = CompositePredictor(_layered_small())
    with pytest.raises(ValueError, match="unknown component slots"):
        model.fit(view, precomputed={"not_a_slot": object()})


def test_component_hashes_require_all_slots() -> None:
    """component_hashes_for_spec refuses a incomplete layered spec."""
    base = _layered_small()
    broken = base.model_copy(update={"components": {"score": base.components["score"]}})
    with pytest.raises(KeyError, match="missing components"):
        component_hashes_for_spec(broken)
    assert len(component_hashes_for_spec(base)) == 4
    assert all(len(h) == 64 for h in component_hashes_for_spec(base).values())
    assert (
        spec_hash(base.components["score"]) == component_hashes_for_spec(base)["score"]
    )
