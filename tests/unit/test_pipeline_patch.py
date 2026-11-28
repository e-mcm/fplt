"""Toolkit 3 / experiments step 2: PipelineSpec patch engine."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from fplt.models.pipeline_patch import (
    PatchDiffError,
    PatchError,
    UnknownPatchPathError,
    apply_patch,
    auto_label,
    expand_grid,
    expected_resolved_diff,
    flat_diff,
    resolved_flat_diff,
    validate_resolved_diff,
)
from fplt.models.pipeline_spec import (
    PipelineSpec,
    live_pipeline_defaults,
    pipeline_hash,
)
from fplt.models.spec import ModelSpec, load_model_specs


def _base(**kwargs: object) -> PipelineSpec:
    """Production-shaped pipeline with a stable python-MC parent model."""
    return live_pipeline_defaults(
        model="dc_layered_v4",
        mc_backend="python",
        name="production",
        **kwargs,  # type: ignore[arg-type]
    )


def test_unknown_path_raises() -> None:
    """An unknown dotted path raises before building an arm."""
    base = _base()
    with pytest.raises(UnknownPatchPathError, match="planner.not_a_field"):
        apply_patch(base, {"planner.not_a_field": 1})
    with pytest.raises(
        UnknownPatchPathError,
        match="model.components.minutes.params.nope",
    ):
        apply_patch(
            base,
            {"model.components.minutes.params.nope": 0.9},
        )


def test_grid_order_does_not_change_hashes() -> None:
    """Cartesian expansion: path key order must not change arm hashes."""
    base = _base()
    grid_a = {
        "planner.horizon": [3, 5],
        "planner.decay": [0.85, 0.9],
    }
    grid_b = {
        "planner.decay": [0.9, 0.85],
        "planner.horizon": [5, 3],
    }
    arms_a = expand_grid(base, grid_a)
    arms_b = expand_grid(base, grid_b)
    hashes_a = {arm.pipeline_hash for arm in arms_a}
    hashes_b = {arm.pipeline_hash for arm in arms_b}
    assert len(arms_a) == 4
    assert hashes_a == hashes_b
    # Each combination is distinct.
    assert len(hashes_a) == 4


def test_two_patches_same_spec_share_hash() -> None:
    """Two patches that resolve to the same pipeline share pipeline_hash."""
    base = _base()
    by_leaf = apply_patch(
        base,
        {"model.components.involvement.params.xg_blend": 0.5},
    )
    # Same leaf change via an equivalent named component fork.
    by_swap = apply_patch(
        base,
        {"model.components.involvement": "involvement_xg_blend_05_v1"},
    )
    assert by_leaf.pipeline_hash == by_swap.pipeline_hash
    assert by_leaf.pipeline_hash == pipeline_hash(by_leaf.pipeline)
    assert by_leaf.pipeline_hash != pipeline_hash(base)


def test_diff_check_catches_hidden_changes() -> None:
    """validate_resolved_diff refuses an arm that changes more than the patch."""
    base = _base()
    honest = apply_patch(base, {"planner.horizon": 9})
    # Tamper: also flip decay, but claim only the horizon patch.
    tampered = honest.pipeline.model_copy(
        update={
            "planner": honest.pipeline.planner.model_copy(update={"decay": 0.5}),
        }
    )
    with pytest.raises(PatchDiffError, match="does not equal"):
        validate_resolved_diff(base, {"planner.horizon": 9}, tampered)

    # Applying only horizon must not touch decay.
    assert honest.pipeline.planner.decay == base.planner.decay
    assert honest.pipeline.planner.horizon == 9


def test_scalar_patch_and_auto_label() -> None:
    """Leaf override changes identity and builds a truthful auto-label."""
    base = _base()
    arm = apply_patch(
        base,
        {"model.components.minutes.params.logistic_blend": 0.8},
    )
    assert arm.label == "production[minutes.logistic_blend=0.8]"
    assert arm.patch == {
        "model.components.minutes.params.logistic_blend": 0.8,
    }
    diff = resolved_flat_diff(base, arm.pipeline)
    assert diff == {
        "model.components.minutes.params.logistic_blend": 0.8,
    }
    assert auto_label("production", {}) == "production"
    assert (
        auto_label("production", {"planner.horizon": 9})
        == "production[planner.horizon=9]"
    )


def test_component_substitution() -> None:
    """Substituting a component by registered name replaces that slot only."""
    base = _base()
    arm = apply_patch(
        base,
        {"model.components.minutes": "empirical_minutes_v2_injuries"},
    )
    assert isinstance(arm.pipeline.model, ModelSpec)
    assert arm.pipeline.model.components["minutes"].name == (
        "empirical_minutes_v2_injuries"
    )
    assert "use_injury_overlay" in (arm.pipeline.model.components["minutes"].params)
    # Score / involvement / other slots unchanged vs base resolved model.
    base_model = (
        base.model
        if isinstance(base.model, ModelSpec)
        else load_model_specs()[base.model_name]
    )
    for slot in ("score", "involvement", "other"):
        assert arm.pipeline.model.components[slot].model_dump(
            mode="json"
        ) == base_model.components[slot].model_dump(mode="json")
    assert arm.label.startswith("production[minutes=empirical_minutes_v2_injuries]")


def test_empty_patch_is_control() -> None:
    """Empty patch is the control arm: same hash, base label."""
    base = _base()
    arm = apply_patch(base, {})
    assert arm.pipeline_hash == pipeline_hash(base)
    assert arm.label == "production"
    assert arm.patch == {}
    assert expand_grid(base, {})[0].pipeline_hash == arm.pipeline_hash


def test_unknown_component_name_raises() -> None:
    """Component substitution with an unregistered name raises KeyError."""
    base = _base()
    with pytest.raises(KeyError, match="no ModelSpec registered"):
        apply_patch(base, {"model.components.minutes": "not_a_real_minutes_spec"})


def test_invalid_leaf_value_fails_validation() -> None:
    """Out-of-range typed values fail PipelineSpec validation."""
    base = _base()
    with pytest.raises(ValidationError):
        apply_patch(base, {"planner.horizon": 0})


def test_expected_diff_matches_apply() -> None:
    """expected_resolved_diff agrees with apply_patch's resolved flat diff."""
    base = _base()
    patch = {
        "planner.horizon": 7,
        "env.mc_backend": "native",
    }
    arm = apply_patch(base, patch)
    assert expected_resolved_diff(base, patch) == resolved_flat_diff(base, arm.pipeline)
    assert flat_diff({"a": 1}, {"a": 2, "b": 3}) == {"a": 2, "b": 3}
    assert flat_diff({"a": {"b": 1}}, {"a": {}}) == {"a.b": None}


def test_invalid_and_empty_paths_raise() -> None:
    """Malformed dotted paths raise UnknownPatchPathError / PatchError."""
    base = _base()
    with pytest.raises(UnknownPatchPathError, match="invalid patch path"):
        apply_patch(base, {"planner..horizon": 9})
    with pytest.raises(PatchError, match="non-empty"):
        apply_patch(base, {"  ": 9})


def test_model_level_substitution_and_inline_spec() -> None:
    """Whole-model name swap and inline ModelSpec substitution both work."""
    base = _base()
    by_name = apply_patch(base, {"model": "dc_layered_v3"})
    assert by_name.pipeline.model_name == "dc_layered_v3"
    assert by_name.pipeline_hash != pipeline_hash(base)

    specs = load_model_specs()
    inline = apply_patch(base, {"model": specs["dc_layered_v3"]})
    assert inline.pipeline_hash == by_name.pipeline_hash
    assert inline.label == "production[model=dc_layered_v3]"


def test_noop_leaf_patch_shares_base_hash() -> None:
    """A patch that sets a leaf to its current value is a control-equivalent arm."""
    base = _base()
    arm = apply_patch(base, {"planner.horizon": base.planner.horizon})
    assert arm.pipeline_hash == pipeline_hash(base)
    assert resolved_flat_diff(base, arm.pipeline) == {}


def test_base_label_override_and_unnamed_pipeline() -> None:
    """Explicit base_label wins; unnamed pipelines fall back to model_name."""
    named = _base()
    arm = apply_patch(
        named,
        {"planner.horizon": 9},
        base_label="prod_line",
    )
    assert arm.label == "prod_line[planner.horizon=9]"

    unnamed = live_pipeline_defaults(model="dc_layered_v4", mc_backend="python")
    assert unnamed.name is None
    control = apply_patch(unnamed, {})
    assert control.label == "dc_layered_v4"


def test_inline_dict_component_and_path_edge_cases() -> None:
    """Inline component dicts resolve; nested-under-scalar paths raise."""
    base = _base()
    specs = load_model_specs()
    inline = specs["empirical_minutes_v2_injuries"].model_dump(mode="json")
    arm = apply_patch(base, {"model.components.minutes": inline})
    model = arm.pipeline.model
    assert isinstance(model, ModelSpec)
    assert model.components["minutes"].params["use_injury_overlay"] is True
    assert "minutes=empirical_minutes_v2_injuries" in arm.label

    class _Dup(dict[str, object]):
        def items(self):  # type: ignore[override]
            yield "planner.horizon", 9
            yield "planner.horizon", 5

    with pytest.raises(PatchError, match="duplicate"):
        apply_patch(base, _Dup())

    with pytest.raises(KeyError, match="non-empty"):
        apply_patch(base, {"model.components.minutes": "   "})

    with pytest.raises(UnknownPatchPathError, match="not a mapping"):
        apply_patch(base, {"planner.horizon.nested": 1})
    with pytest.raises(UnknownPatchPathError, match="data.nope"):
        apply_patch(base, {"data.nope.x": True})

    assert auto_label("x", {"model.components.minutes": "oracle"}) == (
        "x[minutes=oracle]"
    )
    assert auto_label("x", {"model": specs["dc_layered_v3"]}) == (
        "x[model=dc_layered_v3]"
    )
    # Dict without a usable name falls through to str(...).
    assert "minutes=" in auto_label(
        "x",
        {"model.components.minutes": {"kind": "empirical_minutes_v2"}},
    )
