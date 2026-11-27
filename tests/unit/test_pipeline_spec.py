"""Stage B11: PipelineSpec schema, hash, and live-ops reproduction."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from fplt.automation.production import (
    ProductionOps,
    load_production_ops,
    load_production_pipeline,
)
from fplt.models.pipeline_spec import (
    DataSpec,
    EnvSpec,
    PipelineSpec,
    PlannerSpec,
    PolicySpec,
    live_pipeline_defaults,
    pipeline_hash,
    resolve_pipeline_model,
    resolved_pipeline_payload,
)
from fplt.models.spec import ModelSpec, load_model_specs
from fplt.models.spec_identity import spec_hash
from fplt.optimise.candidate_prune import DEFAULT_TARGET_POOL
from fplt.optimise.decisions import (
    DEFAULT_LIVE_CHIP_ARM,
    DEFAULT_LIVE_PRICE_FORECAST_METHOD,
)
from fplt.optimise.highs_options import DEFAULT_HIGHS_OPTIONS
from fplt.optimise.plan import DEFAULT_PLAN_HORIZON
from fplt.optimise.reference_board import load_reference_board
from fplt.optimise.state import load_squad_state


def test_pipeline_hash_excludes_name_and_expands_model() -> None:
    """Hash ignores pipeline name; model name expands to resolved ModelSpec."""
    left = live_pipeline_defaults(model="dc_layered_v4_native", name="a")
    right = live_pipeline_defaults(model="dc_layered_v4_native", name="b")
    assert pipeline_hash(left) == pipeline_hash(right)
    payload = resolved_pipeline_payload(left)
    assert "name" not in payload
    assert isinstance(payload["model"], dict)
    assert "name" not in payload["model"]
    model = resolve_pipeline_model(left)
    assert (
        payload["model"]
        == resolved_pipeline_payload(PipelineSpec(model=model))["model"]
    )
    assert pipeline_hash(left) != spec_hash(model)


def test_pipeline_hash_changes_when_planner_or_policy_changes() -> None:
    """Planner / policy patches change the pipeline identity."""
    base = live_pipeline_defaults(model="dc_layered_v4_native")
    longer = base.model_copy(
        update={"planner": base.planner.model_copy(update={"horizon": 9})}
    )
    no_force = base.model_copy(
        update={
            "policy": base.policy.model_copy(
                update={
                    "chips": base.policy.chips.model_copy(update={"force_play": False})
                }
            )
        }
    )
    assert pipeline_hash(base) != pipeline_hash(longer)
    assert pipeline_hash(base) != pipeline_hash(no_force)


def test_alias_model_names_share_pipeline_hash_when_sections_match() -> None:
    """Two ModelSpec names with one spec_hash share a pipeline_hash."""
    # Known B8 aliases: dc_layered_v4 == dc_layered_cs_v1 (same resolved model).
    left = live_pipeline_defaults(model="dc_layered_v4", mc_backend="python")
    right = live_pipeline_defaults(model="dc_layered_cs_v1", mc_backend="python")
    assert spec_hash(resolve_pipeline_model(left)) == spec_hash(
        resolve_pipeline_model(right)
    )
    assert pipeline_hash(left) == pipeline_hash(right)


def test_committed_production_pipeline_matches_live_ops() -> None:
    """Production PipelineSpec reproduces today's ops behaviour exactly."""
    ops = load_production_ops()
    pipeline = load_production_pipeline()
    assert ops.pipeline == pipeline
    assert ops.models == ["dc_layered_v4_native"]
    assert ops.plan_horizon == DEFAULT_PLAN_HORIZON
    assert pipeline.model_name == "dc_layered_v4_native"
    assert pipeline.planner.horizon == 3
    assert pipeline.planner.decay == pytest.approx(load_squad_state().decay)
    assert pipeline.planner.pool.target == DEFAULT_TARGET_POOL
    assert pipeline.planner.mip_gap == pytest.approx(DEFAULT_HIGHS_OPTIONS.mip_rel_gap)
    assert pipeline.planner.time_limit_s == pytest.approx(
        float(DEFAULT_HIGHS_OPTIONS.time_limit or 0.0)
    )
    assert pipeline.policy.chips.arm == DEFAULT_LIVE_CHIP_ARM
    assert pipeline.policy.chips.force_play is True
    assert pipeline.policy.lineup_reopt is True
    assert pipeline.policy.price_forecast == DEFAULT_LIVE_PRICE_FORECAST_METHOD
    assert pipeline.policy.bench_weight == pytest.approx(
        load_squad_state().bench_weight
    )
    assert pipeline.policy.hit_min_gain == pytest.approx(
        float(load_squad_state().hit_cost)
    )
    assert pipeline.env.mc_backend == "native"
    model = resolve_pipeline_model(pipeline)
    assert str(model.params.get("mc_backend", "python")).strip().lower() == "native"
    # Reference board (season-points) stays aligned with the same knobs.
    board = load_reference_board()
    assert board.model == pipeline.model_name
    assert board.horizon == pipeline.planner.horizon
    assert board.chip_arm == pipeline.policy.chips.arm
    assert board.force_play == pipeline.policy.chips.force_play
    assert board.lineup_reopt == pipeline.policy.lineup_reopt
    assert board.price_forecast == pipeline.policy.price_forecast
    assert board.bench_weight == pytest.approx(pipeline.policy.bench_weight)


def test_production_pipeline_hash_is_stable() -> None:
    """Committed production pipeline hashes deterministically."""
    pipeline = load_production_pipeline()
    digest = pipeline_hash(pipeline)
    assert len(digest) == 64
    assert digest == pipeline_hash(load_production_pipeline())
    # Defaults-only builder for the same model/horizon matches the file.
    synthesised = live_pipeline_defaults(
        model="dc_layered_v4_native",
        horizon=3,
        mc_backend="native",
    )
    assert pipeline_hash(pipeline) == pipeline_hash(synthesised)


def test_legacy_yaml_without_pipeline_synthesises_live_defaults(
    tmp_path: Path,
) -> None:
    """Short-form models/plan_horizon still loads via live_pipeline_defaults."""
    path = tmp_path / "production.yaml"
    path.write_text(
        yaml.safe_dump({"models": ["dc_layered_v4"], "plan_horizon": 5}),
        encoding="utf-8",
    )
    ops = load_production_ops(path, environ={})
    assert ops.models == ["dc_layered_v4"]
    assert ops.plan_horizon == 5
    assert ops.pipeline.model_name == "dc_layered_v4"
    assert ops.pipeline.planner.horizon == 5
    assert ops.pipeline.policy.chips.arm == DEFAULT_LIVE_CHIP_ARM


def test_pipeline_mismatch_with_models_raises(tmp_path: Path) -> None:
    """YAML where pipeline.model disagrees with models[0] is refused."""
    path = tmp_path / "production.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "models": ["dc_layered_v4_native"],
                "plan_horizon": 3,
                "pipeline": {
                    "model": "dc_layered_v4",
                    "planner": {"horizon": 3},
                },
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="pipeline.model"):
        load_production_ops(path, environ={})


def test_env_override_updates_pipeline(tmp_path: Path) -> None:
    """Env model/horizon overrides rewrite the returned PipelineSpec."""
    path = tmp_path / "production.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "models": ["dc_layered_v4_native"],
                "plan_horizon": 3,
                "pipeline": live_pipeline_defaults(
                    model="dc_layered_v4_native", horizon=3
                ).model_dump(mode="json"),
            }
        ),
        encoding="utf-8",
    )
    ops = load_production_ops(
        path,
        environ={
            "FPLT_PRODUCTION_MODELS": "rolling_mean_v1",
            "FPLT_PLAN_HORIZON": "8",
        },
    )
    assert ops.models == ["rolling_mean_v1"]
    assert ops.plan_horizon == 8
    assert ops.pipeline.model_name == "rolling_mean_v1"
    assert ops.pipeline.planner.horizon == 8


def test_inline_model_spec_is_accepted_for_hashing() -> None:
    """Inline ModelSpec objects hash independently of the nested name."""
    inline_model = ModelSpec(
        name="demo",
        kind="involvement_xg",
        params={"window": 10, "xg_blend": 0.7},
    )
    inline = PipelineSpec(
        data=DataSpec(),
        model=inline_model,
        planner=PlannerSpec(),
        policy=PolicySpec(),
        env=EnvSpec(mc_backend="python"),
    )
    renamed = PipelineSpec(
        data=DataSpec(),
        model=inline_model.model_copy(update={"name": "other"}),
        planner=PlannerSpec(),
        policy=PolicySpec(),
        env=EnvSpec(mc_backend="python"),
    )
    assert pipeline_hash(inline) == pipeline_hash(renamed)
    assert "name" not in resolved_pipeline_payload(inline)["model"]
    with pytest.raises(KeyError, match="missing_model"):
        resolve_pipeline_model(
            PipelineSpec(model="missing_model"),
            specs=load_model_specs(),
        )


def test_production_ops_requires_pipeline_field() -> None:
    """Constructing ProductionOps without a pipeline fails validation."""
    with pytest.raises(ValidationError):
        ProductionOps(  # type: ignore[call-arg]
            models=["dc_layered_v4_native"],
            plan_horizon=3,
        )


def test_committed_production_lines_slate() -> None:
    """B12: committed slate is v4_h3 primary + h5/h7/h9 candidates."""
    ops = load_production_ops()
    assert ops.primary_line().line == "v4_h3"
    assert ops.predict_groups() == {"dc_layered_v4_native": 9}
    assert load_production_pipeline().planner.horizon == 3
