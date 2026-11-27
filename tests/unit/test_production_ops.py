"""Production ops YAML + env overrides for unattended planning."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from fplt.automation.production import (
    OfficialLine,
    ProductionOps,
    default_production_path,
    load_production_ops,
)
from fplt.models.pipeline_spec import live_pipeline_defaults


def test_committed_production_yaml_defaults() -> None:
    """Live automation defaults stay dc_layered_v4_native at horizon 3."""
    ops = load_production_ops()
    assert ops.models == ["dc_layered_v4_native"]
    assert ops.plan_horizon == 3
    assert ops.min_prior_seasons == 3
    assert default_production_path().is_file()
    assert default_production_path().name == "production.yaml"
    assert [entry.line for entry in ops.lines] == [
        "v4_h3",
        "v4_h5",
        "v4_h7",
        "v4_h9",
    ]
    primary = ops.primary_line()
    assert primary.role == "primary"
    assert primary.squad_mode == "real"
    assert primary.horizon == 3
    assert ops.predict_groups() == {"dc_layered_v4_native": 9}
    for entry in ops.lines:
        if entry.role == "candidate":
            assert entry.squad_mode == "virtual"


def test_production_model_uses_native_mc_backend() -> None:
    """FDB-293: ops production must resolve mc_backend=native (fail-loud)."""
    from fplt.models.spec import load_model_specs

    ops = load_production_ops()
    specs = load_model_specs()
    prod = specs[ops.models[0]]
    assert str(prod.params.get("mc_backend", "python")).strip().lower() == "native"


def test_min_prior_seasons_rejects_invalid_values(tmp_path: Path) -> None:
    path = tmp_path / "production.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "models": ["dc_layered_v4"],
                "plan_horizon": 3,
                "min_prior_seasons": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="min_prior_seasons"):
        load_production_ops(path, environ={})

    path.write_text(
        yaml.safe_dump(
            {
                "models": ["dc_layered_v4"],
                "plan_horizon": 3,
                "min_prior_seasons": 0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="min_prior_seasons"):
        load_production_ops(path, environ={})


def test_env_overrides_models_and_horizon(tmp_path: Path) -> None:
    path = tmp_path / "production.yaml"
    path.write_text(
        yaml.safe_dump({"models": ["dc_layered_v4"], "plan_horizon": 3}),
        encoding="utf-8",
    )
    ops = load_production_ops(
        path,
        environ={
            "FPLT_PRODUCTION_MODELS": "rolling_mean_v1, ep_next_v1",
            "FPLT_PLAN_HORIZON": "5",
        },
    )
    assert ops.models == ["rolling_mean_v1", "ep_next_v1"]
    assert ops.plan_horizon == 5
    assert len(ops.lines) == 1
    assert ops.primary_line().line == "env_override"
    assert ops.primary_line().spec == "rolling_mean_v1"
    assert ops.primary_line().horizon == 5


def test_env_can_override_one_field_only(tmp_path: Path) -> None:
    path = tmp_path / "production.yaml"
    path.write_text(
        yaml.safe_dump({"models": ["dc_layered_v4"], "plan_horizon": 3}),
        encoding="utf-8",
    )
    models_only = load_production_ops(
        path, environ={"FPLT_PRODUCTION_MODELS": "rolling_mean_v1"}
    )
    assert models_only.models == ["rolling_mean_v1"]
    assert models_only.plan_horizon == 3
    assert models_only.primary_line().spec == "rolling_mean_v1"
    horizon_only = load_production_ops(path, environ={"FPLT_PLAN_HORIZON": "8"})
    assert horizon_only.models == ["dc_layered_v4"]
    assert horizon_only.plan_horizon == 8
    assert horizon_only.primary_line().horizon == 8


def test_empty_env_override_raises(tmp_path: Path) -> None:
    path = tmp_path / "production.yaml"
    path.write_text(
        yaml.safe_dump({"models": ["dc_layered_v4"], "plan_horizon": 3}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="FPLT_PRODUCTION_MODELS"):
        load_production_ops(path, environ={"FPLT_PRODUCTION_MODELS": "  "})
    with pytest.raises(ValueError, match="FPLT_PLAN_HORIZON"):
        load_production_ops(path, environ={"FPLT_PLAN_HORIZON": ""})
    with pytest.raises(ValueError, match="integer"):
        load_production_ops(path, environ={"FPLT_PLAN_HORIZON": "nope"})
    with pytest.raises(ValueError, match=">= 1"):
        load_production_ops(path, environ={"FPLT_PLAN_HORIZON": "0"})


def test_missing_or_invalid_yaml_raises(tmp_path: Path) -> None:
    missing = tmp_path / "missing.yaml"
    with pytest.raises(FileNotFoundError, match="production ops"):
        load_production_ops(missing)
    bad = tmp_path / "production.yaml"
    bad.write_text("- not a mapping\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_production_ops(bad)
    empty_models = tmp_path / "empty.yaml"
    empty_models.write_text(
        yaml.safe_dump({"models": [], "plan_horizon": 3}), encoding="utf-8"
    )
    with pytest.raises(ValueError):
        load_production_ops(empty_models)


def test_production_ops_rejects_blank_model_names() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        ProductionOps(
            models=["dc_layered_v4", "  "],
            plan_horizon=3,
            pipeline=live_pipeline_defaults(model="dc_layered_v4", horizon=3),
            lines=[
                OfficialLine(
                    line="primary",
                    spec="dc_layered_v4",
                    horizon=3,
                    role="primary",
                )
            ],
        )


def test_committed_production_exposes_pipeline() -> None:
    """B11: production.yaml carries a PipelineSpec matching models/horizon."""
    ops = load_production_ops()
    assert ops.pipeline.model_name == ops.models[0]
    assert ops.pipeline.planner.horizon == ops.plan_horizon
    assert ops.pipeline.env.mc_backend == "native"


def test_missing_lines_synthesizes_primary(tmp_path: Path) -> None:
    """Legacy YAML without lines: still loads as a single primary line."""
    path = tmp_path / "production.yaml"
    path.write_text(
        yaml.safe_dump({"models": ["dc_layered_v4"], "plan_horizon": 3}),
        encoding="utf-8",
    )
    ops = load_production_ops(path, environ={})
    assert len(ops.lines) == 1
    assert ops.primary_line().spec == "dc_layered_v4"
    assert ops.primary_line().role == "primary"
    assert ops.primary_line().squad_mode == "real"


def test_lines_require_exactly_one_primary_and_unique_names() -> None:
    pipeline = live_pipeline_defaults(model="dc_layered_v4", horizon=3)
    with pytest.raises(ValidationError, match="exactly one primary"):
        ProductionOps(
            models=["dc_layered_v4"],
            plan_horizon=3,
            pipeline=pipeline,
            lines=[
                OfficialLine(
                    line="a", spec="dc_layered_v4", horizon=3, role="candidate"
                ),
                OfficialLine(
                    line="b", spec="dc_layered_v4", horizon=5, role="candidate"
                ),
            ],
        )
    with pytest.raises(ValidationError, match="unique"):
        ProductionOps(
            models=["dc_layered_v4"],
            plan_horizon=3,
            pipeline=pipeline,
            lines=[
                OfficialLine(
                    line="dup", spec="dc_layered_v4", horizon=3, role="primary"
                ),
                OfficialLine(
                    line="dup", spec="dc_layered_v4", horizon=5, role="candidate"
                ),
            ],
        )


def test_primary_line_must_match_short_form() -> None:
    pipeline = live_pipeline_defaults(model="dc_layered_v4", horizon=3)
    with pytest.raises(ValidationError, match="primary line spec"):
        ProductionOps(
            models=["dc_layered_v4"],
            plan_horizon=3,
            pipeline=pipeline,
            lines=[
                OfficialLine(
                    line="x",
                    spec="rolling_mean_v1",
                    horizon=3,
                    role="primary",
                )
            ],
        )
    with pytest.raises(ValidationError, match="primary line horizon"):
        ProductionOps(
            models=["dc_layered_v4"],
            plan_horizon=3,
            pipeline=pipeline,
            lines=[
                OfficialLine(
                    line="x",
                    spec="dc_layered_v4",
                    horizon=5,
                    role="primary",
                )
            ],
        )


def test_primary_rejects_virtual_squad_mode() -> None:
    with pytest.raises(ValidationError, match="squad_mode=real"):
        OfficialLine(
            line="v4_h3",
            spec="dc_layered_v4_native",
            horizon=3,
            role="primary",
            squad_mode="virtual",
        )


def test_yaml_lines_reject_non_mapping_entries(tmp_path: Path) -> None:
    path = tmp_path / "production.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "models": ["dc_layered_v4"],
                "plan_horizon": 3,
                "lines": ["not-a-mapping"],
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="mapping"):
        load_production_ops(path, environ={})
