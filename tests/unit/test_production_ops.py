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
    assert [entry.line for entry in ops.decision_lines()] == [
        "v4_h3",
        "v4_h5",
        "v4_h7",
        "v4_h9",
    ]
    assert [entry.line for entry in ops.shadow_predict_lines()] == [
        "ep_next_h1",
        "openfpl_h1",
        "dc_v3_h1",
        "dc_v2_h1",
        "rolling_mean_h1",
        "lgbm_probs_h1",
        "forecast_combine_h1",
    ]
    primary = ops.primary_line()
    assert primary.role == "primary"
    assert primary.squad_mode == "real"
    assert primary.horizon == 3
    assert ops.predict_groups() == {"dc_layered_v4_native": 9}
    assert ops.shadow_predict_groups() == {
        "ep_next_v1": 1,
        "openfpl_v1": 1,
        "dc_layered_v3": 1,
        "dc_layered_v2": 1,
        "rolling_mean_v1": 1,
        "lgbm_probs_v1": 1,
        "forecast_combine_probs_v1": 1,
    }
    for entry in ops.lines:
        if entry.role in ("candidate", "shadow_predict"):
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


def test_empty_env_override_means_unset(tmp_path: Path) -> None:
    """GitHub Actions passes \"\" for unset vars; treat as YAML defaults (#362)."""
    path = tmp_path / "production.yaml"
    path.write_text(
        yaml.safe_dump({"models": ["dc_layered_v4"], "plan_horizon": 3}),
        encoding="utf-8",
    )
    ops = load_production_ops(
        path,
        environ={"FPLT_PRODUCTION_MODELS": "", "FPLT_PLAN_HORIZON": ""},
    )
    assert ops.models == ["dc_layered_v4"]
    assert ops.plan_horizon == 3
    whitespace = load_production_ops(
        path, environ={"FPLT_PRODUCTION_MODELS": "  ", "FPLT_PLAN_HORIZON": "  "}
    )
    assert whitespace.models == ["dc_layered_v4"]
    assert whitespace.plan_horizon == 3
    with pytest.raises(ValueError, match="integer"):
        load_production_ops(path, environ={"FPLT_PLAN_HORIZON": "nope"})
    with pytest.raises(ValueError, match=">= 1"):
        load_production_ops(path, environ={"FPLT_PLAN_HORIZON": "0"})
    with pytest.raises(ValueError, match="non-empty names"):
        load_production_ops(path, environ={"FPLT_PRODUCTION_MODELS": "a,,b"})


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
    with pytest.raises(ValueError, match="non-empty"):
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


def test_shadow_predict_defaults_virtual_and_is_excluded_from_slate() -> None:
    """#389: shadow_predict is predict-only; official_slate omits it."""
    pipeline = live_pipeline_defaults(model="dc_layered_v4", horizon=3)
    ops = ProductionOps(
        models=["dc_layered_v4"],
        plan_horizon=3,
        pipeline=pipeline,
        lines=[
            OfficialLine(
                line="primary",
                spec="dc_layered_v4",
                horizon=3,
                role="primary",
            ),
            OfficialLine(
                line="shadow_rm",
                spec="rolling_mean_v1",
                horizon=1,
                role="shadow_predict",
            ),
        ],
    )
    assert ops.shadow_predict_lines()[0].squad_mode == "virtual"
    assert ops.predict_groups() == {"dc_layered_v4": 3}
    assert ops.shadow_predict_groups() == {"rolling_mean_v1": 1}
    horizons, lines, _cp = ops.official_slate()
    assert horizons == {"dc_layered_v4": 3}
    assert [line.line for line in lines] == ["primary"]


def test_shadow_predict_must_not_extend_shared_decision_horizon() -> None:
    """#389: validator — shadow_predict needs no horizon beyond decision predict."""
    pipeline = live_pipeline_defaults(model="dc_layered_v4", horizon=3)
    with pytest.raises(ValidationError, match="must not extend the decision slate"):
        ProductionOps(
            models=["dc_layered_v4"],
            plan_horizon=3,
            pipeline=pipeline,
            lines=[
                OfficialLine(
                    line="primary",
                    spec="dc_layered_v4",
                    horizon=3,
                    role="primary",
                ),
                OfficialLine(
                    line="shadow_same_spec",
                    spec="dc_layered_v4",
                    horizon=9,
                    role="shadow_predict",
                ),
            ],
        )


def test_yaml_shadow_predict_line_loads(tmp_path: Path) -> None:
    path = tmp_path / "production.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "models": ["dc_layered_v4"],
                "plan_horizon": 3,
                "lines": [
                    {
                        "line": "primary",
                        "spec": "dc_layered_v4",
                        "horizon": 3,
                        "role": "primary",
                    },
                    {
                        "line": "rm_h1",
                        "spec": "rolling_mean_v1",
                        "horizon": 1,
                        "role": "shadow_predict",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )
    ops = load_production_ops(path, environ={})
    assert len(ops.shadow_predict_lines()) == 1
    assert ops.shadow_predict_lines()[0].line == "rm_h1"


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


def test_min_run_interval_and_max_build_age_validation(tmp_path: Path) -> None:
    path = tmp_path / "production.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "models": ["dc_layered_v4"],
                "plan_horizon": 3,
                "min_run_interval_minutes": True,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="min_run_interval_minutes"):
        load_production_ops(path, environ={})
    path.write_text(
        yaml.safe_dump(
            {
                "models": ["dc_layered_v4"],
                "plan_horizon": 3,
                "max_build_age_minutes": 0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="max_build_age_minutes"):
        load_production_ops(path, environ={})


def test_committed_catch_up_scope_defaults() -> None:
    """#369: catch-up starts at GW1 of the current season, bounded per tick."""
    ops = load_production_ops()
    assert ops.catch_up_from_gw == 1
    assert ops.max_catch_up_per_tick == 4


def test_background_quiet_minutes_default_and_validation(tmp_path: Path) -> None:
    """#374: quiet window before checkpoints (default 45)."""
    ops = load_production_ops()
    assert ops.background_quiet_minutes == 45
    path = tmp_path / "production.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "models": ["dc_layered_v4"],
                "plan_horizon": 3,
                "background_quiet_minutes": 0,
            }
        ),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="background_quiet_minutes"):
        load_production_ops(path, environ={})


def test_catch_up_scope_parsed_and_kept_by_env_override(tmp_path: Path) -> None:
    path = tmp_path / "production.yaml"
    path.write_text(
        yaml.safe_dump(
            {
                "models": ["dc_layered_v4"],
                "plan_horizon": 3,
                "catch_up_from_gw": 10,
                "max_catch_up_per_tick": 2,
            }
        ),
        encoding="utf-8",
    )
    ops = load_production_ops(path, environ={})
    assert (ops.catch_up_from_gw, ops.max_catch_up_per_tick) == (10, 2)
    # Emergency env override rebuilds ProductionOps; scope must survive it.
    over = load_production_ops(path, environ={"FPLT_PLAN_HORIZON": "5"})
    assert (over.catch_up_from_gw, over.max_catch_up_per_tick) == (10, 2)


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("catch_up_from_gw", 0),
        ("catch_up_from_gw", 39),
        ("catch_up_from_gw", True),
        ("max_catch_up_per_tick", 0),
    ],
)
def test_catch_up_scope_rejects_invalid_values(
    tmp_path: Path, field: str, value: object
) -> None:
    path = tmp_path / "production.yaml"
    path.write_text(
        yaml.safe_dump({"models": ["dc_layered_v4"], "plan_horizon": 3, field: value}),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match=field):
        load_production_ops(path, environ={})
