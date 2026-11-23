"""Production ops YAML + env overrides for unattended planning."""

from pathlib import Path

import pytest
import yaml

from fplt.automation.production import (
    ProductionOps,
    default_production_path,
    load_production_ops,
)


def test_committed_production_yaml_defaults() -> None:
    """Live automation defaults stay dc_layered_v3 at horizon 3."""
    ops = load_production_ops()
    assert ops.models == ["dc_layered_v3"]
    assert ops.plan_horizon == 3
    assert default_production_path().is_file()
    assert default_production_path().name == "production.yaml"


def test_env_overrides_models_and_horizon(tmp_path: Path) -> None:
    path = tmp_path / "production.yaml"
    path.write_text(
        yaml.safe_dump({"models": ["dc_layered_v3"], "plan_horizon": 3}),
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


def test_env_can_override_one_field_only(tmp_path: Path) -> None:
    path = tmp_path / "production.yaml"
    path.write_text(
        yaml.safe_dump({"models": ["dc_layered_v3"], "plan_horizon": 3}),
        encoding="utf-8",
    )
    models_only = load_production_ops(
        path, environ={"FPLT_PRODUCTION_MODELS": "rolling_mean_v1"}
    )
    assert models_only.models == ["rolling_mean_v1"]
    assert models_only.plan_horizon == 3
    horizon_only = load_production_ops(path, environ={"FPLT_PLAN_HORIZON": "8"})
    assert horizon_only.models == ["dc_layered_v3"]
    assert horizon_only.plan_horizon == 8


def test_empty_env_override_raises(tmp_path: Path) -> None:
    path = tmp_path / "production.yaml"
    path.write_text(
        yaml.safe_dump({"models": ["dc_layered_v3"], "plan_horizon": 3}),
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
        ProductionOps(models=["dc_layered_v3", "  "], plan_horizon=3)
