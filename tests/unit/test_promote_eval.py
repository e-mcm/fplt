"""Tests for EvalSpec loading and promote-eval dry-run planning."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from fplt.backtest.promote_eval import (
    EvalSpec,
    format_plan,
    load_eval_spec,
    plan_promote_eval,
    plan_to_json,
)
from fplt.cli import main


def _write_eval(folder: Path, name: str, payload: dict[str, object]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _base_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": "demo_swap",
        "kind": "component",
        "description": "minutes swap",
        "change": {
            "lever": "component_swap",
            "slot": "minutes",
            "control_kind": "empirical_minutes_v2",
            "treatment_kind": "lgbm_minutes",
            "diff": ["components.minutes.kind: empirical_minutes_v2 → lgbm_minutes"],
        },
        "control": "dc_layered_v2",
        "treatment": "dc_layered_minutes_lgbm_v1",
        "window": {
            "seasons": ["2024-25"],
            "from_gw": 1,
            "to_gw": 10,
            "modes": ["walk_forward"],
        },
        "suites": {
            "prediction": True,
            "components": True,
            "decisions": False,
        },
    }
    payload.update(overrides)
    return payload


def test_eval_spec_rejects_kind_lever_mismatch() -> None:
    with pytest.raises(ValidationError):
        EvalSpec.model_validate(
            _base_payload(
                kind="model",
                change={
                    "lever": "component_swap",
                    "slot": "minutes",
                    "diff": ["bad"],
                },
            )
        )


def test_component_swap_requires_slot() -> None:
    with pytest.raises(ValidationError):
        EvalSpec.model_validate(
            _base_payload(
                change={
                    "lever": "component_swap",
                    "slot": None,
                    "diff": ["x"],
                }
            )
        )


def test_model_fork_forbids_slot() -> None:
    with pytest.raises(ValidationError):
        EvalSpec.model_validate(
            _base_payload(
                name="fork",
                kind="model",
                change={
                    "lever": "model_fork",
                    "slot": "minutes",
                    "diff": ["kind swap"],
                },
                control="rolling_mean_v1",
                treatment="lgbm_direct_v1",
                suites={"prediction": True, "components": False},
            )
        )


def test_data_table_requires_tables() -> None:
    with pytest.raises(ValidationError):
        EvalSpec.model_validate(
            _base_payload(
                name="odds",
                kind="data",
                change={
                    "lever": "data_table",
                    "tables": [],
                    "diff": ["match_odds"],
                },
                treatment="dc_layered_odds_v1",
            )
        )


def test_load_and_plan_dry_run(tmp_path: Path) -> None:
    _write_eval(tmp_path, "demo_swap", _base_payload())
    spec = load_eval_spec("demo_swap", directory=tmp_path)
    plan = plan_promote_eval(spec, dry_run=True, tier="S1")
    assert plan.eval_name == "demo_swap"
    assert plan.lever == "component_swap"
    assert plan.to_gw == 10
    assert any("eval-components" in cmd for cmd in plan.commands)
    assert any("compare" in cmd and "dc_layered_v2" in cmd for cmd in plan.commands)
    assert any(
        "ModelSpec 'dc_layered_minutes_lgbm_v1' not found" in w for w in plan.warnings
    )
    text = format_plan(plan)
    assert "component_swap" in text
    assert "dry-run" in text
    payload = plan_to_json(plan)
    assert '"eval_name": "demo_swap"' in payload


def test_tier_disables_decisions(tmp_path: Path) -> None:
    _write_eval(
        tmp_path,
        "demo_swap",
        _base_payload(
            suites={
                "prediction": True,
                "components": False,
                "decisions": True,
                "decisions_horizon": 3,
                "max_mean_ppg": 90,
            }
        ),
    )
    spec = load_eval_spec("demo_swap", directory=tmp_path)
    plan = plan_promote_eval(spec, dry_run=True, tier="S1")
    assert plan.suites.decisions is False
    assert any("disables decisions" in w for w in plan.warnings)
    assert not any("decisions" in cmd for cmd in plan.commands)


def test_repo_example_evals_load() -> None:
    for name in (
        "example_config_dc_v1_vs_v2",
        "example_component_minutes_v1_vs_v2",
        "example_model_lgbm_vs_rolling",
        "example_data_odds_prior",
    ):
        spec = load_eval_spec(name)
        plan = plan_promote_eval(spec, dry_run=True)
        assert plan.eval_name == name
        assert plan.commands


def test_cli_promote_eval_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "promote-eval",
            "--eval",
            "example_model_lgbm_vs_rolling",
            "--dry-run",
            "--tier",
            "S1",
        ]
    )
    out = capsys.readouterr().out
    assert "promote-eval example_model_lgbm_vs_rolling" in out
    assert "model_fork" in out
    assert "fplt compare" in out
    assert "dry-run" in out


def test_cli_promote_eval_execute_exits(
    capsys: pytest.CaptureFixture[str],
) -> None:
    with pytest.raises(SystemExit) as exc:
        main(
            [
                "promote-eval",
                "--eval",
                "example_model_lgbm_vs_rolling",
                "--execute",
            ]
        )
    assert exc.value.code == 2
    err = capsys.readouterr().err
    assert "not implemented" in err


def test_cli_promote_eval_missing_eval() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["promote-eval", "--eval", "does_not_exist_zz"])
    assert exc.value.code == 2
