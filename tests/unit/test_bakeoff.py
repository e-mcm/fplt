"""Tests for the thin multi-arm bakeoff wrapper."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import pytest
import yaml
from pydantic import ValidationError

from fplt.backtest.bakeoff import (
    BakeoffSpec,
    bakeoff_from_models,
    execute_bakeoff,
    format_bakeoff_plan,
    load_bakeoff_spec,
    plan_bakeoff,
    render_ranking,
    selection_payloads,
)
from fplt.backtest.compare import CompareResult, CompareRun
from fplt.backtest.promote_eval import (
    EvalSpec,
    compare_arm_names,
    plan_promote_eval,
)
from fplt.cli import main


def _write_bakeoff(folder: Path, name: str, payload: dict[str, object]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _base_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": "demo_championship",
        "question": "Which arm ranks best on the shared window?",
        "control": "rolling_mean_v1",
        "models": ["rolling_mean_v1", "dc_layered_v2", "lgbm_direct_v1"],
        "window": {
            "seasons": ["2024-25"],
            "from_gw": 1,
            "to_gw": 10,
            "modes": ["walk_forward"],
        },
        "suites": {"prediction": True, "components": False, "decisions": False},
    }
    payload.update(overrides)
    return payload


def test_load_bakeoff_spec(tmp_path: Path) -> None:
    _write_bakeoff(tmp_path, "demo_championship", _base_payload())
    spec = load_bakeoff_spec("demo_championship", directory=tmp_path)
    assert spec.control == "rolling_mean_v1"
    assert len(spec.models) == 3


def test_bakeoff_rejects_evalspec_kind(tmp_path: Path) -> None:
    payload = _base_payload()
    payload["kind"] = "model"
    _write_bakeoff(tmp_path, "demo_championship", payload)
    with pytest.raises(ValueError, match="EvalSpec"):
        load_bakeoff_spec("demo_championship", directory=tmp_path)


def test_bakeoff_requires_control_in_models() -> None:
    with pytest.raises(ValidationError, match="control"):
        BakeoffSpec.model_validate(
            _base_payload(control="ep_next_v1", models=["a", "b", "c"])
        )


def test_bakeoff_requires_three_models() -> None:
    with pytest.raises(ValidationError):
        BakeoffSpec.model_validate(
            _base_payload(models=["rolling_mean_v1", "dc_layered_v2"])
        )


def test_bakeoff_rejects_decisions_suite() -> None:
    with pytest.raises(ValidationError, match="decisions"):
        BakeoffSpec.model_validate(
            _base_payload(
                suites={
                    "prediction": True,
                    "components": False,
                    "decisions": True,
                }
            )
        )


def test_plan_bakeoff_lists_compare_and_rank(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    for name in ("rolling_mean_v1", "dc_layered_v2", "lgbm_direct_v1"):
        (models / f"{name}.yaml").write_text(
            yaml.safe_dump(
                {
                    "name": name,
                    "kind": "rolling_mean",
                    "model_id": name,
                    "params": {},
                    "features": [],
                    "train_window": {"seasons_back": 4},
                    "refit": "every_gw",
                }
            ),
            encoding="utf-8",
        )
    # load_model_specs needs real ModelSpec shape — plan only checks names via
    # load_model_specs; stub by patching when specs are incomplete.
    spec = BakeoffSpec.model_validate(_base_payload())
    with patch(
        "fplt.backtest.bakeoff.load_model_specs",
        return_value={
            "rolling_mean_v1": MagicMock(),
            "dc_layered_v2": MagicMock(),
            "lgbm_direct_v1": MagicMock(),
        },
    ):
        plan = plan_bakeoff(spec, dry_run=True, models_dir=models)
    text = format_bakeoff_plan(plan)
    assert "fplt compare" in text
    assert "rolling_mean_v1 dc_layered_v2 lgbm_direct_v1" in text
    assert "ranking.md" in text
    assert "descriptive" in text


def test_bakeoff_from_models_cli_helper() -> None:
    spec = bakeoff_from_models(
        name="cli",
        control="rolling_mean_v1",
        models=["rolling_mean_v1", "dc_layered_v2", "lgbm_direct_v1"],
        seasons=["2024-25"],
        from_gw=1,
        to_gw=10,
    )
    assert spec.name == "cli"
    assert spec.window.to_gw == 10


def test_execute_bakeoff_dry_run(tmp_path: Path) -> None:
    spec = BakeoffSpec.model_validate(_base_payload())
    with patch(
        "fplt.backtest.bakeoff.load_model_specs",
        return_value={
            "rolling_mean_v1": MagicMock(),
            "dc_layered_v2": MagicMock(),
            "lgbm_direct_v1": MagicMock(),
        },
    ):
        result = execute_bakeoff(spec, data_dir=tmp_path, dry_run=True)
    assert result.exit_code == 0
    assert (result.output_dir / "plan.json").is_file()
    assert (result.output_dir / "ranking.md").is_file()


def test_execute_bakeoff_writes_ranking(tmp_path: Path) -> None:
    spec = BakeoffSpec.model_validate(_base_payload())
    summary = pl.DataFrame(
        {
            "model_name": ["rolling_mean_v1", "dc_layered_v2", "lgbm_direct_v1"],
            "mode": ["walk_forward"] * 3,
            "horizon": [1, 1, 1],
            "mae": [1.2, 1.0, 1.1],
            "rmse": [2.0, 1.8, 1.9],
            "spearman": [0.5, 0.6, 0.55],
            "n": [100, 100, 100],
            "model_id": ["rm", "dc", "lg"],
        }
    )

    def _run(name: str, mid: str) -> CompareRun:
        return CompareRun(
            model_name=name,
            mode="walk_forward",
            spec=MagicMock(model_id=mid),
        )

    compare = CompareResult(
        season="2024-25",
        test_from_gw=1,
        test_to_gw=10,
        fit_gw=1,
        seasons_back=4,
        modes=("walk_forward",),
        runs=(
            _run("rolling_mean_v1", "rm"),
            _run("dc_layered_v2", "dc"),
            _run("lgbm_direct_v1", "lg"),
        ),
        summary=summary,
        detail=summary,
        output_dir=tmp_path / "compare",
    )

    with (
        patch(
            "fplt.backtest.bakeoff.load_model_specs",
            return_value={
                "rolling_mean_v1": MagicMock(),
                "dc_layered_v2": MagicMock(),
                "lgbm_direct_v1": MagicMock(),
            },
        ),
        patch("fplt.backtest.bakeoff.compare_models", return_value=compare),
        patch(
            "fplt.backtest.bakeoff.selection_payloads",
            return_value=(
                {
                    "pvalue_consistent": 0.1,
                    "pvalue_lower": 0.05,
                    "pvalue_upper": 0.2,
                    "n_gw": 10,
                    "n_models": 2,
                },
                {
                    "included": ["dc_layered_v2", "lgbm_direct_v1"],
                    "excluded": ["rolling_mean_v1"],
                    "size": 0.05,
                    "n_gw": 10,
                },
            ),
        ),
    ):
        result = execute_bakeoff(spec, data_dir=tmp_path, dry_run=False)

    assert result.exit_code == 0
    ranking = (result.output_dir / "ranking.md").read_text(encoding="utf-8")
    assert "dc_layered_v2" in ranking
    assert "SPA" in ranking
    assert "descriptive" in ranking.lower() or "fresh window" in ranking
    assert (result.output_dir / "summary.json").is_file()
    assert (result.output_dir / "summary.parquet").is_file()


def test_render_ranking_orders_by_mae() -> None:
    spec = BakeoffSpec.model_validate(_base_payload())
    with patch(
        "fplt.backtest.bakeoff.load_model_specs",
        return_value={
            "rolling_mean_v1": MagicMock(),
            "dc_layered_v2": MagicMock(),
            "lgbm_direct_v1": MagicMock(),
        },
    ):
        baked_plan = plan_bakeoff(spec)
    summary = pl.DataFrame(
        {
            "model_name": ["rolling_mean_v1", "dc_layered_v2", "lgbm_direct_v1"],
            "mode": ["walk_forward"] * 3,
            "horizon": [1, 1, 1],
            "mae": [1.2, 0.9, 1.0],
        }
    )
    text = render_ranking(
        spec,
        baked_plan,
        summary=summary,
        spa=None,
        mcs=None,
    )
    # Best MAE first.
    dc_pos = text.index("`dc_layered_v2`")
    lgbm_pos = text.index("`lgbm_direct_v1`")
    rm_pos = text.index("`rolling_mean_v1` MAE")
    assert dc_pos < lgbm_pos < rm_pos


def test_selection_payloads_needs_three_arms() -> None:
    compare = CompareResult(
        season="2024-25",
        test_from_gw=1,
        test_to_gw=2,
        fit_gw=1,
        seasons_back=4,
        modes=("walk_forward",),
        runs=(
            CompareRun(
                model_name="a",
                mode="walk_forward",
                spec=MagicMock(model_id="a"),
            ),
            CompareRun(
                model_name="b",
                mode="walk_forward",
                spec=MagicMock(model_id="b"),
            ),
        ),
        summary=pl.DataFrame(),
        detail=pl.DataFrame(),
        output_dir=Path("/tmp"),
    )
    spa, mcs = selection_payloads(
        data_dir=Path("/tmp"),
        compare=compare,
        control="a",
        models=["a", "b"],
    )
    assert spa is None and mcs is None


def test_evalspec_arms_compare_names() -> None:
    payload = {
        "name": "dose",
        "kind": "config",
        "question": "Dose readout?",
        "change": {
            "lever": "hyperparam",
            "orientation": "opt_in",
            "slot": "score",
            "control_kind": "dixon_coles",
            "treatment_kind": "dixon_coles",
            "allow_multi_lever": True,
            "diff": ["strength: 0 → 1"],
        },
        "control": "dc_layered_v2",
        "treatment": "dc_layered_odds_v1",
        "arms": ["dc_layered_history_v1"],
        "window": {"seasons": ["2024-25"], "from_gw": 1, "to_gw": 10},
    }
    spec = EvalSpec.model_validate(payload)
    assert compare_arm_names(spec) == [
        "dc_layered_v2",
        "dc_layered_odds_v1",
        "dc_layered_history_v1",
    ]
    with patch(
        "fplt.backtest.promote_eval.load_model_specs",
        return_value={
            "dc_layered_v2": MagicMock(),
            "dc_layered_odds_v1": MagicMock(),
            "dc_layered_history_v1": MagicMock(),
        },
    ):
        plan = plan_promote_eval(spec, dry_run=True)
    assert "dc_layered_history_v1" in plan.commands[0]
    assert any("SPA/MCS" in w or "fresh" in w for w in plan.warnings)


def test_evalspec_arms_reject_duplicate_of_control() -> None:
    with pytest.raises(ValidationError, match="duplicates"):
        EvalSpec.model_validate(
            {
                "name": "bad",
                "kind": "config",
                "question": "x",
                "change": {
                    "lever": "hyperparam",
                    "orientation": "opt_in",
                    "allow_multi_lever": True,
                    "diff": ["x"],
                },
                "control": "a",
                "treatment": "b",
                "arms": ["a"],
            }
        )


def test_cli_bakeoff_dry_run(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _write_bakeoff(tmp_path, "demo_championship", _base_payload())
    monkeypatch.setattr(
        "fplt.cli.load_bakeoff_spec",
        lambda name: load_bakeoff_spec(name, directory=tmp_path),
    )
    with patch(
        "fplt.backtest.bakeoff.load_model_specs",
        return_value={
            "rolling_mean_v1": MagicMock(),
            "dc_layered_v2": MagicMock(),
            "lgbm_direct_v1": MagicMock(),
        },
    ):
        assert main(["bakeoff", "--bakeoff", "demo_championship"]) is None
