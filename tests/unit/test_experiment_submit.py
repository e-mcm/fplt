"""Batch / comparison submit: shared-control dedup and CLI surface."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest
import yaml

from fplt.backtest.promote_eval import EvalSpec
from fplt.cli import main
from fplt.store.experiment_submit import (
    load_eval_ref,
    submit_batch,
    submit_comparison,
)
from fplt.store.experiments import (
    experiment_status,
    list_experiment_ids,
    load_experiment_manifest,
)

_TS = datetime(2026, 11, 26, 14, 0, 0, tzinfo=UTC)


def _write_eval(
    folder: Path,
    *,
    name: str,
    control: str,
    treatment: str,
    question: str = "Does the lever help?",
) -> Path:
    payload = {
        "name": name,
        "kind": "config",
        "question": question,
        "change": {
            "lever": "hyperparam",
            "allow_multi_lever": True,
            "diff": ["test"],
        },
        "control": control,
        "treatment": treatment,
        "window": {
            "seasons": ["2024-25"],
            "from_gw": 1,
            "to_gw": 10,
            "modes": ["walk_forward"],
        },
        "suites": {"prediction": True, "steps": 1},
        "gates": {
            "min_coverage": 0.5,
            "keep_if": {"prediction_any": ["rmse"]},
            "drop_if": {
                "coverage_below_min": True,
                "mean_ppg_gate_fail": True,
            },
        },
    }
    path = folder / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def test_batch_deduplicates_shared_control(tmp_path: Path) -> None:
    evals = tmp_path / "evals"
    evals.mkdir()
    # Both EvalSpecs share rolling_mean_v1 as control and the same window.
    _write_eval(
        evals,
        name="shared_a",
        control="rolling_mean_v1",
        treatment="lgbm_direct_v1",
        question="A vs rolling?",
    )
    _write_eval(
        evals,
        name="shared_b",
        control="rolling_mean_v1",
        treatment="dc_layered_v1",
        question="B vs rolling?",
    )
    specs = [
        load_eval_ref("shared_a", evals_dir=evals),
        load_eval_ref("shared_b", evals_dir=evals),
    ]
    result = submit_batch(
        tmp_path / "data",
        specs,
        batch_label="nightly",
        code_sha="abc1234",
        lockfile_hash="b" * 64,
        build_id=None,
        submitted_at=_TS,
        batch_id="batch_20261126T140000Z_bbbbbb",
        host="test",
        submitted_by="test",
    )
    assert result.batch_id == "batch_20261126T140000Z_bbbbbb"
    assert len(result.experiment_ids) == 2
    # Four arm slots (2 experiments × 2 arms) but shared control is one fresh run.
    assert result.fresh_runs == 3
    assert result.reused_runs == 1
    control_binds = [a for a in result.arms if a.arm == "control"]
    assert len(control_binds) == 2
    assert control_binds[0].reused is False
    assert control_binds[1].reused is True
    assert control_binds[0].run_id == control_binds[1].run_id

    batch_manifest = (
        tmp_path / "data" / "batches" / f"batch={result.batch_id}" / "manifest.json"
    )
    assert batch_manifest.is_file()
    for exp_id in result.experiment_ids:
        manifest = load_experiment_manifest(tmp_path / "data", exp_id)
        assert manifest.batch_id == result.batch_id
        assert manifest.flavour == "ab"
        assert manifest.gates["min_coverage"] == 0.5
        assert experiment_status(tmp_path / "data", exp_id).status == "running"


def test_submit_comparison_pins_build_and_no_gates(tmp_path: Path) -> None:
    result = submit_comparison(
        tmp_path,
        ["rolling_mean_v1", "lgbm_direct_v1"],
        horizon=10,
        season="2024-25",
        gw=5,
        code_sha="abc1234",
        lockfile_hash="b" * 64,
        build_id="build_demo",
        submitted_at=_TS,
        experiment_id="exp_20261126T140000Z_cmp001",
        host="test",
        submitted_by="test",
    )
    assert result.experiment_id == "exp_20261126T140000Z_cmp001"
    assert result.build_id == "build_demo"
    assert result.fresh_runs == 2
    assert result.reused_runs == 0
    manifest = load_experiment_manifest(tmp_path, result.experiment_id)
    assert manifest.flavour == "comparison"
    assert manifest.gates == {}
    assert manifest.eval_spec is None
    assert manifest.horizon == 10
    assert manifest.build_id == "build_demo"
    assert "H=10" in manifest.question


def test_load_eval_ref_path_and_stem(tmp_path: Path) -> None:
    _write_eval(
        tmp_path,
        name="path_eval",
        control="rolling_mean_v1",
        treatment="lgbm_direct_v1",
    )
    from_path = load_eval_ref(str(tmp_path / "path_eval.yaml"))
    from_stem = load_eval_ref("path_eval", evals_dir=tmp_path)
    assert isinstance(from_path, EvalSpec)
    assert from_path.name == from_stem.name == "path_eval"


def test_cli_exp_submit_batch(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    evals = tmp_path / "evals"
    evals.mkdir()
    data = tmp_path / "data"
    data.mkdir()
    _write_eval(
        evals,
        name="cli_a",
        control="rolling_mean_v1",
        treatment="lgbm_direct_v1",
    )
    _write_eval(
        evals,
        name="cli_b",
        control="rolling_mean_v1",
        treatment="dc_layered_v1",
    )
    monkeypatch.chdir(tmp_path)
    main(
        [
            "exp",
            "submit",
            str(evals / "cli_a.yaml"),
            str(evals / "cli_b.yaml"),
            "--batch",
            "nightly",
            "--data-dir",
            str(data),
        ]
    )
    ids = list_experiment_ids(data)
    assert len(ids) == 2
    batches = list((data / "batches").glob("batch=*"))
    assert len(batches) == 1


def test_cli_compare_spec(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data = tmp_path / "data"
    data.mkdir()
    monkeypatch.chdir(tmp_path)
    main(
        [
            "compare",
            "--spec",
            "rolling_mean_v1",
            "--spec",
            "lgbm_direct_v1",
            "--horizon",
            "3",
            "--data-dir",
            str(data),
        ]
    )
    ids = list_experiment_ids(data)
    assert len(ids) == 1
    manifest = load_experiment_manifest(data, ids[0])
    assert manifest.flavour == "comparison"
    assert manifest.horizon == 3


def test_cli_compare_requires_season_or_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(tmp_path)
    with pytest.raises(SystemExit) as exc:
        main(["compare", "--models", "rolling_mean_v1"])
    assert exc.value.code == 2
