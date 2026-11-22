"""Tests for promote-eval equivalence kinds (environment / refactor)."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import patch

import polars as pl
import pytest
import yaml
from pydantic import ValidationError

from fplt.backtest.equivalence import (
    EquivalenceTolerance,
    compare_prediction_frames,
    diff_lockfiles,
    direct_dependency_names,
    equivalence_verdict,
    lockfile_package_versions,
    produce_arm_local,
)
from fplt.backtest.promote_eval import (
    execute_promote_eval,
    load_eval_spec,
    plan_promote_eval,
    preflight_promote_eval,
)
from fplt.models.spec import ModelSpec


def _pred_row(
    *,
    player_key: int = 1,
    fixture_id: int = 10,
    xpts: float = 4.0,
    p_haul: float = 0.2,
) -> dict[str, Any]:
    return {
        "as_of_season": "2024-25",
        "as_of_gw": 1,
        "target_season": "2024-25",
        "target_gw": 1,
        "player_key": player_key,
        "fixture_id": fixture_id,
        "xpts": xpts,
        "xpts_sd": 1.0,
        "p_haul": p_haul,
        "p_60": 0.5,
        "model_id": "m",
    }


def _env_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": "env_demo",
        "kind": "environment",
        "question": "Does a polars upgrade keep forecasts identical?",
        "change": {
            "lever": "dependency_upgrade",
            "packages": ["polars"],
            "control_ref": "control-ref",
            "treatment_ref": "treatment-ref",
            "diff": ["uv.lock: polars"],
        },
        "control": "dc_layered_v2",
        "treatment": "dc_layered_v2",
        "window": {
            "seasons": ["2024-25"],
            "from_gw": 1,
            "to_gw": 2,
            "modes": ["walk_forward"],
        },
        "suites": {"prediction": True, "components": False, "decisions": False},
        "equivalence": {
            "tolerance": {"xpts_abs": 1.0e-6, "prob_abs": 1.0e-9},
            "compare": "predictions",
        },
    }
    payload.update(overrides)
    return payload


def _write_eval(folder: Path, name: str, payload: dict[str, object]) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    (folder / f"{name}.yaml").write_text(yaml.safe_dump(payload), encoding="utf-8")


def test_lockfile_package_versions_and_direct_names(tmp_path: Path) -> None:
    lock = tmp_path / "uv.lock"
    lock.write_text(
        '[[package]]\nname = "numpy"\nversion = "2.0.0"\n\n'
        '[[package]]\nname = "polars"\nversion = "1.0.0"\n',
        encoding="utf-8",
    )
    assert lockfile_package_versions(lock) == {"numpy": "2.0.0", "polars": "1.0.0"}
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "x"\ndependencies = ["numpy>=2", "polars>=1"]\n',
        encoding="utf-8",
    )
    assert direct_dependency_names(pyproject) == {"numpy", "polars"}


def test_diff_lockfiles_missing_direct_package(tmp_path: Path) -> None:
    control = tmp_path / "c.lock"
    treatment = tmp_path / "t.lock"
    control.write_text(
        '[[package]]\nname = "numpy"\nversion = "1.26.0"\n\n'
        '[[package]]\nname = "colorama"\nversion = "0.4.0"\n',
        encoding="utf-8",
    )
    treatment.write_text(
        '[[package]]\nname = "numpy"\nversion = "2.0.0"\n\n'
        '[[package]]\nname = "colorama"\nversion = "0.4.1"\n',
        encoding="utf-8",
    )
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text(
        '[project]\nname = "x"\ndependencies = ["numpy>=1"]\n',
        encoding="utf-8",
    )
    diff = diff_lockfiles(control, treatment, pyproject=pyproject, declared_packages=[])
    assert not diff.identical
    assert diff.missing_packages == ("numpy",)
    assert any(item.startswith("colorama:") for item in diff.transitive_changed)

    declared = diff_lockfiles(
        control, treatment, pyproject=pyproject, declared_packages=["numpy"]
    )
    assert declared.missing_packages == ()


def test_compare_prediction_frames_identical_and_beyond() -> None:
    left = pl.DataFrame([_pred_row(), _pred_row(player_key=2, xpts=5.0)])
    right = left.clone()
    tol = EquivalenceTolerance()
    same = compare_prediction_frames(left, right, tol)
    assert same.identical
    assert same.n_rows == 2
    assert same.n_beyond_tolerance == 0

    moved = pl.DataFrame(
        [_pred_row(), _pred_row(player_key=2, xpts=5.0 + 1.0e-3, p_haul=0.2)]
    )
    diff = compare_prediction_frames(left, moved, tol)
    assert not diff.identical
    assert diff.n_beyond_tolerance >= 1
    assert diff.max_abs_diff is not None and diff.max_abs_diff >= 1.0e-3
    assert diff.top_offenders


def test_compare_prediction_frames_within_tolerance_is_identical() -> None:
    left = pl.DataFrame([_pred_row(xpts=4.0)])
    right = pl.DataFrame([_pred_row(xpts=4.0 + 5.0e-7)])
    tol = EquivalenceTolerance(xpts_abs=1.0e-6, prob_abs=1.0e-9)
    report = compare_prediction_frames(left, right, tol)
    assert report.identical
    assert report.n_beyond_tolerance == 0


def test_equivalence_verdict_order() -> None:
    identical = compare_prediction_frames(
        pl.DataFrame([_pred_row()]),
        pl.DataFrame([_pred_row()]),
        EquivalenceTolerance(),
    )
    assert equivalence_verdict(identical, interval_drop=False) == "keep"
    assert (
        equivalence_verdict(identical, interval_drop=True, force_inconclusive=True)
        == "inconclusive"
    )

    different = compare_prediction_frames(
        pl.DataFrame([_pred_row(xpts=1.0)]),
        pl.DataFrame([_pred_row(xpts=2.0)]),
        EquivalenceTolerance(),
    )
    assert equivalence_verdict(different, interval_drop=False) == "inconclusive"
    assert equivalence_verdict(different, interval_drop=True) == "drop"


def test_environment_loader_requires_packages_and_refs(tmp_path: Path) -> None:
    bad = _env_payload()
    cast_change = dict(bad["change"])  # type: ignore[arg-type]
    cast_change.pop("packages")
    bad["change"] = cast_change
    _write_eval(tmp_path, "env_bad", bad)
    with pytest.raises(ValidationError, match="packages"):
        load_eval_spec("env_bad", directory=tmp_path)

    refactor = {
        "name": "ref_demo",
        "kind": "refactor",
        "question": "Does the refactor keep forecasts identical?",
        "change": {
            "lever": "no_behaviour_change",
            "control_ref": "a",
            "treatment_ref": "b",
            "diff": ["refactor only"],
        },
        "control": "dc_layered_v2",
        "treatment": "dc_layered_v2",
    }
    _write_eval(tmp_path, "ref_demo", refactor)
    spec = load_eval_spec("ref_demo", directory=tmp_path)
    assert spec.kind == "refactor"
    assert spec.control == spec.treatment


def test_equivalence_rejects_differing_model_names() -> None:
    payload = _env_payload(treatment="dc_layered_v1")
    with pytest.raises(ValidationError, match="control == treatment"):
        from fplt.backtest.promote_eval import EvalSpec

        EvalSpec.model_validate(payload)


def test_plan_forces_prediction_only_for_environment(tmp_path: Path) -> None:
    payload = _env_payload()
    change = dict(payload["change"])  # type: ignore[arg-type]
    payload["change"] = change
    payload["suites"] = {
        "prediction": False,
        "components": True,
        "decisions": True,
    }
    _write_eval(tmp_path, "env_demo", payload)
    spec = load_eval_spec("env_demo", directory=tmp_path)
    plan = plan_promote_eval(spec, dry_run=True)
    assert plan.suites.prediction is True
    assert plan.suites.components is False
    assert plan.suites.decisions is False
    assert any("worktree" in cmd for cmd in plan.commands)


def test_environment_preflight_identical_lockfiles_fail(tmp_path: Path) -> None:
    payload = _env_payload()
    _write_eval(tmp_path, "env_demo", payload)
    spec = load_eval_spec("env_demo", directory=tmp_path)
    plan = plan_promote_eval(spec, dry_run=True)
    lock_text = '[[package]]\nname = "numpy"\nversion = "1.0.0"\n'
    with (
        patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False),
        patch(
            "fplt.backtest.promote_eval.resolve_git_sha",
            side_effect=["aaa", "bbb"],
        ),
        patch(
            "fplt.backtest.promote_eval._git_show_file",
            return_value=lock_text,
        ),
    ):
        result = preflight_promote_eval(spec, plan, data_dir=tmp_path)
    assert not result.ok
    assert any("lockfiles are identical" in item for item in result.failures)


def test_environment_preflight_missing_package(tmp_path: Path) -> None:
    payload = _env_payload()
    change = dict(payload["change"])  # type: ignore[arg-type]
    # polars is a direct dep in repo pyproject; omit it from packages.
    change["packages"] = ["scipy"]
    payload["change"] = change
    _write_eval(tmp_path, "env_demo", payload)
    spec = load_eval_spec("env_demo", directory=tmp_path)
    plan = plan_promote_eval(spec, dry_run=True)
    control_lock = '[[package]]\nname = "polars"\nversion = "1.0.0"\n'
    treatment_lock = '[[package]]\nname = "polars"\nversion = "2.0.0"\n'
    with (
        patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False),
        patch(
            "fplt.backtest.promote_eval.resolve_git_sha",
            side_effect=["aaa", "bbb"],
        ),
        patch(
            "fplt.backtest.promote_eval._git_show_file",
            side_effect=[control_lock, treatment_lock],
        ),
    ):
        result = preflight_promote_eval(spec, plan, data_dir=tmp_path)
    assert not result.ok
    assert any("missing from change.packages" in item for item in result.failures)


def test_refactor_preflight_rejects_lock_diff(tmp_path: Path) -> None:
    payload = {
        "name": "ref_demo",
        "kind": "refactor",
        "question": "Does the refactor keep forecasts identical?",
        "change": {
            "lever": "no_behaviour_change",
            "control_ref": "a",
            "treatment_ref": "b",
            "diff": ["refactor"],
        },
        "control": "dc_layered_v2",
        "treatment": "dc_layered_v2",
        "window": {"seasons": ["2024-25"], "from_gw": 1, "to_gw": 2},
    }
    _write_eval(tmp_path, "ref_demo", payload)
    spec = load_eval_spec("ref_demo", directory=tmp_path)
    plan = plan_promote_eval(spec, dry_run=True)
    with (
        patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False),
        patch(
            "fplt.backtest.promote_eval.resolve_git_sha",
            side_effect=["aaa", "bbb"],
        ),
        patch(
            "fplt.backtest.promote_eval._git_show_file",
            side_effect=[
                '[[package]]\nname = "numpy"\nversion = "1.0.0"\n',
                '[[package]]\nname = "numpy"\nversion = "2.0.0"\n',
            ],
        ),
    ):
        result = preflight_promote_eval(spec, plan, data_dir=tmp_path)
    assert not result.ok
    assert any("lockfile differs" in item for item in result.failures)


def test_execute_equivalence_keep(tmp_path: Path) -> None:
    payload = _env_payload()
    _write_eval(tmp_path, "env_demo", payload)
    spec = load_eval_spec("env_demo", directory=tmp_path)
    out = tmp_path / "out"
    frame = pl.DataFrame([_pred_row(), _pred_row(player_key=2)])

    def _fake_produce(worktree: Path, **kwargs: Any) -> dict[str, Any]:
        out_parquet = Path(kwargs["out_parquet"])
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(out_parquet)
        return {"rows": frame.height, "cache_hit": False}

    with (
        patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False),
        patch(
            "fplt.backtest.promote_eval.resolve_git_sha",
            side_effect=["aaa", "bbb", "aaa", "bbb"],
        ),
        patch(
            "fplt.backtest.promote_eval._git_show_file",
            side_effect=[
                '[[package]]\nname = "polars"\nversion = "1.0.0"\n',
                '[[package]]\nname = "polars"\nversion = "2.0.0"\n',
            ],
        ),
        patch("fplt.backtest.promote_eval.add_worktree"),
        patch("fplt.backtest.promote_eval.sync_frozen"),
        patch("fplt.backtest.promote_eval.remove_worktree"),
        patch(
            "fplt.backtest.promote_eval.produce_ledger_in_worktree",
            side_effect=_fake_produce,
        ),
        patch(
            "fplt.backtest.promote_eval.interval_drop_for_frames",
            return_value=(False, []),
        ),
    ):
        result = execute_promote_eval(
            spec,
            data_dir=tmp_path,
            dry_run=False,
            output_dir=out,
            argv=["promote-eval", "--execute"],
        )
    assert result.verdict == "keep"
    assert result.exit_code == 0
    assert result.equivalence_diff is not None
    assert result.equivalence_diff.identical
    rec = (out / "recommendation.md").read_text(encoding="utf-8")
    assert "question_type: equivalence" in rec
    assert "**keep**" in rec
    assert "predictions compared:" in rec


def test_execute_equivalence_drop_on_bad_interval(tmp_path: Path) -> None:
    payload = _env_payload()
    _write_eval(tmp_path, "env_demo", payload)
    spec = load_eval_spec("env_demo", directory=tmp_path)
    out = tmp_path / "out"
    control = pl.DataFrame([_pred_row(xpts=1.0)])
    treatment = pl.DataFrame([_pred_row(xpts=3.0)])
    frames = {"control": control, "treatment": treatment}

    def _fake_produce(worktree: Path, **kwargs: Any) -> dict[str, Any]:
        role = "control" if "control" in str(kwargs["out_parquet"]) else "treatment"
        out_parquet = Path(kwargs["out_parquet"])
        out_parquet.parent.mkdir(parents=True, exist_ok=True)
        frames[role].write_parquet(out_parquet)
        return {"rows": 1, "cache_hit": False}

    with (
        patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False),
        patch(
            "fplt.backtest.promote_eval.resolve_git_sha",
            side_effect=["aaa", "bbb", "aaa", "bbb"],
        ),
        patch(
            "fplt.backtest.promote_eval._git_show_file",
            side_effect=[
                '[[package]]\nname = "polars"\nversion = "1.0.0"\n',
                '[[package]]\nname = "polars"\nversion = "2.0.0"\n',
            ],
        ),
        patch("fplt.backtest.promote_eval.add_worktree"),
        patch("fplt.backtest.promote_eval.sync_frozen"),
        patch("fplt.backtest.promote_eval.remove_worktree"),
        patch(
            "fplt.backtest.promote_eval.produce_ledger_in_worktree",
            side_effect=_fake_produce,
        ),
        patch(
            "fplt.backtest.promote_eval.interval_drop_for_frames",
            return_value=(True, [{"metric": "mae", "entirely_bad": True}]),
        ),
    ):
        result = execute_promote_eval(
            spec,
            data_dir=tmp_path,
            dry_run=False,
            output_dir=out,
        )
    assert result.verdict == "drop"
    assert result.exit_code == 0
    rec = (out / "recommendation.md").read_text(encoding="utf-8")
    assert "**drop**" in rec


def test_produce_arm_local_writes_parquet(tmp_path: Path) -> None:
    spec = ModelSpec(name="rolling_mean_v1", kind="rolling_mean", seed=0)
    # Avoid walking real data: mock produce_ledger.
    fake = pl.DataFrame([_pred_row()])
    with patch(
        "fplt.backtest.equivalence.produce_ledger",
    ) as produce:
        from fplt.backtest.produce_ledger import LedgerCacheKey, ProduceLedgerResult

        key = LedgerCacheKey(
            resolved_spec_sha256="a" * 64,
            uv_lock_sha256="b" * 64,
            data_hashes={},
            season="2024-25",
            test_from_gw=1,
            test_to_gw=1,
            fit_gw=1,
            mode="walk_forward",
            steps=1,
            seasons_back=4,
            n_sims=None,
            seed=0,
        )
        produce.return_value = ProduceLedgerResult(
            frame=fake,
            spec=spec,
            key=key,
            cache_dir=tmp_path / "cache",
            cache_hit=False,
        )
        out = tmp_path / "arm.parquet"
        meta = produce_arm_local(
            data_dir=tmp_path,
            spec=spec,
            season="2024-25",
            mode="walk_forward",
            test_from_gw=1,
            test_to_gw=1,
            fit_gw=1,
            steps=1,
            seasons_back=4,
            out_parquet=out,
            use_cache=False,
            lock_root=tmp_path,
        )
    assert out.is_file()
    assert meta["rows"] == 1
    assert (tmp_path / "arm.meta.json").is_file()


def test_repo_equivalence_examples_load() -> None:
    for name in (
        "example_environment_lock_upgrade",
        "example_refactor_no_behaviour",
    ):
        spec = load_eval_spec(name)
        assert spec.control == spec.treatment
        assert spec.change.control_ref
        assert spec.change.treatment_ref
