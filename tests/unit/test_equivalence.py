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
    assert diff.max_abs_diff is not None
    assert diff.max_abs_diff >= 1.0e-3
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
    from fplt.backtest.promote_eval import EvalSpec

    payload = _env_payload(treatment="dc_layered_v1")
    with pytest.raises(ValidationError, match="control == treatment"):
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


def test_is_equivalence_kind_and_lock_edge_cases(tmp_path: Path) -> None:
    from fplt.backtest.equivalence import (
        direct_dependency_names,
        is_equivalence_kind,
        lock_sha_at,
        lockfile_package_versions,
    )

    assert is_equivalence_kind("environment")
    assert is_equivalence_kind("refactor")
    assert not is_equivalence_kind("feature")
    with pytest.raises(FileNotFoundError):
        lockfile_package_versions(tmp_path / "missing.lock")
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text("name = 'x'\n", encoding="utf-8")
    assert direct_dependency_names(pyproject) == set()
    pyproject.write_text("[project]\nname = 'x'\n", encoding="utf-8")
    assert direct_dependency_names(pyproject) == set()
    pyproject.write_text(
        '[project]\nname = "x"\ndependencies = [1, "numpy>=1", "", "  "]\n',
        encoding="utf-8",
    )
    assert direct_dependency_names(pyproject) == {"numpy"}
    # lock_sha_at via uv_lock_sha256
    lock = tmp_path / "uv.lock"
    lock.write_text('[[package]]\nname = "a"\nversion = "1"\n', encoding="utf-8")
    assert len(lock_sha_at(tmp_path)) == 64


def test_diff_lockfiles_missing_files(tmp_path: Path) -> None:
    pyproject = tmp_path / "pyproject.toml"
    pyproject.write_text('[project]\nname = "x"\ndependencies = []\n', encoding="utf-8")
    diff = diff_lockfiles(
        tmp_path / "c.lock",
        tmp_path / "t.lock",
        pyproject=pyproject,
        declared_packages=[],
    )
    assert diff.control_sha256 == "missing"
    assert diff.treatment_sha256 == "missing"
    assert diff.identical


def test_compare_row_count_mismatch_and_empty() -> None:
    tol = EquivalenceTolerance()
    left = pl.DataFrame([_pred_row()])
    right = pl.DataFrame([_pred_row(), _pred_row(player_key=2)])
    report = compare_prediction_frames(left, right, tol)
    assert not report.identical
    assert report.top_offenders[0]["reason"] == "row_count_mismatch"
    empty = pl.DataFrame([_pred_row()]).clear()
    # Need schema-compatible empty — rebuild
    empty = left.clear()
    same_empty = compare_prediction_frames(empty, empty, tol)
    assert same_empty.identical
    assert same_empty.n_rows == 0


def test_compare_missing_required_columns() -> None:
    from fplt.backtest.equivalence import compare_prediction_frames

    bad = pl.DataFrame({"xpts": [1.0]})
    with pytest.raises(ValueError, match="missing columns"):
        compare_prediction_frames(bad, bad, EquivalenceTolerance())


def test_equivalence_verdict_and_load_arm_spec(tmp_path: Path) -> None:
    from fplt.backtest.equivalence import load_arm_spec

    # Real model name from configs
    spec = load_arm_spec("rolling_mean_v1", models_dir=None)
    assert spec.name == "rolling_mean_v1"
    with pytest.raises(KeyError):
        load_arm_spec("does_not_exist_zzz", models_dir=None)
    # Worktree configs dir
    models = tmp_path / "configs" / "models"
    models.mkdir(parents=True)
    (models / "toy.yaml").write_text(
        "name: toy\nkind: rolling_mean\nparams:\n  window: 3\nseed: 0\n",
        encoding="utf-8",
    )
    loaded = load_arm_spec("toy", models_dir=None, worktree=tmp_path)
    assert loaded.name == "toy"


def test_interval_drop_for_frames_paths() -> None:
    from fplt.backtest.equivalence import interval_drop_for_frames

    control = pl.DataFrame([_pred_row(xpts=1.0)])
    treatment = pl.DataFrame([_pred_row(xpts=3.0)])
    drop, intervals = interval_drop_for_frames(
        control, treatment, control_id="c", treatment_id="t", actuals=None
    )
    assert drop is False
    assert intervals == []
    empty_act = pl.DataFrame(
        {
            "target_season": pl.Series([], dtype=pl.Utf8),
            "target_gw": pl.Series([], dtype=pl.Int64),
            "player_key": pl.Series([], dtype=pl.Int64),
            "fixture_id": pl.Series([], dtype=pl.Int64),
            "actual_pts": pl.Series([], dtype=pl.Int64),
        }
    )
    drop2, _ = interval_drop_for_frames(
        control, treatment, control_id="c", treatment_id="t", actuals=empty_act
    )
    assert drop2 is False
    # All-null actuals
    null_act = pl.DataFrame(
        {
            "target_season": ["2024-25"],
            "target_gw": [1],
            "player_key": [1],
            "fixture_id": [10],
            "actual_pts": [None],
        }
    )
    drop3, ints3 = interval_drop_for_frames(
        control, treatment, control_id="c", treatment_id="t", actuals=null_act
    )
    assert drop3 is False
    # Real actuals → compute intervals (may or may not drop)
    act = pl.DataFrame(
        {
            "target_season": ["2024-25"],
            "target_gw": [1],
            "player_key": [1],
            "fixture_id": [10],
            "actual_pts": [2],
            "position": ["MID"],
        }
    )
    drop4, ints4 = interval_drop_for_frames(
        control, treatment, control_id="c", treatment_id="t", actuals=act
    )
    assert isinstance(drop4, bool)
    assert isinstance(ints4, list)


def test_resolve_git_sha_and_worktree_helpers(tmp_path: Path) -> None:
    from unittest.mock import patch

    from fplt.backtest.equivalence import (
        add_worktree,
        produce_ledger_in_worktree,
        remove_worktree,
        resolve_git_sha,
        sync_frozen,
    )

    with patch("fplt.backtest.equivalence.subprocess.run") as run:
        run.return_value = type(
            "R", (), {"stdout": "abc123\n", "returncode": 0, "stderr": ""}
        )()
        assert resolve_git_sha("HEAD", cwd=tmp_path) == "abc123"
        run.assert_called()

    dest = tmp_path / "wt"
    with patch("fplt.backtest.equivalence.subprocess.run") as run:
        run.return_value = type(
            "R", (), {"returncode": 0, "stdout": "", "stderr": ""}
        )()
        path = add_worktree(tmp_path, "HEAD", dest)
        assert path == dest
    # Exists → error
    dest.mkdir(parents=True, exist_ok=True)
    with pytest.raises(FileExistsError):
        add_worktree(tmp_path, "HEAD", dest)

    # remove success
    with patch("fplt.backtest.equivalence.subprocess.run") as run:
        run.return_value = type(
            "R", (), {"returncode": 0, "stdout": "", "stderr": ""}
        )()
        remove_worktree(tmp_path, dest)
    # remove missing path no-op
    remove_worktree(tmp_path, tmp_path / "nope")
    # remove with CalledProcessError → rmtree fallback
    again = tmp_path / "wt2"
    again.mkdir()
    with patch("fplt.backtest.equivalence.subprocess.run") as run:
        import subprocess as sp

        run.side_effect = [
            sp.CalledProcessError(1, "git"),
            type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})(),
        ]
        remove_worktree(tmp_path, again)

    with patch("fplt.backtest.equivalence.subprocess.run") as run:
        run.return_value = type(
            "R", (), {"returncode": 0, "stdout": "", "stderr": ""}
        )()
        sync_frozen(tmp_path)

    out = tmp_path / "arm.parquet"
    with patch("fplt.backtest.equivalence.subprocess.run") as run:
        run.return_value = type(
            "R",
            (),
            {
                "returncode": 1,
                "stdout": "",
                "stderr": "boom",
            },
        )()
        with pytest.raises(RuntimeError, match="produce-ledger subprocess failed"):
            produce_ledger_in_worktree(
                tmp_path,
                data_dir=tmp_path,
                model_name="m",
                season="2024-25",
                mode="walk_forward",
                test_from_gw=1,
                test_to_gw=1,
                fit_gw=None,
                steps=1,
                seasons_back=None,
                out_parquet=out,
                use_cache=False,
            )

    # Success via meta.json
    with patch("fplt.backtest.equivalence.subprocess.run") as run:

        def _ok(*_a, **_k):
            meta = out.with_suffix(".meta.json")
            meta.write_text('{"rows": 1, "cache_hit": false}\n', encoding="utf-8")
            return type("R", (), {"returncode": 0, "stdout": "", "stderr": ""})()

        run.side_effect = _ok
        meta = produce_ledger_in_worktree(
            tmp_path,
            data_dir=tmp_path,
            model_name="m",
            season="2024-25",
            mode="walk_forward",
            test_from_gw=1,
            test_to_gw=1,
            fit_gw=1,
            steps=1,
            seasons_back=2,
            out_parquet=out,
            use_cache=True,
            models_dir=tmp_path / "models",
        )
        assert meta["rows"] == 1

    # Success via stdout JSON line when meta missing
    out2 = tmp_path / "arm2.parquet"
    with patch("fplt.backtest.equivalence.subprocess.run") as run:
        run.return_value = type(
            "R",
            (),
            {
                "returncode": 0,
                "stdout": '{"rows": 3}\n',
                "stderr": "",
            },
        )()
        # Remove meta if created
        meta_path = out2.with_suffix(".meta.json")
        if meta_path.exists():
            meta_path.unlink()
        meta2 = produce_ledger_in_worktree(
            tmp_path,
            data_dir=tmp_path,
            model_name="m",
            season="2024-25",
            mode="fixed",
            test_from_gw=1,
            test_to_gw=1,
            fit_gw=1,
            steps=1,
            seasons_back=None,
            out_parquet=out2,
            use_cache=False,
        )
        assert meta2["rows"] == 3

    # No metadata at all
    out3 = tmp_path / "arm3.parquet"
    with patch("fplt.backtest.equivalence.subprocess.run") as run:
        run.return_value = type(
            "R", (), {"returncode": 0, "stdout": "\n", "stderr": ""}
        )()
        with pytest.raises(RuntimeError, match="no metadata"):
            produce_ledger_in_worktree(
                tmp_path,
                data_dir=tmp_path,
                model_name="m",
                season="2024-25",
                mode="walk_forward",
                test_from_gw=1,
                test_to_gw=1,
                fit_gw=None,
                steps=1,
                seasons_back=None,
                out_parquet=out3,
                use_cache=False,
            )


def test_attach_actuals_helpers() -> None:
    from fplt.backtest import equivalence as mod

    preds = pl.DataFrame([_pred_row()])
    actuals = pl.DataFrame(
        {
            "target_season": ["2024-25"],
            "target_gw": [1],
            "player_key": [1],
            "fixture_id": [10],
            "actual_pts": [5],
        }
    )
    joined = mod._attach_actuals(preds, actuals)
    assert "actual_pts" in joined.columns
    assert joined["run_kind"][0] == "equivalence"
    assert joined["horizon"][0] == 1
    assert joined["position"][0] == "MID"
    assert joined["actual_minutes"][0] is None


def test_attach_actuals_preserves_existing_columns() -> None:
    from fplt.backtest import equivalence as mod

    preds = pl.DataFrame([_pred_row()]).with_columns(
        pl.lit("custom").alias("run_kind"),
        pl.lit(2).alias("horizon"),
        pl.lit("FWD").alias("position"),
        pl.lit(90).alias("actual_minutes"),
    )
    actuals = pl.DataFrame(
        {
            "target_season": ["2024-25"],
            "target_gw": [1],
            "player_key": [1],
            "fixture_id": [10],
            "actual_pts": [5],
            "position": ["MID"],
        }
    )
    joined = mod._attach_actuals(preds, actuals)
    assert joined["run_kind"][0] == "custom"
    assert joined["horizon"][0] == 2
