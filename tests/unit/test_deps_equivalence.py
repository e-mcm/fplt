"""Tests for ops-12 runtime deps equivalence helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
import yaml

from fplt.automation.deps_equivalence import (
    FPLT_MC,
    RUNTIME_DEPS_EVAL,
    changed_direct_packages,
    check_prod_freeze,
    format_pr_comment,
    gate_from_verdict,
    prepare_runtime_deps_eval,
    run_deps_equivalence,
    write_runtime_deps_eval,
)
from fplt.backtest.promote_eval import load_eval_spec


def _lock_text(*pairs: tuple[str, str]) -> str:
    blocks = [
        f'[[package]]\nname = "{name}"\nversion = "{ver}"\n' for name, ver in pairs
    ]
    return "\n".join(blocks)


def _write_pyproject(path: Path, deps: list[str]) -> None:
    quoted = ", ".join(f'"{d}"' for d in deps)
    path.write_text(
        f'[project]\nname = "fplt"\ndependencies = [{quoted}]\n',
        encoding="utf-8",
    )


def test_runtime_deps_board_loads() -> None:
    spec = load_eval_spec(RUNTIME_DEPS_EVAL)
    assert spec.kind == "environment"
    assert spec.control == spec.treatment == "dc_layered_v4"
    assert spec.window.to_gw == 3
    assert spec.equivalence is not None


def test_check_prod_freeze_soft_skips_without_ops04() -> None:
    result = check_prod_freeze()
    assert result.state == "soft_skip"
    assert "ops-04" in result.reason or "fplt.ops.freeze" in result.reason


def test_check_prod_freeze_honours_hook() -> None:
    fake = MagicMock()
    fake.in_prod_freeze = MagicMock(return_value=True)

    with patch.dict("sys.modules", {"fplt.ops.freeze": fake, "fplt.ops": MagicMock()}):
        # Ensure importlib finds our fake module path.
        with patch(
            "fplt.automation.deps_equivalence.importlib.import_module",
            return_value=fake,
        ):
            frozen = check_prod_freeze()
    assert frozen.state == "frozen"

    fake.in_prod_freeze = MagicMock(return_value=False)
    with patch(
        "fplt.automation.deps_equivalence.importlib.import_module",
        return_value=fake,
    ):
        open_ = check_prod_freeze()
    assert open_.state == "open"


def test_changed_direct_packages_and_fplt_mc(tmp_path: Path) -> None:
    control = tmp_path / "c.lock"
    treatment = tmp_path / "t.lock"
    control.write_text(
        _lock_text(("polars", "1.0.0"), ("ruff", "0.1.0"), (FPLT_MC, "0.5.0")),
        encoding="utf-8",
    )
    treatment.write_text(
        _lock_text(("polars", "2.0.0"), ("ruff", "0.1.0"), (FPLT_MC, "0.5.0")),
        encoding="utf-8",
    )
    pyproject = tmp_path / "pyproject.toml"
    _write_pyproject(pyproject, ["polars>=1", "fplt-mc==0.5.0"])

    diff = changed_direct_packages(control, treatment, pyproject=pyproject)
    assert diff.packages == ("polars",)
    assert diff.fplt_mc_changed is False
    assert any(line.startswith("polars:") for line in diff.direct_changed)

    treatment.write_text(
        _lock_text(("polars", "2.0.0"), (FPLT_MC, "0.6.0")),
        encoding="utf-8",
    )
    bumped = changed_direct_packages(control, treatment, pyproject=pyproject)
    assert bumped.fplt_mc_changed is True


def test_write_and_prepare_runtime_deps_eval(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_pyproject(repo / "pyproject.toml", ["polars>=1", "numpy>=2"])
    control = tmp_path / "c.lock"
    treatment = tmp_path / "t.lock"
    control.write_text(
        _lock_text(("polars", "1.0.0"), ("numpy", "2.0.0")),
        encoding="utf-8",
    )
    treatment.write_text(
        _lock_text(("polars", "1.1.0"), ("numpy", "2.0.0")),
        encoding="utf-8",
    )
    board = load_eval_spec(RUNTIME_DEPS_EVAL)
    evals = tmp_path / "evals"
    path = write_runtime_deps_eval(
        board=board,
        packages=["polars"],
        control_ref="aaa",
        treatment_ref="bbb",
        evals_dir=evals,
    )
    loaded = load_eval_spec(RUNTIME_DEPS_EVAL, directory=evals)
    assert loaded.change.packages == ["polars"]
    assert loaded.change.control_ref == "aaa"
    assert loaded.change.treatment_ref == "bbb"
    assert path.is_file()

    prep = prepare_runtime_deps_eval(
        repo_root=repo,
        control_ref="base",
        treatment_ref="head",
        control_lock=control,
        treatment_lock=treatment,
        evals_dir=tmp_path / "evals2",
    )
    assert prep.packages == ("polars",)
    rewritten = yaml.safe_load(
        (prep.evals_dir / f"{RUNTIME_DEPS_EVAL}.yaml").read_text(encoding="utf-8")
    )
    assert rewritten["change"]["control_ref"] == "base"
    assert rewritten["change"]["packages"] == ["polars"]


def test_prepare_rejects_fplt_mc_and_identical(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_pyproject(repo / "pyproject.toml", ["polars>=1", "fplt-mc==0.5.0"])
    control = tmp_path / "c.lock"
    treatment = tmp_path / "t.lock"
    text = _lock_text(("polars", "1.0.0"), (FPLT_MC, "0.5.0"))
    control.write_text(text, encoding="utf-8")
    treatment.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="identical"):
        prepare_runtime_deps_eval(
            repo_root=repo,
            control_ref="a",
            treatment_ref="b",
            control_lock=control,
            treatment_lock=treatment,
            evals_dir=tmp_path / "e1",
        )

    treatment.write_text(
        _lock_text(("polars", "1.0.0"), (FPLT_MC, "0.6.0")),
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="fplt-mc"):
        prepare_runtime_deps_eval(
            repo_root=repo,
            control_ref="a",
            treatment_ref="b",
            control_lock=control,
            treatment_lock=treatment,
            evals_dir=tmp_path / "e2",
        )


def test_gate_from_verdict_and_comment() -> None:
    assert gate_from_verdict("keep") == "pass"
    assert gate_from_verdict("inconclusive") == "pass"
    assert gate_from_verdict("drop") == "fail"
    assert gate_from_verdict("tune") == "fail"

    from fplt.automation.deps_equivalence import DepsEquivalenceGate, FreezeCheck

    gate = DepsEquivalenceGate(
        gate="pass",
        verdict="keep",
        summary="ok",
        recommendation="# keep\n",
        freeze=FreezeCheck(state="soft_skip", reason="no ops-04"),
        packages=("polars",),
        control_ref="aaa",
        treatment_ref="bbb",
    )
    body = format_pr_comment(
        gate,
        packages=gate.packages,
        control_ref="aaa",
        treatment_ref="bbb",
    )
    assert "ops-12" in body
    assert "`keep`" in body
    assert "polars" in body
    assert "soft_skip" in body


def test_run_deps_equivalence_skip_and_freeze(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_pyproject(repo / "pyproject.toml", ["polars>=1"])
    control = tmp_path / "c.lock"
    treatment = tmp_path / "t.lock"
    control.write_text(_lock_text(("polars", "1.0.0")), encoding="utf-8")
    treatment.write_text(_lock_text(("polars", "1.1.0")), encoding="utf-8")
    out = tmp_path / "out"

    skipped = run_deps_equivalence(
        repo_root=repo,
        control_ref="base",
        treatment_ref="head",
        data_dir=tmp_path / "data",
        output_dir=out,
        control_lock=control,
        treatment_lock=treatment,
        live=False,
    )
    assert skipped.gate == "skip"
    assert skipped.packages == ("polars",)
    assert (out / "evals" / f"{RUNTIME_DEPS_EVAL}.yaml").is_file()

    with patch(
        "fplt.automation.deps_equivalence.check_prod_freeze",
        return_value=__import__(
            "fplt.automation.deps_equivalence", fromlist=["FreezeCheck"]
        ).FreezeCheck(state="frozen", reason="deadline soon"),
    ):
        blocked = run_deps_equivalence(
            repo_root=repo,
            control_ref="base",
            treatment_ref="head",
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "out2",
            control_lock=control,
            treatment_lock=treatment,
            live=True,
        )
    assert blocked.gate == "fail"
    assert blocked.verdict is None
    assert "freeze" in blocked.summary.lower()


def test_run_deps_equivalence_execute_maps_verdict(tmp_path: Path) -> None:
    repo = tmp_path / "repo"
    repo.mkdir()
    _write_pyproject(repo / "pyproject.toml", ["polars>=1"])
    control = tmp_path / "c.lock"
    treatment = tmp_path / "t.lock"
    control.write_text(_lock_text(("polars", "1.0.0")), encoding="utf-8")
    treatment.write_text(_lock_text(("polars", "1.1.0")), encoding="utf-8")

    fake_result = MagicMock()
    fake_result.verdict = "keep"
    fake_result.exit_code = 0
    fake_result.recommendation = "**keep**\n"

    with patch(
        "fplt.automation.deps_equivalence.execute_promote_eval",
        return_value=fake_result,
    ):
        gate = run_deps_equivalence(
            repo_root=repo,
            control_ref="base",
            treatment_ref="head",
            data_dir=tmp_path / "data",
            output_dir=tmp_path / "out",
            control_lock=control,
            treatment_lock=treatment,
            live=True,
            execute=True,
        )
    assert gate.gate == "pass"
    assert gate.verdict == "keep"
    assert gate.recommendation is not None


def test_write_rejects_empty_and_fplt_mc() -> None:
    board = load_eval_spec(RUNTIME_DEPS_EVAL)
    with pytest.raises(ValueError, match="at least one"):
        write_runtime_deps_eval(
            board=board,
            packages=[],
            control_ref="a",
            treatment_ref="b",
            evals_dir=Path("/tmp"),
        )
    with pytest.raises(ValueError, match="fplt-mc"):
        write_runtime_deps_eval(
            board=board,
            packages=["polars", "fplt-mc"],
            control_ref="a",
            treatment_ref="b",
            evals_dir=Path("/tmp"),
        )


def test_renovate_config_groups_and_fplt_mc() -> None:
    raw: dict[str, Any] = yaml.safe_load(
        # renovate.json is JSON; yaml.safe_load accepts it.
        Path("renovate.json").read_text(encoding="utf-8")
    )
    assert "before 6am on Monday" in raw["schedule"]
    rules = raw["packageRules"]
    fplt_mc = next(r for r in rules if "fplt-mc" in str(r.get("matchPackageNames")))
    assert fplt_mc["enabled"] is False
    runtime = next(r for r in rules if r.get("groupName") == "runtime libraries")
    assert runtime["automerge"] is False
    assert "equivalence" in runtime["labels"]
    assert "deps:runtime" in runtime["labels"]
    dev = next(r for r in rules if r.get("groupName") == "dev tools (patch/minor)")
    assert dev["automerge"] is True
    assert "deps:dev" in dev["labels"]
