"""Stage C experiment storage: status folding, arm reuse, frozen gates."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest
from pydantic import ValidationError

from fplt.schemas.l2 import SCHEMA_VERSION
from fplt.schemas.manifests import (
    AsOfDeadline,
    BatchManifest,
    ExperimentArm,
    ExperimentEvent,
    ExperimentManifest,
    ExperimentVerdict,
    ExperimentWindow,
)
from fplt.store.experiments import (
    append_experiment_event,
    arm_reuse_key,
    bind_arm_run,
    experiment_status,
    find_reusable_arm,
    fold_experiment_status,
    frozen_gates,
    hash_eval_spec_payload,
    load_experiment_manifest,
    mint_batch_id,
    mint_experiment_id,
    resolve_or_create_arm_run,
    submit_experiment,
    write_batch_manifest,
    write_experiment_results,
)

_TS = datetime(2026, 11, 26, 12, 0, 0, tzinfo=UTC)


def _arm(name: str = "control", *, role: str = "control") -> ExperimentArm:
    return ExperimentArm(
        arm=name,
        role=role,  # type: ignore[arg-type]
        spec_hash="a" * 64,
        spec_label=name,
    )


def _window() -> ExperimentWindow:
    return ExperimentWindow(role="primary", seasons=["2024-25"], from_gw=1, to_gw=10)


def _deadlines() -> list[AsOfDeadline]:
    return [
        AsOfDeadline(season="2024-25", gw=1, deadline_utc=_TS),
        AsOfDeadline(season="2024-25", gw=10, deadline_utc=_TS),
    ]


def test_mint_experiment_and_batch_ids() -> None:
    exp = mint_experiment_id(when=_TS, suffix="aaaaaa")
    batch = mint_batch_id(when=_TS, suffix="bbbbbb")
    assert exp == "exp_20261126T120000Z_aaaaaa"
    assert batch == "batch_20261126T120000Z_bbbbbb"


def test_experiment_manifest_ab_requires_eval_hash() -> None:
    with pytest.raises(ValidationError, match="eval_spec"):
        ExperimentManifest(
            experiment_id="exp_x",
            flavour="ab",
            question="Does x help?",
            arms=[_arm()],
            windows=[_window()],
            horizon=1,
            gates={},
            code_sha="abc",
            lockfile_hash="b" * 64,
            submitted_by="test",
            host="test",
            submitted_at=_TS,
        )


def test_fold_status_planned_running_analysed() -> None:
    folded = fold_experiment_status([], n_arms=2)
    assert folded.status == "planned"
    submitted = [ExperimentEvent(event="submitted", at=_TS)]
    folded = fold_experiment_status(submitted, n_arms=2)
    assert folded.status == "running"
    assert folded.label == "running (0/2 arms)"
    events = [
        *submitted,
        ExperimentEvent(event="arm_succeeded", at=_TS, arm="control", run_id="r1"),
        ExperimentEvent(event="arm_succeeded", at=_TS, arm="treatment", run_id="r2"),
        ExperimentEvent(event="analysed", at=_TS),
    ]
    folded = fold_experiment_status(events, n_arms=2)
    assert folded.status == "analysed"
    assert folded.arms_succeeded == 2


def test_fold_status_out_of_order_and_duplicates() -> None:
    """Arrival order and duplicate (event, arm) pairs must not change the fold."""
    out_of_order = [
        ExperimentEvent(event="analysed", at=_TS),
        ExperimentEvent(event="arm_succeeded", at=_TS, arm="b", run_id="r2"),
        ExperimentEvent(event="submitted", at=_TS),
        ExperimentEvent(event="arm_succeeded", at=_TS, arm="a", run_id="r1"),
        # duplicates
        ExperimentEvent(event="arm_succeeded", at=_TS, arm="a", run_id="r1"),
        ExperimentEvent(event="submitted", at=_TS),
    ]
    folded = fold_experiment_status(out_of_order, n_arms=2)
    assert folded.status == "analysed"
    assert folded.arms_succeeded == 2

    aborted = [
        ExperimentEvent(event="arm_succeeded", at=_TS, arm="a", run_id="r1"),
        ExperimentEvent(event="aborted", at=_TS),
        ExperimentEvent(event="analysed", at=_TS),
    ]
    assert fold_experiment_status(aborted, n_arms=2).status == "aborted"


def test_fold_status_failed_when_all_arms_terminal_with_failure() -> None:
    events = [
        ExperimentEvent(event="submitted", at=_TS),
        ExperimentEvent(event="arm_succeeded", at=_TS, arm="control", run_id="r1"),
        ExperimentEvent(event="arm_failed", at=_TS, arm="treatment", detail="boom"),
    ]
    folded = fold_experiment_status(events, n_arms=2)
    assert folded.status == "failed"


def test_arm_reuse_exact_match_only(tmp_path: Path) -> None:
    key = arm_reuse_key(
        spec_hash="a" * 64,
        code_sha="abc1234",
        lockfile_hash="b" * 64,
        build_id="build_1",
        deadlines=_deadlines(),
        horizon=3,
        seeds={"base": 7},
    )
    run_id, reused = resolve_or_create_arm_run(
        tmp_path,
        key=key,
        spec_label="control",
        model_id="m1",
        experiment_id="exp_demo",
        arm="control",
    )
    assert reused is False
    hit = find_reusable_arm(tmp_path, key)
    assert hit == run_id

    # Any single-field mismatch is a miss.
    for kwargs in (
        {"spec_hash": "c" * 64},
        {"code_sha": "deadbeef"},
        {"lockfile_hash": "d" * 64},
        {"build_id": "build_2"},
        {"build_id": None},
        {"horizon": 4},
        {"seeds": {"base": 8}},
    ):
        base = {
            "spec_hash": "a" * 64,
            "code_sha": "abc1234",
            "lockfile_hash": "b" * 64,
            "build_id": "build_1",
            "deadlines": _deadlines(),
            "horizon": 3,
            "seeds": {"base": 7},
        }
        base.update(kwargs)
        assert find_reusable_arm(tmp_path, arm_reuse_key(**base)) is None

    # Exact match reuses.
    again, reused_again = resolve_or_create_arm_run(
        tmp_path,
        key=key,
        spec_label="control",
        model_id="m1",
    )
    assert reused_again is True
    assert again == run_id


def test_gates_frozen_at_submit(tmp_path: Path) -> None:
    gates = {
        "min_coverage": 0.8,
        "allow_low_coverage": False,
        "coverage": {"metric": "odds_join", "read": "strict_pit"},
        "keep_if": {"prediction_any": ["rmse"], "components_any": []},
        "drop_if": {"coverage_below_min": True, "mean_ppg_gate_fail": True},
    }
    eval_hash = hash_eval_spec_payload(
        {"name": "demo_ab", "gates": gates, "question": "q"}
    )
    manifest = submit_experiment(
        tmp_path,
        flavour="ab",
        question="Does coverage gate freeze?",
        arms=[_arm("control"), _arm("treatment", role="treatment")],
        windows=[_window()],
        horizon=1,
        code_sha="abc",
        lockfile_hash="b" * 64,
        gates=gates,
        eval_spec="demo_ab",
        eval_spec_hash=eval_hash,
        submitted_by="test",
        host="test",
        experiment_id="exp_20261126T120000Z_gate01",
        submitted_at=_TS,
    )
    assert frozen_gates(manifest)["min_coverage"] == 0.8

    # Mutating the in-memory gates dict after submit must not change disk.
    gates["min_coverage"] = 0.1
    reloaded = load_experiment_manifest(tmp_path, manifest.experiment_id)
    assert reloaded.gates["min_coverage"] == 0.8
    assert reloaded.eval_spec_hash == eval_hash
    # A different EvalSpec payload hashes differently; the frozen hash stays.
    mutated_hash = hash_eval_spec_payload(
        {"name": "demo_ab", "gates": gates, "question": "q"}
    )
    assert mutated_hash != eval_hash
    assert reloaded.eval_spec_hash != mutated_hash

    status = experiment_status(tmp_path, manifest.experiment_id)
    assert status.status == "running"
    assert status.label == "running (0/2 arms)"


def test_submit_bind_results_and_batch(tmp_path: Path) -> None:
    manifest = submit_experiment(
        tmp_path,
        flavour="comparison",
        question="H=3 xPts A vs B",
        arms=[_arm("a"), _arm("b", role="treatment")],
        windows=[_window()],
        horizon=3,
        code_sha="abc",
        lockfile_hash="b" * 64,
        gates={},
        seeds={"base": 1},
        as_of=_deadlines(),
        submitted_by="test",
        host="test",
        experiment_id="exp_20261126T120000Z_cmp001",
        submitted_at=_TS,
    )
    key = arm_reuse_key(
        spec_hash="a" * 64,
        code_sha="abc",
        lockfile_hash="b" * 64,
        build_id=None,
        deadlines=_deadlines(),
        horizon=3,
        seeds={"base": 1},
    )
    run_a, _ = resolve_or_create_arm_run(
        tmp_path,
        key=key,
        spec_label="a",
        model_id="m_a",
        experiment_id=manifest.experiment_id,
        arm="a",
    )
    bind_arm_run(
        tmp_path,
        manifest.experiment_id,
        arm="a",
        run_id=run_a,
        reused=False,
        at=_TS,
    )
    arm_metrics = pl.DataFrame(
        {
            "arm": ["a"],
            "window": ["primary"],
            "metric": ["rmse"],
            "value": [1.5],
            "se": [None],
            "n": [10],
            "stage": ["prediction"],
            "schema_version": [SCHEMA_VERSION],
        }
    )
    write_experiment_results(
        tmp_path,
        manifest.experiment_id,
        arm_metrics=arm_metrics,
        verdict=ExperimentVerdict(
            overall="inconclusive",
            gates=[],
            computed_at=_TS,
            eval_spec_hash=None,
        ),
        report_md="# report\n",
    )
    append_experiment_event(
        tmp_path,
        manifest.experiment_id,
        ExperimentEvent(event="analysed", at=_TS),
    )
    assert experiment_status(tmp_path, manifest.experiment_id).status == "analysed"

    batch = BatchManifest(
        batch_id="batch_20261126T120000Z_bbbbbb",
        label="nightly",
        experiment_ids=[manifest.experiment_id],
        submitted_at=_TS,
        submitted_by="test",
        host="test",
    )
    write_batch_manifest(tmp_path, batch)
    batch_manifest = tmp_path / "batches" / f"batch={batch.batch_id}" / "manifest.json"
    assert batch_manifest.is_file()


def test_results_registry_accepts_experiment_id() -> None:
    from fplt.backtest.results_registry import PromoteResult

    entry = PromoteResult(
        id="demo",
        date="2026-11-26",
        path="modelspec",
        verdict="keep",
        eval_spec="demo_ab",
        experiment_id="exp_20261126T120000Z_aaaaaa",
    )
    assert entry.experiment_id is not None
