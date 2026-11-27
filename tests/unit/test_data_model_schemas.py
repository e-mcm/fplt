"""Stage B2 data-model schema contracts (manifests, outcomes, L2 keys)."""

from __future__ import annotations

import json
from datetime import UTC, datetime

import polars as pl
import pytest
from pydantic import ValidationError

from fplt.schemas.l2 import (
    SCHEMA_VERSION,
    L2AsOfTargetKeys,
    L2StandardKeys,
    L2TeamFixtureKeys,
)
from fplt.schemas.manifests import (
    AppliedRecord,
    AsOfDeadline,
    BatchManifest,
    BuildManifest,
    BuildTableEntry,
    ExperimentArm,
    ExperimentEvent,
    ExperimentManifest,
    ExperimentWindow,
    InterventionRecord,
    OfficialPointer,
    RunManifest,
    SubmittedRecord,
)
from fplt.schemas.outcomes import FixtureOutcomesSchema, PlayerFixtureOutcomesSchema
from fplt.schemas.squad_state_table import SquadStateTableSchema

_TS = datetime(2026, 11, 24, 12, 56, 1, tzinfo=UTC)


def _as_of() -> AsOfDeadline:
    return AsOfDeadline(season="2026-27", gw=12, deadline_utc=_TS)


def _run_kwargs(**overrides: object) -> dict[str, object]:
    base: dict[str, object] = {
        "run_id": "r_20261124T125601Z_7f3a2c",
        "kind": "official",
        "mode": "live",
        "purpose": "GW12 T-90m",
        "run_type": "predict",
        "spec_hash": "a" * 64,
        "spec_label": "dc_layered_v4_native",
        "model_id": "deadbeefdeadbeef",
        "code_sha": "abc1234",
        "lockfile_hash": "b" * 64,
        "build_id": "build_cafe",
        "as_of": [_as_of()],
        "horizon": 9,
        "checkpoint": "T-90m",
        "seeds": {"base": 0},
        "host": "gha",
        "started_at": _TS,
        "finished_at": _TS,
        "status": "succeeded",
    }
    base.update(overrides)
    return base


def test_run_manifest_round_trip_matches_design_example() -> None:
    manifest = RunManifest.model_validate(_run_kwargs(line="v4_h3", plan_horizon=None))
    payload = manifest.model_dump(mode="json")
    restored = RunManifest.model_validate(payload)
    assert restored.run_id == manifest.run_id
    assert restored.as_of[0].season == "2026-27"
    assert restored.status == "succeeded"
    assert json.loads(json.dumps(payload))["spec_label"] == "dc_layered_v4_native"


def test_run_manifest_accepts_legacy_kind() -> None:
    manifest = RunManifest.model_validate(
        _run_kwargs(kind="legacy", code_sha="unknown", spec_hash="unknown")
    )
    assert manifest.kind == "legacy"
    assert manifest.code_sha == "unknown"


def test_run_manifest_allows_pre_build_staged_digests() -> None:
    manifest = RunManifest.model_validate(
        _run_kwargs(
            build_id=None,
            raw_manifest_hash="c" * 64,
            staged_digests={"players": "d" * 64},
            status="running",
            finished_at=None,
        )
    )
    assert manifest.build_id is None
    assert manifest.staged_digests["players"] == "d" * 64


def test_run_manifest_rejects_succeeded_without_finished_at() -> None:
    with pytest.raises(ValidationError, match="finished_at"):
        RunManifest.model_validate(_run_kwargs(finished_at=None, status="succeeded"))


def test_run_manifest_plan_requires_plan_horizon() -> None:
    with pytest.raises(ValidationError, match="plan_horizon"):
        RunManifest.model_validate(_run_kwargs(run_type="plan", plan_horizon=None))


def test_build_manifest_requires_tables() -> None:
    manifest = BuildManifest(
        build_id="build_cafe",
        created_at=_TS,
        staging_version=1,
        code_sha="abc1234",
        raw_manifest_hash="e" * 64,
        raw_cutoff=_TS,
        tables={
            "players": BuildTableEntry(sha="f" * 64, rows=10, schema_hash="g" * 64),
        },
        quarantined_captures=["raw/bad.json.gz"],
    )
    assert manifest.tables["players"].rows == 10
    with pytest.raises(ValidationError):
        BuildManifest(
            build_id="x",
            created_at=_TS,
            staging_version=1,
            code_sha="abc",
            raw_manifest_hash="h" * 64,
            raw_cutoff=_TS,
            tables={},
        )


def test_official_pointer_and_applied_and_intervention() -> None:
    pointer = OfficialPointer(
        predict_run_id="r_predict",
        plan_run_id="r_plan",
        role="primary",
        squad_mode="real",
        written_at=_TS,
    )
    applied = AppliedRecord(
        line="v4_h3",
        plan_run_id="r_plan",
        submitted_at=_TS,
    )
    intervention = InterventionRecord(
        who="eamon",
        intervened_at=_TS,
        what_changed="vetoed transfer Salah out",
        reason="price rise expected",
        line="v4_h3",
        plan_run_id="r_plan",
    )
    assert pointer.role == "primary"
    assert applied.line == "v4_h3"
    assert "Salah" in intervention.what_changed
    submitted = SubmittedRecord(
        line="v4_h3",
        plan_run_id="r_plan",
        submitted_at=_TS,
        entry_id=1,
    )
    assert submitted.entry_id == 1
    with pytest.raises(ValidationError):
        OfficialPointer(
            predict_run_id="r_predict",
            plan_run_id="r_plan",
            role="shadow",  # type: ignore[arg-type]
            squad_mode="real",
            written_at=_TS,
        )


def test_l2_standard_keys_mixin_validates_and_exports_schema_version() -> None:
    assert SCHEMA_VERSION == 1
    frame = pl.DataFrame(
        {
            "run_id": ["r_1"],
            "as_of_season": ["2026-27"],
            "as_of_gw": [12],
            "deadline_utc": [_TS],
            "target_season": ["2026-27"],
            "target_gw": [13],
            "horizon": [1],
            "fixture_id": [100],
            "player_key": [42],
            "team_key": [7],
            "schema_version": [SCHEMA_VERSION],
        }
    )
    checked = L2StandardKeys.validate(frame)
    assert checked["schema_version"].to_list() == [1]
    as_of = frame.drop("player_key", "team_key")
    L2AsOfTargetKeys.validate(as_of)
    team = as_of.with_columns(pl.lit(7).cast(pl.Int64).alias("team_key"))
    L2TeamFixtureKeys.validate(team)


def test_player_fixture_outcomes_schema() -> None:
    frame = pl.DataFrame(
        {
            "season": ["2026-27"],
            "gw": [12],
            "fixture_id": [100],
            "player_key": [42],
            "team_key": [7],
            "pts": [6],
            "minutes": [90],
            "goals": [1],
            "assists": [0],
            "clean_sheets": [0],
            "available_at": [_TS],
            "schema_version": [SCHEMA_VERSION],
            "source": ["fpl_api"],
        }
    )
    checked = PlayerFixtureOutcomesSchema.validate(frame)
    assert checked["pts"].to_list() == [6]


def test_fixture_outcomes_schema() -> None:
    frame = pl.DataFrame(
        {
            "season": ["2026-27"],
            "gw": [12],
            "fixture_id": [100],
            "team_h_key": [1],
            "team_a_key": [2],
            "team_h_score": [2],
            "team_a_score": [1],
            "available_at": [_TS],
            "schema_version": [SCHEMA_VERSION],
            "source": ["fpl_api"],
        }
    )
    checked = FixtureOutcomesSchema.validate(frame)
    assert checked["team_h_score"].to_list() == [2]


def test_squad_state_table_schema_empty_owned() -> None:
    frame = pl.DataFrame(
        {
            "run_id": ["r_plan"],
            "as_of_season": ["2026-27"],
            "as_of_gw": [12],
            "deadline_utc": [_TS],
            "schema_version": [SCHEMA_VERSION],
            "bank": [1000],
            "free_transfers": [1],
            "ft_bank_cap": [5],
            "hit_cost": [4],
            "decay": [1.0],
            "bench_weight": [0.0],
            "owned": ["[]"],
            "chips_remaining": ['{"wildcard":1,"freehit":1,"bboost":1,"3xc":1}'],
            "chips_played": ["[]"],
        }
    )
    checked = SquadStateTableSchema.validate(frame)
    assert checked["bank"].to_list() == [1000]
    assert json.loads(checked["owned"].to_list()[0]) == []


def test_experiment_and_batch_manifest_round_trip() -> None:
    """Stage C experiment / batch manifests validate and round-trip."""
    arm = ExperimentArm(
        arm="control",
        role="control",
        spec_hash="a" * 64,
        spec_label="demo",
    )
    window = ExperimentWindow(role="primary", seasons=["2024-25"], from_gw=1, to_gw=10)
    experiment = ExperimentManifest(
        experiment_id="exp_20261124T125601Z_aaaaaa",
        flavour="ab",
        question="Does the lever help?",
        eval_spec="demo_ab",
        eval_spec_hash="c" * 64,
        arms=[arm],
        windows=[window],
        horizon=3,
        gates={"min_coverage": 0.5},
        code_sha="abc1234",
        lockfile_hash="b" * 64,
        submitted_by="test",
        host="test",
        submitted_at=_TS,
    )
    restored = ExperimentManifest.model_validate(
        json.loads(experiment.model_dump_json())
    )
    assert restored.experiment_id == experiment.experiment_id
    event = ExperimentEvent(event="submitted", at=_TS)
    assert event.event == "submitted"
    batch = BatchManifest(
        batch_id="batch_20261124T125601Z_bbbbbb",
        label="nightly",
        experiment_ids=[experiment.experiment_id],
        submitted_at=_TS,
        submitted_by="test",
        host="test",
    )
    assert batch.experiment_ids == [experiment.experiment_id]
