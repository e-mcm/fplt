"""B12: write-once official pointers, applied / submitted / intervention."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from fplt.automation.production import load_production_ops
from fplt.schemas.manifests import (
    AppliedRecord,
    AsOfDeadline,
    InterventionRecord,
    OfficialPointer,
    SubmittedRecord,
)
from fplt.store.official import (
    LineRunPair,
    OfficialExistsError,
    finalize_official_checkpoint,
    pointer_path,
    write_intervention_record,
    write_official_pointer,
    write_submitted_record,
)
from fplt.store.runs import RunWriter

_TS = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)


def _succeed_run(tmp_path: Path, run_id: str, **extra: object) -> str:
    kwargs: dict[str, object] = {
        "kind": "official",
        "mode": "live",
        "run_type": "predict",
        "spec_hash": "a" * 64,
        "spec_label": "dc_layered_v4_native",
        "model_id": "deadbeef",
        "as_of": [AsOfDeadline(season="2026-27", gw=12, deadline_utc=_TS)],
        "horizon": 9,
        "code_sha": "abc1234",
        "code_dirty": False,
        "lockfile_hash": "b" * 64,
        "host": "test",
        "raw_manifest_hash": "c" * 64,
        "staged_digests": {},
        "started_at": _TS,
        "run_id": run_id,
        "checkpoint": "T-90m",
    }
    kwargs.update(extra)
    with RunWriter(tmp_path, **kwargs) as writer:  # type: ignore[arg-type]
        pass
    return writer.run_id


def test_official_slate_from_committed_production() -> None:
    ops = load_production_ops()
    horizons, lines, checkpoint = ops.official_slate()
    assert checkpoint == "T-90m"
    assert horizons == {"dc_layered_v4_native": 9}
    assert [line.line for line in lines] == ["v4_h3", "v4_h5", "v4_h7", "v4_h9"]


def test_finalize_writes_pointers_and_applied(tmp_path: Path) -> None:
    predict = _succeed_run(tmp_path, "r_20261126T120000Z_pred001", run_type="predict")
    plan_primary = _succeed_run(
        tmp_path,
        "r_20261126T120000Z_plan001",
        run_type="plan",
        plan_horizon=3,
        line="v4_h3",
        squad_mode="real",
        parent_run_id=predict,
        horizon=3,
    )
    plan_cand = _succeed_run(
        tmp_path,
        "r_20261126T120000Z_plan002",
        run_type="plan",
        plan_horizon=5,
        line="v4_h5",
        squad_mode="virtual",
        parent_run_id=predict,
        horizon=5,
    )
    pairs = [
        LineRunPair(
            line="v4_h3",
            role="primary",
            squad_mode="real",
            predict_run_id=predict,
            plan_run_id=plan_primary,
        ),
        LineRunPair(
            line="v4_h5",
            role="candidate",
            squad_mode="virtual",
            predict_run_id=predict,
            plan_run_id=plan_cand,
        ),
    ]
    written = finalize_official_checkpoint(
        tmp_path,
        season="2026-27",
        gw=12,
        pairs=pairs,
        written_at=_TS,
    )
    assert len(written) == 3
    primary = OfficialPointer.model_validate_json(
        pointer_path(tmp_path, season="2026-27", gw=12, line="v4_h3").read_text(
            encoding="utf-8"
        )
    )
    assert primary.predict_run_id == predict
    assert primary.plan_run_id == plan_primary
    assert primary.role == "primary"
    assert primary.squad_mode == "real"
    applied = AppliedRecord.model_validate_json(
        (
            tmp_path
            / "official"
            / "season=2026-27"
            / "gw=12"
            / "checkpoint=T-90m"
            / "applied.json"
        ).read_text(encoding="utf-8")
    )
    assert applied.line == "v4_h3"
    assert applied.plan_run_id == plan_primary


def test_pointer_write_once_and_requires_success(tmp_path: Path) -> None:
    predict = _succeed_run(tmp_path, "r_20261126T120000Z_pred003")
    plan = _succeed_run(
        tmp_path,
        "r_20261126T120000Z_plan003",
        run_type="plan",
        plan_horizon=3,
        horizon=3,
    )
    pointer = OfficialPointer(
        predict_run_id=predict,
        plan_run_id=plan,
        role="primary",
        squad_mode="real",
        written_at=_TS,
    )
    write_official_pointer(
        tmp_path, season="2026-27", gw=12, line="v4_h3", pointer=pointer
    )
    with pytest.raises(OfficialExistsError):
        write_official_pointer(
            tmp_path, season="2026-27", gw=12, line="v4_h3", pointer=pointer
        )
    with pytest.raises(FileNotFoundError, match="run directory missing|_SUCCESS"):
        write_official_pointer(
            tmp_path,
            season="2026-27",
            gw=13,
            line="v4_h3",
            pointer=OfficialPointer(
                predict_run_id="r_missing",
                plan_run_id=plan,
                role="primary",
                squad_mode="real",
                written_at=_TS,
            ),
        )


def test_submitted_and_intervention_write_once(tmp_path: Path) -> None:
    plan = _succeed_run(
        tmp_path,
        "r_20261126T120000Z_plan004",
        run_type="plan",
        plan_horizon=3,
        horizon=3,
    )
    write_submitted_record(
        tmp_path,
        season="2026-27",
        gw=12,
        record=SubmittedRecord(
            line="v4_h3",
            plan_run_id=plan,
            submitted_at=_TS,
            entry_id=42,
        ),
    )
    write_intervention_record(
        tmp_path,
        season="2026-27",
        gw=12,
        record=InterventionRecord(
            who="eamon",
            intervened_at=_TS,
            what_changed="vetoed chip",
            reason="save for DGW",
            line="v4_h3",
            plan_run_id=plan,
        ),
    )
    with pytest.raises(OfficialExistsError):
        write_submitted_record(
            tmp_path,
            season="2026-27",
            gw=12,
            record=SubmittedRecord(line="v4_h3", plan_run_id=plan, submitted_at=_TS),
        )


def test_run_writer_records_official_line_fields(tmp_path: Path) -> None:
    with RunWriter(
        tmp_path,
        kind="official",
        mode="live",
        run_type="plan",
        spec_hash="a" * 64,
        spec_label="dc_layered_v4_native",
        model_id="deadbeef",
        as_of=[AsOfDeadline(season="2026-27", gw=12, deadline_utc=_TS)],
        horizon=5,
        plan_horizon=5,
        code_sha="abc1234",
        code_dirty=False,
        lockfile_hash="b" * 64,
        host="test",
        raw_manifest_hash="c" * 64,
        staged_digests={},
        started_at=_TS,
        run_id="r_20261126T120000Z_off001",
        line="v4_h5",
        squad_mode="virtual",
        prev_run_id="r_prev",
        checkpoint="T-90m",
        prod_tag="prod-20261120-ab12cd3",
        parent_run_id="r_predict",
    ) as writer:
        pass
    assert writer.manifest.line == "v4_h5"
    assert writer.manifest.squad_mode == "virtual"
    assert writer.manifest.prev_run_id == "r_prev"
    assert writer.manifest.checkpoint == "T-90m"
    assert writer.manifest.prod_tag == "prod-20261120-ab12cd3"
