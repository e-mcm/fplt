"""Stage C Store read API: runs / official / load / experiment stubs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fplt.models.predictions import empty_predictions
from fplt.schemas.manifests import (
    AppliedRecord,
    AsOfDeadline,
    ExperimentArm,
    ExperimentWindow,
    OfficialPointer,
)
from fplt.store import Store
from fplt.store.catalog import rebuild_runs_catalog
from fplt.store.experiments import bind_arm_run, submit_experiment
from fplt.store.runs import RunWriter

_TS = datetime(2026, 11, 24, 12, 56, 1, tzinfo=UTC)


def _deadline() -> AsOfDeadline:
    return AsOfDeadline(season="2026-27", gw=12, deadline_utc=_TS)


def _writer(tmp_path: Path, run_id: str, **overrides: object) -> RunWriter:
    kwargs: dict[str, object] = {
        "kind": "adhoc",
        "mode": "live",
        "run_type": "predict",
        "spec_hash": "a" * 64,
        "spec_label": "dc_layered_v5",
        "model_id": "deadbeef",
        "as_of": [_deadline()],
        "horizon": 3,
        "code_sha": "abc1234",
        "code_dirty": False,
        "lockfile_hash": "b" * 64,
        "host": "test",
        "raw_manifest_hash": "c" * 64,
        "staged_digests": {},
        "started_at": _TS,
        "run_id": run_id,
    }
    kwargs.update(overrides)
    return RunWriter(tmp_path, **kwargs)  # type: ignore[arg-type]


def test_runs_filters_catalog_and_hides_incomplete(tmp_path: Path) -> None:
    good = "r_20261124T125601Z_aaaaaa"
    with _writer(tmp_path, good):
        pass
    bad = "r_20261124T125601Z_bbbbbb"
    writer = _writer(tmp_path, bad, spec_label="other")
    writer.__enter__()
    rebuild_runs_catalog(tmp_path)

    store = Store(tmp_path)
    assert store.manifest(bad) is None
    all_rows = store.runs()
    assert all_rows["run_id"].to_list() == [good]
    filtered = store.runs(spec_name="dc_layered_v5", kind="adhoc")
    assert filtered.height == 1
    assert store.runs(spec_name="missing").is_empty()


def test_official_missing_sources_return_none(tmp_path: Path) -> None:
    store = Store(tmp_path)
    assert store.official(gw=12, season="2026-27") is None
    assert store.official(gw=12, season="2026-27", line="v4_h9") is None


def test_official_applied_and_line(tmp_path: Path) -> None:
    plan_id = "r_20261124T125601Z_plan01"
    other_id = "r_20261124T125601Z_plan02"
    with _writer(
        tmp_path,
        plan_id,
        kind="official",
        run_type="plan",
        plan_horizon=3,
        spec_label="dc_layered_v4",
    ):
        pass
    with _writer(
        tmp_path,
        other_id,
        kind="official",
        run_type="plan",
        plan_horizon=9,
        spec_label="dc_layered_v4",
    ):
        pass

    base = tmp_path / "official" / "season=2026-27" / "gw=12" / "checkpoint=T-90m"
    base.mkdir(parents=True)
    (base / "applied.json").write_text(
        AppliedRecord(
            line="v4_h3",
            plan_run_id=plan_id,
            submitted_at=_TS,
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    (base / "line=v4_h9.json").write_text(
        OfficialPointer(
            predict_run_id="r_unused",
            plan_run_id=other_id,
            role="candidate",
            squad_mode="virtual",
            written_at=_TS,
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )

    store = Store(tmp_path)
    applied = store.official(season="2026-27", gw=12)
    assert applied is not None
    assert applied.run_id == plan_id
    line = store.official(season="2026-27", gw=12, line="v4_h9")
    assert line is not None
    assert line.run_id == other_id


def test_load_player_fixture_and_rejects_incomplete(tmp_path: Path) -> None:
    run_id = "r_20261124T125601Z_load01"
    with _writer(tmp_path, run_id) as writer:
        writer.write_parquet(
            "predictions/player_fixture.parquet",
            empty_predictions(),
        )
    store = Store(tmp_path)
    frame = store.load("player_fixture", run=run_id).collect()
    assert isinstance(frame, pl.DataFrame)

    incomplete = "r_20261124T125601Z_load02"
    w = _writer(tmp_path, incomplete)
    w.__enter__()
    with pytest.raises(FileNotFoundError, match="_SUCCESS"):
        store.load("player_fixture", run=incomplete)


def test_experiment_arms_stub(tmp_path: Path) -> None:
    manifest = submit_experiment(
        tmp_path,
        question="does xg_blend help?",
        flavour="comparison",
        arms=[
            ExperimentArm(
                arm="control",
                role="control",
                spec_hash="a" * 64,
                spec_label="base",
            ),
            ExperimentArm(
                arm="treatment",
                role="treatment",
                spec_hash="b" * 64,
                spec_label="blend",
            ),
        ],
        windows=[
            ExperimentWindow(
                role="primary",
                seasons=["2024-25"],
                from_gw=1,
                to_gw=10,
            )
        ],
        build_id="build" + "0" * 58,
        horizon=3,
        metrics=["rmse"],
        gates={},
        code_sha="abc1234",
        lockfile_hash="d" * 64,
        experiment_id="exp_20261124T125601Z_xgblend",
        submitted_at=_TS,
    )
    run_id = "r_20261124T125601Z_arm001"
    with _writer(tmp_path, run_id, kind="experiment"):
        pass
    bind_arm_run(
        tmp_path,
        manifest.experiment_id,
        arm="control",
        run_id=run_id,
        reused=False,
        at=_TS,
    )

    handle = Store(tmp_path).experiment(manifest.experiment_id)
    assert handle.arms["control"] == run_id
    assert handle.arms["treatment"] is None
    assert handle.status.status in {"planned", "running"}
