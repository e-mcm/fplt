"""Stage B4 RunWriter: mint, write-once, _SUCCESS, dual-write entry points."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fplt.backtest.predict_live import run_predict
from fplt.backtest.produce_ledger import hash_staged_data
from fplt.backtest.run import run_backtest
from fplt.data.store import DataStore
from fplt.models.predictions import empty_predictions
from fplt.models.spec import load_model_specs
from fplt.schemas.manifests import AsOfDeadline, RunManifest
from fplt.store.runs import (
    RunExistsError,
    RunWriter,
    hash_raw_manifest,
    mint_run_id,
    run_dir,
    write_plan_run,
    write_predict_run,
)
from fplt.types import GWKey
from tests.builders import build_league

_TS = datetime(2026, 11, 24, 12, 56, 1, tzinfo=UTC)
_SEASON = "2024-25"


def _deadline() -> AsOfDeadline:
    return AsOfDeadline(season="2026-27", gw=12, deadline_utc=_TS)


def _writer(tmp_path: Path, **overrides: object) -> RunWriter:
    kwargs: dict[str, object] = {
        "kind": "adhoc",
        "mode": "live",
        "run_type": "predict",
        "spec_hash": "a" * 64,
        "spec_label": "demo",
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
        "run_id": "r_20261124T125601Z_7f3a2c",
    }
    kwargs.update(overrides)
    return RunWriter(tmp_path, **kwargs)  # type: ignore[arg-type]


def test_mint_run_id_is_sortable_and_unique() -> None:
    first = mint_run_id(when=_TS, suffix="aaaaaa")
    second = mint_run_id(when=_TS, suffix="bbbbbb")
    assert first == "r_20261124T125601Z_aaaaaa"
    assert first < second
    with ThreadPoolExecutor(max_workers=8) as pool:
        ids = list(pool.map(lambda _: mint_run_id(when=_TS), range(64)))
    assert len(set(ids)) == 64


def test_run_writer_manifest_first_success_last(tmp_path: Path) -> None:
    with _writer(tmp_path) as writer:
        assert (writer.path / "manifest.json").exists()
        running = RunManifest.model_validate_json(
            (writer.path / "manifest.json").read_text(encoding="utf-8")
        )
        assert running.status == "running"
        assert running.finished_at is None
        assert running.build_id is None
        assert not (writer.path / "_SUCCESS").exists()
        writer.write_parquet(
            "predictions/player_fixture.parquet",
            empty_predictions(),
        )
    final = RunManifest.model_validate_json(
        (writer.path / "manifest.json").read_text(encoding="utf-8")
    )
    assert final.status == "succeeded"
    assert final.finished_at is not None
    assert (writer.path / "_SUCCESS").exists()
    assert final.raw_manifest_hash == "c" * 64
    assert final.build_id is None


def test_run_writer_records_explicit_build_id(tmp_path: Path) -> None:
    build_id = "cafe" * 8
    with _writer(tmp_path, build_id=build_id, run_id="r_20261124T125601Z_bld001"):
        pass
    path = run_dir(tmp_path, "r_20261124T125601Z_bld001")
    manifest = RunManifest.model_validate_json(
        (path / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.build_id == build_id


def test_run_writer_auto_detects_latest_build_id(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "fplt.store.builds.latest_build_id",
        lambda _data_dir: "auto" * 16,
    )
    with _writer(tmp_path, run_id="r_20261124T125601Z_bldaut"):
        pass
    path = run_dir(tmp_path, "r_20261124T125601Z_bldaut")
    manifest = RunManifest.model_validate_json(
        (path / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.build_id == "auto" * 16


def test_run_writer_refuses_existing_run(tmp_path: Path) -> None:
    with _writer(tmp_path):
        pass
    with pytest.raises(RunExistsError, match="already exists"):
        with _writer(tmp_path):
            pass


def test_run_writer_failed_has_no_success(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="boom"):
        with _writer(tmp_path, run_id="r_20261124T125601Z_failed1"):
            raise RuntimeError("boom")
    path = run_dir(tmp_path, "r_20261124T125601Z_failed1")
    manifest = RunManifest.model_validate_json(
        (path / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.status == "failed"
    assert not (path / "_SUCCESS").exists()


def test_run_writer_table_write_once(tmp_path: Path) -> None:
    with _writer(tmp_path, run_id="r_20261124T125601Z_once01") as writer:
        writer.write_parquet("metrics/summary.parquet", empty_predictions())
        with pytest.raises(FileExistsError):
            writer.write_parquet("metrics/summary.parquet", empty_predictions())


def test_hash_raw_manifest_stable(tmp_path: Path) -> None:
    raw = tmp_path / "raw" / "fpl_api" / "season=2024-25" / "captured_at=x"
    raw.mkdir(parents=True)
    (raw / "bootstrap_static.json.gz").write_bytes(b"abc")
    first = hash_raw_manifest(tmp_path)
    second = hash_raw_manifest(tmp_path)
    assert first == second
    assert len(first) == 64
    (raw / "bootstrap_static.json.gz").write_bytes(b"abcd")
    assert hash_raw_manifest(tmp_path) != first


def test_write_predict_run_records_staged_digests(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    digests = hash_staged_data(tmp_path)
    assert digests
    spec = load_model_specs()["rolling_mean_v1"]
    as_of = GWKey(_SEASON, 2)
    manifest = write_predict_run(
        tmp_path,
        spec=spec,
        as_of=as_of,
        deadline_utc=store.deadline(as_of),
        horizon=1,
        frame=empty_predictions(),
        purpose="unit",
    )
    assert manifest.status == "succeeded"
    assert manifest.staged_digests == digests
    assert (run_dir(tmp_path, manifest.run_id) / "_SUCCESS").exists()
    assert (
        run_dir(tmp_path, manifest.run_id) / "predictions" / "player_fixture.parquet"
    ).exists()


def test_run_predict_dual_writes_run(tmp_path: Path) -> None:
    build_league(DataStore(tmp_path))
    path = run_predict(tmp_path, "rolling_mean_v1", 1, season=_SEASON, gw=2)
    assert path.exists()
    runs = list((tmp_path / "runs").glob("run=*/_SUCCESS"))
    assert len(runs) == 1
    manifest = RunManifest.model_validate_json(
        (runs[0].parent / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.run_type == "predict"
    assert manifest.mode == "live"
    assert manifest.kind == "adhoc"


def test_run_backtest_dual_writes_run(tmp_path: Path) -> None:
    build_league(DataStore(tmp_path))
    run_backtest(tmp_path, _SEASON, 1, 2, ["rolling_mean_v1"], steps=1)
    runs = list((tmp_path / "runs").glob("run=*/manifest.json"))
    assert len(runs) == 1
    manifest = RunManifest.model_validate_json(runs[0].read_text(encoding="utf-8"))
    assert manifest.mode == "backtest"
    assert len(manifest.as_of) == 2


def test_write_plan_run_writes_plan_parquet(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    spec = load_model_specs()["rolling_mean_v1"]
    as_of = GWKey(_SEASON, 1)
    frame = pl.DataFrame(
        {
            "model_id": [spec.model_id],
            "as_of_season": [_SEASON],
            "as_of_gw": [1],
            "target_gw": [1],
            "step": [0],
            "player_key": [1],
            "position": ["MID"],
            "team_key": [1],
            "in_xi": [True],
            "captain": [True],
            "vice": [False],
            "bought": [False],
            "sold": [False],
            "xpts": [1.0],
            "hit": [0.0],
        }
    )
    manifest = write_plan_run(
        tmp_path,
        spec=spec,
        as_of=as_of,
        deadline_utc=store.deadline(as_of),
        horizon=1,
        frame=frame,
        purpose="unit-plan",
    )
    assert manifest.run_type == "plan"
    assert manifest.plan_horizon == 1
    assert (run_dir(tmp_path, manifest.run_id) / "plans" / "plan.parquet").exists()
    assert (run_dir(tmp_path, manifest.run_id) / "_SUCCESS").exists()
