"""Stage B10: wrap local migration trees as kind=legacy runs."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fplt.cli import main
from fplt.schemas.ledger import LEDGER_COLUMNS
from fplt.schemas.manifests import AsOfDeadline, RunManifest
from fplt.schemas.predictions import PREDICTION_COLUMNS
from fplt.store.legacy import (
    UNKNOWN,
    import_legacy,
    legacy_run_id,
    parse_as_of_dir,
)
from fplt.store.runs import RunWriter, run_dir

_TS = datetime(2024, 8, 16, 17, tzinfo=UTC)
_MODEL = "deadbeefcafe"


def _prediction_frame(*, model_id: str = _MODEL, gw: int = 1) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model_id": [model_id],
            "as_of_season": ["2024-25"],
            "as_of_gw": [gw],
            "target_season": ["2024-25"],
            "target_gw": [gw],
            "player_key": [100],
            "fixture_id": [1],
            "xpts": [4.5],
            "xpts_sd": [1.0],
            "p_played": [0.9],
            "p_60": [0.7],
            "p_goal": [0.2],
            "p_assist": [0.1],
            "p_cs": [None],
            "p_haul": [0.05],
        },
        schema=PREDICTION_COLUMNS,
    )


def _ledger_frame(*, model_id: str = _MODEL, gw: int = 2) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model_id": [model_id],
            "run_kind": ["backtest"],
            "as_of_season": ["2024-25"],
            "as_of_gw": [gw],
            "target_season": ["2024-25"],
            "target_gw": [gw],
            "horizon": [1],
            "player_key": [100],
            "position": ["MID"],
            "fixture_id": [2],
            "xpts": [3.0],
            "xpts_sd": [0.5],
            "p_played": [0.8],
            "p_60": [0.6],
            "p_goal": [0.1],
            "p_assist": [0.1],
            "p_cs": [None],
            "p_haul": [0.02],
            "actual_pts": [2],
            "actual_minutes": [90],
            "actual_goals": [0],
            "actual_assists": [0],
            "actual_cs": [0],
        },
        schema=LEDGER_COLUMNS,
    )


def _plan_frame(
    *, model_id: str = _MODEL, gw: int = 1, horizon: int = 3
) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model_id": [model_id] * horizon,
            "as_of_season": ["2024-25"] * horizon,
            "as_of_gw": [gw] * horizon,
            "target_gw": [gw + i for i in range(horizon)],
            "step": list(range(horizon)),
            "player_key": [100] * horizon,
            "action": ["start"] * horizon,
            "xpts": [4.0] * horizon,
        }
    )


def test_parse_as_of_dir_with_and_without_horizon() -> None:
    plain = parse_as_of_dir("as_of=2024-25_gw12")
    assert plain is not None
    assert plain.season == "2024-25"
    assert plain.gw == 12
    assert plain.horizon is None
    assert plain.stamp == "2024-25_gw12"
    planned = parse_as_of_dir("as_of=2024-25_gw03_h05")
    assert planned is not None
    assert planned.horizon == 5
    assert planned.stamp == "2024-25_gw03_h05"
    assert parse_as_of_dir("not-an-as-of") is None


def test_legacy_run_id_format() -> None:
    assert legacy_run_id("abc", "2024-25_gw01") == "legacy-abc-2024-25_gw01"
    assert legacy_run_id("abc", "as_of=2024-25_gw01") == "legacy-abc-2024-25_gw01"


def test_import_predictions_plans_ledger_and_artifacts(tmp_path: Path) -> None:
    model = _MODEL
    pred_dir = tmp_path / "predictions" / model / "as_of=2024-25_gw01"
    pred_dir.mkdir(parents=True)
    _prediction_frame().write_parquet(pred_dir / "part.parquet")
    (pred_dir / "code_sha.txt").write_text("abc1234\n", encoding="utf-8")

    art_dir = tmp_path / "artifacts" / model / "as_of=2024-25_gw01"
    art_dir.mkdir(parents=True)
    (art_dir / "spec.json").write_text('{"name": "demo"}\n', encoding="utf-8")
    (art_dir / "git_sha.txt").write_text("abc1234\n", encoding="utf-8")

    plan_dir = tmp_path / "plans" / model / "as_of=2024-25_gw01_h03"
    plan_dir.mkdir(parents=True)
    _plan_frame().write_parquet(plan_dir / "part.parquet")

    ledger_dir = tmp_path / "ledger" / "predictions"
    ledger_dir.mkdir(parents=True)
    _ledger_frame().write_parquet(ledger_dir / "part.parquet")

    only_art = tmp_path / "artifacts" / model / "as_of=2024-25_gw03"
    only_art.mkdir(parents=True)
    (only_art / "model.bin").write_bytes(b"\x00\x01")

    compare_dir = tmp_path / "compare" / "2024-25_gw01-02"
    compare_dir.mkdir(parents=True)
    (compare_dir / "summary.json").write_text('{"ok": true}\n', encoding="utf-8")

    bench_dir = tmp_path / "benchmark_projections" / "as_of=2024-25_gw01"
    bench_dir.mkdir(parents=True)
    _prediction_frame(model_id="ep_next").write_parquet(bench_dir / "part.parquet")

    result = import_legacy(tmp_path)
    assert result.n_skipped == 0
    assert result.n_imported >= 5

    pred_id = legacy_run_id(model, "2024-25_gw01")
    plan_id = legacy_run_id(model, "2024-25_gw01_h03")
    ledger_id = legacy_run_id(model, "2024-25_gw02")
    art_id = legacy_run_id(model, "2024-25_gw03")
    assert pred_id in result.imported
    assert plan_id in result.imported
    assert ledger_id in result.imported
    assert art_id in result.imported

    pred_run = run_dir(tmp_path, pred_id)
    assert (pred_run / "_SUCCESS").is_file()
    assert (pred_run / "predictions" / "player_fixture.parquet").is_file()
    assert (pred_run / "artifacts" / "spec.json").is_file()
    manifest = RunManifest.model_validate_json(
        (pred_run / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest.kind == "legacy"
    assert manifest.code_sha == "abc1234"
    assert manifest.spec_hash == UNKNOWN
    assert manifest.lockfile_hash == UNKNOWN
    assert manifest.tags["legacy_source"] == "predictions"
    assert manifest.as_of[0].gw == 1

    plan_manifest = RunManifest.model_validate_json(
        (run_dir(tmp_path, plan_id) / "manifest.json").read_text(encoding="utf-8")
    )
    assert plan_manifest.run_type == "plan"
    assert plan_manifest.plan_horizon == 3
    assert (run_dir(tmp_path, plan_id) / "plans" / "plan.parquet").is_file()

    ledger_manifest = RunManifest.model_validate_json(
        (run_dir(tmp_path, ledger_id) / "manifest.json").read_text(encoding="utf-8")
    )
    assert ledger_manifest.mode == "backtest"
    assert ledger_manifest.tags["legacy_source"] == "ledger"

    second = import_legacy(tmp_path)
    assert second.n_imported == 0
    assert pred_id in second.skipped
    assert plan_id in second.skipped


def test_dry_run_writes_nothing(tmp_path: Path) -> None:
    pred_dir = tmp_path / "predictions" / _MODEL / "as_of=2024-25_gw01"
    pred_dir.mkdir(parents=True)
    _prediction_frame().write_parquet(pred_dir / "part.parquet")
    result = import_legacy(tmp_path, dry_run=True)
    assert result.n_imported == 1
    assert not (tmp_path / "runs").exists()


def test_cli_legacy_import_dry_run(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    pred_dir = tmp_path / "predictions" / _MODEL / "as_of=2024-25_gw01"
    pred_dir.mkdir(parents=True)
    _prediction_frame().write_parquet(pred_dir / "part.parquet")
    main(["legacy", "import", "--data-dir", str(tmp_path), "--dry-run"])
    out = capsys.readouterr().out
    assert "would import=1" in out
    assert legacy_run_id(_MODEL, "2024-25_gw01") in out
    assert not (tmp_path / "runs").exists()


def test_run_writer_copy_file(tmp_path: Path) -> None:
    source = tmp_path / "src.bin"
    source.write_bytes(b"xyz")
    with RunWriter(
        tmp_path,
        kind="legacy",
        mode="live",
        run_type="predict",
        spec_hash=UNKNOWN,
        spec_label="legacy:test",
        model_id="m1",
        as_of=[AsOfDeadline(season="2024-25", gw=1, deadline_utc=_TS)],
        horizon=1,
        run_id="legacy-m1-2024-25_gw01",
        code_sha=UNKNOWN,
        code_dirty=False,
        lockfile_hash=UNKNOWN,
        host="test",
        raw_manifest_hash=UNKNOWN,
        staged_digests={},
        build_id=None,
    ) as writer:
        writer.copy_file("artifacts/src.bin", source)
    assert (
        run_dir(tmp_path, "legacy-m1-2024-25_gw01") / "artifacts" / "src.bin"
    ).read_bytes() == b"xyz"
