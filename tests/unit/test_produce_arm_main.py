"""Unit tests for the produce-arm subprocess entrypoint."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from fplt.backtest._produce_arm_main import main


@dataclass(frozen=True)
class _FakeKey:
    uv_lock_sha256: str = "lock" * 16
    resolved_spec_sha256: str = "spec" * 16

    def digest(self) -> str:
        return "digest" * 8


def test_produce_arm_main_usage_when_argv_wrong(
    capsys: pytest.CaptureFixture[str],
) -> None:
    assert main([]) == 2
    assert "usage:" in capsys.readouterr().err


def test_produce_arm_main_missing_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    request = tmp_path / "req.json"
    request.write_text(
        json.dumps(
            {
                "data_dir": str(tmp_path),
                "model_name": "missing_model",
                "season": "2024-25",
                "mode": "walk_forward",
                "test_from_gw": 1,
                "test_to_gw": 2,
                "out_parquet": str(tmp_path / "out.parquet"),
            }
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "fplt.models.spec.load_model_specs",
        lambda _models_dir=None: {},
    )
    assert main([str(request)]) == 2
    assert "missing" in capsys.readouterr().err


def test_produce_arm_main_writes_parquet_and_meta(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    out_parquet = tmp_path / "arm" / "preds.parquet"
    request = tmp_path / "req.json"
    request.write_text(
        json.dumps(
            {
                "data_dir": str(tmp_path / "data"),
                "model_name": "rolling_mean_v1",
                "season": "2024-25",
                "mode": "fixed",
                "test_from_gw": 1,
                "test_to_gw": 3,
                "fit_gw": 1,
                "steps": 2,
                "seasons_back": 1,
                "use_cache": False,
                "lock_root": str(tmp_path),
                "eval_seed": 7,
                "out_parquet": str(out_parquet),
            }
        ),
        encoding="utf-8",
    )
    spec = SimpleNamespace(name="rolling_mean_v1")
    frame = pl.DataFrame({"player_key": [1], "xpts": [4.5]})
    produced = SimpleNamespace(
        frame=frame,
        cache_hit=False,
        key=_FakeKey(),
    )
    calls: dict[str, Any] = {}

    def fake_produce(*_args: object, **kwargs: object) -> SimpleNamespace:
        calls.update(kwargs)
        return produced

    monkeypatch.setattr(
        "fplt.models.spec.load_model_specs",
        lambda _models_dir=None: {"rolling_mean_v1": spec},
    )
    monkeypatch.setattr(
        "fplt.backtest.produce_ledger.produce_ledger",
        fake_produce,
    )

    assert main([str(request)]) == 0
    assert out_parquet.is_file()
    meta_path = out_parquet.with_suffix(".meta.json")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    assert meta["rows"] == 1
    assert meta["cache_hit"] is False
    assert meta["digest"] == "digest" * 8
    printed = json.loads(capsys.readouterr().out)
    assert printed["rows"] == 1
    assert calls["mode"] == "fixed"
    assert calls["fit_gw"] == 1
    assert calls["steps"] == 2
    assert calls["seasons_back"] == 1
    assert calls["use_cache"] is False
    assert calls["eval_seed"] == 7
