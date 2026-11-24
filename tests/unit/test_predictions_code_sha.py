"""``model_id`` collision: predictions must not reuse across code shas."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from fplt.models.predictions import predictions_path
from fplt.models.spec import ModelSpec, TrainWindow
from fplt.orchestration.factories import (
    predictions_code_sha_path,
    write_predictions_or_existing,
)
from fplt.types import GWKey

_AS_OF = GWKey("2024-25", 1)


def _spec() -> ModelSpec:
    return ModelSpec(
        name="collision_probe",
        kind="rolling_mean",
        params={"window": 3},
        features=[],
        train_window=TrainWindow(seasons_back=1),
    )


def _frame(spec: ModelSpec, *, xpts: float) -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model_id": [spec.model_id],
            "as_of_season": [_AS_OF.season],
            "as_of_gw": [_AS_OF.gw],
            "target_season": [_AS_OF.season],
            "target_gw": [_AS_OF.gw],
            "player_key": [1],
            "fixture_id": [10],
            "xpts": [xpts],
            "xpts_sd": [1.0],
            "p_played": [0.9],
            "p_60": [0.8],
            "p_goal": [0.2],
            "p_assist": [0.1],
            "p_cs": [0.3],
            "p_haul": [0.05],
        }
    )


def test_same_code_sha_reuses_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Matching sidecar allows returning the existing parquet."""
    monkeypatch.setattr("fplt.orchestration.factories.git_sha", lambda: "sha_aaaa")
    spec = _spec()
    first = _frame(spec, xpts=4.0)
    path = write_predictions_or_existing(tmp_path, spec, _AS_OF, first)
    assert path == predictions_path(tmp_path, spec, _AS_OF)
    sidecar = predictions_code_sha_path(path)
    assert sidecar.read_text(encoding="utf-8").strip() == "sha_aaaa"

    second = _frame(spec, xpts=9.0)
    reused = write_predictions_or_existing(tmp_path, spec, _AS_OF, second)
    assert reused == path
    assert pl.read_parquet(path)["xpts"].to_list() == [4.0]


def test_different_code_sha_refuses_prediction_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two code shas with the same spec must not reuse each other's predictions."""
    monkeypatch.setattr("fplt.orchestration.factories.git_sha", lambda: "sha_old")
    spec = _spec()
    path = write_predictions_or_existing(tmp_path, spec, _AS_OF, _frame(spec, xpts=4.0))
    assert path.exists()

    monkeypatch.setattr("fplt.orchestration.factories.git_sha", lambda: "sha_new")
    with pytest.raises(FileExistsError, match="code_sha='sha_old'"):
        write_predictions_or_existing(tmp_path, spec, _AS_OF, _frame(spec, xpts=9.0))
    assert pl.read_parquet(path)["xpts"].to_list() == [4.0]


def test_missing_code_sha_sidecar_refuses_reuse(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Legacy parquet without a sidecar must not be reused silently."""
    monkeypatch.setattr("fplt.orchestration.factories.git_sha", lambda: "sha_now")
    spec = _spec()
    path = predictions_path(tmp_path, spec, _AS_OF)
    path.parent.mkdir(parents=True, exist_ok=True)
    _frame(spec, xpts=1.0).write_parquet(path)
    assert not predictions_code_sha_path(path).exists()

    with pytest.raises(FileExistsError, match="no code_sha.txt sidecar"):
        write_predictions_or_existing(tmp_path, spec, _AS_OF, _frame(spec, xpts=2.0))
