"""Live predict writes prediction parquet and can reuse a fitted artifact."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fplt.backtest.ledger import read_predictions
from fplt.backtest.predict_live import resolve_as_of, run_predict
from fplt.cli import main
from fplt.data.store import DataStore
from fplt.models.predictions import empty_predictions, require_scored_predictions
from fplt.models.spec import artifact_dir, load_model_specs
from fplt.settings import get_settings
from fplt.types import GWKey
from tests.builders import build_league

_SEASON = "2024-25"
_AS_OF = GWKey(_SEASON, 2)
_BEFORE_GW2 = datetime(2024, 8, 20, tzinfo=UTC)
_AFTER_ALL = datetime(2025, 1, 1, tzinfo=UTC)


def test_run_predict_writes_predictions_and_ledger(tmp_path: Path) -> None:
    build_league(DataStore(tmp_path))
    path = run_predict(
        tmp_path,
        "rolling_mean_v1",
        1,
        season=_SEASON,
        gw=2,
    )
    spec = load_model_specs()["rolling_mean_v1"]
    assert "runs" in path.parts
    assert path.name == "player_fixture.parquet"
    assert path.exists()
    frame = pl.read_parquet(path)
    assert frame.height == 2
    assert set(frame["target_gw"].to_list()) == {2}
    ledger = read_predictions(tmp_path)
    assert not ledger.is_empty()
    assert ledger["run_kind"].unique().to_list() == ["live"]
    assert (artifact_dir(tmp_path, spec, _AS_OF) / "spec.json").exists()


def test_run_predict_loads_an_existing_artifact(tmp_path: Path) -> None:
    build_league(DataStore(tmp_path))
    first = run_predict(tmp_path, "rolling_mean_v1", 1, season=_SEASON, gw=2)
    second = run_predict(tmp_path, "rolling_mean_v1", 1, season=_SEASON, gw=2)
    assert second.exists()
    assert "runs" in second.parts
    assert first != second
    assert pl.read_parquet(second).height == 2


def test_run_predict_rejects_an_all_zero_frame(tmp_path: Path) -> None:
    build_league(DataStore(tmp_path))
    with pytest.raises(ValueError, match="all zero|predicted 0"):
        run_predict(tmp_path, "rolling_mean_v1", 1, season=_SEASON, gw=1)
    assert not (tmp_path / "predictions").exists()
    require_scored_predictions(empty_predictions(), "rolling_mean_v1")


def test_resolve_as_of_picks_the_next_deadline(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    assert resolve_as_of(store, now=_BEFORE_GW2) == _AS_OF
    assert resolve_as_of(store, now=_AFTER_ALL) == _AS_OF
    assert resolve_as_of(store, season=_SEASON, gw=1) == GWKey(_SEASON, 1)
    with pytest.raises(ValueError, match="together"):
        resolve_as_of(store, season=_SEASON)
    store.write(
        "gameweeks",
        store.read("gameweeks").with_columns(pl.lit(True).alias("data_checked")),
    )
    with pytest.raises(ValueError, match="no upcoming"):
        resolve_as_of(store, now=_AFTER_ALL)


def test_predict_command_smoke(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    build_league(DataStore(tmp_path))
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        main(
            [
                "predict",
                "--model",
                "rolling_mean_v1",
                "--horizon",
                "1",
                "--season",
                _SEASON,
                "--gw",
                "2",
            ]
        )
        printed = capsys.readouterr().out.strip()
        out_path = Path(printed.splitlines()[-1])
        assert "runs" in out_path.parts
        assert out_path.exists()
        with pytest.raises(SystemExit) as raised:
            main(
                [
                    "predict",
                    "--model",
                    "missing",
                    "--horizon",
                    "1",
                    "--season",
                    _SEASON,
                    "--gw",
                    "2",
                ]
            )
        assert raised.value.code == 2
    finally:
        get_settings.cache_clear()


def test_bad_predict_requests_are_rejected(tmp_path: Path) -> None:
    build_league(DataStore(tmp_path))
    with pytest.raises(ValueError, match="horizon"):
        run_predict(tmp_path, "rolling_mean_v1", 0, season=_SEASON, gw=2)
    with pytest.raises(ValueError, match="unknown model"):
        run_predict(tmp_path, "missing", 1, season=_SEASON, gw=2)
    first = run_predict(tmp_path, "rolling_mean_v1", 1, season=_SEASON, gw=2)
    second = run_predict(tmp_path, "rolling_mean_v1", 1, season=_SEASON, gw=2)
    assert first != second
    assert second.exists()


def test_fit_or_load_predictor_fit_fp_reuse_and_mismatch(tmp_path: Path) -> None:
    from fplt.backtest.predict_live import fit_or_load_predictor
    from fplt.data.pit import PITData
    from fplt.models.spec import load_model_specs

    store = build_league(DataStore(tmp_path))
    key = GWKey(_SEASON, 2)
    spec = load_model_specs()["rolling_mean_v1"]
    view = PITData(store).view(key)
    fit_fp = "a" * 64
    first = fit_or_load_predictor(tmp_path, spec, key, view, fit_fp=fit_fp)
    second = fit_or_load_predictor(tmp_path, spec, key, view, fit_fp=fit_fp)
    assert first is not second  # freshly loaded instance
    dest = artifact_dir(tmp_path, spec, key) / f"fit_fp={fit_fp}"
    assert (dest / "fit_fp.txt").read_text(encoding="utf-8").strip() == fit_fp
    (dest / "fit_fp.txt").write_text("wrong\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="fit_fp mismatch"):
        fit_or_load_predictor(tmp_path, spec, key, view, fit_fp=fit_fp)
