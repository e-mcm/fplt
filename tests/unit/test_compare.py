"""Fair compare runs walk-forward and fixed modes over a shared window."""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from fplt.backtest.compare import (
    compare_models,
    format_summary_table,
)
from fplt.backtest.ledger import read_predictions
from fplt.backtest.metrics import overall_summary
from fplt.cli import main
from fplt.data.store import DataStore
from fplt.models.spec import ModelSpec
from fplt.settings import get_settings

_PRIOR = "2023-24"
_TEST = "2024-25"
_START = datetime(2024, 7, 1, tzinfo=UTC)
_GW1 = datetime(2024, 8, 16, 17, tzinfo=UTC)
_GW1_RESULT = datetime(2024, 8, 17, 18, tzinfo=UTC)
_GW2 = datetime(2024, 8, 23, 17, tzinfo=UTC)
_PRIOR_RESULT = datetime(2023, 8, 12, 17, tzinfo=UTC)
_ZERO = (
    "goals_scored",
    "assists",
    "clean_sheets",
    "goals_conceded",
    "saves",
    "bonus",
    "yellow_cards",
    "red_cards",
    "own_goals",
    "penalties_saved",
    "penalties_missed",
)


def test_compare_runs_both_modes_and_writes_summary(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    _store(tmp_path)
    with caplog.at_level(logging.INFO, logger="fplt.backtest.compare"):
        result = compare_models(
            tmp_path,
            _TEST,
            models=["rolling_mean_v1"],
            test_from_gw=1,
            test_to_gw=2,
            seasons_back=2,
            steps=1,
        )
    assert "compare 2024-25 gw 1-2 fit_gw=1 steps=1 modes=walk_forward,fixed" in (
        caplog.text
    )
    assert "walk_forward rolling_mean_v1 starting refit=every_gw" in caplog.text
    assert "fixed rolling_mean_v1 starting refit=never" in caplog.text
    assert "staging outcomes for checked gameweeks" in caplog.text
    assert "joining ledger view" in caplog.text
    assert "writing" in caplog.text
    assert [run.mode for run in result.runs] == ["walk_forward", "fixed"]
    assert {run.spec.refit for run in result.runs} == {"every_gw", "never"}
    assert len({run.spec.model_id for run in result.runs}) == 2
    ledger = read_predictions(tmp_path)
    assert set(ledger["model_id"].unique().to_list()) == {
        run.spec.model_id for run in result.runs
    }
    summary_json = result.output_dir / "summary.json"
    assert summary_json.is_file()
    assert (result.output_dir / "summary.parquet").is_file()
    payload = json.loads(summary_json.read_text(encoding="utf-8"))
    assert payload["season"] == _TEST
    assert payload["fit_gw"] == 1
    assert {row["mode"] for row in payload["runs"]} == {"walk_forward", "fixed"}
    assert "rmse" in result.summary.columns
    assert "mae" in result.summary.columns
    assert list(result.summary.columns).index("rmse") < list(
        result.summary.columns
    ).index("mae")
    assert payload.get("headline_metric") == "rmse"
    assert "decision-relevant" in str(payload.get("player_pool", ""))
    assert "model_name" in result.summary.columns
    assert set(result.summary["model_name"].to_list()) == {"rolling_mean_v1"}
    assert set(result.summary["mode"].to_list()) == {"walk_forward", "fixed"}
    table = format_summary_table(result.summary)
    assert "shape:" in table or table.startswith("(no scored")
    if "rmse" in table and "mae" in table:
        assert table.index("rmse") < table.index("mae")


def test_fixed_fit_gw_before_test_drops_pre_test_rows(tmp_path: Path) -> None:
    _store(tmp_path)
    result = compare_models(
        tmp_path,
        _TEST,
        models=["rolling_mean_v1"],
        test_from_gw=2,
        test_to_gw=2,
        fit_gw=1,
        seasons_back=2,
        modes=["fixed"],
        steps=1,
    )
    ledger = read_predictions(tmp_path).filter(
        pl.col("model_id") == result.runs[0].spec.model_id
    )
    assert ledger["as_of_gw"].unique().to_list() == [2]


def test_walk_forward_and_fixed_differ_on_rolling_mean(tmp_path: Path) -> None:
    _store(tmp_path)
    result = compare_models(
        tmp_path,
        _TEST,
        models=["rolling_mean_v1"],
        test_from_gw=1,
        test_to_gw=2,
        seasons_back=2,
        steps=1,
    )
    by_mode = {run.mode: run.spec.model_id for run in result.runs}
    walk = _points(tmp_path, by_mode["walk_forward"])
    fixed = _points(tmp_path, by_mode["fixed"])
    assert walk[1] == pytest.approx(2.0)
    assert walk[2] == pytest.approx(6.0)
    assert fixed[1] == pytest.approx(2.0)
    assert fixed[2] == pytest.approx(2.0)


def test_compare_rejects_bad_windows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="test gameweeks"):
        compare_models(
            tmp_path,
            _TEST,
            models=["rolling_mean_v1"],
            test_from_gw=3,
            test_to_gw=1,
        )
    with pytest.raises(ValueError, match="fit_gw"):
        compare_models(
            tmp_path,
            _TEST,
            models=["rolling_mean_v1"],
            test_from_gw=1,
            test_to_gw=2,
            fit_gw=3,
        )
    with pytest.raises(ValueError, match="modes"):
        compare_models(tmp_path, _TEST, models=["rolling_mean_v1"], modes=[])
    with pytest.raises(ValueError, match="unknown model"):
        compare_models(tmp_path, _TEST, models=["missing_model"])


def test_overall_summary_pools_positions() -> None:
    ledger = pl.DataFrame(
        {
            "model_id": ["m1", "m1", "m1", "m1"],
            "horizon": [1, 1, 1, 1],
            "position": ["MID", "MID", "FWD", "FWD"],
            "target_season": ["2024-25"] * 4,
            "target_gw": [1, 1, 1, 1],
            "player_key": [1, 2, 3, 4],
            "xpts": [10.0, 5.0, 8.0, 1.0],
            "actual_pts": [9, 4, 7, 2],
            "actual_minutes": [90, 90, 90, 90],
            "p_haul": [None, None, None, None],
            "p_60": [None, None, None, None],
        }
    )
    summary = overall_summary(ledger)
    assert summary.height == 1
    assert summary["n"][0] == 4
    assert "position" not in summary.columns
    assert overall_summary(ledger.head(0)).is_empty()


def test_format_summary_empty() -> None:
    assert "no scored" in format_summary_table(pl.DataFrame())


def test_format_summary_lists_rmse_before_mae() -> None:
    summary = pl.DataFrame(
        {
            "model_name": ["a"],
            "mode": ["walk_forward"],
            "horizon": [1],
            "mae": [1.0],
            "rmse": [1.5],
            "spearman": [0.2],
            "top10_precision": [0.3],
            "brier_haul": [0.1],
            "n": [10],
            "model_id": ["m"],
        }
    )
    # Prefer column order over the truncated polars string render.
    cols = [
        c
        for c in (
            "model_name",
            "mode",
            "horizon",
            "rmse",
            "mae",
            "spearman",
            "top10_precision",
            "brier_haul",
            "n",
            "model_id",
        )
        if c in summary.columns
    ]
    assert cols.index("rmse") < cols.index("mae")
    assert "rmse" in format_summary_table(summary)


def test_cli_compare(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    _store(tmp_path)
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    main(
        [
            "compare",
            "--season",
            _TEST,
            "--models",
            "rolling_mean_v1",
            "--test-from-gw",
            "1",
            "--test-to-gw",
            "2",
            "--seasons-back",
            "2",
            "--modes",
            "fixed",
        ]
    )
    captured = capsys.readouterr()
    assert "rolling_mean_v1 mode=fixed" in captured.out
    assert "summary.json" in captured.out
    assert "compare 2024-25 gw 1-2" in captured.err
    assert "fixed rolling_mean_v1 starting" in captured.err
    assert "gw 1 (1/2) fitting" in captured.err
    get_settings.cache_clear()


def test_cli_compare_maps_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()

    def boom(*_a: Any, **_k: Any) -> None:
        raise ValueError("bad compare")

    monkeypatch.setattr("fplt.cli.compare_models", boom)
    with pytest.raises(SystemExit) as exc:
        main(["compare", "--season", _TEST, "--models", "rolling_mean_v1"])
    assert exc.value.code == 2
    get_settings.cache_clear()


def test_compare_delegates_model_resolution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    seen: dict[str, Any] = {}
    empty = pl.DataFrame(
        schema={
            "model_id": pl.Utf8(),
            "run_kind": pl.Utf8(),
            "as_of_season": pl.Utf8(),
            "as_of_gw": pl.Int64(),
            "target_season": pl.Utf8(),
            "target_gw": pl.Int64(),
            "horizon": pl.Int64(),
            "player_key": pl.Int64(),
            "position": pl.Utf8(),
            "fixture_id": pl.Int64(),
            "xpts": pl.Float64(),
            "xpts_sd": pl.Float64(),
            "p_haul": pl.Float64(),
            "p_60": pl.Float64(),
            "actual_pts": pl.Int64(),
            "actual_minutes": pl.Int64(),
        }
    )

    def fake_resolve(
        season: str,
        *,
        models: list[str] | None = None,
        with_openfpl: bool = False,
        openfpl_model_dir: str | None = None,
    ) -> list[str]:
        seen["season"] = season
        seen["models"] = models
        return ["rolling_mean_v1"]

    def fake_produce(
        data_dir: Path,
        spec: ModelSpec,
        season: str,
        **kwargs: Any,
    ) -> Any:
        from fplt.backtest.produce_ledger import (
            LedgerCacheKey,
            ProduceLedgerResult,
        )

        key = LedgerCacheKey(
            resolved_spec_sha256="abc",
            uv_lock_sha256="lock",
            data_hashes={},
            season=season,
            test_from_gw=1,
            test_to_gw=1,
            fit_gw=1,
            mode="walk_forward",
            steps=1,
            seasons_back=None,
            n_sims=None,
            seed=0,
        )
        return ProduceLedgerResult(
            frame=empty,
            spec=spec,
            key=key,
            cache_dir=tmp_path / "ledger_cache" / "x",
            cache_hit=False,
        )

    monkeypatch.setattr("fplt.backtest.compare.resolve_eval_models", fake_resolve)
    monkeypatch.setattr("fplt.backtest.compare.produce_ledger", fake_produce)
    monkeypatch.setattr(
        "fplt.backtest.compare.write_backtest_run",
        lambda *a, **k: type("M", (), {"run_id": "r_test"})(),
    )
    monkeypatch.setattr(
        "fplt.backtest.compare.stage_outcomes",
        lambda *_: {},
    )

    class _FakeStore:
        def __init__(self, *_args: object, **_kwargs: object) -> None:
            pass

        def ledger(self, *_args: object, **_kwargs: object) -> pl.DataFrame:
            return empty

    monkeypatch.setattr("fplt.backtest.compare.Store", _FakeStore)
    monkeypatch.setattr(
        "fplt.backtest.compare.load_model_specs",
        lambda: {
            "rolling_mean_v1": ModelSpec(
                name="rolling_mean_v1",
                kind="rolling_mean",
            )
        },
    )
    result = compare_models(
        tmp_path,
        _TEST,
        models=None,
        test_from_gw=1,
        test_to_gw=1,
        modes=["walk_forward"],
        with_openfpl=True,
    )
    assert seen["season"] == _TEST
    assert seen["models"] is None
    assert result.runs[0].model_name == "rolling_mean_v1"


def _points(root: Path, model_id: str) -> dict[int, float]:
    frame = read_predictions(root).filter(pl.col("model_id") == model_id)
    return {
        int(gw): float(points)
        for gw, points in frame.select("as_of_gw", "xpts").iter_rows()
    }


def _store(root: Path, source: str = "fpl_api") -> DataStore:
    store = DataStore(root)
    store.write(
        "players",
        _utc(
            pl.DataFrame(
                {
                    "season": [_TEST],
                    "player_key": [100],
                    "element_id": [10],
                    "team_key": [1],
                    "position": ["MID"],
                    "web_name": ["A"],
                    "available_at": [_START],
                    "source": [source],
                }
            )
        ),
    )
    store.write(
        "gameweeks",
        _utc(
            pl.DataFrame(
                {
                    "season": [_TEST, _TEST],
                    "gw": [1, 2],
                    "deadline_time": [_GW1, _GW2],
                    "finished": [True, False],
                    "data_checked": [True, False],
                    "available_at": [_START, _START],
                    "source": [source, source],
                }
            )
        ),
    )
    store.write(
        "fixtures",
        _utc(
            pl.DataFrame(
                {
                    "season": [_TEST, _TEST],
                    "fixture_id": [101, 102],
                    "gw": [1, 2],
                    "team_h_key": [1, 1],
                    "team_a_key": [2, 2],
                    "kickoff_time": [_GW1, _GW2],
                    "team_h_score": [1, None],
                    "team_a_score": [0, None],
                    "finished": [True, False],
                    "available_at": [_START, _START],
                    "source": [source, source],
                }
            )
        ),
    )
    store.write(
        "player_status",
        _utc(
            pl.DataFrame(
                {
                    "season": [_TEST],
                    "player_key": [100],
                    "status": ["a"],
                    "news": [""],
                    "price": [50],
                    "ep_next": pl.Series("ep_next", [None], dtype=pl.Float64),
                    "chance_of_playing": pl.Series(
                        "chance_of_playing", [None], dtype=pl.Float64
                    ),
                    "available_at": [_START],
                    "source": [source],
                }
            )
        ),
    )
    store.write("player_fixture_stats", _utc(_appearances(source)))
    return store


def _appearances(source: str) -> pl.DataFrame:
    rows = [
        (_PRIOR, 1, 1, 2, 90, _PRIOR_RESULT),
        (_TEST, 1, 101, 10, 90, _GW1_RESULT),
    ]
    return pl.DataFrame(
        {
            "season": [row[0] for row in rows],
            "gw": [row[1] for row in rows],
            "player_key": [100, 100],
            "fixture_id": [row[2] for row in rows],
            "minutes": [row[4] for row in rows],
            "total_points": [row[3] for row in rows],
            "starts": [1, 1],
            "available_at": [row[5] for row in rows],
            "source": [source, source],
            **{name: [0, 0] for name in _ZERO},
        }
    )


def _utc(frame: pl.DataFrame) -> pl.DataFrame:
    columns = [
        pl.col(name).cast(pl.Datetime("us", "UTC"))
        for name, dtype in frame.schema.items()
        if isinstance(dtype, pl.Datetime)
    ]
    return frame.with_columns(columns) if columns else frame
