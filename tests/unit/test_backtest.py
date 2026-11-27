"""Walk-forward refits on the spec's policy and fills actuals only when checked."""

import logging
from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fplt.backtest.ledger import empty_ledger, read_predictions
from fplt.backtest.metrics import summarise
from fplt.backtest.walk_forward import (
    allowed_seasons,
    horizon_for,
    should_refit,
    walk_forward,
)
from fplt.data.outcomes import stage_outcomes
from fplt.data.pit import PITData, PITView
from fplt.data.store import DataStore
from fplt.models.baselines import RollingMeanPredictor
from fplt.models.spec import ModelSpec, TrainWindow
from fplt.store import Store
from fplt.store.runs import write_backtest_run
from fplt.types import GWKey
from tests.builders import build_league


class _CountingMean(RollingMeanPredictor):
    def __init__(self, spec: ModelSpec) -> None:
        super().__init__(spec)
        self.fits = 0

    def fit(self, view: PITView) -> None:
        self.fits += 1
        super().fit(view)


def test_horizon_and_refit_policy() -> None:
    assert [key.gw for key in horizon_for(GWKey("2024-25", 37), 5)] == [37, 38]
    spec = ModelSpec(name="m", kind="rolling_mean", refit="every_n_gw", refit_every=2)
    refits = [should_refit(spec, index) for index in range(4)]
    assert refits == [True, False, True, False]
    never = ModelSpec(name="m", kind="rolling_mean", refit="never")
    assert should_refit(never, 0) is True
    assert should_refit(never, 1) is False
    assert allowed_seasons(GWKey("2024-25", 2), 1) == {"2024-25"}


def test_empty_horizon_is_an_empty_ledger(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    spec = ModelSpec(name="rolling_mean", kind="rolling_mean", params={"window": 5})
    assert read_predictions(tmp_path).is_empty()
    frame = walk_forward(
        RollingMeanPredictor(spec),
        PITData(store),
        [GWKey("2024-25", 1)],
        steps=0,
    )
    assert frame.is_empty()
    assert empty_ledger().is_empty()


def test_walk_forward_logs_each_deadline(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    store = build_league(DataStore(tmp_path))
    spec = ModelSpec(name="rolling_mean", kind="rolling_mean", params={"window": 5})
    as_ofs = [GWKey("2024-25", 1), GWKey("2024-25", 2)]
    with caplog.at_level(logging.INFO, logger="fplt.backtest.walk_forward"):
        walk_forward(RollingMeanPredictor(spec), PITData(store), as_ofs, steps=1)
    assert "rolling_mean 2024-25 gw 1 (1/2) fitting" in caplog.text
    assert "rolling_mean 2024-25 gw 2 (2/2) fitting" in caplog.text
    assert "gw 1 done in" in caplog.text
    assert "fit=" in caplog.text
    assert "predict=" in caplog.text
    assert "walk_forward timing model=rolling_mean" in caplog.text


def test_walk_forward_refits_and_leaves_actuals_null(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    spec = ModelSpec(name="rolling_mean", kind="rolling_mean", params={"window": 5})
    model = _CountingMean(spec)
    as_ofs = [GWKey("2024-25", 1), GWKey("2024-25", 2)]
    frame = walk_forward(model, PITData(store), as_ofs, steps=1)
    assert model.fits == 2
    assert frame["actual_pts"].null_count() == frame.height
    assert set(frame["position"].to_list()) == {"MID", "FWD"}


def test_never_refits_only_once(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    spec = ModelSpec(
        name="rolling_mean",
        kind="rolling_mean",
        params={"window": 5},
        refit="never",
    )
    model = _CountingMean(spec)
    walk_forward(
        model,
        PITData(store),
        [GWKey("2024-25", 1), GWKey("2024-25", 2)],
        steps=1,
    )
    assert model.fits == 1


def test_training_window_hides_an_older_season(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    existing = store.read("player_fixture_stats")
    older = (
        existing.filter(pl.col("player_key") == 100)
        .head(1)
        .with_columns(
            pl.lit("2023-24").alias("season"),
            pl.lit(16).cast(pl.Int64).alias("total_points"),
            pl.lit(datetime(2023, 8, 20, tzinfo=UTC)).alias("available_at"),
        )
    )
    store.write("player_fixture_stats", pl.concat([existing, older]))
    narrow = ModelSpec(
        name="rolling_mean",
        kind="rolling_mean",
        params={"window": 5},
        train_window=TrainWindow(seasons_back=1),
    )
    wide = ModelSpec(
        name="rolling_mean",
        kind="rolling_mean",
        params={"window": 5},
        train_window=TrainWindow(seasons_back=4),
    )
    as_ofs = [GWKey("2024-25", 2)]
    narrow_rows = walk_forward(
        RollingMeanPredictor(narrow), PITData(store), as_ofs, steps=1
    )
    wide_rows = walk_forward(
        RollingMeanPredictor(wide), PITData(store), as_ofs, steps=1
    )
    narrow_x = narrow_rows.filter(pl.col("player_key") == 100)["xpts"][0]
    wide_x = wide_rows.filter(pl.col("player_key") == 100)["xpts"][0]
    assert narrow_x == 8.0
    assert wide_x == 12.0


def test_ledger_view_waits_until_the_gameweek_is_checked(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    spec = ModelSpec(name="rolling_mean", kind="rolling_mean", params={"window": 5})
    frame = walk_forward(
        RollingMeanPredictor(spec),
        PITData(store),
        [GWKey("2024-25", 1), GWKey("2024-25", 2)],
        steps=1,
    )
    write_backtest_run(
        tmp_path,
        spec=spec,
        as_ofs=[
            (GWKey("2024-25", 1), store.deadline(GWKey("2024-25", 1))),
            (GWKey("2024-25", 2), store.deadline(GWKey("2024-25", 2))),
        ],
        horizon=1,
        frame=frame,
    )
    stage_outcomes(store)
    filled = Store(tmp_path).ledger()
    gw1 = filled.filter(pl.col("target_gw") == 1)
    gw2 = filled.filter(pl.col("target_gw") == 2)
    assert gw2["actual_pts"].null_count() == gw2.height
    assert gw2["actual_goals"].null_count() == gw2.height
    player_200 = gw1.filter(pl.col("player_key") == 200)["actual_pts"].to_list()
    assert player_200 == [2]
    assert "actual_goals" in filled.columns
    assert "actual_assists" in filled.columns
    assert "actual_cs" in filled.columns
    assert gw1["actual_goals"].null_count() == 0


def test_summarise_skips_rows_without_actuals() -> None:
    ledger = pl.DataFrame(
        {
            "model_id": ["m", "m", "m", "m"],
            "run_kind": ["backtest", "backtest", "backtest", "backtest"],
            "as_of_season": ["2024-25", "2024-25", "2024-25", "2024-25"],
            "as_of_gw": [1, 1, 1, 2],
            "target_season": ["2024-25", "2024-25", "2024-25", "2024-25"],
            "target_gw": [1, 1, 1, 2],
            "horizon": [1, 1, 1, 1],
            "player_key": [1, 2, 4, 3],
            "position": ["MID", "MID", "FWD", "MID"],
            "fixture_id": [1, 1, 1, 2],
            "xpts": [2.0, 0.0, 1.0, 9.0],
            "xpts_sd": [0.0, 0.0, 0.0, 0.0],
            "p_haul": [0.0, 0.0, None, 1.0],
            "p_60": [1.0, 0.0, None, 1.0],
            "actual_pts": [2, 0, 1, None],
            "actual_minutes": [90, 0, 90, None],
        }
    )
    summary = summarise(ledger)
    mid = summary.filter(pl.col("position") == "MID")
    forward = summary.filter(pl.col("position") == "FWD")
    assert summary.height == 2
    assert mid["mae"].to_list() == [0.0]
    assert list(summary.columns).index("rmse") < list(summary.columns).index("mae")
    assert mid["brier_60"].to_list() == [0.0]
    assert forward["brier_haul"].to_list() == [None]
    assert forward["brier_60"].to_list() == [None]
    assert summarise(ledger.clear()).is_empty()


def test_overall_summary_topk_is_mean_over_gameweeks() -> None:
    """Pooled top-10 over a season is wrong; mean per-GW precision is the default."""
    from fplt.backtest.metrics import overall_summary

    # 12 players per GW so k=10 is strict (not the whole set).
    rows: list[dict[str, object]] = []
    for gw in (1, 2):
        for i in range(12):
            key = gw * 100 + i
            # Predictions: prefer low ids. Actuals: prefer high ids on GW2 only.
            if gw == 1:
                xpts = float(12 - i)
                actual = 12 - i
            else:
                xpts = float(12 - i)
                actual = i
            rows.append(
                {
                    "model_id": "m",
                    "horizon": 1,
                    "position": "MID",
                    "target_season": "2024-25",
                    "target_gw": gw,
                    "player_key": key,
                    "xpts": xpts,
                    "actual_pts": actual,
                    "actual_minutes": 90,
                    "p_haul": None,
                    "p_60": None,
                }
            )
    ledger = pl.DataFrame(rows)
    # GW1: pred top-10 = keys 100..109; actual top-10 = same → 1.0
    # GW2: pred top-10 = 200..209; actual top-10 = 211..202 descending i=11..2
    #   → keys 202..211; intersection {202..209} = 8/10 = 0.8
    # mean = 0.9. Pooled over 24 rows would pick 10 globally and differ.
    summary = overall_summary(ledger)
    assert summary["top10_precision"][0] == pytest.approx(0.9)
    assert "rmse" in summary.columns
    assert list(summary.columns).index("rmse") < list(summary.columns).index("mae")
