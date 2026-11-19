"""Walk-forward refits on the spec's policy and fills actuals only when checked."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from fplt.backtest.ledger import (
    append_predictions,
    fill_actuals,
    ledger_path,
    read_predictions,
)
from fplt.backtest.metrics import summarise
from fplt.backtest.walk_forward import (
    allowed_seasons,
    horizon_for,
    should_refit,
    walk_forward,
)
from fplt.data.pit import PITData, PITView
from fplt.data.store import DataStore
from fplt.models.baselines import RollingMeanPredictor
from fplt.models.spec import ModelSpec, TrainWindow
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
    append_predictions(tmp_path, frame)
    assert not ledger_path(tmp_path).exists()


def test_walk_forward_refits_and_leaves_actuals_null(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    spec = ModelSpec(name="rolling_mean", kind="rolling_mean", params={"window": 5})
    model = _CountingMean(spec)
    as_ofs = [GWKey("2024-25", 1), GWKey("2024-25", 2)]
    frame = walk_forward(model, PITData(store), as_ofs, steps=1)
    assert model.fits == 2
    assert frame["actual_pts"].null_count() == frame.height
    assert set(frame["position"].to_list()) == {"MID", "FWD"}
    append_predictions(tmp_path, frame)
    append_predictions(tmp_path, frame)
    assert read_predictions(tmp_path).height == frame.height * 2


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


def test_fill_actuals_waits_until_the_gameweek_is_checked(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    spec = ModelSpec(name="rolling_mean", kind="rolling_mean", params={"window": 5})
    frame = walk_forward(
        RollingMeanPredictor(spec),
        PITData(store),
        [GWKey("2024-25", 1), GWKey("2024-25", 2)],
        steps=1,
    )
    append_predictions(tmp_path, frame)
    fill_actuals(tmp_path, store)
    filled = read_predictions(tmp_path)
    gw1 = filled.filter(pl.col("target_gw") == 1)
    gw2 = filled.filter(pl.col("target_gw") == 2)
    assert gw2["actual_pts"].null_count() == gw2.height
    player_200 = gw1.filter(pl.col("player_key") == 200)["actual_pts"].to_list()
    assert player_200 == [2]
    fill_actuals(tmp_path / "missing", store)


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
    assert mid["brier_60"].to_list() == [0.0]
    assert forward["brier_haul"].to_list() == [None]
    assert forward["brier_60"].to_list() == [None]
    assert summarise(ledger.clear()).is_empty()
