"""B9: outcomes staging and Store.ledger view."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from fplt.backtest.walk_forward import walk_forward
from fplt.data.outcomes import (
    build_fixture_outcomes,
    build_player_fixture_outcomes,
    stage_outcomes,
)
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.baselines.rolling_mean import RollingMeanPredictor
from fplt.models.spec import ModelSpec
from fplt.store import Store
from fplt.store.runs import write_backtest_run
from fplt.types import GWKey
from tests.builders import build_league


def test_stage_outcomes_only_checked_gameweeks(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    written = stage_outcomes(store)
    player = written["player_fixture_outcomes"]
    assert set(player["gw"].to_list()) == {1}
    assert player.filter(pl.col("player_key") == 200)["pts"].to_list() == [2]
    # Latest stats row for player 100 (40 pts, not the earlier 8).
    assert player.filter(pl.col("player_key") == 100)["pts"].to_list() == [40]
    fixture = written["fixture_outcomes"]
    assert fixture.height == 1
    assert fixture["team_h_score"].to_list() == [2]
    assert fixture["team_a_score"].to_list() == [0]


def test_build_outcomes_empty_when_nothing_checked(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    assert build_player_fixture_outcomes(store).is_empty()
    assert build_fixture_outcomes(store).is_empty()
    assert stage_outcomes(store) == {}


def test_ledger_view_joins_outcomes_for_checked_only(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    spec = ModelSpec(name="rolling_mean", kind="rolling_mean", params={"window": 5})
    frame = walk_forward(
        RollingMeanPredictor(spec),
        PITData(store),
        [GWKey("2024-25", 1), GWKey("2024-25", 2)],
        steps=1,
    )
    run = write_backtest_run(
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
    view = Store(tmp_path).ledger(runs=[run.run_id])
    gw1 = view.filter(pl.col("target_gw") == 1)
    gw2 = view.filter(pl.col("target_gw") == 2)
    assert gw2["actual_pts"].null_count() == gw2.height
    assert gw1.filter(pl.col("player_key") == 200)["actual_pts"].to_list() == [2]
    assert gw1["actual_goals"].null_count() == 0


def test_store_ledger_discovers_runs(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    spec = ModelSpec(name="rolling_mean", kind="rolling_mean", params={"window": 5})
    frame = walk_forward(
        RollingMeanPredictor(spec),
        PITData(store),
        [GWKey("2024-25", 1)],
        steps=1,
    )
    write_backtest_run(
        tmp_path,
        spec=spec,
        as_ofs=[(GWKey("2024-25", 1), store.deadline(GWKey("2024-25", 1)))],
        horizon=1,
        frame=frame,
    )
    stage_outcomes(store)
    ledger = Store(tmp_path).ledger()
    gw1 = ledger.filter(pl.col("target_gw") == 1)
    assert gw1.filter(pl.col("player_key") == 200)["actual_pts"].to_list() == [2]
    assert gw1["actual_pts"].null_count() == 0
