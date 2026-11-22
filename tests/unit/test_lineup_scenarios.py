"""Sample-average lineup selection on a fixed fifteen."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fplt.optimise.lineup import (
    select_lineup_expected,
    select_lineup_scenarios,
    verify_lineup_mean,
)
from fplt.optimise.scenarios import ScenarioBundle


def test_select_lineup_prefers_bench_insurance() -> None:
    """Risky XI star is benched when a safe sub scores in the blank scenarios."""
    squad = _squad()
    keys = squad["player_key"].to_list()
    n = len(keys)
    # Two sims. Player 11 (FWD) is high EV but blanks in sim1; bench MID 13
    # always scores 6. Starting 11 is only optimal if we ignore blanks.
    points = np.zeros((n, 2), dtype=np.float64)
    minutes = np.full((n, 2), 90, dtype=np.int64)
    for key in keys:
        idx = keys.index(key)
        points[idx] = 2.0
    # Inflate FWD 11 in sim0 only; blank in sim1.
    i11 = keys.index(11)
    points[i11, 0] = 10.0
    points[i11, 1] = 0.0
    minutes[i11, 1] = 0
    i13 = keys.index(13)
    points[i13] = 6.0
    minutes[i13] = 90
    bundle = ScenarioBundle(
        player_keys=np.asarray(keys, dtype=np.int64),
        points=points,
        minutes=minutes,
    )
    chosen = select_lineup_scenarios(squad, bundle)
    assert chosen.mean_points == pytest.approx(
        verify_lineup_mean(chosen, bundle), rel=0, abs=1e-9
    )
    # Either 11 starts and 13 is first outfield bench, or 13 starts — both
    # capture the insurance. Assert the mean beats "naive always start 11
    # with 13 last on bench and no autosub credit" by checking autosub path.
    assert chosen.players.filter(pl.col("in_xi")).height == 11
    assert int(chosen.players["captain"].sum()) == 1
    assert int(chosen.players["vice"].sum()) == 1
    bench = chosen.players.filter(~pl.col("in_xi")).sort("bench_slot")
    assert bench.height == 4
    assert set(bench["bench_slot"].to_list()) == {1, 2, 3, 4}


def test_p_haul_breaks_captain_ties() -> None:
    squad = _squad().with_columns(
        pl.when(pl.col("player_key") == 9)
        .then(0.9)
        .when(pl.col("player_key") == 10)
        .then(0.1)
        .otherwise(0.0)
        .alias("p_haul"),
        pl.lit(5.0).alias("xpts"),
    )
    # Equal xpts everywhere so EV mean is flat; haul should prefer player 9.
    chosen = select_lineup_expected(squad, haul_weight=1.0)
    captain = chosen.players.filter(pl.col("captain"))["player_key"][0]
    assert int(captain) == 9


def test_select_lineup_rejects_bad_squad() -> None:
    squad = _squad().head(14)
    bundle = ScenarioBundle(
        player_keys=np.arange(1, 15, dtype=np.int64),
        points=np.zeros((14, 1)),
        minutes=np.zeros((14, 1), dtype=np.int64),
    )
    with pytest.raises(ValueError, match="15"):
        select_lineup_scenarios(squad, bundle)


def _squad() -> pl.DataFrame:
    positions = [
        "GK",
        "DEF",
        "DEF",
        "DEF",
        "MID",
        "MID",
        "MID",
        "MID",
        "FWD",
        "FWD",
        "FWD",
        "GK",
        "MID",
        "DEF",
        "DEF",
    ]
    return pl.DataFrame(
        {
            "player_key": list(range(1, 16)),
            "position": positions,
            "team_key": [1, 1, 2, 2, 3, 3, 4, 4, 5, 5, 6, 6, 1, 2, 3],
            "price": [45] * 15,
            "xpts": [2.0] * 15,
        }
    )
