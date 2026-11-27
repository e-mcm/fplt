"""Forecast × optimiser 2×2 swap harness on tiny synthetic projections."""

from __future__ import annotations

import json

import polars as pl
import pytest
from pandera.errors import SchemaError

from fplt.backtest.forecast_optimiser_swap import (
    SwapArm,
    build_swap_arms,
    run_forecast_optimiser_swap,
)
from fplt.optimise.season_optimiser import (
    FplOptimizationToolsAdapter,
    IdentityReferenceOptimiser,
    OursSeasonOptimiser,
    freeze_reference_state,
)
from fplt.optimise.state import SquadState
from fplt.schemas.swap import SeasonPointsSchema, SwapProjectionSchema


def test_build_swap_arms_is_cartesian() -> None:
    arms = build_swap_arms(("ours", "benchmark"), (OursSeasonOptimiser(),))
    assert [arm.arm_id for arm in arms] == ["ours|ours", "benchmark|ours"]


def test_build_swap_arms_rejects_empty() -> None:
    with pytest.raises(ValueError, match="forecast_sources"):
        build_swap_arms((), (OursSeasonOptimiser(),))
    with pytest.raises(ValueError, match="optimisers"):
        build_swap_arms(("ours",), ())


def test_freeze_reference_state_caps_free_transfers() -> None:
    state = _state(free_transfers=2, ft_bank_cap=5, decay=0.9, hit_cost=8)
    frozen = freeze_reference_state(state)
    assert frozen.hit_cost == 4
    assert frozen.ft_bank_cap == 2
    assert frozen.decay == 1.0
    assert frozen.bench_weight == 0.0
    assert frozen.free_transfers == 2
    assert frozen.bank == state.bank


def test_identity_matches_ours_under_frozen_settings() -> None:
    candidates, projections, actuals = _tiny_world()
    state = freeze_reference_state(_state())
    result = run_forecast_optimiser_swap(
        candidates,
        projections,
        actuals,
        state,
        season="2024-25",
        gameweeks=[1, 2],
        horizon=1,
        forecast_sources=("ours",),
        optimisers=(OursSeasonOptimiser(), IdentityReferenceOptimiser()),
    )
    by_arm = {row["arm_id"]: row for row in result.iter_rows(named=True)}
    ours = by_arm["ours|ours"]
    reference = by_arm["ours|identity_milp"]
    assert ours["total_points"] == reference["total_points"]
    assert ours["per_gw_json"] == reference["per_gw_json"]
    assert ours["gameweeks"] == 2
    assert ours["mean_ppg"] == pytest.approx(ours["total_points"] / 2)


def test_benchmark_forecasts_change_season_points() -> None:
    candidates, projections, actuals = _tiny_world()
    # Flip who is valuable under the benchmark so the draft diverges.
    flipped = projections.with_columns(
        pl.when(pl.col("source") == "benchmark")
        .then(10.0 - pl.col("xpts"))
        .otherwise(pl.col("xpts"))
        .alias("xpts")
    )
    result = run_forecast_optimiser_swap(
        candidates,
        flipped,
        actuals,
        freeze_reference_state(_state()),
        season="2024-25",
        gameweeks=[1, 2],
        horizon=1,
        forecast_sources=("ours", "benchmark"),
        optimisers=(IdentityReferenceOptimiser(),),
    )
    SeasonPointsSchema.validate(result)
    by_source = {
        row["forecast_source"]: int(row["total_points"])
        for row in result.iter_rows(named=True)
    }
    assert by_source["ours"] != by_source["benchmark"]
    assert set(result["arm_id"].to_list()) == {
        "ours|identity_milp",
        "benchmark|identity_milp",
    }
    for row in result.iter_rows(named=True):
        assert row["captaincy_regret"] is not None
        assert float(row["captaincy_regret"]) >= 0.0
        assert row["chip_regret"] is None
        assert json.loads(row["per_gw_json"])


def test_full_default_grid_has_four_arms() -> None:
    candidates, projections, actuals = _tiny_world()
    result = run_forecast_optimiser_swap(
        candidates,
        projections,
        actuals,
        freeze_reference_state(_state()),
        season="2024-25",
        gameweeks=[1],
        horizon=1,
    )
    assert result.height == 4
    assert set(result["arm_id"].to_list()) == {
        "ours|ours",
        "ours|identity_milp",
        "benchmark|ours",
        "benchmark|identity_milp",
    }


def test_missing_forecast_source_raises() -> None:
    candidates, projections, actuals = _tiny_world()
    only_ours = projections.filter(pl.col("source") == "ours")
    with pytest.raises(ValueError, match="missing forecast source"):
        run_forecast_optimiser_swap(
            candidates,
            only_ours,
            actuals,
            _state(),
            season="2024-25",
            gameweeks=[1],
            horizon=1,
        )


def test_fpl_optimization_tools_adapter_is_stub() -> None:
    candidates, projections, actuals = _tiny_world()
    adapter = FplOptimizationToolsAdapter()
    assert adapter.optimiser_id == "fpl_optimization_tools"
    with pytest.raises(NotImplementedError, match="not an installable"):
        adapter.simulate_season(
            candidates,
            projections.filter(pl.col("source") == "ours").drop("source"),
            actuals,
            _state(),
            season="2024-25",
            gameweeks=[1],
            horizon=1,
            model_id="stub",
        )


def test_swap_projection_schema_rejects_dupes() -> None:
    frame = pl.DataFrame(
        {
            "source": ["ours", "ours"],
            "as_of_gw": [1, 1],
            "target_gw": [1, 1],
            "player_key": [1, 1],
            "xpts": [1.0, 2.0],
        }
    )
    with pytest.raises(SchemaError):
        SwapProjectionSchema.validate(frame)


def test_swap_arm_id() -> None:
    arm = SwapArm("ours", IdentityReferenceOptimiser())
    assert arm.arm_id == "ours|identity_milp"


def _state(**overrides: object) -> SquadState:
    values: dict[str, object] = {
        "bank": 1000,
        "free_transfers": 1,
        "ft_bank_cap": 2,
        "hit_cost": 4,
        "decay": 1.0,
        "bench_weight": 0.0,
        "owned": [],
    }
    values.update(overrides)
    return SquadState.model_validate(values)


def _tiny_world() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Build a legal 15-from-30 pool over two gameweeks with two forecast arms."""
    players: list[dict[str, object]] = []
    player = 1
    for club in range(1, 7):
        for position in ("GK", "DEF", "DEF", "MID", "FWD"):
            players.append(
                {
                    "player_key": player,
                    "position": position,
                    "team_key": club,
                    "price": 40 + club,
                    "base_xpts": float(club + (2 if position == "MID" else 0)),
                }
            )
            player += 1
    pool = pl.DataFrame(players)
    candidates = pl.concat(
        [
            pool.select("player_key", "position", "team_key", "price").with_columns(
                pl.lit(gw).alias("gw")
            )
            for gw in (1, 2)
        ]
    )
    projection_rows: list[dict[str, object]] = []
    actual_rows: list[dict[str, object]] = []
    for row in pool.iter_rows(named=True):
        key = int(row["player_key"])
        base = float(row["base_xpts"])
        for gw in (1, 2):
            for source in ("ours", "benchmark"):
                # Benchmark is a mild shift so both arms remain solvable.
                xpts = base if source == "ours" else base + 0.1 * (key % 3)
                projection_rows.append(
                    {
                        "source": source,
                        "as_of_gw": gw,
                        "target_gw": gw,
                        "player_key": key,
                        "xpts": xpts,
                    }
                )
            actual_rows.append(
                {
                    "gw": gw,
                    "player_key": key,
                    "actual_pts": int(base) + gw,
                    "actual_minutes": 90,
                }
            )
    projections = SwapProjectionSchema.validate(pl.DataFrame(projection_rows))
    assert isinstance(projections, pl.DataFrame)
    return candidates, projections, pl.DataFrame(actual_rows)
