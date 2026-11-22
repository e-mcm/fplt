"""2×2 forecast × optimiser swap harness (north-star N4).

Arms are the Cartesian product of forecast sources and season optimisers.
Default arms: ``(ours | benchmark) × (ours | identity_milp)``. Outputs
``SeasonPointsSchema`` rows for promote-eval decisions later. No live FPL
Review fetch; callers supply projection frames.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from typing import cast

import polars as pl

from fplt.optimise.realised import summarise_decisions
from fplt.optimise.season_optimiser import (
    IdentityReferenceOptimiser,
    OursSeasonOptimiser,
    SeasonOptimiser,
    validate_decisions,
)
from fplt.optimise.state import SquadState
from fplt.schemas.swap import SeasonPointsSchema, SwapProjectionSchema

DEFAULT_FORECAST_SOURCES: tuple[str, ...] = ("ours", "benchmark")
DEFAULT_OPTIMISERS: tuple[SeasonOptimiser, ...] = (
    OursSeasonOptimiser(),
    IdentityReferenceOptimiser(),
)


@dataclass(frozen=True)
class SwapArm:
    """One cell of the forecast × optimiser grid."""

    forecast_source: str
    optimiser: SeasonOptimiser

    @property
    def arm_id(self) -> str:
        """Stable ``forecast|optimiser`` label."""
        return f"{self.forecast_source}|{self.optimiser.optimiser_id}"


def build_swap_arms(
    forecast_sources: Sequence[str] = DEFAULT_FORECAST_SOURCES,
    optimisers: Sequence[SeasonOptimiser] = DEFAULT_OPTIMISERS,
) -> tuple[SwapArm, ...]:
    """Build the Cartesian product of forecast sources and optimisers.

    Args:
        forecast_sources: Labels that must appear in the projection frame's
            ``source`` column.
        optimisers: Season-optimiser adapters to run for each source.

    Returns:
        Ordered arms (forecast major, optimiser minor).

    Raises:
        ValueError: Empty sources or optimisers, or a duplicate ``arm_id``.
    """
    if not forecast_sources:
        raise ValueError("forecast_sources must be non-empty")
    if not optimisers:
        raise ValueError("optimisers must be non-empty")
    arms = tuple(
        SwapArm(forecast_source=source, optimiser=optimiser)
        for source in forecast_sources
        for optimiser in optimisers
    )
    ids = [arm.arm_id for arm in arms]
    if len(ids) != len(set(ids)):
        raise ValueError(f"duplicate arm_id in swap grid: {ids}")
    return arms


def run_forecast_optimiser_swap(
    candidates: pl.DataFrame,
    projections: pl.DataFrame,
    actuals: pl.DataFrame,
    state: SquadState,
    *,
    season: str,
    gameweeks: Sequence[int],
    horizon: int,
    forecast_sources: Sequence[str] = DEFAULT_FORECAST_SOURCES,
    optimisers: Sequence[SeasonOptimiser] = DEFAULT_OPTIMISERS,
) -> pl.DataFrame:
    """Run every forecast × optimiser arm and return season-point rows.

    Args:
        candidates: ``gw``, ``player_key``, ``position``, ``team_key``,
            ``price`` for every deadline in ``gameweeks``.
        projections: Rows matching ``SwapProjectionSchema``.
        actuals: ``gw``, ``player_key``, ``actual_pts``, optional
            ``actual_minutes``.
        state: Starting squad for every arm (typically empty owned, bank 1000).
        season: Season label.
        gameweeks: Deadlines to play, in order.
        horizon: Planner look-ahead.
        forecast_sources: Which ``source`` values to include.
        optimisers: Which season optimisers to cross with those sources.

    Returns:
        A frame matching ``SeasonPointsSchema`` (one row per arm).

    Raises:
        ValueError: Bad grid, empty gameweeks, or a requested source missing
            from ``projections``.
    """
    weeks = list(gameweeks)
    if not weeks:
        raise ValueError("gameweeks must be non-empty")
    checked = SwapProjectionSchema.validate(projections)
    assert isinstance(checked, pl.DataFrame)
    arms = build_swap_arms(forecast_sources, optimisers)
    present = set(checked["source"].unique().to_list())
    missing = sorted({arm.forecast_source for arm in arms} - present)
    if missing:
        raise ValueError(f"projections missing forecast source(s): {missing}")

    rows: list[dict[str, object]] = []
    for arm in arms:
        part = checked.filter(pl.col("source") == arm.forecast_source).drop("source")
        decisions = validate_decisions(
            arm.optimiser.simulate_season(
                candidates,
                part,
                actuals,
                state,
                season=season,
                gameweeks=weeks,
                horizon=horizon,
                model_id=arm.arm_id,
            )
        )
        rows.append(_season_points_row(arm, season, decisions))
    return _season_points(pl.DataFrame(rows) if rows else _empty_season_points())


def _season_points_row(
    arm: SwapArm,
    season: str,
    decisions: pl.DataFrame,
) -> dict[str, object]:
    summary = summarise_decisions(decisions)
    if summary.is_empty():
        return {
            "arm_id": arm.arm_id,
            "forecast_source": arm.forecast_source,
            "optimiser_id": arm.optimiser.optimiser_id,
            "season": season,
            "total_points": 0,
            "gameweeks": 0,
            "mean_ppg": 0.0,
            "per_gw_json": "[]",
            "captaincy_regret": None,
            "xi_regret": None,
            "transfer_regret": None,
            "chip_regret": None,
        }
    row = summary.row(0, named=True)
    gameweeks = int(row["gameweeks"])
    total = int(row["total_points"])
    nets = _per_gw_nets(row["per_gw"])
    return {
        "arm_id": arm.arm_id,
        "forecast_source": arm.forecast_source,
        "optimiser_id": arm.optimiser.optimiser_id,
        "season": season,
        "total_points": total,
        "gameweeks": gameweeks,
        "mean_ppg": float(total) / float(gameweeks) if gameweeks else 0.0,
        "per_gw_json": json.dumps(nets),
        "captaincy_regret": None,
        "xi_regret": None,
        "transfer_regret": None,
        "chip_regret": None,
    }


def _per_gw_nets(raw: object) -> list[int]:
    """Coerce a ``summarise_decisions`` ``per_gw`` cell to a list of ints."""
    if raw is None:
        return []
    items = cast(Sequence[object], raw)
    return [int(cast(int, value)) for value in items]


def _season_points(frame: pl.DataFrame) -> pl.DataFrame:
    checked = SeasonPointsSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked


def _empty_season_points() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "arm_id": pl.Utf8(),
            "forecast_source": pl.Utf8(),
            "optimiser_id": pl.Utf8(),
            "season": pl.Utf8(),
            "total_points": pl.Int64(),
            "gameweeks": pl.Int64(),
            "mean_ppg": pl.Float64(),
            "per_gw_json": pl.Utf8(),
            "captaincy_regret": pl.Float64(),
            "xi_regret": pl.Float64(),
            "transfer_regret": pl.Float64(),
            "chip_regret": pl.Float64(),
        }
    )
