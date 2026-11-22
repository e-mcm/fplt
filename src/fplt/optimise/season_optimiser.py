"""Season-optimiser protocol and reference adapters for the 2×2 swap.

``SeasonOptimiser`` is the thin boundary the forecast × optimiser harness
calls. ``OursSeasonOptimiser`` wraps ``simulate_decisions`` unchanged.
``IdentityReferenceOptimiser`` reuses the same MILP with frozen rule knobs so
the harness is testable without an external solver.

FPL-Optimization-Tools / open-fpl-solver is a CLI/script repository (pandas
CSV in, ``solve.py`` out), not an installable library with a season-sim API.
It does not cover ~90% of this Protocol without a large glue layer, so
``FplOptimizationToolsAdapter`` is a documented stub that raises until a
subprocess/CSV bridge is justified.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import polars as pl

from fplt.optimise.decisions import simulate_decisions
from fplt.optimise.state import SquadState
from fplt.schemas.plan import DecisionSchema

# Canonical chips-off decision-sim knobs. Bank / owned / free_transfers still
# come from the caller's starting state; only rule hyperparameters freeze.
REFERENCE_HIT_COST = 4
REFERENCE_FT_BANK_CAP = 2
REFERENCE_DECAY = 1.0
REFERENCE_BENCH_WEIGHT = 0.0


@runtime_checkable
class SeasonOptimiser(Protocol):
    """Play a season from opaque projections; return decision-ledger rows."""

    @property
    def optimiser_id(self) -> str:
        """Stable id written onto swap arm rows."""
        ...

    def simulate_season(
        self,
        candidates: pl.DataFrame,
        projections: pl.DataFrame,
        actuals: pl.DataFrame,
        state: SquadState,
        *,
        season: str,
        gameweeks: list[int],
        horizon: int,
        model_id: str,
    ) -> pl.DataFrame:
        """Simulate decisions for one forecast arm.

        Args:
            candidates: ``gw``, ``player_key``, ``position``, ``team_key``,
                ``price`` (same shape as ``simulate_decisions``).
            projections: ``as_of_gw``, ``target_gw``, ``player_key``, ``xpts``.
            actuals: ``gw``, ``player_key``, ``actual_pts``, optional
                ``actual_minutes``.
            state: Starting bank, free transfers, and owned players.
            season: Season label written onto every decision row.
            gameweeks: Deadlines to play, in order.
            horizon: Planner look-ahead in weeks.
            model_id: Label written into ``DecisionSchema.model_id``.

        Returns:
            A frame matching ``DecisionSchema``.
        """
        ...


@dataclass(frozen=True)
class OursSeasonOptimiser:
    """Production path: ``simulate_decisions`` with the caller's SquadState."""

    optimiser_id: str = "ours"

    def simulate_season(
        self,
        candidates: pl.DataFrame,
        projections: pl.DataFrame,
        actuals: pl.DataFrame,
        state: SquadState,
        *,
        season: str,
        gameweeks: list[int],
        horizon: int,
        model_id: str,
    ) -> pl.DataFrame:
        """Delegate to ``simulate_decisions`` without freezing rule knobs."""
        return simulate_decisions(
            candidates,
            projections,
            actuals,
            state,
            model_id=model_id,
            season=season,
            gameweeks=gameweeks,
            horizon=horizon,
        )


@dataclass(frozen=True)
class IdentityReferenceOptimiser:
    """Reference arm: our MILP with frozen hit/decay/banking/bench knobs.

    When the caller already uses these frozen settings, results match
    ``OursSeasonOptimiser`` (identity). Divergent knobs on ``state`` are
    overwritten so the reference stays comparable across runs.
    """

    optimiser_id: str = "identity_milp"

    def simulate_season(
        self,
        candidates: pl.DataFrame,
        projections: pl.DataFrame,
        actuals: pl.DataFrame,
        state: SquadState,
        *,
        season: str,
        gameweeks: list[int],
        horizon: int,
        model_id: str,
    ) -> pl.DataFrame:
        """Simulate with ``freeze_reference_state(state)``."""
        return simulate_decisions(
            candidates,
            projections,
            actuals,
            freeze_reference_state(state),
            model_id=model_id,
            season=season,
            gameweeks=gameweeks,
            horizon=horizon,
        )


@dataclass(frozen=True)
class FplOptimizationToolsAdapter:
    """Documented stub for sertalpbilal/FPL-Optimization-Tools (open-fpl-solver).

    The upstream project is a uv-managed script tree: projections as CSV under
    ``data/``, settings JSON, and ``run/solve.py``. There is no pip-installable
    season-sim entry point that accepts our ``SwapProjectionSchema`` /
    ``SquadState`` / ``DecisionSchema`` contracts. Bridging it would need a
    pandas CSV round-trip, settings translation, and a pick→decision mapper —
    well below the ~90% coverage bar for adopting it as the reference today.
    """

    optimiser_id: str = "fpl_optimization_tools"

    def simulate_season(
        self,
        candidates: pl.DataFrame,
        projections: pl.DataFrame,
        actuals: pl.DataFrame,
        state: SquadState,
        *,
        season: str,
        gameweeks: list[int],
        horizon: int,
        model_id: str,
    ) -> pl.DataFrame:
        """Refuse until a justified subprocess/CSV bridge exists.

        Raises:
            NotImplementedError: Always; the adapter is a stub.
        """
        _ = (
            candidates,
            projections,
            actuals,
            state,
            season,
            gameweeks,
            horizon,
            model_id,
        )
        raise NotImplementedError(
            "FPL-Optimization-Tools / open-fpl-solver is not an installable "
            "season-sim library for this Protocol (~CLI/CSV workflow). Use "
            "IdentityReferenceOptimiser until a thin subprocess adapter is "
            "justified; see docs/forecast-optimiser-swap.md in the Project store."
        )


def freeze_reference_state(state: SquadState) -> SquadState:
    """Copy ``state`` with the canonical reference rule hyperparameters.

    Args:
        state: Starting squad (bank, owned, free transfers preserved).

    Returns:
        A ``SquadState`` with frozen hit cost, FT bank cap, decay, and bench
        weight. Free transfers are capped at the frozen bank cap.
    """
    free = min(state.free_transfers, REFERENCE_FT_BANK_CAP)
    if free < 1:
        free = 1
    return state.model_copy(
        update={
            "hit_cost": REFERENCE_HIT_COST,
            "ft_bank_cap": REFERENCE_FT_BANK_CAP,
            "decay": REFERENCE_DECAY,
            "bench_weight": REFERENCE_BENCH_WEIGHT,
            "free_transfers": free,
        }
    )


def validate_decisions(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate a season-optimiser output against ``DecisionSchema``.

    Args:
        frame: Candidate decision rows.

    Returns:
        The validated polars frame.
    """
    if "chip" not in frame.columns:
        frame = frame.with_columns(pl.lit(None).cast(pl.Utf8).alias("chip"))
    checked = DecisionSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked
