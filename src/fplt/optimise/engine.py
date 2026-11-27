"""Transfer MILP boundary: opaque expected points in, squad decisions out.

Does not import predictors, ledgers, or the data store. The solver is the
multi-period HiGHS model in ``plan_squad``.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import polars as pl

from fplt.optimise.plan import GameweekPlan, plan_squad
from fplt.optimise.prices import apply_market_prices, market_prices_from_candidates
from fplt.optimise.state import SquadState
from fplt.schemas.plan import (
    CandidateSchema,
    ExpectedPointsSchema,
    HorizonPickSchema,
    HorizonPointsSchema,
)
from fplt.utils.frames import require_dataframe


@dataclass(frozen=True)
class TransferPlan:
    """Solved horizon with the first week's actionable transfers."""

    gameweeks: tuple[int, ...]
    picks: pl.DataFrame
    hits: tuple[float, ...]
    free_transfers: tuple[int, ...]
    points: float

    @property
    def is_draft(self) -> bool:
        """Whether the first step is an initial fifteen, not a transfer window."""
        return bool(self.free_transfers) and self.free_transfers[0] == 0

    @property
    def transfers_in(self) -> tuple[int, ...]:
        """Player keys bought at the first horizon step.

        An initial draft marks every pick as bought in the pick table; those
        are not transfers, so this property returns an empty tuple.
        """
        if self.is_draft:
            return ()
        return _keys(self.picks, bought=True)

    @property
    def transfers_out(self) -> tuple[int, ...]:
        """Player keys sold at the first horizon step."""
        if self.is_draft:
            return ()
        return _keys(self.picks, sold=True)

    @property
    def xi(self) -> tuple[int, ...]:
        """Starting XI player keys at the first horizon step."""
        return _keys(self.picks, in_xi=True)

    @property
    def captain(self) -> int | None:
        """Captain player key at the first horizon step, if any."""
        keys = _keys(self.picks, captain=True)
        return keys[0] if keys else None

    @property
    def vice(self) -> int | None:
        """Vice-captain player key at the first horizon step, if any."""
        keys = _keys(self.picks, vice=True)
        return keys[0] if keys else None

    @property
    def hit(self) -> float:
        """Hit cost applied at the first horizon step."""
        return self.hits[0] if self.hits else 0.0


def horizon_points(
    expected_points: pl.DataFrame,
    gameweeks: Sequence[int],
) -> pl.DataFrame:
    """Map player-gameweek expected points onto horizon steps.

    Args:
        expected_points: Rows matching ``ExpectedPointsSchema``. Predictions
            are opaque: only ``player_key``, ``gw``, and ``xpts`` are read.
        gameweeks: Target gameweeks in horizon order. Step ``i`` is
            ``gameweeks[i]``. Missing player-week pairs score zero.

    Returns:
        A frame matching ``HorizonPointsSchema``.

    Raises:
        ValueError: ``gameweeks`` is empty or repeats a gameweek.
    """
    window = _window(gameweeks)
    scored = ExpectedPointsSchema.validate(expected_points)
    scored = require_dataframe(scored)
    if scored.is_empty():
        return _empty_horizon()
    step_of = {gameweek: step for step, gameweek in enumerate(window)}
    chosen = scored.filter(pl.col("gw").is_in(window))
    if chosen.is_empty():
        return _empty_horizon()
    frame = (
        chosen.group_by("player_key", "gw")
        .agg(pl.col("xpts").sum())
        .with_columns(pl.col("gw").replace_strict(step_of).alias("step"))
        .select("player_key", "step", "xpts")
    )
    checked = HorizonPointsSchema.validate(frame)
    checked = require_dataframe(checked)
    return checked


def optimize_transfers(
    candidates: pl.DataFrame,
    expected_points: pl.DataFrame,
    state: SquadState,
    *,
    gameweeks: Sequence[int],
) -> TransferPlan:
    """Solve transfers, XI, and captaincy from opaque expected points.

    Args:
        candidates: Rows matching ``CandidateSchema`` (prices, positions,
            clubs). Not predictions.
        expected_points: Rows matching ``ExpectedPointsSchema``. Only the
            listed gameweeks matter; others are ignored.
        state: Bank, free transfers, hit rules, and owned selling prices.
            An empty ``owned`` list is an initial draft (no transfer hits).
        gameweeks: Horizon targets in order. The first week is the one to
            act on; later weeks exist so free transfers can be banked.

    Returns:
        First-week transfers plus the full horizon picks.

    Raises:
        ValueError: The horizon is empty, inputs disagree, or no squad
            meets the budget, position, and club rules.
    """
    window = _window(gameweeks)
    pool = CandidateSchema.validate(candidates)
    pool = require_dataframe(pool)
    state = apply_market_prices(state, market_prices_from_candidates(pool))
    solved = plan_squad(
        pool,
        horizon_points(expected_points, window),
        state,
        initial=not state.owned,
        horizon=len(window),
    )
    return _from_plan(solved, window)


def _from_plan(plan: GameweekPlan, gameweeks: Sequence[int]) -> TransferPlan:
    picks = HorizonPickSchema.validate(plan.picks)
    picks = require_dataframe(picks)
    return TransferPlan(
        gameweeks=tuple(gameweeks),
        picks=picks,
        hits=plan.hits,
        free_transfers=plan.free_transfers,
        points=plan.points,
    )


def _window(gameweeks: Sequence[int]) -> list[int]:
    window = list(gameweeks)
    if not window:
        raise ValueError("gameweeks must be non-empty")
    if len(set(window)) != len(window):
        raise ValueError("gameweeks must not repeat a gameweek")
    for gameweek in window:
        if gameweek < 1 or gameweek > 38:
            raise ValueError(f"gameweek {gameweek} is outside 1..38")
    return window


def _keys(picks: pl.DataFrame, **flags: bool) -> tuple[int, ...]:
    part = picks.filter(pl.col("step") == 0)
    for name, wanted in flags.items():
        part = part.filter(pl.col(name) == wanted)
    return tuple(int(key) for key in part["player_key"].sort().to_list())


def _empty_horizon() -> pl.DataFrame:
    return pl.DataFrame(
        schema={"player_key": pl.Int64(), "step": pl.Int64(), "xpts": pl.Float64()}
    )
