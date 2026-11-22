"""Long-horizon MILP heuristics for H≥6 transfer plans.

Only week 1 is executed. Later steps exist so free transfers bank and
chips schedule correctly; they may be approximate. This module holds the
policy knobs and pick-shifting helpers used to seed HiGHS from a prior
plan. Relax-and-fix and coarsening hook off ``LongHorizonPolicy``.
"""

from __future__ import annotations

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, field_validator

# Auto-apply relax-and-fix / coarsening at this horizon and above (§4).
LONG_HORIZON_MIN = 6


class LongHorizonPolicy(BaseModel):
    """Optional MILP shrink knobs for long horizons.

    ``plan_squad`` applies enabled heuristics when
    ``horizon >= auto_min_horizon``. Warm starts are always available via
    ``warm_start=`` regardless of horizon.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    auto_min_horizon: int = Field(
        default=LONG_HORIZON_MIN,
        ge=2,
        description="Minimum horizon before relax-and-fix / coarsening auto-apply.",
    )
    relax_and_fix: bool = Field(
        default=True,
        description=(
            "Keep the first ``binary_steps`` steps integer; relax later steps "
            "to continuous, solve, then fix week-1 and re-solve."
        ),
    )
    binary_steps: int = Field(
        default=2,
        ge=1,
        description="Leading steps that stay binary under relax-and-fix.",
    )
    coarsen_from_step: int | None = Field(
        default=None,
        description=(
            "Drop captain/vice detail from this step onward (0-indexed). "
            "None disables coarsening."
        ),
    )

    @field_validator("coarsen_from_step")
    @classmethod
    def _coarsen_non_negative(cls, value: int | None) -> int | None:
        if value is not None and value < 0:
            raise ValueError(f"coarsen_from_step must be >= 0, got {value}")
        return value

    def should_auto_apply(self, horizon: int) -> bool:
        """True when ``horizon`` is long enough for these heuristics."""
        return horizon >= self.auto_min_horizon


# Relax-and-fix on at H≥6; coarsening stays off until a follow-up enables it.
DEFAULT_LONG_HORIZON_POLICY = LongHorizonPolicy()


def shift_horizon_picks(picks: pl.DataFrame, *, steps: int = 1) -> pl.DataFrame:
    """Drop the first ``steps`` horizon rows and renumber remaining steps from 0.

    Args:
        picks: Horizon picks (``step`` column required).
        steps: How many leading steps to discard (default 1 = last week → this).

    Returns:
        A copy with ``step' = step - steps`` for rows with ``step >= steps``.

    Raises:
        ValueError: ``steps`` is negative or ``step`` is missing.
    """
    if steps < 0:
        raise ValueError(f"steps must be >= 0, got {steps}")
    if "step" not in picks.columns:
        raise ValueError("picks must include a step column")
    if steps == 0:
        return picks
    return picks.filter(pl.col("step") >= steps).with_columns(
        (pl.col("step") - steps).alias("step")
    )
