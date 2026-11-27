"""Pandera schemas for the candidate-pruning loss harness."""

from __future__ import annotations

import pandera.polars as pa


class PruningLossRowSchema(pa.DataFrameModel):
    """One scenario × prune-arm comparison against the reference solve.

    ``week1_agree`` is the primary metric (identical transfers, XI, captain,
    and vice). ``objective_gap`` is reference objective minus the arm's
    objective in decayed points — descriptive only; do not gate pytest on it.
    """

    scenario_id: str
    scenario_tag: str
    arm_id: str
    target_pool: int = pa.Field(nullable=True)
    pool_size: int = pa.Field(ge=0)
    week1_agree: bool
    objective: float
    objective_gap: float
    solve_elapsed: float = pa.Field(ge=0.0, nullable=True)
    mip_status: str = pa.Field(nullable=True)
    horizon: int = pa.Field(ge=1)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = ["scenario_id", "arm_id"]  # noqa: RUF012
