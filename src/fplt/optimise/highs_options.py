"""HiGHS solve options for the transfer MILP.

open-fpl-solver (FPL-Optimization-Tools) sets ``time_limit`` (``secs``),
``mip_rel_gap`` (``gap``), ``presolve``, and ``parallel``. We previously
only forced ``mip_rel_gap=0`` with no wall-clock limit, so fat H=8 models
could hang indefinitely. This policy makes those knobs explicit.
"""

from __future__ import annotations

import logging

import highspy
from pydantic import BaseModel, ConfigDict, Field, field_validator

_log = logging.getLogger(__name__)


class HighsSolveOptions(BaseModel):
    """HiGHS knobs applied before ``solver.run()``.

    When ``time_limit`` elapses with a feasible incumbent,
    ``plan_squad`` accepts ``HighsModelStatus.kTimeLimit`` instead of
    requiring proven ``kOptimal`` (unless ``accept_time_limit`` is false).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    time_limit: float | None = Field(
        default=600.0,
        description=(
            "Wall-clock seconds for HiGHS. None disables the limit. "
            "open-fpl default is 600."
        ),
    )
    mip_rel_gap: float = Field(
        default=0.0,
        ge=0.0,
        description=(
            "Relative MIP gap. 0 proves optimality when the solve finishes. "
            "open-fpl ``gap`` default is 0; try 1e-3 to stop earlier."
        ),
    )
    output_flag: bool = Field(
        default=False,
        description="HiGHS console logging.",
    )
    presolve: str = Field(
        default="choose",
        description="HiGHS presolve: choose / on / off.",
    )
    parallel: str | None = Field(
        default="on",
        description="HiGHS parallel mode: on / off. None leaves the default.",
    )
    threads: int | None = Field(
        default=None,
        ge=0,
        description="Thread count. None / 0 leaves HiGHS default.",
    )
    random_seed: int | None = Field(
        default=None,
        ge=0,
        description="Optional MIP random seed for reproducibility.",
    )
    accept_time_limit: bool = Field(
        default=True,
        description=(
            "Treat ``kTimeLimit`` with a primal solution as success. "
            "If false, only ``kOptimal`` is accepted."
        ),
    )

    @field_validator("time_limit")
    @classmethod
    def _positive_time(cls, value: float | None) -> float | None:
        if value is not None and value <= 0:
            raise ValueError(f"time_limit must be > 0 when set, got {value}")
        return value

    @field_validator("presolve")
    @classmethod
    def _presolve_token(cls, value: str) -> str:
        allowed = {"choose", "on", "off"}
        if value not in allowed:
            raise ValueError(
                f"presolve must be one of {sorted(allowed)}, got {value!r}"
            )
        return value

    @field_validator("parallel")
    @classmethod
    def _parallel_token(cls, value: str | None) -> str | None:
        if value is None:
            return value
        allowed = {"on", "off"}
        if value not in allowed:
            raise ValueError(
                f"parallel must be one of {sorted(allowed)}, got {value!r}"
            )
        return value

    @classmethod
    def open_fpl_defaults(cls) -> HighsSolveOptions:
        """Return open-fpl-like HiGHS settings (600s, gap 0, parallel on)."""
        return cls(
            time_limit=600.0,
            mip_rel_gap=0.0,
            output_flag=False,
            presolve="on",
            parallel="on",
            accept_time_limit=True,
        )

    @classmethod
    def unlimited_optimal(cls) -> HighsSolveOptions:
        """Return the pre-policy behaviour: no time limit, prove optimal."""
        return cls(
            time_limit=None,
            mip_rel_gap=0.0,
            output_flag=False,
            presolve="choose",
            parallel=None,
            accept_time_limit=False,
        )

    def apply(self, solver: highspy.Highs) -> None:
        """Write these options onto ``solver``."""
        solver.setOptionValue("output_flag", self.output_flag)
        solver.setOptionValue("mip_rel_gap", self.mip_rel_gap)
        solver.setOptionValue("presolve", self.presolve)
        if self.time_limit is not None:
            solver.setOptionValue("time_limit", float(self.time_limit))
        if self.parallel is not None:
            solver.setOptionValue("parallel", self.parallel)
        if self.threads is not None:
            solver.setOptionValue("threads", int(self.threads))
        if self.random_seed is not None:
            solver.setOptionValue("random_seed", int(self.random_seed))

    def accepts_status(self, status: highspy.HighsModelStatus) -> bool:
        """True when ``status`` is good enough to read a plan from."""
        if status == highspy.HighsModelStatus.kOptimal:
            return True
        if self.accept_time_limit and status == highspy.HighsModelStatus.kTimeLimit:
            return True
        return False


DEFAULT_HIGHS_OPTIONS = HighsSolveOptions.open_fpl_defaults()


def log_solve_status(
    status: highspy.HighsModelStatus,
    *,
    options: HighsSolveOptions,
) -> None:
    """Log a non-optimal but accepted solve status once."""
    if status == highspy.HighsModelStatus.kTimeLimit:
        _log.warning(
            "HiGHS hit time_limit=%ss (mip_rel_gap=%s); using incumbent",
            options.time_limit,
            options.mip_rel_gap,
        )
