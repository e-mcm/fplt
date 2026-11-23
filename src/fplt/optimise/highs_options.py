"""HiGHS solve options for the transfer MILP.

open-fpl-solver (FPL-Optimization-Tools) sets ``time_limit`` (``secs``),
``mip_rel_gap`` (``gap``), ``presolve``, and ``parallel``. We ship a 600s
time limit and a small ``mip_rel_gap`` (1e-3) so fat H≥6 models finish;
reference / harness solves still pass ``mip_rel_gap=0`` when proving
optimality matters.
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
        default=1e-3,
        ge=0.0,
        description=(
            "Relative MIP gap. 0 proves optimality when the solve finishes. "
            "Production default is 1e-3 so fat H≥6 models stop earlier; "
            "week-1 actions stay stable on the pruning-loss harness. "
            "Use 0 for reference / equivalence solves."
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
        """Return production HiGHS settings (600s, gap 1e-3, parallel on).

        open-fpl uses gap 0; we ship a small relative gap so long-horizon
        MILPs finish sooner without changing executable week-1 actions.
        """
        return cls(
            time_limit=600.0,
            mip_rel_gap=1e-3,
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

    @classmethod
    def board_deterministic(cls) -> HighsSolveOptions:
        """Return reproducible board knobs (no time limit, 1 thread, seed 0).

        Chip / season-points boards use this so a second run on the same tip
        and staged data yields identical totals (``docs/eval-protocol.md`` §5).
        Live ``fplt plan`` keeps ``open_fpl_defaults`` so unattended solves
        stay wall-clock bounded.

        ``threads=1`` is required for determinism: even with ``parallel="off"``,
        HiGHS still uses the ``threads`` pool for MIP symmetry detection and
        clique-table work, which can pick different alternate optima across
        re-runs. ``apply`` resets the thread-local scheduler before pinning
        so a prior prune / default-thread solve cannot leave ``kNotset``.
        """
        return cls(
            time_limit=None,
            mip_rel_gap=0.0,
            output_flag=False,
            presolve="on",
            parallel="off",
            threads=1,
            random_seed=0,
            accept_time_limit=False,
        )

    def apply(self, solver: highspy.Highs) -> None:
        """Write these options onto ``solver``.

        When ``threads`` is set, reset HiGHS' thread-local scheduler first.
        The first ``run()`` / ``presolve()`` on a thread binds the scheduler
        size; candidate prune (and other default-thread HiGHS calls) often
        initialize it to N cores. Setting a different nonzero ``threads``
        after that returns ``kError`` and leaves ``HighsModelStatus.kNotset``.
        """
        solver.setOptionValue("output_flag", self.output_flag)
        solver.setOptionValue("mip_rel_gap", self.mip_rel_gap)
        solver.setOptionValue("presolve", self.presolve)
        if self.time_limit is not None:
            solver.setOptionValue("time_limit", float(self.time_limit))
        if self.parallel is not None:
            solver.setOptionValue("parallel", self.parallel)
        if self.threads is not None:
            highspy.Highs.resetThreadScheduler(True)
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
