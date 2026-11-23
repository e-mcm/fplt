"""L6 native MC system-equivalence gates (native-mc-build-spec §4).

Compares python ``dc_layered_v4`` ledgers to ``dc_layered_v4_native`` under
paired seeds. Auto promote-eval stays ``inconclusive`` for improvement kinds;
this module is the written L6 board readout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import polars as pl

from fplt.backtest.intervals import (
    PairedInterval,
    interval_to_dict,
    intervals_for_pair,
    paired_interval,
)

# native-mc-build-spec §4 L6: n_sims study 10k-vs-20k jitter ≈ 0.011;
# two independent 10k runs ≈ √2 of that → bar at 0.02.
MEAN_ABS_XPTS_MAX = 0.02

_JOIN_KEYS = (
    "as_of_season",
    "as_of_gw",
    "horizon",
    "player_key",
    "fixture_id",
)


@dataclass(frozen=True)
class L6GateResult:
    """Pass/fail for one L6 gate with supporting numbers."""

    name: str
    passed: bool
    detail: dict[str, Any]


@dataclass(frozen=True)
class L6Board:
    """Full L6 gate board for one paired window."""

    mean_abs_xpts: float
    n_rows: int
    gates: tuple[L6GateResult, ...]
    intervals: tuple[dict[str, Any], ...]

    @property
    def passed(self) -> bool:
        """True when every assessed gate passed."""
        return all(gate.passed for gate in self.gates)


def _ci_covers_zero(interval: PairedInterval) -> bool:
    """Return whether any available CI interval covers zero."""
    checks: list[bool] = []
    if interval.hac_ci_low is not None and interval.hac_ci_high is not None:
        checks.append(interval.hac_ci_low <= 0.0 <= interval.hac_ci_high)
    if (
        interval.bootstrap_ci_low is not None
        and interval.bootstrap_ci_high is not None
    ):
        checks.append(
            interval.bootstrap_ci_low <= 0.0 <= interval.bootstrap_ci_high
        )
    if not checks:
        # Too few gameweeks for a CI — do not fail the gate on missing bounds.
        return True
    return any(checks)


def mean_abs_xpts_delta(
    control: pl.DataFrame, treatment: pl.DataFrame
) -> tuple[float, int]:
    """Return mean |ΔxPts| and joined row count for paired prediction ledgers.

    Args:
        control: Control prediction rows (python MC).
        treatment: Treatment prediction rows (native MC).

    Returns:
        ``(mean_abs_delta, n_joined_rows)``.
    """
    left = control.select(*_JOIN_KEYS, "xpts").rename({"xpts": "xpts_c"})
    right = treatment.select(*_JOIN_KEYS, "xpts").rename({"xpts": "xpts_t"})
    joined = left.join(right, on=list(_JOIN_KEYS), how="inner")
    if joined.is_empty():
        return 0.0, 0
    delta = (joined["xpts_t"] - joined["xpts_c"]).abs()
    return float(delta.mean()), int(joined.height)


def signed_xpts_delta_intervals(
    control: pl.DataFrame,
    treatment: pl.DataFrame,
    *,
    positions: tuple[str, ...] = ("GK", "DEF", "MID", "FWD"),
) -> list[PairedInterval]:
    """Per-position (and overall) signed mean ΔxPts intervals (treatment − control).

    Builds a per-gameweek series of mean predicted-xPts differences, then the
    usual HAC / stationary-bootstrap intervals. Positive mean means native
    predicts higher xPts than python.

    Args:
        control: Control prediction rows.
        treatment: Treatment prediction rows.
        positions: Position slices to report.

    Returns:
        Overall interval first, then one per position that has rows.
    """
    left = control.select(*_JOIN_KEYS, "position", "xpts").rename(
        {"xpts": "xpts_c"}
    )
    right = treatment.select(*_JOIN_KEYS, "xpts").rename({"xpts": "xpts_t"})
    joined = left.join(right, on=list(_JOIN_KEYS), how="inner")
    if joined.is_empty():
        return []
    joined = joined.with_columns(
        (pl.col("xpts_t") - pl.col("xpts_c")).alias("delta_xpts")
    )
    results: list[PairedInterval] = []
    slices: list[str | None] = [None, *positions]
    for position in slices:
        part = joined if position is None else joined.filter(pl.col("position") == position)
        if part.is_empty():
            continue
        per_gw = (
            part.group_by("as_of_gw")
            .agg(pl.col("delta_xpts").mean().alias("d"))
            .sort("as_of_gw")
        )
        series = [float(v) for v in per_gw["d"].to_list() if v is not None]
        results.append(
            paired_interval(
                series,
                metric="delta_xpts",
                position=position,
            )
        )
    return results


def assess_l6_gates(
    control: pl.DataFrame,
    treatment: pl.DataFrame,
    *,
    control_id: str,
    treatment_id: str,
    scored_ledger: pl.DataFrame | None = None,
) -> L6Board:
    """Evaluate L6 gates on paired prediction frames.

    Args:
        control: Control (python) prediction rows for the window.
        treatment: Treatment (native) prediction rows for the window.
        control_id: Control ``model_id`` (for haul-Brier intervals).
        treatment_id: Treatment ``model_id``.
        scored_ledger: Optional scored ledger covering both arms (needs
            ``actual_pts`` / ``p_haul``). When absent, the haul-Brier gate is
            skipped and recorded as not assessed.

    Returns:
        ``L6Board`` with per-gate pass/fail. Season-points gate is out of
        scope here (decision / sim-season board).
    """
    mean_abs, n_rows = mean_abs_xpts_delta(control, treatment)
    gates: list[L6GateResult] = [
        L6GateResult(
            name="mean_abs_xpts",
            passed=n_rows > 0 and mean_abs <= MEAN_ABS_XPTS_MAX,
            detail={
                "mean_abs_xpts": mean_abs,
                "threshold": MEAN_ABS_XPTS_MAX,
                "n_rows": n_rows,
            },
        )
    ]
    xpts_intervals = signed_xpts_delta_intervals(control, treatment)
    for item in xpts_intervals:
        label = "overall" if item.position is None else item.position
        gates.append(
            L6GateResult(
                name=f"signed_xpts_ci_{label}",
                passed=_ci_covers_zero(item),
                detail=interval_to_dict(item),
            )
        )

    interval_payloads = [interval_to_dict(item) for item in xpts_intervals]
    if scored_ledger is not None and not scored_ledger.is_empty():
        haul = intervals_for_pair(
            scored_ledger,
            control_id=control_id,
            treatment_id=treatment_id,
            metrics=("brier_haul",),
        )
        for item in haul:
            label = "overall" if item.position is None else item.position
            gates.append(
                L6GateResult(
                    name=f"haul_brier_ci_{label}",
                    passed=_ci_covers_zero(item),
                    detail=interval_to_dict(item),
                )
            )
            interval_payloads.append(interval_to_dict(item))
    else:
        gates.append(
            L6GateResult(
                name="haul_brier_ci",
                passed=True,
                detail={"skipped": True, "reason": "no scored ledger"},
            )
        )

    return L6Board(
        mean_abs_xpts=mean_abs,
        n_rows=n_rows,
        gates=tuple(gates),
        intervals=tuple(interval_payloads),
    )


__all__ = [
    "MEAN_ABS_XPTS_MAX",
    "L6Board",
    "L6GateResult",
    "assess_l6_gates",
    "mean_abs_xpts_delta",
    "signed_xpts_delta_intervals",
]
