"""Locked-fifteen formation checks for plan picks and owned squads.

Under ``LongHorizonPolicy.relax_and_fix``, far horizon steps stay continuous.
Thresholding those LP values at 0.5 can yield fifteen rows whose position
counts are not ``{GK:2, DEF:5, MID:5, FWD:3}`` even though the LP satisfied
the fractional equalities. That is an extraction artefact, not a team we
field: only binary / actionable steps must be legal locked fifteens.

Use these helpers to:
- assert week-1 (and other binary steps) after ``plan_squad``;
- decide whether ``lineup_reopt`` may treat a step as a locked fifteen;
- dump per-step integrity for post-mortems / ``--explain-gw``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import polars as pl

_log = logging.getLogger(__name__)

# FPL full-squad quotas (same as the MILP ``_SQUAD`` constraints).
SQUAD_POSITION_QUOTA: Mapping[str, int] = {
    "GK": 2,
    "DEF": 5,
    "MID": 5,
    "FWD": 3,
}
SQUAD_SIZE = 15
_POS = ("GK", "DEF", "MID", "FWD")


@dataclass(frozen=True)
class StepSquadIntegrity:
    """Formation check for one horizon step's kept (non-sold) rows."""

    step: int
    chip: str | None
    n_kept: int
    counts: Mapping[str, int]
    player_keys: tuple[int, ...]
    issue: str | None

    @property
    def ok(self) -> bool:
        """True when kept rows form a legal locked fifteen."""
        return self.issue is None


def squad_position_counts(squad: pl.DataFrame) -> dict[str, int]:
    """Count ``position`` values; unknown labels appear as their own keys.

    Args:
        squad: Frame with a ``position`` column.

    Returns:
        Counts for GK/DEF/MID/FWD (0 when absent) plus any unknown labels.
    """
    counts = {position: 0 for position in _POS}
    if "position" not in squad.columns:
        return counts
    for position in squad["position"].to_list():
        label = str(position)
        counts[label] = counts.get(label, 0) + 1
    return counts


def locked_fifteen_issue(squad: pl.DataFrame) -> str | None:
    """Return a short issue string when ``squad`` is not a legal fifteen.

    Args:
        squad: Candidate locked fifteen (``player_key``, ``position``).

    Returns:
        ``None`` when valid; otherwise a message suitable for logs / raises.
    """
    required = {"player_key", "position"}
    missing = required - set(squad.columns)
    if missing:
        return f"squad missing columns: {sorted(missing)}"
    if squad["player_key"].n_unique() != squad.height:
        return "player_key must be unique"
    if squad.height != SQUAD_SIZE:
        return f"squad must have 15 players, got {squad.height}"
    counts = squad_position_counts(squad)
    unknown = sorted(key for key in counts if key not in SQUAD_POSITION_QUOTA)
    if unknown:
        return f"unknown position(s) {unknown}"
    expected = dict(SQUAD_POSITION_QUOTA)
    got = {position: counts[position] for position in _POS}
    if got != expected:
        return f"squad position counts must be {expected}, got {got}"
    return None


def kept_rows_for_step(picks: pl.DataFrame, step: int) -> pl.DataFrame:
    """Non-sold pick rows for one horizon step (the candidate locked fifteen)."""
    part = picks.filter(pl.col("step") == step)
    if "sold" in part.columns:
        return part.filter(~pl.col("sold"))
    return part


def inspect_horizon_picks(
    picks: pl.DataFrame,
    chips: Sequence[str | None] | None = None,
) -> list[StepSquadIntegrity]:
    """Per-step locked-fifteen integrity for a ``GameweekPlan.picks`` frame.

    Args:
        picks: Horizon picks (``step``, ``player_key``, ``position``, ``sold``).
        chips: Optional per-step chip labels (same length as the horizon).

    Returns:
        One ``StepSquadIntegrity`` per distinct ``step``, sorted ascending.
    """
    if picks.is_empty() or "step" not in picks.columns:
        return []
    steps = sorted(int(value) for value in picks["step"].unique().to_list())
    rows: list[StepSquadIntegrity] = []
    for step in steps:
        kept = kept_rows_for_step(picks, step)
        chip = None
        if chips is not None and 0 <= step < len(chips):
            chip = chips[step]
        keys = tuple(
            int(key) for key in kept.sort("player_key")["player_key"].to_list()
        )
        counts = squad_position_counts(kept)
        rows.append(
            StepSquadIntegrity(
                step=step,
                chip=chip,
                n_kept=kept.height,
                counts=counts,
                player_keys=keys,
                issue=locked_fifteen_issue(kept),
            )
        )
    return rows


def format_step_integrity(row: StepSquadIntegrity) -> str:
    """One-line summary for logs and error messages."""
    chip = row.chip if row.chip is not None else "-"
    counts = {position: row.counts.get(position, 0) for position in _POS}
    if row.ok:
        return f"step={row.step} chip={chip} n={row.n_kept} counts={counts} ok"
    return (
        f"step={row.step} chip={chip} n={row.n_kept} counts={counts} "
        f"issue={row.issue!r} keys={list(row.player_keys)}"
    )


def require_locked_steps(
    integrity: Sequence[StepSquadIntegrity],
    steps: Sequence[int],
    *,
    context: str,
) -> None:
    """Raise if any required step fails the locked-fifteen check.

    Args:
        integrity: Output of ``inspect_horizon_picks``.
        steps: Steps that must be legal (typically actionable week-1 / binary).
        context: Prefix for the error (e.g. ``plan_squad after kOptimal``).

    Raises:
        ValueError: A required step is not a legal locked fifteen.
    """
    by_step = {row.step: row for row in integrity}
    for step in steps:
        row = by_step.get(int(step))
        if row is None:
            raise ValueError(f"{context}: missing pick rows for required step {step}")
        if row.ok:
            continue
        raise ValueError(f"{context}: {format_step_integrity(row)}")


def log_horizon_integrity(
    integrity: Sequence[StepSquadIntegrity],
    *,
    binary_steps: Sequence[int] | None = None,
) -> None:
    """Log per-step integrity; WARN on non-integral far steps, INFO on binary."""
    binary = set(int(step) for step in binary_steps) if binary_steps else {0}
    for row in integrity:
        line = format_step_integrity(row)
        if row.ok:
            if row.step in binary:
                _log.info("plan integrity %s", line)
            else:
                _log.debug("plan integrity %s", line)
            continue
        if row.step in binary:
            _log.error("plan integrity %s", line)
        else:
            # Expected under relax-and-fix when 0.5-thresholding spreads mass.
            _log.warning(
                "plan integrity non-integral far step (look-ahead only): %s",
                line,
            )
