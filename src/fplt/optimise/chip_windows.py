"""Coarse chip-timing windows for the short-horizon MILP.

Mirrors FPL-Optimization-Tools ``allowed_chip_gws`` / ``forced_chip_gws`` /
``no_chip_gws``. Wave C3 exposes the hooks and a rule-based baseline that
forces BB / TC / FH / WC onto double and blank landmarks. Wave C5's
``schedule_chip_windows`` (``fplt.optimise.chip_scheduler``) fills the same
contract with option-value scoring over the rest of the half-season.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fplt.scoring.chips import ChipName, ChipRules

ChipArm = Literal["off", "rule_based", "optimised"]
_CHIP_ORDER: tuple[ChipName, ...] = ("wildcard", "freehit", "bboost", "3xc")


class ChipUseWindows(BaseModel):
    """Allowed, forced, and no-chip gameweeks for each chip.

    ``allowed`` restricts where a chip may fire when the key is present (an
    empty frozenset forbids it entirely). Missing keys leave that chip
    unrestricted beyond ``ChipRules``. ``forced`` requires the chip binary on
    at matching horizon steps. ``no_chip`` forbids every chip that week.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    allowed: dict[ChipName, frozenset[int]] = Field(
        default_factory=lambda: dict[ChipName, frozenset[int]]()
    )
    forced: dict[ChipName, frozenset[int]] = Field(
        default_factory=lambda: dict[ChipName, frozenset[int]]()
    )
    no_chip: frozenset[int] = Field(default_factory=lambda: frozenset[int]())

    @model_validator(mode="after")
    def _consistent(self) -> ChipUseWindows:
        for chip, weeks in self.forced.items():
            allowed = self.allowed.get(chip)
            if allowed is not None and not weeks <= allowed:
                raise ValueError(
                    f"forced weeks for {chip!r} must be a subset of allowed"
                )
            overlap = weeks & self.no_chip
            if overlap:
                raise ValueError(
                    f"forced weeks for {chip!r} collide with no_chip: {sorted(overlap)}"
                )
        for weeks in (*self.allowed.values(), *self.forced.values(), self.no_chip):
            for gameweek in weeks:
                if gameweek < 1 or gameweek > 38:
                    raise ValueError(f"gameweek must be in 1..38, got {gameweek}")
        by_gw: dict[int, list[ChipName]] = {}
        for chip, weeks in self.forced.items():
            for gameweek in weeks:
                by_gw.setdefault(gameweek, []).append(chip)
        for gameweek, chips in by_gw.items():
            if len(chips) > 1:
                raise ValueError(
                    f"at most one forced chip per gameweek; GW{gameweek} has {chips}"
                )
        return self

    def allows(self, chip: ChipName, gameweek: int) -> bool:
        """Return whether ``chip`` may be played in ``gameweek`` under these hooks."""
        if gameweek in self.no_chip:
            return False
        allowed = self.allowed.get(chip)
        if allowed is not None and gameweek not in allowed:
            return False
        return True

    def requires(self, chip: ChipName, gameweek: int) -> bool:
        """Return whether ``chip`` is forced on in ``gameweek``."""
        return gameweek in self.forced.get(chip, frozenset())


def empty_chip_windows() -> ChipUseWindows:
    """Return unrestricted chip-use windows (ChipRules alone still apply)."""
    return ChipUseWindows()


def team_fixture_counts(fixtures: pl.DataFrame, *, season: str) -> pl.DataFrame:
    """Count fixtures per team per gameweek for ``season``.

    Args:
        fixtures: Staged fixtures with ``season``, ``gw``, ``team_h_key``,
            ``team_a_key``.
        season: Season label to keep.

    Returns:
        ``gw``, ``team_key``, ``n_fixtures``.
    """
    part = fixtures.filter(pl.col("season") == season)
    if part.is_empty():
        return pl.DataFrame(
            schema={"gw": pl.Int64(), "team_key": pl.Int64(), "n_fixtures": pl.Int64()}
        )
    home = part.select(pl.col("gw"), pl.col("team_h_key").alias("team_key"))
    away = part.select(pl.col("gw"), pl.col("team_a_key").alias("team_key"))
    return (
        pl.concat([home, away])
        .group_by("gw", "team_key")
        .len()
        .rename({"len": "n_fixtures"})
        .sort("gw", "team_key")
    )


def gameweek_landmarks(counts: pl.DataFrame) -> pl.DataFrame:
    """Summarise doubles and blanks per gameweek from team fixture counts.

    Args:
        counts: Output of ``team_fixture_counts``.

    Returns:
        ``gw``, ``n_double_teams``, ``n_blank_teams``, ``n_fixtures``.
    """
    if counts.is_empty():
        return pl.DataFrame(
            schema={
                "gw": pl.Int64(),
                "n_double_teams": pl.Int64(),
                "n_blank_teams": pl.Int64(),
                "n_fixtures": pl.Int64(),
            }
        )
    teams = counts.select("team_key").unique()
    gws = counts.select("gw").unique()
    grid = gws.join(teams, how="cross")
    filled = (
        grid.join(counts, on=["gw", "team_key"], how="left")
        .with_columns(pl.col("n_fixtures").fill_null(0))
        .group_by("gw")
        .agg(
            (pl.col("n_fixtures") >= 2).sum().alias("n_double_teams"),
            (pl.col("n_fixtures") == 0).sum().alias("n_blank_teams"),
            pl.col("n_fixtures").sum().alias("n_fixtures"),
        )
        .sort("gw")
    )
    return filled


def rule_based_chip_windows(
    fixtures: pl.DataFrame,
    *,
    season: str,
    rules: ChipRules,
    predictions: pl.DataFrame | None = None,
    gameweeks: Sequence[int] | None = None,
) -> ChipUseWindows:
    """Build forced chip weeks from doubles / blanks (eval §5.5 baseline).

    Per ``ChipRules`` window: Bench Boost on the biggest double, Triple Captain
    on the best double-gameweek captain opportunity, Free Hit on the biggest
    blank, Wildcard on the gameweek before the biggest double. Collisions pick
    the next-best landmark so at most one chip is forced per gameweek.

    Args:
        fixtures: Staged fixtures for landmark detection.
        season: Season whose fixtures to use.
        rules: Chip allotment (often 2025-26 rules, possibly counterfactual).
        predictions: Optional ``gw`` / ``player_key`` / ``xpts`` (or
            ``target_gw``) rows for TC ranking among doubles.
        gameweeks: Optional deadline filter; landmarks outside are ignored.

    Returns:
        Windows with ``allowed`` and ``forced`` set to the chosen weeks.
    """
    counts = team_fixture_counts(fixtures, season=season)
    landmarks = gameweek_landmarks(counts)
    if gameweeks is not None:
        allowed_gws = set(gameweeks)
        landmarks = landmarks.filter(pl.col("gw").is_in(list(allowed_gws)))
    captain_by_gw = _captain_xpts_by_gw(predictions)
    forced: dict[ChipName, set[int]] = {name: set() for name in _CHIP_ORDER}
    taken: set[int] = set()
    for window in rules.windows:
        part = landmarks.filter(
            (pl.col("gw") >= window.start_event) & (pl.col("gw") <= window.stop_event)
        )
        grants = {grant.name for grant in rules.grants_for_window(window.id)}
        double_gws = _ranked_doubles(part, captain_by_gw)
        blank_gws = _ranked_blanks(part)
        if "bboost" in grants and double_gws:
            _assign(forced, taken, "bboost", double_gws)
        if "3xc" in grants and double_gws:
            _assign(forced, taken, "3xc", double_gws)
        if "freehit" in grants and blank_gws:
            _assign(forced, taken, "freehit", blank_gws)
        if "wildcard" in grants and double_gws:
            before = [
                gw - 1
                for gw in double_gws
                if window.start_event <= gw - 1 <= window.stop_event
            ]
            _assign(forced, taken, "wildcard", before)
    allowed: dict[ChipName, frozenset[int]] = {
        chip: frozenset(weeks) for chip, weeks in forced.items() if weeks
    }
    forced_frozen: dict[ChipName, frozenset[int]] = {
        chip: frozenset(weeks) for chip, weeks in forced.items() if weeks
    }
    return ChipUseWindows(allowed=allowed, forced=forced_frozen)


def chip_arm_model_id(model_id: str, arm: ChipArm) -> str:
    """Encode the chip arm into a decision-ledger ``model_id``."""
    return f"{model_id}::chip_{arm}"


def parse_chip_arm_model_id(model_id: str) -> tuple[str, ChipArm | None]:
    """Split a chip-arm ledger id into base model id and arm.

    Returns:
        ``(base_model_id, arm)``. ``arm`` is ``None`` when the id has no
        ``::chip_`` suffix.
    """
    marker = "::chip_"
    if marker not in model_id:
        return model_id, None
    base, _, suffix = model_id.partition(marker)
    if suffix == "off":
        return base, "off"
    if suffix == "rule_based":
        return base, "rule_based"
    if suffix == "optimised":
        return base, "optimised"
    return model_id, None


def rules_label(season: str, chip_rules_season: str) -> str:
    """Human label for the chip-rules year used when scoring ``season``.

    When the fixture season differs from the chip-rules season the result is
    tagged as a counterfactual (e.g. ``2024-25-under-2025/26-rules``).
    """
    if season == chip_rules_season:
        return season
    pretty = chip_rules_season.replace("-", "/")
    return f"{season}-under-{pretty}-rules"


def is_counterfactual_rules(season: str, chip_rules_season: str) -> bool:
    """Return whether ``chip_rules_season`` is not the fixture season's own rules."""
    return season != chip_rules_season


def _captain_xpts_by_gw(predictions: pl.DataFrame | None) -> dict[int, float]:
    if predictions is None or predictions.is_empty():
        return {}
    gw_col = "gw" if "gw" in predictions.columns else "target_gw"
    if gw_col not in predictions.columns or "xpts" not in predictions.columns:
        return {}
    peak = predictions.group_by(gw_col).agg(pl.col("xpts").max().alias("xpts"))
    return {int(gw): float(xpts) for gw, xpts in peak.iter_rows()}


def _ranked_doubles(
    landmarks: pl.DataFrame, captain_by_gw: Mapping[int, float]
) -> list[int]:
    if landmarks.is_empty():
        return []
    doubles = landmarks.filter(pl.col("n_double_teams") > 0)
    if doubles.is_empty():
        return []
    rows = [
        (
            -int(row["n_double_teams"]),
            -float(captain_by_gw.get(int(row["gw"]), 0.0)),
            int(row["gw"]),
        )
        for row in doubles.iter_rows(named=True)
    ]
    rows.sort()
    return [gw for _, _, gw in rows]


def _ranked_blanks(landmarks: pl.DataFrame) -> list[int]:
    if landmarks.is_empty():
        return []
    blanks = landmarks.filter(pl.col("n_blank_teams") > 0)
    if blanks.is_empty():
        return []
    rows = [
        (-int(row["n_blank_teams"]), int(row["gw"]))
        for row in blanks.iter_rows(named=True)
    ]
    rows.sort()
    return [gw for _, gw in rows]


def _assign(
    forced: dict[ChipName, set[int]],
    taken: set[int],
    chip: ChipName,
    candidates: Sequence[int],
) -> None:
    for gameweek in candidates:
        if gameweek in taken or gameweek < 1:
            continue
        forced[chip].add(gameweek)
        taken.add(gameweek)
        return
