"""Season scoring tables loaded from ``configs/scoring``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

PositionName = Literal["GK", "DEF", "MID", "FWD"]
DefensiveAction = Literal[
    "clearances_blocks_interceptions",
    "tackles",
    "recoveries",
]
_POSITIONS = frozenset({"GK", "DEF", "MID", "FWD"})
_CONFIGS = Path(__file__).resolve().parents[3] / "configs" / "scoring"


def _no_contributions() -> list[DefensiveContribution]:
    return []


class AppearanceRule(BaseModel):
    """Points for a short appearance and for playing the long-minutes threshold."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    long_minutes: int = Field(ge=1)
    short: int
    long: int


class CountRule(BaseModel):
    """Points awarded once per ``every`` events, for the listed positions."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    positions: list[PositionName] = Field(min_length=1)
    every: int = Field(ge=1)
    points: int


class DefensiveContribution(BaseModel):
    """A flat award once the summed actions reach ``threshold``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    positions: list[PositionName] = Field(min_length=1)
    actions: list[DefensiveAction] = Field(min_length=1)
    threshold: int = Field(ge=1)
    points: int


class MatchStats(BaseModel):
    """Integer match stats for one player, as published on ``event/{gw}/live``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    position: PositionName
    minutes: int = 0
    goals_scored: int = 0
    assists: int = 0
    clean_sheets: int = 0
    goals_conceded: int = 0
    saves: int = 0
    penalties_saved: int = 0
    penalties_missed: int = 0
    yellow_cards: int = 0
    red_cards: int = 0
    own_goals: int = 0
    bonus: int = 0
    clearances_blocks_interceptions: int = 0
    tackles: int = 0
    recoveries: int = 0


class Rules(BaseModel):
    """The FPL points table for one season."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    appearance: AppearanceRule
    goals: dict[PositionName, int]
    assists: int
    clean_sheets: dict[PositionName, int]
    goals_conceded: CountRule
    saves: CountRule
    penalties_saved: int
    penalties_missed: int
    yellow_cards: int
    red_cards: int
    own_goals: int
    bonus: int
    defensive_contributions: list[DefensiveContribution] = Field(
        default_factory=_no_contributions
    )

    @model_validator(mode="after")
    def _complete(self) -> Rules:
        for name, table in (("goals", self.goals), ("clean_sheets", self.clean_sheets)):
            missing = _POSITIONS - set(table)
            if missing:
                raise ValueError(f"{name} is missing {sorted(missing)}")
        seen: list[str] = []
        for rule in self.defensive_contributions:
            overlap = set(seen) & set(rule.positions)
            if overlap:
                raise ValueError(
                    f"defensive contributions overlap on {sorted(overlap)}"
                )
            seen.extend(rule.positions)
        return self

    @classmethod
    def for_season(cls, season: str) -> Rules:
        """Load and validate ``configs/scoring/<season>.yaml``.

        Args:
            season: Season label, such as ``"2024-25"``.

        Returns:
            The validated table.

        Raises:
            ValueError: ``season`` is not a plain file name.
            FileNotFoundError: That season has no rules file.
        """
        if not season or Path(season).name != season:
            raise ValueError(f"season must be a file name, got {season!r}")
        path = _CONFIGS / f"{season}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"no scoring rules for season {season!r}")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(loaded)

    def score(self, stats: MatchStats) -> int:
        """Return the points ``stats`` score under this table.

        Clean sheets and goals conceded are used as published. The live feed
        already applies the 60-minute clean-sheet gate and keeps conceding
        after a red card, so this does not recompute them from the scoreline.

        Args:
            stats: One player's integer match stats. ``MatchStats``.

        Returns:
            The points total.
        """
        points = self._appearance(stats.minutes)
        points += stats.goals_scored * self.goals[stats.position]
        points += stats.assists * self.assists
        points += stats.clean_sheets * self.clean_sheets[stats.position]
        points += self._banded(
            self.goals_conceded, stats.position, stats.goals_conceded
        )
        points += self._banded(self.saves, stats.position, stats.saves)
        points += stats.penalties_saved * self.penalties_saved
        points += stats.penalties_missed * self.penalties_missed
        points += stats.yellow_cards * self.yellow_cards
        points += stats.red_cards * self.red_cards
        points += stats.own_goals * self.own_goals
        points += stats.bonus * self.bonus
        points += self._defensive(stats)
        return points

    def _appearance(self, minutes: int) -> int:
        if minutes >= self.appearance.long_minutes:
            return self.appearance.long
        if minutes > 0:
            return self.appearance.short
        return 0

    def _banded(self, rule: CountRule, position: str, count: int) -> int:
        if position not in rule.positions:
            return 0
        return rule.points * (count // rule.every)

    def _defensive(self, stats: MatchStats) -> int:
        awarded = 0
        for rule in self.defensive_contributions:
            if stats.position not in rule.positions:
                continue
            actions = sum(getattr(stats, name) for name in rule.actions)
            if actions >= rule.threshold:
                awarded += rule.points
        return awarded
