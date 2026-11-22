"""Season chip allotments loaded from ``configs/chips``."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

ChipName = Literal["wildcard", "freehit", "bboost", "3xc"]
_ALL_CHIPS: tuple[ChipName, ...] = ("wildcard", "freehit", "bboost", "3xc")
_CONFIGS = Path(__file__).resolve().parents[3] / "configs" / "chips"


class ChipWindow(BaseModel):
    """A closed gameweek interval when a chip grant may be played."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    id: str = Field(min_length=1)
    start_event: int = Field(ge=1, le=38)
    stop_event: int = Field(ge=1, le=38)
    carries_over: bool = False

    @model_validator(mode="after")
    def _ordered(self) -> ChipWindow:
        if self.start_event > self.stop_event:
            raise ValueError(f"window {self.id!r}: start_event must be <= stop_event")
        return self


class ChipGrant(BaseModel):
    """One use of a named chip, playable inside ``window``."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: ChipName
    window: str = Field(min_length=1)


class ChipRules(BaseModel):
    """The FPL chip table for one season."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    one_chip_per_gameweek: bool
    free_hit_keeps_free_transfers: bool
    windows: list[ChipWindow] = Field(min_length=1)
    grants: list[ChipGrant] = Field(min_length=1)

    @model_validator(mode="after")
    def _consistent(self) -> ChipRules:
        by_id = {window.id: window for window in self.windows}
        if len(by_id) != len(self.windows):
            raise ValueError("chip window ids must be unique")
        for grant in self.grants:
            if grant.window not in by_id:
                raise ValueError(
                    f"grant {grant.name!r} references unknown window {grant.window!r}"
                )
        return self

    @classmethod
    def for_season(cls, season: str) -> ChipRules:
        """Load and validate ``configs/chips/<season>.yaml``.

        Args:
            season: Season label, such as ``"2025-26"``.

        Returns:
            The validated chip rules.

        Raises:
            ValueError: ``season`` is not a plain file name.
            FileNotFoundError: That season has no chip rules file.
        """
        if not season or Path(season).name != season:
            raise ValueError(f"season must be a file name, got {season!r}")
        path = _CONFIGS / f"{season}.yaml"
        if not path.is_file():
            raise FileNotFoundError(f"no chip rules for season {season!r}")
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
        return cls.model_validate(loaded)

    def remaining_counts(self) -> dict[ChipName, int]:
        """Return how many uses of each chip the season grants in total.

        Returns:
            Counts keyed by FPL chip name (``wildcard``, ``freehit``,
            ``bboost``, ``3xc``). Missing names are omitted.
        """
        counts: dict[ChipName, int] = {}
        for grant in self.grants:
            counts[grant.name] = counts.get(grant.name, 0) + 1
        return counts

    def grants_for_window(self, window_id: str) -> list[ChipGrant]:
        """Return grants that belong to ``window_id``.

        Args:
            window_id: A ``ChipWindow.id`` from this season.

        Returns:
            Matching grants in config order.

        Raises:
            KeyError: ``window_id`` is not a defined window.
        """
        if window_id not in {window.id for window in self.windows}:
            raise KeyError(f"unknown chip window {window_id!r}")
        return [grant for grant in self.grants if grant.window == window_id]


def empty_remaining() -> dict[ChipName, int]:
    """Return zero remaining uses for every chip name."""
    return {name: 0 for name in _ALL_CHIPS}
