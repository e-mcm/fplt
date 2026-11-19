"""Shared domain types for fplt."""

from dataclasses import dataclass
from enum import IntEnum


class Position(IntEnum):
    """FPL player position.

    Integer values match the ``position_enc`` column produced by
    ``features.engineer`` so they can be used directly as LightGBM features
    and in DataFrame comparisons without conversion.
    """

    GK = 0
    DEF = 1
    MID = 2
    FWD = 3

    @classmethod
    def from_str(cls, s: str) -> "Position":
        """Parse a position string, treating AM as MID.

        Args:
            s: Position string e.g. "GK", "DEF", "MID", "FWD", "AM".

        Returns:
            Corresponding Position enum member.

        Raises:
            ValueError: If the string is not a recognised position.
        """
        mapping = {
            "GK": cls.GK,
            "DEF": cls.DEF,
            "MID": cls.MID,
            "AM": cls.MID,
            "FWD": cls.FWD,
        }
        try:
            return mapping[s.upper()]
        except KeyError:
            raise ValueError(f"Unknown position: {s!r}") from None


@dataclass(frozen=True, order=True)
class GWKey:
    """A season and gameweek, the unit of a point-in-time cutoff."""

    season: str
    gw: int

    def __post_init__(self) -> None:
        if not 1 <= self.gw <= 38:
            raise ValueError(f"gameweek must be in 1..38, got {self.gw}")
