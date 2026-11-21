"""Point-in-time views over staged tables.

Models receive a ``PITView`` and never a store. Rows with ``available_at``
after the gameweek deadline are invisible.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Protocol

import polars as pl

from fplt.types import GWKey


class TableSource(Protocol):
    """Anything that can hand back a staged table by name."""

    def table(self, name: str) -> pl.LazyFrame: ...

    def deadline(self, gw: GWKey) -> datetime: ...


@dataclass(frozen=True)
class _SeasonWindow:
    """Hide rows from seasons outside a training window."""

    inner: TableSource
    seasons: frozenset[str]

    def table(self, name: str) -> pl.LazyFrame:
        frame = self.inner.table(name)
        if "season" not in frame.collect_schema().names():
            return frame
        return frame.filter(pl.col("season").is_in(sorted(self.seasons)))

    def deadline(self, gw: GWKey) -> datetime:
        return self.inner.deadline(gw)


@dataclass(frozen=True)
class PITView:
    """Rows visible at one gameweek deadline."""

    as_of: GWKey
    cutoff: datetime
    _source: TableSource

    def within_seasons(self, seasons: set[str]) -> PITView:
        """A view that also hides rows from other seasons."""
        window = _SeasonWindow(self._source, frozenset(seasons))
        return PITView(as_of=self.as_of, cutoff=self.cutoff, _source=window)

    def table(self, name: str) -> pl.LazyFrame:
        """Rows of ``name`` with ``available_at`` at or before the cutoff."""
        frame = self._source.table(name)
        if "available_at" not in frame.collect_schema().names():
            return frame
        return frame.filter(pl.col("available_at") <= self.cutoff)

    def latest(self, name: str, by: list[str]) -> pl.LazyFrame:
        """Last visible row for each key in ``by``."""
        return self.table(name).sort("available_at").unique(subset=by, keep="last")

    def closing_proxy_match_odds(self) -> pl.LazyFrame:
        """``match_odds`` rows with ``odds_kind=closing_proxy``, ignoring cutoff.

        Closing lines are staged with ``available_at = kickoff`` (after the FPL
        deadline). This escape hatch is for the labelled closing-proxy scoreline
        baseline and DC market prior only — never claim deadline-safe PIT.
        Season windows from ``within_seasons`` still apply.
        """
        frame = self._source.table("match_odds")
        return frame.filter(pl.col("odds_kind") == "closing_proxy")


class PITData:
    """Build a view of a store as of a gameweek deadline."""

    def __init__(self, source: TableSource) -> None:
        self._source = source

    def deadline(self, gw: GWKey) -> datetime:
        """Deadline of ``gw`` according to the store."""
        return self._source.deadline(gw)

    def view(self, as_of: GWKey) -> PITView:
        """A view that hides anything published after ``as_of``'s deadline."""
        return PITView(as_of=as_of, cutoff=self.deadline(as_of), _source=self._source)
