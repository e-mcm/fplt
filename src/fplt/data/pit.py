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
class PITView:
    """Rows visible at one gameweek deadline."""

    as_of: GWKey
    cutoff: datetime
    _source: TableSource

    def table(self, name: str) -> pl.LazyFrame:
        """Rows of ``name`` with ``available_at`` at or before the cutoff."""
        frame = self._source.table(name)
        if "available_at" not in frame.collect_schema().names():
            return frame
        return frame.filter(pl.col("available_at") <= self.cutoff)

    def latest(self, name: str, by: list[str]) -> pl.LazyFrame:
        """Last visible row for each key in ``by``."""
        return self.table(name).sort("available_at").unique(subset=by, keep="last")


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
