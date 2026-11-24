"""Point-in-time views over staged or curated tables.

Models receive a ``PITView`` and never a store. Rows with ``available_at``
after the gameweek deadline are invisible. Stage B6: construct a view from a
content-addressed curated build via ``PITView.from_build`` /
``PITData.from_build``.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
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
    """Rows visible at one gameweek deadline.

    Optional ``build_id`` records which curated build backed the view
    (``None`` when reading the migration ``staged/`` tree via ``DataStore``).
    """

    as_of: GWKey
    cutoff: datetime
    _source: TableSource
    build_id: str | None = None

    @classmethod
    def from_build(
        cls,
        data_dir: Path,
        build_id: str,
        *,
        deadline: datetime,
        as_of: GWKey,
    ) -> PITView:
        """Construct a view over a curated build at an explicit deadline.

        Args:
            data_dir: Local data root holding ``curated/``.
            build_id: Curated build id (``builds/build=<id>.json``).
            deadline: PIT cutoff (usually the FPL deadline for ``as_of``).
            as_of: Gameweek key models use for horizon / season context.

        Returns:
            A ``PITView`` whose tables come from the build's content-addressed
            parquet files.
        """
        from fplt.store.builds import CuratedBuildSource

        source = CuratedBuildSource.load(data_dir, build_id)
        return cls(
            as_of=as_of,
            cutoff=deadline,
            _source=source,
            build_id=build_id,
        )

    def within_seasons(self, seasons: set[str]) -> PITView:
        """A view that also hides rows from other seasons."""
        window = _SeasonWindow(self._source, frozenset(seasons))
        return PITView(
            as_of=self.as_of,
            cutoff=self.cutoff,
            _source=window,
            build_id=self.build_id,
        )

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
    """Build a view of a store (or curated build) as of a gameweek deadline."""

    def __init__(self, source: TableSource, *, build_id: str | None = None) -> None:
        self._source = source
        self.build_id = build_id

    @classmethod
    def from_build(cls, data_dir: Path, build_id: str) -> PITData:
        """PIT accessor over one curated build.

        Args:
            data_dir: Local data root holding ``curated/``.
            build_id: Curated build id.

        Returns:
            ``PITData`` whose ``view`` / ``deadline`` read the build's tables.
        """
        from fplt.store.builds import CuratedBuildSource

        return cls(CuratedBuildSource.load(data_dir, build_id), build_id=build_id)

    def deadline(self, gw: GWKey) -> datetime:
        """Deadline of ``gw`` according to the store."""
        return self._source.deadline(gw)

    def view(self, as_of: GWKey) -> PITView:
        """A view that hides anything published after ``as_of``'s deadline."""
        return PITView(
            as_of=as_of,
            cutoff=self.deadline(as_of),
            _source=self._source,
            build_id=self.build_id,
        )
