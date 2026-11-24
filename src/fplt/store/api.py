"""Store read API: ledger view over predictions ⋈ outcomes.

Full catalog / official / experiment helpers land in a later Stage B PR
(data-model migration step 8). B9: ``Store.ledger`` is the only path to
scored prediction rows (``append_predictions`` / ``fill_actuals`` removed).
"""

from __future__ import annotations

import logging
from collections.abc import Sequence
from pathlib import Path

import polars as pl

from fplt.backtest.ledger import align_ledger_columns, empty_ledger, ledger_path
from fplt.data.store import DataStore
from fplt.schemas.ledger import LEDGER_COLUMNS, LedgerSchema
from fplt.settings import get_settings
from fplt.store.builds import CuratedBuildSource, latest_build_id
from fplt.store.runs import list_succeeded_predict_run_ids, run_dir

logger = logging.getLogger(__name__)

_ACTUAL_COLS = (
    "actual_pts",
    "actual_minutes",
    "actual_goals",
    "actual_assists",
    "actual_cs",
)


class Store:
    """Local-first analytic read API over ``$FPLT_DATA_DIR``.

    Model and feature code must not use this class; they only see a ``PITView``.
    """

    def __init__(self, data_dir: Path) -> None:
        """Bind to a local data root.

        Args:
            data_dir: Directory with ``staged/``, ``runs/``, ``curated/``, and
                the optional legacy ``ledger/`` archive.
        """
        self.data_dir = data_dir

    @classmethod
    def from_settings(cls) -> Store:
        """Construct from ``FPLT_DATA_DIR`` / settings."""
        return cls(get_settings().data_dir)

    def ledger(
        self,
        runs: Sequence[str] | None = None,
        *,
        outcomes_build: str | None = None,
    ) -> pl.DataFrame:
        """Join predictions to ``player_fixture_outcomes`` (ledger-as-view).

        Join key: ``(target_season, target_gw, player_key, fixture_id)`` ↔
        ``(season, gw, player_key, fixture_id)``. Unchecked gameweeks have no
        outcomes rows, so actuals stay null.

        When ``runs`` is omitted, every succeeded predict run under ``runs/``
        is included. If none exist, a legacy ``ledger/predictions`` parquet
        (B10 archive) is read when present.

        Args:
            runs: Optional run ids whose ``predictions/player_fixture.parquet``
                frames are concatenated. ``None`` discovers local predict runs.
            outcomes_build: Curated build id to pin; ``None`` uses the latest
                build that contains outcomes, else staged
                ``player_fixture_outcomes``.

        Returns:
            ``LedgerSchema`` rows (possibly empty).
        """
        predictions = self._predictions(runs)
        if predictions.is_empty():
            return empty_ledger()
        outcomes = self._player_outcomes(outcomes_build)
        if outcomes.is_empty():
            logger.debug("ledger view: no outcomes; returning null actuals")
            return align_ledger_columns(predictions)
        joined = (
            predictions.join(
                outcomes,
                left_on=["target_season", "target_gw", "player_key", "fixture_id"],
                right_on=["season", "gw", "player_key", "fixture_id"],
                how="left",
            )
            .with_columns(
                pl.col("pts").alias("actual_pts"),
                pl.col("minutes").alias("actual_minutes"),
                pl.col("goals").alias("actual_goals"),
                pl.col("assists").alias("actual_assists"),
                pl.col("clean_sheets").alias("actual_cs"),
            )
            .select(list(LEDGER_COLUMNS))
        )
        checked = LedgerSchema.validate(align_ledger_columns(joined))
        assert isinstance(checked, pl.DataFrame)
        return checked

    def _predictions(self, runs: Sequence[str] | None) -> pl.DataFrame:
        run_ids: Sequence[str]
        if runs is None:
            run_ids = list_succeeded_predict_run_ids(self.data_dir)
            if not run_ids:
                return _legacy_predictions(self.data_dir)
        else:
            run_ids = runs
        frames: list[pl.DataFrame] = []
        for run_id in run_ids:
            path = (
                run_dir(self.data_dir, run_id)
                / "predictions"
                / ("player_fixture.parquet")
            )
            if not path.is_file():
                raise FileNotFoundError(path)
            raw = pl.read_parquet(path)
            frames.append(_strip_actuals(align_ledger_columns(raw)))
        if not frames:
            return empty_ledger()
        return align_ledger_columns(pl.concat(frames, how="vertical_relaxed"))

    def _player_outcomes(self, outcomes_build: str | None) -> pl.DataFrame:
        build_id = (
            outcomes_build
            if outcomes_build is not None
            else latest_build_id(self.data_dir)
        )
        if build_id is not None:
            frame = (
                CuratedBuildSource.load(self.data_dir, build_id)
                .table("player_fixture_outcomes")
                .collect()
            )
            if not frame.is_empty():
                return _latest_outcomes(frame)
        staged = DataStore(self.data_dir).read("player_fixture_outcomes")
        if staged.is_empty():
            return staged
        return _latest_outcomes(staged)


def _legacy_predictions(data_dir: Path) -> pl.DataFrame:
    """Read the pre-B9 ledger parquet when no runs exist (archive only)."""
    path = ledger_path(data_dir)
    if not path.is_file():
        return empty_ledger()
    return _strip_actuals(align_ledger_columns(pl.read_parquet(path)))


def _latest_outcomes(frame: pl.DataFrame) -> pl.DataFrame:
    """Keep the newest outcomes row per player–fixture key."""
    return frame.sort("available_at").unique(
        subset=["season", "gw", "player_key", "fixture_id"],
        keep="last",
    )


def _strip_actuals(frame: pl.DataFrame) -> pl.DataFrame:
    """Drop filled actual columns so the outcomes join is authoritative."""
    aligned = align_ledger_columns(frame)
    return aligned.with_columns(
        *[pl.lit(None).cast(pl.Int64).alias(col) for col in _ACTUAL_COLS]
    ).select(list(LEDGER_COLUMNS))
