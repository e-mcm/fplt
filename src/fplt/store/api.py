"""Store read API over catalog, official pointers, runs, and ledger.

Data-model migration step 8 / Stage C: ``Store.runs``, ``official``, ``load``,
``experiment``, ``pit``, plus the existing ``ledger`` view. Incomplete runs
(no ``_SUCCESS``) stay invisible. Missing official pointers return ``None``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import polars as pl

from fplt.backtest.ledger import (
    align_ledger_columns,
    empty_ledger,
    ledger_path,
    predictions_from_benchmarks,
)
from fplt.data.pit import PITView
from fplt.data.store import DataStore
from fplt.schemas.ledger import LEDGER_COLUMNS, LedgerSchema
from fplt.schemas.manifests import (
    AppliedRecord,
    ExperimentManifest,
    OfficialPointer,
    RunManifest,
)
from fplt.settings import get_settings
from fplt.store.builds import CuratedBuildSource, latest_build_id
from fplt.store.catalog import read_runs_catalog
from fplt.store.experiments import (
    FoldedStatus,
    experiment_dir,
    fold_experiment_status,
    load_experiment_events,
    load_experiment_manifest,
)
from fplt.store.push import load_run_manifest
from fplt.store.runs import list_succeeded_predict_run_ids, run_dir
from fplt.types import GWKey
from fplt.utils.frames import require_dataframe

logger = logging.getLogger(__name__)

_SUCCESS = "_SUCCESS"
_DEFAULT_CHECKPOINT = "T-90m"

_ACTUAL_COLS = (
    "actual_pts",
    "actual_minutes",
    "actual_goals",
    "actual_assists",
    "actual_cs",
)

_LOAD_TABLES: dict[str, str] = {
    "player_fixture": "predictions/player_fixture.parquet",
    "minutes": "predictions/components/minutes.parquet",
    "scoreline": "predictions/components/scoreline.parquet",
    "involvement": "predictions/components/involvement.parquet",
    "other": "predictions/components/other.parquet",
    "points_pmf": "predictions/points_pmf.parquet",
    "points_pmf_gw": "predictions/points_pmf_gw.parquet",
    "plans": "plans/plan.parquet",
    "plan": "plans/plan.parquet",
    "decisions": "plans/decisions.parquet",
    # post_gw analysis run tables (#396)
    "scorecard": "scorecard.parquet",
    "attribution": "attribution.parquet",
    "season_to_date": "season_to_date.parquet",
    "biggest_misses": "biggest_misses.parquet",
    "forecast_scores": "forecast_scores.parquet",
    "forecast_score_agg": "forecast_score_agg.parquet",
}


@dataclass(frozen=True)
class ExperimentHandle:
    """Read view of one experiment: manifest, folded status, arm run ids."""

    experiment_id: str
    manifest: ExperimentManifest
    status: FoldedStatus
    arms: dict[str, str | None]


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

    def runs(
        self,
        *,
        spec_name: str | None = None,
        kind: str | None = None,
        run_type: str | None = None,
        status: str | None = "succeeded",
    ) -> pl.DataFrame:
        """Query the derived runs catalog (``CatalogRunSchema``).

        Only succeeded ``_SUCCESS`` runs appear in the catalog. Filters are
        optional equality matches on catalog columns.

        Args:
            spec_name: Match ``spec_label`` (human registered / auto label).
            kind: Match ``kind`` (``official`` / ``adhoc`` / …).
            run_type: Match ``run_type`` (``predict`` / ``plan`` / ``eval`` /
                ``component``).
            status: Match ``status``; default ``succeeded``. Pass ``None`` to
                skip the status filter.

        Returns:
            Catalog rows sorted by ``run_id`` (possibly empty).
        """
        frame = read_runs_catalog(self.data_dir)
        if frame.is_empty():
            return frame
        expr = pl.lit(True)
        if spec_name is not None:
            expr = expr & (pl.col("spec_label") == spec_name)
        if kind is not None:
            expr = expr & (pl.col("kind") == kind)
        if run_type is not None:
            expr = expr & (pl.col("run_type") == run_type)
        if status is not None:
            expr = expr & (pl.col("status") == status)
        return frame.filter(expr).sort("run_id")

    def manifest(self, run_id: str) -> RunManifest | None:
        """Load a succeeded run's manifest, or ``None`` if incomplete/missing.

        Args:
            run_id: Run folder id under ``runs/run=<id>/``.

        Returns:
            Parsed ``RunManifest`` when ``_SUCCESS`` exists; else ``None``.
        """
        path = run_dir(self.data_dir, run_id)
        if not (path / _SUCCESS).is_file():
            return None
        try:
            return load_run_manifest(self.data_dir, run_id)
        except FileNotFoundError:
            return None

    def official(
        self,
        *,
        season: str | None = None,
        gw: int,
        line: str | None = None,
        checkpoint: str = _DEFAULT_CHECKPOINT,
    ) -> RunManifest | None:
        """Resolve the official plan run for a gameweek checkpoint.

        Without ``line``, reads ``applied.json`` and returns that plan run.
        With ``line``, reads the per-line pointer. Missing pointers (B12 still
        landing) return ``None`` rather than raising.

        Args:
            season: Season key; when omitted, inferred from applied/line paths
                or the sole season under ``official/``.
            gw: Gameweek (1–38).
            line: Optional line name (e.g. ``v4_h9``); default is the applied
                line.
            checkpoint: Official checkpoint id (default ``T-90m``).

        Returns:
            Succeeded plan ``RunManifest``, or ``None`` when the pointer or
            run is missing / incomplete.
        """
        run_id = self._official_plan_run_id(
            season=season, gw=gw, line=line, checkpoint=checkpoint
        )
        if run_id is None:
            return None
        return self.manifest(run_id)

    def load(
        self,
        table: str,
        *,
        run: str | RunManifest,
    ) -> pl.LazyFrame:
        """Lazy-scan one L2 table from a succeeded run folder.

        Args:
            table: Logical name (``player_fixture``, ``minutes``, ``plans``, …).
            run: Run id string or a ``RunManifest`` (e.g. from ``official``).

        Returns:
            ``pl.LazyFrame`` over the parquet file.

        Raises:
            KeyError: Unknown ``table`` name.
            FileNotFoundError: Run incomplete or table file missing.
            TypeError: ``run`` is neither ``str`` nor ``RunManifest``.
        """
        if table not in _LOAD_TABLES:
            raise KeyError(
                f"unknown store table {table!r}; expected one of {sorted(_LOAD_TABLES)}"
            )
        run_id = _run_id(run)
        path = run_dir(self.data_dir, run_id)
        if not (path / _SUCCESS).is_file():
            raise FileNotFoundError(
                f"run {run_id} has no _SUCCESS; incomplete runs are invisible"
            )
        target = path / _LOAD_TABLES[table]
        if not target.is_file():
            raise FileNotFoundError(target)
        return pl.scan_parquet(target)

    def experiment(self, experiment_id: str) -> ExperimentHandle:
        """Load an experiment handle with folded status and arm run ids.

        Args:
            experiment_id: Id under ``experiments/exp=<id>/``.

        Returns:
            ``ExperimentHandle`` with ``.arms`` mapping arm → run_id (null
            until bound).

        Raises:
            FileNotFoundError: Experiment manifest is missing.
        """
        manifest = load_experiment_manifest(self.data_dir, experiment_id)
        events = load_experiment_events(self.data_dir, experiment_id)
        status = fold_experiment_status(events, n_arms=len(manifest.arms))
        arms = {arm.arm: arm.run_id for arm in manifest.arms}
        arms_dir = experiment_dir(self.data_dir, experiment_id) / "arms"
        if arms_dir.is_dir():
            for path in arms_dir.glob("*.json"):
                payload = json.loads(path.read_text(encoding="utf-8"))
                name = str(payload.get("arm", path.stem))
                run_id = payload.get("run_id")
                arms[name] = str(run_id) if run_id is not None else None
        return ExperimentHandle(
            experiment_id=experiment_id,
            manifest=manifest,
            status=status,
            arms=arms,
        )

    def pit(
        self,
        *,
        build_id: str,
        deadline: datetime,
        as_of: GWKey,
    ) -> PITView:
        """Build a ``PITView`` from a curated build (reproduce a run's inputs).

        Args:
            build_id: Curated build id.
            deadline: PIT cutoff (usually the FPL deadline).
            as_of: Gameweek key for horizon / season context.

        Returns:
            ``PITView`` over the content-addressed curated tables.
        """
        return PITView.from_build(
            self.data_dir, build_id, deadline=deadline, as_of=as_of
        )

    def ledger(
        self,
        runs: Sequence[str] | None = None,
        *,
        outcomes_build: str | None = None,
        include_benchmarks: bool | None = None,
    ) -> pl.DataFrame:
        """Join predictions to ``player_fixture_outcomes`` (ledger-as-view).

        Join key: ``(target_season, target_gw, player_key, fixture_id)`` ↔
        ``(season, gw, player_key, fixture_id)``. Unchecked gameweeks have no
        outcomes rows, so actuals stay null.

        When ``runs`` is omitted, every succeeded predict run under ``runs/``
        is included. If none exist, a legacy ``ledger/predictions`` parquet
        (B10 archive) is read when present. Archived ``benchmark_projections``
        (ep_next / OpenFPL) are appended as mean-only pseudo-models by default
        when discovering the full view (#390); OpenFPL seasons before
        ``OPENFPL_COMPARE_FROM`` are excluded.

        Args:
            runs: Optional run ids whose ``predictions/player_fixture.parquet``
                frames are concatenated. ``None`` discovers local predict runs.
            outcomes_build: Curated build id to pin; ``None`` uses the latest
                build that contains outcomes, else staged
                ``player_fixture_outcomes``.
            include_benchmarks: When ``None`` (default), include archives only
                for the full discovered view (``runs is None``). Pass ``True``
                / ``False`` to force.

        Returns:
            ``LedgerSchema`` rows (possibly empty).
        """
        predictions = self._predictions(runs)
        with_benchmarks = (
            include_benchmarks if include_benchmarks is not None else runs is None
        )
        if with_benchmarks:
            predictions = _concat_predictions(
                predictions, predictions_from_benchmarks(self.data_dir)
            )
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
        checked = require_dataframe(checked)
        return checked

    def _official_plan_run_id(
        self,
        *,
        season: str | None,
        gw: int,
        line: str | None,
        checkpoint: str,
    ) -> str | None:
        root = self.data_dir / "official"
        if not root.is_dir():
            return None
        resolved_season = season if season is not None else _infer_official_season(root)
        if resolved_season is None:
            return None
        base = (
            root / f"season={resolved_season}" / f"gw={gw}" / f"checkpoint={checkpoint}"
        )
        if line is None:
            applied_path = base / "applied.json"
            if not applied_path.is_file():
                return None
            try:
                applied = AppliedRecord.model_validate_json(
                    applied_path.read_text(encoding="utf-8")
                )
            except ValueError as exc:
                logger.debug("skip applied %s: %s", applied_path, exc)
                return None
            return applied.plan_run_id
        pointer_path = base / f"line={line}.json"
        if not pointer_path.is_file():
            return None
        try:
            pointer = OfficialPointer.model_validate_json(
                pointer_path.read_text(encoding="utf-8")
            )
        except ValueError as exc:
            logger.debug("skip official pointer %s: %s", pointer_path, exc)
            return None
        return pointer.plan_run_id

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


def _run_id(run: str | RunManifest) -> str:
    if isinstance(run, RunManifest):
        return run.run_id
    return run


def _infer_official_season(root: Path) -> str | None:
    seasons = sorted(
        p.name.removeprefix("season=") for p in root.glob("season=*") if p.is_dir()
    )
    if len(seasons) == 1:
        return seasons[0]
    return None


def _legacy_predictions(data_dir: Path) -> pl.DataFrame:
    """Read the pre-B9 ledger parquet when no runs exist (archive only)."""
    path = ledger_path(data_dir)
    if not path.is_file():
        return empty_ledger()
    return _strip_actuals(align_ledger_columns(pl.read_parquet(path)))


def _concat_predictions(left: pl.DataFrame, right: pl.DataFrame) -> pl.DataFrame:
    """Stack two ledger-shaped prediction frames, tolerating empties."""
    if left.is_empty():
        return right
    if right.is_empty():
        return left
    return align_ledger_columns(pl.concat([left, right], how="vertical_relaxed"))


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
