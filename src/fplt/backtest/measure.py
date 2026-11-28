"""Common Measure interface + adapters over existing scorers.

Toolkit §2.3 / ``fplt-experiments.md`` §9 step 3: one protocol
``(stage, runs, outcomes, slices) → metrics table`` with a shared pandera
schema. Adapters wrap ``component_eval``, ``minutes_measurement``,
``other_points_measurement``, ``forecast_scores``, the decisions board, and
``simulated_season`` without rewriting those scorers.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

import polars as pl

from fplt.backtest.forecast_scores import aggregate_forecast_scores
from fplt.backtest.minutes_measurement import summarise_minutes_detail
from fplt.backtest.other_points_measurement import (
    captain_regret_by_gw,
    summarise_other_points_detail,
)
from fplt.backtest.simulated_season import mean_policy_totals
from fplt.optimise.realised import summarise_decisions
from fplt.schemas.measure import (
    MEASURE_METRICS_COLUMNS,
    MEASURE_STAGES,
    SCHEMA_VERSION,
    MeasureMetricsSchema,
)
from fplt.utils.frames import require_dataframe

DEFAULT_WINDOW = "primary"

_Runs = pl.DataFrame | Mapping[str, pl.DataFrame]


@runtime_checkable
class Measure(Protocol):
    """Score arm run outputs against outcomes into a standard metrics table."""

    @property
    def name(self) -> str:
        """Stable adapter id (e.g. ``forecast_scores``)."""
        ...

    def score(
        self,
        stage: str,
        runs: _Runs,
        outcomes: pl.DataFrame,
        slices: Sequence[str] | None = None,
        *,
        window: str = DEFAULT_WINDOW,
    ) -> pl.DataFrame:
        """Return ``MeasureMetricsSchema`` rows for one measurement stage.

        Args:
            stage: Pipeline stage tag (``components``, ``prediction``,
                ``decision``, ``season``, …).
            runs: Scorer-specific run outputs (a frame or named frames).
            outcomes: Realised outcomes when the scorer needs them; may be
                empty when actuals already sit on ``runs``.
            slices: Optional slice labels to keep (``None`` = all).
            window: Eval window label (default ``primary``).

        Returns:
            Validated metrics table.
        """
        ...


def empty_measure_metrics() -> pl.DataFrame:
    """Return an empty validated ``MeasureMetricsSchema`` frame."""
    return require_dataframe(
        MeasureMetricsSchema.validate(pl.DataFrame(schema=MEASURE_METRICS_COLUMNS))
    )


def validate_measure_metrics(frame: pl.DataFrame) -> pl.DataFrame:
    """Validate a metrics table against ``MeasureMetricsSchema``."""
    return require_dataframe(MeasureMetricsSchema.validate(frame))


def filter_slices(
    frame: pl.DataFrame,
    slices: Sequence[str] | None,
) -> pl.DataFrame:
    """Keep rows whose ``slice`` is in ``slices`` (``None`` keeps all)."""
    if slices is None:
        return frame
    wanted = list(slices)
    if not wanted or frame.is_empty():
        return frame.clear() if not wanted else frame
    return frame.filter(pl.col("slice").is_in(wanted))


def _as_frame(runs: _Runs, key: str | None = None) -> pl.DataFrame:
    """Resolve ``runs`` to a DataFrame (optional mapping key)."""
    if isinstance(runs, pl.DataFrame):
        return runs
    if key is None:
        if len(runs) == 1:
            return next(iter(runs.values()))
        raise TypeError(
            "runs mapping needs an explicit key when it has multiple frames"
        )
    if key not in runs:
        raise KeyError(f"runs mapping missing {key!r}")
    return runs[key]


def _rows_from_summary(
    summary: pl.DataFrame,
    *,
    stage: str,
    window: str,
    arm_col: str,
    slice_col: str = "slice",
    metric_col: str = "metric",
    value_col: str = "value",
    n_col: str = "n",
) -> pl.DataFrame:
    """Map a scorer summary (arm/metric/slice/value/n) onto Measure rows."""
    if summary.is_empty():
        return empty_measure_metrics()
    missing = {arm_col, metric_col, value_col, n_col, slice_col} - set(summary.columns)
    if missing:
        raise ValueError(f"summary missing columns: {sorted(missing)}")
    rows: list[dict[str, object]] = []
    for row in summary.iter_rows(named=True):
        raw_val = row.get(value_col)
        rows.append(
            {
                "arm": str(row[arm_col]),
                "window": window,
                "metric": str(row[metric_col]),
                "value": None if raw_val is None else float(raw_val),
                "se": None,
                "n": int(row[n_col]),
                "stage": stage,
                "slice": str(row[slice_col]),
                "schema_version": SCHEMA_VERSION,
            }
        )
    return validate_measure_metrics(pl.DataFrame(rows))


class ComponentEvalMeasure:
    """Adapter over ``component_eval`` summary rows (no re-fit)."""

    @property
    def name(self) -> str:
        """Adapter id."""
        return "component_eval"

    def score(
        self,
        stage: str,
        runs: _Runs,
        outcomes: pl.DataFrame,
        slices: Sequence[str] | None = None,
        *,
        window: str = DEFAULT_WINDOW,
    ) -> pl.DataFrame:
        """Reshape a ``ComponentEvalSummarySchema`` frame into Measure rows.

        ``runs`` is the component-eval summary (or ``{"summary": …}``).
        ``outcomes`` is unused (actuals already folded into the summary).
        """
        del outcomes
        summary = _as_frame(runs, "summary" if isinstance(runs, Mapping) else None)
        metrics = _rows_from_summary(
            summary, stage=stage, window=window, arm_col="model_name"
        )
        return validate_measure_metrics(filter_slices(metrics, slices))


class MinutesMeasurementMeasure:
    """Adapter over ``minutes_measurement`` (summarise detail → Measure)."""

    def __init__(self, model_name: str = "minutes") -> None:
        """Bind the arm label written into the minutes summary."""
        self._model_name = model_name

    @property
    def name(self) -> str:
        """Adapter id."""
        return "minutes_measurement"

    def score(
        self,
        stage: str,
        runs: _Runs,
        outcomes: pl.DataFrame,
        slices: Sequence[str] | None = None,
        *,
        window: str = DEFAULT_WINDOW,
    ) -> pl.DataFrame:
        """Score minutes detail via ``summarise_minutes_detail``.

        ``runs`` is the minutes detail frame (or ``{"detail": …}``).
        ``outcomes`` is unused (actuals already on the detail frame).
        """
        del outcomes
        detail = _as_frame(runs, "detail" if isinstance(runs, Mapping) else None)
        if detail.is_empty():
            return empty_measure_metrics()
        summary = summarise_minutes_detail(detail, model_name=self._model_name)
        metrics = _rows_from_summary(
            summary, stage=stage, window=window, arm_col="model_name"
        )
        return validate_measure_metrics(filter_slices(metrics, slices))


class OtherPointsMeasurementMeasure:
    """Adapter over ``other_points_measurement`` (summarise detail → Measure)."""

    def __init__(self, model_name: str = "other") -> None:
        """Bind the arm label written into the other-points summary."""
        self._model_name = model_name

    @property
    def name(self) -> str:
        """Adapter id."""
        return "other_points_measurement"

    def score(
        self,
        stage: str,
        runs: _Runs,
        outcomes: pl.DataFrame,
        slices: Sequence[str] | None = None,
        *,
        window: str = DEFAULT_WINDOW,
    ) -> pl.DataFrame:
        """Score other-points detail via ``summarise_other_points_detail``.

        ``runs`` is the detail frame, or a mapping with ``detail`` and optional
        ``captain_regret``. ``outcomes`` is unused.
        """
        del outcomes
        if isinstance(runs, Mapping):
            detail = _as_frame(runs, "detail")
            captain = runs.get("captain_regret")
            if captain is None:
                captain = captain_regret_by_gw(detail)
        else:
            detail = runs
            captain = captain_regret_by_gw(detail)
        if detail.is_empty():
            return empty_measure_metrics()
        summary = summarise_other_points_detail(
            detail, captain, model_name=self._model_name
        )
        metrics = _rows_from_summary(
            summary, stage=stage, window=window, arm_col="model_name"
        )
        return validate_measure_metrics(filter_slices(metrics, slices))


class ForecastScoresMeasure:
    """Adapter over ``forecast_scores`` aggregations."""

    @property
    def name(self) -> str:
        """Adapter id."""
        return "forecast_scores"

    def score(
        self,
        stage: str,
        runs: _Runs,
        outcomes: pl.DataFrame,
        slices: Sequence[str] | None = None,
        *,
        window: str = DEFAULT_WINDOW,
    ) -> pl.DataFrame:
        """Aggregate forecast score rows (or accept a pre-built agg frame).

        ``runs`` is either a ``ForecastScoresSchema`` detail frame, a
        ``ForecastScoreAggSchema`` frame (detected via ``ci_lo``), or a mapping
        with ``scores`` / ``agg``. ``outcomes`` is unused when scores are
        already joined; pass an empty frame.
        """
        del outcomes
        if isinstance(runs, Mapping):
            if "agg" in runs:
                agg = runs["agg"]
            elif "scores" in runs:
                agg = aggregate_forecast_scores(runs["scores"])
            else:
                raise KeyError("runs mapping needs 'scores' or 'agg'")
        else:
            if runs.is_empty():
                return empty_measure_metrics()
            if "ci_lo" in runs.columns and "metric" in runs.columns:
                agg = runs
            else:
                agg = aggregate_forecast_scores(runs)
        if agg.is_empty():
            return empty_measure_metrics()
        metrics = _rows_from_summary(
            agg,
            stage=stage,
            window=window,
            arm_col="model_id",
        )
        return validate_measure_metrics(filter_slices(metrics, slices))


class DecisionsBoardMeasure:
    """Adapter over the decisions board (``summarise_decisions``)."""

    @property
    def name(self) -> str:
        """Adapter id."""
        return "decisions_board"

    def score(
        self,
        stage: str,
        runs: _Runs,
        outcomes: pl.DataFrame,
        slices: Sequence[str] | None = None,
        *,
        window: str = DEFAULT_WINDOW,
    ) -> pl.DataFrame:
        """Summarise decision-ledger rows into season-points Measure rows.

        ``runs`` is the decision ledger (actuals on the frame). ``outcomes`` is
        unused. Slice labels are ``all`` and ``season=<season>``.
        """
        del outcomes
        frame = _as_frame(runs, "decisions" if isinstance(runs, Mapping) else None)
        if frame.is_empty():
            return empty_measure_metrics()
        summary = summarise_decisions(frame)
        if summary.is_empty():
            return empty_measure_metrics()
        rows: list[dict[str, object]] = []
        for row in summary.iter_rows(named=True):
            arm = str(row["model_id"])
            season = str(row["season"])
            total = float(row["total_points"])
            n_gw = int(row["gameweeks"])
            for slice_name in ("all", f"season={season}"):
                rows.append(
                    {
                        "arm": arm,
                        "window": window,
                        "metric": "season_points",
                        "value": total,
                        "se": None,
                        "n": n_gw,
                        "stage": stage,
                        "slice": slice_name,
                        "schema_version": SCHEMA_VERSION,
                    }
                )
                if n_gw > 0:
                    rows.append(
                        {
                            "arm": arm,
                            "window": window,
                            "metric": "mean_points_per_gw",
                            "value": total / n_gw,
                            "se": None,
                            "n": n_gw,
                            "stage": stage,
                            "slice": slice_name,
                            "schema_version": SCHEMA_VERSION,
                        }
                    )
        metrics = validate_measure_metrics(pl.DataFrame(rows))
        return validate_measure_metrics(filter_slices(metrics, slices))


class SimulatedSeasonMeasure:
    """Adapter over ``simulated_season`` path totals."""

    @property
    def name(self) -> str:
        """Adapter id."""
        return "simulated_season"

    def score(
        self,
        stage: str,
        runs: _Runs,
        outcomes: pl.DataFrame,
        slices: Sequence[str] | None = None,
        *,
        window: str = DEFAULT_WINDOW,
    ) -> pl.DataFrame:
        """Map path-level policy totals onto Measure rows via mean totals.

        ``runs`` is a ``SimulatedSeasonPathSchema`` frame (or
        ``{"paths": …}``). ``outcomes`` is unused (truth already on paths).
        Values match ``mean_policy_totals``.
        """
        del outcomes
        paths = _as_frame(runs, "paths" if isinstance(runs, Mapping) else None)
        if paths.is_empty():
            return empty_measure_metrics()
        means = mean_policy_totals(paths)
        n_paths = int(paths.select("path_id").n_unique())
        has_share = (
            "share_of_attainable" in paths.columns
            and paths["share_of_attainable"].null_count() < paths.height
        )
        rows: list[dict[str, object]] = []
        for policy_id, mean_pts in means.items():
            rows.append(
                {
                    "arm": policy_id,
                    "window": window,
                    "metric": "season_points",
                    "value": float(mean_pts),
                    "se": None,
                    "n": n_paths,
                    "stage": stage,
                    "slice": "all",
                    "schema_version": SCHEMA_VERSION,
                }
            )
        if has_share:
            share_means = (
                paths.group_by("policy_id")
                .agg(pl.col("share_of_attainable").mean().alias("mean_share"))
                .sort("policy_id")
            )
            for row in share_means.iter_rows(named=True):
                rows.append(
                    {
                        "arm": str(row["policy_id"]),
                        "window": window,
                        "metric": "share_of_attainable",
                        "value": float(row["mean_share"]),
                        "se": None,
                        "n": n_paths,
                        "stage": stage,
                        "slice": "all",
                        "schema_version": SCHEMA_VERSION,
                    }
                )
        metrics = validate_measure_metrics(pl.DataFrame(rows))
        return validate_measure_metrics(filter_slices(metrics, slices))


ADAPTERS: dict[str, Measure] = {
    "component_eval": ComponentEvalMeasure(),
    "minutes_measurement": MinutesMeasurementMeasure(),
    "other_points_measurement": OtherPointsMeasurementMeasure(),
    "forecast_scores": ForecastScoresMeasure(),
    "decisions_board": DecisionsBoardMeasure(),
    "simulated_season": SimulatedSeasonMeasure(),
}


def get_measure(name: str) -> Measure:
    """Return a registered Measure adapter by name.

    Raises:
        KeyError: Unknown adapter name.
    """
    try:
        return ADAPTERS[name]
    except KeyError as exc:
        known = ", ".join(sorted(ADAPTERS))
        raise KeyError(f"unknown Measure {name!r}; known: {known}") from exc


__all__ = [
    "ADAPTERS",
    "DEFAULT_WINDOW",
    "MEASURE_STAGES",
    "ComponentEvalMeasure",
    "DecisionsBoardMeasure",
    "ForecastScoresMeasure",
    "Measure",
    "MinutesMeasurementMeasure",
    "OtherPointsMeasurementMeasure",
    "SimulatedSeasonMeasure",
    "empty_measure_metrics",
    "filter_slices",
    "get_measure",
    "validate_measure_metrics",
]
