"""Last-week page shaping: post_gw catalog, season-to-date, week detail.

Pure polars helpers for ``fplt ui`` (#396 / #397). No Streamlit / network —
callers load frames through ``DashboardStore`` / ``Store`` and pass them in.
Season-to-date panels lead; the selected gameweek is detail.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC
from math import sqrt
from typing import Any

import polars as pl

from fplt.backtest.eval_set import OPENFPL_COMPARE_FROM
from fplt.schemas.benchmark_projections import SOURCE_EP_NEXT, SOURCE_OPENFPL
from fplt.schemas.last_week import (
    LAST_WEEK_BENCHMARKS_COLUMNS,
    LAST_WEEK_CATALOG_COLUMNS,
    LAST_WEEK_LINES_STD_COLUMNS,
    LAST_WEEK_PIT_HIST_COLUMNS,
    LAST_WEEK_RELIABILITY_COLUMNS,
    LAST_WEEK_SELECTION_BIAS_COLUMNS,
    LAST_WEEK_SKILL_COLUMNS,
    LastWeekBenchmarksSchema,
    LastWeekCatalogSchema,
    LastWeekLinesStdSchema,
    LastWeekOpsReviewSchema,
    LastWeekPitHistSchema,
    LastWeekRankContextSchema,
    LastWeekReliabilitySchema,
    LastWeekSelectionBiasSchema,
    LastWeekSkillSchema,
)
from fplt.schemas.manifests import RunManifest
from fplt.utils.frames import require_dataframe

_BENCHMARK_ARMS: tuple[str, ...] = (SOURCE_EP_NEXT, SOURCE_OPENFPL)

_SKILL_METRICS: tuple[tuple[str, str], ...] = (
    ("mean_rps", "RPS (CRPS)"),
    ("mean_log_score", "Log score"),
    ("rps_skill_clim", "RPS skill vs climatology"),
    ("pit_coverage_50", "PIT 50% coverage"),
    ("pit_coverage_80", "PIT 80% coverage"),
    ("pit_coverage_90", "PIT 90% coverage"),
)

_RELIABILITY_EVENTS: tuple[tuple[str, str, str], ...] = (
    ("played", "p_played", "y_played"),
    ("ge60", "p_ge60", "y_ge60"),
    ("goal", "p_goal", "y_goal"),
    ("assist", "p_assist", "y_assist"),
    ("cs", "p_cs", "y_cs"),
    ("haul", "p_haul", "y_haul"),
    ("blank", "p_blank", "y_blank"),
)


@dataclass(frozen=True)
class PostGwSelection:
    """One post_gw analysis run chosen for the Last-week page."""

    run_id: str
    season: str
    gw: int
    model_id: str
    line: str | None


@dataclass(frozen=True)
class LastWeekSeasonPanel:
    """Season-to-date tables shown before the week detail."""

    lines: pl.DataFrame
    skill: pl.DataFrame
    pit_hist: pl.DataFrame
    reliability: pl.DataFrame
    selection_bias: pl.DataFrame
    season_to_date: pl.DataFrame


@dataclass(frozen=True)
class LastWeekDetailPanel:
    """Single-gameweek scorecard, attribution, misses, benchmarks, ops."""

    scorecard: pl.DataFrame
    attribution: pl.DataFrame
    biggest_misses: pl.DataFrame
    rank_context: pl.DataFrame
    benchmarks: pl.DataFrame
    ops_review: pl.DataFrame


def post_gw_catalog(manifests: Sequence[RunManifest]) -> pl.DataFrame:
    """Build a selector table from succeeded ``purpose=post_gw`` manifests.

    Args:
        manifests: Parsed run manifests (any purpose); non-post_gw rows drop.

    Returns:
        ``LastWeekCatalogSchema`` rows, newest gameweek first.
    """
    rows: list[dict[str, object]] = []
    for manifest in manifests:
        if manifest.purpose != "post_gw":
            continue
        first = manifest.as_of[0]
        finished = manifest.finished_at
        finished_s: str | None = None
        if finished is not None:
            stamp = (
                finished
                if finished.tzinfo is not None
                else finished.replace(tzinfo=UTC)
            )
            finished_s = stamp.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")
        rows.append(
            {
                "run_id": manifest.run_id,
                "season": first.season,
                "gw": first.gw,
                "model_id": manifest.model_id,
                "line": manifest.line,
                "finished_at": finished_s,
            }
        )
    if not rows:
        return require_dataframe(
            LastWeekCatalogSchema.validate(
                pl.DataFrame(schema=LAST_WEEK_CATALOG_COLUMNS)
            )
        )
    frame = require_dataframe(LastWeekCatalogSchema.validate(pl.DataFrame(rows)))
    return frame.sort(["season", "gw", "run_id"], descending=[True, True, True])


def default_post_gw_selection(catalog: pl.DataFrame) -> PostGwSelection | None:
    """Pick the newest post_gw run from a catalog frame.

    Args:
        catalog: ``LastWeekCatalogSchema`` rows.

    Returns:
        Selection for the first row, or ``None`` when empty.
    """
    if catalog.is_empty():
        return None
    row = catalog.row(0, named=True)
    return PostGwSelection(
        run_id=str(row["run_id"]),
        season=str(row["season"]),
        gw=int(row["gw"]),
        model_id=str(row["model_id"]),
        line=None if row["line"] is None else str(row["line"]),
    )


def selection_from_catalog(
    catalog: pl.DataFrame, *, season: str, gw: int
) -> PostGwSelection | None:
    """Resolve a catalog row for ``season`` / ``gw`` (newest run_id wins).

    Args:
        catalog: ``LastWeekCatalogSchema`` rows.
        season: Season key.
        gw: Gameweek.

    Returns:
        Matching selection, or ``None`` when absent.
    """
    part = catalog.filter((pl.col("season") == season) & (pl.col("gw") == gw))
    if part.is_empty():
        return None
    return default_post_gw_selection(part)


def lines_season_to_date(
    scorecards: pl.DataFrame,
    *,
    season: str,
    through_gw: int,
    season_to_date: pl.DataFrame | None = None,
) -> pl.DataFrame:
    """Roll scorecard chips / ex-chips into a season-to-date lines table.

    Args:
        scorecards: One or more ``PostGwScorecardSchema`` rows (any GWs).
        season: Season key to keep.
        through_gw: Inclusive GW ceiling.
        season_to_date: Optional ``SeasonToDateSchema`` for plan_ev_sum fill.

    Returns:
        ``LastWeekLinesStdSchema`` (empty when no matching scorecard rows).
    """
    empty = require_dataframe(
        LastWeekLinesStdSchema.validate(
            pl.DataFrame(schema=LAST_WEEK_LINES_STD_COLUMNS)
        )
    )
    if scorecards.is_empty():
        return empty
    part = scorecards.filter(
        (pl.col("season") == season) & (pl.col("gw") <= through_gw)
    )
    if part.is_empty():
        return empty
    rolled = part.group_by(["model_id", "season"]).agg(
        pl.lit(through_gw).alias("through_gw"),
        pl.col("gw").n_unique().alias("gameweeks"),
        pl.col("realised_points").sum().alias("points_with_chips"),
        pl.col("realised_ex_chips")
        .fill_null(pl.col("realised_points"))
        .sum()
        .alias("points_ex_chips"),
        pl.col("plan_ev").sum().alias("plan_ev_sum"),
    )
    if season_to_date is not None and not season_to_date.is_empty():
        std = season_to_date.filter(
            (pl.col("season") == season) & (pl.col("through_gw") == through_gw)
        )
        if not std.is_empty() and "plan_ev_sum" in std.columns:
            rolled = rolled.drop("plan_ev_sum").join(
                std.select("model_id", "season", "plan_ev_sum"),
                on=["model_id", "season"],
                how="left",
            )
    return require_dataframe(LastWeekLinesStdSchema.validate(rolled))


def skill_table(forecast_agg: pl.DataFrame, *, slice_name: str = "all") -> pl.DataFrame:
    """Extract headline CRPS / log / PIT coverage metrics from aggregations.

    Args:
        forecast_agg: ``ForecastScoreAggSchema`` rows (STD; ``gw`` null).
        slice_name: Aggregation slice (default ``all``).

    Returns:
        ``LastWeekSkillSchema`` rows in display order.
    """
    empty = require_dataframe(
        LastWeekSkillSchema.validate(pl.DataFrame(schema=LAST_WEEK_SKILL_COLUMNS))
    )
    if forecast_agg.is_empty():
        return empty
    part = forecast_agg.filter(pl.col("slice") == slice_name)
    if "gw" in part.columns:
        std = part.filter(pl.col("gw").is_null())
        if not std.is_empty():
            part = std
    rows: list[dict[str, object]] = []
    for metric, label in _SKILL_METRICS:
        hit = part.filter(pl.col("metric") == metric)
        if hit.is_empty():
            continue
        r = hit.row(0, named=True)
        rows.append(
            {
                "metric": metric,
                "label": label,
                "value": float(r["value"]),
                "ci_lo": r["ci_lo"],
                "ci_hi": r["ci_hi"],
                "n": int(r["n"]),
                "slice": slice_name,
            }
        )
    if not rows:
        return empty
    return require_dataframe(LastWeekSkillSchema.validate(pl.DataFrame(rows)))


def pit_histogram(forecast_agg: pl.DataFrame) -> pl.DataFrame:
    """Shape ``pit_hist`` aggregation rows for the ``pit_hist`` renderer.

    Args:
        forecast_agg: ``ForecastScoreAggSchema`` including ``metric=pit_hist``.

    Returns:
        ``LastWeekPitHistSchema`` densitiy by bin.
    """
    empty = require_dataframe(
        LastWeekPitHistSchema.validate(pl.DataFrame(schema=LAST_WEEK_PIT_HIST_COLUMNS))
    )
    if forecast_agg.is_empty():
        return empty
    part = forecast_agg.filter(pl.col("metric") == "pit_hist")
    if part.is_empty():
        return empty
    rows: list[dict[str, object]] = []
    for r in part.iter_rows(named=True):
        label = str(r["slice"])
        lo, hi = _parse_bin_slice(label)
        rows.append(
            {
                "bin_lo": lo,
                "bin_hi": hi,
                "density": float(r["value"]),
                "label": label.removeprefix("bin="),
            }
        )
    frame = pl.DataFrame(rows).sort("bin_lo")
    return require_dataframe(LastWeekPitHistSchema.validate(frame))


def reliability_diagram(
    forecast_scores: pl.DataFrame, *, n_bins: int = 10
) -> pl.DataFrame:
    """Build event reliability rows from per-player-fixture forecast scores.

    Args:
        forecast_scores: ``ForecastScoresSchema`` (season-to-date or one GW).
        n_bins: Equal-width probability bins on ``[0, 1]`` (default 10).

    Returns:
        ``LastWeekReliabilitySchema`` for the ``reliability`` renderer.
    """
    empty = require_dataframe(
        LastWeekReliabilitySchema.validate(
            pl.DataFrame(schema=LAST_WEEK_RELIABILITY_COLUMNS)
        )
    )
    if forecast_scores.is_empty() or n_bins < 1:
        return empty
    edges = [i / float(n_bins) for i in range(n_bins + 1)]
    rows: list[dict[str, object]] = []
    for event, p_col, y_col in _RELIABILITY_EVENTS:
        if p_col not in forecast_scores.columns or y_col not in forecast_scores.columns:
            continue
        usable = forecast_scores.filter(
            pl.col(p_col).is_not_null() & pl.col(y_col).is_not_null()
        )
        if usable.is_empty():
            continue
        for i in range(n_bins):
            lo, hi = edges[i], edges[i + 1]
            if i == n_bins - 1:
                mask = (pl.col(p_col) >= lo) & (pl.col(p_col) <= hi)
            else:
                mask = (pl.col(p_col) >= lo) & (pl.col(p_col) < hi)
            bucket = usable.filter(mask)
            if bucket.is_empty():
                continue
            mean_pred = float(bucket[p_col].mean())  # type: ignore[arg-type]
            observed = float(bucket[y_col].mean())  # type: ignore[arg-type]
            rows.append(
                {
                    "event": event,
                    "pred_bin": f"{lo:.1f}-{hi:.1f}",
                    "mean_pred": mean_pred,
                    "observed": observed,
                    "n": int(bucket.height),
                }
            )
    if not rows:
        return empty
    return require_dataframe(LastWeekReliabilitySchema.validate(pl.DataFrame(rows)))


def selection_bias_table(forecast_agg: pl.DataFrame) -> pl.DataFrame:
    """Extract selection-conditional bias metrics from aggregations.

    Args:
        forecast_agg: ``ForecastScoreAggSchema`` with ``line=…`` slices.

    Returns:
        ``LastWeekSelectionBiasSchema`` rows.
    """
    empty = require_dataframe(
        LastWeekSelectionBiasSchema.validate(
            pl.DataFrame(schema=LAST_WEEK_SELECTION_BIAS_COLUMNS)
        )
    )
    if forecast_agg.is_empty():
        return empty
    part = forecast_agg.filter(
        pl.col("metric").is_in(["selection_bias", "selection_bias_delta"])
        & pl.col("slice").str.starts_with("line=")
    )
    if part.is_empty():
        return empty
    rows = [
        {
            "line": str(r["slice"]).removeprefix("line="),
            "metric": str(r["metric"]),
            "value": float(r["value"]),
            "ci_lo": r["ci_lo"],
            "ci_hi": r["ci_hi"],
            "n": int(r["n"]),
        }
        for r in part.iter_rows(named=True)
    ]
    return require_dataframe(LastWeekSelectionBiasSchema.validate(pl.DataFrame(rows)))


def season_panel(
    *,
    scorecards: pl.DataFrame,
    season_to_date: pl.DataFrame,
    forecast_agg: pl.DataFrame,
    forecast_scores: pl.DataFrame,
    season: str,
    through_gw: int,
) -> LastWeekSeasonPanel:
    """Assemble every season-to-date panel for the Last-week page.

    Args:
        scorecards: Combined scorecards across post_gw runs (chips / ex-chips).
        season_to_date: Rollup from the selected run.
        forecast_agg: STD aggregations from the selected run.
        forecast_scores: Per-row scores (STD preferred) for reliability.
        season: Season key.
        through_gw: Inclusive GW of the selected analysis run.

    Returns:
        Shaped ``LastWeekSeasonPanel``.
    """
    return LastWeekSeasonPanel(
        lines=lines_season_to_date(
            scorecards,
            season=season,
            through_gw=through_gw,
            season_to_date=season_to_date,
        ),
        skill=skill_table(forecast_agg),
        pit_hist=pit_histogram(forecast_agg),
        reliability=reliability_diagram(forecast_scores),
        selection_bias=selection_bias_table(forecast_agg),
        season_to_date=season_to_date,
    )


def week_detail(
    *,
    scorecard: pl.DataFrame,
    attribution: pl.DataFrame,
    biggest_misses: pl.DataFrame,
    primary: pl.DataFrame | None = None,
    benchmarks: pl.DataFrame | None = None,
    outcomes: pl.DataFrame | None = None,
    season: str | None = None,
    gw: int | None = None,
    ops_rows: pl.DataFrame | None = None,
) -> LastWeekDetailPanel:
    """Assemble week-detail panels including #397 rank / bench / ops.

    Args:
        scorecard: ``PostGwScorecardSchema`` for the selected GW.
        attribution: ``PostGwAttributionSchema`` for the selected GW.
        biggest_misses: ``PostGwBiggestMissSchema`` for the selected GW.
        primary: Optional primary ``player_fixture`` predictions.
        benchmarks: Optional ``BenchmarkProjectionSchema`` archive rows.
        outcomes: Optional ``player_fixture_outcomes`` (``pts`` column).
        season: Season for benchmark filter (defaults from scorecard).
        gw: Gameweek for benchmark filter (defaults from scorecard).
        ops_rows: Optional pre-shaped ``LastWeekOpsReviewSchema`` rows.

    Returns:
        ``LastWeekDetailPanel``.
    """
    season_key = season
    gw_key = gw
    if (season_key is None or gw_key is None) and not scorecard.is_empty():
        season_key = season_key or str(scorecard["season"][0])
        gw_key = gw_key if gw_key is not None else int(scorecard["gw"][0])
    bench = (
        benchmarks_table(
            primary if primary is not None else pl.DataFrame(),
            benchmarks if benchmarks is not None else pl.DataFrame(),
            outcomes if outcomes is not None else pl.DataFrame(),
            season=season_key or "",
            gw=gw_key or 1,
        )
        if season_key is not None and gw_key is not None
        else _empty_benchmarks()
    )
    return LastWeekDetailPanel(
        scorecard=scorecard,
        attribution=attribution,
        biggest_misses=biggest_misses,
        rank_context=rank_context(scorecard),
        benchmarks=bench,
        ops_review=(
            ops_rows
            if ops_rows is not None
            else ops_review_table(applied=None, freshness=[], timings=[], alarms=[])
        ),
    )


def rank_context(scorecard: pl.DataFrame) -> pl.DataFrame:
    """Shape FPL average / highest and rank change from a scorecard row.

    Args:
        scorecard: ``PostGwScorecardSchema`` (one GW preferred).

    Returns:
        ``LastWeekRankContextSchema``; values null when missing.
    """
    fields: tuple[tuple[str, str], ...] = (
        ("fpl_average", "FPL average"),
        ("fpl_highest", "FPL highest"),
        ("rank_change", "Rank change"),
    )
    if scorecard.is_empty():
        return require_dataframe(
            LastWeekRankContextSchema.validate(
                pl.DataFrame(
                    [
                        {
                            "field": key,
                            "label": label,
                            "value": None,
                            "note": "no scorecard",
                        }
                        for key, label in fields
                    ]
                )
            )
        )
    row = scorecard.row(0, named=True)
    rows: list[dict[str, object]] = []
    for key, label in fields:
        raw = row.get(key) if key in scorecard.columns else None
        note = None if raw is not None else "unavailable"
        if key == "rank_change" and raw is not None:
            note = "positive = climbed"
        rows.append({"field": key, "label": label, "value": raw, "note": note})
    return require_dataframe(LastWeekRankContextSchema.validate(pl.DataFrame(rows)))


def benchmarks_table(
    primary: pl.DataFrame,
    benchmarks: pl.DataFrame,
    outcomes: pl.DataFrame,
    *,
    season: str,
    gw: int,
) -> pl.DataFrame:
    """Compare primary vs ep_next / OpenFPL on the same fixtures.

    OpenFPL is omitted (with a note row) when ``season`` is before
    ``OPENFPL_COMPARE_FROM``. Missing inputs degrade to empty / note rows.

    Args:
        primary: Primary predictions with ``player_key``, ``fixture_id``, ``xpts``.
        benchmarks: ``BenchmarkProjectionSchema`` rows (any as_of window).
        outcomes: Outcomes with ``player_key``, ``fixture_id``, and ``pts``.
        season: Season key.
        gw: Target gameweek.

    Returns:
        ``LastWeekBenchmarksSchema`` (primary + each archive arm).
    """
    empty = _empty_benchmarks()
    if not season:
        return empty
    actuals = _outcome_pts(outcomes)
    ours = _arm_on_fixtures(primary, actuals, arm="primary")
    rows: list[dict[str, object]] = [ours]
    archive = pl.DataFrame() if benchmarks.is_empty() else benchmarks
    if not archive.is_empty():
        archive = archive.filter(
            (pl.col("as_of_season") == season)
            & (pl.col("as_of_gw") == gw)
            & (pl.col("target_season") == season)
            & (pl.col("target_gw") == gw)
        )
    for source in _BENCHMARK_ARMS:
        if source == SOURCE_OPENFPL and _season_before(season, OPENFPL_COMPARE_FROM):
            rows.append(
                {
                    "arm": source,
                    "n": 0,
                    "rmse": None,
                    "mae": None,
                    "mean_error": None,
                    "note": f"season guard: OpenFPL from {OPENFPL_COMPARE_FROM}",
                }
            )
            continue
        if archive.is_empty():
            rows.append(
                {
                    "arm": source,
                    "n": 0,
                    "rmse": None,
                    "mae": None,
                    "mean_error": None,
                    "note": "no benchmark archive",
                }
            )
            continue
        part = archive.filter(pl.col("source") == source).select(
            "player_key", "fixture_id", "xpts"
        )
        if part.is_empty():
            rows.append(
                {
                    "arm": source,
                    "n": 0,
                    "rmse": None,
                    "mae": None,
                    "mean_error": None,
                    "note": "no rows for source",
                }
            )
            continue
        # Same-fixture intersection with primary when primary has scored rows.
        if not primary.is_empty() and {"player_key", "fixture_id"}.issubset(
            set(primary.columns)
        ):
            keys = primary.select("player_key", "fixture_id").unique()
            part = part.join(keys, on=["player_key", "fixture_id"], how="inner")
        rows.append(_arm_on_fixtures(part, actuals, arm=source))
    return require_dataframe(LastWeekBenchmarksSchema.validate(pl.DataFrame(rows)))


def ops_review_table(
    *,
    applied: Mapping[str, Any] | None,
    freshness: Sequence[Mapping[str, Any]],
    timings: Sequence[Mapping[str, Any]],
    alarms: Sequence[Mapping[str, Any]],
    last_error: str | None = None,
) -> pl.DataFrame:
    """Shape the ops review section from local applied / status / timing / alarms.

    Args:
        applied: Optional dict with ``checkpoint``, ``line``, ``plan_run_id``,
            ``submitted_at``.
        freshness: Capture / heartbeat rows with ``key`` and ``value``
            (optional ``detail``).
        timings: Checkpoint timing rows with ``key`` / ``value`` / ``detail``.
        alarms: Alarm marker rows with ``key`` / ``value`` / ``detail``.
        last_error: Optional last-error message.

    Returns:
        ``LastWeekOpsReviewSchema`` (empty schema only when nothing at all).
    """
    rows: list[dict[str, object]] = []
    if applied:
        rows.extend(
            [
                {
                    "section": "applied",
                    "key": "checkpoint",
                    "value": _as_str(applied.get("checkpoint")),
                    "detail": None,
                },
                {
                    "section": "applied",
                    "key": "line",
                    "value": _as_str(applied.get("line")),
                    "detail": None,
                },
                {
                    "section": "applied",
                    "key": "plan_run_id",
                    "value": _as_str(applied.get("plan_run_id")),
                    "detail": None,
                },
                {
                    "section": "applied",
                    "key": "submitted_at",
                    "value": _as_str(applied.get("submitted_at")),
                    "detail": None,
                },
            ]
        )
    else:
        rows.append(
            {
                "section": "applied",
                "key": "checkpoint",
                "value": None,
                "detail": "no applied.json",
            }
        )
    if freshness:
        for item in freshness:
            rows.append(
                {
                    "section": "freshness",
                    "key": str(item.get("key", "source")),
                    "value": _as_str(item.get("value")),
                    "detail": _as_str(item.get("detail")),
                }
            )
    else:
        rows.append(
            {
                "section": "freshness",
                "key": "captures",
                "value": None,
                "detail": "no status freshness",
            }
        )
    if timings:
        for item in timings:
            rows.append(
                {
                    "section": "timing",
                    "key": str(item.get("key", "checkpoint")),
                    "value": _as_str(item.get("value")),
                    "detail": _as_str(item.get("detail")),
                }
            )
    else:
        rows.append(
            {
                "section": "timing",
                "key": "ops_timing",
                "value": None,
                "detail": "no ops_timing.json",
            }
        )
    if alarms:
        for item in alarms:
            rows.append(
                {
                    "section": "alarm",
                    "key": str(item.get("key", "alarm")),
                    "value": _as_str(item.get("value")),
                    "detail": _as_str(item.get("detail")),
                }
            )
    else:
        rows.append(
            {
                "section": "alarm",
                "key": "alarms",
                "value": "none",
                "detail": None,
            }
        )
    if last_error:
        rows.append(
            {
                "section": "failure",
                "key": "last_error",
                "value": last_error,
                "detail": None,
            }
        )
    else:
        rows.append(
            {
                "section": "failure",
                "key": "last_error",
                "value": None,
                "detail": "none recorded",
            }
        )
    return require_dataframe(LastWeekOpsReviewSchema.validate(pl.DataFrame(rows)))


def header_fields(
    selection: PostGwSelection, *, finished_at: str | None = None
) -> dict[str, Any]:
    """Build header caption fields for the Last-week page.

    Args:
        selection: Chosen post_gw run.
        finished_at: Optional ISO finish stamp from the catalog.

    Returns:
        Dict of display fields.
    """
    return {
        "season": selection.season,
        "gw": selection.gw,
        "run_id": selection.run_id,
        "model_id": selection.model_id,
        "line": selection.line or "—",
        "finished_at": finished_at or "—",
    }


def _empty_benchmarks() -> pl.DataFrame:
    return require_dataframe(
        LastWeekBenchmarksSchema.validate(
            pl.DataFrame(schema=LAST_WEEK_BENCHMARKS_COLUMNS)
        )
    )


def _outcome_pts(outcomes: pl.DataFrame) -> pl.DataFrame:
    if outcomes.is_empty():
        return pl.DataFrame(
            schema={
                "player_key": pl.Int64(),
                "fixture_id": pl.Int64(),
                "pts": pl.Int64(),
            }
        )
    cols = set(outcomes.columns)
    if "pts" in cols:
        return outcomes.select("player_key", "fixture_id", "pts")
    if "actual_pts" in cols:
        return outcomes.select(
            "player_key",
            "fixture_id",
            pl.col("actual_pts").alias("pts"),
        )
    return pl.DataFrame(
        schema={
            "player_key": pl.Int64(),
            "fixture_id": pl.Int64(),
            "pts": pl.Int64(),
        }
    )


def _arm_on_fixtures(
    preds: pl.DataFrame, actuals: pl.DataFrame, *, arm: str
) -> dict[str, object]:
    need = {"player_key", "fixture_id", "xpts"}
    if preds.is_empty() or not need.issubset(set(preds.columns)) or actuals.is_empty():
        note = "no predictions" if arm == "primary" else "no scored overlap"
        if preds.is_empty() and arm == "primary":
            note = "no primary predictions"
        elif actuals.is_empty():
            note = "no outcomes"
        return {
            "arm": arm,
            "n": 0,
            "rmse": None,
            "mae": None,
            "mean_error": None,
            "note": note,
        }
    joined = preds.select("player_key", "fixture_id", "xpts").join(
        actuals, on=["player_key", "fixture_id"], how="inner"
    )
    if joined.is_empty():
        return {
            "arm": arm,
            "n": 0,
            "rmse": None,
            "mae": None,
            "mean_error": None,
            "note": "no scored overlap",
        }
    err = (joined["xpts"] - joined["pts"].cast(pl.Float64)).to_list()
    n = len(err)
    mae = sum(abs(e) for e in err) / n
    mse = sum(e * e for e in err) / n
    mean_err = sum(err) / n
    return {
        "arm": arm,
        "n": n,
        "rmse": sqrt(mse),
        "mae": mae,
        "mean_error": mean_err,
        "note": None,
    }


def _season_before(season: str, floor: str) -> bool:
    try:
        return int(season.split("-", 1)[0]) < int(floor.split("-", 1)[0])
    except (ValueError, IndexError):
        return True


def _as_str(value: object) -> str | None:
    if value is None:
        return None
    return str(value)


def _parse_bin_slice(label: str) -> tuple[float, float]:
    text = label.removeprefix("bin=")
    try:
        left, right = text.split("-", 1)
        return float(left), float(right)
    except ValueError:
        return 0.0, 1.0
