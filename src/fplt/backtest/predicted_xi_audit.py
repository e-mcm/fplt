"""Predicted-XI accuracy audit vs official starts (FDB-253 / PR8).

Measurement only — no keep A/B and no ``PRODUCTION_MODELS`` flip. Expands
``score_predicted_starts`` with source / hours-to-KO / confidence slices,
start recall, bench→start false-bench rate, and fixture coverage. Use before
promoting ``dc_layered_predicted_xi_v1``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from fplt.data.predicted_lineups import ROLE_BENCH, ROLE_START, score_predicted_starts
from fplt.data.store import DataStore
from fplt.schemas.predicted_xi_audit import (
    PREDICTED_XI_AUDIT_DETAIL_COLUMNS,
    PREDICTED_XI_AUDIT_SUMMARY_COLUMNS,
    PredictedXiAuditDetailSchema,
    PredictedXiAuditSummarySchema,
)
from fplt.utils.frames import require_dataframe

logger = logging.getLogger(__name__)

# Promote-smoke default: abort keep claims below this start precision.
DEFAULT_MIN_START_ACCURACY = 0.55

_HOUR_BUCKETS: tuple[tuple[str, float | None, float | None], ...] = (
    ("0-6", 0.0, 6.0),
    ("6-24", 6.0, 24.0),
    ("24-72", 24.0, 72.0),
    ("72+", 72.0, None),
)

_CONF_BUCKETS: tuple[tuple[str, float, float], ...] = (
    ("0.0-0.5", 0.0, 0.5),
    ("0.5-0.8", 0.5, 0.8),
    ("0.8-1.0", 0.8, 1.0000001),
)


@dataclass(frozen=True)
class PredictedXiAuditResult:
    """Summary metrics, detail rows, and the directory they were written to."""

    summary: pl.DataFrame
    detail: pl.DataFrame
    output_dir: Path
    passed_min_accuracy: bool


def run_predicted_xi_audit(
    data_dir: Path,
    season: str,
    *,
    from_gw: int = 1,
    to_gw: int = 38,
    source: str | None = None,
    min_accuracy: float = DEFAULT_MIN_START_ACCURACY,
    output_dir: Path | None = None,
) -> PredictedXiAuditResult:
    """Score staged predicted lineups against official starts for ``season``.

    Args:
        data_dir: Root with ``staged/``.
        season: Evaluation season such as ``"2025-26"``.
        from_gw: First fixture gameweek included.
        to_gw: Last fixture gameweek, inclusive.
        source: Optional ``predicted_lineups.source`` filter.
        min_accuracy: Start-precision gate for ``passed_min_accuracy``.
        output_dir: Where to write parquet / JSON / markdown.

    Returns:
        Summary, detail, output path, and whether overall start precision
        meets ``min_accuracy`` (false when there are no scored starts).

    Raises:
        ValueError: Gameweek range is invalid or ``min_accuracy`` is outside
            ``[0, 1]``.
    """
    if not 1 <= from_gw <= to_gw <= 38:
        raise ValueError(f"gameweeks must be in 1..38, got {from_gw}..{to_gw}")
    if not 0.0 <= min_accuracy <= 1.0:
        raise ValueError(f"min_accuracy must be in [0, 1], got {min_accuracy}")

    store = DataStore(data_dir)
    detail = build_audit_detail(
        store, season, from_gw=from_gw, to_gw=to_gw, source=source
    )
    summary = summarise_audit(detail, store, season, from_gw=from_gw, to_gw=to_gw)
    checked_summary = PredictedXiAuditSummarySchema.validate(summary)
    checked_summary = require_dataframe(checked_summary)
    checked_detail = PredictedXiAuditDetailSchema.validate(detail)
    checked_detail = require_dataframe(checked_detail)
    overall = checked_summary.filter(
        (pl.col("metric") == "start_precision") & (pl.col("slice") == "all")
    )
    if overall.is_empty() or overall["value"][0] is None:
        passed = False
    else:
        passed = float(overall["value"][0]) >= min_accuracy

    out = output_dir or (
        data_dir / "predicted_xi_audit" / f"{season}_gw{from_gw}-{to_gw}"
    )
    out.mkdir(parents=True, exist_ok=True)
    checked_summary.write_parquet(out / "summary.parquet")
    checked_detail.write_parquet(out / "detail.parquet")
    (out / "summary.json").write_text(
        json.dumps(
            {
                "season": season,
                "from_gw": from_gw,
                "to_gw": to_gw,
                "source": source,
                "min_accuracy": min_accuracy,
                "passed_min_accuracy": passed,
                "rows": checked_summary.to_dicts(),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "report.md").write_text(
        format_audit_markdown(
            checked_summary, passed=passed, min_accuracy=min_accuracy
        ),
        encoding="utf-8",
    )
    logger.info("wrote predicted XI audit under %s (passed=%s)", out, passed)
    return PredictedXiAuditResult(
        summary=checked_summary,
        detail=checked_detail,
        output_dir=out,
        passed_min_accuracy=passed,
    )


def build_audit_detail(
    store: DataStore,
    season: str,
    *,
    from_gw: int,
    to_gw: int,
    source: str | None = None,
) -> pl.DataFrame:
    """Join predicted starts to official labels with hours-to-KO context.

    Uses the latest predicted row per ``(player_key, fixture_id, team_key)``
    that still has a joinable ``fixture_id``. Empty when lineups or stats are
    missing.

    Args:
        store: Staged tables.
        season: Season label.
        from_gw: First fixture gameweek.
        to_gw: Last fixture gameweek, inclusive.
        source: Optional lineup source filter.

    Returns:
        A ``PredictedXiAuditDetailSchema`` frame (possibly empty).
    """
    empty = pl.DataFrame(schema=PREDICTED_XI_AUDIT_DETAIL_COLUMNS)
    if not store.path("predicted_lineups").exists():
        return empty
    predicted = store.read("predicted_lineups").filter(pl.col("season") == season)
    if source is not None:
        predicted = predicted.filter(pl.col("source") == source)
    if predicted.is_empty():
        return empty

    fixtures = (
        store.read("fixtures")
        .filter(pl.col("season") == season)
        .filter((pl.col("gw") >= from_gw) & (pl.col("gw") <= to_gw))
        .sort("available_at")
        .unique(subset=["fixture_id"], keep="last")
        .select("fixture_id", "gw", "kickoff_time")
    )
    if fixtures.is_empty():
        return empty

    latest = (
        predicted.filter(
            pl.col("fixture_id").is_not_null() & pl.col("player_key").is_not_null()
        )
        .sort("available_at")
        .unique(subset=["season", "team_key", "fixture_id", "player_key"], keep="last")
    )
    framed = latest.join(fixtures, on="fixture_id", how="inner").with_columns(
        ((pl.col("kickoff_time") - pl.col("available_at")).dt.total_hours()).alias(
            "hours_to_ko"
        )
    )
    starts = framed.filter(pl.col("role") == ROLE_START)
    if starts.is_empty():
        return empty

    scored = score_predicted_starts(starts, store.read("player_fixture_stats"))
    if scored.is_empty():
        return empty

    meta = starts.select(
        "season",
        "fixture_id",
        "player_key",
        "team_key",
        "source",
        "role",
        "confidence",
        "hours_to_ko",
        "gw",
    )
    return (
        scored.join(
            meta, on=["season", "fixture_id", "player_key", "source"], how="left"
        )
        .select(list(PREDICTED_XI_AUDIT_DETAIL_COLUMNS))
        .sort(["gw", "fixture_id", "player_key"])
    )


def summarise_audit(
    detail: pl.DataFrame,
    store: DataStore,
    season: str,
    *,
    from_gw: int,
    to_gw: int,
) -> pl.DataFrame:
    """Build summary metric rows from audit detail plus coverage / recall.

    Args:
        detail: Output of ``build_audit_detail``.
        store: Staged tables (for recall + coverage denominators).
        season: Season label.
        from_gw: First fixture gameweek.
        to_gw: Last fixture gameweek, inclusive.

    Returns:
        A ``PredictedXiAuditSummarySchema`` frame.
    """
    rows: list[dict[str, object]] = []
    if not detail.is_empty():
        rows.extend(_precision_rows(detail, "all"))
        for src in sorted(detail["source"].unique().to_list()):
            rows.extend(
                _precision_rows(detail.filter(pl.col("source") == src), f"source={src}")
            )
        for label, lo, hi in _HOUR_BUCKETS:
            part = _hours_slice(detail, lo, hi)
            rows.extend(_precision_rows(part, f"hours_to_ko={label}"))
        for label, lo, hi in _CONF_BUCKETS:
            part = detail.filter(
                pl.col("confidence").is_not_null()
                & (pl.col("confidence") >= lo)
                & (pl.col("confidence") < hi)
            )
            rows.extend(_precision_rows(part, f"confidence={label}"))

    rows.extend(
        _recall_rows(store, season, from_gw=from_gw, to_gw=to_gw, detail=detail)
    )
    rows.extend(_bench_false_start_rows(store, season, from_gw=from_gw, to_gw=to_gw))
    rows.extend(_coverage_rows(store, season, from_gw=from_gw, to_gw=to_gw))

    if not rows:
        return pl.DataFrame(schema=PREDICTED_XI_AUDIT_SUMMARY_COLUMNS)
    return pl.DataFrame(rows, schema=PREDICTED_XI_AUDIT_SUMMARY_COLUMNS)


def format_audit_markdown(
    summary: pl.DataFrame,
    *,
    passed: bool,
    min_accuracy: float,
) -> str:
    """Render the audit summary as a short markdown report."""
    gate = "PASS" if passed else "FAIL"
    lines = [
        "# Predicted XI audit",
        "",
        f"Start-precision gate (≥ {min_accuracy:.2f}): **{gate}**",
        "",
        "| metric | slice | value | n |",
        "|---|---|---:|---:|",
    ]
    if summary.is_empty():
        lines.append("| _(empty)_ | | | |")
    else:
        for row in summary.sort(["metric", "slice"]).iter_rows(named=True):
            value = row["value"]
            value_s = "" if value is None else f"{float(value):.4f}"
            lines.append(
                f"| {row['metric']} | {row['slice']} | {value_s} | {row['n']} |"
            )
    lines.append("")
    return "\n".join(lines)


def format_summary_table(summary: pl.DataFrame) -> str:
    """Pretty-print the audit summary for the CLI."""
    if summary.is_empty():
        return "(no predicted XI audit rows)"
    return str(summary.sort(["metric", "slice"]))


def _precision_rows(detail: pl.DataFrame, slice_name: str) -> list[dict[str, object]]:
    n = detail.height
    if n == 0:
        return [
            {
                "metric": "start_precision",
                "slice": slice_name,
                "value": None,
                "n": 0,
            }
        ]
    correct = float(detail["correct"].sum())
    return [
        {
            "metric": "start_precision",
            "slice": slice_name,
            "value": correct / n,
            "n": n,
        }
    ]


def _hours_slice(
    detail: pl.DataFrame, lo: float | None, hi: float | None
) -> pl.DataFrame:
    framed = detail.filter(pl.col("hours_to_ko").is_not_null())
    if lo is not None:
        framed = framed.filter(pl.col("hours_to_ko") >= lo)
    if hi is not None:
        framed = framed.filter(pl.col("hours_to_ko") < hi)
    return framed


def _recall_rows(
    store: DataStore,
    season: str,
    *,
    from_gw: int,
    to_gw: int,
    detail: pl.DataFrame,
) -> list[dict[str, object]]:
    """Share of official starters that appear as a predicted start."""
    stats = _official_starters(store, season, from_gw=from_gw, to_gw=to_gw)
    n = stats.height
    if n == 0:
        return [{"metric": "start_recall", "slice": "all", "value": None, "n": 0}]
    if detail.is_empty():
        return [{"metric": "start_recall", "slice": "all", "value": 0.0, "n": n}]
    predicted_keys = detail.select("season", "fixture_id", "player_key").unique()
    hit = stats.join(
        predicted_keys, on=["season", "fixture_id", "player_key"], how="inner"
    ).height
    return [
        {
            "metric": "start_recall",
            "slice": "all",
            "value": hit / n,
            "n": n,
        }
    ]


def _bench_false_start_rows(
    store: DataStore,
    season: str,
    *,
    from_gw: int,
    to_gw: int,
) -> list[dict[str, object]]:
    """Rate at which predicted ``bench`` players actually started (sub/XI miss)."""
    if not store.path("predicted_lineups").exists():
        return [{"metric": "bench_started_rate", "slice": "all", "value": None, "n": 0}]
    fixtures = (
        store.read("fixtures")
        .filter(pl.col("season") == season)
        .filter((pl.col("gw") >= from_gw) & (pl.col("gw") <= to_gw))
        .select("fixture_id")
        .unique()
    )
    benches = (
        store.read("predicted_lineups")
        .filter(pl.col("season") == season)
        .filter(pl.col("role") == ROLE_BENCH)
        .filter(pl.col("fixture_id").is_not_null() & pl.col("player_key").is_not_null())
        .join(fixtures, on="fixture_id", how="inner")
        .sort("available_at")
        .unique(subset=["season", "fixture_id", "player_key"], keep="last")
    )
    if benches.is_empty():
        return [{"metric": "bench_started_rate", "slice": "all", "value": None, "n": 0}]
    labels = _official_starter_flags(store, season)
    joined = benches.join(labels, on=["season", "fixture_id", "player_key"], how="left")
    started = joined.with_columns(pl.col("actual_started").fill_null(False))
    n = started.height
    rate = float(started["actual_started"].sum()) / n
    return [
        {
            "metric": "bench_started_rate",
            "slice": "all",
            "value": rate,
            "n": n,
        }
    ]


def _coverage_rows(
    store: DataStore,
    season: str,
    *,
    from_gw: int,
    to_gw: int,
) -> list[dict[str, object]]:
    """Share of team-fixtures with a full (≥11) predicted XI."""
    fixtures = (
        store.read("fixtures")
        .filter(pl.col("season") == season)
        .filter((pl.col("gw") >= from_gw) & (pl.col("gw") <= to_gw))
        .sort("available_at")
        .unique(subset=["fixture_id"], keep="last")
        .select("fixture_id", "team_h_key", "team_a_key")
    )
    if fixtures.is_empty() or not store.path("predicted_lineups").exists():
        return [
            {
                "metric": "full_xi_coverage",
                "slice": "team_fixture",
                "value": None,
                "n": 0,
            }
        ]
    team_fixtures = pl.concat(
        [
            fixtures.select("fixture_id", pl.col("team_h_key").alias("team_key")),
            fixtures.select("fixture_id", pl.col("team_a_key").alias("team_key")),
        ]
    ).unique()
    n = team_fixtures.height
    starts = (
        store.read("predicted_lineups")
        .filter(pl.col("season") == season)
        .filter(pl.col("role") == ROLE_START)
        .filter(pl.col("fixture_id").is_not_null())
        .sort("available_at")
        .unique(
            subset=["season", "team_key", "fixture_id", "player_key"],
            keep="last",
        )
        .group_by(["fixture_id", "team_key"])
        .len()
        .filter(pl.col("len") >= 11)
        .select("fixture_id", "team_key", pl.lit(True).alias("full_xi"))
    )
    covered = team_fixtures.join(
        starts, on=["fixture_id", "team_key"], how="inner"
    ).height
    return [
        {
            "metric": "full_xi_coverage",
            "slice": "team_fixture",
            "value": covered / n if n else None,
            "n": n,
        }
    ]


def _official_starters(
    store: DataStore,
    season: str,
    *,
    from_gw: int,
    to_gw: int,
) -> pl.DataFrame:
    fixtures = (
        store.read("fixtures")
        .filter(pl.col("season") == season)
        .filter((pl.col("gw") >= from_gw) & (pl.col("gw") <= to_gw))
        .select("fixture_id")
        .unique()
    )
    return (
        _official_starter_flags(store, season)
        .filter(pl.col("actual_started"))
        .join(fixtures, on="fixture_id", how="inner")
        .select("season", "fixture_id", "player_key")
    )


def _official_starter_flags(store: DataStore, season: str) -> pl.DataFrame:
    stats = store.read("player_fixture_stats").filter(pl.col("season") == season)
    if "available_at" in stats.columns:
        stats = stats.sort("available_at").unique(
            subset=["season", "player_key", "fixture_id"], keep="last"
        )
    else:
        stats = stats.unique(subset=["season", "player_key", "fixture_id"], keep="last")
    return stats.select(
        "season",
        "fixture_id",
        "player_key",
        (pl.col("starts") > 0).alias("actual_started"),
    )


__all__ = [
    "DEFAULT_MIN_START_ACCURACY",
    "PredictedXiAuditResult",
    "build_audit_detail",
    "format_audit_markdown",
    "format_summary_table",
    "run_predicted_xi_audit",
    "summarise_audit",
]
