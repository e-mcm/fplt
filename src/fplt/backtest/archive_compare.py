"""Paired prospective compare against ``benchmark_projections`` archives (N2).

Joins our ledger forecasts to archived arms (``ep_next``, ``openfpl``, later
``fpl_review``) on identical player–fixtures, keeps only rows where every
compared arm has a pre-deadline snapshot, and scores the default north-star
board: RMSE (headline), MAE (secondary), Spearman by position, plus OpenFPL
outcome groups (zeros / blanks / tickers / haulers). Paired significance uses
the existing Diebold–Mariano / HAC + stationary-bootstrap stack in
``fplt.backtest.intervals``.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from fplt.backtest.intervals import (
    DEFAULT_BOOTSTRAP_REPS,
    DEFAULT_BOOTSTRAP_SEED,
    PairedInterval,
    interval_to_dict,
    intervals_for_pair,
)
from fplt.backtest.ledger import align_ledger_columns, read_predictions
from fplt.backtest.metrics import overall_summary, summarise
from fplt.data.benchmark_projections import read_benchmark_archive
from fplt.data.store import DataStore
from fplt.models.spec import load_model_specs
from fplt.schemas.benchmark_projections import (
    SOURCE_EP_NEXT,
    SOURCE_FPL_REVIEW,
    SOURCE_OPENFPL,
)
from fplt.schemas.ledger import LEDGER_COLUMNS, LedgerSchema
from fplt.types import GWKey

logger = logging.getLogger(__name__)

HEADLINE_METRIC = "rmse"
DEFAULT_SOURCES: tuple[str, ...] = (SOURCE_EP_NEXT, SOURCE_OPENFPL)
ALLOWED_SOURCES: frozenset[str] = frozenset(
    {SOURCE_EP_NEXT, SOURCE_OPENFPL, SOURCE_FPL_REVIEW}
)
ARCHIVE_MODEL_PREFIX = "archive:"

OUTCOME_ZEROS = "zeros"
OUTCOME_BLANKS = "blanks"
OUTCOME_TICKERS = "tickers"
OUTCOME_HAULERS = "haulers"
OUTCOME_GROUPS: tuple[str, ...] = (
    OUTCOME_ZEROS,
    OUTCOME_BLANKS,
    OUTCOME_TICKERS,
    OUTCOME_HAULERS,
)

_PAIR_KEYS = (
    "as_of_season",
    "as_of_gw",
    "target_season",
    "target_gw",
    "player_key",
    "fixture_id",
)


@dataclass(frozen=True)
class CoverageGateResult:
    """How many player–fixtures survived the all-arms pre-deadline gate."""

    n_ours: int
    n_benchmark_by_source: dict[str, int]
    n_covered: int
    n_scored: int
    sources: tuple[str, ...]


@dataclass(frozen=True)
class ArchiveCompareResult:
    """Artifacts from ``compare_to_archives``."""

    season: str
    from_gw: int
    to_gw: int
    ours_model_id: str
    ours_label: str
    sources: tuple[str, ...]
    coverage: CoverageGateResult
    paired_ledger: pl.DataFrame
    summary: pl.DataFrame
    by_position: pl.DataFrame
    outcome_groups: pl.DataFrame
    intervals: tuple[PairedInterval, ...]
    output_dir: Path
    openfpl_degraded_rows: int = 0


def archive_model_id(source: str) -> str:
    """Ledger ``model_id`` used for an archived benchmark ``source``."""
    return f"{ARCHIVE_MODEL_PREFIX}{source}"


def outcome_group(actual_pts: int, actual_minutes: int | None) -> str:
    """Map realised points (and minutes) to an OpenFPL-style outcome group.

    Args:
        actual_pts: Realised FPL points for the player–fixture.
        actual_minutes: Minutes played when known; ``None`` falls back to points.

    Returns:
        One of ``zeros``, ``blanks``, ``tickers``, ``haulers``.
    """
    if actual_minutes is not None and actual_minutes <= 0:
        return OUTCOME_ZEROS
    if actual_pts <= 0 and (actual_minutes is None or actual_minutes <= 0):
        return OUTCOME_ZEROS
    if actual_pts <= 2:
        return OUTCOME_BLANKS
    if actual_pts <= 4:
        return OUTCOME_TICKERS
    return OUTCOME_HAULERS


def resolve_ours_model_id(model: str) -> str:
    """Resolve a ModelSpec name or raw ``model_id`` hash to a ledger id.

    Args:
        model: YAML spec name (e.g. ``dc_layered_v2``) or a 16-char ``model_id``.

    Returns:
        The ``model_id`` string stored on ledger rows.

    Raises:
        KeyError: Spec name is unknown and ``model`` is not treated as an id.
    """
    specs = load_model_specs()
    if model in specs:
        return specs[model].model_id
    return model


def deadlines_frame(
    store: DataStore, season: str, from_gw: int, to_gw: int
) -> pl.DataFrame:
    """Deadline clock for each as-of gameweek in ``season``.

    Args:
        store: Staged data store.
        season: Season label.
        from_gw: First as-of gameweek inclusive.
        to_gw: Last as-of gameweek inclusive.

    Returns:
        Columns ``as_of_season``, ``as_of_gw``, ``deadline_time`` (UTC).
    """
    rows: list[dict[str, object]] = []
    for gw in range(from_gw, to_gw + 1):
        key = GWKey(season, gw)
        try:
            cut = store.deadline(key)
        except KeyError:
            continue
        rows.append(
            {
                "as_of_season": season,
                "as_of_gw": gw,
                "deadline_time": _aware(cut),
            }
        )
    if not rows:
        return pl.DataFrame(
            schema={
                "as_of_season": pl.Utf8(),
                "as_of_gw": pl.Int64(),
                "deadline_time": pl.Datetime(time_zone="UTC"),
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("deadline_time").cast(pl.Datetime(time_zone="UTC"))
    )


def covered_pair_keys(
    ours: pl.DataFrame,
    benchmarks: pl.DataFrame,
    deadlines: pl.DataFrame,
    sources: Sequence[str],
) -> pl.DataFrame:
    """Player–fixtures present on every arm with a pre-deadline snapshot.

    Ours is treated as covered when a ledger row exists for that as-of (walk-
    forward / live predict is already a deadline snapshot). Archive rows must
    have ``available_at`` strictly before the gameweek deadline.

    Args:
        ours: Ledger rows for the candidate model (any extra columns ok).
        benchmarks: ``BenchmarkProjectionSchema`` rows.
        deadlines: Output of ``deadlines_frame``.
        sources: Archive ``source`` values that must all be present.

    Returns:
        Distinct ``_PAIR_KEYS`` rows that clear the gate.

    Raises:
        ValueError: Empty ``sources`` or an unknown source name.
    """
    chosen = _validate_sources(sources)
    if ours.is_empty() or deadlines.is_empty():
        return _empty_keys()

    ours_keys = (
        ours.select(list(_PAIR_KEYS))
        .unique()
        .join(
            deadlines.select("as_of_season", "as_of_gw"),
            on=["as_of_season", "as_of_gw"],
        )
    )
    covered = ours_keys
    if benchmarks.is_empty():
        return _empty_keys()

    stamped = _pre_deadline_rows(benchmarks, deadlines)
    for source in chosen:
        part = (
            stamped.filter(pl.col("source") == source).select(list(_PAIR_KEYS)).unique()
        )
        covered = covered.join(part, on=list(_PAIR_KEYS), how="inner")
        if covered.is_empty():
            return _empty_keys()
    return covered.unique().sort(list(_PAIR_KEYS))


def benchmarks_to_ledger(
    benchmarks: pl.DataFrame,
    template: pl.DataFrame,
    *,
    sources: Sequence[str] | None = None,
) -> pl.DataFrame:
    """Project archive rows onto ``LedgerSchema`` using ``template`` metadata.

    Args:
        benchmarks: Archive rows (already coverage-filtered preferred).
        template: Our ledger rows supplying ``position``, actuals, ``horizon``,
            and ``run_kind`` for the same pair keys.
        sources: Optional source filter; default keeps every archive source.

    Returns:
        A ``LedgerSchema`` frame with ``model_id`` = ``archive:<source>``.
    """
    if benchmarks.is_empty() or template.is_empty():
        return pl.DataFrame(schema=LEDGER_COLUMNS)
    frame = benchmarks
    if sources is not None:
        frame = frame.filter(pl.col("source").is_in(list(sources)))
    if frame.is_empty():
        return pl.DataFrame(schema=LEDGER_COLUMNS)

    meta = template.select(
        *(_PAIR_KEYS),
        "run_kind",
        "horizon",
        "position",
        "actual_pts",
        "actual_minutes",
    ).unique(subset=list(_PAIR_KEYS))
    joined = frame.join(meta, on=list(_PAIR_KEYS), how="inner")
    if joined.is_empty():
        return pl.DataFrame(schema=LEDGER_COLUMNS)
    out = align_ledger_columns(
        joined.select(
            pl.concat_str(pl.lit(ARCHIVE_MODEL_PREFIX), pl.col("source")).alias(
                "model_id"
            ),
            "run_kind",
            "as_of_season",
            "as_of_gw",
            "target_season",
            "target_gw",
            "horizon",
            "player_key",
            "position",
            "fixture_id",
            pl.col("xpts").cast(pl.Float64),
            pl.lit(None).cast(pl.Float64).alias("xpts_sd"),
            pl.lit(None).cast(pl.Float64).alias("p_haul"),
            pl.lit(None).cast(pl.Float64).alias("p_60"),
            "actual_pts",
            "actual_minutes",
        )
    )
    checked = LedgerSchema.validate(out)
    assert isinstance(checked, pl.DataFrame)
    return checked


def outcome_group_metrics(ledger: pl.DataFrame) -> pl.DataFrame:
    """RMSE / MAE / n by model and OpenFPL outcome group.

    Args:
        ledger: Paired scored ledger (null actuals skipped).

    Returns:
        Columns ``model_id``, ``outcome_group``, ``n``, ``rmse``, ``mae``.
    """
    scored = ledger.filter(pl.col("actual_pts").is_not_null())
    if scored.is_empty():
        return pl.DataFrame(
            schema={
                "model_id": pl.Utf8(),
                "outcome_group": pl.Utf8(),
                "n": pl.Int64(),
                "rmse": pl.Float64(),
                "mae": pl.Float64(),
            }
        )
    labelled = scored.with_columns(_outcome_group_expr().alias("outcome_group"))
    error = (pl.col("xpts") - pl.col("actual_pts")).cast(pl.Float64)
    return (
        labelled.group_by("model_id", "outcome_group")
        .agg(
            pl.len().alias("n"),
            error.pow(2).mean().sqrt().alias("rmse"),
            error.abs().mean().alias("mae"),
        )
        .sort("model_id", "outcome_group")
    )


def compare_to_archives(
    data_dir: Path,
    *,
    season: str,
    model: str,
    from_gw: int = 1,
    to_gw: int = 38,
    sources: Sequence[str] = DEFAULT_SOURCES,
    horizon: int = 1,
    ledger: pl.DataFrame | None = None,
    benchmarks: pl.DataFrame | None = None,
    deadlines: pl.DataFrame | None = None,
    output_dir: Path | None = None,
    bootstrap_reps: int = DEFAULT_BOOTSTRAP_REPS,
    bootstrap_seed: int = DEFAULT_BOOTSTRAP_SEED,
) -> ArchiveCompareResult:
    """Join our forecasts to archived benchmarks and score the default board.

    Args:
        data_dir: Root with ``ledger/`` and ``benchmark_projections/``.
        season: Evaluation season.
        model: ModelSpec name or ledger ``model_id`` for the candidate arm.
        from_gw: First as-of gameweek inclusive.
        to_gw: Last as-of gameweek inclusive.
        sources: Archive sources to pair against (default ``ep_next``,
            ``openfpl``; ``fpl_review`` when archived).
        horizon: Ledger horizon to keep (default 1).
        ledger: Optional injected ledger (tests); else read from disk.
        benchmarks: Optional injected archives (tests).
        deadlines: Optional injected deadlines frame (tests).
        output_dir: Where to write JSON/parquet/markdown.
        bootstrap_reps: Stationary bootstrap replications for intervals.
        bootstrap_seed: Bootstrap seed.

    Returns:
        Board tables, coverage stats, and paired intervals (ours vs each source).

    Raises:
        ValueError: Bad window, empty sources, unknown source, or missing ours.
        KeyError: ModelSpec name unknown when resolving ``model``.
    """
    if not 1 <= from_gw <= to_gw <= 38:
        raise ValueError(f"gameweeks must be in 1..38, got {from_gw}..{to_gw}")
    if horizon < 1:
        raise ValueError(f"horizon must be positive, got {horizon}")
    chosen = _validate_sources(sources)
    ours_id = resolve_ours_model_id(model)
    ours_label = model if model != ours_id else ours_id

    store = DataStore(data_dir)
    full_ledger = read_predictions(data_dir) if ledger is None else ledger
    archive = read_benchmark_archive(data_dir) if benchmarks is None else benchmarks
    cuts = (
        deadlines_frame(store, season, from_gw, to_gw)
        if deadlines is None
        else deadlines
    )

    ours = full_ledger.filter(
        (pl.col("model_id") == ours_id)
        & (pl.col("target_season") == season)
        & (pl.col("as_of_season") == season)
        & (pl.col("as_of_gw") >= from_gw)
        & (pl.col("as_of_gw") <= to_gw)
        & (pl.col("horizon") == horizon)
    )
    if ours.is_empty():
        raise ValueError(
            f"no ledger rows for model_id={ours_id!r} in {season} "
            f"gw {from_gw}-{to_gw} horizon={horizon}"
        )

    archive = archive.filter(
        (pl.col("as_of_season") == season)
        & (pl.col("target_season") == season)
        & (pl.col("as_of_gw") >= from_gw)
        & (pl.col("as_of_gw") <= to_gw)
        & pl.col("source").is_in(list(chosen))
    )

    keys = covered_pair_keys(ours, archive, cuts, chosen)
    n_by_source = {
        source: int(
            _pre_deadline_rows(archive.filter(pl.col("source") == source), cuts)
            .select(list(_PAIR_KEYS))
            .unique()
            .height
        )
        for source in chosen
    }
    ours_covered = align_ledger_columns(
        ours.join(keys, on=list(_PAIR_KEYS), how="inner")
    )
    archive_covered = archive.join(keys, on=list(_PAIR_KEYS), how="inner")
    bench_ledger = benchmarks_to_ledger(archive_covered, ours_covered, sources=chosen)
    paired = pl.concat([ours_covered.select(list(LEDGER_COLUMNS)), bench_ledger])
    scored_n = int(paired.filter(pl.col("actual_pts").is_not_null()).height)
    coverage = CoverageGateResult(
        n_ours=int(ours.select(list(_PAIR_KEYS)).unique().height),
        n_benchmark_by_source=n_by_source,
        n_covered=int(keys.height),
        n_scored=scored_n,
        sources=chosen,
    )

    degraded = 0
    if "quality_flag" in archive_covered.columns and not archive_covered.is_empty():
        degraded = int(
            archive_covered.filter(pl.col("quality_flag") == "openfpl_degraded").height
        )

    summary = overall_summary(paired)
    by_position = summarise(paired)
    groups = outcome_group_metrics(paired)

    positions = tuple(
        sorted(
            {
                str(value)
                for value in paired["position"].unique().to_list()
                if value is not None
            }
        )
    )
    intervals: list[PairedInterval] = []
    for source in chosen:
        intervals.extend(
            intervals_for_pair(
                paired,
                control_id=archive_model_id(source),
                treatment_id=ours_id,
                metrics=("rmse", "mae", "spearman"),
                horizon=horizon,
                positions=positions or None,
                bootstrap_reps=bootstrap_reps,
                bootstrap_seed=bootstrap_seed,
            )
        )

    out = output_dir or _default_output_dir(data_dir, season, from_gw, to_gw, ours_id)
    out.mkdir(parents=True, exist_ok=True)
    result = ArchiveCompareResult(
        season=season,
        from_gw=from_gw,
        to_gw=to_gw,
        ours_model_id=ours_id,
        ours_label=ours_label,
        sources=chosen,
        coverage=coverage,
        paired_ledger=paired,
        summary=summary,
        by_position=by_position,
        outcome_groups=groups,
        intervals=tuple(intervals),
        output_dir=out,
        openfpl_degraded_rows=degraded,
    )
    _write_outputs(result)
    logger.info(
        "archive-compare %s gw%s-%s covered=%s scored_rows=%s out=%s",
        season,
        from_gw,
        to_gw,
        coverage.n_covered,
        coverage.n_scored,
        out,
    )
    return result


def format_archive_board(result: ArchiveCompareResult) -> str:
    """Render a short text board (RMSE first) for the CLI."""
    lines = [
        f"archive-compare {result.season} gw{result.from_gw}-{result.to_gw}",
        f"ours={result.ours_label} ({result.ours_model_id})",
        f"sources={','.join(result.sources)}",
        (
            f"coverage: ours_keys={result.coverage.n_ours} "
            f"covered={result.coverage.n_covered} "
            f"scored_rows={result.coverage.n_scored} "
            f"bench={result.coverage.n_benchmark_by_source}"
        ),
        f"headline_metric={HEADLINE_METRIC}",
    ]
    if result.openfpl_degraded_rows:
        lines.append(
            f"openfpl_degraded_rows={result.openfpl_degraded_rows} "
            "(Deep/PPDA missing at archive time)"
        )
    if result.summary.is_empty():
        lines.append("(no scored rows)")
    else:
        cols = [
            c
            for c in ("model_id", "n", "rmse", "mae", "spearman", "top10_precision")
            if c in result.summary.columns
        ]
        board = result.summary.select(cols).sort("rmse")
        lines.append(str(board))
    if not result.by_position.is_empty() and "spearman" in result.by_position.columns:
        spear = result.by_position.select(
            "model_id", "position", "spearman", "rmse", "mae"
        ).sort("position", "model_id")
        lines.append("spearman by position:")
        lines.append(str(spear))
    if not result.outcome_groups.is_empty():
        lines.append("outcome groups (zeros/blanks/tickers/haulers):")
        lines.append(str(result.outcome_groups.sort("outcome_group", "model_id")))
    return "\n".join(lines)


def _write_outputs(result: ArchiveCompareResult) -> None:
    out = result.output_dir
    summary_path = out / "summary.parquet"
    detail_path = out / "by_position.parquet"
    groups_path = out / "outcome_groups.parquet"
    paired_path = out / "paired_ledger.parquet"
    json_path = out / "summary.json"
    md_path = out / "board.md"
    result.summary.write_parquet(summary_path)
    result.by_position.write_parquet(detail_path)
    result.outcome_groups.write_parquet(groups_path)
    result.paired_ledger.write_parquet(paired_path)
    payload: dict[str, Any] = {
        "headline_metric": HEADLINE_METRIC,
        "season": result.season,
        "from_gw": result.from_gw,
        "to_gw": result.to_gw,
        "ours_model_id": result.ours_model_id,
        "ours_label": result.ours_label,
        "sources": list(result.sources),
        "coverage": {
            "n_ours": result.coverage.n_ours,
            "n_benchmark_by_source": result.coverage.n_benchmark_by_source,
            "n_covered": result.coverage.n_covered,
            "n_scored": result.coverage.n_scored,
            "sources": list(result.coverage.sources),
        },
        "openfpl_degraded_rows": result.openfpl_degraded_rows,
        "summary": result.summary.to_dicts(),
        "by_position": result.by_position.to_dicts(),
        "outcome_groups": result.outcome_groups.to_dicts(),
        "intervals": [interval_to_dict(item) for item in result.intervals],
        "files": {
            "summary_parquet": str(summary_path),
            "by_position_parquet": str(detail_path),
            "outcome_groups_parquet": str(groups_path),
            "paired_ledger_parquet": str(paired_path),
            "summary_json": str(json_path),
            "board_md": str(md_path),
        },
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    md_path.write_text(format_archive_board(result) + "\n", encoding="utf-8")


def _outcome_group_expr() -> pl.Expr:
    """Polars expression matching ``outcome_group`` for frame labelling."""
    zero_minutes = pl.col("actual_minutes").is_not_null() & (
        pl.col("actual_minutes") <= 0
    )
    unknown_zero = pl.col("actual_minutes").is_null() & (pl.col("actual_pts") <= 0)
    return (
        pl.when(zero_minutes | unknown_zero)
        .then(pl.lit(OUTCOME_ZEROS))
        .when(pl.col("actual_pts") <= 2)
        .then(pl.lit(OUTCOME_BLANKS))
        .when(pl.col("actual_pts") <= 4)
        .then(pl.lit(OUTCOME_TICKERS))
        .otherwise(pl.lit(OUTCOME_HAULERS))
    )


def _validate_sources(sources: Sequence[str]) -> tuple[str, ...]:
    if not sources:
        raise ValueError("sources must be non-empty")
    seen: list[str] = []
    for source in sources:
        if source not in ALLOWED_SOURCES:
            raise ValueError(
                f"unknown archive source {source!r}; "
                f"expected one of {sorted(ALLOWED_SOURCES)}"
            )
        if source not in seen:
            seen.append(source)
    return tuple(seen)


def _empty_keys() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "as_of_season": pl.Utf8(),
            "as_of_gw": pl.Int64(),
            "target_season": pl.Utf8(),
            "target_gw": pl.Int64(),
            "player_key": pl.Int64(),
            "fixture_id": pl.Int64(),
        }
    )


def _default_output_dir(
    data_dir: Path, season: str, from_gw: int, to_gw: int, ours_id: str
) -> Path:
    stamp = f"{season}_gw{from_gw:02d}-{to_gw:02d}_{ours_id}"
    return data_dir / "archive_compare" / stamp


def _pre_deadline_rows(
    benchmarks: pl.DataFrame, deadlines: pl.DataFrame
) -> pl.DataFrame:
    """Archive rows whose ``available_at`` is strictly before the deadline."""
    if benchmarks.is_empty() or deadlines.is_empty():
        return benchmarks.head(0)
    stamped = benchmarks.join(deadlines, on=["as_of_season", "as_of_gw"], how="inner")
    stamped = _ensure_utc(stamped, "available_at")
    stamped = _ensure_utc(stamped, "deadline_time")
    return stamped.filter(pl.col("available_at") < pl.col("deadline_time"))


def _ensure_utc(frame: pl.DataFrame, column: str) -> pl.DataFrame:
    """Cast a datetime column to UTC-aware (naive values treated as UTC)."""
    if column not in frame.columns or frame.is_empty():
        return frame
    dtype = frame.schema[column]
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
        return frame.with_columns(pl.col(column).dt.replace_time_zone("UTC"))
    if isinstance(dtype, pl.Datetime) and dtype.time_zone != "UTC":
        return frame.with_columns(pl.col(column).dt.convert_time_zone("UTC"))
    return frame


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = [
    "ALLOWED_SOURCES",
    "ARCHIVE_MODEL_PREFIX",
    "DEFAULT_SOURCES",
    "HEADLINE_METRIC",
    "OUTCOME_BLANKS",
    "OUTCOME_GROUPS",
    "OUTCOME_HAULERS",
    "OUTCOME_TICKERS",
    "OUTCOME_ZEROS",
    "ArchiveCompareResult",
    "CoverageGateResult",
    "archive_model_id",
    "benchmarks_to_ledger",
    "compare_to_archives",
    "covered_pair_keys",
    "deadlines_frame",
    "format_archive_board",
    "outcome_group",
    "outcome_group_metrics",
    "resolve_ours_model_id",
]
