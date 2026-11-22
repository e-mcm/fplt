"""Read/write the canonical ``benchmark_projections`` archive (N1).

On disk: ``$FPLT_DATA_DIR/benchmark_projections/as_of=<season>_gwNN/part.parquet``.
Schema: ``fplt.schemas.benchmark_projections.BenchmarkProjectionSchema``.
``available_at`` is always strictly before the deadline. Manual FPL Review
CSV/JSON drops use ``fplt.data.fpl_review`` (``source=fpl_review``); the
``fpl_review_projections`` stub stays for the automated archive path.
"""

from __future__ import annotations

import logging
from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

import fplt.models.baselines  # noqa: F401  # pyright: ignore[reportUnusedImport]
import fplt.models.external.openfpl  # noqa: F401  # pyright: ignore[reportUnusedImport]
from fplt.backtest.walk_forward import horizon_for
from fplt.data.pit import PITData, PITView
from fplt.data.store import DataStore
from fplt.models.spec import get_model, load_model_specs
from fplt.schemas.benchmark_projections import (
    BENCHMARK_COLUMNS,
    QUALITY_OPENFPL_DEGRADED,
    SOURCE_EP_NEXT,
    SOURCE_FPL_REVIEW,
    SOURCE_OPENFPL,
    BenchmarkProjectionSchema,
)
from fplt.types import GWKey

logger = logging.getLogger(__name__)

EP_NEXT_SPEC = "ep_next_v1"
OPENFPL_SPEC = "openfpl_v1"
_REVIEW_TERMS_NOTE = (
    "Automated FPL Review fetch is not implemented (no scrape). "
    "Use fplt ingest-review / fplt.data.fpl_review to merge a manual "
    "CSV/JSON drop into this same benchmark_projections table. "
    "See docs/fpl_review.md and docs/benchmark-archives.md."
)


class BenchmarkArchiveError(RuntimeError):
    """A required benchmark source could not be produced for the deadline."""


def empty_benchmarks() -> pl.DataFrame:
    """An empty frame that still matches ``BenchmarkProjectionSchema``."""
    return pl.DataFrame(schema=BENCHMARK_COLUMNS)


def benchmarks_path(data_dir: Path, as_of: GWKey) -> Path:
    """Parquet path for one deadline's ``benchmark_projections`` archive."""
    stamp = f"as_of={as_of.season}_gw{as_of.gw:02d}"
    return data_dir / "benchmark_projections" / stamp / "part.parquet"


def write_benchmark_archive(data_dir: Path, as_of: GWKey, frame: pl.DataFrame) -> Path:
    """Validate ``frame`` and write it once for ``as_of``.

    Args:
        data_dir: Root that receives ``benchmark_projections/``.
        as_of: Deadline the archive belongs to.
        frame: A ``BenchmarkProjectionSchema`` frame.

    Returns:
        The parquet path that was written.

    Raises:
        FileExistsError: That deadline was already archived.
    """
    checked = BenchmarkProjectionSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    path = benchmarks_path(data_dir, as_of)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    checked.write_parquet(path)
    return path


def merge_benchmark_source(data_dir: Path, as_of: GWKey, frame: pl.DataFrame) -> Path:
    """Replace one ``source``'s rows in the deadline archive (or create it).

    Used by the manual FPL Review drop path so ``ep_next`` / ``openfpl`` rows
    written earlier are kept. All rows in ``frame`` must share one ``source``
    and match ``as_of``.

    Args:
        data_dir: Root that contains ``benchmark_projections/``.
        as_of: Deadline the archive belongs to.
        frame: A ``BenchmarkProjectionSchema`` frame for a single source.

    Returns:
        The parquet path written.

    Raises:
        ValueError: ``frame`` is empty, mixes sources, or mismatches ``as_of``.
    """
    checked = BenchmarkProjectionSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    checked = _ensure_utc(checked)
    if checked.is_empty():
        raise ValueError("cannot merge an empty benchmark frame")
    sources = checked["source"].unique().to_list()
    if len(sources) != 1:
        raise ValueError(f"merge expects a single source, got {sources!r}")
    seasons = checked["as_of_season"].unique().to_list()
    gws = checked["as_of_gw"].unique().to_list()
    if seasons != [as_of.season] or gws != [as_of.gw]:
        raise ValueError(
            f"frame as_of {seasons!r} gw{gws!r} does not match archive {as_of}"
        )
    source = str(sources[0])
    existing = read_benchmark_archive(data_dir, as_of)
    if existing.is_empty():
        merged = checked
    else:
        kept = existing.filter(pl.col("source") != source)
        merged = pl.concat([kept, checked], how="vertical_relaxed")
    merged = BenchmarkProjectionSchema.validate(_ensure_utc(merged))
    assert isinstance(merged, pl.DataFrame)
    path = benchmarks_path(data_dir, as_of)
    path.parent.mkdir(parents=True, exist_ok=True)
    merged.write_parquet(path)
    return path


def read_benchmark_archive(data_dir: Path, as_of: GWKey | None = None) -> pl.DataFrame:
    """Read one deadline archive, or every ``benchmark_projections`` parquet.

    Args:
        data_dir: Root that contains ``benchmark_projections/``.
        as_of: When set, read only that deadline's parquet.

    Returns:
        A ``BenchmarkProjectionSchema`` frame (possibly empty).
    """
    if as_of is not None:
        path = benchmarks_path(data_dir, as_of)
        if not path.exists():
            return empty_benchmarks()
        return _ensure_utc(pl.read_parquet(path))
    root = data_dir / "benchmark_projections"
    if not root.exists():
        return empty_benchmarks()
    paths = sorted(root.glob("as_of=*/part.parquet"))
    if not paths:
        return empty_benchmarks()
    return _ensure_utc(pl.concat([pl.read_parquet(path) for path in paths]))


def visible_benchmarks(data_dir: Path, cutoff: datetime) -> pl.DataFrame:
    """Benchmark rows with ``available_at`` at or before ``cutoff``.

    Args:
        data_dir: Root that contains ``benchmark_projections/``.
        cutoff: Inclusive UTC visibility time (PIT).

    Returns:
        Rows that would be visible at ``cutoff``.
    """
    frame = read_benchmark_archive(data_dir)
    if frame.is_empty():
        return frame
    moment = _aware(cutoff)
    return frame.filter(pl.col("available_at") <= moment)


def openfpl_deep_ppda_degraded(view: PITView) -> bool:
    """Whether OpenFPL Deep/PPDA inputs are missing at this PIT cutoff.

    Args:
        view: Point-in-time view used for the OpenFPL run.

    Returns:
        True when ``understat_team_matches`` has no visible rows.
    """
    return view.table("understat_team_matches").collect().is_empty()


def fpl_review_projections(
    view: PITView, horizon: list[GWKey], available_at: datetime
) -> pl.DataFrame:
    """Automated Review fetch stub — always raises (no scrape).

    Manual CSV/JSON drops go through ``fplt.data.fpl_review`` /
    ``fplt ingest-review`` into the same ``benchmark_projections`` archive.

    Args:
        view: Unused PIT view (kept for API symmetry with other sources).
        horizon: Unused target gameweeks.
        available_at: Unused archive stamp.

    Returns:
        Never returns.

    Raises:
        NotImplementedError: Always — use the manual ingest path instead.
    """
    del view, horizon, available_at
    raise NotImplementedError(_REVIEW_TERMS_NOTE)


def archive_stamp(deadline: datetime, now: datetime) -> datetime:
    """Choose ``available_at`` strictly before ``deadline``.

    Live saves use ``now`` when it is before the deadline. Late backfills
    stamp one second before the deadline so PIT at the deadline still sees
    the archive without claiming a post-deadline save.

    Args:
        deadline: Gameweek deadline (UTC).
        now: Clock reading (UTC).

    Returns:
        A timestamp strictly earlier than ``deadline``.

    Raises:
        ValueError: ``deadline`` cannot yield a pre-deadline stamp.
    """
    cut = _aware(deadline)
    moment = _aware(now)
    if moment < cut:
        return moment
    stamp = cut - timedelta(seconds=1)
    if stamp >= cut:
        raise ValueError(f"cannot place available_at before deadline {cut}")
    return stamp


def archive_deadline_benchmarks(
    data_dir: Path,
    as_of: GWKey,
    *,
    now: datetime | None = None,
    include_openfpl: bool = True,
    include_review: bool = False,
) -> Path:
    """Build and save ``ep_next`` (+ OpenFPL) for one deadline.

    Raises when a required source cannot be produced. OpenFPL rows that
    run without ``understat_team_matches`` are kept and labelled
    ``openfpl_degraded``. Does not scrape FPL Review.

    Args:
        data_dir: Root with ``staged/``; receives ``benchmark_projections/``.
        as_of: Deadline to archive.
        now: Clock reading. Defaults to current UTC.
        include_openfpl: Require OpenFPL rows (fail if they cannot be built).
        include_review: When true, call the Review stub (raises).

    Returns:
        The parquet path written under ``benchmark_projections/``.

    Raises:
        FileExistsError: Archive for ``as_of`` already exists.
        BenchmarkArchiveError: A required source is missing.
        NotImplementedError: ``include_review`` is true.
        KeyError: Required model specs are missing.
    """
    store = DataStore(data_dir)
    deadline = _aware(store.deadline(as_of))
    moment = _aware(now if now is not None else datetime.now(UTC))
    available_at = archive_stamp(deadline, moment)
    view = PITData(store).view(as_of)
    horizon = horizon_for(as_of, steps=1)
    frames = [_ep_next_rows(view, horizon, available_at)]
    if include_openfpl:
        frames.append(_openfpl_rows(view, horizon, available_at))
    if include_review:
        frames.append(fpl_review_projections(view, horizon, available_at))
    non_empty = [part for part in frames if not part.is_empty()]
    frame = pl.concat(non_empty, how="vertical") if non_empty else empty_benchmarks()
    _require_sources(frame, include_openfpl=include_openfpl)
    return write_benchmark_archive(data_dir, as_of, frame)


def archive_if_missing(
    data_dir: Path,
    as_of: GWKey,
    *,
    now: datetime | None = None,
    include_openfpl: bool = True,
) -> Path | None:
    """Archive ``as_of`` when no parquet exists yet.

    Args:
        data_dir: Data root.
        as_of: Deadline to archive.
        now: Clock for ``available_at``.
        include_openfpl: Forwarded to ``archive_deadline_benchmarks``.

    Returns:
        The path written, or ``None`` when the archive already existed.
    """
    path = benchmarks_path(data_dir, as_of)
    if path.exists():
        logger.info("benchmark archive already exists for %s", as_of)
        return None
    return archive_deadline_benchmarks(
        data_dir, as_of, now=now, include_openfpl=include_openfpl
    )


def _require_sources(frame: pl.DataFrame, *, include_openfpl: bool) -> None:
    sources: set[str] = (
        {str(value) for value in frame["source"].to_list()}
        if not frame.is_empty()
        else set()
    )
    missing: list[str] = []
    if SOURCE_EP_NEXT not in sources:
        missing.append(SOURCE_EP_NEXT)
    if include_openfpl and SOURCE_OPENFPL not in sources:
        missing.append(SOURCE_OPENFPL)
    if missing:
        raise BenchmarkArchiveError(
            "benchmark archive missing required source(s): " + ", ".join(missing)
        )


def _ep_next_rows(
    view: PITView, horizon: list[GWKey], available_at: datetime
) -> pl.DataFrame:
    specs = load_model_specs()
    try:
        spec = specs[EP_NEXT_SPEC]
    except KeyError as exc:
        raise BenchmarkArchiveError(f"missing model spec {EP_NEXT_SPEC}") from exc
    try:
        model = get_model(spec.kind)(spec)
        model.fit(view)
        predicted = model.predict(view, horizon)
    except Exception as exc:
        raise BenchmarkArchiveError(f"ep_next failed: {exc}") from exc
    return _from_predictions(predicted, SOURCE_EP_NEXT, available_at, quality_flag=None)


def _openfpl_rows(
    view: PITView, horizon: list[GWKey], available_at: datetime
) -> pl.DataFrame:
    specs = load_model_specs()
    if OPENFPL_SPEC not in specs:
        raise BenchmarkArchiveError(f"missing model spec {OPENFPL_SPEC}")
    spec = specs[OPENFPL_SPEC]
    degraded = openfpl_deep_ppda_degraded(view)
    quality = QUALITY_OPENFPL_DEGRADED if degraded else None
    if degraded:
        logger.warning(
            "OpenFPL labelled %s at %s (no understat_team_matches)",
            QUALITY_OPENFPL_DEGRADED,
            view.as_of,
        )
    try:
        model = get_model(spec.kind)(spec)
        model.fit(view)
        predicted = model.predict(view, horizon)
    except Exception as exc:
        raise BenchmarkArchiveError(f"openfpl failed: {exc}") from exc
    return _from_predictions(
        predicted, SOURCE_OPENFPL, available_at, quality_flag=quality
    )


def _from_predictions(
    predicted: pl.DataFrame,
    source: str,
    available_at: datetime,
    *,
    quality_flag: str | None,
) -> pl.DataFrame:
    if predicted.is_empty():
        return empty_benchmarks()
    frame = predicted.select(
        pl.lit(source).alias("source"),
        "as_of_season",
        "as_of_gw",
        "target_season",
        "target_gw",
        "player_key",
        "fixture_id",
        pl.col("xpts").cast(pl.Float64),
        pl.lit(available_at).alias("available_at"),
        pl.lit(quality_flag).cast(pl.Utf8).alias("quality_flag"),
    ).select(list(BENCHMARK_COLUMNS))
    checked = BenchmarkProjectionSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked


def _ensure_utc(frame: pl.DataFrame) -> pl.DataFrame:
    if "available_at" not in frame.columns:
        return frame
    dtype = frame.schema["available_at"]
    if isinstance(dtype, pl.Datetime) and dtype.time_zone is None:
        return frame.with_columns(pl.col("available_at").dt.replace_time_zone("UTC"))
    return frame


def _aware(value: datetime) -> datetime:
    if value.tzinfo is None:
        return value.replace(tzinfo=UTC)
    return value.astimezone(UTC)


__all__ = [
    "EP_NEXT_SPEC",
    "OPENFPL_SPEC",
    "SOURCE_EP_NEXT",
    "SOURCE_FPL_REVIEW",
    "SOURCE_OPENFPL",
    "BenchmarkArchiveError",
    "archive_deadline_benchmarks",
    "archive_if_missing",
    "archive_stamp",
    "benchmarks_path",
    "empty_benchmarks",
    "fpl_review_projections",
    "merge_benchmark_source",
    "openfpl_deep_ppda_degraded",
    "read_benchmark_archive",
    "visible_benchmarks",
    "write_benchmark_archive",
]
