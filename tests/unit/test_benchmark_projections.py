"""Write/read and PIT visibility for benchmark_projections archives."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from fplt.data.benchmark_projections import (
    BenchmarkArchiveError,
    archive_deadline_benchmarks,
    archive_stamp,
    benchmarks_path,
    empty_benchmarks,
    fpl_review_projections,
    openfpl_deep_ppda_degraded,
    read_benchmark_archive,
    visible_benchmarks,
    write_benchmark_archive,
)
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.schemas.benchmark_projections import (
    QUALITY_OPENFPL_DEGRADED,
    SOURCE_EP_NEXT,
    SOURCE_FPL_REVIEW,
    SOURCE_OPENFPL,
    BenchmarkProjectionSchema,
)
from fplt.types import GWKey
from tests.builders import build_league

_AS_OF = GWKey("2024-25", 2)
_SAVE = datetime(2024, 8, 23, 16, tzinfo=UTC)
_DEADLINE = datetime(2024, 8, 23, 17, tzinfo=UTC)


def _rows(
    *,
    source: str = SOURCE_EP_NEXT,
    available_at: datetime = _SAVE,
    quality_flag: str | None = None,
) -> pl.DataFrame:
    frame = pl.DataFrame(
        {
            "source": [source, source],
            "as_of_season": ["2024-25", "2024-25"],
            "as_of_gw": [2, 2],
            "target_season": ["2024-25", "2024-25"],
            "target_gw": [2, 2],
            "player_key": [100, 200],
            "fixture_id": [2, 2],
            "xpts": [6.5, 4.0],
            "available_at": [available_at, available_at],
            "quality_flag": [quality_flag, quality_flag],
        }
    )
    checked = BenchmarkProjectionSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked


def test_schema_accepts_fpl_review_source() -> None:
    frame = _rows(source=SOURCE_FPL_REVIEW)
    assert frame["source"].to_list() == [SOURCE_FPL_REVIEW, SOURCE_FPL_REVIEW]


def test_write_read_refuses_overwrite(tmp_path: Path) -> None:
    path = write_benchmark_archive(tmp_path, _AS_OF, _rows())
    assert path == benchmarks_path(tmp_path, _AS_OF)
    got = read_benchmark_archive(tmp_path, _AS_OF)
    assert got["player_key"].to_list() == [100, 200]
    assert got["source"].to_list() == [SOURCE_EP_NEXT, SOURCE_EP_NEXT]
    with pytest.raises(FileExistsError):
        write_benchmark_archive(tmp_path, _AS_OF, _rows())


def test_pit_hides_rows_before_available_at(tmp_path: Path) -> None:
    write_benchmark_archive(tmp_path, _AS_OF, _rows(available_at=_SAVE))
    before = visible_benchmarks(tmp_path, _SAVE - timedelta(seconds=1))
    assert before.is_empty()
    at_save = visible_benchmarks(tmp_path, _SAVE)
    assert at_save.height == 2
    after = visible_benchmarks(tmp_path, _DEADLINE)
    assert after.height == 2


def test_archive_stamp_is_strictly_before_deadline() -> None:
    live = archive_stamp(_DEADLINE, _SAVE)
    assert live == _SAVE
    assert live < _DEADLINE
    late = archive_stamp(_DEADLINE, _DEADLINE + timedelta(hours=1))
    assert late < _DEADLINE


def test_fpl_review_stub_raises(tmp_path: Path) -> None:
    view = PITData(build_league(DataStore(tmp_path))).view(_AS_OF)
    with pytest.raises(NotImplementedError, match="terms-of-use"):
        fpl_review_projections(view, [_AS_OF], _SAVE)


def test_openfpl_degraded_without_team_table(tmp_path: Path) -> None:
    view = PITData(build_league(DataStore(tmp_path))).view(_AS_OF)
    assert openfpl_deep_ppda_degraded(view) is True


def test_archive_deadline_labels_openfpl_degraded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_league(DataStore(tmp_path))

    def fake_openfpl(view, horizon, available_at):  # noqa: ANN001
        del view, horizon
        return _rows(
            source=SOURCE_OPENFPL,
            available_at=available_at,
            quality_flag=QUALITY_OPENFPL_DEGRADED,
        )

    monkeypatch.setattr("fplt.data.benchmark_projections._openfpl_rows", fake_openfpl)
    path = archive_deadline_benchmarks(
        tmp_path, _AS_OF, now=_SAVE, include_openfpl=True
    )
    frame = read_benchmark_archive(tmp_path, _AS_OF)
    assert path.exists()
    sources = sorted(frame["source"].unique().to_list())
    assert sources == [SOURCE_EP_NEXT, SOURCE_OPENFPL]
    openfpl = frame.filter(pl.col("source") == SOURCE_OPENFPL)
    assert openfpl["quality_flag"].to_list() == [
        QUALITY_OPENFPL_DEGRADED,
        QUALITY_OPENFPL_DEGRADED,
    ]
    assert (frame["available_at"] < _DEADLINE).all()


def test_archive_fails_loudly_when_openfpl_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    build_league(DataStore(tmp_path))
    monkeypatch.setattr(
        "fplt.data.benchmark_projections._openfpl_rows",
        lambda *a, **k: empty_benchmarks(),
    )
    with pytest.raises(BenchmarkArchiveError, match="openfpl"):
        archive_deadline_benchmarks(tmp_path, _AS_OF, now=_SAVE, include_openfpl=True)
