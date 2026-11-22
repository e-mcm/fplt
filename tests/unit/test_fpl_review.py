"""Manual FPL Review drop loader merges into benchmark_projections."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fplt.cli import main
from fplt.data.benchmark_projections import (
    read_benchmark_archive,
    write_benchmark_archive,
)
from fplt.data.fpl_review import (
    copy_drop_to_raw,
    fpl_review_raw_root,
    ingest_fpl_review_drop,
    load_fpl_review_drop,
    parse_available_at,
)
from fplt.schemas.benchmark_projections import (
    SOURCE_EP_NEXT,
    SOURCE_FPL_REVIEW,
    BenchmarkProjectionSchema,
)
from fplt.settings import get_settings
from fplt.types import GWKey

_FIXTURES = Path(__file__).resolve().parents[1] / "fixtures" / "fpl_review"
_CSV = _FIXTURES / "projections.csv"
_JSON = _FIXTURES / "projections.json"
_STAMP = datetime(2025, 8, 15, 10, 0, tzinfo=UTC)
_AS_OF = GWKey("2025-26", 1)


def test_parse_available_at_iso_and_naive() -> None:
    assert parse_available_at("2025-08-15T10:00:00Z") == _STAMP
    assert parse_available_at("2025-08-15T10:00:00") == _STAMP
    assert parse_available_at(_STAMP) == _STAMP
    with pytest.raises(ValueError, match="empty"):
        parse_available_at("  ")


def test_load_csv_fixture_forces_fpl_review_source() -> None:
    frame = load_fpl_review_drop(_CSV)
    BenchmarkProjectionSchema.validate(frame)
    assert frame.height == 2
    assert frame["source"].unique().to_list() == [SOURCE_FPL_REVIEW]
    assert frame["xpts"].to_list() == pytest.approx([6.2, 4.1])
    assert frame["as_of_gw"].to_list() == [1, 1]
    assert frame["target_gw"].to_list() == [1, 1]
    stamps = frame["available_at"].dt.replace_time_zone("UTC").to_list()
    assert stamps == [_STAMP, _STAMP]


def test_load_json_aliases_ev() -> None:
    frame = load_fpl_review_drop(_JSON)
    assert set(frame["source"].unique()) == {SOURCE_FPL_REVIEW}
    assert frame["xpts"].to_list() == pytest.approx([6.2, 4.1])


def test_load_requires_available_at_when_absent(tmp_path: Path) -> None:
    path = tmp_path / "bare.csv"
    path.write_text(
        "season,gw,player_key,fixture_id,xpts\n2025-26,1,1,1,3.0\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="available_at"):
        load_fpl_review_drop(path)
    frame = load_fpl_review_drop(path, available_at=_STAMP)
    assert frame["available_at"].dt.replace_time_zone("UTC").to_list() == [_STAMP]


def test_load_rejects_missing_columns(tmp_path: Path) -> None:
    path = tmp_path / "bad.csv"
    path.write_text("season,gw,xpts\n2025-26,1,1.0\n", encoding="utf-8")
    with pytest.raises(ValueError, match="missing required columns"):
        load_fpl_review_drop(path, available_at=_STAMP)


def test_ingest_merges_alongside_existing_ep_next(tmp_path: Path) -> None:
    existing = pl.DataFrame(
        {
            "source": [SOURCE_EP_NEXT],
            "as_of_season": ["2025-26"],
            "as_of_gw": [1],
            "target_season": ["2025-26"],
            "target_gw": [1],
            "player_key": [999],
            "fixture_id": [9],
            "xpts": [5.0],
            "available_at": [_STAMP],
            "quality_flag": [None],
        }
    )
    write_benchmark_archive(tmp_path, _AS_OF, existing)
    path, n = ingest_fpl_review_drop(tmp_path, _CSV)
    assert n == 2
    assert path.exists()
    staged = read_benchmark_archive(tmp_path, _AS_OF)
    sources = sorted(staged["source"].unique().to_list())
    assert sources == [SOURCE_EP_NEXT, SOURCE_FPL_REVIEW]
    assert staged.height == 3
    # Re-ingest replaces Review rows only.
    ingest_fpl_review_drop(tmp_path, _CSV)
    again = read_benchmark_archive(tmp_path, _AS_OF)
    assert again.height == 3
    assert again.filter(pl.col("source") == SOURCE_EP_NEXT).height == 1


def test_copy_drop_to_raw_is_write_once(tmp_path: Path) -> None:
    dest = copy_drop_to_raw(tmp_path, _CSV, as_of=_AS_OF, captured_at=_STAMP)
    assert dest.is_file()
    assert dest.parent == (
        fpl_review_raw_root(tmp_path)
        / "season=2025-26"
        / "gw=1"
        / "captured_at=2025-08-15T10-00-00Z"
    )
    with pytest.raises(FileExistsError):
        copy_drop_to_raw(tmp_path, _CSV, as_of=_AS_OF, captured_at=_STAMP)


def test_cli_ingest_review(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    try:
        main(["ingest-review", str(_CSV), "--copy-raw"])
    finally:
        get_settings.cache_clear()
    out = capsys.readouterr().out
    assert "staged 2 fpl_review rows" in out
    assert "raw:" in out
    staged = read_benchmark_archive(tmp_path, _AS_OF)
    assert staged.height == 2
    assert list(fpl_review_raw_root(tmp_path).rglob("projections.csv"))
