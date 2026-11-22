"""Paired archive compare: coverage gate, board metrics, HAC intervals."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from fplt.backtest.archive_compare import (
    HEADLINE_METRIC,
    OUTCOME_BLANKS,
    OUTCOME_HAULERS,
    OUTCOME_TICKERS,
    OUTCOME_ZEROS,
    archive_model_id,
    benchmarks_to_ledger,
    compare_to_archives,
    covered_pair_keys,
    format_archive_board,
    outcome_group,
    outcome_group_metrics,
)
from fplt.backtest.ledger import align_ledger_columns
from fplt.data.benchmark_projections import write_benchmark_archive
from fplt.schemas.benchmark_projections import (
    SOURCE_EP_NEXT,
    SOURCE_OPENFPL,
    BenchmarkProjectionSchema,
)
from fplt.schemas.ledger import LedgerSchema
from fplt.types import GWKey

_SEASON = "2024-25"
_OURS = "ours_test_id"
_DEADLINE_GW2 = datetime(2024, 8, 23, 17, tzinfo=UTC)
_DEADLINE_GW3 = datetime(2024, 8, 30, 17, tzinfo=UTC)
_SAVE = datetime(2024, 8, 23, 16, tzinfo=UTC)
_LATE = datetime(2024, 8, 23, 18, tzinfo=UTC)


def test_outcome_group_buckets() -> None:
    assert outcome_group(0, 0) == OUTCOME_ZEROS
    assert outcome_group(0, 90) == OUTCOME_BLANKS
    assert outcome_group(2, 90) == OUTCOME_BLANKS
    assert outcome_group(3, 90) == OUTCOME_TICKERS
    assert outcome_group(4, None) == OUTCOME_TICKERS
    assert outcome_group(5, 90) == OUTCOME_HAULERS
    assert outcome_group(0, None) == OUTCOME_ZEROS


def test_coverage_gate_requires_every_arm_pre_deadline() -> None:
    ours = _ours_ledger()
    benches = pl.concat(
        [
            _bench_rows(SOURCE_EP_NEXT, available_at=_SAVE),
            # GW2 openfpl: only player 100 pre-deadline; 200 arrives late.
            _bench_rows(
                SOURCE_OPENFPL,
                available_at=_SAVE,
                player_key=100,
                fixture_id=2,
                as_of_gw=2,
            ),
            _bench_rows(
                SOURCE_OPENFPL,
                available_at=_LATE,
                player_key=200,
                fixture_id=2,
                as_of_gw=2,
                xpts=3.0,
            ),
            # GW3 both sources complete pre-deadline.
            _bench_rows(
                SOURCE_EP_NEXT,
                available_at=_DEADLINE_GW3 - timedelta(hours=1),
                as_of_gw=3,
            ),
            _bench_rows(
                SOURCE_OPENFPL,
                available_at=_DEADLINE_GW3 - timedelta(hours=1),
                as_of_gw=3,
            ),
        ]
    )
    deadlines = _deadlines()
    keys = covered_pair_keys(ours, benches, deadlines, (SOURCE_EP_NEXT, SOURCE_OPENFPL))
    # GW2 player 100 covered; player 200 missing valid openfpl; GW3 both ok.
    pairs = {(row["player_key"], row["as_of_gw"]) for row in keys.iter_rows(named=True)}
    assert (100, 2) in pairs
    assert (200, 2) not in pairs
    assert (100, 3) in pairs
    assert (200, 3) in pairs


def test_post_deadline_archive_excluded() -> None:
    ours = _ours_ledger()
    late_only = pl.concat(
        [
            _bench_rows(SOURCE_EP_NEXT, available_at=_LATE, as_of_gw=2),
            _bench_rows(SOURCE_OPENFPL, available_at=_SAVE, as_of_gw=2),
        ]
    )
    keys = covered_pair_keys(
        ours, late_only, _deadlines(), (SOURCE_EP_NEXT, SOURCE_OPENFPL)
    )
    assert keys.filter(pl.col("as_of_gw") == 2).is_empty()


def test_benchmarks_to_ledger_and_outcome_metrics() -> None:
    ours = _ours_ledger()
    benches = pl.concat(
        [
            _bench_rows(SOURCE_EP_NEXT, available_at=_SAVE),
            _bench_rows(SOURCE_OPENFPL, available_at=_SAVE, xpts=1.0),
        ]
    )
    keys = covered_pair_keys(
        ours, benches, _deadlines(), (SOURCE_EP_NEXT, SOURCE_OPENFPL)
    )
    ours_c = ours.join(keys, on=list(keys.columns), how="inner")
    led = benchmarks_to_ledger(benches.join(keys, on=list(keys.columns)), ours_c)
    assert set(led["model_id"].unique().to_list()) == {
        archive_model_id(SOURCE_EP_NEXT),
        archive_model_id(SOURCE_OPENFPL),
    }
    paired = pl.concat([ours_c, led])
    groups = outcome_group_metrics(paired)
    assert "rmse" in groups.columns
    assert set(groups["outcome_group"].unique().to_list()) <= set(
        (OUTCOME_ZEROS, OUTCOME_BLANKS, OUTCOME_TICKERS, OUTCOME_HAULERS)
    )


def test_compare_to_archives_end_to_end(tmp_path: Path) -> None:
    _write_fixture_archives(tmp_path)
    ledger = _ours_ledger()
    result = compare_to_archives(
        tmp_path,
        season=_SEASON,
        model=_OURS,
        from_gw=2,
        to_gw=3,
        sources=(SOURCE_EP_NEXT, SOURCE_OPENFPL),
        ledger=ledger,
        benchmarks=read_all_benches(tmp_path),
        deadlines=_deadlines(),
        output_dir=tmp_path / "out",
        bootstrap_reps=50,
        bootstrap_seed=0,
    )
    assert result.coverage.n_covered >= 1
    assert HEADLINE_METRIC == "rmse"
    assert (tmp_path / "out" / "summary.json").exists()
    assert (tmp_path / "out" / "board.md").exists()
    assert (tmp_path / "out" / "paired_ledger.parquet").exists()
    ids = (
        set(result.summary["model_id"].to_list())
        if not result.summary.is_empty()
        else set()
    )
    assert _OURS in ids or result.coverage.n_scored == 0
    assert any(item.metric == "rmse" for item in result.intervals)
    board = format_archive_board(result)
    assert "headline_metric=rmse" in board
    assert "archive:ep_next" in board or result.coverage.n_covered == 0


def test_compare_rejects_unknown_source(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown archive source"):
        compare_to_archives(
            tmp_path,
            season=_SEASON,
            model=_OURS,
            sources=("not_a_source",),
            ledger=_ours_ledger(),
            benchmarks=_bench_rows(SOURCE_EP_NEXT, available_at=_SAVE),
            deadlines=_deadlines(),
            bootstrap_reps=10,
        )


def test_compare_requires_ours_rows(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="no ledger rows"):
        compare_to_archives(
            tmp_path,
            season=_SEASON,
            model="missing_model",
            ledger=_ours_ledger(),
            benchmarks=_bench_rows(SOURCE_EP_NEXT, available_at=_SAVE),
            deadlines=_deadlines(),
            bootstrap_reps=10,
        )


def read_all_benches(data_dir: Path) -> pl.DataFrame:
    from fplt.data.benchmark_projections import read_benchmark_archive

    return read_benchmark_archive(data_dir)


def _write_fixture_archives(data_dir: Path) -> None:
    for gw, save in ((2, _SAVE), (3, _DEADLINE_GW3 - timedelta(hours=1))):
        frame = pl.concat(
            [
                _bench_rows(SOURCE_EP_NEXT, available_at=save, as_of_gw=gw),
                _bench_rows(SOURCE_OPENFPL, available_at=save, as_of_gw=gw, xpts=2.5),
            ]
        )
        write_benchmark_archive(data_dir, GWKey(_SEASON, gw), frame)


def _deadlines() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "as_of_season": [_SEASON, _SEASON],
            "as_of_gw": [2, 3],
            "deadline_time": [_DEADLINE_GW2, _DEADLINE_GW3],
        }
    )


def _ours_ledger() -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for gw, fixture in ((2, 2), (3, 3)):
        for player, pts, minutes, xpts, position in (
            (100, 6, 90, 5.0, "MID"),
            (200, 0, 0, 1.0, "FWD"),
        ):
            rows.append(
                {
                    "model_id": _OURS,
                    "run_kind": "walk_forward",
                    "as_of_season": _SEASON,
                    "as_of_gw": gw,
                    "target_season": _SEASON,
                    "target_gw": gw,
                    "horizon": 1,
                    "player_key": player,
                    "position": position,
                    "fixture_id": fixture,
                    "xpts": xpts,
                    "xpts_sd": None,
                    "p_haul": None,
                    "p_60": None,
                    "actual_pts": pts,
                    "actual_minutes": minutes,
                }
            )
    frame = align_ledger_columns(pl.DataFrame(rows))
    checked = LedgerSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked


def _bench_rows(
    source: str,
    *,
    available_at: datetime,
    as_of_gw: int = 2,
    player_key: int | None = None,
    fixture_id: int | None = None,
    xpts: float = 4.0,
    quality_flag: str | None = None,
) -> pl.DataFrame:
    if player_key is not None:
        players = [(player_key, fixture_id or as_of_gw, xpts)]
    else:
        players = [(100, as_of_gw, xpts), (200, as_of_gw, xpts - 1.0)]
    frame = pl.DataFrame(
        {
            "source": [source] * len(players),
            "as_of_season": [_SEASON] * len(players),
            "as_of_gw": [as_of_gw] * len(players),
            "target_season": [_SEASON] * len(players),
            "target_gw": [as_of_gw] * len(players),
            "player_key": [p[0] for p in players],
            "fixture_id": [p[1] for p in players],
            "xpts": [p[2] for p in players],
            "available_at": [available_at] * len(players),
            "quality_flag": [quality_flag] * len(players),
        }
    )
    checked = BenchmarkProjectionSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked
