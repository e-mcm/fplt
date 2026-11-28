"""#390: archived benchmarks as mean-only ledger pseudo-models."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from fplt.backtest.ledger import (
    ARCHIVE_MODEL_PREFIX,
    BENCHMARK_RUN_KIND,
    archive_model_id,
    guard_openfpl_benchmarks,
    predictions_from_benchmarks,
)
from fplt.backtest.walk_forward import walk_forward
from fplt.data.benchmark_projections import write_benchmark_archive
from fplt.data.outcomes import stage_outcomes
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.baselines.rolling_mean import RollingMeanPredictor
from fplt.models.spec import ModelSpec
from fplt.schemas.benchmark_projections import (
    SOURCE_EP_NEXT,
    SOURCE_OPENFPL,
    BenchmarkProjectionSchema,
)
from fplt.store import Store
from fplt.store.runs import write_backtest_run
from fplt.types import GWKey
from fplt.utils.frames import require_dataframe
from tests.builders import build_league

_SEASON = "2024-25"
_SAVE = datetime(2024, 8, 16, 16, tzinfo=UTC)


def test_guard_openfpl_drops_training_seasons() -> None:
    """OpenFPL must never be scored on 2020-21…2023-24 (AGENTS.md §6)."""
    rows = pl.concat(
        [
            _bench_frame(SOURCE_EP_NEXT, season="2022-23", xpts=4.0),
            _bench_frame(SOURCE_OPENFPL, season="2022-23", xpts=3.0),
            _bench_frame(SOURCE_OPENFPL, season="2023-24", xpts=3.5),
            _bench_frame(SOURCE_OPENFPL, season="2024-25", xpts=5.0),
            _bench_frame(SOURCE_EP_NEXT, season="2024-25", xpts=6.0),
        ]
    )
    guarded = guard_openfpl_benchmarks(rows)
    openfpl = guarded.filter(pl.col("source") == SOURCE_OPENFPL)
    assert set(openfpl["as_of_season"].to_list()) == {"2024-25"}
    ep = guarded.filter(pl.col("source") == SOURCE_EP_NEXT)
    assert set(ep["as_of_season"].to_list()) == {"2022-23", "2024-25"}


def test_predictions_from_benchmarks_mean_only(tmp_path: Path) -> None:
    build_league(DataStore(tmp_path))
    write_benchmark_archive(
        tmp_path,
        GWKey(_SEASON, 1),
        _bench_frame(SOURCE_EP_NEXT, season=_SEASON, xpts=6.5),
    )
    preds = predictions_from_benchmarks(tmp_path)
    assert preds.height == 2
    assert preds["model_id"].unique().to_list() == [archive_model_id(SOURCE_EP_NEXT)]
    assert preds["run_kind"].unique().to_list() == [BENCHMARK_RUN_KIND]
    assert preds["xpts_sd"].null_count() == preds.height
    assert preds["p_60"].null_count() == preds.height
    assert preds["p_haul"].null_count() == preds.height
    assert set(preds["position"].to_list()) == {"MID", "FWD"}


def test_ledger_view_includes_benchmark_rows_for_archived_gw(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    spec = ModelSpec(name="rolling_mean", kind="rolling_mean", params={"window": 5})
    frame = walk_forward(
        RollingMeanPredictor(spec),
        PITData(store),
        [GWKey(_SEASON, 1)],
        steps=1,
    )
    write_backtest_run(
        tmp_path,
        spec=spec,
        as_ofs=[(GWKey(_SEASON, 1), store.deadline(GWKey(_SEASON, 1)))],
        horizon=1,
        frame=frame,
    )
    write_benchmark_archive(
        tmp_path,
        GWKey(_SEASON, 1),
        pl.concat(
            [
                _bench_frame(SOURCE_EP_NEXT, season=_SEASON, xpts=6.5),
                _bench_frame(SOURCE_OPENFPL, season=_SEASON, xpts=5.0),
            ]
        ),
    )
    stage_outcomes(store)
    ledger = Store(tmp_path).ledger()
    gw1 = ledger.filter(pl.col("target_gw") == 1)
    ids = set(gw1["model_id"].unique().to_list())
    assert archive_model_id(SOURCE_EP_NEXT) in ids
    assert archive_model_id(SOURCE_OPENFPL) in ids
    assert any(not mid.startswith(ARCHIVE_MODEL_PREFIX) for mid in ids)
    ep = gw1.filter(pl.col("model_id") == archive_model_id(SOURCE_EP_NEXT))
    assert ep.filter(pl.col("player_key") == 200)["actual_pts"].to_list() == [2]
    assert ep["actual_pts"].null_count() == 0
    from fplt.store.runs import list_succeeded_predict_run_ids

    run_ids = list_succeeded_predict_run_ids(tmp_path)
    assert run_ids
    scoped = Store(tmp_path).ledger(runs=run_ids)
    scoped_ids = set(scoped["model_id"].unique().to_list())
    assert archive_model_id(SOURCE_EP_NEXT) not in scoped_ids
    forced = Store(tmp_path).ledger(runs=run_ids, include_benchmarks=True)
    assert archive_model_id(SOURCE_EP_NEXT) in set(
        forced["model_id"].unique().to_list()
    )


def test_openfpl_training_season_absent_from_ledger(tmp_path: Path) -> None:
    build_league(DataStore(tmp_path))
    early = "2022-23"
    # Staged players are 2024-25; inject archive + players for the early season.
    players = (
        DataStore(tmp_path).read("players").with_columns(pl.lit(early).alias("season"))
    )
    archive = pl.concat(
        [
            _bench_frame(SOURCE_EP_NEXT, season=early, xpts=4.0),
            _bench_frame(SOURCE_OPENFPL, season=early, xpts=3.0),
        ]
    )
    write_benchmark_archive(tmp_path, GWKey(early, 1), archive)
    preds = predictions_from_benchmarks(tmp_path, players=players)
    ids = set(preds["model_id"].unique().to_list())
    assert archive_model_id(SOURCE_EP_NEXT) in ids
    assert archive_model_id(SOURCE_OPENFPL) not in ids


def test_guard_openfpl_empty_and_missing_source() -> None:
    assert guard_openfpl_benchmarks(pl.DataFrame()).is_empty()
    bare = pl.DataFrame({"xpts": [1.0]})
    assert guard_openfpl_benchmarks(bare).equals(bare)


def test_predictions_from_benchmarks_empty_after_guard(tmp_path: Path) -> None:
    """Only early-season OpenFPL → nothing left for the ledger."""
    early = _bench_frame(SOURCE_OPENFPL, season="2021-22", xpts=2.0)
    preds = predictions_from_benchmarks(
        tmp_path, benchmarks=early, players=pl.DataFrame()
    )
    assert preds.is_empty()


def test_predictions_from_benchmarks_incomplete_players(tmp_path: Path) -> None:
    build_league(DataStore(tmp_path))
    write_benchmark_archive(
        tmp_path,
        GWKey(_SEASON, 1),
        _bench_frame(SOURCE_EP_NEXT, season=_SEASON, xpts=6.5),
    )
    missing_cols = predictions_from_benchmarks(
        tmp_path, players=pl.DataFrame({"player_key": [100]})
    )
    assert missing_cols["position"].to_list() == ["UNK", "UNK"]
    empty_players = predictions_from_benchmarks(tmp_path, players=pl.DataFrame())
    assert empty_players["position"].to_list() == ["UNK", "UNK"]


def test_ledger_benchmarks_only_without_runs(tmp_path: Path) -> None:
    build_league(DataStore(tmp_path))
    write_benchmark_archive(
        tmp_path,
        GWKey(_SEASON, 1),
        _bench_frame(SOURCE_EP_NEXT, season=_SEASON, xpts=6.5),
    )
    stage_outcomes(DataStore(tmp_path))
    ledger = Store(tmp_path).ledger()
    assert set(ledger["model_id"].unique().to_list()) == {
        archive_model_id(SOURCE_EP_NEXT)
    }
    assert ledger.filter(pl.col("player_key") == 200)["actual_pts"].to_list() == [2]


def _bench_frame(
    source: str,
    *,
    season: str,
    xpts: float,
    as_of_gw: int = 1,
    available_at: datetime | None = None,
) -> pl.DataFrame:
    stamp = available_at if available_at is not None else _SAVE
    if season != _SEASON:
        # Offset stamp into that season's calendar year for realism only.
        year = int(season.split("-", 1)[0])
        stamp = datetime(year, 8, 16, 16, tzinfo=UTC)
    frame = pl.DataFrame(
        {
            "source": [source, source],
            "as_of_season": [season, season],
            "as_of_gw": [as_of_gw, as_of_gw],
            "target_season": [season, season],
            "target_gw": [as_of_gw, as_of_gw],
            "player_key": [100, 200],
            "fixture_id": [1, 1],
            "xpts": [xpts, xpts - 1.0],
            "available_at": [stamp, stamp + timedelta(seconds=1)],
            "quality_flag": [None, None],
        }
    ).with_columns(pl.col("available_at").cast(pl.Datetime(time_zone="UTC")))
    return require_dataframe(BenchmarkProjectionSchema.validate(frame))
