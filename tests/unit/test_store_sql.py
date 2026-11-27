"""Stage C fplt sql: DuckDB views over catalog + succeeded runs only."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import polars as pl
import pytest

from fplt.models.predictions import empty_predictions
from fplt.schemas.manifests import AsOfDeadline, OfficialPointer
from fplt.store.catalog import (
    catalog_runs_path,
    empty_runs_catalog,
    rebuild_runs_catalog,
)
from fplt.store.runs import RunWriter
from fplt.store.sql import (
    _concat_run_tables,
    _official_frame,
    _outcomes_frame,
    _register_frame,
    open_sql,
    run_sql,
    sql_repl,
)

_TS = datetime(2026, 11, 24, 12, 56, 1, tzinfo=UTC)


def _deadline() -> AsOfDeadline:
    return AsOfDeadline(season="2026-27", gw=12, deadline_utc=_TS)


def _writer(tmp_path: Path, run_id: str, **overrides: object) -> RunWriter:
    kwargs: dict[str, object] = {
        "kind": "adhoc",
        "mode": "live",
        "run_type": "predict",
        "spec_hash": "a" * 64,
        "spec_label": "demo",
        "model_id": "deadbeef",
        "as_of": [_deadline()],
        "horizon": 3,
        "code_sha": "abc1234",
        "code_dirty": False,
        "lockfile_hash": "b" * 64,
        "host": "test",
        "raw_manifest_hash": "c" * 64,
        "staged_digests": {},
        "started_at": _TS,
        "run_id": run_id,
    }
    kwargs.update(overrides)
    return RunWriter(tmp_path, **kwargs)  # type: ignore[arg-type]


def test_sql_views_empty_when_sources_missing(tmp_path: Path) -> None:
    frame = run_sql("SELECT * FROM runs", tmp_path)
    assert frame.is_empty()
    official = run_sql("SELECT * FROM official", tmp_path)
    assert official.is_empty()
    ledger = run_sql("SELECT * FROM ledger", tmp_path)
    assert ledger.is_empty()


def test_sql_hides_incomplete_runs(tmp_path: Path) -> None:
    good = "r_20261124T125601Z_sql001"
    with _writer(tmp_path, good) as writer:
        writer.write_parquet(
            "predictions/player_fixture.parquet",
            empty_predictions(),
        )
    incomplete = "r_20261124T125601Z_sql002"
    w = _writer(tmp_path, incomplete)
    w.__enter__()
    # Incomplete dir still has a predictions file, but no _SUCCESS.
    (tmp_path / "runs" / f"run={incomplete}" / "predictions").mkdir(
        parents=True, exist_ok=True
    )
    empty_predictions().write_parquet(
        tmp_path
        / "runs"
        / f"run={incomplete}"
        / "predictions"
        / "player_fixture.parquet"
    )
    rebuild_runs_catalog(tmp_path)

    runs = run_sql("SELECT run_id FROM runs ORDER BY run_id", tmp_path)
    assert runs["run_id"].to_list() == [good]
    # Incomplete run's parquet exists on disk but must not enter the view via
    # catalog (only _SUCCESS runs are scanned). View still opens.
    pf = run_sql("SELECT COUNT(*) AS n FROM player_fixture", tmp_path)
    assert pf["n"].to_list() == [empty_predictions().height]


def test_sql_official_and_rebuild_flag(tmp_path: Path) -> None:
    run_id = "r_20261124T125601Z_sql003"
    with _writer(tmp_path, run_id, kind="official", run_type="plan", plan_horizon=3):
        pass
    base = tmp_path / "official" / "season=2026-27" / "gw=12" / "checkpoint=T-90m"
    base.mkdir(parents=True)
    (base / "line=v4_h3.json").write_text(
        OfficialPointer(
            predict_run_id="r_pred",
            plan_run_id=run_id,
            role="primary",
            squad_mode="real",
            written_at=_TS,
        ).model_dump_json()
        + "\n",
        encoding="utf-8",
    )
    frame = run_sql(
        "SELECT line, plan_run_id FROM official",
        tmp_path,
        rebuild_catalog=True,
    )
    assert frame.to_dicts() == [{"line": "v4_h3", "plan_run_id": run_id}]
    con = open_sql(tmp_path)
    names = {
        row[0]
        for row in con.execute(
            "SELECT table_name FROM information_schema.tables WHERE table_type = 'VIEW'"
        ).fetchall()
    }
    con.close()
    assert {
        "runs",
        "official",
        "player_fixture",
        "minutes",
        "plans",
        "outcomes",
        "ledger",
    }.issubset(names)


def test_register_frame_zero_width_stub() -> None:
    import duckdb

    con = duckdb.connect(database=":memory:")
    _register_frame(con, "emptyish", pl.DataFrame())
    assert con.sql("SELECT * FROM emptyish").pl().columns == ["_empty"]
    con.close()


def test_sql_repl_quit_multiline_error_and_eof(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import duckdb

    con = duckdb.connect(database=":memory:")
    con.execute("CREATE VIEW runs AS SELECT 1 AS n")
    answers = iter(["SELECT", "1 AS n;", "not sql;", "quit"])

    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    sql_repl(con)
    out = capsys.readouterr().out
    assert "fplt sql" in out
    assert "Error:" in out
    con.close()

    con = duckdb.connect(database=":memory:")

    def _eof(_prompt: str = "") -> str:
        raise EOFError

    monkeypatch.setattr("builtins.input", _eof)
    sql_repl(con)
    con.close()


def test_concat_skips_catalog_runs_without_success(tmp_path: Path) -> None:
    incomplete = "r_incomplete_sql"
    # Catalog row for a run that has parquet but no _SUCCESS marker.
    catalog_path = catalog_runs_path(tmp_path)
    catalog_path.parent.mkdir(parents=True, exist_ok=True)
    frame = pl.DataFrame(
        {
            "run_id": [incomplete],
            "kind": ["adhoc"],
            "mode": ["live"],
            "purpose": ["predict"],
            "experiment_id": [None],
            "arm": [None],
            "run_type": ["predict"],
            "spec_hash": ["a" * 64],
            "spec_label": ["demo"],
            "model_id": ["deadbeef"],
            "line": [None],
            "plan_horizon": [None],
            "squad_mode": [None],
            "prev_run_id": [None],
            "code_sha": ["abc1234"],
            "code_dirty": [False],
            "prod_tag": [None],
            "fplt_mc_version": [None],
            "lockfile_hash": ["b" * 64],
            "build_id": [None],
            "as_of_season": ["2026-27"],
            "as_of_gw": [12],
            "deadline_utc": [_TS],
            "n_as_of": [1],
            "horizon": [3],
            "checkpoint": [None],
            "host": ["test"],
            "started_at": [_TS],
            "finished_at": [_TS],
            "status": ["succeeded"],
            "parent_run_id": [None],
        },
        schema=empty_runs_catalog().schema,
    )
    frame.write_parquet(catalog_path)
    (tmp_path / "runs" / f"run={incomplete}" / "predictions").mkdir(parents=True)
    empty_predictions().write_parquet(
        tmp_path
        / "runs"
        / f"run={incomplete}"
        / "predictions"
        / "player_fixture.parquet"
    )
    out = _concat_run_tables(tmp_path, "predictions/player_fixture.parquet")
    assert out.columns == ["run_id"]
    assert out.is_empty()


def test_official_frame_skips_bad_paths_and_invalid_json(tmp_path: Path) -> None:
    root = tmp_path / "official"
    bad_layout = root / "season=2026-27" / "not-a-gw"
    bad_layout.mkdir(parents=True)
    (bad_layout / "line=orphan.json").write_text("{}", encoding="utf-8")
    good_dir = root / "season=2026-27" / "gw=12" / "checkpoint=T-90m"
    good_dir.mkdir(parents=True)
    (good_dir / "line=broken.json").write_text("{not-json", encoding="utf-8")
    frame = _official_frame(tmp_path)
    assert frame.is_empty()
    assert "season" in frame.columns


def test_outcomes_frame_prefers_nonempty_curated_build(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    outcomes = pl.DataFrame({"player_key": [1], "fixture_id": [10], "points": [6]})

    class _Source:
        def table(self, _name: str) -> Any:
            return SimpleNamespace(collect=lambda: outcomes)

    monkeypatch.setattr("fplt.store.sql.latest_build_id", lambda _root: "build123")
    monkeypatch.setattr(
        "fplt.store.sql.CuratedBuildSource.load",
        lambda _root, _build_id: _Source(),
    )
    assert _outcomes_frame(tmp_path).equals(outcomes)


def test_outcomes_frame_falls_back_when_build_empty(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    empty = pl.DataFrame(schema={"player_key": pl.Int64()})
    staged = pl.DataFrame({"player_key": [2]})

    class _Source:
        def table(self, _name: str) -> Any:
            return SimpleNamespace(collect=lambda: empty)

    class _Store:
        def __init__(self, _root: Path) -> None:
            pass

        def read(self, _name: str) -> pl.DataFrame:
            return staged

    monkeypatch.setattr("fplt.store.sql.latest_build_id", lambda _root: "build123")
    monkeypatch.setattr(
        "fplt.store.sql.CuratedBuildSource.load",
        lambda _root, _build_id: _Source(),
    )
    monkeypatch.setattr("fplt.data.store.DataStore", _Store)
    assert _outcomes_frame(tmp_path).equals(staged)
