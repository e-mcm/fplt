"""Stage C fplt sql: DuckDB views over catalog + succeeded runs only."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

from fplt.models.predictions import empty_predictions
from fplt.schemas.manifests import AsOfDeadline, OfficialPointer
from fplt.store.catalog import rebuild_runs_catalog
from fplt.store.runs import RunWriter
from fplt.store.sql import open_sql, run_sql

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
