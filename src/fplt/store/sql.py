"""DuckDB session with catalog / run / ledger views for ``fplt sql``.

Registers read-only views over the local ``$FPLT_DATA_DIR`` mirror so ad-hoc
questions can join the same tables the ``Store`` API exposes
(``docs/proposals/fplt-data-model.md`` §6).
"""

from __future__ import annotations

import logging
import re
from pathlib import Path

import duckdb
import polars as pl

from fplt.schemas.manifests import OfficialPointer
from fplt.settings import get_settings
from fplt.store.api import Store
from fplt.store.builds import CuratedBuildSource, latest_build_id
from fplt.store.catalog import read_runs_catalog, rebuild_runs_catalog
from fplt.store.runs import run_dir

logger = logging.getLogger(__name__)

_SUCCESS = "_SUCCESS"
_OFFICIAL_RE = re.compile(
    r"^season=(?P<season>[^/]+)/gw=(?P<gw>\d+)/"
    r"checkpoint=(?P<checkpoint>[^/]+)/line=(?P<line>[^/]+)\.json$"
)

_VIEW_NAMES = (
    "runs",
    "official",
    "player_fixture",
    "minutes",
    "plans",
    "outcomes",
    "ledger",
)


def open_sql(
    data_dir: Path | None = None,
    *,
    rebuild_catalog: bool = False,
) -> duckdb.DuckDBPyConnection:
    """Open an in-memory DuckDB connection with Store views registered.

    Args:
        data_dir: Local data root; defaults to settings ``data_dir``.
        rebuild_catalog: When true, rescan runs into ``catalog/runs.parquet``
            before registering the ``runs`` view.

    Returns:
        DuckDB connection with views ``runs``, ``official``, ``player_fixture``,
        ``minutes``, ``plans``, ``outcomes``, and ``ledger``.
    """
    root = data_dir if data_dir is not None else get_settings().data_dir
    if rebuild_catalog:
        rebuild_runs_catalog(root)
    con = duckdb.connect(database=":memory:")
    store = Store(root)
    _register_frame(con, "runs", read_runs_catalog(root))
    _register_frame(con, "official", _official_frame(root))
    _register_frame(
        con,
        "player_fixture",
        _concat_run_tables(root, "predictions/player_fixture.parquet"),
    )
    _register_frame(
        con,
        "minutes",
        _concat_run_tables(root, "predictions/components/minutes.parquet"),
    )
    _register_frame(con, "plans", _concat_run_tables(root, "plans/plan.parquet"))
    _register_frame(con, "outcomes", _outcomes_frame(root))
    _register_frame(con, "ledger", store.ledger())
    logger.info("fplt sql: registered views %s over %s", _VIEW_NAMES, root)
    return con


def run_sql(
    query: str,
    data_dir: Path | None = None,
    *,
    rebuild_catalog: bool = False,
) -> pl.DataFrame:
    """Execute one SQL query against the Store views and return a DataFrame.

    Args:
        query: DuckDB SQL (may reference the registered views).
        data_dir: Local data root.
        rebuild_catalog: Rebuild ``catalog/runs.parquet`` first when true.

    Returns:
        Query result as a polars DataFrame.
    """
    con = open_sql(data_dir, rebuild_catalog=rebuild_catalog)
    try:
        return con.sql(query).pl()
    finally:
        con.close()


def sql_repl(con: duckdb.DuckDBPyConnection) -> None:
    """Read SQL statements from stdin until EOF / quit and print results.

    Args:
        con: Connection from ``open_sql``.
    """
    print(
        "fplt sql — views: " + ", ".join(_VIEW_NAMES) + " (quit / exit / EOF to leave)",
        flush=True,
    )
    buffer = ""
    while True:
        try:
            prompt = "fplt-sql> " if not buffer else "....> "
            line = input(prompt)
        except EOFError:
            print()
            break
        stripped = line.strip()
        if not buffer and stripped.lower() in {"quit", "exit", "\\q"}:
            break
        buffer = f"{buffer}\n{line}" if buffer else line
        if not stripped.endswith(";"):
            continue
        try:
            print(con.sql(buffer).pl())
        except Exception as exc:  # noqa: BLE001 — surface SQL errors to the REPL
            print(f"Error: {exc}", flush=True)
        buffer = ""


def _register_frame(
    con: duckdb.DuckDBPyConnection, name: str, frame: pl.DataFrame
) -> None:
    # DuckDB rejects zero-column frames; keep a typed stub so views always exist.
    if frame.width == 0:
        frame = pl.DataFrame(schema={"_empty": pl.Utf8()})
    con.register(f"_{name}_df", frame)
    con.execute(f"CREATE OR REPLACE VIEW {name} AS SELECT * FROM _{name}_df")


def _concat_run_tables(data_dir: Path, relative: str) -> pl.DataFrame:
    catalog = read_runs_catalog(data_dir)
    if catalog.is_empty():
        return pl.DataFrame(schema={"run_id": pl.Utf8()})
    frames: list[pl.DataFrame] = []
    for run_id in catalog["run_id"].to_list():
        path = run_dir(data_dir, str(run_id))
        if not (path / _SUCCESS).is_file():
            continue
        target = path / relative
        if target.is_file():
            frames.append(pl.read_parquet(target))
    if not frames:
        return pl.DataFrame(schema={"run_id": pl.Utf8()})
    return pl.concat(frames, how="diagonal_relaxed")


def _official_frame(data_dir: Path) -> pl.DataFrame:
    root = data_dir / "official"
    if not root.is_dir():
        return pl.DataFrame(
            schema={
                "season": pl.Utf8(),
                "gw": pl.Int64(),
                "checkpoint": pl.Utf8(),
                "line": pl.Utf8(),
                "predict_run_id": pl.Utf8(),
                "plan_run_id": pl.Utf8(),
                "role": pl.Utf8(),
                "squad_mode": pl.Utf8(),
            }
        )
    rows: list[dict[str, object]] = []
    for path in sorted(root.rglob("line=*.json")):
        rel = path.relative_to(root).as_posix()
        match = _OFFICIAL_RE.match(rel)
        if match is None:
            continue
        try:
            pointer = OfficialPointer.model_validate_json(
                path.read_text(encoding="utf-8")
            )
        except ValueError:
            continue
        rows.append(
            {
                "season": match.group("season"),
                "gw": int(match.group("gw")),
                "checkpoint": match.group("checkpoint"),
                "line": match.group("line"),
                "predict_run_id": pointer.predict_run_id,
                "plan_run_id": pointer.plan_run_id,
                "role": pointer.role,
                "squad_mode": pointer.squad_mode,
            }
        )
    if not rows:
        return pl.DataFrame(
            schema={
                "season": pl.Utf8(),
                "gw": pl.Int64(),
                "checkpoint": pl.Utf8(),
                "line": pl.Utf8(),
                "predict_run_id": pl.Utf8(),
                "plan_run_id": pl.Utf8(),
                "role": pl.Utf8(),
                "squad_mode": pl.Utf8(),
            }
        )
    return pl.DataFrame(rows)


def _outcomes_frame(data_dir: Path) -> pl.DataFrame:
    build_id = latest_build_id(data_dir)
    if build_id is not None:
        frame = (
            CuratedBuildSource.load(data_dir, build_id)
            .table("player_fixture_outcomes")
            .collect()
        )
        if not frame.is_empty():
            return frame
    from fplt.data.store import DataStore

    return DataStore(data_dir).read("player_fixture_outcomes")
