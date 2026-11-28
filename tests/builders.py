"""Synthetic staged seasons for contract tests. No network and no real FPL dumps."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from fplt.data.store import DataStore

_START = datetime(2024, 8, 1, tzinfo=UTC)
_DEADLINE_GW1 = datetime(2024, 8, 16, 17, tzinfo=UTC)
_KICKOFF_GW1 = datetime(2024, 8, 17, 14, tzinfo=UTC)
_DEADLINE_GW2 = datetime(2024, 8, 23, 17, tzinfo=UTC)
_KICKOFF_GW2 = datetime(2024, 8, 24, 14, tzinfo=UTC)


def _utc(frame: pl.DataFrame) -> pl.DataFrame:
    exprs = [
        pl.col(name).cast(pl.Datetime("us", "UTC"))
        for name, dtype in frame.schema.items()
        if isinstance(dtype, pl.Datetime)
    ]
    return frame.with_columns(exprs) if exprs else frame


def build_league(root: DataStore) -> DataStore:
    """Two clubs, one gameweek of results, and one future gameweek."""
    season = "2024-25"
    root.write(
        "teams",
        _utc(
            pl.DataFrame(
                {
                    "season": [season, season],
                    "team_key": [3, 7],
                    "team_id": [1, 2],
                    "name": ["Arsenal", "Spurs"],
                    "short_name": ["ARS", "TOT"],
                    "available_at": [_START, _START],
                    "source": ["fpl_api", "fpl_api"],
                }
            )
        ),
    )
    root.write(
        "players",
        _utc(
            pl.DataFrame(
                {
                    "season": [season, season],
                    "player_key": [100, 200],
                    "element_id": [10, 20],
                    "team_key": [3, 7],
                    "position": ["MID", "FWD"],
                    "web_name": ["A", "B"],
                    "available_at": [_START, _START],
                    "source": ["fpl_api", "fpl_api"],
                }
            )
        ),
    )
    root.write(
        "gameweeks",
        _utc(
            pl.DataFrame(
                {
                    "season": [season, season],
                    "gw": [1, 2],
                    "deadline_time": [_DEADLINE_GW1, _DEADLINE_GW2],
                    "finished": [True, False],
                    "data_checked": [True, False],
                    "available_at": [_START, _START],
                    "source": ["fpl_api", "fpl_api"],
                }
            )
        ),
    )
    result_at = _KICKOFF_GW1 + timedelta(hours=3)
    root.write(
        "fixtures",
        _utc(
            pl.DataFrame(
                {
                    "season": [season, season, season],
                    "fixture_id": [1, 1, 2],
                    "gw": [1, 1, 2],
                    "team_h_key": [3, 3, 7],
                    "team_a_key": [7, 7, 3],
                    "kickoff_time": [_KICKOFF_GW1, _KICKOFF_GW1, _KICKOFF_GW2],
                    "team_h_score": [None, 2, None],
                    "team_a_score": [None, 0, None],
                    "finished": [False, True, False],
                    "available_at": [_START, result_at, _START],
                    "source": ["fpl_api", "fpl_api", "fpl_api"],
                }
            )
        ),
    )
    root.write(
        "player_status",
        _utc(
            pl.DataFrame(
                {
                    "season": [season, season],
                    "player_key": [100, 200],
                    "status": ["a", "a"],
                    "news": ["", ""],
                    "price": [100, 80],
                    "ep_next": [6.5, 4.0],
                    "chance_of_playing": [100.0, 75.0],
                    "penalties_order": [1, None],
                    "available_at": [_START, _START],
                    "source": ["fpl_api", "fpl_api"],
                }
            )
        ),
    )
    root.write(
        "player_fixture_stats",
        _utc(
            pl.DataFrame(
                {
                    "season": [season, season, season],
                    "gw": [1, 1, 1],
                    "player_key": [100, 200, 100],
                    "fixture_id": [1, 1, 1],
                    "minutes": [90, 90, 90],
                    "total_points": [8, 2, 40],
                    "goals_scored": [1, 0, 5],
                    "assists": [0, 0, 0],
                    "clean_sheets": [1, 0, 1],
                    "goals_conceded": [0, 2, 0],
                    "saves": [0, 0, 0],
                    "bonus": [2, 0, 3],
                    "yellow_cards": [0, 0, 0],
                    "red_cards": [0, 0, 0],
                    "own_goals": [0, 0, 0],
                    "penalties_saved": [0, 0, 0],
                    "penalties_missed": [0, 0, 0],
                    "starts": [1, 1, 1],
                    "influence": [20.0, 5.0, 80.0],
                    "creativity": [10.0, 2.0, 40.0],
                    "threat": [15.0, 1.0, 60.0],
                    "bps": [25, 5, 90],
                    "available_at": [
                        result_at,
                        result_at,
                        _DEADLINE_GW2 + timedelta(days=1),
                    ],
                    "source": ["fpl_api", "fpl_api", "fpl_api"],
                }
            )
        ),
    )
    return root


def dual_write_plan(
    data_dir: Path,
    frame: pl.DataFrame,
    *,
    report: str | None = None,
    code_sha: str = "stub",
    deadline: datetime | None = None,
) -> Path:
    """Write a plan the way ``run_plan`` does: legacy ``plans/`` + a plan run.

    Readers resolve plans from ``runs/`` (data-model cut-over), so tests that
    need "a plan exists" must create the run, not only the legacy parquet.
    The spec is found by ``model_id`` among ``configs/models``.

    Returns:
        The legacy parquet path (``plans/<model_id>/as_of=…/part.parquet``).
    """
    from fplt.models.spec import load_model_specs
    from fplt.optimise.plan import write_plan
    from fplt.store.runs import write_plan_run
    from fplt.types import GWKey

    path = write_plan(data_dir, frame, code_sha=code_sha)
    if report is not None:
        (path.parent / "report.md").write_text(report, encoding="utf-8")
    model_id = str(frame["model_id"][0])
    spec = next(s for s in load_model_specs().values() if s.model_id == model_id)
    write_plan_run(
        data_dir,
        spec=spec,
        as_of=GWKey(str(frame["as_of_season"][0]), int(frame["as_of_gw"][0])),
        deadline_utc=deadline or datetime(2024, 8, 16, 17, 30, tzinfo=UTC),
        horizon=int(frame["step"].n_unique()),
        frame=frame,
        report_md=report,
    )
    return path
