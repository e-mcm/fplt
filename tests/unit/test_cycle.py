"""Catch-up plans a skipped week. The deadline sensor fires inside three hours."""

from __future__ import annotations

import shutil
from datetime import UTC, datetime, timedelta
from pathlib import Path

import dagster as dg
import polars as pl
import pytest

from fplt.automation.cycle import (
    PRODUCTION_MODELS,
    CatchUpSeasonError,
    catch_up,
    current_season,
    due_gameweek,
    pending_checked,
    plans_exist,
    pre_deadline,
    reports,
)
from fplt.automation.production import (
    OfficialLine,
    ProductionOps,
    load_production_ops,
)
from fplt.backtest.ledger import align_ledger_columns
from fplt.data.fpl_client import CurrentTeam
from fplt.data.store import DataStore
from fplt.models.spec import load_model_specs
from fplt.optimise.plan import DEFAULT_PLAN_HORIZON, covered_horizon
from fplt.optimise.state import SquadState
from fplt.orchestration import defs
from fplt.orchestration.assets import recommendation_report
from fplt.orchestration.sensors import catchup_sensor, pre_deadline_sensor
from fplt.schemas.ledger import LedgerSchema
from fplt.settings import get_settings
from fplt.store.runs import write_backtest_run
from fplt.types import GWKey
from tests.builders import dual_write_plan

_WHEN = datetime(2024, 8, 1, tzinfo=UTC)
_GW3 = datetime(2024, 9, 1, 17, tzinfo=UTC)
_ROOT = Path(__file__).resolve().parents[2]
_MODEL = "rolling_mean_v1"


@pytest.fixture(scope="module")
def _league_template(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Build the synthetic league once per worker; tests copy into tmp_path."""
    root = tmp_path_factory.mktemp("cycle_league")
    _league(root)
    return root


@pytest.fixture
def league_dir(tmp_path: Path, _league_template: Path) -> Path:
    """Fresh data dir with the shared synthetic league copied in."""
    dest = tmp_path / "data"
    shutil.copytree(_league_template, dest)
    return dest


def test_production_models_are_dc_layered_v4_native_at_default_horizon() -> None:
    """Unattended plans use native MC production at the live planner horizon."""
    ops = load_production_ops()
    assert ops.models == ["dc_layered_v4_native"]
    assert ops.plan_horizon == 3
    assert PRODUCTION_MODELS == ["dc_layered_v4_native"]
    assert DEFAULT_PLAN_HORIZON == 3
    assert catch_up.__defaults__ is not None
    assert catch_up.__defaults__[-1] is None
    assert pre_deadline.__defaults__ is not None
    assert pre_deadline.__defaults__[-1] is None
    assert pending_checked.__defaults__ is not None
    assert pending_checked.__defaults__[-1] is None
    assert plans_exist.__defaults__ is not None
    assert plans_exist.__defaults__[-1] is None


def test_production_models_is_lazy_not_import_time() -> None:
    """``PRODUCTION_MODELS`` must not be evaluated at cycle import (#362)."""
    import fplt.automation.cycle as cycle

    assert "PRODUCTION_MODELS" not in cycle.__dict__
    assert cycle.PRODUCTION_MODELS == ["dc_layered_v4_native"]


def test_automation_import_tolerates_empty_and_bad_overrides() -> None:
    """Empty/bad production env must not crash ``import fplt.automation`` (#362)."""
    import os
    import subprocess
    import sys

    for environ in (
        {
            "FPLT_PRODUCTION_MODELS": "",
            "FPLT_PLAN_HORIZON": "",
            "PYTEST_CURRENT_TEST": "1",
        },
        {
            "FPLT_PLAN_HORIZON": "nope",
            "PYTEST_CURRENT_TEST": "1",
        },
    ):
        completed = subprocess.run(  # noqa: S603 — controlled interpreter + const argv
            [sys.executable, "-c", "import fplt.automation"],
            check=False,
            capture_output=True,
            text=True,
            env={**os.environ, **environ},
        )
        assert completed.returncode == 0, completed.stderr


def test_a_deadline_is_due_only_inside_three_hours(league_dir: Path) -> None:
    store = DataStore(league_dir)
    assert due_gameweek(store, _GW3 - timedelta(hours=4)) is None
    assert due_gameweek(store, _GW3 + timedelta(minutes=1)) is None
    due = due_gameweek(store, _GW3 - timedelta(hours=1))
    assert due is not None
    assert due.gw == 3
    waiting = pending_checked(store, league_dir, [_MODEL], horizon=1)
    assert [key.gw for key in waiting] == [1, 2]
    assert plans_exist(league_dir, [_MODEL], GWKey("2024-25", 1), 0) is False


def test_catch_up_plans_a_skipped_checked_week(
    league_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Real predict+plan path: catch-up writes reports for skipped checked weeks."""
    monkeypatch.setattr(
        "fplt.automation.cycle.archive_if_missing", lambda *a, **k: None
    )
    done = catch_up(league_dir, [_MODEL], horizon=1)
    assert [key.gw for key in done] == [1, 2]
    assert catch_up(league_dir, [_MODEL], horizon=1) == []
    text = reports(league_dir)
    assert "GW 1" in text
    assert "GW 2" in text
    moment = _GW3 - timedelta(hours=1)
    assert pre_deadline(league_dir, moment, [_MODEL], horizon=1) is not None
    assert pre_deadline(league_dir, moment, [_MODEL], horizon=1) is None


def test_pre_deadline_refreshes_when_availability_signal_is_newer(
    league_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Refresh wiring only: stub predict/plan; assert stale signal triggers rebuild."""
    _stub_predict_and_plan(monkeypatch)
    moment = _GW3 - timedelta(hours=1)
    assert pre_deadline(league_dir, moment, [_MODEL], horizon=1) is not None
    assert pre_deadline(league_dir, moment, [_MODEL], horizon=1) is None

    store = DataStore(league_dir)
    store.write(
        "player_status",
        pl.DataFrame(
            {
                "season": ["2024-25"],
                "player_key": [100],
                "status": ["d"],
                "news": ["late knock"],
                "price": [100],
                "ep_next": [2.0],
                "chance_of_playing": [25.0],
                "penalties_order": [None],
                "available_at": [moment + timedelta(minutes=5)],
                "source": ["fpl_api"],
            }
        ),
    )
    # Plan run finished "now"; force it older than the new status signal.
    monkeypatch.setattr(
        "fplt.automation.late_refresh.latest_plan_finished_at",
        lambda *_a, **_k: moment - timedelta(hours=2),
    )
    later = moment + timedelta(minutes=10)
    assert pre_deadline(league_dir, later, [_MODEL], horizon=1) is not None
    assert (
        pre_deadline(
            league_dir,
            later,
            [_MODEL],
            horizon=1,
            refresh_if_stale=False,
        )
        is None
    )


def test_catch_up_passes_live_squad_when_entry_id_set(
    league_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Entry-id wiring: catch-up must pass the live squad into run_plan."""
    monkeypatch.setenv("FPLT_ENTRY_ID", "42")
    monkeypatch.setenv("FPLT_FPL_SESSION", "cookie")
    get_settings.cache_clear()
    team = CurrentTeam.from_my_team(42, _live_payload())
    seen: list[SquadState | None] = []

    def load_team(**_: object) -> CurrentTeam:
        return team

    monkeypatch.setattr("fplt.optimise.decisions.load_current_team", load_team)

    def load_public(entry_id: int, view: object, settings: object, **_: object):
        from fplt.optimise.public_squad import PublicSquadState
        from fplt.optimise.state import ChipInventory

        return PublicSquadState(
            entry_id=entry_id,
            team=team.model_copy(update={"source": "public"}),
            chips=ChipInventory(),
            last_finished=1,
        )

    monkeypatch.setattr("fplt.optimise.decisions.public_squad_state", load_public)
    _stub_predict_and_plan(monkeypatch, seen=seen)
    try:
        done = catch_up(league_dir, [_MODEL], horizon=1)
        assert [key.gw for key in done] == [1, 2]
        # Regression: stubbed path still must receive a resolved live squad
        # (bank / selling price / 15 owned) — not an empty YAML draft.
        assert len(seen) == 2
        for squad in seen:
            assert squad is not None
            assert squad.bank == 21
            assert len(squad.owned) == 15
            assert squad.owned[0].selling_price == 46
            assert squad.free_transfers == 1
        assert plans_exist(league_dir, [_MODEL], GWKey("2024-25", 1), 1)
        assert plans_exist(league_dir, [_MODEL], GWKey("2024-25", 2), 1)
    finally:
        get_settings.cache_clear()


def test_catch_up_without_entry_id_keeps_yaml_draft(
    league_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """YAML-draft wiring: without entry id, run_plan sees empty owned + bank 1000."""
    monkeypatch.delenv("FPLT_ENTRY_ID", raising=False)
    monkeypatch.delenv("FPLT_FPL_SESSION", raising=False)
    get_settings.cache_clear()
    seen: list[SquadState | None] = []

    def boom(**_: object) -> CurrentTeam:
        raise AssertionError("load_current_team must not run without entry id")

    monkeypatch.setattr("fplt.optimise.decisions.load_current_team", boom)
    _stub_predict_and_plan(monkeypatch, seen=seen)
    try:
        catch_up(league_dir, [_MODEL], horizon=1)
        # Regression: stubbed path must still see YAML draft squad state.
        assert seen
        assert all(squad is not None and squad.owned == [] for squad in seen)
        assert all(squad is not None and squad.bank == 1000 for squad in seen)
        assert plans_exist(league_dir, [_MODEL], GWKey("2024-25", 1), 1)
    finally:
        get_settings.cache_clear()


def _gameweeks_only(root: Path, rows: list[tuple[str, int, bool]]) -> DataStore:
    """Data dir whose only staged table is ``gameweeks`` (season, gw, checked)."""
    store = DataStore(root)
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": [season for season, _, _ in rows],
                "gw": [gw for _, gw, _ in rows],
                "deadline_time": [
                    datetime(int(season[:4]), 8, 1, tzinfo=UTC) + timedelta(weeks=gw)
                    for season, gw, _ in rows
                ],
                "finished": [checked for _, _, checked in rows],
                "data_checked": [checked for _, _, checked in rows],
                "available_at": [_WHEN] * len(rows),
                "source": ["test"] * len(rows),
            }
        ),
    )
    return store


def _ops_with(**update: object) -> ProductionOps:
    return load_production_ops().model_copy(update=update)


def test_pending_checked_ignores_past_seasons(tmp_path: Path) -> None:
    """Regression #369: staged history (e.g. vaastav 2021-22) is never pending.

    Before the fix, every checked GW since 2021-22 counted as a missed live
    week, and catch-up tried to plan 2021-22 GW1 with the 2026-27 squad.
    """
    rows = [("2021-22", gw, True) for gw in range(1, 39)]
    rows += [("2025-26", gw, True) for gw in range(1, 39)]
    rows += [("2026-27", gw, gw <= 12) for gw in range(1, 39)]
    store = _gameweeks_only(tmp_path, rows)
    assert current_season(store) == "2026-27"
    waiting = pending_checked(store, tmp_path, [_MODEL], horizon=1)
    assert waiting == [GWKey("2026-27", gw) for gw in range(1, 13)]


def test_pending_checked_respects_catch_up_from_gw(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = _gameweeks_only(tmp_path, [("2026-27", gw, True) for gw in range(1, 6)])
    monkeypatch.setattr(
        "fplt.automation.cycle.load_production_ops",
        lambda: _ops_with(catch_up_from_gw=4),
    )
    waiting = pending_checked(store, tmp_path, [_MODEL], horizon=1)
    assert waiting == [GWKey("2026-27", 4), GWKey("2026-27", 5)]


def test_pending_checked_empty_gameweeks(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    assert current_season(store) is None
    assert pending_checked(store, tmp_path, [_MODEL], horizon=1) == []


def test_catch_up_is_bounded_per_tick(
    league_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#369: at most max_catch_up_per_tick GWs per call; the rest next tick."""
    monkeypatch.delenv("FPLT_ENTRY_ID", raising=False)
    get_settings.cache_clear()
    _stub_predict_and_plan(monkeypatch)
    monkeypatch.setattr(
        "fplt.automation.cycle.load_production_ops",
        lambda: _ops_with(max_catch_up_per_tick=1),
    )
    try:
        assert catch_up(league_dir, [_MODEL], horizon=1) == [GWKey("2024-25", 1)]
        assert catch_up(league_dir, [_MODEL], horizon=1) == [GWKey("2024-25", 2)]
        assert catch_up(league_dir, [_MODEL], horizon=1) == []
    finally:
        get_settings.cache_clear()


def test_play_refuses_a_past_season(
    league_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression #369: never resolve the live squad against a past season."""
    import fplt.automation.cycle as cycle

    def boom(*_: object, **__: object) -> None:
        raise AssertionError("must refuse before predicting or resolving a squad")

    monkeypatch.setattr("fplt.automation.cycle.run_backtest", boom)
    monkeypatch.setattr("fplt.automation.cycle.resolve_squad_state", boom)
    with pytest.raises(CatchUpSeasonError, match="2023-24"):
        cycle._play(league_dir, [_MODEL], GWKey("2023-24", 1), 1)


def test_stub_plan_writer_marks_plans_exist(league_dir: Path) -> None:
    """Stub helper writes a plan parquet that ``plans_exist`` recognises.

    Guards against a stub that records squad kwargs but forgets to satisfy
    the catch-up / pre-deadline completion check (assertion weakening).
    """
    path = _write_stub_plan(league_dir, _MODEL, "2024-25", 1, 1)
    assert path.exists()
    assert (path.parent / "report.md").read_text(encoding="utf-8")
    assert "GW 1" in (path.parent / "report.md").read_text(encoding="utf-8")
    assert plans_exist(league_dir, [_MODEL], GWKey("2024-25", 1), 1) is True


def test_sensors_request_a_run_and_then_skip(
    league_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use(league_dir, monkeypatch)
    catch = catchup_sensor.evaluate_tick(dg.build_sensor_context())
    assert catch.run_requests[0].run_key == "catchup-2024-25-1"
    moment = _GW3 - timedelta(hours=1)
    monkeypatch.setattr(
        "fplt.orchestration.sensors.datetime",
        type("Clock", (), {"now": staticmethod(lambda _tz: moment)}),
    )
    upcoming = pre_deadline_sensor.evaluate_tick(dg.build_sensor_context())
    assert upcoming.run_requests[0].run_key == "deadline-2024-25-3"
    names = {sensor.name for sensor in defs.sensors}
    assert "notify_run_failure" in names
    get_settings.cache_clear()


def test_deploy_templates_name_the_dagster_processes() -> None:
    web = (_ROOT / "deploy" / "launchd" / "dev.fplt.webserver.plist").read_text()
    daemon = (_ROOT / "deploy" / "launchd" / "dev.fplt.daemon.plist").read_text()
    compose = (_ROOT / "deploy" / "docker-compose.yml").read_text()
    assert "dagster-webserver" in web
    assert "dagster-daemon" in daemon
    assert "dagster-webserver" in compose
    assert "dagster-daemon" in compose


@pytest.mark.slow
def test_materialize_attaches_the_report(
    league_dir: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use(league_dir, monkeypatch)
    # Synthetic league has rolling-mean ledger rows only; keep the asset path
    # under that model and H=1 so the report-attachment check stays cheap.
    from fplt.models.pipeline_spec import live_pipeline_defaults

    monkeypatch.setattr(
        "fplt.automation.cycle.load_production_ops",
        lambda: ProductionOps(
            models=[_MODEL],
            plan_horizon=1,
            pipeline=live_pipeline_defaults(
                model=_MODEL, horizon=1, mc_backend="python"
            ),
            lines=[
                OfficialLine(
                    line="primary",
                    spec=_MODEL,
                    horizon=1,
                    role="primary",
                )
            ],
        ),
    )
    result = dg.materialize([recommendation_report])
    assert result.success
    events = result.get_asset_materialization_events()
    metadata = events[0].event_specific_data.materialization.metadata
    assert "GW 1" in metadata["report"].value
    assert "2024-25 gw1" in metadata["processed"].value
    get_settings.cache_clear()


def _use(data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(data_dir))
    get_settings.cache_clear()


def _stub_predict_and_plan(
    monkeypatch: pytest.MonkeyPatch,
    *,
    seen: list[SquadState | None] | None = None,
) -> None:
    """Skip real backtest/MILP; write a tiny plan so ``plans_exist`` succeeds."""
    monkeypatch.setattr(
        "fplt.automation.cycle.archive_if_missing", lambda *a, **k: None
    )
    monkeypatch.setattr("fplt.automation.cycle.run_backtest", lambda *a, **k: None)

    def fake_run_plan(
        data_dir: Path,
        model_name: str,
        season: str,
        gameweek: int,
        horizon: int,
        entry_json: Path | None = None,
        squad: SquadState | None = None,
        **_: object,
    ) -> tuple[None, Path, str, str]:
        if seen is not None:
            seen.append(squad)
        path = _write_stub_plan(data_dir, model_name, season, gameweek, horizon)
        report = (path.parent / "report.md").read_text(encoding="utf-8")
        return None, path, report, "stub-run"

    monkeypatch.setattr("fplt.automation.cycle.run_plan", fake_run_plan)


def _write_stub_plan(
    data_dir: Path,
    model_name: str,
    season: str,
    gameweek: int,
    horizon: int,
) -> Path:
    """Write a minimal plan parquet (+ report) for one deadline/horizon."""
    model_id = load_model_specs()[model_name].model_id
    weeks = covered_horizon(gameweek, horizon)
    rows = [
        {
            "model_id": model_id,
            "as_of_season": season,
            "as_of_gw": gameweek,
            "target_gw": gameweek + step,
            "step": step,
            "player_key": 1,
            "position": "MID",
            "team_key": 1,
            "in_xi": True,
            "captain": True,
            "vice": False,
            "bought": False,
            "sold": False,
            "xpts": 1.0,
            "hit": 0.0,
        }
        for step in range(weeks)
    ]
    return dual_write_plan(
        data_dir, pl.DataFrame(rows), report=f"# {season}\n\nGW {gameweek}\n"
    )


def _live_payload() -> dict[str, object]:
    # Valid 2/5/5/3 with ≤3 per club among the synthetic league keys.
    elements = [1, 16, 7, 12, 17, 22, 27, 4, 9, 14, 19, 24, 5, 10, 15]
    picks = [
        {
            "element": element,
            "position": index,
            "selling_price": 45 + index,
            "multiplier": 1 if index <= 11 else 0,
            "purchase_price": 45 + index,
            "is_captain": index == 9,
            "is_vice_captain": index == 6,
        }
        for index, element in enumerate(elements, start=1)
    ]
    return {
        "picks": picks,
        "chips": [],
        "transfers": {
            "cost": 4,
            "status": "cost",
            "limit": 1,
            "made": 0,
            "bank": 21,
            "value": 1022,
        },
    }


def _league(root: Path) -> DataStore:
    store = DataStore(root)
    players: list[dict[str, object]] = []
    status: list[dict[str, object]] = []
    player = 1
    for club in range(1, 7):
        for position in ("GK", "DEF", "DEF", "MID", "FWD"):
            players.append(
                {
                    "season": "2024-25",
                    "player_key": player,
                    "element_id": player,
                    "team_key": club,
                    "position": position,
                    "web_name": f"p{player}",
                    "available_at": _WHEN,
                    "source": "test",
                }
            )
            status.append(
                {
                    "season": "2024-25",
                    "player_key": player,
                    "status": "a",
                    "news": "",
                    "price": 40 + club,
                    "ep_next": None,
                    "chance_of_playing": None,
                    "available_at": _WHEN,
                    "source": "test",
                }
            )
            player += 1
    store.write("players", pl.DataFrame(players))
    store.write(
        "player_status",
        pl.DataFrame(status).with_columns(
            pl.col("ep_next").cast(pl.Float64()),
            pl.col("chance_of_playing").cast(pl.Float64()),
        ),
    )
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": ["2024-25", "2024-25", "2024-25"],
                "gw": [1, 2, 3],
                "deadline_time": [
                    datetime(2024, 8, 16, 18, tzinfo=UTC),
                    datetime(2024, 8, 23, 18, tzinfo=UTC),
                    _GW3,
                ],
                "finished": [True, True, False],
                "data_checked": [True, True, False],
                "available_at": [_WHEN, _WHEN, _WHEN],
                "source": ["test", "test", "test"],
            }
        ),
    )
    pairs = [(1, 2), (3, 4), (5, 6)]
    fixture_rows: list[dict[str, object]] = []
    fixture_id = 1
    for gw in (1, 2, 3):
        kickoff = datetime(2024, 8, 16 + 7 * (gw - 1), 14, tzinfo=UTC)
        for home, away in pairs:
            fixture_rows.append(
                {
                    "season": "2024-25",
                    "fixture_id": fixture_id,
                    "gw": gw,
                    "team_h_key": home,
                    "team_a_key": away,
                    "kickoff_time": kickoff,
                    "team_h_score": None,
                    "team_a_score": None,
                    "finished": False,
                    "available_at": _WHEN,
                    "source": "test",
                }
            )
            fixture_id += 1
    store.write(
        "fixtures",
        pl.DataFrame(fixture_rows).with_columns(
            pl.col("team_h_score").cast(pl.Int64()),
            pl.col("team_a_score").cast(pl.Int64()),
        ),
    )
    _predictions(root, players)
    return store


def _predictions(root: Path, players: list[dict[str, object]]) -> None:
    """Seed non-zero scores as a predict run so catch-up can plan.

    The synthetic league has no ``player_fixture_stats``, so walk-forward
    rolling-mean xPts are zero; this run supplies usable rows for ``run_plan``.
    """
    spec = load_model_specs()[_MODEL]
    records: list[dict[str, object]] = []
    for gameweek in (1, 2, 3):
        for row in players:
            records.append(
                {
                    "model_id": spec.model_id,
                    "run_kind": "live",
                    "as_of_season": "2024-25",
                    "as_of_gw": gameweek,
                    "target_season": "2024-25",
                    "target_gw": gameweek,
                    "horizon": 1,
                    "player_key": row["player_key"],
                    "position": row["position"],
                    "fixture_id": row["player_key"],
                    "xpts": float(row["player_key"]),
                    "xpts_sd": None,
                    "p_haul": None,
                    "p_60": None,
                    "actual_pts": None,
                    "actual_minutes": None,
                }
            )
    frame = pl.DataFrame(records).with_columns(
        pl.col("xpts_sd").cast(pl.Float64()),
        pl.col("p_haul").cast(pl.Float64()),
        pl.col("p_60").cast(pl.Float64()),
        pl.col("actual_pts").cast(pl.Int64()),
        pl.col("actual_minutes").cast(pl.Int64()),
    )
    checked = LedgerSchema.validate(align_ledger_columns(frame))
    assert isinstance(checked, pl.DataFrame)
    store = DataStore(root)
    as_ofs = [
        (GWKey("2024-25", gw), store.deadline(GWKey("2024-25", gw))) for gw in (1, 2, 3)
    ]
    write_backtest_run(
        root,
        spec=spec,
        as_ofs=as_ofs,
        horizon=1,
        frame=checked,
        purpose="cycle test seed",
    )
