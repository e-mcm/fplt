"""Catch-up plans a skipped week. The deadline sensor fires inside three hours."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import dagster as dg
import polars as pl
import pytest

from fplt.automation.cycle import (
    PRODUCTION_MODELS,
    catch_up,
    due_gameweek,
    pending_checked,
    plans_exist,
    pre_deadline,
    reports,
)
from fplt.automation.production import ProductionOps, load_production_ops
from fplt.backtest.ledger import append_predictions
from fplt.data.fpl_client import CurrentTeam
from fplt.data.store import DataStore
from fplt.models.spec import load_model_specs
from fplt.optimise.plan import DEFAULT_PLAN_HORIZON
from fplt.optimise.state import SquadState
from fplt.orchestration import defs
from fplt.orchestration.assets import recommendation_report
from fplt.orchestration.sensors import catchup_sensor, pre_deadline_sensor
from fplt.settings import get_settings
from fplt.types import GWKey

_WHEN = datetime(2024, 8, 1, tzinfo=UTC)
_GW3 = datetime(2024, 9, 1, 17, tzinfo=UTC)
_ROOT = Path(__file__).resolve().parents[2]


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


def test_a_deadline_is_due_only_inside_three_hours(tmp_path: Path) -> None:
    store = _league(tmp_path)
    assert due_gameweek(store, _GW3 - timedelta(hours=4)) is None
    assert due_gameweek(store, _GW3 + timedelta(minutes=1)) is None
    due = due_gameweek(store, _GW3 - timedelta(hours=1))
    assert due is not None
    assert due.gw == 3
    waiting = pending_checked(store, tmp_path, ["rolling_mean_v1"], horizon=1)
    assert [key.gw for key in waiting] == [1, 2]
    assert plans_exist(tmp_path, ["rolling_mean_v1"], GWKey("2024-25", 1), 0) is False


def test_catch_up_plans_a_skipped_checked_week(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _league(tmp_path)
    monkeypatch.setattr(
        "fplt.automation.cycle.archive_if_missing", lambda *a, **k: None
    )
    done = catch_up(tmp_path, ["rolling_mean_v1"], horizon=1)
    assert [key.gw for key in done] == [1, 2]
    assert catch_up(tmp_path, ["rolling_mean_v1"], horizon=1) == []
    text = reports(tmp_path)
    assert "GW 1" in text
    assert "GW 2" in text
    moment = _GW3 - timedelta(hours=1)
    assert pre_deadline(tmp_path, moment, ["rolling_mean_v1"], horizon=1) is not None
    assert pre_deadline(tmp_path, moment, ["rolling_mean_v1"], horizon=1) is None


def test_pre_deadline_refreshes_when_availability_signal_is_newer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _league(tmp_path)
    monkeypatch.setattr(
        "fplt.automation.cycle.archive_if_missing", lambda *a, **k: None
    )
    moment = _GW3 - timedelta(hours=1)
    assert pre_deadline(tmp_path, moment, ["rolling_mean_v1"], horizon=1) is not None
    assert pre_deadline(tmp_path, moment, ["rolling_mean_v1"], horizon=1) is None

    store = DataStore(tmp_path)
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
    # Plan mtime is "now"; force it older than the new status signal.
    monkeypatch.setattr(
        "fplt.automation.late_refresh.artifact_mtime",
        lambda _path: moment - timedelta(hours=2),
    )
    later = moment + timedelta(minutes=10)
    assert pre_deadline(tmp_path, later, ["rolling_mean_v1"], horizon=1) is not None
    assert (
        pre_deadline(
            tmp_path,
            later,
            ["rolling_mean_v1"],
            horizon=1,
            refresh_if_stale=False,
        )
        is None
    )


def test_catch_up_passes_live_squad_when_entry_id_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _league(tmp_path)
    monkeypatch.setenv("FPLT_ENTRY_ID", "42")
    monkeypatch.setenv("FPLT_FPL_SESSION", "cookie")
    get_settings.cache_clear()
    team = CurrentTeam.from_my_team(42, _live_payload())
    seen: list[SquadState | None] = []

    def load_team(**_: object) -> CurrentTeam:
        return team

    def spy_plan(
        data_dir: Path,
        model_name: str,
        season: str,
        gameweek: int,
        horizon: int,
        entry_json: Path | None = None,
        squad: SquadState | None = None,
    ):
        seen.append(squad)
        from fplt.optimise.decisions import run_plan as real

        return real(
            data_dir,
            model_name,
            season,
            gameweek,
            horizon,
            entry_json=entry_json,
            squad=squad,
        )

    monkeypatch.setattr("fplt.optimise.decisions.load_current_team", load_team)
    monkeypatch.setattr("fplt.automation.cycle.run_plan", spy_plan)
    try:
        done = catch_up(tmp_path, ["rolling_mean_v1"], horizon=1)
        assert [key.gw for key in done] == [1, 2]
        assert len(seen) == 2
        for squad in seen:
            assert squad is not None
            assert squad.bank == 21
            assert len(squad.owned) == 15
            assert squad.owned[0].selling_price == 46
            assert squad.free_transfers == 1
    finally:
        get_settings.cache_clear()


def test_catch_up_without_entry_id_keeps_yaml_draft(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _league(tmp_path)
    monkeypatch.delenv("FPLT_ENTRY_ID", raising=False)
    monkeypatch.delenv("FPLT_FPL_SESSION", raising=False)
    get_settings.cache_clear()
    seen: list[SquadState | None] = []

    def boom(**_: object) -> CurrentTeam:
        raise AssertionError("load_current_team must not run without entry id")

    def spy_plan(
        data_dir: Path,
        model_name: str,
        season: str,
        gameweek: int,
        horizon: int,
        entry_json: Path | None = None,
        squad: SquadState | None = None,
    ):
        seen.append(squad)
        from fplt.optimise.decisions import run_plan as real

        return real(
            data_dir,
            model_name,
            season,
            gameweek,
            horizon,
            entry_json=entry_json,
            squad=squad,
        )

    monkeypatch.setattr("fplt.optimise.decisions.load_current_team", boom)
    monkeypatch.setattr("fplt.automation.cycle.run_plan", spy_plan)
    try:
        catch_up(tmp_path, ["rolling_mean_v1"], horizon=1)
        assert seen
        assert all(squad is not None and squad.owned == [] for squad in seen)
        assert all(squad is not None and squad.bank == 1000 for squad in seen)
    finally:
        get_settings.cache_clear()


def test_sensors_request_a_run_and_then_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use(tmp_path, monkeypatch)
    _league(tmp_path)
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
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use(tmp_path, monkeypatch)
    _league(tmp_path)
    # Synthetic league has rolling-mean ledger rows only; keep the asset path
    # under that model and H=1 so the report-attachment check stays cheap.
    from fplt.models.pipeline_spec import live_pipeline_defaults

    monkeypatch.setattr(
        "fplt.automation.cycle.load_production_ops",
        lambda: ProductionOps(
            models=["rolling_mean_v1"],
            plan_horizon=1,
            pipeline=live_pipeline_defaults(
                model="rolling_mean_v1", horizon=1, mc_backend="python"
            ),
        ),
    )
    result = dg.materialize([recommendation_report])
    assert result.success
    events = result.get_asset_materialization_events()
    metadata = events[0].event_specific_data.materialization.metadata
    assert "GW 1" in metadata["report"].value
    assert "2024-25 gw1" in metadata["processed"].value
    get_settings.cache_clear()


def _use(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()


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
    """Give each checked week a non-zero score so catch-up can plan."""
    model_id = load_model_specs()["rolling_mean_v1"].model_id
    records: list[dict[str, object]] = []
    for gameweek in (1, 2, 3):
        for row in players:
            records.append(
                {
                    "model_id": model_id,
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
    append_predictions(root, frame)
