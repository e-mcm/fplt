"""Catch-up plans a skipped week. The deadline sensor fires inside three hours."""

from datetime import UTC, datetime, timedelta
from pathlib import Path
from urllib.request import Request

import dagster as dg
import polars as pl
import pytest

from fplt.automation.cycle import (
    catch_up,
    due_gameweek,
    pending_checked,
    pre_deadline,
    reports,
)
from fplt.automation.notify import notify
from fplt.data.store import DataStore
from fplt.orchestration import defs
from fplt.orchestration.assets import recommendation_report
from fplt.orchestration.sensors import catchup_sensor, pre_deadline_sensor
from fplt.settings import get_settings

_WHEN = datetime(2024, 8, 1, tzinfo=UTC)
_GW3 = datetime(2024, 9, 1, 17, tzinfo=UTC)
_ROOT = Path(__file__).resolve().parents[2]


def test_notify_posts_only_when_a_url_is_set() -> None:
    sent: list[Request] = []

    def post(request: Request) -> object:
        sent.append(request)
        return None

    assert notify(None, "title", "body", post) is False
    assert notify("", "title", "body", post) is False
    assert sent == []
    assert notify("https://ntfy.sh/fplt", "title", "hello", post) is True
    assert sent[0].data == b"hello"
    assert sent[0].get_header("Title") == "title"


def test_a_deadline_is_due_only_inside_three_hours(tmp_path: Path) -> None:
    store = _league(tmp_path)
    assert due_gameweek(store, _GW3 - timedelta(hours=4)) is None
    assert due_gameweek(store, _GW3 + timedelta(minutes=1)) is None
    due = due_gameweek(store, _GW3 - timedelta(hours=1))
    assert due is not None
    assert due.gw == 3
    waiting = pending_checked(store, tmp_path, ["rolling_mean_v1"])
    assert [key.gw for key in waiting] == [1, 2]


def test_catch_up_plans_a_skipped_checked_week(tmp_path: Path) -> None:
    _league(tmp_path)
    done = catch_up(tmp_path, ["rolling_mean_v1"], horizon=1)
    assert [key.gw for key in done] == [1, 2]
    assert catch_up(tmp_path, ["rolling_mean_v1"], horizon=1) == []
    text = reports(tmp_path)
    assert "GW 1" in text
    assert "GW 2" in text
    moment = _GW3 - timedelta(hours=1)
    assert pre_deadline(tmp_path, moment, ["rolling_mean_v1"], horizon=1) is not None
    assert pre_deadline(tmp_path, moment, ["rolling_mean_v1"], horizon=1) is None


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
    return store
