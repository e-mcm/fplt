"""ops-tick / ops-catchup GHA workflows: schedule, concurrency, heartbeats."""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOWS = _REPO / ".github" / "workflows"


def _load(name: str) -> dict:
    return yaml.safe_load((_WORKFLOWS / name).read_text(encoding="utf-8"))


def _on(doc: dict) -> dict:
    value = doc.get("on", doc.get(True))
    assert isinstance(value, dict), "workflow missing on:"
    return value


def test_ops_tick_workflow_has_offset_schedule_and_dispatch() -> None:
    doc = _load("ops-tick.yml")
    on = _on(doc)
    crons = [entry["cron"] for entry in on["schedule"]]
    assert "7,22,37,52 * * * *" in crons
    assert "workflow_dispatch" in on
    assert on["workflow_dispatch"]["inputs"]["simulate_failure"]["type"] == "boolean"


def test_ops_catchup_workflow_is_dispatch_only() -> None:
    doc = _load("ops-catchup.yml")
    on = _on(doc)
    assert "workflow_dispatch" in on
    assert "schedule" not in on


def test_ops_workflows_share_concurrency_group() -> None:
    for name in ("ops-tick.yml", "ops-catchup.yml"):
        doc = _load(name)
        assert doc["concurrency"]["group"] == "ops"
        assert doc["concurrency"]["cancel-in-progress"] is False


def test_ops_workflows_have_timeouts() -> None:
    tick = _load("ops-tick.yml")
    assert tick["jobs"]["tick"]["timeout-minutes"] == 90
    assert tick["jobs"]["alerts"]["timeout-minutes"] == 5
    catchup = _load("ops-catchup.yml")
    assert catchup["jobs"]["catchup"]["timeout-minutes"] == 180
    assert catchup["jobs"]["alerts"]["timeout-minutes"] == 5


def test_ops_workflows_gate_on_ops_live_and_call_cli() -> None:
    tick = _load("ops-tick.yml")
    tick_steps = tick["jobs"]["tick"]["steps"]
    assert any(s.get("name") == "Gate on OPS_LIVE" for s in tick_steps)
    run_step = next(s for s in tick_steps if s.get("name") == "Run ops-tick")
    assert "fplt ops-tick" in run_step["run"]
    assert "OPS_LIVE" in next(
        s["run"] for s in tick_steps if s.get("name") == "Gate on OPS_LIVE"
    )

    catchup = _load("ops-catchup.yml")
    catchup_steps = catchup["jobs"]["catchup"]["steps"]
    run_catchup = next(s for s in catchup_steps if s.get("name") == "Run ops-catchup")
    assert "fplt ops-catchup" in run_catchup["run"]


def test_ops_workflows_ping_heartbeat_and_notify_failure() -> None:
    for name, job in (("ops-tick.yml", "tick"), ("ops-catchup.yml", "catchup")):
        doc = _load(name)
        alerts = doc["jobs"]["alerts"]
        assert alerts["needs"] == job
        assert alerts["if"] == "always()"
        steps = alerts["steps"]
        heartbeats = [
            s for s in steps if s.get("uses") == "./.github/actions/heartbeat"
        ]
        failures = [
            s for s in steps if s.get("uses") == "./.github/actions/notify-failure"
        ]
        assert len(heartbeats) == 1, name
        assert len(failures) == 1, name
        assert failures[0]["with"]["ntfy_url"] == "${{ secrets.NTFY_URL }}"


def test_ops_tick_heartbeat_secret_is_ops_tick() -> None:
    steps = _load("ops-tick.yml")["jobs"]["alerts"]["steps"]
    heartbeat = next(s for s in steps if s.get("uses") == "./.github/actions/heartbeat")
    assert heartbeat["with"]["ping_url"] == "${{ secrets.HC_PING_URL_OPS_TICK }}"


def test_ops_catchup_heartbeat_secret_is_ops_catchup() -> None:
    steps = _load("ops-catchup.yml")["jobs"]["alerts"]["steps"]
    heartbeat = next(s for s in steps if s.get("uses") == "./.github/actions/heartbeat")
    assert heartbeat["with"]["ping_url"] == "${{ secrets.HC_PING_URL_OPS_CATCHUP }}"


def test_dual_write_capture_workflows_untouched() -> None:
    """Foundation must not strip branch+R2 dual-write from capture jobs."""
    for name in (
        "snapshot.yml",
        "understat.yml",
        "api_football.yml",
        "football_data.yml",
    ):
        text = (_WORKFLOWS / name).read_text(encoding="utf-8")
        assert "push-raw" in text, name
        assert "Publish to the snapshots branch" in text, name
