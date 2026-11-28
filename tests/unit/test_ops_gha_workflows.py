"""ops-tick / ops-catchup / release GHA workflows: schedule, prod, gates."""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOWS = _REPO / ".github" / "workflows"
_ACTIONS = _REPO / ".github" / "actions"


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


def test_ops_workflows_export_job_started_at() -> None:
    for name, job in (("ops-tick.yml", "tick"), ("ops-catchup.yml", "catchup")):
        steps = _load(name)["jobs"][job]["steps"]
        record = steps[0]
        assert record.get("name") == "Record job start"
        assert "JOB_STARTED_AT=" in record["run"]
        run_step = next(
            s for s in steps if str(s.get("name", "")).startswith("Run ops-")
        )
        assert (
            run_step.get("env", {}).get("JOB_STARTED_AT") == "${{ env.JOB_STARTED_AT }}"
        )


def test_ops_workflows_checkout_prod_with_bootstrap() -> None:
    for name, job in (("ops-tick.yml", "tick"), ("ops-catchup.yml", "catchup")):
        steps = _load(name)["jobs"][job]["steps"]
        assert any(s.get("uses") == "./.github/actions/checkout-prod" for s in steps), (
            name
        )
    action = (_ACTIONS / "checkout-prod" / "action.yml").read_text(encoding="utf-8")
    assert "refs/tags/prod" in action
    assert "bootstrapping" in action
    assert "FPLT_PROD_TAG=prod" in action


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


def test_release_workflow_refuses_on_red_ci_freeze_or_smoke() -> None:
    doc = _load("release.yml")
    on = _on(doc)
    assert "workflow_dispatch" in on
    assert "ref" in on["workflow_dispatch"]["inputs"]
    text = (_WORKFLOWS / "release.yml").read_text(encoding="utf-8")
    assert "refusing release: CI is red" in text
    assert "fplt release" in text
    assert "ci-status" in text
    job = doc["jobs"]["release"]
    assert job["timeout-minutes"] == 30
    step_names = [s.get("name") for s in job["steps"]]
    assert "Require CI green on candidate" in step_names
    assert "Run freeze + smoke gates" in step_names


def test_snapshot_curate_job_after_capture() -> None:
    """#345: curate needs capture success, own concurrency, fplt data curate."""
    doc = _load("snapshot.yml")
    curate = doc["jobs"]["curate"]
    assert curate["needs"] == "capture"
    assert curate["if"] == "always() && needs.capture.result == 'success'"
    assert curate["environment"] == "r2"
    assert curate["timeout-minutes"] == 60
    assert curate["concurrency"]["group"] == "curate"
    assert curate["concurrency"]["cancel-in-progress"] is False
    steps = curate["steps"]
    cache = next(s for s in steps if s.get("name") == "Restore raw cache")
    assert cache["with"]["path"] == "data/raw"
    assert "curate-raw-" in cache["with"]["key"]
    run_step = next(s for s in steps if s.get("name") == "Curate")
    assert "fplt data curate" in run_step["run"]
    assert "--source bucket" in run_step["run"]
    assert run_step["env"]["FPLT_DATA_BUCKET"] == "${{ vars.DATA_BUCKET }}"
    # Capture alerts stay independent of curate (curate failure never blocks capture).
    assert doc["jobs"]["alerts"]["needs"] == "capture"


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
