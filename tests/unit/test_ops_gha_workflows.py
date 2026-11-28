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


def test_ops_tick_checks_secrets_before_run() -> None:
    """ops-tick preflight fails empty secrets before Run ops-tick (#362)."""
    steps = _load("ops-tick.yml")["jobs"]["tick"]["steps"]
    names = [s.get("name") for s in steps]
    assert "Check required secrets are non-empty" in names
    assert "Run ops-tick" in names
    assert names.index("Check required secrets are non-empty") < names.index(
        "Run ops-tick"
    )
    check = next(
        s for s in steps if s.get("name") == "Check required secrets are non-empty"
    )
    assert check.get("uses") == "./.github/actions/check-nonempty-env"
    assert "FPLT_ENTRY_ID" in check["with"]["names"]
    action = (_ACTIONS / "check-nonempty-env" / "action.yml").read_text(
        encoding="utf-8"
    )
    assert "::error::" in action
    assert "never echo" in action.lower() or "Never echoes" in action


def test_r2_environment_jobs_check_nonempty_secrets() -> None:
    """Every environment:r2 job that reads R2 secrets has the preflight (#362)."""
    for path in sorted(_WORKFLOWS.glob("*.yml")):
        doc = _load(path.name)
        for job_id, body in doc["jobs"].items():
            if body.get("environment") != "r2":
                continue
            dumped = yaml.safe_dump(body)
            if "secrets.R2_" not in dumped:
                continue
            steps = body.get("steps") or []
            assert any(
                isinstance(s, dict)
                and s.get("uses") == "./.github/actions/check-nonempty-env"
                for s in steps
            ), f"{path.name}::{job_id}"


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


def test_dual_write_capture_workflows_untouched() -> None:
    """Foundation must not strip branch+R2 dual-write from capture jobs."""
    for name in (
        "snapshot.yml",
        "understat.yml",
        "api_football.yml",
        "football_data.yml",
        "vaastav.yml",
    ):
        text = (_WORKFLOWS / name).read_text(encoding="utf-8")
        assert "push-raw" in text, name
        assert "Publish to the snapshots branch" in text, name
        # Dot-path guard must run before the worktree publish (#339 / #350).
        assert "reject_raw_dot_paths.py" in text, name
        assert "git worktree" in text, name
        guard_at = text.index("reject_raw_dot_paths.py")
        worktree_at = text.index("git worktree")
        assert guard_at < worktree_at, name


def test_vaastav_workflow_keeps_cache_outside_raw() -> None:
    """vaastav.yml passes --cache-dir under RUNNER_TEMP, not under raw/ (#339)."""
    text = (_WORKFLOWS / "vaastav.yml").read_text(encoding="utf-8")
    assert '--cache-dir "$RUNNER_TEMP/vaastav_cache"' in text
    assert '--status-dir "$RUNNER_TEMP/status"' in text
    assert '--out "$RUNNER_TEMP/raw"' in text
    # Publish copies only non-dot top-level entries (#339 / #350).
    assert 'for entry in "$RUNNER_TEMP/raw"/*' in text


def test_vaastav_capture_timeout_is_90() -> None:
    """Full vaastav mirror needs headroom beyond the default 30 (#353)."""
    doc = _load("vaastav.yml")
    assert doc["jobs"]["capture"]["timeout-minutes"] == 90


def test_capture_status_pushed_only_after_successful_upload() -> None:
    """Status push runs after Upload and only when uploaded=true (#353)."""
    for name in (
        "vaastav.yml",
        "understat.yml",
        "api_football.yml",
        "football_data.yml",
    ):
        doc = _load(name)
        steps = doc["jobs"]["capture"]["steps"]
        names = [s.get("name") for s in steps]
        assert "Upload to R2" in names, name
        assert "Push capture status" in names, name
        assert names.index("Upload to R2") < names.index("Push capture status"), name
        status = next(s for s in steps if s.get("name") == "Push capture status")
        assert status.get("if") == "steps.upload.outputs.uploaded == 'true'", name
