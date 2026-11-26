"""Scheduled workflows wire ops-03 heartbeats and failure alerts."""

from __future__ import annotations

import subprocess
import textwrap
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOWS = _REPO / ".github" / "workflows"
_ACTIONS = _REPO / ".github" / "actions"

# Every scheduled (or ops dispatch) workflow that must ping + alert.
_SCHEDULED = (
    "snapshot.yml",
    "understat.yml",
    "api_football.yml",
    "football_data.yml",
    "ops-tick.yml",
    "ops-catchup.yml",
    "raw_parity.yml",
    "secret-rotation-reminder.yml",
    "auth-check.yml",
)

_MAIN_JOB = {
    "snapshot.yml": "capture",
    "understat.yml": "capture",
    "api_football.yml": "capture",
    "football_data.yml": "capture",
    "ops-tick.yml": "tick",
    "ops-catchup.yml": "catchup",
    "raw_parity.yml": "parity",
    "secret-rotation-reminder.yml": "remind",
    "auth-check.yml": "check",
}

_PING_SECRETS = {
    "snapshot.yml": "HC_PING_URL_SNAPSHOT",
    "understat.yml": "HC_PING_URL_UNDERSTAT",
    "api_football.yml": "HC_PING_URL_API_FOOTBALL",
    "football_data.yml": "HC_PING_URL_FOOTBALL_DATA",
    "ops-tick.yml": "HC_PING_URL_OPS_TICK",
    "ops-catchup.yml": "HC_PING_URL_OPS_CATCHUP",
    "raw_parity.yml": "HC_PING_URL_RAW_PARITY",
    "secret-rotation-reminder.yml": "HC_PING_URL_SECRET_ROTATION",
    "auth-check.yml": "HC_PING_URL_AUTH_CHECK",
}


def _load(name: str) -> dict:
    return yaml.safe_load((_WORKFLOWS / name).read_text(encoding="utf-8"))


def _on(doc: dict) -> dict:
    # PyYAML 1.1 turns the key ``on`` into ``True``.
    value = doc.get("on", doc.get(True))
    assert isinstance(value, dict), "workflow missing on:"
    return value


def _alert_steps(doc: dict) -> list[dict]:
    return list(doc["jobs"]["alerts"]["steps"])


def _main_steps(doc: dict, name: str) -> list[dict]:
    return list(doc["jobs"][_MAIN_JOB[name]]["steps"])


def _action_run_script(action: str) -> str:
    """Return the composite action's bash ``run:`` body."""
    doc = yaml.safe_load((_ACTIONS / action / "action.yml").read_text(encoding="utf-8"))
    step = doc["runs"]["steps"][0]
    assert step["shell"] == "bash"
    return str(step["run"])


def test_heartbeat_and_notify_failure_actions_exist() -> None:
    """Composite actions used by scheduled workflows are present."""
    assert (_ACTIONS / "heartbeat" / "action.yml").is_file()
    assert (_ACTIONS / "notify-failure" / "action.yml").is_file()


def test_action_inputs_are_optional_for_missing_secrets() -> None:
    """Ping / ntfy inputs must not be required so absent secrets soft-skip."""
    for action in ("heartbeat", "notify-failure"):
        doc = yaml.safe_load(
            (_ACTIONS / action / "action.yml").read_text(encoding="utf-8")
        )
        for name, spec in doc["inputs"].items():
            assert spec.get("required") is False, f"{action}.{name} must be optional"
            assert spec.get("default") == "", f"{action}.{name} default must be empty"


def test_heartbeat_soft_skips_when_ping_url_unset() -> None:
    """Empty PING_URL exits 0 with a soft-skip log (no network)."""
    script = _action_run_script("heartbeat")
    result = subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={"PING_URL": ""},
    )
    assert result.returncode == 0
    assert "soft skip" in result.stdout


def test_push_raw_soft_skips_when_r2_secrets_unset(tmp_path: Path) -> None:
    """Empty R2_* / DATA_BUCKET exits 0 with a soft-skip log (no rclone)."""
    doc = yaml.safe_load(
        (_ACTIONS / "push-raw" / "action.yml").read_text(encoding="utf-8")
    )
    gate = doc["runs"]["steps"][0]
    assert gate["name"] == "Gate on R2 credentials"
    script = str(gate["run"])
    env = {
        "R2_ACCOUNT_ID": "",
        "R2_ACCESS_KEY_ID": "",
        "R2_SECRET_ACCESS_KEY": "",
        "DATA_BUCKET": "fplt-data",
        "GITHUB_OUTPUT": str(tmp_path / "out"),
    }
    (tmp_path / "out").write_text("", encoding="utf-8")
    result = subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode == 0
    assert "soft skip" in result.stdout
    assert "skip=true" in (tmp_path / "out").read_text(encoding="utf-8")


def test_notify_failure_soft_skips_when_ntfy_url_unset() -> None:
    """Empty NTFY_URL exits 0 with a soft-skip log (no network)."""
    script = _action_run_script("notify-failure")
    result = subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env={
            "NTFY_URL": "",
            "WORKFLOW": "snapshot",
            "JOB": "alerts",
            "RUN_URL": "https://example.test/run/1",
        },
    )
    assert result.returncode == 0
    assert "soft skip" in result.stdout


def test_scheduled_workflows_have_simulate_failure_input() -> None:
    """workflow_dispatch exposes simulate_failure for the forced-alert smoke test."""
    for name in _SCHEDULED:
        doc = _load(name)
        inputs = _on(doc)["workflow_dispatch"]["inputs"]
        assert "simulate_failure" in inputs, name
        assert inputs["simulate_failure"]["type"] == "boolean"
        assert inputs["simulate_failure"]["default"] is False


def test_alerts_job_runs_after_main_even_on_failure() -> None:
    """alerts needs the main job and uses if: always() so failures still notify."""
    for name in _SCHEDULED:
        doc = _load(name)
        alerts = doc["jobs"]["alerts"]
        assert alerts["needs"] == _MAIN_JOB[name], name
        assert alerts["if"] == "always()"


def test_scheduled_workflows_ping_heartbeat_on_success() -> None:
    """Each alerts job pings its HC_PING_URL_* secret when the main job succeeds."""
    for name in _SCHEDULED:
        steps = _alert_steps(_load(name))
        heartbeats = [
            s for s in steps if s.get("uses") == "./.github/actions/heartbeat"
        ]
        assert len(heartbeats) == 1, name
        step = heartbeats[0]
        main = _MAIN_JOB[name]
        assert step.get("if") == f"needs.{main}.result == 'success'"
        assert step["with"]["ping_url"] == f"${{{{ secrets.{_PING_SECRETS[name]} }}}}"


def test_scheduled_workflows_notify_on_failure() -> None:
    """Each alerts job posts one ntfy alert via NTFY_URL when the main job fails."""
    for name in _SCHEDULED:
        steps = _alert_steps(_load(name))
        alerts = [
            s for s in steps if s.get("uses") == "./.github/actions/notify-failure"
        ]
        assert len(alerts) == 1, name
        step = alerts[0]
        main = _MAIN_JOB[name]
        assert step.get("if") == f"needs.{main}.result == 'failure'"
        assert step["with"]["ntfy_url"] == "${{ secrets.NTFY_URL }}"


def test_simulate_failure_step_runs_only_on_dispatch() -> None:
    """Forced failure is gated so schedules cannot trip it accidentally."""
    for name in _SCHEDULED:
        steps = _main_steps(_load(name), name)
        sims = [s for s in steps if s.get("name") == "Simulate failure"]
        assert len(sims) == 1, name
        cond = sims[0]["if"]
        assert "simulate_failure" in cond
        # ops-catchup is dispatch-only, so the workflow_dispatch guard is optional.
        if name != "ops-catchup.yml":
            assert "workflow_dispatch" in cond


def test_actions_never_echo_secret_env_names_in_run_scripts() -> None:
    """Heartbeat and notify-failure scripts must not echo ping/ntfy URLs."""
    for action in ("heartbeat", "notify-failure"):
        text = (_ACTIONS / action / "action.yml").read_text(encoding="utf-8")
        for line in text.splitlines():
            stripped = line.strip()
            if stripped.startswith("#"):
                continue
            assert 'echo "$PING_URL"' not in stripped
            assert 'echo "$NTFY_URL"' not in stripped
            assert "echo $PING_URL" not in stripped
            assert "echo $NTFY_URL" not in stripped


def test_deploy_readme_documents_checks_and_runbook_choice() -> None:
    """deploy/README.md lists check periods/grace and records runbook vs keepalive."""
    text = (_REPO / "deploy" / "README.md").read_text(encoding="utf-8")
    for name in (
        "fplt-snapshot",
        "fplt-understat",
        "fplt-api-football",
        "fplt-football-data",
        "fplt-ops-tick",
        "fplt-raw-parity",
        "fplt-ops-catchup",
        "fplt-secret-rotation",
        "fplt-auth-check",
    ):
        assert name in text, name
    assert "15 min" in text
    assert "45 min" in text
    assert "soft skip" in text.lower() or "soft skip" in text
    assert "runbook, not keepalive" in text
    assert "Do not add a keepalive" not in text or "no keepalive workflow" in text


def test_every_scheduled_workflow_file_is_covered() -> None:
    """Guardrail: any new cron workflow must appear in _SCHEDULED."""
    scheduled: list[str] = []
    for path in sorted(_WORKFLOWS.glob("*.yml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        on = _on(doc)
        if "schedule" in on:
            scheduled.append(path.name)
    expected = {n for n in _SCHEDULED if n != "ops-catchup.yml"}
    assert set(scheduled) == expected, textwrap.dedent(
        f"""
        scheduled workflows drifted:
          on disk: {sorted(scheduled)}
          tested:  {sorted(expected)}
        """
    )
