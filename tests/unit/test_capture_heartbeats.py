"""Capture workflows wire ops-03 / A7 heartbeats and failure alerts."""

from __future__ import annotations

from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOWS = _REPO / ".github" / "workflows"
_ACTIONS = _REPO / ".github" / "actions"

_CAPTURE = (
    "snapshot.yml",
    "understat.yml",
    "api_football.yml",
    "football_data.yml",
)

_PING_SECRETS = {
    "snapshot.yml": "HC_PING_URL_SNAPSHOT",
    "understat.yml": "HC_PING_URL_UNDERSTAT",
    "api_football.yml": "HC_PING_URL_API_FOOTBALL",
    "football_data.yml": "HC_PING_URL_FOOTBALL_DATA",
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


def _capture_steps(doc: dict) -> list[dict]:
    return list(doc["jobs"]["capture"]["steps"])


def test_heartbeat_and_notify_failure_actions_exist() -> None:
    """Composite actions used by capture workflows are present."""
    assert (_ACTIONS / "heartbeat" / "action.yml").is_file()
    assert (_ACTIONS / "notify-failure" / "action.yml").is_file()


def test_capture_workflows_have_simulate_failure_input() -> None:
    """workflow_dispatch exposes simulate_failure for the forced-alert smoke test."""
    for name in _CAPTURE:
        doc = _load(name)
        inputs = _on(doc)["workflow_dispatch"]["inputs"]
        assert "simulate_failure" in inputs, name
        assert inputs["simulate_failure"]["type"] == "boolean"
        assert inputs["simulate_failure"]["default"] is False


def test_alerts_job_runs_after_capture_even_on_failure() -> None:
    """alerts needs capture and uses if: always() so failures still notify."""
    for name in _CAPTURE:
        doc = _load(name)
        alerts = doc["jobs"]["alerts"]
        assert alerts["needs"] == "capture"
        assert alerts["if"] == "always()"


def test_capture_workflows_ping_heartbeat_on_success() -> None:
    """Each alerts job pings its HC_PING_URL_* secret when capture succeeds."""
    for name in _CAPTURE:
        steps = _alert_steps(_load(name))
        heartbeats = [
            s for s in steps if s.get("uses") == "./.github/actions/heartbeat"
        ]
        assert len(heartbeats) == 1, name
        step = heartbeats[0]
        assert step.get("if") == "needs.capture.result == 'success'"
        assert step["with"]["ping_url"] == f"${{{{ secrets.{_PING_SECRETS[name]} }}}}"


def test_capture_workflows_notify_on_failure() -> None:
    """Each alerts job posts one ntfy alert via NTFY_URL when capture fails."""
    for name in _CAPTURE:
        steps = _alert_steps(_load(name))
        alerts = [
            s for s in steps if s.get("uses") == "./.github/actions/notify-failure"
        ]
        assert len(alerts) == 1, name
        step = alerts[0]
        assert step.get("if") == "needs.capture.result == 'failure'"
        assert step["with"]["ntfy_url"] == "${{ secrets.NTFY_URL }}"


def test_simulate_failure_step_runs_only_on_dispatch() -> None:
    """Forced failure is gated so schedules cannot trip it accidentally."""
    for name in _CAPTURE:
        steps = _capture_steps(_load(name))
        sims = [s for s in steps if s.get("name") == "Simulate failure"]
        assert len(sims) == 1, name
        assert "workflow_dispatch" in sims[0]["if"]
        assert "simulate_failure" in sims[0]["if"]


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
