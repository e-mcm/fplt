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
    "vaastav.yml",
    "ops-tick.yml",
    "ops-catchup.yml",
    "raw_parity.yml",
    "curate-full.yml",
    "secret-rotation-reminder.yml",
    "auth-check.yml",
    "bench.yml",
)

_MAIN_JOB = {
    "snapshot.yml": "capture",
    "understat.yml": "capture",
    "api_football.yml": "capture",
    "football_data.yml": "capture",
    "vaastav.yml": "capture",
    "ops-tick.yml": "tick",
    "ops-catchup.yml": "catchup",
    "raw_parity.yml": "parity",
    "curate-full.yml": "curate_full",
    "secret-rotation-reminder.yml": "remind",
    "auth-check.yml": "check",
    "bench.yml": "bench",
}

_PING_SECRETS = {
    "snapshot.yml": "HC_PING_URL_SNAPSHOT",
    "understat.yml": "HC_PING_URL_UNDERSTAT",
    "api_football.yml": "HC_PING_URL_API_FOOTBALL",
    "football_data.yml": "HC_PING_URL_FOOTBALL_DATA",
    "vaastav.yml": "HC_PING_URL_VAASTAV",
    "ops-tick.yml": "HC_PING_URL_OPS_TICK",
    "ops-catchup.yml": "HC_PING_URL_OPS_CATCHUP",
    "raw_parity.yml": "HC_PING_URL_RAW_PARITY",
    "curate-full.yml": "HC_PING_URL_CURATE_FULL",
    "secret-rotation-reminder.yml": "HC_PING_URL_SECRET_ROTATION",
    "auth-check.yml": "HC_PING_URL_AUTH_CHECK",
    "bench.yml": "HC_PING_URL_BENCH",
}

# Capture workflows that dual-write via push-raw: heartbeat only when uploaded.
_PUSH_RAW_CAPTURE = frozenset(
    {
        "snapshot.yml",
        "understat.yml",
        "api_football.yml",
        "football_data.yml",
        "vaastav.yml",
    }
)


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


def _push_raw_gate_script() -> str:
    """Return the bash body of the push-raw R2 credential gate."""
    doc = yaml.safe_load(
        (_ACTIONS / "push-raw" / "action.yml").read_text(encoding="utf-8")
    )
    gate = doc["runs"]["steps"][0]
    assert gate["name"] == "Gate on R2 credentials"
    return str(gate["run"])


def test_push_raw_soft_skips_when_r2_secrets_unset_off_main(tmp_path: Path) -> None:
    """Empty R2_* off main exits 0 with a soft-skip log (no rclone)."""
    script = _push_raw_gate_script()
    env = {
        "R2_ACCOUNT_ID": "",
        "R2_ACCESS_KEY_ID": "",
        "R2_SECRET_ACCESS_KEY": "",
        "DATA_BUCKET": "fplt-data",
        "GITHUB_REF": "refs/heads/cursor/feature",
        "GITHUB_OUTPUT": str(tmp_path / "out"),
        "PATH": "/usr/bin:/bin",
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
    out = (tmp_path / "out").read_text(encoding="utf-8")
    assert "skip=true" in out
    # uploaded is set by the result step after the gate, not here (#340).
    assert "uploaded=" not in out


def test_push_raw_fails_on_main_when_r2_secrets_unset(tmp_path: Path) -> None:
    """Empty R2_* on main must fail the gate (fires notify-failure) (#278)."""
    script = _push_raw_gate_script()
    env = {
        "R2_ACCOUNT_ID": "",
        "R2_ACCESS_KEY_ID": "",
        "R2_SECRET_ACCESS_KEY": "",
        "DATA_BUCKET": "fplt-data",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_OUTPUT": str(tmp_path / "out"),
        "PATH": "/usr/bin:/bin",
    }
    (tmp_path / "out").write_text("", encoding="utf-8")
    result = subprocess.run(
        ["bash", "-c", script],
        check=False,
        capture_output=True,
        text=True,
        env=env,
    )
    assert result.returncode != 0
    combined = result.stdout + result.stderr
    assert "soft skip" not in combined
    assert "main" in combined.lower()


def test_push_raw_soft_skips_on_pull_request_even_when_ref_main(tmp_path: Path) -> None:
    """pull_request events soft-skip even if GITHUB_REF looks like main."""
    script = _push_raw_gate_script()
    env = {
        "R2_ACCOUNT_ID": "",
        "R2_ACCESS_KEY_ID": "",
        "R2_SECRET_ACCESS_KEY": "",
        "DATA_BUCKET": "fplt-data",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_EVENT_NAME": "pull_request",
        "GITHUB_OUTPUT": str(tmp_path / "out"),
        "PATH": "/usr/bin:/bin",
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


def test_push_raw_gate_does_not_claim_uploaded_when_creds_present(
    tmp_path: Path,
) -> None:
    """Credentials alone must not set uploaded=true before rclone (#340)."""
    script = _push_raw_gate_script()
    env = {
        "R2_ACCOUNT_ID": "acct",
        "R2_ACCESS_KEY_ID": "key",
        "R2_SECRET_ACCESS_KEY": "secret",
        "DATA_BUCKET": "fplt-data",
        "GITHUB_REF": "refs/heads/main",
        "GITHUB_OUTPUT": str(tmp_path / "out"),
        "PATH": "/usr/bin:/bin",
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
    out = (tmp_path / "out").read_text(encoding="utf-8")
    assert "skip=false" in out
    assert "uploaded=" not in out


def test_push_raw_exposes_uploaded_output() -> None:
    """Composite action declares an ``uploaded`` output for heartbeat gating."""
    doc = yaml.safe_load(
        (_ACTIONS / "push-raw" / "action.yml").read_text(encoding="utf-8")
    )
    assert "uploaded" in doc.get("outputs", {})
    assert doc["outputs"]["uploaded"]["value"] == (
        "${{ steps.result.outputs.uploaded }}"
    )


def test_push_raw_uses_install_rclone_not_apt() -> None:
    """push-raw must install the pinned rclone action, not apt (#340)."""
    text = (_ACTIONS / "push-raw" / "action.yml").read_text(encoding="utf-8")
    assert "apt-get install" not in text
    assert "install-rclone" in text
    doc = yaml.safe_load(text)
    uses = [s.get("uses") for s in doc["runs"]["steps"] if "uses" in s]
    assert "./.github/actions/install-rclone" in uses


def test_push_raw_rejects_dot_path_components() -> None:
    """push_raw_upload.sh guards '.'-prefixed paths before rclone (#339)."""
    helper = (_REPO / "scripts" / "push_raw_upload.sh").read_text(encoding="utf-8")
    assert "reject_raw_dot_paths.py" in helper
    # Strip comments so the header docstring mentioning rclone does not confuse order.
    code_lines = [
        line
        for line in helper.splitlines()
        if line.strip() and not line.lstrip().startswith("#")
    ]
    code = "\n".join(code_lines)
    assert "reject_raw_dot_paths.py" in code
    assert "rclone copy" in code
    assert code.index("reject_raw_dot_paths.py") < code.index("rclone copy")
    # Completion marker last (#353).
    assert "--exclude '**/_source.json'" in code
    assert "--include '**/_source.json'" in code
    assert code.index("--exclude") < code.index("--include")
    assert "transfers=" in helper or 'transfers="${RCLONE_TRANSFERS:-32}"' in helper
    action = (_ACTIONS / "push-raw" / "action.yml").read_text(encoding="utf-8")
    assert "push_raw_upload.sh" in action
    assert "::error::" in (_REPO / "scripts" / "reject_raw_dot_paths.py").read_text(
        encoding="utf-8"
    )


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
    """Each alerts job pings its HC_PING_URL_* secret when the main job succeeds.

    Capture workflows that use push-raw also require ``uploaded == true`` so a
    soft-skipped R2 upload does not keep the capture heartbeat green (#278).
    """
    for name in _SCHEDULED:
        steps = _alert_steps(_load(name))
        heartbeats = [
            s for s in steps if s.get("uses") == "./.github/actions/heartbeat"
        ]
        assert len(heartbeats) == 1, name
        step = heartbeats[0]
        main = _MAIN_JOB[name]
        if name in _PUSH_RAW_CAPTURE:
            # Folded YAML ``>-`` becomes a single line; capture sources also
            # require source_ok so unavailable runs do not ping (#338).
            expected_if = (
                f"needs.{main}.result == 'success' "
                f"&& needs.{main}.outputs.uploaded == 'true' "
                f"&& needs.{main}.outputs.source_ok == 'true'"
            )
            # snapshot.yml keeps uploaded-only gating (no capture_health yet).
            if name == "snapshot.yml":
                expected_if = (
                    f"needs.{main}.result == 'success' "
                    f"&& needs.{main}.outputs.uploaded == 'true'"
                )
            assert step.get("if") == expected_if
            doc = _load(name)
            assert doc["jobs"][main]["outputs"]["uploaded"] == (
                "${{ steps.upload.outputs.uploaded }}"
            )
            if name != "snapshot.yml":
                assert doc["jobs"][main]["outputs"]["source_ok"] == (
                    "${{ steps.capture.outputs.source_ok }}"
                )
        else:
            assert step.get("if") == f"needs.{main}.result == 'success'"
        assert step["with"]["ping_url"] == f"${{{{ secrets.{_PING_SECRETS[name]} }}}}"


def test_r2_secret_jobs_use_environment_r2() -> None:
    """Every job that reads secrets.R2_* declares environment: r2 (#278)."""
    for name in (
        "snapshot.yml",
        "understat.yml",
        "api_football.yml",
        "football_data.yml",
        "ops-tick.yml",
        "ops-catchup.yml",
        "raw_parity.yml",
        "curate-full.yml",
        "auth-check.yml",
        "release.yml",
        "deps-equivalence.yml",
    ):
        doc = _load(name)
        for job_id, body in doc["jobs"].items():
            dumped = yaml.safe_dump(body)
            if "secrets.R2_" not in dumped:
                continue
            assert body.get("environment") == "r2", f"{name}::{job_id}"


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
        "fplt-ci",
    ):
        assert name in text, name
    assert "15 min" in text
    assert "45 min" in text
    assert "soft skip" in text.lower() or "soft skip" in text
    assert "runbook, not keepalive" in text
    assert "Do not add a keepalive" not in text or "no keepalive workflow" in text


# Capture / ops schedules that must queue (ci.yml uses sha-keyed concurrency).
_CAPTURE_CONCURRENCY = (
    "snapshot.yml",
    "understat.yml",
    "api_football.yml",
    "football_data.yml",
    "raw_parity.yml",
    "vaastav.yml",
    "auth-check.yml",
    "secret-rotation-reminder.yml",
)


def test_capture_workflows_queue_without_cancel() -> None:
    """Capture schedules declare concurrency with cancel-in-progress: false (#305)."""
    for name in _CAPTURE_CONCURRENCY:
        doc = _load(name)
        conc = doc["concurrency"]
        assert conc["cancel-in-progress"] is False, name
        assert conc["group"], name


# Scheduled workflows that are not capture/ops heartbeats (no alerts job).
_SCHEDULED_WITHOUT_HEARTBEAT = frozenset(
    {
        "ci.yml",  # nightly full matrix (#305); gate job owns notify
        "codeql.yml",  # #309 weekly CodeQL
        "dependency-audit.yml",  # #309 weekly osv-scanner SARIF
    }
)


def test_every_scheduled_workflow_file_is_covered() -> None:
    """Guardrail: any new cron workflow must appear in _SCHEDULED (or be listed).

    ``ci.yml``, ``codeql.yml``, and ``dependency-audit.yml`` are scheduled but
    use non-capture notify paths (CI ``gate``, or security SARIF uploads), so
    they are excluded from ``_SCHEDULED`` heartbeats.
    """
    scheduled: list[str] = []
    for path in sorted(_WORKFLOWS.glob("*.yml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        on = _on(doc)
        if "schedule" in on:
            scheduled.append(path.name)
    expected = {
        n for n in _SCHEDULED if n != "ops-catchup.yml"
    } | _SCHEDULED_WITHOUT_HEARTBEAT
    assert set(scheduled) == expected, textwrap.dedent(
        f"""
        scheduled workflows drifted:
          on disk: {sorted(scheduled)}
          tested:  {sorted(expected)}
        """
    )
