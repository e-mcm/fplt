"""ops-11: every workflow job must set timeout-minutes."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.check_workflow_timeouts import (
    iter_workflow_jobs,
    main,
    missing_timeouts,
)

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOWS = _REPO / ".github" / "workflows"
_OPS_PROD = _REPO / "configs" / "ops" / "production.yaml"


def test_repo_workflows_all_have_timeout_minutes() -> None:
    """Live ``.github/workflows`` must pass the lint (acceptance criterion)."""
    assert missing_timeouts(_WORKFLOWS) == []
    assert main([]) == 0


def test_missing_timeout_is_reported(tmp_path: Path) -> None:
    """A job without timeout-minutes is a hard failure."""
    wf = tmp_path / "hung.yml"
    wf.write_text(
        "\n".join(
            [
                "name: hung",
                "on: workflow_dispatch",
                "jobs:",
                "  capture:",
                "    runs-on: ubuntu-latest",
                "    steps:",
                "      - run: echo hi",
                "",
            ]
        ),
        encoding="utf-8",
    )
    violations = missing_timeouts(tmp_path)
    assert violations == ["hung.yml::capture missing timeout-minutes"]
    assert main(["--workflows-dir", str(tmp_path)]) == 1


def test_non_positive_timeout_is_reported(tmp_path: Path) -> None:
    """timeout-minutes must be a positive int."""
    wf = tmp_path / "bad.yml"
    wf.write_text(
        "\n".join(
            [
                "name: bad",
                "on: workflow_dispatch",
                "jobs:",
                "  zero:",
                "    runs-on: ubuntu-latest",
                "    timeout-minutes: 0",
                "    steps:",
                "      - run: true",
                "  flag:",
                "    runs-on: ubuntu-latest",
                "    timeout-minutes: true",
                "    steps:",
                "      - run: true",
                "",
            ]
        ),
        encoding="utf-8",
    )
    violations = missing_timeouts(tmp_path)
    assert any("zero" in v and "> 0" in v for v in violations)
    assert any("flag" in v and "positive int" in v for v in violations)


def test_iter_workflow_jobs_skips_non_mapping_jobs(tmp_path: Path) -> None:
    """Malformed job bodies are ignored rather than crashing the lint."""
    wf = tmp_path / "odd.yml"
    wf.write_text(
        "\n".join(
            [
                "name: odd",
                "on: workflow_dispatch",
                "jobs:",
                "  ok:",
                "    runs-on: ubuntu-latest",
                "    timeout-minutes: 5",
                "    steps: []",
                "  reused: some-reusable-job",
                "",
            ]
        ),
        encoding="utf-8",
    )
    jobs = iter_workflow_jobs(tmp_path)
    assert [(p.name, jid) for p, jid, _ in jobs] == [("odd.yml", "ok")]
    assert missing_timeouts(tmp_path) == []


def test_ops_timeouts_exceed_solver_wall_clock() -> None:
    """ops-tick / ops-catchup must outlive HiGHS time_limit budgets.

    production planner ``time_limit_s`` is 600s per solve. Chip boards and
    catch-up can run several solves; tick ≥ 90m and catchup ≥ 180m leave
    headroom above a handful of full budgets (ops-11).
    """
    prod = yaml.safe_load(_OPS_PROD.read_text(encoding="utf-8"))
    time_limit_s = float(prod["pipeline"]["planner"]["time_limit_s"])
    assert time_limit_s == 600.0

    tick = yaml.safe_load((_WORKFLOWS / "ops-tick.yml").read_text(encoding="utf-8"))
    catchup = yaml.safe_load(
        (_WORKFLOWS / "ops-catchup.yml").read_text(encoding="utf-8")
    )
    tick_s = int(tick["jobs"]["tick"]["timeout-minutes"]) * 60
    catchup_s = int(catchup["jobs"]["catchup"]["timeout-minutes"]) * 60
    # At least 5 full solver budgets for a tick, 15 for catch-up.
    assert tick_s > 5 * time_limit_s
    assert catchup_s > 15 * time_limit_s


def test_reusable_workflow_call_skips_timeout_requirement(tmp_path: Path) -> None:
    """Jobs that only ``uses:`` a reusable workflow cannot set timeout-minutes."""
    wf = tmp_path / "call.yml"
    wf.write_text(
        "\n".join(
            [
                "name: call",
                "on: workflow_dispatch",
                "jobs:",
                "  scan:",
                "    uses: org/repo/.github/workflows/scan.yml@0123456789abcdef0123456789abcdef01234567",
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert missing_timeouts(tmp_path) == []
    assert main(["--workflows-dir", str(tmp_path)]) == 0


def test_main_fails_when_workflows_dir_missing(tmp_path: Path) -> None:
    """CLI exits 1 when the workflows directory does not exist."""
    assert main(["--workflows-dir", str(tmp_path / "nope")]) == 1
