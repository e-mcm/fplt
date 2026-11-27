"""Lint: secret-handling workflow steps must not enable ``set -x`` (#308)."""

from __future__ import annotations

from pathlib import Path

from scripts.check_workflow_secret_xtrace import main, xtrace_secret_violations

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOWS = _REPO / ".github" / "workflows"
_ACTIONS = _REPO / ".github" / "actions"


def test_repo_workflows_have_no_secret_xtrace() -> None:
    """Live workflows and composite actions must pass the lint."""
    assert (
        xtrace_secret_violations(workflows_dir=_WORKFLOWS, actions_dir=_ACTIONS) == []
    )
    assert main([]) == 0


def test_set_x_with_secrets_is_reported(tmp_path: Path) -> None:
    """A run step that handles secrets and enables xtrace fails the lint."""
    wf = tmp_path / "leak.yml"
    wf.write_text(
        "\n".join(
            [
                "name: leak",
                "on: workflow_dispatch",
                "jobs:",
                "  tick:",
                "    runs-on: ubuntu-latest",
                "    steps:",
                "      - name: bad",
                "        env:",
                "          FPLT_FPL_API_TOKEN: ${{ secrets.FPLT_FPL_API_TOKEN }}",
                "        run: |",
                "          set -euxo pipefail",
                "          echo hi",
                "",
            ]
        ),
        encoding="utf-8",
    )
    violations = xtrace_secret_violations(workflows_dir=tmp_path, actions_dir=None)
    assert len(violations) == 1
    assert "set -x" in violations[0]
    assert main(["--workflows-dir", str(tmp_path), "--skip-actions"]) == 1


def test_set_x_without_secrets_is_ok(tmp_path: Path) -> None:
    """xtrace is allowed when the step does not touch secrets."""
    wf = tmp_path / "ok.yml"
    wf.write_text(
        "\n".join(
            [
                "name: ok",
                "on: workflow_dispatch",
                "jobs:",
                "  build:",
                "    runs-on: ubuntu-latest",
                "    steps:",
                "      - run: |",
                "          set -x",
                "          echo public",
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert xtrace_secret_violations(workflows_dir=tmp_path, actions_dir=None) == []


def test_secret_step_without_xtrace_is_ok(tmp_path: Path) -> None:
    """Secret env is fine when the script does not enable xtrace."""
    wf = tmp_path / "masked.yml"
    wf.write_text(
        "\n".join(
            [
                "name: masked",
                "on: workflow_dispatch",
                "jobs:",
                "  tick:",
                "    runs-on: ubuntu-latest",
                "    steps:",
                "      - env:",
                "          FPLT_FPL_API_TOKEN: ${{ secrets.FPLT_FPL_API_TOKEN }}",
                "        run: |",
                "          set -euo pipefail",
                '          tok="${FPLT_FPL_API_TOKEN#Bearer }"',
                '          echo "::add-mask::${tok}"',
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert xtrace_secret_violations(workflows_dir=tmp_path, actions_dir=None) == []
