"""Jobs that read secrets.R2_* must declare environment: r2 (#278)."""

from __future__ import annotations

from pathlib import Path

from scripts.check_workflow_r2_environment import (
    iter_workflow_jobs,
    jobs_missing_r2_environment,
    main,
)

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOWS = _REPO / ".github" / "workflows"


def test_repo_workflows_r2_jobs_have_environment() -> None:
    """Live ``.github/workflows`` must pass the R2 environment lint."""
    assert jobs_missing_r2_environment(_WORKFLOWS) == []
    assert main([]) == 0


def test_missing_r2_environment_is_reported(tmp_path: Path) -> None:
    """A job that references secrets.R2_ without environment: r2 fails the lint."""
    wf = tmp_path / "capture.yml"
    wf.write_text(
        "\n".join(
            [
                "name: capture",
                "on: workflow_dispatch",
                "jobs:",
                "  upload:",
                "    runs-on: ubuntu-latest",
                "    timeout-minutes: 10",
                "    steps:",
                "      - run: echo ${{ secrets.R2_ACCOUNT_ID }}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    violations = jobs_missing_r2_environment(tmp_path)
    assert violations == [
        "capture.yml::upload references secrets.R2_ without environment: r2"
    ]
    assert main(["--workflows-dir", str(tmp_path)]) == 1


def test_environment_r2_string_or_mapping_passes(tmp_path: Path) -> None:
    """Both ``environment: r2`` and ``environment: {name: r2}`` are accepted."""
    (tmp_path / "ok.yml").write_text(
        "\n".join(
            [
                "name: ok",
                "on: workflow_dispatch",
                "jobs:",
                "  as_string:",
                "    runs-on: ubuntu-latest",
                "    timeout-minutes: 5",
                "    environment: r2",
                "    steps:",
                "      - run: echo ${{ secrets.R2_ACCESS_KEY_ID }}",
                "  as_mapping:",
                "    runs-on: ubuntu-latest",
                "    timeout-minutes: 5",
                "    environment:",
                "      name: r2",
                "    steps:",
                "      - run: echo ${{ secrets.R2_SECRET_ACCESS_KEY }}",
                "  hc_only:",
                "    runs-on: ubuntu-latest",
                "    timeout-minutes: 5",
                "    steps:",
                "      - run: echo ${{ secrets.HC_PING_URL_SNAPSHOT }}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert jobs_missing_r2_environment(tmp_path) == []
    jobs = iter_workflow_jobs(tmp_path)
    assert {jid for _, jid, _ in jobs} == {"as_string", "as_mapping", "hc_only"}


def test_main_fails_when_workflows_dir_missing(tmp_path: Path) -> None:
    """CLI exits 1 when the workflows directory does not exist."""
    assert main(["--workflows-dir", str(tmp_path / "nope")]) == 1
