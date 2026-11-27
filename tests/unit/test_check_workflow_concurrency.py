"""#305: CI must not cancel main; scheduled workflows need concurrency."""

from __future__ import annotations

from pathlib import Path

import yaml

from scripts.check_workflow_concurrency import (
    ci_cancels_main_runs,
    concurrency_violations,
    load_workflow,
    main,
    scheduled_concurrency_violations,
    workflow_on,
)

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOWS = _REPO / ".github" / "workflows"


def test_repo_workflows_pass_concurrency_lint() -> None:
    """Live ``.github/workflows`` must pass the concurrency lint (#305)."""
    assert concurrency_violations(_WORKFLOWS) == []
    assert main([]) == 0


def test_ci_cancel_true_is_reported(tmp_path: Path) -> None:
    """``cancel-in-progress: true`` on ci.yml is a hard failure."""
    (tmp_path / "ci.yml").write_text(
        "\n".join(
            [
                "name: CI",
                "on:",
                "  push:",
                "    branches: [main]",
                "concurrency:",
                "  group: ci-${{ github.ref }}",
                "  cancel-in-progress: true",
                "jobs:",
                "  lint:",
                "    runs-on: ubuntu-latest",
                "    timeout-minutes: 5",
                "    steps: []",
                "",
            ]
        ),
        encoding="utf-8",
    )
    violations = concurrency_violations(tmp_path)
    assert any("cancels main" in v for v in violations)
    assert main(["--workflows-dir", str(tmp_path)]) == 1


def test_ci_pr_only_cancel_expression_passes(tmp_path: Path) -> None:
    """PR-gated cancel-in-progress expression is accepted."""
    (tmp_path / "ci.yml").write_text(
        "\n".join(
            [
                "name: CI",
                "on:",
                "  push:",
                "    branches: [main]",
                "  pull_request:",
                "    branches: [main]",
                "concurrency:",
                "  group: >-",
                "    ci-${{ github.event_name == 'pull_request'",
                "    && github.ref || github.sha }}",
                "  cancel-in-progress: >-",
                "    ${{ github.event_name == 'pull_request' }}",
                "jobs:",
                "  lint:",
                "    runs-on: ubuntu-latest",
                "    timeout-minutes: 5",
                "    steps: []",
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert concurrency_violations(tmp_path) == []
    assert (
        ci_cancels_main_runs(
            __import__("yaml").safe_load(
                (tmp_path / "ci.yml").read_text(encoding="utf-8")
            )
        )
        is None
    )


def test_scheduled_workflow_missing_concurrency_is_reported(tmp_path: Path) -> None:
    """A scheduled workflow without concurrency.group fails the lint."""
    (tmp_path / "capture.yml").write_text(
        "\n".join(
            [
                "name: capture",
                "on:",
                "  schedule:",
                "    - cron: '0 6 * * *'",
                "jobs:",
                "  run:",
                "    runs-on: ubuntu-latest",
                "    timeout-minutes: 5",
                "    steps: []",
                "",
            ]
        ),
        encoding="utf-8",
    )
    violations = scheduled_concurrency_violations(tmp_path)
    assert violations == ["capture.yml: scheduled workflow missing concurrency.group"]
    assert main(["--workflows-dir", str(tmp_path)]) == 1


def test_scheduled_workflow_cancel_true_is_reported(tmp_path: Path) -> None:
    """Scheduled workflows must queue (cancel-in-progress: false), not cancel."""
    (tmp_path / "capture.yml").write_text(
        "\n".join(
            [
                "name: capture",
                "on:",
                "  schedule:",
                "    - cron: '0 6 * * *'",
                "concurrency:",
                "  group: capture",
                "  cancel-in-progress: true",
                "jobs:",
                "  run:",
                "    runs-on: ubuntu-latest",
                "    timeout-minutes: 5",
                "    steps: []",
                "",
            ]
        ),
        encoding="utf-8",
    )
    violations = scheduled_concurrency_violations(tmp_path)
    assert any("must not set cancel-in-progress: true" in v for v in violations)


def test_scheduled_workflow_with_group_passes(tmp_path: Path) -> None:
    """``concurrency.group`` + ``cancel-in-progress: false`` is enough."""
    (tmp_path / "capture.yml").write_text(
        "\n".join(
            [
                "name: capture",
                "on:",
                "  schedule:",
                "    - cron: '0 6 * * *'",
                "concurrency:",
                "  group: capture",
                "  cancel-in-progress: false",
                "jobs:",
                "  run:",
                "    runs-on: ubuntu-latest",
                "    timeout-minutes: 5",
                "    steps: []",
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert scheduled_concurrency_violations(tmp_path) == []
    assert concurrency_violations(tmp_path) == []


def test_main_fails_when_workflows_dir_missing(tmp_path: Path) -> None:
    """CLI exits 1 when the workflows directory does not exist."""
    assert main(["--workflows-dir", str(tmp_path / "nope")]) == 1


def test_load_workflow_rejects_non_mapping(tmp_path: Path) -> None:
    """Scalar / list YAML documents are ignored rather than crashing."""
    path = tmp_path / "list.yml"
    path.write_text("- just a list\n", encoding="utf-8")
    assert load_workflow(path) is None


def test_workflow_on_handles_bool_key_and_non_mapping() -> None:
    """PyYAML ``True`` key is accepted; non-mapping ``on`` yields {}."""
    assert workflow_on({True: {"schedule": [{"cron": "0 0 * * *"}]}}) == {
        "schedule": [{"cron": "0 0 * * *"}]
    }
    assert workflow_on({"on": "workflow_dispatch"}) == {}


def test_ci_missing_or_false_cancel(tmp_path: Path) -> None:
    """Missing concurrency fails; cancel false/None is safe for main."""
    (tmp_path / "ci.yml").write_text(
        "name: CI\non: {push: {branches: [main]}}\njobs: {}\n",
        encoding="utf-8",
    )
    assert any("missing concurrency" in v for v in concurrency_violations(tmp_path))
    assert ci_cancels_main_runs({"concurrency": {"group": "ci"}}) is None
    assert (
        ci_cancels_main_runs(
            {"concurrency": {"group": "ci", "cancel-in-progress": False}}
        )
        is None
    )


def test_ci_bad_cancel_expression_and_type() -> None:
    """Unknown cancel expressions / types are rejected."""
    assert "must only cancel pull_request" in (
        ci_cancels_main_runs(
            {
                "concurrency": {
                    "group": "ci",
                    "cancel-in-progress": "${{ true }}",
                }
            }
        )
        or ""
    )
    assert "unexpected type" in (
        ci_cancels_main_runs({"concurrency": {"group": "ci", "cancel-in-progress": 1}})
        or ""
    )


def test_scheduled_empty_group_and_non_mapping_ci(tmp_path: Path) -> None:
    """Empty concurrency.group and non-mapping ci.yml are violations."""
    (tmp_path / "capture.yml").write_text(
        "\n".join(
            [
                "name: capture",
                "on:",
                "  schedule:",
                "    - cron: '0 6 * * *'",
                "concurrency:",
                "  group: ''",
                "  cancel-in-progress: false",
                "jobs:",
                "  run:",
                "    runs-on: ubuntu-latest",
                "    timeout-minutes: 5",
                "    steps: []",
                "",
            ]
        ),
        encoding="utf-8",
    )
    (tmp_path / "ci.yml").write_text("- not a mapping\n", encoding="utf-8")
    violations = concurrency_violations(tmp_path)
    assert any("missing concurrency.group" in v for v in violations)
    assert "ci.yml is not a YAML mapping" in violations


def test_scheduled_skips_unparseable_workflow(tmp_path: Path) -> None:
    """Non-mapping scheduled files are skipped without a false positive."""
    (tmp_path / "odd.yml").write_text("null\n", encoding="utf-8")
    assert scheduled_concurrency_violations(tmp_path) == []
    assert yaml.safe_load("null\n") is None
