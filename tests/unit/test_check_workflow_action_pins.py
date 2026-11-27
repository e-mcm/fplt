"""#309: third-party GitHub Actions must be SHA-pinned with a version comment."""

from __future__ import annotations

from pathlib import Path

from scripts.check_workflow_action_pins import main, unpinned_actions

_REPO = Path(__file__).resolve().parents[2]
_GITHUB = _REPO / ".github"


def test_repo_actions_are_sha_pinned() -> None:
    """Live ``.github`` workflows and composite actions must pass the lint."""
    assert unpinned_actions(_GITHUB) == []
    assert main([]) == 0


def test_tag_ref_is_reported(tmp_path: Path) -> None:
    """A mutable tag ``@v4`` is a hard failure."""
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "bad.yml").write_text(
        "\n".join(
            [
                "name: bad",
                "on: workflow_dispatch",
                "jobs:",
                "  x:",
                "    runs-on: ubuntu-latest",
                "    steps:",
                "      - uses: actions/checkout@v4",
                "",
            ]
        ),
        encoding="utf-8",
    )
    violations = unpinned_actions(tmp_path)
    assert len(violations) == 1
    assert "unpinned action" in violations[0]
    assert main(["--root", str(tmp_path)]) == 1


def test_sha_without_version_comment_is_reported(tmp_path: Path) -> None:
    """SHA pin without a human version comment is rejected."""
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    sha = "11d5960a326750d5838078e36cf38b85af677262"
    (workflows / "nocomment.yml").write_text(
        "\n".join(
            [
                "name: nocomment",
                "on: workflow_dispatch",
                "jobs:",
                "  x:",
                "    runs-on: ubuntu-latest",
                "    steps:",
                f"      - uses: actions/checkout@{sha}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    violations = unpinned_actions(tmp_path)
    assert len(violations) == 1
    assert "missing version comment" in violations[0]


def test_local_composite_actions_are_exempt(tmp_path: Path) -> None:
    """``./.github/actions/...`` refs do not need a SHA."""
    workflows = tmp_path / "workflows"
    workflows.mkdir()
    (workflows / "local.yml").write_text(
        "\n".join(
            [
                "name: local",
                "on: workflow_dispatch",
                "jobs:",
                "  x:",
                "    runs-on: ubuntu-latest",
                "    steps:",
                "      - uses: ./.github/actions/heartbeat",
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert unpinned_actions(tmp_path) == []
    assert main(["--root", str(tmp_path)]) == 0


def test_composite_action_uses_are_scanned(tmp_path: Path) -> None:
    """Third-party ``uses:`` inside composite actions are linted too."""
    actions = tmp_path / "actions" / "demo"
    actions.mkdir(parents=True)
    (actions / "action.yml").write_text(
        "\n".join(
            [
                "name: demo",
                "runs:",
                "  using: composite",
                "  steps:",
                "    - uses: actions/cache@v4",
                "",
            ]
        ),
        encoding="utf-8",
    )
    violations = unpinned_actions(tmp_path)
    assert len(violations) == 1
    assert "actions/demo/action.yml" in violations[0].replace("\\", "/")
