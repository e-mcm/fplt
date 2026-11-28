"""#350: workflow ``run:`` steps must not switch the main workspace branch.

Branch publish (snapshots dual-write) must use ``git worktree`` or
``git -C <dir>`` so composite action post steps still find
``.github/actions`` after the publish step.
"""

from __future__ import annotations

import re
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
_WORKFLOWS = _REPO / ".github" / "workflows"

# ``git checkout`` / ``git switch`` that change the current branch in-place.
_BRANCH_SWITCH = re.compile(
    r"""
    \bgit\s+
    (?:
        checkout\s+--orphan\b
        | checkout\s+--track\b
        | checkout\s+-B\b
        | checkout\s+-b\b
        | checkout\s+-t\b
        | switch\b
    )
    """,
    re.VERBOSE,
)

# ``git checkout <ref>`` without ``--`` path restore / ``-C`` / worktree.
# Matches a bare ref after optional short flags that are not -b/-B/-t.
_BARE_CHECKOUT_REF = re.compile(
    r"""
    \bgit\s+checkout\s+
    (?:-[a-zA-Z]\s+)*          # optional short flags (-f, -q, …)
    (?!--)                     # not a path restore
    (?!-b\b|-B\b|-t\b)         # branch-create flags handled above
    (?P<ref>[^\s-][^\s]*)      # a ref / branch name
    (?:\s|$)
    """,
    re.VERBOSE,
)

_GIT_C_OR_WORKTREE = re.compile(r"\bgit\s+(?:-C\b|worktree\b)")


def _run_scripts(doc: object) -> list[tuple[str, str]]:
    """Collect ``(job_id/step_name, run_script)`` from a workflow document."""
    if not isinstance(doc, dict):
        return []
    jobs = doc.get("jobs")
    if not isinstance(jobs, dict):
        return []
    found: list[tuple[str, str]] = []
    for job_id, job in jobs.items():
        if not isinstance(job, dict):
            continue
        steps = job.get("steps")
        if not isinstance(steps, list):
            continue
        for index, step in enumerate(steps):
            if not isinstance(step, dict):
                continue
            run = step.get("run")
            if not isinstance(run, str):
                continue
            name = step.get("name")
            label = str(name) if isinstance(name, str) else f"step[{index}]"
            found.append((f"{job_id}/{label}", run))
    return found


def _strip_shell_comment(line: str) -> str:
    """Drop a trailing ``#`` comment, ignoring ``#`` inside single quotes."""
    in_single = False
    for index, char in enumerate(line):
        if char == "'" and not in_single:
            in_single = True
            continue
        if char == "'" and in_single:
            in_single = False
            continue
        if char == "#" and not in_single:
            return line[:index]
    return line


def main_workspace_branch_switches(script: str) -> list[str]:
    """Return script lines that switch branches in the main workspace.

    Lines that already use ``git -C <dir>`` or ``git worktree`` are allowed.
    Path restores (``git checkout -- <paths>``) are ignored.

    Args:
        script: Bash body of a workflow ``run:`` step.

    Returns:
        Offending lines (stripped), empty when the script is clean.
    """
    offenders: list[str] = []
    for raw in script.splitlines():
        line = _strip_shell_comment(raw).strip()
        if not line or _GIT_C_OR_WORKTREE.search(line):
            continue
        if _BRANCH_SWITCH.search(line):
            offenders.append(line)
            continue
        # ``git checkout -- file`` / ``git checkout -f -- file`` is a restore.
        if re.search(r"\bgit\s+checkout\b", line) and re.search(
            r"(^|\s)--(\s|$)", line
        ):
            continue
        if _BARE_CHECKOUT_REF.search(line):
            offenders.append(line)
    return offenders


def test_no_workflow_run_switches_main_workspace_branch() -> None:
    """Every ``.github/workflows`` ``run:`` keeps the main checkout intact (#350)."""
    violations: list[str] = []
    for path in sorted(_WORKFLOWS.glob("*.yml")):
        doc = yaml.safe_load(path.read_text(encoding="utf-8"))
        for step_id, script in _run_scripts(doc):
            for line in main_workspace_branch_switches(script):
                violations.append(f"{path.name}:{step_id}: {line}")
    assert violations == [], "main-workspace branch switches:\n" + "\n".join(violations)


def test_capture_workflows_publish_via_worktree() -> None:
    """The five dual-write captures publish snapshots through a worktree (#350)."""
    for name in (
        "snapshot.yml",
        "understat.yml",
        "api_football.yml",
        "football_data.yml",
        "vaastav.yml",
    ):
        text = (_WORKFLOWS / name).read_text(encoding="utf-8")
        assert "git worktree add" in text, name
        assert 'git -C "$wt"' in text, name
        assert "git checkout --track" not in text, name
        assert "git checkout --orphan" not in text, name


def test_scanner_flags_main_checkout_and_allows_worktree() -> None:
    """Unit-check the scanner against synthetic scripts."""
    bad = main_workspace_branch_switches(
        "\n".join(
            [
                "git checkout --track origin/snapshots",
                "git checkout --orphan snapshots",
                "git switch main",
                "git checkout -B snapshots origin/snapshots",
                "git checkout feature-branch",
            ]
        )
    )
    assert len(bad) == 5

    good = main_workspace_branch_switches(
        "\n".join(
            [
                'git worktree add -B snapshots "$wt" origin/snapshots',
                'git worktree add --orphan -b snapshots "$wt"',
                'git -C "$wt" add data/raw',
                'git -C "$wt" checkout --orphan snapshots',
                "git checkout -- README.md",
                "git checkout -f -- path/to/file",
                "git fetch origin snapshots || true",
            ]
        )
    )
    assert good == []
