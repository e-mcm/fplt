"""Notify call sites must name a docs/ops-runbook.md#ops-XX-… anchor.

Tolerant of alerts that are not wired yet: every existing notify / ntfy
failure body must point at a real runbook heading, but ops-01..ops-11 need
not all appear in code.
"""

from __future__ import annotations

import ast
import re
from pathlib import Path

from fplt.automation import runbook as runbook_mod

_REPO = Path(__file__).resolve().parents[2]
_RUNBOOK = _REPO / "docs" / "ops-runbook.md"
_SRC = _REPO / "src"
_NOTIFY_FAILURE = _REPO / ".github" / "actions" / "notify-failure" / "action.yml"
_NOTIFY_PY = _SRC / "fplt" / "automation" / "notify.py"

_ANCHOR_RE = re.compile(r"docs/ops-runbook\.md#([a-z0-9-]+)")
_HEADING_RE = re.compile(r"^#{2,3}\s+(.+)$", re.MULTILINE)
_OPS_HEADING_RE = re.compile(r"^#{2,3}\s+ops-(\d{2}):", re.IGNORECASE | re.MULTILINE)
_RUNBOOK_CONST_RE = re.compile(
    r"\b(OPS_\d{2}_[A-Z0-9_]+|RECOMMENDATION|ALERT_ANCHORS|runbook_anchor)\b"
)


def _github_slug(heading: str) -> str:
    """Approximate GitHub's heading anchor slug."""
    text = heading.strip().lower()
    text = re.sub(r"[^\w\s-]", "", text)
    return re.sub(r"\s+", "-", text).strip("-")


def _runbook_slugs() -> set[str]:
    text = _RUNBOOK.read_text(encoding="utf-8")
    return {_github_slug(match.group(1)) for match in _HEADING_RE.finditer(text)}


def _call_name(node: ast.Call) -> str | None:
    func = node.func
    if isinstance(func, ast.Name):
        return func.id
    if isinstance(func, ast.Attribute):
        return func.attr
    return None


def _python_notify_sites() -> list[Path]:
    """Return source files that call notify / notify_recommendation."""
    sites: list[Path] = []
    for path in _SRC.rglob("*.py"):
        if path == _NOTIFY_PY:
            continue
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Call) and _call_name(node) in {
                "notify",
                "notify_recommendation",
            }:
                sites.append(path)
                break
    return sorted(sites)


def _site_names_runbook(path: Path) -> bool:
    """True when the file names a runbook section (literal, const, or default)."""
    text = path.read_text(encoding="utf-8")
    if _ANCHOR_RE.search(text):
        return True
    if _RUNBOOK_CONST_RE.search(text):
        return True
    # notify_recommendation applies RECOMMENDATION by default in notify.py.
    tree = ast.parse(text, filename=str(path))
    names = {_call_name(node) for node in ast.walk(tree) if isinstance(node, ast.Call)}
    return names == {"notify_recommendation"} or (
        "notify_recommendation" in names and "notify" not in names
    )


def test_runbook_documents_ops_01_through_ops_11() -> None:
    text = _RUNBOOK.read_text(encoding="utf-8")
    found = {int(m.group(1)) for m in _OPS_HEADING_RE.finditer(text)}
    missing = set(range(1, 12)) - found
    assert not missing, f"missing ops sections: {missing}"
    assert "## Season rollover" in text
    assert "## Run a gameweek by hand" in text


def test_runbook_anchor_constants_match_headings() -> None:
    slugs = _runbook_slugs()
    for anchor in (*runbook_mod.ALERT_ANCHORS, runbook_mod.RECOMMENDATION):
        match = _ANCHOR_RE.fullmatch(anchor)
        assert match is not None, anchor
        assert match.group(1) in slugs, f"{anchor} missing from {_RUNBOOK}"


def test_notify_call_sites_name_a_runbook_anchor() -> None:
    """Every live notify site greps to a real runbook anchor (ops-13)."""
    slugs = _runbook_slugs()
    sites = _python_notify_sites()
    assert sites, "expected at least one notify call site under src/"

    missing = [
        str(path.relative_to(_REPO)) for path in sites if not _site_names_runbook(path)
    ]
    assert not missing, "notify sites without a runbook anchor: " + ", ".join(missing)

    # Module constants + the GHA failure action carry the literal anchors.
    runbook_path = Path(runbook_mod.__file__)
    module_anchors = _ANCHOR_RE.findall(runbook_path.read_text(encoding="utf-8"))
    assert module_anchors, "runbook.py must define docs/ops-runbook.md#… literals"
    for slug in module_anchors:
        assert slug in slugs, f"runbook.py → #{slug}"

    action = _NOTIFY_FAILURE.read_text(encoding="utf-8")
    action_anchors = _ANCHOR_RE.findall(action)
    assert action_anchors, "notify-failure action must name a runbook section"
    for slug in action_anchors:
        assert slug in slugs, f"notify-failure → #{slug}"


def test_docs_and_deploy_readme_link_the_runbook() -> None:
    for rel in ("docs/README.md", "deploy/README.md"):
        text = (_REPO / rel).read_text(encoding="utf-8")
        assert "ops-runbook.md" in text, rel
