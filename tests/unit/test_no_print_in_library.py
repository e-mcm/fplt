"""Lint: library modules under ``src/fplt`` must not call ``print(`` (#310)."""

from __future__ import annotations

import ast
from pathlib import Path

_SRC_ROOT = Path(__file__).resolve().parents[2] / "src" / "fplt"

# Operator-facing surfaces may print; everything else must use logging.
_ALLOWED_RELATIVE = frozenset(
    {
        "cli.py",
    }
)
_ALLOWED_PREFIXES = ("ui/",)


def _is_allowed(relative: str) -> bool:
    if relative in _ALLOWED_RELATIVE:
        return True
    return any(relative.startswith(prefix) for prefix in _ALLOWED_PREFIXES)


def _print_call_lines(path: Path) -> list[int]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    hits: list[int] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        func = node.func
        if isinstance(func, ast.Name) and func.id == "print":
            hits.append(node.lineno)
    return hits


def test_no_print_outside_allowed_modules() -> None:
    """Fail when ``print(`` appears outside CLI / UI modules."""
    offenders: list[str] = []
    for path in sorted(_SRC_ROOT.rglob("*.py")):
        relative = path.relative_to(_SRC_ROOT).as_posix()
        if _is_allowed(relative):
            continue
        lines = _print_call_lines(path)
        if lines:
            joined = ", ".join(str(n) for n in lines)
            offenders.append(f"{relative}: print( at line(s) {joined}")
    assert not offenders, "library print( calls:\n" + "\n".join(offenders)


def test_allowed_modules_still_exist() -> None:
    """Guard the allowlist paths so renames update the lint."""
    assert (_SRC_ROOT / "cli.py").is_file()
    assert (_SRC_ROOT / "ui").is_dir()
