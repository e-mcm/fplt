#!/usr/bin/env python3
"""Decide which secret-rotation reminders are due (ops-11).

Reads ``configs/ops/secret_rotation.yaml``. A rotation is due for a reminder
when ``today`` falls in ``[next_due - warn_days, next_due]`` (inclusive).
The scheduled workflow opens one GitHub issue per due entry; this module is
pure so tests never touch the network.

Usage::

    uv run python scripts/secret_rotation_reminders.py --today 2027-01-08
    uv run python scripts/secret_rotation_reminders.py --today 2027-01-08 --json
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import yaml

_REPO_ROOT = Path(__file__).resolve().parents[1]
_DEFAULT_CONFIG = _REPO_ROOT / "configs" / "ops" / "secret_rotation.yaml"


def load_rotation_config(path: Path) -> dict[str, Any]:
    """Load and lightly validate the secret-rotation YAML.

    Args:
        path: Path to ``secret_rotation.yaml``.

    Returns:
        Parsed mapping with ``warn_days`` (int) and ``rotations`` (list).

    Raises:
        ValueError: When required keys are missing or mistyped.
    """
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"{path}: expected a mapping at the root")
    doc = cast(dict[str, Any], raw)
    warn_days = doc.get("warn_days", 7)
    if not isinstance(warn_days, int) or isinstance(warn_days, bool) or warn_days < 0:
        raise ValueError(f"{path}: warn_days must be a non-negative int")
    rotations_raw = doc.get("rotations")
    if not isinstance(rotations_raw, list):
        raise ValueError(f"{path}: rotations must be a list")
    rotations = cast(list[Any], rotations_raw)
    return {"warn_days": warn_days, "rotations": rotations}


def parse_due_date(value: str | date) -> date:
    """Parse an ISO date string or pass through a ``date``.

    Args:
        value: ``YYYY-MM-DD`` string or a ``date``.

    Returns:
        The calendar date.

    Raises:
        ValueError: When the string is not ISO ``YYYY-MM-DD``.
    """
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return date.fromisoformat(str(value))


def due_reminders(
    rotations: list[dict[str, Any]],
    *,
    today: date,
    warn_days: int = 7,
) -> list[dict[str, Any]]:
    """Return rotations whose reminder window includes ``today``.

    Args:
        rotations: Entries from ``secret_rotation.yaml`` (each needs ``id``,
            ``title``, ``next_due``).
        today: Reference calendar date (UTC in the reminder workflow).
        warn_days: Inclusive days before ``next_due`` to start reminding.

    Returns:
        Matching rotation dicts (same objects as input), sorted by
        ``next_due`` then ``id``. Entries missing ``next_due`` are skipped.
    """
    due: list[dict[str, Any]] = []
    for entry in rotations:
        raw_due = entry.get("next_due")
        if raw_due is None or raw_due == "":
            continue
        next_due = parse_due_date(cast(str | date, raw_due))
        window_start = next_due - timedelta(days=warn_days)
        if window_start <= today <= next_due:
            due.append(entry)
    due.sort(
        key=lambda e: (
            parse_due_date(cast(str | date, e["next_due"])),
            str(e.get("id", "")),
        )
    )
    return due


def issue_title(entry: dict[str, Any]) -> str:
    """Build the GitHub issue title for a due rotation reminder.

    Args:
        entry: A rotation mapping with ``id``, ``title``, ``next_due``.

    Returns:
        Stable title including the rotation id and due date (for dedupe).
    """
    due = parse_due_date(cast(str | date, entry["next_due"]))
    return f"[ops-11] Rotate secrets: {entry['id']} (due {due.isoformat()})"


def issue_body(entry: dict[str, Any]) -> str:
    """Build the GitHub issue body for a due rotation reminder.

    Args:
        entry: A rotation mapping.

    Returns:
        Markdown body with cadence, secrets list, and a pointer to
        ``deploy/README.md``.
    """
    secrets_val = entry.get("secrets")
    secrets: list[Any] = (
        cast(list[Any], secrets_val) if isinstance(secrets_val, list) else []
    )
    secret_lines = "\n".join(f"- `{name}`" for name in secrets) or ("- _(none listed)_")
    notes = str(entry.get("notes") or "").strip()
    due = parse_due_date(cast(str | date, entry["next_due"]))
    parts = [
        f"## {entry.get('title', entry['id'])}",
        "",
        f"**Due:** {due.isoformat()}",
        f"**Cadence:** {entry.get('cadence', 'unspecified')}",
        f"**Scope:** {entry.get('scope', 'unspecified')}",
        "",
        "### Secrets",
        secret_lines,
        "",
    ]
    if notes:
        parts.extend([notes, ""])
    parts.extend(
        [
            "Inventory and owner steps: `deploy/README.md` (ops-11).",
            "After rotating, bump `next_due` in "
            "`configs/ops/secret_rotation.yaml` and close this issue.",
        ]
    )
    return "\n".join(parts) + "\n"


def reminder_payloads(
    rotations: list[dict[str, Any]],
    *,
    today: date,
    warn_days: int = 7,
) -> list[dict[str, str]]:
    """Build ``{id, title, body}`` payloads for due rotations.

    Args:
        rotations: Config rotation entries.
        today: Reference date.
        warn_days: Reminder lead time in days.

    Returns:
        JSON-serialisable list for the reminder workflow.
    """
    return [
        {
            "id": str(entry["id"]),
            "title": issue_title(entry),
            "body": issue_body(entry),
        }
        for entry in due_reminders(rotations, today=today, warn_days=warn_days)
    ]


def _as_rotation_dicts(raw: list[Any]) -> list[dict[str, Any]]:
    """Filter config rotations down to mapping entries."""
    out: list[dict[str, Any]] = []
    for item in raw:
        if isinstance(item, dict):
            out.append(cast(dict[str, Any], item))
    return out


def main(argv: list[str] | None = None) -> int:
    """Print due reminder titles (or JSON payloads) for today.

    Args:
        argv: CLI args (defaults to ``sys.argv[1:]``).

    Returns:
        ``0`` when parsing succeeds; ``1`` on config errors.
    """
    parser = argparse.ArgumentParser(
        description="List secret-rotation reminders due today (ops-11)."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="Path to secret_rotation.yaml",
    )
    parser.add_argument(
        "--today",
        type=str,
        default=None,
        help="Reference date YYYY-MM-DD (default: UTC today)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print a JSON array of {id, title, body} for due reminders",
    )
    args = parser.parse_args(argv)
    try:
        cfg = load_rotation_config(cast(Path, args.config))
    except (OSError, ValueError, yaml.YAMLError) as exc:
        print(f"secret_rotation: {exc}", file=sys.stderr)
        return 1
    today = date.fromisoformat(args.today) if args.today else datetime.now(UTC).date()
    rotations = _as_rotation_dicts(cast(list[Any], cfg["rotations"]))
    payloads = reminder_payloads(
        rotations,
        today=today,
        warn_days=int(cfg["warn_days"]),
    )
    if args.json:
        print(json.dumps(payloads, indent=2))
        return 0
    if not payloads:
        print("secret_rotation: no reminders due")
        return 0
    for item in payloads:
        print(item["title"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
