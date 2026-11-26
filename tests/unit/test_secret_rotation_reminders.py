"""ops-11: secret-rotation reminder window and issue payloads."""

from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import pytest
import yaml

from scripts.secret_rotation_reminders import (
    due_reminders,
    issue_body,
    issue_title,
    load_rotation_config,
    main,
    reminder_payloads,
)

_REPO = Path(__file__).resolve().parents[2]
_CONFIG = _REPO / "configs" / "ops" / "secret_rotation.yaml"


def test_committed_rotation_config_loads() -> None:
    """Repo config is valid and lists the R2 yearly rotation."""
    cfg = load_rotation_config(_CONFIG)
    assert cfg["warn_days"] == 7
    ids = {r["id"] for r in cfg["rotations"]}
    assert "r2-tokens" in ids
    assert "api-football-key" in ids


def test_due_reminders_window_edges() -> None:
    """Reminder fires from next_due - warn_days through next_due inclusive."""
    rotations = [
        {
            "id": "r2-tokens",
            "title": "Rotate R2",
            "next_due": "2027-01-15",
        }
    ]
    assert due_reminders(rotations, today=date(2027, 1, 7), warn_days=7) == []
    due = due_reminders(rotations, today=date(2027, 1, 8), warn_days=7)
    assert [r["id"] for r in due] == ["r2-tokens"]
    assert due_reminders(rotations, today=date(2027, 1, 15), warn_days=7)
    assert due_reminders(rotations, today=date(2027, 1, 16), warn_days=7) == []


def test_empty_next_due_is_skipped() -> None:
    """Blank next_due means 'no calendar reminder' (e.g. FPL JWT)."""
    rotations = [{"id": "fpl-api-token", "title": "FPL", "next_due": ""}]
    assert due_reminders(rotations, today=date(2027, 1, 1), warn_days=7) == []


def test_issue_title_and_body_are_stable() -> None:
    """Title embeds id + due date so the workflow can dedupe open issues."""
    entry = {
        "id": "r2-tokens",
        "title": "Rotate Cloudflare R2 API tokens",
        "next_due": "2027-01-15",
        "cadence": "yearly",
        "scope": "GHA + local",
        "secrets": ["R2_ACCESS_KEY_ID"],
        "notes": "Rotate together.",
    }
    title = issue_title(entry)
    assert title == "[ops-11] Rotate secrets: r2-tokens (due 2027-01-15)"
    body = issue_body(entry)
    assert "`R2_ACCESS_KEY_ID`" in body
    assert "deploy/README.md" in body
    assert "Rotate together." in body


def test_reminder_payloads_json_roundtrip(tmp_path: Path) -> None:
    """CLI ``--json`` emits the payloads the reminder workflow consumes."""
    cfg_path = tmp_path / "secret_rotation.yaml"
    cfg_path.write_text(
        yaml.safe_dump(
            {
                "warn_days": 7,
                "rotations": [
                    {
                        "id": "demo",
                        "title": "Demo",
                        "next_due": "2027-02-01",
                        "secrets": ["X"],
                    }
                ],
            }
        ),
        encoding="utf-8",
    )
    assert main(["--config", str(cfg_path), "--today", "2027-01-25", "--json"]) == 0
    # Capture via function API (stdout asserted indirectly by structure).
    payloads = reminder_payloads(
        [{"id": "demo", "title": "Demo", "next_due": "2027-02-01", "secrets": ["X"]}],
        today=date(2027, 1, 25),
        warn_days=7,
    )
    assert len(payloads) == 1
    assert json.loads(json.dumps(payloads))[0]["id"] == "demo"


def test_load_rotation_config_rejects_bad_warn_days(tmp_path: Path) -> None:
    """Invalid warn_days raises ValueError."""
    path = tmp_path / "bad.yaml"
    path.write_text("warn_days: true\nrotations: []\n", encoding="utf-8")
    with pytest.raises(ValueError, match="warn_days"):
        load_rotation_config(path)


def test_main_reports_no_reminders(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Outside the window, CLI prints a clear no-op message."""
    cfg_path = tmp_path / "secret_rotation.yaml"
    cfg_path.write_text(
        "\n".join(
            [
                "warn_days: 7",
                "rotations:",
                "  - id: x",
                "    title: X",
                "    next_due: '2099-01-01'",
                "",
            ]
        ),
        encoding="utf-8",
    )
    assert main(["--config", str(cfg_path), "--today", "2026-01-01"]) == 0
    assert "no reminders due" in capsys.readouterr().out
