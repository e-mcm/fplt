"""Ops runbook anchors for notification footers.

Alert bodies name a section in ``docs/ops-runbook.md`` so an ntfy / SMTP
message points at the one-command fix. Anchors match GitHub heading slugs.
Literals stay contiguous so notify→runbook greps can find them.
"""

from __future__ import annotations

RUNBOOK_PATH = "docs/ops-runbook.md"

# Stable anchors — keep in sync with headings in docs/ops-runbook.md.
OPS_01_TOKEN_EXPIRY = "docs/ops-runbook.md#ops-01-token-expiry"
OPS_02_NO_PLAN = "docs/ops-runbook.md#ops-02-no-plan"
OPS_03_WORKFLOW_FAILURE = "docs/ops-runbook.md#ops-03-workflow-failure"
OPS_03_HEARTBEAT_MISSING = "docs/ops-runbook.md#ops-03-heartbeat-missing"
OPS_04_BAD_RELEASE = "docs/ops-runbook.md#ops-04-bad-release"
OPS_05_APPLY_POLICY = "docs/ops-runbook.md#ops-05-apply-policy"
OPS_06_QUARANTINE = "docs/ops-runbook.md#ops-06-quarantine"
OPS_07_SEASON_ROLLOVER = "docs/ops-runbook.md#ops-07-season-rollover"
OPS_08_WEEKLY_REPORT = "docs/ops-runbook.md#ops-08-weekly-report"
OPS_09_MODEL_DRIFT = "docs/ops-runbook.md#ops-09-model-drift"
OPS_10_STATUS = "docs/ops-runbook.md#ops-10-status"
OPS_11_SECRET_ROTATION = "docs/ops-runbook.md#ops-11-secret-rotation"
RECOMMENDATION = "docs/ops-runbook.md#recommendation"

# Every ops-01..ops-11 alert anchor (heartbeat shares ops-03's pair).
ALERT_ANCHORS: tuple[str, ...] = (
    OPS_01_TOKEN_EXPIRY,
    OPS_02_NO_PLAN,
    OPS_03_WORKFLOW_FAILURE,
    OPS_03_HEARTBEAT_MISSING,
    OPS_04_BAD_RELEASE,
    OPS_05_APPLY_POLICY,
    OPS_06_QUARANTINE,
    OPS_07_SEASON_ROLLOVER,
    OPS_08_WEEKLY_REPORT,
    OPS_09_MODEL_DRIFT,
    OPS_10_STATUS,
    OPS_11_SECRET_ROTATION,
)


def footer(anchor: str) -> str:
    """Return a one-line runbook pointer for a notification body.

    Args:
        anchor: Full ``docs/ops-runbook.md#…`` path from this module.

    Returns:
        A ``Runbook: …`` line safe to append to an ntfy / email body.
    """
    return f"Runbook: {anchor}"
