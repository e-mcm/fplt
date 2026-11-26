"""ops-04 release gates: freeze, red CI, failed smoke, promote commands."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

from fplt.automation.freeze import FREEZE_LEAD
from fplt.automation.release import (
    PROD_TAG,
    ReleaseCandidate,
    ReleaseGateError,
    assert_ci_green,
    assert_outside_freeze,
    evaluate_release_gates,
    history_tag_name,
    promote_prod_tags,
    resolve_candidate,
    resolve_prod_tag_label,
    run_smoke_plan,
)

_SEASON = "2025-26"
_DEADLINE = datetime(2025, 11, 29, 11, 0, tzinfo=UTC)
_KICKOFF = datetime(2025, 11, 29, 12, 30, tzinfo=UTC)
_SHA = "abcdef0123456789abcdef0123456789abcdef01"


def _gw() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [_SEASON],
            "gw": [13],
            "deadline_time": [_DEADLINE],
        }
    )


def _fx() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "season": [_SEASON],
            "gw": [13],
            "kickoff_time": [_KICKOFF],
        }
    )


def test_assert_ci_green_accepts_success() -> None:
    assert_ci_green(_SHA, status_fn=lambda _sha: "success")


@pytest.mark.parametrize(
    ("status", "needle"),
    [
        ("failure", "CI is red"),
        ("pending", "CI still pending"),
        ("unknown", "CI status"),
    ],
)
def test_assert_ci_green_refuses_non_success(status: str, needle: str) -> None:
    with pytest.raises(ReleaseGateError, match=needle):
        assert_ci_green(_SHA, status_fn=lambda _sha: status)  # type: ignore[arg-type]


def test_assert_outside_freeze_refuses_inside() -> None:
    with pytest.raises(ReleaseGateError, match="inside freeze"):
        assert_outside_freeze(_DEADLINE - timedelta(hours=1), _gw(), _fx())


def test_assert_outside_freeze_allows_outside() -> None:
    assert_outside_freeze(_DEADLINE - FREEZE_LEAD - timedelta(hours=1), _gw(), _fx())
    assert_outside_freeze(_KICKOFF, _gw(), _fx())


def test_run_smoke_plan_loads_production() -> None:
    result = run_smoke_plan()
    assert result.model
    assert result.plan_horizon >= 1
    assert "loaded production model" in result.detail


def test_run_smoke_plan_fails_on_bad_env(tmp_path: Path) -> None:
    with pytest.raises(ReleaseGateError, match="smoke plan failed"):
        run_smoke_plan(environ={"FPLT_PRODUCTION_MODELS": "no_such_model_xyz"})


def test_evaluate_release_gates_happy_path() -> None:
    now = _DEADLINE - FREEZE_LEAD - timedelta(hours=2)
    result = evaluate_release_gates(
        "main",
        now=now,
        gameweeks=_gw(),
        fixtures=_fx(),
        status_fn=lambda _sha: "success",
        rev_parse=lambda _ref: _SHA,
    )
    assert result.candidate.sha == _SHA
    assert result.history_tag == history_tag_name(_SHA, when=now)
    assert result.freeze is None


def test_evaluate_release_gates_refuses_red_ci() -> None:
    now = _DEADLINE - FREEZE_LEAD - timedelta(hours=2)
    with pytest.raises(ReleaseGateError, match="CI is red"):
        evaluate_release_gates(
            "main",
            now=now,
            gameweeks=_gw(),
            fixtures=_fx(),
            status_fn=lambda _sha: "failure",
            rev_parse=lambda _ref: _SHA,
        )


def test_evaluate_release_gates_refuses_freeze() -> None:
    with pytest.raises(ReleaseGateError, match="inside freeze"):
        evaluate_release_gates(
            "main",
            now=_DEADLINE - timedelta(hours=1),
            gameweeks=_gw(),
            fixtures=_fx(),
            status_fn=lambda _sha: "success",
            rev_parse=lambda _ref: _SHA,
        )


def test_evaluate_release_gates_refuses_failed_smoke() -> None:
    now = _DEADLINE - FREEZE_LEAD - timedelta(hours=2)
    with pytest.raises(ReleaseGateError, match="smoke plan failed"):
        evaluate_release_gates(
            "main",
            now=now,
            gameweeks=_gw(),
            fixtures=_fx(),
            status_fn=lambda _sha: "success",
            rev_parse=lambda _ref: _SHA,
            environ={"FPLT_PRODUCTION_MODELS": "no_such_model_xyz"},
        )


def test_resolve_candidate_rejects_empty() -> None:
    with pytest.raises(ReleaseGateError, match="cannot resolve"):
        resolve_candidate("main", rev_parse=lambda _ref: "")


def test_promote_prod_tags_dry_run_commands() -> None:
    candidate = ReleaseCandidate(ref="main", sha=_SHA, short_sha=_SHA[:7])
    tag = history_tag_name(_SHA, when=datetime(2025, 11, 20, tzinfo=UTC))
    cmds = promote_prod_tags(candidate, history_tag=tag, dry_run=True)
    assert any(tag in c for c in cmds)
    assert any(f"refs/tags/{PROD_TAG}" in c for c in cmds)


def test_promote_prod_tags_runs_via_injector() -> None:
    seen: list[list[str]] = []
    candidate = ReleaseCandidate(ref="main", sha=_SHA, short_sha=_SHA[:7])
    tag = "prod-20251120-abcdef0"
    promote_prod_tags(
        candidate,
        history_tag=tag,
        runner=lambda cmd: seen.append(list(cmd)),
    )
    assert seen[0][:3] == ["git", "tag", "-a"]
    assert seen[1][:4] == ["git", "tag", "-f", "-a"]
    assert seen[1][4] == PROD_TAG


def test_resolve_prod_tag_label() -> None:
    assert resolve_prod_tag_label(environ={}) is None
    assert resolve_prod_tag_label(environ={"FPLT_PROD_TAG": "  "}) is None
    assert resolve_prod_tag_label(environ={"FPLT_PROD_TAG": "prod"}) == "prod"
