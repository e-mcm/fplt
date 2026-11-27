"""Host CI / FPLT env must not leak into unit tests (#287)."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
import yaml

import tests.conftest as conftest
from fplt.data.parity import check_raw_parity
from fplt.data.remote import Remote, RemoteError, github_requires_r2
from fplt.data.snapshots import fetch_raw
from fplt.settings import Settings

_REPO = Path(__file__).resolve().parents[2]
_CI_YML = _REPO / ".github" / "workflows" / "ci.yml"


def test_host_ci_and_fplt_env_stripped() -> None:
    """Autouse fixture must clear GITHUB_*, CI, and FPLT_* for every test.

    Fails if the fixture is removed (marker stays False) or if any of those
    names remain visible inside a test body.
    """
    assert conftest.HOST_CI_ENV_STRIP_STATE["ok"] is True
    leaked = [
        key
        for key in os.environ
        if key == "CI" or key.startswith("GITHUB_") or key.startswith("FPLT_")
    ]
    assert leaked == [], f"host CI/FPLT env leaked into test: {leaked}"


@pytest.mark.parametrize(
    ("ref", "event", "expected"),
    [
        ("refs/heads/main", "push", True),
        ("refs/heads/main", "schedule", True),
        ("refs/heads/main", "workflow_dispatch", True),
        ("refs/heads/main", "pull_request", False),
        ("refs/heads/cursor/feature", "push", False),
        (None, None, False),
    ],
    ids=[
        "main-push",
        "main-schedule",
        "main-workflow_dispatch",
        "pull_request",
        "other-branch",
        "not-on-gha",
    ],
)
def test_github_requires_r2_matrix(
    monkeypatch: pytest.MonkeyPatch,
    ref: str | None,
    event: str | None,
    expected: bool,
) -> None:
    """``github_requires_r2`` covers main events, PRs, other branches, and local."""
    if ref is None:
        monkeypatch.delenv("GITHUB_REF", raising=False)
    else:
        monkeypatch.setenv("GITHUB_REF", ref)
    if event is None:
        monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    else:
        monkeypatch.setenv("GITHUB_EVENT_NAME", event)
    assert github_requires_r2() is expected


def test_soft_skip_after_fixture_clears_main_host_env() -> None:
    """Regression (#287): soft-skip works once the host main ref is stripped.

    Push-to-main CI sets ``GITHUB_REF=refs/heads/main`` / ``GITHUB_ACTIONS=true``
    in the process environment; before the autouse strip that made soft-skip
    raise. After the strip the same call returns ``None``.
    """
    assert "GITHUB_REF" not in os.environ
    assert github_requires_r2() is False
    assert Remote.from_settings(Settings(data_bucket="fplt-data")) is None


def test_four_soft_skip_paths_raise_when_main_ref_set(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Regression (#287): the four red-main call sites raise under an explicit main ref.

    Setting ``GITHUB_REF`` / ``GITHUB_ACTIONS`` first reproduces the pre-fix
    failure mode locally; after the autouse strip, only an explicit ``setenv``
    re-enables require-R2.
    """
    monkeypatch.setenv("GITHUB_REF", "refs/heads/main")
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("GITHUB_EVENT_NAME", raising=False)
    settings = Settings(data_bucket="fplt-data")

    with pytest.raises(RemoteError, match="credentials incomplete on main"):
        Remote.from_settings(settings)

    with pytest.raises(RemoteError, match="credentials incomplete on main"):
        fetch_raw(tmp_path / "raw-auto", settings, source="auto", remote=str(tmp_path))

    with pytest.raises(RemoteError, match="credentials incomplete on main"):
        fetch_raw(tmp_path / "raw-bucket", settings, source="bucket")

    with pytest.raises(RemoteError, match="credentials incomplete on main"):
        check_raw_parity(settings, remote=str(tmp_path / "missing"), branch="snapshots")


def test_ci_yml_notifies_on_main_push_gate_failure() -> None:
    """``ci.yml`` gate posts ntfy when a push to main fails the gate (#287)."""
    doc = yaml.safe_load(_CI_YML.read_text(encoding="utf-8"))
    steps = doc["jobs"]["gate"]["steps"]
    notifies = [s for s in steps if s.get("uses") == "./.github/actions/notify-failure"]
    assert len(notifies) == 1
    step = notifies[0]
    cond = step.get("if", "")
    assert "failure()" in cond
    assert "push" in cond
    assert "main" in cond
    assert step["with"]["ntfy_url"] == "${{ secrets.NTFY_URL }}"
