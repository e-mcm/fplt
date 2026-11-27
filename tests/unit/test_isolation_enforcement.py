"""Isolation enforcement meta-tests (#306).

Guards for network blocking, pytest-randomly seed reporting, per-test
timeouts, and Hypothesis CI/dev profiles. Extends the #287 host-env strip
guard pattern: removing the enforcement should fail these tests.
"""

from __future__ import annotations

import os
import socket
from pathlib import Path

import pytest
import yaml
from hypothesis import settings as hypothesis_settings
from pytest_socket import SocketBlockedError

_REPO = Path(__file__).resolve().parents[2]
_CI_YML = _REPO / ".github" / "workflows" / "ci.yml"
_PYPROJECT = _REPO / "pyproject.toml"

pytest_plugins = ["pytester"]


def test_socket_create_connection_blocked() -> None:
    """Outbound TCP must fail inside tests with a clear SocketBlockedError."""
    # pytest-socket also blocks getaddrinfo and emits a UserWarning before raise.
    with (
        pytest.warns(UserWarning, match="getaddrinfo"),
        pytest.raises(SocketBlockedError),
    ):
        socket.create_connection(("example.com", 80), timeout=0.5)


def test_randomly_seed_is_configured(pytestconfig: pytest.Config) -> None:
    """pytest-randomly must expose an integer seed (printed in the CI log)."""
    seed = pytestconfig.getoption("randomly_seed")
    assert isinstance(seed, int), f"expected int randomly_seed, got {seed!r}"


def test_timeout_plugin_active(pytestconfig: pytest.Config) -> None:
    """Global timeout must be armed (60s default from pyproject addopts)."""
    timeout = pytestconfig.getoption("timeout")
    assert float(timeout) == 60.0, f"expected 60s global timeout, got {timeout!r}"


def test_sleeping_test_fails_on_timeout(pytester: pytest.Pytester) -> None:
    """A deliberately hanging test must fail via pytest-timeout (stack dump).

    Uses the signal method in a subprocess so the hang is interrupted cleanly
    and we still get a terminal summary (thread dumps can leave the child
    stuck in ``sleep`` without a pytest summary).
    """
    pytester.makepyfile(
        """
        import time

        def test_hangs() -> None:
            time.sleep(30)
        """
    )
    result = pytester.runpytest_subprocess(
        "--timeout=1",
        "--timeout-method=signal",
        "-p",
        "no:cacheprovider",
        "-p",
        "no:randomly",
        "-o",
        "addopts=",
    )
    text = result.stdout.str() + result.stderr.str()
    assert result.ret != 0, text
    assert "timeout" in text.lower(), text
    result.assert_outcomes(failed=1)


def test_hypothesis_profiles_registered() -> None:
    """``ci`` and ``dev`` Hypothesis profiles must stay registered in conftest."""
    ci = hypothesis_settings.get_profile("ci")
    assert ci.deadline is None
    assert ci.print_blob is True
    assert ci.derandomize is False
    assert ci.max_examples == 100

    dev = hypothesis_settings.get_profile("dev")
    assert dev.deadline is None
    assert dev.print_blob is False
    assert dev.max_examples == 25


def test_hypothesis_profile_loaded_from_env() -> None:
    """Active profile must match ``HYPOTHESIS_PROFILE`` (default ``dev``)."""
    expected = os.environ.get("HYPOTHESIS_PROFILE", "dev")
    active = hypothesis_settings()
    profile = hypothesis_settings.get_profile(expected)
    assert active.max_examples == profile.max_examples
    assert active.print_blob == profile.print_blob


def test_ci_yml_sets_hypothesis_profile_ci() -> None:
    """``ci.yml`` test-linux must select the Hypothesis ``ci`` profile (#306)."""
    doc = yaml.safe_load(_CI_YML.read_text(encoding="utf-8"))
    step = next(
        s
        for s in doc["jobs"]["test-linux"]["steps"]
        if s.get("name") == "Pytest (fplt)"
    )
    assert step.get("env", {}).get("HYPOTHESIS_PROFILE") == "ci"


def test_pyproject_addopts_disable_socket_and_timeout() -> None:
    """pyproject addopts must keep socket disable + unix allow + timeout."""
    text = _PYPROJECT.read_text(encoding="utf-8")
    assert "--disable-socket" in text
    assert "--allow-unix-socket" in text
    assert "--timeout=60" in text
    assert 'timeout_method = "thread"' in text
