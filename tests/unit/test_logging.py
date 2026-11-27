"""CLI logging configuration."""

from __future__ import annotations

import logging

import pytest

from fplt.utils.logging import (
    SecretRedactionFilter,
    configure_logging,
    reset_logging_for_tests,
)


@pytest.fixture(autouse=True)
def _reset_logging() -> None:
    """Keep the redaction filter from leaking across tests."""
    reset_logging_for_tests()
    yield
    reset_logging_for_tests()


def test_configure_logging_defaults_to_info() -> None:
    configure_logging(secrets=())
    root = logging.getLogger()
    assert root.level == logging.INFO
    assert logging.getLogger("httpx").level == logging.WARNING
    assert any(
        isinstance(f, SecretRedactionFilter) for h in root.handlers for f in h.filters
    )


def test_verbose_enables_debug() -> None:
    configure_logging(verbose=True, secrets=())
    assert logging.getLogger().level == logging.DEBUG
    assert logging.getLogger("httpx").level == logging.DEBUG


def test_quiet_raises_threshold() -> None:
    configure_logging(quiet=True, secrets=())
    assert logging.getLogger().level == logging.WARNING


def test_reconfigure_changes_level_without_extra_handlers() -> None:
    configure_logging(secrets=())
    root = logging.getLogger()
    n_handlers = len(root.handlers)
    configure_logging(verbose=True, secrets=("sekrit",))
    assert root.level == logging.DEBUG
    assert len(root.handlers) == n_handlers
    redactors = [
        f
        for h in root.handlers
        for f in h.filters
        if isinstance(f, SecretRedactionFilter)
    ]
    assert len(redactors) >= 1
    assert redactors[0].secrets == ("sekrit",)
