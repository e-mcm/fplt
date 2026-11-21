"""CLI logging configuration."""

from __future__ import annotations

import logging

from fplt.utils.logging import configure_logging, reset_logging_for_tests


def test_configure_logging_defaults_to_info() -> None:
    reset_logging_for_tests()
    configure_logging()
    assert logging.getLogger().level == logging.INFO
    assert logging.getLogger("httpx").level == logging.WARNING


def test_verbose_enables_debug() -> None:
    reset_logging_for_tests()
    configure_logging(verbose=True)
    assert logging.getLogger().level == logging.DEBUG
    assert logging.getLogger("httpx").level == logging.DEBUG


def test_quiet_raises_threshold() -> None:
    reset_logging_for_tests()
    configure_logging(quiet=True)
    assert logging.getLogger().level == logging.WARNING


def test_reconfigure_changes_level_without_extra_handlers() -> None:
    reset_logging_for_tests()
    configure_logging()
    configure_logging(verbose=True)
    root = logging.getLogger()
    assert root.level == logging.DEBUG
    assert len(root.handlers) == 1
