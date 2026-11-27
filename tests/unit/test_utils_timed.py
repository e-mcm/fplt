"""Coverage for ``fplt.utils.timed``."""

from __future__ import annotations

import logging

import pytest

from fplt.utils import timed


def test_timed_logs_elapsed(caplog: pytest.LogCaptureFixture) -> None:
    """Decorator returns the wrapped result and logs duration."""

    @timed
    def add(left: int, right: int) -> int:
        return left + right

    with caplog.at_level(logging.INFO, logger="fplt.utils"):
        assert add(2, 3) == 5
    assert any("add finished in" in record.message for record in caplog.records)
