"""Process-wide logging setup for CLI and scripts."""

from __future__ import annotations

import logging
import sys
from typing import Literal

_configured = False

LogLevel = Literal["DEBUG", "INFO", "WARNING", "ERROR"]


def configure_logging(
    *,
    verbose: bool = False,
    quiet: bool = False,
    level: LogLevel | None = None,
) -> None:
    """Configure the root logger for fplt operators.

    Default is ``INFO`` (progress for plan / backtest / compare / API).
    ``verbose`` selects ``DEBUG``; ``quiet`` selects ``WARNING``. An explicit
    ``level`` wins over both flags. Idempotent for the process: later calls
    only change the level.

    Args:
        verbose: Log DEBUG (HTTP paths, per-gameweek walk-forward steps).
        quiet: Log WARNING and above only.
        level: Explicit level name; overrides ``verbose`` / ``quiet``.
    """
    global _configured
    if level is not None:
        chosen = getattr(logging, level)
    elif quiet:
        chosen = logging.WARNING
    elif verbose:
        chosen = logging.DEBUG
    else:
        chosen = logging.INFO

    root = logging.getLogger()
    if not _configured:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                fmt="%(asctime)s %(levelname)s [%(name)s] %(message)s",
                datefmt="%H:%M:%S",
            )
        )
        root.addHandler(handler)
        _configured = True
    root.setLevel(chosen)
    # Keep httpx chatter down unless the operator asked for DEBUG.
    logging.getLogger("httpx").setLevel(
        logging.DEBUG if chosen <= logging.DEBUG else logging.WARNING
    )
    logging.getLogger("httpcore").setLevel(
        logging.DEBUG if chosen <= logging.DEBUG else logging.WARNING
    )


def reset_logging_for_tests() -> None:
    """Clear handlers so unit tests can reconfigure cleanly."""
    global _configured
    root = logging.getLogger()
    for handler in list(root.handlers):
        root.removeHandler(handler)
        handler.close()
    _configured = False
