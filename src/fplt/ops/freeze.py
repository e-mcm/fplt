"""Prod freeze hook for ops-12 / Renovate (ops-04).

``fplt.automation.deps_equivalence.check_prod_freeze`` imports
``fplt.ops.freeze.in_prod_freeze`` when present. This module wraps the pure
freeze windows in ``fplt.automation.freeze`` against staged gameweeks/fixtures.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl

from fplt.automation.freeze import format_freeze_block, in_freeze_window
from fplt.data.store import DataStore
from fplt.settings import get_settings


def in_prod_freeze(
    *,
    now: datetime | None = None,
    data_dir: Path | None = None,
) -> tuple[bool, str]:
    """Return whether the moving ``prod`` tag must stay frozen.

    Args:
        now: Clock reading (default: UTC now).
        data_dir: Staged data root (default: ``Settings.data_dir``).

    Returns:
        ``(frozen, detail)``. ``frozen`` is true inside any freeze window.
        Empty / missing staged tables mean outside freeze.
    """
    moment = now if now is not None else datetime.now(UTC)
    root = get_settings().data_dir if data_dir is None else data_dir
    store = DataStore(root)
    gameweeks = pl.DataFrame()
    fixtures = pl.DataFrame()
    if store.path("gameweeks").exists():
        try:
            gameweeks = store.read("gameweeks")
        except Exception:  # noqa: BLE001 — treat unreadable as empty
            gameweeks = pl.DataFrame()
    if store.path("fixtures").exists():
        try:
            fixtures = store.read("fixtures")
        except Exception:  # noqa: BLE001 — treat unreadable as empty
            fixtures = pl.DataFrame()
    window = in_freeze_window(moment, gameweeks, fixtures)
    if window is None:
        return False, "outside freeze"
    return True, format_freeze_block(window, now=moment)
