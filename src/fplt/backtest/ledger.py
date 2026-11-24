"""Ledger helpers: column alignment and the read-only view entry point.

Predictions live under ``runs/``; actuals live in curated/staged outcomes.
``Store.ledger()`` joins them at read time. ``read_predictions`` is a thin
alias for that view so older call sites keep working. The former
``append_predictions`` / ``fill_actuals`` mutators are gone (B9 PR2).
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from fplt.schemas.ledger import LEDGER_COLUMNS


def ledger_path(root: Path) -> Path:
    """Legacy parquet path (read-only archive until B10 import)."""
    return root / "ledger" / "predictions" / "part.parquet"


def empty_ledger() -> pl.DataFrame:
    """An empty frame that still matches ``LedgerSchema``."""
    return pl.DataFrame(schema=LEDGER_COLUMNS)


def read_predictions(root: Path) -> pl.DataFrame:
    """Return the ledger view (predictions ⋈ outcomes).

    Prefer ``Store(root).ledger()`` at new call sites. This alias exists so
    boards and optimisers that still import ``read_predictions`` see the same
    view without rewriting every reader in B9.

    Args:
        root: Local data root (``$FPLT_DATA_DIR``).

    Returns:
        ``LedgerSchema`` rows (possibly empty).
    """
    from fplt.store import Store

    return Store(root).ledger()


def align_ledger_columns(frame: pl.DataFrame) -> pl.DataFrame:
    """Pad missing event-prob / actual columns so older ledgers still load.

    Args:
        frame: Ledger-shaped rows that may omit newer nullable columns.

    Returns:
        Frame with every ``LEDGER_COLUMNS`` key present (null-filled when absent).
    """
    extras: list[pl.Expr] = []
    for name, dtype in LEDGER_COLUMNS.items():
        if name in frame.columns:
            continue
        extras.append(pl.lit(None).cast(dtype).alias(name))
    if extras:
        frame = frame.with_columns(extras)
    return frame.select(list(LEDGER_COLUMNS))
