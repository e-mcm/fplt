"""Immutable analytic-output store (runs, later catalog / official readers).

Stage B4 introduces ``RunWriter`` and ``push_run`` under ``fplt.store``.
The full ``Store`` read API arrives later (data-model migration step 8).
"""

from fplt.store.push import IncompleteRunError, maybe_push_run, push_run
from fplt.store.runs import RunWriter, mint_run_id, run_dir

__all__ = [
    "IncompleteRunError",
    "RunWriter",
    "maybe_push_run",
    "mint_run_id",
    "push_run",
    "run_dir",
]
