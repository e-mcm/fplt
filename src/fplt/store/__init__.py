"""Immutable analytic-output store (runs, later catalog / official readers).

Stage B4 introduces ``RunWriter`` under ``fplt.store.runs``. The full
``Store`` read API arrives later (data-model migration step 8).
"""

from fplt.store.runs import RunWriter, mint_run_id, run_dir

__all__ = ["RunWriter", "mint_run_id", "run_dir"]
