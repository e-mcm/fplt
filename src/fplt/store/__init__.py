"""Immutable analytic-output store (runs, curated builds; later catalog).

Stage B4 introduced ``RunWriter`` / ``push_run``. Stage B6 adds curated
builds under ``fplt.store.builds``. The full ``Store`` read API arrives later
(data-model migration step 8).
"""

from fplt.store.builds import (
    STAGING_VERSION,
    BuildExistsError,
    build_from_staged,
    load_build_manifest,
)
from fplt.store.push import IncompleteRunError, maybe_push_run, push_run
from fplt.store.runs import RunWriter, mint_run_id, run_dir

__all__ = [
    "BuildExistsError",
    "IncompleteRunError",
    "RunWriter",
    "STAGING_VERSION",
    "build_from_staged",
    "load_build_manifest",
    "maybe_push_run",
    "mint_run_id",
    "push_run",
    "run_dir",
]
