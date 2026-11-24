"""Immutable analytic-output store (runs, curated builds; later catalog).

Stage B4 introduced ``RunWriter`` / ``push_run``. Stage B6 adds curated
builds under ``fplt.store.builds``. B9 adds ``Store.ledger`` (predictions ⋈
outcomes). The rest of the ``Store`` read API arrives in migration step 8.
"""

from fplt.store.api import Store
from fplt.store.builds import (
    STAGING_VERSION,
    BuildExistsError,
    CuratedBuildSource,
    build_from_staged,
    latest_build_id,
    load_build_manifest,
)
from fplt.store.push import (
    IncompleteRunError,
    maybe_push_build,
    maybe_push_run,
    push_build,
    push_run,
)
from fplt.store.runs import RunWriter, mint_run_id, run_dir

__all__ = [
    "BuildExistsError",
    "CuratedBuildSource",
    "IncompleteRunError",
    "RunWriter",
    "STAGING_VERSION",
    "Store",
    "build_from_staged",
    "latest_build_id",
    "load_build_manifest",
    "maybe_push_build",
    "maybe_push_run",
    "mint_run_id",
    "push_build",
    "push_run",
    "run_dir",
]
