"""Immutable analytic-output store (runs, curated builds, catalog, experiments).

Stage B4 introduced ``RunWriter`` / ``push_run``. Stage B6 adds curated
builds under ``fplt.store.builds``. B9 adds ``Store.ledger`` (predictions ⋈
outcomes). B12 adds write-once ``official/`` pointers, virtual-squad tables,
and official ``points_pmf`` helpers. Stage C adds ``fplt.store.experiments``,
``catalog/`` rebuild, and the ``Store`` read API (``runs`` / ``official`` /
``load`` / ``experiment`` / ``pit``) plus ``fplt sql`` from migration step 8.
"""

from fplt.store.api import ExperimentHandle, Store
from fplt.store.builds import (
    STAGING_VERSION,
    BuildExistsError,
    CuratedBuildSource,
    build_from_staged,
    latest_build_id,
    load_build_manifest,
)
from fplt.store.catalog import (
    catalog_runs_path,
    list_succeeded_run_ids,
    read_runs_catalog,
    rebuild_runs_catalog,
    write_run_inbox_row,
)
from fplt.store.experiments import (
    ArmReuseKey,
    FoldedStatus,
    find_reusable_arm,
    fold_experiment_status,
    mint_batch_id,
    mint_experiment_id,
    submit_experiment,
)
from fplt.store.legacy import LegacyImportResult, import_legacy, legacy_run_id
from fplt.store.official import (
    LineRunPair,
    OfficialExistsError,
    finalize_official_checkpoint,
    write_applied_record,
    write_intervention_record,
    write_official_pointer,
    write_submitted_record,
)
from fplt.store.points_pmf import (
    POINTS_PMF_MAX,
    POINTS_PMF_MIN,
    PointsPmfGwSchema,
    PointsPmfSchema,
    points_pmf_from_draws,
)
from fplt.store.push import (
    IncompleteRunError,
    maybe_push_build,
    maybe_push_official,
    maybe_push_run,
    push_build,
    push_official,
    push_run,
)
from fplt.store.runs import (
    RunWriter,
    list_succeeded_predict_run_ids,
    mint_run_id,
    run_dir,
)
from fplt.store.sql import open_sql, run_sql
from fplt.store.virtual_squad import (
    chip_remaining,
    resolve_virtual_squad,
    write_squad_state_pair,
)

__all__ = [
    "ArmReuseKey",
    "BuildExistsError",
    "CuratedBuildSource",
    "ExperimentHandle",
    "FoldedStatus",
    "IncompleteRunError",
    "LegacyImportResult",
    "LineRunPair",
    "OfficialExistsError",
    "POINTS_PMF_MAX",
    "POINTS_PMF_MIN",
    "PointsPmfGwSchema",
    "PointsPmfSchema",
    "RunWriter",
    "STAGING_VERSION",
    "Store",
    "build_from_staged",
    "catalog_runs_path",
    "chip_remaining",
    "finalize_official_checkpoint",
    "find_reusable_arm",
    "fold_experiment_status",
    "import_legacy",
    "latest_build_id",
    "legacy_run_id",
    "list_succeeded_predict_run_ids",
    "list_succeeded_run_ids",
    "load_build_manifest",
    "maybe_push_build",
    "maybe_push_official",
    "maybe_push_run",
    "mint_batch_id",
    "mint_experiment_id",
    "mint_run_id",
    "open_sql",
    "points_pmf_from_draws",
    "push_build",
    "push_official",
    "push_run",
    "read_runs_catalog",
    "rebuild_runs_catalog",
    "resolve_virtual_squad",
    "run_dir",
    "run_sql",
    "submit_experiment",
    "write_applied_record",
    "write_intervention_record",
    "write_official_pointer",
    "write_run_inbox_row",
    "write_squad_state_pair",
    "write_submitted_record",
]
