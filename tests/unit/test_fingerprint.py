"""Unit tests for run identity fingerprints (#346)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fplt.models.spec import ModelSpec, load_model_specs
from fplt.schemas.manifests import BuildManifest, BuildTableEntry
from fplt.schemas.staged import STAGED_SCHEMAS
from fplt.store.fingerprint import (
    PLANNER_AND_OUTCOME_TABLES,
    fit_fingerprint,
    fp_index_exists,
    last_completed_gw,
    load_fp_index,
    required_tables,
    run_fingerprint,
    write_fp_index,
)
from fplt.types import GWKey

_TS = datetime(2024, 9, 1, 12, 0, tzinfo=UTC)


def _entry(sha: str) -> BuildTableEntry:
    return BuildTableEntry(sha=sha, rows=1, schema_hash="s" * 64)


def _build(tables: dict[str, str]) -> BuildManifest:
    return BuildManifest(
        build_id="b" * 64,
        created_at=_TS,
        staging_version=1,
        code_sha="abc1234",
        raw_manifest_hash="r" * 64,
        raw_cutoff=_TS,
        tables={name: _entry(sha) for name, sha in tables.items()},
    )


def test_required_tables_includes_planner_set_and_feature_inputs() -> None:
    specs = load_model_specs()
    # rolling_mean has no @feature list; kind inputs + planner set.
    tables = required_tables(specs["rolling_mean_v1"])
    assert PLANNER_AND_OUTCOME_TABLES <= tables
    assert "player_fixture_stats" in tables
    # Feature inputs must be STAGED_SCHEMAS keys (registry contract).
    for name in tables:
        assert name in STAGED_SCHEMAS


def test_required_tables_rejects_unknown_feature_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fplt.features.registry import Feature
    from fplt.store import fingerprint as fp_mod

    class _Schema:  # minimal stand-in; never validated
        pass

    fake = Feature(
        name="_bad_input_feature",
        version=1,
        grain="player_fixture",
        inputs=("not_a_real_table",),
        schema=_Schema,  # type: ignore[arg-type]
        fn=lambda view: pl.DataFrame(),
    )
    monkeypatch.setattr(fp_mod, "ensure_features_registered", lambda: None)

    def _get(name: str, version: int) -> Feature:
        assert name == "_bad_input_feature"
        assert version == 1
        return fake

    monkeypatch.setattr("fplt.features.registry.get_feature", _get)
    spec = ModelSpec(
        name="tmp",
        kind="rolling_mean",
        features=["_bad_input_feature@1"],
    )
    with pytest.raises(ValueError, match="not_a_real_table"):
        required_tables(spec)


def test_run_fingerprint_stable_for_unread_table_change() -> None:
    specs = load_model_specs()
    spec = specs["rolling_mean_v1"]
    needed = required_tables(spec)
    base = {name: f"{i:064d}" for i, name in enumerate(sorted(needed))}
    # understat is not required for rolling_mean.
    with_extra = {**base, "understat_player_matches": "u" * 64}
    changed_unread = {**with_extra, "understat_player_matches": "v" * 64}
    key = GWKey("2024-25", 3)
    fp1 = run_fingerprint(
        spec=spec,
        build=_build(with_extra),
        target_gw=key,
        run_type="predict",
        tables=needed,
    )
    fp2 = run_fingerprint(
        spec=spec,
        build=_build(changed_unread),
        target_gw=key,
        run_type="predict",
        tables=needed,
    )
    assert fp1 == fp2


def test_run_fingerprint_changes_when_required_table_changes() -> None:
    specs = load_model_specs()
    spec = specs["rolling_mean_v1"]
    needed = required_tables(spec)
    base = {name: f"{i:064d}" for i, name in enumerate(sorted(needed))}
    changed = dict(base)
    first = sorted(needed)[0]
    changed[first] = "f" * 64
    key = GWKey("2024-25", 3)
    fp1 = run_fingerprint(
        spec=spec, build=_build(base), target_gw=key, run_type="predict", tables=needed
    )
    fp2 = run_fingerprint(
        spec=spec,
        build=_build(changed),
        target_gw=key,
        run_type="predict",
        tables=needed,
    )
    assert fp1 != fp2


def test_fit_fingerprint_ignores_player_status_churn() -> None:
    specs = load_model_specs()
    spec = specs["rolling_mean_v1"]
    needed = required_tables(spec)
    base = {name: f"{i:064d}" for i, name in enumerate(sorted(needed))}
    churn = dict(base)
    churn["player_status"] = "z" * 64
    completed = GWKey("2024-25", 2)
    fp1 = fit_fingerprint(
        spec=spec, build=_build(base), last_completed=completed, tables=needed
    )
    fp2 = fit_fingerprint(
        spec=spec, build=_build(churn), last_completed=completed, tables=needed
    )
    assert fp1 == fp2


def test_fit_fingerprint_changes_after_gw_completes() -> None:
    specs = load_model_specs()
    spec = specs["rolling_mean_v1"]
    needed = required_tables(spec)
    tables = {name: f"{i:064d}" for i, name in enumerate(sorted(needed))}
    build = _build(tables)
    fp1 = fit_fingerprint(
        spec=spec, build=build, last_completed=GWKey("2024-25", 1), tables=needed
    )
    fp2 = fit_fingerprint(
        spec=spec, build=build, last_completed=GWKey("2024-25", 2), tables=needed
    )
    assert fp1 != fp2


def test_by_fp_index_write_once(tmp_path: Path) -> None:
    fp = "a" * 64
    write_fp_index(tmp_path, fp, "r_test_000001")
    entry = load_fp_index(tmp_path, fp)
    assert entry is not None
    assert entry.run_id == "r_test_000001"
    assert fp_index_exists(tmp_path, fp)
    with pytest.raises(FileExistsError):
        write_fp_index(tmp_path, fp, "r_test_000002")


def test_last_completed_gw_from_gameweeks() -> None:
    frame = pl.DataFrame(
        {
            "season": ["2024-25", "2024-25", "2024-25"],
            "gw": [1, 2, 3],
            "finished": [True, True, False],
            "data_checked": [True, False, False],
        }
    )
    assert last_completed_gw(frame) == GWKey("2024-25", 2)


def test_last_completed_gw_edge_cases() -> None:
    assert last_completed_gw(None) is None
    assert last_completed_gw(pl.DataFrame()) is None
    only_finished = pl.DataFrame({"season": ["2024-25"], "gw": [1], "finished": [True]})
    assert last_completed_gw(only_finished) == GWKey("2024-25", 1)
    only_checked = pl.DataFrame(
        {"season": ["2024-25"], "gw": [4], "data_checked": [True]}
    )
    assert last_completed_gw(only_checked) == GWKey("2024-25", 4)
    no_flags = pl.DataFrame({"season": ["2024-25"], "gw": [1]})
    assert last_completed_gw(no_flags) is None
    none_done = pl.DataFrame(
        {
            "season": ["2024-25"],
            "gw": [1],
            "finished": [False],
            "data_checked": [False],
        }
    )
    assert last_completed_gw(none_done) is None


def test_fp_index_exists_prefers_remote_listing(tmp_path: Path) -> None:
    fp = "b" * 64
    assert not fp_index_exists(tmp_path, fp)
    assert fp_index_exists(tmp_path, fp, remote_listing={f"fp={fp}.json"})
    assert fp_index_exists(tmp_path, fp, remote_listing={f"runs/by_fp/fp={fp}.json"})
    assert load_fp_index(tmp_path, fp) is None


def test_run_fingerprint_missing_table_uses_empty_sha() -> None:
    """Empty staged tables are omitted from builds; absence is sha=\"\"."""
    specs = load_model_specs()
    spec = specs["rolling_mean_v1"]
    key = GWKey("2024-25", 1)
    fp_absent = run_fingerprint(
        spec=spec,
        build=_build({"players": "p" * 64}),
        target_gw=key,
        run_type="predict",
        tables={"players", "gameweeks"},
    )
    fp_present = run_fingerprint(
        spec=spec,
        build=_build({"players": "p" * 64, "gameweeks": "g" * 64}),
        target_gw=key,
        run_type="predict",
        tables={"players", "gameweeks"},
    )
    assert fp_absent != fp_present


def test_required_tables_walks_nested_components() -> None:
    specs = load_model_specs()
    # dc_layered has nested components; union must include kind inputs.
    if "dc_layered_v4" in specs:
        tables = required_tables(specs["dc_layered_v4"])
        assert "match_odds" in tables
        assert PLANNER_AND_OUTCOME_TABLES <= tables


def test_ensure_features_registered_ok() -> None:
    from fplt.store.fingerprint import ensure_features_registered

    ensure_features_registered()
