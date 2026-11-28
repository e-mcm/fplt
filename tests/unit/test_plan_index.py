"""Catch-up plan-done index: local plans/ fast path + remote catalog (#371)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fplt.automation.cycle import pending_checked, plans_exist
from fplt.automation.plan_index import (
    catalog_has_plans,
    fetch_remote_plan_catalog,
    plan_index_remotes,
    resolve_plan_catalog,
)
from fplt.data.remote import Remote
from fplt.data.store import DataStore
from fplt.schemas.catalog import CATALOG_RUN_COLUMNS, CatalogRunSchema
from fplt.settings import Settings, get_settings
from fplt.store.catalog import catalog_runs_path
from fplt.types import GWKey
from fplt.utils.frames import require_dataframe

_MODEL = "rolling_mean_v1"
_TS = datetime(2026, 8, 1, tzinfo=UTC)


def _one_step_plan(model_id: str, season: str, gw: int) -> pl.DataFrame:
    """Minimal ``PlanSchema`` rows for one deadline at horizon 1."""
    return pl.DataFrame(
        [
            {
                "model_id": model_id,
                "as_of_season": season,
                "as_of_gw": gw,
                "target_gw": gw,
                "step": 0,
                "player_key": 1,
                "position": "MID",
                "team_key": 1,
                "in_xi": True,
                "captain": True,
                "vice": False,
                "bought": False,
                "sold": False,
                "xpts": 1.0,
                "hit": 0.0,
            }
        ]
    )


def _gameweeks_only(root: Path, rows: list[tuple[str, int, bool]]) -> DataStore:
    """Data dir whose only staged table is ``gameweeks`` (season, gw, checked)."""
    store = DataStore(root)
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": [season for season, _, _ in rows],
                "gw": [gw for _, gw, _ in rows],
                "deadline_time": [
                    datetime(int(season[:4]), 8, 1, tzinfo=UTC)
                    for season, gw, _ in rows
                ],
                "finished": [checked for _, _, checked in rows],
                "data_checked": [checked for _, _, checked in rows],
                "available_at": [_TS] * len(rows),
                "source": ["test"] * len(rows),
            }
        ),
    )
    return store


def _plan_catalog_row(
    *,
    run_id: str,
    spec_label: str,
    season: str,
    gw: int,
    horizon: int,
) -> dict[str, object]:
    return {
        "run_id": run_id,
        "kind": "adhoc",
        "mode": "live",
        "purpose": f"plan {spec_label}",
        "experiment_id": None,
        "arm": None,
        "run_type": "plan",
        "spec_hash": "a" * 64,
        "spec_label": spec_label,
        "model_id": "deadbeef",
        "line": None,
        "plan_horizon": horizon,
        "squad_mode": None,
        "prev_run_id": None,
        "code_sha": "abc1234",
        "code_dirty": False,
        "prod_tag": None,
        "fplt_mc_version": None,
        "lockfile_hash": "b" * 64,
        "build_id": None,
        "as_of_season": season,
        "as_of_gw": gw,
        "deadline_utc": _TS,
        "n_as_of": 1,
        "horizon": horizon,
        "checkpoint": None,
        "host": "test",
        "started_at": _TS,
        "finished_at": _TS,
        "status": "succeeded",
        "parent_run_id": None,
    }


def _plan_catalog(
    spec_label: str, season: str, gws: list[int], *, horizon: int = 1
) -> pl.DataFrame:
    rows = [
        _plan_catalog_row(
            run_id=f"r_plan_gw{gw:02d}",
            spec_label=spec_label,
            season=season,
            gw=gw,
            horizon=horizon,
        )
        for gw in gws
    ]
    frame = CatalogRunSchema.validate(pl.DataFrame(rows, schema=CATALOG_RUN_COLUMNS))
    return require_dataframe(frame)


def test_catalog_has_plans_matches_spec_season_gw_horizon() -> None:
    catalog = _plan_catalog(_MODEL, "2026-27", [1, 2, 3], horizon=1)
    assert catalog_has_plans(catalog, [_MODEL], GWKey("2026-27", 2), 1) is True
    assert catalog_has_plans(catalog, [_MODEL], GWKey("2026-27", 4), 1) is False
    assert catalog_has_plans(catalog, [_MODEL], GWKey("2026-27", 2), 3) is False
    assert catalog_has_plans(catalog, ["other"], GWKey("2026-27", 2), 1) is False


def test_catalog_has_plans_falls_back_to_horizon_when_plan_horizon_null() -> None:
    catalog = _plan_catalog(_MODEL, "2026-27", [1], horizon=2)
    catalog = catalog.with_columns(pl.lit(None).cast(pl.Int64).alias("plan_horizon"))
    assert catalog_has_plans(catalog, [_MODEL], GWKey("2026-27", 1), 2) is True
    assert catalog_has_plans(catalog, [_MODEL], GWKey("2026-27", 1), 1) is False


def test_plan_index_remotes_empty_without_data_bucket() -> None:
    assert plan_index_remotes(Settings()) == []


def test_plan_index_remotes_prefers_injected_list() -> None:
    remote = Remote(target="/tmp/lab", kind="lab", env={})
    assert plan_index_remotes(Settings(), remotes=[remote]) == [remote]


def test_resolve_plan_catalog_ignores_injected_frame_without_bucket() -> None:
    catalog = _plan_catalog(_MODEL, "2026-27", [1], horizon=1)
    assert resolve_plan_catalog(settings=Settings(), remote_catalog=catalog).is_empty()


def test_resolve_plan_catalog_uses_injected_frame_when_bucket_set() -> None:
    catalog = _plan_catalog(_MODEL, "2026-27", [1], horizon=1)
    settings = Settings(data_bucket="fplt-data")
    got = resolve_plan_catalog(settings=settings, remote_catalog=catalog)
    assert got.height == 1


def test_pending_checked_uses_remote_catalog_when_local_plans_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#371: cold plans/ + remote index for GW1–3 → pending starts at GW4."""
    store = _gameweeks_only(tmp_path, [("2026-27", gw, True) for gw in range(1, 6)])
    catalog = _plan_catalog(_MODEL, "2026-27", [1, 2, 3], horizon=1)
    monkeypatch.setenv("FPLT_DATA_BUCKET", "fplt-data")
    get_settings.cache_clear()
    try:
        waiting = pending_checked(
            store,
            tmp_path,
            [_MODEL],
            horizon=1,
            remote_catalog=catalog,
        )
        assert waiting == [GWKey("2026-27", 4), GWKey("2026-27", 5)]
        assert (
            plans_exist(
                tmp_path,
                [_MODEL],
                GWKey("2026-27", 3),
                1,
                remote_catalog=catalog,
            )
            is True
        )
        assert (
            plans_exist(
                tmp_path,
                [_MODEL],
                GWKey("2026-27", 4),
                1,
                remote_catalog=catalog,
            )
            is False
        )
    finally:
        get_settings.cache_clear()


def test_pending_checked_unchanged_without_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """#371: no data bucket → local-only behaviour even if a catalog is passed."""
    monkeypatch.delenv("FPLT_DATA_BUCKET", raising=False)
    get_settings.cache_clear()
    try:
        store = _gameweeks_only(tmp_path, [("2026-27", gw, True) for gw in range(1, 5)])
        catalog = _plan_catalog(_MODEL, "2026-27", [1, 2, 3], horizon=1)
        waiting = pending_checked(
            store,
            tmp_path,
            [_MODEL],
            horizon=1,
            settings=Settings(),
            remote_catalog=catalog,
        )
        assert waiting == [GWKey("2026-27", gw) for gw in range(1, 5)]
    finally:
        get_settings.cache_clear()


def test_fetch_remote_plan_catalog_from_local_dir_remote(tmp_path: Path) -> None:
    """Local-dir remote holding catalog/runs.parquet is readable without R2."""
    bucket = tmp_path / "bucket"
    catalog_dir = bucket / "catalog"
    catalog_dir.mkdir(parents=True)
    catalog = _plan_catalog(_MODEL, "2026-27", [1, 2, 3], horizon=1)
    catalog.write_parquet(catalog_runs_path(bucket))
    remote = Remote(target=str(bucket), kind="lab", env={})
    frame = fetch_remote_plan_catalog(remotes=[remote], cache_dir=tmp_path / "cache")
    assert catalog_has_plans(frame, [_MODEL], GWKey("2026-27", 1), 1)
    assert not catalog_has_plans(frame, [_MODEL], GWKey("2026-27", 4), 1)


def test_pending_checked_with_local_dir_remote(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """End-to-end: fresh data dir, remote catalog GW1–3, pending from GW4."""
    data = tmp_path / "data"
    data.mkdir()
    store = _gameweeks_only(data, [("2026-27", gw, True) for gw in range(1, 6)])
    bucket = tmp_path / "bucket"
    (bucket / "catalog").mkdir(parents=True)
    _plan_catalog(_MODEL, "2026-27", [1, 2, 3], horizon=1).write_parquet(
        catalog_runs_path(bucket)
    )
    remote = Remote(target=str(bucket), kind="lab", env={})
    monkeypatch.setenv("FPLT_DATA_BUCKET", "fplt-data")
    get_settings.cache_clear()
    try:
        waiting = pending_checked(store, data, [_MODEL], horizon=1, remotes=[remote])
        assert waiting == [GWKey("2026-27", 4), GWKey("2026-27", 5)]
    finally:
        get_settings.cache_clear()


def test_fetch_remote_plan_catalog_soft_skips_pull_errors(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.data.remote import RemoteError

    remote = Remote(target=str(tmp_path / "missing"), kind="lab", env={})

    def boom(*_a: object, **_k: object) -> list[Path]:
        raise RemoteError("directory not found")

    monkeypatch.setattr("fplt.automation.plan_index.pull", boom)
    frame = fetch_remote_plan_catalog(remotes=[remote], cache_dir=tmp_path / "cache")
    assert frame.is_empty()


def test_fetch_remote_plan_catalog_unions_lab_and_data(tmp_path: Path) -> None:
    lab = tmp_path / "lab"
    data = tmp_path / "data"
    for root, gws in ((lab, [1]), (data, [2])):
        (root / "catalog").mkdir(parents=True)
        _plan_catalog(_MODEL, "2026-27", gws, horizon=1).write_parquet(
            catalog_runs_path(root)
        )
    frame = fetch_remote_plan_catalog(
        remotes=[
            Remote(target=str(lab), kind="lab", env={}),
            Remote(target=str(data), kind="data", env={}),
        ],
        cache_dir=tmp_path / "cache",
    )
    assert catalog_has_plans(frame, [_MODEL], GWKey("2026-27", 1), 1)
    assert catalog_has_plans(frame, [_MODEL], GWKey("2026-27", 2), 1)


def test_fetch_remote_plan_catalog_empty_remote_and_default_cache(
    tmp_path: Path,
) -> None:
    empty_bucket = tmp_path / "empty"
    empty_bucket.mkdir()
    # Prefix missing → rclone pull may create empty dest; catalog read is empty.
    frame = fetch_remote_plan_catalog(
        remotes=[Remote(target=str(empty_bucket), kind="lab", env={})]
    )
    assert frame.is_empty()
    assert fetch_remote_plan_catalog(remotes=[]).is_empty()


def test_plan_index_remotes_resolves_from_settings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    lab = Remote(target=str(tmp_path / "lab"), kind="lab", env={})
    data = Remote(target=str(tmp_path / "data"), kind="data", env={})

    def fake_from_settings(
        _settings: Settings, bucket: str | None = None
    ) -> Remote | None:
        if bucket == "fplt-lab":
            return lab
        if bucket == "fplt-data":
            return data
        return None

    monkeypatch.setattr(
        "fplt.automation.plan_index.Remote.from_settings", fake_from_settings
    )
    settings = Settings(data_bucket="fplt-data", lab_bucket="fplt-lab")
    assert plan_index_remotes(settings) == [lab, data]
    # Soft-skip None + de-dupe same target.
    monkeypatch.setattr(
        "fplt.automation.plan_index.Remote.from_settings",
        lambda _s, bucket=None: lab,
    )
    assert plan_index_remotes(settings) == [lab]
    # lab_bucket unset → skip that iteration, keep data.
    monkeypatch.setattr(
        "fplt.automation.plan_index.Remote.from_settings", fake_from_settings
    )
    assert plan_index_remotes(Settings(data_bucket="fplt-data")) == [data]


def test_pending_checked_skips_local_plans_and_empty_pending(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.models.spec import load_model_specs
    from fplt.store.runs import write_plan_run

    store = _gameweeks_only(tmp_path, [("2026-27", 1, True), ("2026-27", 2, True)])
    spec = load_model_specs()[_MODEL]
    for gw in (1, 2):
        # A local plan *run* (runs/), not a legacy plans/ parquet (cut-over).
        write_plan_run(
            tmp_path,
            spec=spec,
            as_of=GWKey("2026-27", gw),
            deadline_utc=datetime(2026, 8, 1, tzinfo=UTC),
            horizon=1,
            frame=_one_step_plan(spec.model_id, "2026-27", gw),
        )
    monkeypatch.setenv("FPLT_DATA_BUCKET", "fplt-data")
    get_settings.cache_clear()
    try:
        assert pending_checked(store, tmp_path, [_MODEL], horizon=1) == []
        assert plans_exist(tmp_path, [_MODEL], GWKey("2026-27", 1), 1) is True
    finally:
        get_settings.cache_clear()
