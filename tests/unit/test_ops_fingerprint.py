"""Ops fingerprint path: skip-if-done, required tables, throttle, freshness."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from fplt.automation.ops_data import (
    StaleBuildError,
    build_is_fresh,
    decisions_for_slate,
    enforce_build_freshness,
    pull_required_tables,
    tables_for_slate,
    throttle_allows,
)
from fplt.automation.production import OfficialLine, ProductionOps
from fplt.models.pipeline_spec import live_pipeline_defaults
from fplt.models.spec import load_model_specs
from fplt.schemas.manifests import (
    BuildManifest,
    BuildTableEntry,
    CuratedLatestPointer,
)
from fplt.settings import get_settings
from fplt.store.builds import build_manifest_path, table_sha_path
from fplt.store.fingerprint import (
    required_tables,
    run_fingerprint,
    write_fp_index,
)
from fplt.types import GWKey

_TS = datetime(2024, 9, 1, 12, 0, tzinfo=UTC)


def _tiny_ops(*, model: str = "rolling_mean_v1", horizon: int = 1) -> ProductionOps:
    pipeline = live_pipeline_defaults(model=model, horizon=horizon)
    return ProductionOps(
        models=[model],
        plan_horizon=horizon,
        pipeline=pipeline,
        lines=[
            OfficialLine(
                line="primary",
                spec=model,
                horizon=horizon,
                role="primary",
                squad_mode="real",
            )
        ],
        min_run_interval_minutes=60,
        max_build_age_minutes=30,
    )


def _write_build(tmp_path: Path, tables: dict[str, str]) -> BuildManifest:
    from fplt.store.builds import STAGING_VERSION

    entries: dict[str, BuildTableEntry] = {}
    for name, sha in tables.items():
        frame = pl.DataFrame({"x": [1]})
        # Minimal parquet bytes via helper when available; else write directly.
        dest = table_sha_path(tmp_path, name, sha)
        dest.parent.mkdir(parents=True, exist_ok=True)
        frame.write_parquet(dest)
        entries[name] = BuildTableEntry(sha=sha, rows=1, schema_hash="s" * 64)
    manifest = BuildManifest(
        build_id="c" * 64,
        created_at=_TS,
        staging_version=STAGING_VERSION,
        code_sha="abc1234",
        raw_manifest_hash="r" * 64,
        raw_cutoff=_TS,
        tables=entries,
    )
    path = build_manifest_path(tmp_path, manifest.build_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(manifest.model_dump_json(indent=2) + "\n", encoding="utf-8")
    return manifest


def test_pull_required_tables_only(tmp_path: Path) -> None:
    needed = {"gameweeks", "players"}
    extra = "understat_player_matches"
    tables = {
        "gameweeks": "a" * 64,
        "players": "b" * 64,
        extra: "c" * 64,
    }
    build = _write_build(tmp_path, tables)
    # Place curated files; pull_required_tables with remote=None hydrates only.
    written = pull_required_tables(tmp_path, build.build_id, needed)
    assert set(written) == needed
    assert (tmp_path / "staged" / "gameweeks" / "part.parquet").is_file()
    assert (tmp_path / "staged" / "players" / "part.parquet").is_file()
    assert not (tmp_path / "staged" / extra).exists()


def test_second_tick_skips_when_fp_unchanged(tmp_path: Path) -> None:
    ops = _tiny_ops()
    specs = load_model_specs()
    spec = specs["rolling_mean_v1"]
    needed = required_tables(spec)
    table_map = {name: f"{i:064d}" for i, name in enumerate(sorted(needed))}
    build = _write_build(tmp_path, table_map)
    key = GWKey("2024-25", 3)
    decisions = decisions_for_slate(ops, build, key, data_dir=tmp_path)
    assert len(decisions) == 1
    assert decisions[0].skip is False
    write_fp_index(tmp_path, decisions[0].predict_fp, "r_pred")
    write_fp_index(tmp_path, decisions[0].plan_fp, "r_plan")
    again = decisions_for_slate(ops, build, key, data_dir=tmp_path)
    assert again[0].skip is True
    assert again[0].predict_run_id == "r_pred"


def test_unread_table_change_keeps_fp(tmp_path: Path) -> None:
    ops = _tiny_ops()
    specs = load_model_specs()
    spec = specs["rolling_mean_v1"]
    needed = required_tables(spec)
    table_map = {name: f"{i:064d}" for i, name in enumerate(sorted(needed))}
    table_map["understat_player_matches"] = "u" * 64
    build = _write_build(tmp_path, table_map)
    key = GWKey("2024-25", 3)
    fp1 = run_fingerprint(spec=spec, build=build, target_gw=key, run_type="predict")
    # Change unread table sha in a new build object.
    tables = dict(build.tables)
    tables["understat_player_matches"] = BuildTableEntry(
        sha="v" * 64, rows=1, schema_hash="s" * 64
    )
    build2 = build.model_copy(update={"tables": tables})
    fp2 = run_fingerprint(spec=spec, build=build2, target_gw=key, run_type="predict")
    assert fp1 == fp2
    assert "understat_player_matches" not in tables_for_slate(ops)


def test_throttle_and_freshness_branches(tmp_path: Path) -> None:
    get_settings.cache_clear()
    now = datetime(2024, 9, 1, 15, 0, tzinfo=UTC)
    assert throttle_allows(
        tmp_path,
        "primary",
        now=now,
        min_interval=timedelta(minutes=60),
        checkpoint=False,
    )
    from fplt.automation.ops_data import record_line_run

    record_line_run(tmp_path, "primary", now=now)
    assert not throttle_allows(
        tmp_path,
        "primary",
        now=now + timedelta(minutes=30),
        min_interval=timedelta(minutes=60),
        checkpoint=False,
    )
    assert throttle_allows(
        tmp_path,
        "primary",
        now=now + timedelta(minutes=30),
        min_interval=timedelta(minutes=60),
        checkpoint=True,
    )

    pointer = CuratedLatestPointer(
        build_id="c" * 64,
        created_at=now - timedelta(minutes=45),
        raw_cutoff=now - timedelta(minutes=45),
        code_sha="abc",
    )
    assert not build_is_fresh(pointer, now=now, max_age=timedelta(minutes=30))
    ops = _tiny_ops()
    with pytest.raises(StaleBuildError, match="stale"):
        enforce_build_freshness(
            pointer,
            now=now,
            ops=ops,
            settings=get_settings(),
            notify=False,
            checkpoint="T-90m",
        )


def test_hydrated_build_matches_staged_predictions(tmp_path: Path) -> None:
    """Parity: PIT predictions from hydrated curated tables match staged (#346)."""
    from fplt.backtest.predict_live import fit_or_load_predictor
    from fplt.backtest.walk_forward import horizon_for
    from fplt.data.pit import PITData
    from fplt.data.store import DataStore
    from fplt.models.spec import load_model_specs
    from fplt.store.builds import build_from_staged, hydrate_staged_from_build
    from tests.builders import build_league

    store = build_league(DataStore(tmp_path))
    key = GWKey("2024-25", 2)
    spec = load_model_specs()["rolling_mean_v1"]
    view = PITData(store).view(key)
    model = fit_or_load_predictor(tmp_path, spec, key, view)
    from_staged = model.predict(view, horizon_for(key, 1)).sort(
        ["player_key", "fixture_id"]
    )

    build = build_from_staged(tmp_path, code_sha="parity")
    # Clear staged and rehydrate tables present in the build ∩ required.
    import shutil

    shutil.rmtree(tmp_path / "staged")
    present = set(build.tables) & required_tables(spec)
    hydrate_staged_from_build(tmp_path, build.build_id, tables=present)
    view2 = PITData(DataStore(tmp_path)).view(key)
    model2 = fit_or_load_predictor(tmp_path, spec, key, view2)
    from_build = model2.predict(view2, horizon_for(key, 1)).sort(
        ["player_key", "fixture_id"]
    )
    assert from_staged.equals(from_build)


def test_ops_tick_idle_no_table_pull(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Same build + spec + GW → second path does nothing (no required pulls)."""
    from fplt.automation.ops_tick import run_ops_tick
    from fplt.schemas.manifests import CuratedLatestPointer
    from fplt.store.fingerprint import required_tables

    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("JOB_STARTED_AT", "1700000000")
    get_settings.cache_clear()

    ops = _tiny_ops()
    specs = load_model_specs()
    needed = required_tables(specs["rolling_mean_v1"])
    table_map = {name: f"{i:064d}" for i, name in enumerate(sorted(needed))}
    # gameweeks must be a real-ish frame for next_upcoming_deadline.
    build = _write_build(tmp_path, table_map)
    # Overwrite gameweeks parquet with usable schema rows.
    gw_sha = table_map["gameweeks"]
    deadline = datetime(2024, 9, 2, 17, 0, tzinfo=UTC)
    pl.DataFrame(
        {
            "season": ["2024-25"],
            "gw": [3],
            "deadline_time": [deadline],
            "finished": [False],
            "data_checked": [False],
            "available_at": [deadline - timedelta(days=20)],
        }
    ).write_parquet(table_sha_path(tmp_path, "gameweeks", gw_sha))

    pointer = CuratedLatestPointer(
        build_id=build.build_id,
        created_at=_TS,
        raw_cutoff=_TS,
        code_sha="abc",
    )
    (tmp_path / "status" / "curated").mkdir(parents=True)
    (tmp_path / "status" / "curated" / "latest.json").write_text(
        pointer.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    key = GWKey("2024-25", 3)
    decisions = decisions_for_slate(ops, build, key, data_dir=tmp_path)
    write_fp_index(tmp_path, decisions[0].predict_fp, "r_pred")
    write_fp_index(tmp_path, decisions[0].plan_fp, "r_plan")

    pulls: list[Any] = []

    def fake_pull(*args: Any, **kwargs: Any) -> list[str]:
        pulls.append((args, kwargs))
        # Still hydrate gameweeks for deadline discovery on first call pattern.
        tables = args[2] if len(args) > 2 else kwargs.get("tables", set())
        if tables == {"gameweeks"} or tables == set(tables) == {"gameweeks"}:
            from fplt.store.builds import hydrate_staged_from_build

            return hydrate_staged_from_build(
                tmp_path, build.build_id, tables=frozenset({"gameweeks"})
            )
        raise AssertionError(f"unexpected pull_required_tables: {tables!r}")

    monkeypatch.setattr("fplt.automation.ops_tick.pull_required_tables", fake_pull)
    monkeypatch.setattr("fplt.automation.ops_tick.load_production_ops", lambda: ops)
    monkeypatch.setattr(
        "fplt.automation.ops_tick.load_ops_build_context",
        lambda *a, **k: type(
            "Ctx",
            (),
            {
                "pointer": pointer,
                "build": build,
                "remote_by_fp": frozenset(),
            },
        )(),
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.resolve_done_sets",
        lambda *a, **k: (frozenset({"T-24h", "T-6h", "T-90m", "T-30m"}), frozenset()),
    )

    result = run_ops_tick(
        tmp_path,
        now=deadline - timedelta(hours=12),
        notify=False,
    )
    assert result.checkpoints_run == []
    # Only the cheap gameweeks bootstrap pull — never full required tables.
    assert len(pulls) == 1
    assert pulls[0][0][2] == {"gameweeks"} or pulls[0][1].get("tables") == {"gameweeks"}


def test_load_ops_build_context_none_and_local(tmp_path: Path) -> None:
    from fplt.automation.ops_data import load_ops_build_context

    assert load_ops_build_context(tmp_path, remote=None) is None
    build = _write_build(tmp_path, {"gameweeks": "g" * 64})
    pointer = CuratedLatestPointer(
        build_id=build.build_id,
        created_at=_TS,
        raw_cutoff=_TS,
        code_sha="abc",
    )
    (tmp_path / "status" / "curated").mkdir(parents=True)
    (tmp_path / "status" / "curated" / "latest.json").write_text(
        pointer.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    ctx = load_ops_build_context(tmp_path, remote=None)
    assert ctx is not None
    assert ctx.build.build_id == build.build_id
    assert ctx.remote_by_fp == frozenset()


def test_load_ops_build_context_remote_lsf_fail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.automation.ops_data import load_ops_build_context
    from fplt.data.remote import RemoteError

    build = _write_build(tmp_path, {"gameweeks": "g" * 64})
    pointer = CuratedLatestPointer(
        build_id=build.build_id,
        created_at=_TS,
        raw_cutoff=_TS,
        code_sha="abc",
    )
    (tmp_path / "status" / "curated").mkdir(parents=True)
    (tmp_path / "status" / "curated" / "latest.json").write_text(
        pointer.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )

    class _Remote:
        pass

    monkeypatch.setattr(
        "fplt.automation.ops_data.pull_curated_latest", lambda *a, **k: pointer
    )
    monkeypatch.setattr(
        "fplt.automation.ops_data.lsf",
        lambda *a, **k: (_ for _ in ()).throw(RemoteError("nope")),
    )
    ctx = load_ops_build_context(tmp_path, remote=_Remote())  # type: ignore[arg-type]
    assert ctx is not None
    assert ctx.remote_by_fp == frozenset()


def test_pull_required_tables_missing_manifest(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        pull_required_tables(tmp_path, "d" * 64, {"gameweeks"}, remote=None)


def test_push_fp_index_noop_paths(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.automation.ops_data import push_fp_index

    push_fp_index(tmp_path, "e" * 64, remote=None)
    write_fp_index(tmp_path, "e" * 64, "r1")
    pushed: list[Any] = []

    def fake_push(remote: Any, local: Any, remote_key: str) -> None:
        pushed.append(remote_key)

    monkeypatch.setattr("fplt.data.remote.push", fake_push)
    monkeypatch.setattr("fplt.data.remote.Remote.from_settings", lambda *a, **k: None)
    push_fp_index(tmp_path, "e" * 64)
    assert pushed == []

    class _R:
        pass

    push_fp_index(tmp_path, "e" * 64, remote=_R())  # type: ignore[arg-type]
    assert pushed == ["runs/by_fp"]


def test_freshness_notify_and_naive_created(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.automation.ops_data import enforce_build_freshness, line_tables

    now = datetime(2024, 9, 1, 15, 0, tzinfo=UTC)
    pointer = CuratedLatestPointer(
        build_id="c" * 64,
        created_at=datetime(2024, 9, 1, 14, 45),  # naive, within 30m
        raw_cutoff=_TS,
        code_sha="abc",
    )
    assert build_is_fresh(pointer, now=now, max_age=timedelta(minutes=30))
    notes: list[str] = []
    monkeypatch.setattr(
        "fplt.automation.ops_data.notify_recommendation",
        lambda *a, **k: notes.append(a[1] if len(a) > 1 else "x"),
    )
    stale = CuratedLatestPointer(
        build_id="c" * 64,
        created_at=now - timedelta(minutes=90),
        raw_cutoff=_TS,
        code_sha="abc",
    )
    with pytest.raises(StaleBuildError):
        enforce_build_freshness(
            stale,
            now=now,
            ops=_tiny_ops(),
            settings=get_settings(),
            notify=True,
            checkpoint="T-30m",
        )
    assert notes
    ops = _tiny_ops()
    assert "players" in line_tables(ops.lines[0])


def test_throttle_invalid_and_naive(tmp_path: Path) -> None:
    from fplt.automation.ops_data import record_line_run

    path = tmp_path / "status" / "ops" / "last_run_by_line.json"
    path.parent.mkdir(parents=True)
    path.write_text("[1, 2]\n", encoding="utf-8")
    now = datetime(2024, 9, 1, 15, 0, tzinfo=UTC)
    assert throttle_allows(
        tmp_path,
        "primary",
        now=now,
        min_interval=timedelta(minutes=60),
        checkpoint=False,
    )
    # Naive stamp in file.
    path.write_text('{"primary": "2024-09-01T14:30:00"}\n', encoding="utf-8")
    assert not throttle_allows(
        tmp_path,
        "primary",
        now=now.replace(tzinfo=None),
        min_interval=timedelta(minutes=60),
        checkpoint=False,
    )
    record_line_run(tmp_path, "primary", now=now.replace(tzinfo=None))


def test_hydrate_missing_table_raises(tmp_path: Path) -> None:
    from fplt.store.builds import hydrate_staged_from_build

    build = _write_build(tmp_path, {"gameweeks": "g" * 64})
    with pytest.raises(KeyError, match="missing from build"):
        hydrate_staged_from_build(
            tmp_path, build.build_id, tables=frozenset({"players"})
        )


def test_ensure_build_manifest_missing_local(tmp_path: Path) -> None:
    from fplt.automation.ops_data import _ensure_build_manifest

    with pytest.raises(FileNotFoundError):
        _ensure_build_manifest(tmp_path, "z" * 64, remote=None)


def test_ops_tick_idle_without_curated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.automation.ops_tick import run_ops_tick

    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("JOB_STARTED_AT", "1700000000")
    get_settings.cache_clear()
    monkeypatch.setattr(
        "fplt.automation.ops_tick.load_ops_build_context", lambda *a, **k: None
    )
    monkeypatch.setattr("fplt.automation.ops_tick.load_production_ops", _tiny_ops)
    result = run_ops_tick(tmp_path, now=_TS, notify=False)
    assert result.checkpoints_run == []
    assert result.models == ["rolling_mean_v1"]
    get_settings.cache_clear()


def test_ops_tick_ensure_when_fp_missing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.automation.ops_data import LineFpDecision
    from fplt.automation.ops_tick import run_ops_tick
    from fplt.schemas.manifests import CuratedLatestPointer

    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("JOB_STARTED_AT", "1700000000")
    get_settings.cache_clear()

    ops = _tiny_ops()
    specs = load_model_specs()
    needed = required_tables(specs["rolling_mean_v1"])
    table_map = {name: f"{i:064d}" for i, name in enumerate(sorted(needed))}
    build = _write_build(tmp_path, table_map)
    gw_sha = table_map["gameweeks"]
    deadline = datetime(2024, 9, 2, 17, 0, tzinfo=UTC)
    pl.DataFrame(
        {
            "season": ["2024-25"],
            "gw": [3],
            "deadline_time": [deadline],
            "finished": [False],
            "data_checked": [False],
            "available_at": [deadline - timedelta(days=20)],
        }
    ).write_parquet(table_sha_path(tmp_path, "gameweeks", gw_sha))
    pointer = CuratedLatestPointer(
        build_id=build.build_id,
        created_at=datetime.now(UTC),
        raw_cutoff=_TS,
        code_sha="abc",
    )
    decision = LineFpDecision(
        line="primary",
        spec_name="rolling_mean_v1",
        predict_fp="p" * 64,
        plan_fp="q" * 64,
        skip=False,
    )
    ensures: list[Any] = []

    def fake_ensure(*args: Any, **kwargs: Any) -> tuple[str, tuple[()]]:
        ensures.append(kwargs)
        return build.build_id, ()

    monkeypatch.setattr("fplt.automation.ops_tick.load_production_ops", lambda: ops)
    monkeypatch.setattr(
        "fplt.automation.ops_tick.load_ops_build_context",
        lambda *a, **k: type(
            "Ctx",
            (),
            {"pointer": pointer, "build": build, "remote_by_fp": frozenset()},
        )(),
    )

    def fake_pull(*args: Any, **kwargs: Any) -> list[str]:
        tables = args[2] if len(args) > 2 else kwargs.get("tables", set())
        if tables == {"gameweeks"}:
            from fplt.store.builds import hydrate_staged_from_build

            return hydrate_staged_from_build(
                tmp_path, build.build_id, tables=frozenset({"gameweeks"})
            )
        return []

    monkeypatch.setattr("fplt.automation.ops_tick.pull_required_tables", fake_pull)
    monkeypatch.setattr(
        "fplt.automation.ops_tick.decisions_for_slate", lambda *a, **k: [decision]
    )
    monkeypatch.setattr("fplt.automation.ops_tick.ensure_slate_runs", fake_ensure)
    monkeypatch.setattr(
        "fplt.automation.ops_tick.enforce_training_history",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.resolve_done_sets",
        lambda *a, **k: (
            frozenset({"T-24h", "T-6h", "T-90m", "T-30m"}),
            frozenset(),
        ),
    )
    result = run_ops_tick(tmp_path, now=deadline - timedelta(hours=12), notify=False)
    assert ensures
    assert result.checkpoints_run == []
    get_settings.cache_clear()


def test_ops_tick_stale_build_skips_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.automation.ops_tick import run_ops_tick
    from fplt.data.store import DataStore
    from fplt.schemas.manifests import CuratedLatestPointer

    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("JOB_STARTED_AT", "1700000000")
    get_settings.cache_clear()
    ops = _tiny_ops()
    deadline = datetime(2024, 9, 2, 17, 0, tzinfo=UTC)
    store = DataStore(tmp_path)
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": ["2024-25"],
                "gw": [3],
                "deadline_time": [deadline],
                "finished": [False],
                "data_checked": [False],
                "available_at": [deadline - timedelta(days=20)],
            }
        ),
    )
    build = _write_build(tmp_path, {"gameweeks": "g" * 64})
    pointer = CuratedLatestPointer(
        build_id=build.build_id,
        created_at=deadline - timedelta(hours=5),
        raw_cutoff=_TS,
        code_sha="abc",
    )
    monkeypatch.setattr("fplt.automation.ops_tick.load_production_ops", lambda: ops)
    monkeypatch.setattr(
        "fplt.automation.ops_tick.load_ops_build_context",
        lambda *a, **k: type(
            "Ctx",
            (),
            {"pointer": pointer, "build": build, "remote_by_fp": frozenset()},
        )(),
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.pull_required_tables", lambda *a, **k: []
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.decisions_for_slate", lambda *a, **k: []
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.ensure_slate_runs",
        lambda *a, **k: (build.build_id, ()),
    )
    monkeypatch.setattr("fplt.automation.ops_tick.catch_up", lambda *a, **k: [])
    monkeypatch.setattr("fplt.automation.ops_tick.pre_deadline", lambda *a, **k: None)
    monkeypatch.setattr("fplt.automation.ops_tick.pending_checked", lambda *a, **k: [])
    monkeypatch.setattr("fplt.automation.ops_tick.plans_exist", lambda *a, **k: True)
    monkeypatch.setattr(
        "fplt.automation.ops_tick.enforce_training_history", lambda *a, **k: None
    )
    from fplt.automation.checkpoints import (
        write_notified_marker,
        write_plan_checkpoint_marker,
    )

    key = GWKey("2024-25", 3)
    write_plan_checkpoint_marker(tmp_path, key, "T-24h", primary_line="primary")
    write_plan_checkpoint_marker(tmp_path, key, "T-6h", primary_line="primary")
    write_notified_marker(tmp_path, key)
    official_calls: list[str] = []

    def fake_official(*a: Any, **k: Any) -> Any:
        official_calls.append("called")
        raise AssertionError("should not plan on stale build")

    monkeypatch.setattr(
        "fplt.automation.ops_tick.run_official_checkpoint", fake_official
    )
    result = run_ops_tick(tmp_path, now=deadline - timedelta(minutes=80), notify=False)
    assert "T-90m" not in result.checkpoints_run
    assert official_calls == []
    get_settings.cache_clear()


def _four_line_ops(model: str = "rolling_mean_v1") -> ProductionOps:
    pipeline = live_pipeline_defaults(model=model, horizon=3)
    return ProductionOps(
        models=[model],
        plan_horizon=3,
        pipeline=pipeline,
        lines=[
            OfficialLine(
                line="v4_h3", spec=model, horizon=3, role="primary", squad_mode="real"
            ),
            *(
                OfficialLine(
                    line=f"v4_h{h}",
                    spec=model,
                    horizon=h,
                    role="candidate",
                    squad_mode="virtual",
                )
                for h in (5, 7, 9)
            ),
        ],
        min_run_interval_minutes=60,
        max_build_age_minutes=30,
    )


def _slate_build(tmp_path: Path) -> BuildManifest:
    spec = load_model_specs()["rolling_mean_v1"]
    needed = required_tables(spec)
    return _write_build(
        tmp_path, {name: f"{i:064d}" for i, name in enumerate(sorted(needed))}
    )


def test_plan_fp_distinct_per_line_sharing_a_spec(tmp_path: Path) -> None:
    """Regression #380: v4_h3/h5/h7/h9 shared one plan_fp, so pointers reused
    the h3 plan for every line (by_fp already present for fp=9c528d235e9f)."""
    build = _slate_build(tmp_path)
    decisions = decisions_for_slate(
        _four_line_ops(), build, GWKey("2024-25", 3), data_dir=tmp_path
    )
    assert len({d.plan_fp for d in decisions}) == 4
    # One predict per spec at the longest horizon: predict fp is shared.
    assert len({d.predict_fp for d in decisions}) == 1


def test_plan_fp_changes_with_squad_digest_only(tmp_path: Path) -> None:
    build = _slate_build(tmp_path)
    ops = _four_line_ops()
    key = GWKey("2024-25", 3)
    a = decisions_for_slate(ops, build, key, data_dir=tmp_path, squad_digest="s1")
    same = decisions_for_slate(ops, build, key, data_dir=tmp_path, squad_digest="s1")
    b = decisions_for_slate(ops, build, key, data_dir=tmp_path, squad_digest="s2")
    assert [d.plan_fp for d in a] == [d.plan_fp for d in same]
    assert all(x.plan_fp != y.plan_fp for x, y in zip(a, b, strict=True))
    assert [d.predict_fp for d in a] == [d.predict_fp for d in b]
    # A transfer (new digest) means a previously indexed plan is not reused.
    write_fp_index(tmp_path, a[0].predict_fp, "r_pred")
    write_fp_index(tmp_path, a[0].plan_fp, "r_plan_s1")
    after = decisions_for_slate(ops, build, key, data_dir=tmp_path, squad_digest="s2")
    assert after[0].skip is False


def test_run_fingerprint_extra_is_optional_and_hashed(tmp_path: Path) -> None:
    build = _slate_build(tmp_path)
    spec = load_model_specs()["rolling_mean_v1"]
    key = GWKey("2024-25", 3)
    base = run_fingerprint(spec=spec, build=build, target_gw=key, run_type="plan")
    again = run_fingerprint(
        spec=spec, build=build, target_gw=key, run_type="plan", extra=None
    )
    tagged = run_fingerprint(
        spec=spec, build=build, target_gw=key, run_type="plan", extra={"line": "x"}
    )
    assert base == again
    assert tagged != base


class _FakeFPL:
    def __init__(
        self,
        history: dict[str, Any],
        transfers: list[dict[str, Any]],
        *,
        fail: bool = False,
    ) -> None:
        self.history = history
        self.transfers = transfers
        self.fail = fail

    def entry_history(self, entry_id: int) -> dict[str, Any]:
        from fplt.data.fpl_client import FPLAPIError

        if self.fail:
            raise FPLAPIError("boom")
        return self.history

    def entry_transfers(self, entry_id: int) -> list[dict[str, Any]]:
        return self.transfers


def _history(points: int = 50, bank: int = 30) -> dict[str, Any]:
    return {
        "current": [
            {
                "event": 12,
                "points": points,
                "total_points": 600 + points,
                "rank": 1000,
                "bank": bank,
                "event_transfers": 1,
                "event_transfers_cost": 0,
            }
        ],
        "chips": [{"name": "wildcard", "event": 5, "time": "x"}],
    }


_TRANSFER = {
    "event": 12,
    "element_in": 10,
    "element_out": 20,
    "element_in_cost": 55,
    "element_out_cost": 60,
    "time": "2026-11-20T10:00:00Z",
}


def test_entry_state_digest_tracks_squad_not_points() -> None:
    from fplt.automation.ops_data import entry_state_digest
    from fplt.settings import Settings

    cfg = Settings(entry_id=42)
    base = entry_state_digest(cfg, api=_FakeFPL(_history(), [_TRANSFER]))  # type: ignore[arg-type]
    live_points = entry_state_digest(
        cfg,
        api=_FakeFPL(_history(points=80), [_TRANSFER]),  # type: ignore[arg-type]
    )
    moved = entry_state_digest(
        cfg,
        api=_FakeFPL(  # type: ignore[arg-type]
            _history(bank=25), [_TRANSFER, {**_TRANSFER, "event": 13}]
        ),
    )
    assert base is not None
    assert len(base) == 64
    assert live_points == base  # live scoring does not re-plan
    assert moved != base  # a transfer does
    assert entry_state_digest(Settings(entry_id=None)) is None


def test_safe_entry_state_digest_marks_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fplt.automation import ops_data
    from fplt.data.fpl_client import FPLAPIError
    from fplt.settings import Settings

    def boom(*_: object, **__: object) -> str:
        raise FPLAPIError("down")

    monkeypatch.setattr(ops_data, "entry_state_digest", boom)
    got = ops_data.safe_entry_state_digest(Settings(entry_id=42))
    assert got == ops_data.SQUAD_DIGEST_UNAVAILABLE


def test_reused_plan_must_match_line(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A skipped line must never point at another line's plan run (#380)."""
    from types import SimpleNamespace

    from fplt.automation import official_slate

    line_h5 = _four_line_ops().lines[1]
    monkeypatch.setattr(
        official_slate,
        "load_run_manifest",
        lambda *_: SimpleNamespace(line="v4_h3", plan_horizon=3),
    )
    assert official_slate._plan_run_matches_line(tmp_path, "r", line_h5) is False
    monkeypatch.setattr(
        official_slate,
        "load_run_manifest",
        lambda *_: SimpleNamespace(line="v4_h5", plan_horizon=5),
    )
    assert official_slate._plan_run_matches_line(tmp_path, "r", line_h5) is True

    def missing(*_: object) -> None:
        raise FileNotFoundError("r")

    monkeypatch.setattr(official_slate, "load_run_manifest", missing)
    assert official_slate._plan_run_matches_line(tmp_path, "r", line_h5) is True
