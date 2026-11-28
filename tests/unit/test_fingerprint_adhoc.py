"""Light adhoc fingerprint runs (#374): primary only, no PMF, quiet window."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from fplt.automation.checkpoints import in_background_quiet_window
from fplt.automation.ops_data import LineFpDecision, decisions_for_slate
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
from fplt.store.fingerprint import required_tables, run_fingerprint
from fplt.types import GWKey

_TS = datetime(2024, 9, 1, 12, 0, tzinfo=UTC)
_DEADLINE = datetime(2024, 9, 2, 17, 0, tzinfo=UTC)


def _ops_with_candidates(*, model: str = "rolling_mean_v1") -> ProductionOps:
    pipeline = live_pipeline_defaults(model=model, horizon=1)
    return ProductionOps(
        models=[model],
        plan_horizon=1,
        pipeline=pipeline,
        lines=[
            OfficialLine(
                line="primary",
                spec=model,
                horizon=1,
                role="primary",
                squad_mode="real",
            ),
            OfficialLine(
                line="candidate",
                spec=model,
                horizon=3,
                role="candidate",
                squad_mode="virtual",
            ),
        ],
        min_run_interval_minutes=60,
        max_build_age_minutes=30,
        background_quiet_minutes=45,
    )


def _write_build(tmp_path: Path, tables: dict[str, str]) -> BuildManifest:
    from fplt.store.builds import STAGING_VERSION

    entries: dict[str, BuildTableEntry] = {}
    for name, sha in tables.items():
        dest = table_sha_path(tmp_path, name, sha)
        dest.parent.mkdir(parents=True, exist_ok=True)
        pl.DataFrame({"x": [1]}).write_parquet(dest)
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


def test_run_fingerprint_includes_kind(tmp_path: Path) -> None:
    specs = load_model_specs()
    spec = specs["rolling_mean_v1"]
    needed = required_tables(spec)
    table_map = {name: f"{i:064d}" for i, name in enumerate(sorted(needed))}
    build = _write_build(tmp_path, table_map)
    key = GWKey("2024-25", 3)
    official = run_fingerprint(
        spec=spec, build=build, target_gw=key, run_type="predict", kind="official"
    )
    adhoc = run_fingerprint(
        spec=spec, build=build, target_gw=key, run_type="predict", kind="adhoc"
    )
    assert official != adhoc
    assert len(official) == 64


def test_decisions_for_fingerprint_primary_only(tmp_path: Path) -> None:
    ops = _ops_with_candidates()
    specs = load_model_specs()
    needed = required_tables(specs["rolling_mean_v1"])
    table_map = {name: f"{i:064d}" for i, name in enumerate(sorted(needed))}
    build = _write_build(tmp_path, table_map)
    key = GWKey("2024-25", 3)
    primary = ops.primary_line()
    fp_decisions = decisions_for_slate(
        ops,
        build,
        key,
        data_dir=tmp_path,
        kind="adhoc",
        lines=(primary,),
    )
    assert len(fp_decisions) == 1
    assert fp_decisions[0].line == "primary"
    official = decisions_for_slate(ops, build, key, data_dir=tmp_path, kind="official")
    assert len(official) == 2
    assert fp_decisions[0].predict_fp != official[0].predict_fp


def test_in_background_quiet_window() -> None:
    done: frozenset[str] = frozenset()
    quiet = timedelta(minutes=45)
    # Far from any checkpoint → not quiet.
    assert not in_background_quiet_window(
        _DEADLINE - timedelta(hours=30), _DEADLINE, done, quiet
    )
    # Within 45 min of T−24h fire (deadline − 24h) → quiet.
    t24_fire = _DEADLINE - timedelta(hours=24)
    assert in_background_quiet_window(
        t24_fire - timedelta(minutes=30), _DEADLINE, done, quiet
    )
    # Already due (past fire time) → quiet.
    assert in_background_quiet_window(
        t24_fire + timedelta(minutes=5), _DEADLINE, done, quiet
    )
    # Done T−24h; far from T−6h → not quiet.
    after_t24 = frozenset({"T-24h"})
    assert not in_background_quiet_window(
        t24_fire + timedelta(hours=10), _DEADLINE, after_t24, quiet
    )
    assert not in_background_quiet_window(_TS, None, done, quiet)
    # After the deadline → not quiet (nothing left to protect).
    assert not in_background_quiet_window(
        _DEADLINE + timedelta(minutes=1), _DEADLINE, done, quiet
    )


def test_ensure_fingerprint_slate_adhoc_no_pmf(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-checkpoint fp change → one adhoc primary run, no points_pmf."""
    from fplt.automation.official_slate import ensure_slate_runs
    from fplt.data.store import DataStore
    from fplt.optimise.public_squad import ResolvedSquad
    from fplt.optimise.state import SquadState
    from fplt.schemas.manifests import AsOfDeadline, RunManifest
    from fplt.store.fingerprint import write_fp_index
    from fplt.store.official import LineRunPair
    from fplt.store.runs import run_dir
    from tests.builders import build_league

    build_league(DataStore(tmp_path))
    ops = _ops_with_candidates()
    key = GWKey("2024-25", 2)
    specs = load_model_specs()
    needed = required_tables(specs["rolling_mean_v1"])
    table_map = {name: f"{i:064d}" for i, name in enumerate(sorted(needed))}
    build = _write_build(tmp_path, table_map)

    predict_kwargs: list[dict[str, Any]] = []
    plan_kwargs: list[dict[str, Any]] = []

    def fake_predict(*args: Any, **kwargs: Any) -> str:
        predict_kwargs.append(kwargs)
        run_id = "r_fp_pred"
        path = run_dir(tmp_path, run_id)
        path.mkdir(parents=True, exist_ok=True)
        (path / "_SUCCESS").write_text("", encoding="utf-8")
        manifest = RunManifest(
            run_id=run_id,
            kind=kwargs["kind"],
            mode="live",
            purpose=kwargs["purpose"] or "",
            run_type="predict",
            spec_hash="s" * 64,
            spec_label="rolling_mean_v1",
            model_id="rolling_mean_v1",
            code_sha="abc",
            lockfile_hash="l" * 64,
            as_of=[
                AsOfDeadline(
                    season=key.season,
                    gw=key.gw,
                    deadline_utc=_DEADLINE,
                )
            ],
            horizon=kwargs["horizon"],
            host="test",
            started_at=_TS,
            finished_at=_TS,
            status="succeeded",
            fp=kwargs["fp"],
            build_id=build.build_id,
        )
        (path / "manifest.json").write_text(
            manifest.model_dump_json(indent=2) + "\n", encoding="utf-8"
        )
        write_fp_index(tmp_path, kwargs["fp"], run_id)
        return run_id

    def fake_plan(*args: Any, **kwargs: Any) -> str:
        plan_kwargs.append(kwargs)
        run_id = "r_fp_plan"
        write_fp_index(tmp_path, kwargs["fp"], run_id)
        return run_id

    monkeypatch.setattr(
        "fplt.automation.official_slate.pull_required_tables", lambda *a, **k: []
    )
    monkeypatch.setattr(
        "fplt.automation.official_slate.push_fp_index", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "fplt.automation.official_slate.record_line_run", lambda *a, **k: None
    )
    monkeypatch.setattr(
        "fplt.automation.official_slate.resolve_squad_state",
        lambda *a, **k: ResolvedSquad(
            state=SquadState(bank=0, free_transfers=1, ft_bank_cap=5, owned=[]),
            source="yaml",
        ),
    )
    monkeypatch.setattr("fplt.automation.official_slate._slate_predict", fake_predict)
    monkeypatch.setattr("fplt.automation.official_slate._slate_plan", fake_plan)

    decision = LineFpDecision(
        line="primary",
        spec_name="rolling_mean_v1",
        predict_fp="a" * 64,
        plan_fp="b" * 64,
        skip=False,
    )
    bid, pairs = ensure_slate_runs(
        tmp_path,
        ops,
        key,
        build_id=build.build_id,
        decisions=[decision],
        checkpoint=None,
        slate="fingerprint",
        now=_DEADLINE - timedelta(hours=12),
    )
    assert bid == build.build_id
    assert len(pairs) == 1
    assert pairs[0] == LineRunPair(
        line="primary",
        role="primary",
        squad_mode="real",
        predict_run_id="r_fp_pred",
        plan_run_id="r_fp_plan",
    )
    assert len(predict_kwargs) == 1
    assert predict_kwargs[0]["kind"] == "adhoc"
    assert predict_kwargs[0]["purpose"] == "fingerprint"
    assert predict_kwargs[0]["write_pmf"] is False
    assert predict_kwargs[0]["horizon"] == 1  # primary, not candidate h=3
    assert plan_kwargs[0]["kind"] == "adhoc"
    assert plan_kwargs[0]["purpose"] == "fingerprint"
    pred_dir = run_dir(tmp_path, "r_fp_pred")
    assert not (pred_dir / "predictions" / "points_pmf.parquet").exists()
    assert not (pred_dir / "predictions" / "points_pmf_gw.parquet").exists()

    with pytest.raises(ValueError, match="fingerprint slate"):
        ensure_slate_runs(
            tmp_path,
            ops,
            key,
            build_id=build.build_id,
            decisions=[decision],
            checkpoint="T-90m",
            slate="fingerprint",
        )


def test_ops_tick_quiet_window_skips_fingerprint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.automation.ops_tick import run_ops_tick

    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("JOB_STARTED_AT", "1700000000")
    get_settings.cache_clear()
    ops = _ops_with_candidates()
    specs = load_model_specs()
    needed = required_tables(specs["rolling_mean_v1"])
    table_map = {name: f"{i:064d}" for i, name in enumerate(sorted(needed))}
    build = _write_build(tmp_path, table_map)
    gw_sha = table_map["gameweeks"]
    pl.DataFrame(
        {
            "season": ["2024-25"],
            "gw": [3],
            "deadline_time": [_DEADLINE],
            "finished": [False],
            "data_checked": [False],
            "available_at": [_DEADLINE - timedelta(days=20)],
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
        "fplt.automation.ops_tick.enforce_training_history", lambda *a, **k: None
    )
    monkeypatch.setattr("fplt.automation.ops_tick.catch_up", lambda *a, **k: [])
    monkeypatch.setattr("fplt.automation.ops_tick.pre_deadline", lambda *a, **k: None)
    monkeypatch.setattr("fplt.automation.ops_tick.pending_checked", lambda *a, **k: [])
    monkeypatch.setattr("fplt.automation.ops_tick.plans_exist", lambda *a, **k: True)
    # All checkpoints done so none are due; quiet via proximity to… wait,
    # if all done, quiet is False. Use none done and now within 45m of T-24h.
    monkeypatch.setattr(
        "fplt.automation.ops_tick.resolve_done_sets",
        lambda *a, **k: (frozenset(), frozenset()),
    )
    # T−24h fires at deadline−24h; sit 20 min before that fire → quiet.
    now = _DEADLINE - timedelta(hours=24) + timedelta(minutes=-20)
    result = run_ops_tick(tmp_path, now=now, notify=False)
    assert ensures == []
    assert result.checkpoints_run == []
    get_settings.cache_clear()


def test_ops_tick_fingerprint_slate_kwarg(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.automation.ops_tick import run_ops_tick

    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("JOB_STARTED_AT", "1700000000")
    get_settings.cache_clear()
    ops = _ops_with_candidates()
    specs = load_model_specs()
    needed = required_tables(specs["rolling_mean_v1"])
    table_map = {name: f"{i:064d}" for i, name in enumerate(sorted(needed))}
    build = _write_build(tmp_path, table_map)
    gw_sha = table_map["gameweeks"]
    pl.DataFrame(
        {
            "season": ["2024-25"],
            "gw": [3],
            "deadline_time": [_DEADLINE],
            "finished": [False],
            "data_checked": [False],
            "available_at": [_DEADLINE - timedelta(days=20)],
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
        "fplt.automation.ops_tick.enforce_training_history", lambda *a, **k: None
    )
    monkeypatch.setattr("fplt.automation.ops_tick.catch_up", lambda *a, **k: [])
    monkeypatch.setattr("fplt.automation.ops_tick.pre_deadline", lambda *a, **k: None)
    monkeypatch.setattr("fplt.automation.ops_tick.pending_checked", lambda *a, **k: [])
    monkeypatch.setattr("fplt.automation.ops_tick.plans_exist", lambda *a, **k: True)
    monkeypatch.setattr(
        "fplt.automation.ops_tick.resolve_done_sets",
        lambda *a, **k: (
            frozenset({"T-24h", "T-6h", "T-90m", "T-30m"}),
            frozenset(),
        ),
    )
    result = run_ops_tick(tmp_path, now=_DEADLINE - timedelta(hours=12), notify=False)
    assert ensures
    assert ensures[0].get("slate") == "fingerprint"
    assert ensures[0].get("checkpoint") is None
    assert result.checkpoints_run == []
    get_settings.cache_clear()
