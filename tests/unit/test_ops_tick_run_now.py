"""ops-tick ``run_now`` manual-<stamp> preview (#364), with #346 skip-if-done."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from fplt.automation.checkpoints import (
    notified_path,
    write_notified_marker,
    write_plan_checkpoint_marker,
)
from fplt.automation.official_slate import OfficialCheckpointResult
from fplt.automation.ops_data import LineFpDecision
from fplt.automation.ops_tick import manual_checkpoint_id, run_ops_tick
from fplt.automation.production import OfficialLine, ProductionOps
from fplt.data.store import DataStore
from fplt.models.pipeline_spec import live_pipeline_defaults
from fplt.optimise.state import SquadState
from fplt.schemas.manifests import (
    BuildManifest,
    BuildTableEntry,
    CuratedLatestPointer,
    OfficialPointer,
)
from fplt.settings import Settings, get_settings
from fplt.store.builds import STAGING_VERSION
from fplt.store.official import LineRunPair, official_checkpoint_dir
from fplt.types import GWKey

_DEADLINE = datetime(2024, 9, 1, 17, 0, tzinfo=UTC)
_PRIMARY = "e2e_primary"
_TS = datetime(2024, 8, 1, tzinfo=UTC)
_TABLE = BuildTableEntry(sha="a" * 64, rows=1, schema_hash="b" * 64)


def _tiny_ops(*, n_lines: int = 2) -> ProductionOps:
    pipeline = live_pipeline_defaults(model="rolling_mean_v1", horizon=1)
    lines = [
        OfficialLine(
            line=_PRIMARY,
            spec="rolling_mean_v1",
            horizon=1,
            role="primary",
            squad_mode="real",
        )
    ]
    if n_lines > 1:
        lines.append(
            OfficialLine(
                line="e2e_h5",
                spec="rolling_mean_v1",
                horizon=1,
                role="candidate",
                squad_mode="virtual",
            )
        )
    return ProductionOps(
        models=["rolling_mean_v1"],
        plan_horizon=1,
        pipeline=pipeline,
        lines=lines,
        min_prior_seasons=1,
    )


def _league(root: Path) -> None:
    store = DataStore(root)
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": ["2024-25"],
                "gw": [3],
                "deadline_time": [_DEADLINE],
                "finished": [False],
                "data_checked": [False],
                "available_at": [_DEADLINE - timedelta(days=20)],
            }
        ),
    )


def _build_ctx(build_id: str = "build-manual") -> Any:
    build = BuildManifest(
        build_id=build_id,
        created_at=_TS,
        staging_version=STAGING_VERSION,
        code_sha="abc",
        raw_manifest_hash="d" * 64,
        raw_cutoff=_TS,
        tables={"gameweeks": _TABLE},
    )
    pointer = CuratedLatestPointer(
        build_id=build_id,
        created_at=_TS,
        raw_cutoff=_TS,
        code_sha="abc",
    )
    return type(
        "Ctx",
        (),
        {"pointer": pointer, "build": build, "remote_by_fp": frozenset()},
    )()


def _patch_curated(
    monkeypatch: pytest.MonkeyPatch,
    *,
    decisions: list[LineFpDecision] | None = None,
) -> None:
    monkeypatch.setattr("fplt.automation.ops_tick.load_production_ops", _tiny_ops)
    monkeypatch.setattr(
        "fplt.automation.ops_tick.load_ops_build_context",
        lambda *a, **k: _build_ctx(),
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.pull_required_tables", lambda *a, **k: []
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.decisions_for_slate",
        lambda *a, **k: decisions if decisions is not None else [],
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.enforce_training_history",
        lambda *a, **k: None,
    )


def test_manual_checkpoint_id_second_resolution() -> None:
    stamp = manual_checkpoint_id(datetime(2026, 11, 28, 7, 30, 15, tzinfo=UTC))
    assert stamp == "manual-20261128T073015Z"


def test_run_now_writes_manual_pointers_without_side_effects(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_now`` writes every slate line under manual-* and leaves T-* alone."""
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("JOB_STARTED_AT", "1700000000")
    get_settings.cache_clear()
    _league(tmp_path)
    _patch_curated(monkeypatch)

    key = GWKey("2024-25", 3)
    write_plan_checkpoint_marker(tmp_path, key, "T-24h", primary_line=_PRIMARY)
    write_plan_checkpoint_marker(tmp_path, key, "T-6h", primary_line=_PRIMARY)
    write_notified_marker(tmp_path, key)
    before_t24 = (
        official_checkpoint_dir(tmp_path, season="2024-25", gw=3, checkpoint="T-24h")
        / f"line={_PRIMARY}.json"
    ).read_text(encoding="utf-8")
    before_notified = notified_path(tmp_path, key).read_text(encoding="utf-8")

    calls: list[dict[str, Any]] = []
    notify_calls: list[str] = []
    now = _DEADLINE - timedelta(days=3)

    def fake_official(
        data_dir: Path,
        ops: ProductionOps,
        gw_key: GWKey,
        *,
        settings: Settings | None = None,
        chip_arm: str = "optimised",
        build_id: str | None = None,
        prod_tag: str | None = None,
        decisions: list[LineFpDecision] | None = None,
        now: datetime | None = None,
        checkpoint: str | None = None,
        apply_primary: bool | None = None,
    ) -> OfficialCheckpointResult:
        assert checkpoint is not None
        assert checkpoint.startswith("manual-")
        assert apply_primary is False
        assert build_id == "build-manual"
        cp = str(checkpoint)
        cp_dir = official_checkpoint_dir(
            data_dir, season=gw_key.season, gw=gw_key.gw, checkpoint=cp
        )
        cp_dir.mkdir(parents=True, exist_ok=True)
        pairs_list: list[LineRunPair] = []
        for line in ops.lines:
            pairs_list.append(
                LineRunPair(
                    line=line.line,
                    role=line.role,
                    squad_mode=line.squad_mode,
                    predict_run_id=f"pred-{line.line}",
                    plan_run_id=f"plan-{line.line}",
                )
            )
            (cp_dir / f"line={line.line}.json").write_text(
                OfficialPointer(
                    predict_run_id=f"pred-{line.line}",
                    plan_run_id=f"plan-{line.line}",
                    role=line.role,
                    squad_mode=line.squad_mode,
                    written_at=now or _DEADLINE,
                ).model_dump_json(),
                encoding="utf-8",
            )
        calls.append(
            {
                "checkpoint": cp,
                "apply_primary": apply_primary,
                "build_id": build_id,
                "decisions": decisions,
                "prod_tag": prod_tag,
            }
        )
        return OfficialCheckpointResult(
            season=gw_key.season,
            gw=gw_key.gw,
            checkpoint=cp,
            pairs=tuple(pairs_list),
            checkpoint_dir=cp_dir,
            build_id=str(build_id),
        )

    monkeypatch.setattr(
        "fplt.automation.ops_tick.run_official_checkpoint", fake_official
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.notify_recommendation",
        lambda *a, **k: notify_calls.append("sent") or True,
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.catch_up",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("catch_up on run_now")),
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.pre_deadline",
        lambda *a, **k: (_ for _ in ()).throw(
            AssertionError("pre_deadline on run_now")
        ),
    )

    result = run_ops_tick(
        tmp_path,
        now=now,
        models=["rolling_mean_v1"],
        horizon=1,
        notify=True,
        run_now=True,
    )
    assert len(result.checkpoints_run) == 1
    assert result.checkpoints_run[0] == manual_checkpoint_id(now)
    assert result.no_plan_alarms == []
    assert notify_calls == []
    assert len(calls) == 1

    cp = result.checkpoints_run[0]
    for line in (_PRIMARY, "e2e_h5"):
        path = (
            official_checkpoint_dir(tmp_path, season="2024-25", gw=3, checkpoint=cp)
            / f"line={line}.json"
        )
        assert path.is_file(), path
    assert not (
        official_checkpoint_dir(tmp_path, season="2024-25", gw=3, checkpoint=cp)
        / "applied.json"
    ).exists()
    assert (
        official_checkpoint_dir(tmp_path, season="2024-25", gw=3, checkpoint="T-24h")
        / f"line={_PRIMARY}.json"
    ).read_text(encoding="utf-8") == before_t24
    assert notified_path(tmp_path, key).read_text(encoding="utf-8") == before_notified
    get_settings.cache_clear()


def test_two_run_now_calls_distinct_manual_checkpoints(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("JOB_STARTED_AT", "1700000000")
    get_settings.cache_clear()
    _league(tmp_path)
    _patch_curated(monkeypatch)

    def fake_official(
        data_dir: Path,
        ops: ProductionOps,
        gw_key: GWKey,
        *,
        settings: Settings | None = None,
        chip_arm: str = "optimised",
        build_id: str | None = None,
        prod_tag: str | None = None,
        decisions: list[LineFpDecision] | None = None,
        now: datetime | None = None,
        checkpoint: str | None = None,
        apply_primary: bool | None = None,
    ) -> OfficialCheckpointResult:
        cp = str(checkpoint)
        cp_dir = official_checkpoint_dir(
            data_dir, season=gw_key.season, gw=gw_key.gw, checkpoint=cp
        )
        cp_dir.mkdir(parents=True, exist_ok=True)
        pairs_list: list[LineRunPair] = []
        for line in ops.lines:
            pairs_list.append(
                LineRunPair(
                    line=line.line,
                    role=line.role,
                    squad_mode=line.squad_mode,
                    predict_run_id=f"pred-{cp}-{line.line}",
                    plan_run_id=f"plan-{cp}-{line.line}",
                )
            )
            (cp_dir / f"line={line.line}.json").write_text(
                OfficialPointer(
                    predict_run_id=f"pred-{cp}-{line.line}",
                    plan_run_id=f"plan-{cp}-{line.line}",
                    role=line.role,
                    squad_mode=line.squad_mode,
                    written_at=_DEADLINE,
                ).model_dump_json(),
                encoding="utf-8",
            )
        return OfficialCheckpointResult(
            season=gw_key.season,
            gw=gw_key.gw,
            checkpoint=cp,
            pairs=tuple(pairs_list),
            checkpoint_dir=cp_dir,
            build_id=str(build_id),
        )

    monkeypatch.setattr(
        "fplt.automation.ops_tick.run_official_checkpoint", fake_official
    )
    first = run_ops_tick(
        tmp_path,
        now=_DEADLINE - timedelta(days=4, seconds=10),
        notify=False,
        run_now=True,
    )
    second = run_ops_tick(
        tmp_path,
        now=_DEADLINE - timedelta(days=4, seconds=9),
        notify=False,
        run_now=True,
    )
    assert first.checkpoints_run[0] != second.checkpoints_run[0]
    for cp in (first.checkpoints_run[0], second.checkpoints_run[0]):
        assert cp.startswith("manual-")
        assert (
            official_checkpoint_dir(tmp_path, season="2024-25", gw=3, checkpoint=cp)
            / f"line={_PRIMARY}.json"
        ).is_file()
    get_settings.cache_clear()


def test_run_now_reuses_fingerprint_and_still_writes_pointer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Unchanged fp → reuse runs; still write a new manual-* pointer (#346/#364)."""
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("JOB_STARTED_AT", "1700000000")
    get_settings.cache_clear()
    _league(tmp_path)
    skip_decisions = [
        LineFpDecision(
            line=_PRIMARY,
            spec_name="rolling_mean_v1",
            predict_fp="a" * 64,
            plan_fp="b" * 64,
            skip=True,
            predict_run_id="r_pred_reused",
            plan_run_id="r_plan_reused",
        ),
        LineFpDecision(
            line="e2e_h5",
            spec_name="rolling_mean_v1",
            predict_fp="a" * 64,
            plan_fp="c" * 64,
            skip=True,
            predict_run_id="r_pred_reused",
            plan_run_id="r_plan_h5_reused",
        ),
    ]
    _patch_curated(monkeypatch, decisions=skip_decisions)

    calls: list[dict[str, Any]] = []

    def fake_official(
        data_dir: Path,
        ops: ProductionOps,
        gw_key: GWKey,
        *,
        settings: Settings | None = None,
        chip_arm: str = "optimised",
        build_id: str | None = None,
        prod_tag: str | None = None,
        decisions: list[LineFpDecision] | None = None,
        now: datetime | None = None,
        checkpoint: str | None = None,
        apply_primary: bool | None = None,
    ) -> OfficialCheckpointResult:
        assert decisions is not None
        assert all(item.skip for item in decisions)
        cp = str(checkpoint)
        cp_dir = official_checkpoint_dir(
            data_dir, season=gw_key.season, gw=gw_key.gw, checkpoint=cp
        )
        cp_dir.mkdir(parents=True, exist_ok=True)
        pairs_list: list[LineRunPair] = []
        for item in decisions:
            line = next(entry for entry in ops.lines if entry.line == item.line)
            pairs_list.append(
                LineRunPair(
                    line=item.line,
                    role=line.role,
                    squad_mode=line.squad_mode,
                    predict_run_id=str(item.predict_run_id),
                    plan_run_id=str(item.plan_run_id),
                )
            )
            (cp_dir / f"line={item.line}.json").write_text(
                OfficialPointer(
                    predict_run_id=str(item.predict_run_id),
                    plan_run_id=str(item.plan_run_id),
                    role=line.role,
                    squad_mode=line.squad_mode,
                    written_at=_DEADLINE,
                ).model_dump_json(),
                encoding="utf-8",
            )
        calls.append({"checkpoint": cp, "decisions": decisions})
        return OfficialCheckpointResult(
            season=gw_key.season,
            gw=gw_key.gw,
            checkpoint=cp,
            pairs=tuple(pairs_list),
            checkpoint_dir=cp_dir,
            build_id=str(build_id),
        )

    monkeypatch.setattr(
        "fplt.automation.ops_tick.run_official_checkpoint", fake_official
    )
    # Idle short-circuit must not fire for run_now even when all fps skip.
    result = run_ops_tick(
        tmp_path,
        now=_DEADLINE - timedelta(days=2),
        notify=False,
        run_now=True,
    )
    assert len(calls) == 1
    assert result.checkpoints_run[0].startswith("manual-")
    pointer = OfficialPointer.model_validate_json(
        (
            official_checkpoint_dir(
                tmp_path,
                season="2024-25",
                gw=3,
                checkpoint=result.checkpoints_run[0],
            )
            / f"line={_PRIMARY}.json"
        ).read_text(encoding="utf-8")
    )
    assert pointer.predict_run_id == "r_pred_reused"
    assert pointer.plan_run_id == "r_plan_reused"
    get_settings.cache_clear()


def test_run_now_requires_upcoming_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    # No gameweeks → no upcoming deadline (still needs curated ctx).
    _patch_curated(monkeypatch)
    with pytest.raises(ValueError, match="upcoming deadline"):
        run_ops_tick(tmp_path, now=_DEADLINE, notify=False, run_now=True)
    get_settings.cache_clear()


def test_ops_tick_drift_alert_and_failure_nonfatal(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Non-run_now path still logs drift alerts / swallows drift errors."""
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("JOB_STARTED_AT", "1700000000")
    get_settings.cache_clear()
    _league(tmp_path)
    _patch_curated(monkeypatch)
    monkeypatch.setattr(
        "fplt.automation.ops_tick.catch_up",
        lambda *a, **k: [GWKey("2024-25", 2)],
    )
    monkeypatch.setattr("fplt.automation.ops_tick.pre_deadline", lambda *a, **k: None)
    monkeypatch.setattr(
        "fplt.automation.ops_tick.resolve_done_sets",
        lambda *a, **k: (
            frozenset({"T-24h", "T-6h", "T-90m", "T-30m"}),
            frozenset(),
        ),
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.ensure_slate_runs", lambda *a, **k: ("b", ())
    )
    monkeypatch.setattr("fplt.automation.ops_tick.pending_checked", lambda *a, **k: [])
    monkeypatch.setattr(
        "fplt.automation.ops_tick.notify_recommendation", lambda *a, **k: True
    )

    drift = type(
        "Drift",
        (),
        {
            "alerted": True,
            "model_id": "m",
            "alerts": [type("A", (), {"metric": "rmse"})()],
        },
    )()
    monkeypatch.setattr(
        "fplt.automation.drift.run_live_drift_monitor",
        lambda *a, **k: [drift],
    )
    run_ops_tick(tmp_path, now=_DEADLINE - timedelta(days=5), notify=False)

    monkeypatch.setattr(
        "fplt.automation.drift.run_live_drift_monitor",
        lambda *a, **k: (_ for _ in ()).throw(ValueError("boom")),
    )
    run_ops_tick(tmp_path, now=_DEADLINE - timedelta(days=5), notify=False)
    get_settings.cache_clear()


def test_run_now_ops_log_exists_is_tolerated(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Pre-existing ops_log.jsonl is write-once; run_now continues (#364)."""
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("JOB_STARTED_AT", "1700000000")
    get_settings.cache_clear()
    _league(tmp_path)
    _patch_curated(monkeypatch)

    def fake_official(
        data_dir: Path,
        ops: ProductionOps,
        gw_key: GWKey,
        *,
        settings: Settings | None = None,
        chip_arm: str = "optimised",
        build_id: str | None = None,
        prod_tag: str | None = None,
        decisions: list[LineFpDecision] | None = None,
        now: datetime | None = None,
        checkpoint: str | None = None,
        apply_primary: bool | None = None,
    ) -> OfficialCheckpointResult:
        cp = str(checkpoint)
        cp_dir = official_checkpoint_dir(
            data_dir, season=gw_key.season, gw=gw_key.gw, checkpoint=cp
        )
        cp_dir.mkdir(parents=True, exist_ok=True)
        # Force FileExistsError inside write_ops_log_once.
        (cp_dir / "ops_log.jsonl").write_text("{}\n", encoding="utf-8")
        return OfficialCheckpointResult(
            season=gw_key.season,
            gw=gw_key.gw,
            checkpoint=cp,
            pairs=(),
            checkpoint_dir=cp_dir,
            build_id=str(build_id),
        )

    monkeypatch.setattr(
        "fplt.automation.ops_tick.run_official_checkpoint", fake_official
    )
    result = run_ops_tick(
        tmp_path, now=_DEADLINE - timedelta(days=1), notify=False, run_now=True
    )
    assert result.checkpoints_run[0].startswith("manual-")
    get_settings.cache_clear()


def test_run_official_checkpoint_manual_skips_applied(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``apply_primary=False`` / manual-* leaves applied.json absent."""
    from fplt.automation.official_slate import run_official_checkpoint
    from tests.builders import build_league

    build_league(DataStore(tmp_path))
    squad = SquadState(bank=1000, free_transfers=1, ft_bank_cap=5, owned=[])
    monkeypatch.setattr(
        "fplt.automation.official_slate.ensure_slate_runs",
        lambda *a, **k: (
            "build1",
            (
                LineRunPair(
                    line=_PRIMARY,
                    role="primary",
                    squad_mode="real",
                    predict_run_id="predict-run",
                    plan_run_id="plan-run",
                ),
            ),
        ),
    )
    finalized: list[dict[str, Any]] = []

    def fake_finalize(*_a: Any, **kwargs: Any) -> list[Path]:
        finalized.append(kwargs)
        cp = official_checkpoint_dir(
            tmp_path,
            season="2024-25",
            gw=2,
            checkpoint=str(kwargs.get("checkpoint")),
        )
        cp.mkdir(parents=True, exist_ok=True)
        (cp / f"line={_PRIMARY}.json").write_text("{}", encoding="utf-8")
        return [cp / f"line={_PRIMARY}.json"]

    monkeypatch.setattr(
        "fplt.automation.official_slate.finalize_official_checkpoint", fake_finalize
    )
    monkeypatch.setattr(
        "fplt.automation.official_slate.maybe_push_official",
        lambda *a, **k: None,
    )
    _ = squad  # keep import used for parity with other official tests
    result = run_official_checkpoint(
        tmp_path,
        _tiny_ops(n_lines=1),
        GWKey("2024-25", 2),
        build_id="build1",
        checkpoint="manual-20261128T120000Z",
        apply_primary=False,
    )
    assert result.checkpoint == "manual-20261128T120000Z"
    assert finalized
    assert finalized[0].get("apply_primary") is False
    assert finalized[0].get("checkpoint") == "manual-20261128T120000Z"
