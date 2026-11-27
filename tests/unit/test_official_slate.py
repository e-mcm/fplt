"""Unit tests for T−90m official slate wiring (#307 product gaps)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl
import pytest

from fplt.automation.official_slate import (
    OfficialCheckpointResult,
    run_official_checkpoint,
)
from fplt.automation.ops_feeds import FeedRefreshResult
from fplt.automation.ops_tick import run_ops_tick
from fplt.automation.production import OfficialLine, ProductionOps, load_production_ops
from fplt.data.store import DataStore
from fplt.models.pipeline_spec import live_pipeline_defaults
from fplt.settings import Settings, get_settings
from fplt.store.official import official_checkpoint_dir
from fplt.store.push import maybe_push_official, push_official
from fplt.types import GWKey
from fplt.utils.timing import ops_timing_path as timing_ops_path

_DEADLINE = datetime(2024, 9, 1, 17, 0, tzinfo=UTC)


def _tiny_ops(*, model: str = "rolling_mean_v1", horizon: int = 1) -> ProductionOps:
    pipeline = live_pipeline_defaults(model=model, horizon=horizon)
    return ProductionOps(
        models=[model],
        plan_horizon=horizon,
        pipeline=pipeline,
        lines=[
            OfficialLine(
                line="e2e_primary",
                spec=model,
                horizon=horizon,
                role="primary",
                squad_mode="real",
            )
        ],
    )


def test_ops_tick_t90m_calls_official_slate(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("JOB_STARTED_AT", "1700000000")
    get_settings.cache_clear()
    store = DataStore(tmp_path)
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
    calls: list[str] = []
    cp_dir = tmp_path / "official" / "season=2024-25" / "gw=3" / "checkpoint=T-90m"
    cp_dir.mkdir(parents=True)

    def fake_official(
        data_dir: Path,
        ops: ProductionOps,
        key: GWKey,
        *,
        settings: Settings | None = None,
        chip_arm: str = "optimised",
    ) -> OfficialCheckpointResult:
        calls.append(f"{key.season} gw{key.gw}:{ops.primary_line().line}:{chip_arm}")
        return OfficialCheckpointResult(
            season=key.season,
            gw=key.gw,
            checkpoint="T-90m",
            pairs=(),
            checkpoint_dir=cp_dir,
            build_id="abc",
        )

    monkeypatch.setattr(
        "fplt.automation.ops_tick.refresh_ops_feeds",
        lambda *a, **k: FeedRefreshResult(0, 0, 0, 0),
    )
    monkeypatch.setattr("fplt.automation.ops_tick.catch_up", lambda *a, **k: [])
    monkeypatch.setattr("fplt.automation.ops_tick.pre_deadline", lambda *a, **k: None)
    monkeypatch.setattr("fplt.automation.ops_tick.pending_checked", lambda *a, **k: [])
    monkeypatch.setattr("fplt.automation.ops_tick.plans_exist", lambda *a, **k: True)
    monkeypatch.setattr(
        "fplt.automation.ops_tick.run_official_checkpoint", fake_official
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.load_production_ops",
        _tiny_ops,
    )
    monkeypatch.setattr(
        "fplt.automation.ops_tick.enforce_training_history",
        lambda *a, **k: None,
    )
    from fplt.automation.checkpoints import CheckpointState, save_checkpoint_state

    save_checkpoint_state(
        tmp_path,
        CheckpointState(
            key=GWKey("2024-25", 3),
            deadline=_DEADLINE,
            done=frozenset({"T-24h", "T-6h"}),
            alarms_done=frozenset(),
        ),
    )

    result = run_ops_tick(
        tmp_path,
        now=_DEADLINE - timedelta(minutes=80),
        models=["rolling_mean_v1"],
        horizon=1,
        notify=False,
    )
    assert "T-90m" in result.checkpoints_run
    assert calls == ["2024-25 gw3:e2e_primary:optimised"]
    # Official ops_timing.json written when checkpoint_dir is passed through.
    assert timing_ops_path(cp_dir).is_file()
    get_settings.cache_clear()


def test_maybe_push_official_skips_without_bucket(tmp_path: Path) -> None:
    cp = official_checkpoint_dir(tmp_path, season="2024-25", gw=3, checkpoint="T-90m")
    cp.mkdir(parents=True)
    (cp / "applied.json").write_text("{}", encoding="utf-8")
    assert (
        maybe_push_official(tmp_path, season="2024-25", gw=3, settings=Settings())
        is None
    )


def test_push_official_to_local_remote(tmp_path: Path) -> None:
    from fplt.data.remote import Remote

    data = tmp_path / "data"
    bucket = tmp_path / "bucket"
    cp = official_checkpoint_dir(data, season="2024-25", gw=3, checkpoint="T-90m")
    cp.mkdir(parents=True)
    (cp / "line=e2e_primary.json").write_text('{"ok": true}\n', encoding="utf-8")
    remote = Remote(target=str(bucket), kind="data", env={})
    push_official(data, season="2024-25", gw=3, checkpoint="T-90m", remote=remote)
    dest = (
        bucket
        / "official"
        / "season=2024-25"
        / "gw=3"
        / "checkpoint=T-90m"
        / "line=e2e_primary.json"
    )
    assert dest.is_file()


def test_run_official_checkpoint_requires_sampled_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Opaque models cannot write points_pmf; fail with a clear message."""
    from tests.builders import build_league

    build_league(DataStore(tmp_path))
    # Seed minimal staged content so build_from_staged can run.
    monkeypatch.setattr(
        "fplt.automation.official_slate.build_from_staged",
        lambda *a, **k: type("B", (), {"build_id": "deadbeef"})(),
    )

    def boom_predict(*_a: Any, **_k: Any) -> str:
        raise ValueError(
            "official predict requires SampledPointsPredictor.sample() "
            "for points PMFs; got RollingMeanPredictor"
        )

    monkeypatch.setattr(
        "fplt.automation.official_slate._official_predict", boom_predict
    )
    ops = _tiny_ops()
    with pytest.raises(ValueError, match="SampledPointsPredictor"):
        run_official_checkpoint(tmp_path, ops, GWKey("2024-25", 2))


def test_committed_production_slate_still_loads() -> None:
    ops = load_production_ops()
    horizons, lines, checkpoint = ops.official_slate()
    assert checkpoint == "T-90m"
    assert "dc_layered_v4_native" in horizons
    assert any(line.role == "primary" for line in lines)


def test_write_predict_run_persists_points_pmf(tmp_path: Path) -> None:
    from fplt.models.predictions import empty_predictions
    from fplt.models.spec import load_model_specs
    from fplt.store.api import Store
    from fplt.store.points_pmf import points_pmf_from_draws
    from fplt.store.runs import write_predict_run
    from tests.builders import build_league

    store = build_league(DataStore(tmp_path))
    as_of = GWKey("2024-25", 2)
    deadline = store.deadline(as_of)
    spec = load_model_specs()["rolling_mean_v1"]
    draws = pl.DataFrame(
        {
            "player_key": [100, 100],
            "team_key": [3, 3],
            "fixture_id": [2, 2],
            "target_season": ["2024-25", "2024-25"],
            "target_gw": [2, 2],
            "horizon": [0, 0],
            "draw_id": [0, 1],
            "points": [4.0, 6.0],
        }
    )

    def _pmf(run_id: str) -> tuple[pl.DataFrame, pl.DataFrame]:
        fx = points_pmf_from_draws(
            draws,
            run_id=run_id,
            as_of_season=as_of.season,
            as_of_gw=as_of.gw,
            deadline_utc=deadline,
            grain="fixture",
        )
        gw = points_pmf_from_draws(
            draws,
            run_id=run_id,
            as_of_season=as_of.season,
            as_of_gw=as_of.gw,
            deadline_utc=deadline,
            grain="gameweek",
        )
        return fx, gw

    manifest = write_predict_run(
        tmp_path,
        spec=spec,
        as_of=as_of,
        deadline_utc=deadline,
        horizon=1,
        frame=empty_predictions(),
        kind="official",
        checkpoint="T-90m",
        points_pmf_for_run_id=_pmf,
    )
    analytic = Store(tmp_path)
    assert not analytic.load("points_pmf", run=manifest).collect().is_empty()
    assert not analytic.load("points_pmf_gw", run=manifest).collect().is_empty()


def test_run_official_checkpoint_happy_path(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wire predict → plan → finalize → push with stubs for heavy work."""
    from fplt.optimise.state import SquadState
    from fplt.store.official import LineRunPair
    from tests.builders import build_league

    build_league(DataStore(tmp_path))
    squad = SquadState(bank=1000, free_transfers=1, ft_bank_cap=5, owned=[])
    monkeypatch.setattr(
        "fplt.automation.official_slate.build_from_staged",
        lambda *a, **k: type("B", (), {"build_id": "build1"})(),
    )
    monkeypatch.setattr(
        "fplt.automation.official_slate._official_predict",
        lambda *a, **k: "predict-run",
    )
    monkeypatch.setattr(
        "fplt.automation.official_slate.resolve_squad_state",
        lambda *a, **k: squad,
    )
    monkeypatch.setattr(
        "fplt.automation.official_slate._official_plan",
        lambda *a, **k: "plan-run",
    )
    finalized: list[object] = []

    def fake_finalize(*_a: Any, **kwargs: Any) -> list[Path]:
        finalized.append(kwargs.get("pairs"))
        cp = official_checkpoint_dir(
            tmp_path, season="2024-25", gw=2, checkpoint="T-90m"
        )
        cp.mkdir(parents=True, exist_ok=True)
        return [cp / "applied.json"]

    monkeypatch.setattr(
        "fplt.automation.official_slate.finalize_official_checkpoint", fake_finalize
    )
    pushed: list[tuple[str, int]] = []

    def fake_push(
        data_dir: Path,
        *,
        season: str,
        gw: int,
        checkpoint: str = "T-90m",
        settings: Settings | None = None,
    ) -> None:
        pushed.append((season, gw))

    monkeypatch.setattr("fplt.automation.official_slate.maybe_push_official", fake_push)

    result = run_official_checkpoint(tmp_path, _tiny_ops(), GWKey("2024-25", 2))
    assert result.build_id == "build1"
    assert result.checkpoint == "T-90m"
    assert result.pairs == (
        LineRunPair(
            line="e2e_primary",
            role="primary",
            squad_mode="real",
            predict_run_id="predict-run",
            plan_run_id="plan-run",
        ),
    )
    assert finalized
    assert pushed == [("2024-25", 2)]


def test_official_predict_writes_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``_official_predict`` fits, scores, and writes an official run id."""
    from fplt.automation.official_slate import _official_predict
    from fplt.models.predictions import empty_predictions
    from fplt.models.spec import load_model_specs
    from tests.builders import build_league

    build_league(DataStore(tmp_path))
    key = GWKey("2024-25", 2)
    spec = load_model_specs()["rolling_mean_v1"]

    class _Model:
        def predict(self, view: object, horizon: object) -> pl.DataFrame:
            return empty_predictions()

    monkeypatch.setattr(
        "fplt.automation.official_slate.fit_or_load_predictor",
        lambda *a, **k: _Model(),
    )
    monkeypatch.setattr(
        "fplt.automation.official_slate.write_predictions",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "fplt.automation.official_slate.ledger_rows",
        lambda *a, **k: empty_predictions(),
    )
    monkeypatch.setattr(
        "fplt.automation.official_slate.n_sims_of",
        lambda *_a, **_k: 2,
    )
    monkeypatch.setattr(
        "fplt.automation.official_slate.component_builder_for",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "fplt.automation.official_slate._pmf_builder",
        lambda *a, **k: None,
    )

    def fake_write(*_a: Any, **kwargs: Any) -> Any:
        assert kwargs["kind"] == "official"
        assert kwargs["checkpoint"] == "T-90m"
        return type("M", (), {"run_id": "pred-1"})()

    monkeypatch.setattr("fplt.automation.official_slate.write_predict_run", fake_write)
    assert (
        _official_predict(tmp_path, spec.name, key=key, horizon=1, checkpoint="T-90m")
        == "pred-1"
    )


def test_official_predict_unknown_model(tmp_path: Path) -> None:
    from fplt.automation.official_slate import _official_predict

    with pytest.raises(ValueError, match="unknown model spec"):
        _official_predict(
            tmp_path,
            "no_such_model",
            key=GWKey("2024-25", 1),
            horizon=1,
            checkpoint="T-90m",
        )


def test_official_predict_tolerates_legacy_predictions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.automation.official_slate import _official_predict
    from fplt.models.predictions import empty_predictions
    from fplt.models.spec import load_model_specs
    from tests.builders import build_league

    build_league(DataStore(tmp_path))
    key = GWKey("2024-25", 2)
    spec = load_model_specs()["rolling_mean_v1"]

    class _Model:
        def predict(self, view: object, horizon: object) -> pl.DataFrame:
            return empty_predictions()

    monkeypatch.setattr(
        "fplt.automation.official_slate.fit_or_load_predictor",
        lambda *a, **k: _Model(),
    )
    monkeypatch.setattr(
        "fplt.automation.official_slate.write_predictions",
        lambda *a, **k: (_ for _ in ()).throw(FileExistsError("exists")),
    )
    monkeypatch.setattr(
        "fplt.automation.official_slate.ledger_rows",
        lambda *a, **k: empty_predictions(),
    )
    monkeypatch.setattr(
        "fplt.automation.official_slate.n_sims_of", lambda *_a, **_k: None
    )
    monkeypatch.setattr(
        "fplt.automation.official_slate.component_builder_for",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "fplt.automation.official_slate._pmf_builder",
        lambda *a, **k: None,
    )
    monkeypatch.setattr(
        "fplt.automation.official_slate.write_predict_run",
        lambda *a, **k: type("M", (), {"run_id": "pred-legacy"})(),
    )
    assert (
        _official_predict(tmp_path, spec.name, key=key, horizon=1, checkpoint="T-90m")
        == "pred-legacy"
    )


def test_official_plan_success_and_exists(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.automation.official_slate import _official_plan
    from fplt.models.spec import load_model_specs
    from fplt.optimise.decisions import PlanExistsError
    from fplt.optimise.state import SquadState
    from tests.builders import build_league

    build_league(DataStore(tmp_path))
    line = OfficialLine(
        line="e2e_primary",
        spec="rolling_mean_v1",
        horizon=1,
        role="primary",
        squad_mode="real",
    )
    squad = SquadState(bank=1000, free_transfers=1, ft_bank_cap=5, owned=[])
    key = GWKey("2024-25", 2)

    monkeypatch.setattr(
        "fplt.automation.official_slate.run_plan",
        lambda *a, **k: (None, None, None, "plan-ok"),
    )
    assert (
        _official_plan(
            tmp_path,
            line,
            key=key,
            parent_run_id="p",
            checkpoint="T-90m",
            squad=squad,
            chip_arm="off",
        )
        == "plan-ok"
    )

    with pytest.raises(TypeError, match="SquadState"):
        _official_plan(
            tmp_path,
            line,
            key=key,
            parent_run_id="p",
            checkpoint="T-90m",
            squad="nope",
            chip_arm="off",
        )

    plan_path = tmp_path / "legacy_plan.parquet"
    pl.DataFrame(
        {
            "model_id": ["x"],
            "as_of_season": ["2024-25"],
            "as_of_gw": [2],
            "target_gw": [2],
            "step": [0],
            "player_key": [100],
            "position": ["MID"],
            "team_key": [3],
            "in_xi": [True],
            "captain": [True],
            "vice": [False],
            "bought": [False],
            "sold": [False],
            "xpts": [1.0],
            "hit": [0.0],
        }
    ).write_parquet(plan_path)

    def boom(*_a: Any, **_k: Any) -> Any:
        raise PlanExistsError(plan_path, "already planned")

    monkeypatch.setattr("fplt.automation.official_slate.run_plan", boom)
    monkeypatch.setattr(
        "fplt.automation.official_slate.write_plan_run",
        lambda *a, **k: type("M", (), {"run_id": "plan-from-legacy"})(),
    )
    assert load_model_specs()["rolling_mean_v1"]
    assert (
        _official_plan(
            tmp_path,
            line,
            key=key,
            parent_run_id="p",
            checkpoint="T-90m",
            squad=squad,
            chip_arm="off",
        )
        == "plan-from-legacy"
    )


def test_pmf_builder_paths(tmp_path: Path) -> None:
    """``_pmf_builder`` validates sample() output and joins team_key."""
    from fplt.automation.official_slate import _pmf_builder
    from fplt.data.pit import PITData
    from fplt.models.spec import load_model_specs
    from tests.builders import build_league

    store = build_league(DataStore(tmp_path))
    key = GWKey("2024-25", 2)
    view = PITData(store).view(key)
    deadline = store.deadline(key)

    class _Opaque:
        pass

    opaque = _pmf_builder(
        _Opaque(), view, as_of=key, deadline_utc=deadline, horizon=1, n_sims=2
    )
    with pytest.raises(ValueError, match="SampledPointsPredictor"):
        opaque("run-x")

    class _Empty:
        spec = load_model_specs()["rolling_mean_v1"]

        def sample(self, view: object, horizon: object, n: int) -> pl.DataFrame:
            return pl.DataFrame(
                schema={
                    "player_key": pl.Int64,
                    "fixture_id": pl.Int64,
                    "target_season": pl.Utf8,
                    "target_gw": pl.Int64,
                    "sim": pl.Int64,
                    "points": pl.Float64,
                }
            )

    empty_builder = _pmf_builder(
        _Empty(), view, as_of=key, deadline_utc=deadline, horizon=1, n_sims=2
    )
    with pytest.raises(ValueError, match="returned no rows"):
        empty_builder("run-x")

    class _MissingTeam:
        spec = load_model_specs()["rolling_mean_v1"]

        def sample(self, view: object, horizon: object, n: int) -> pl.DataFrame:
            return pl.DataFrame(
                {
                    "player_key": [999],
                    "fixture_id": [2],
                    "target_season": ["2024-25"],
                    "target_gw": [2],
                    "sim": [0],
                    "points": [1.0],
                }
            )

    missing = _pmf_builder(
        _MissingTeam(), view, as_of=key, deadline_utc=deadline, horizon=1, n_sims=1
    )
    with pytest.raises(ValueError, match="missing team_key"):
        missing("run-x")

    class _Ok:
        spec = load_model_specs()["rolling_mean_v1"]

        def sample(self, view: object, horizon: object, n: int) -> pl.DataFrame:
            return pl.DataFrame(
                {
                    "player_key": [100, 100],
                    "fixture_id": [2, 2],
                    "target_season": ["2024-25", "2024-25"],
                    "target_gw": [2, 2],
                    "sim": [0, 1],
                    "points": [4.0, 6.0],
                }
            )

    ok = _pmf_builder(
        _Ok(), view, as_of=key, deadline_utc=deadline, horizon=1, n_sims=2
    )
    fx, gw = ok("run-ok")
    assert not fx.is_empty()
    assert not gw.is_empty()


def test_push_official_error_paths(tmp_path: Path) -> None:
    from fplt.data.remote import Remote

    with pytest.raises(FileNotFoundError):
        push_official(
            tmp_path,
            season="2024-25",
            gw=1,
            remote=Remote(target=str(tmp_path / "b"), kind="data", env={}),
        )
    cp = official_checkpoint_dir(tmp_path, season="2024-25", gw=1, checkpoint="T-90m")
    cp.mkdir(parents=True)
    (cp / "applied.json").write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="data bucket"):
        push_official(
            tmp_path,
            season="2024-25",
            gw=1,
            remote=Remote(target=str(tmp_path / "b"), kind="lab", env={}),
        )


def test_maybe_push_official_with_bucket(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.data.remote import Remote

    cp = official_checkpoint_dir(tmp_path, season="2024-25", gw=3, checkpoint="T-90m")
    cp.mkdir(parents=True)
    (cp / "applied.json").write_text("{}", encoding="utf-8")
    bucket = tmp_path / "bucket"
    settings = Settings(data_bucket=str(bucket))
    remote = Remote(target=str(bucket), kind="data", env={})
    monkeypatch.setattr(
        "fplt.store.push._remote_for",
        lambda wanted, settings: remote,
    )
    assert (
        maybe_push_official(tmp_path, season="2024-25", gw=3, settings=settings)
        is remote
    )


def test_write_points_pmf_tables_ignores_bad_builders(tmp_path: Path) -> None:
    from fplt.models.predictions import empty_predictions
    from fplt.models.spec import load_model_specs
    from fplt.store.runs import write_predict_run
    from tests.builders import build_league

    store = build_league(DataStore(tmp_path))
    as_of = GWKey("2024-25", 2)
    deadline = store.deadline(as_of)
    spec = load_model_specs()["rolling_mean_v1"]

    for builder in (
        lambda _rid: "not-a-tuple",
        lambda _rid: (pl.DataFrame({"a": [1]}),),
        lambda _rid: ("x", "y"),
    ):
        write_predict_run(
            tmp_path,
            spec=spec,
            as_of=as_of,
            deadline_utc=deadline,
            horizon=1,
            frame=empty_predictions(),
            kind="official",
            checkpoint="T-90m",
            purpose=f"bad-{id(builder)}",
            points_pmf_for_run_id=builder,
        )
