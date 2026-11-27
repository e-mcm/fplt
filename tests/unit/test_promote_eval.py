"""Tests for EvalSpec loading, preflight, coverage, and promote-eval execute."""

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, ClassVar
from unittest.mock import patch

import polars as pl
import pytest
import yaml
from pydantic import ValidationError

from fplt.backtest.compare import CompareResult, CompareRun
from fplt.backtest.component_eval import ComponentEvalResult
from fplt.backtest.equivalence import EquivalenceDiff, LockfileDiff
from fplt.backtest.intervals import McsResult, PairedInterval, SpaResult
from fplt.backtest.promote_eval import (
    CoverageResult,
    EvalSpec,
    PreflightResult,
    PromoteEvalPlan,
    _check_equivalence_preflight,
    _check_kinds,
    _check_one_lever,
    _check_registered,
    _component_deltas,
    _component_param_diffs,
    _coverage_pass,
    _coverage_set_piece_order,
    _dict_diff_paths,
    _format_interval_lines,
    _git_show_file,
    _is_production_arm,
    _lever_is_position_specific,
    _n_sims,
    _pooled_arm_metric,
    _prediction_deltas,
    _run_components,
    _spa_mcs_from_ledger,
    _verdict_paragraph,
    decisions_claim_failures,
    detect_points_claim_phrase,
    effective_param,
    execute_promote_eval,
    format_plan,
    git_is_dirty,
    is_effectively_on,
    is_n_sims_lever,
    is_scoreline_lever,
    load_eval_spec,
    plan_promote_eval,
    plan_to_json,
    preflight_promote_eval,
    raw_de_vig_spec,
    render_recommendation,
    slot_component_names,
)
from fplt.cli import main
from fplt.data.store import DataStore
from fplt.models.spec import ModelSpec

_START = datetime(2025, 8, 1, tzinfo=UTC)
_DEADLINE_GW1 = datetime(2025, 8, 15, 17, tzinfo=UTC)
_KICKOFF_GW1 = datetime(2025, 8, 16, 14, tzinfo=UTC)


def _write_eval(folder: Path, name: str, payload: dict[str, object]) -> Path:
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / f"{name}.yaml"
    path.write_text(yaml.safe_dump(payload), encoding="utf-8")
    return path


def _base_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": "demo_swap",
        "kind": "component",
        "question": "Does a minutes swap move xPts?",
        "description": "minutes swap",
        "change": {
            "lever": "component_swap",
            "orientation": "opt_in",
            "slot": "minutes",
            "control_kind": "empirical_minutes_v2",
            "treatment_kind": "lgbm_minutes",
            "allow_multi_lever": True,
            "diff": ["components.minutes.kind: empirical_minutes_v2 → lgbm_minutes"],
        },
        "control": "dc_layered_v2",
        "treatment": "dc_layered_minutes_lgbm_v1",
        "window": {
            "seasons": ["2024-25"],
            "from_gw": 1,
            "to_gw": 10,
            "modes": ["walk_forward"],
        },
        "suites": {
            "prediction": True,
            "components": True,
            "decisions": False,
        },
    }
    payload.update(overrides)
    return payload


def _utc(frame: pl.DataFrame) -> pl.DataFrame:
    exprs = [
        pl.col(name).cast(pl.Datetime("us", "UTC"))
        for name, dtype in frame.schema.items()
        if isinstance(dtype, pl.Datetime)
    ]
    return frame.with_columns(exprs) if exprs else frame


def _seed_pens_store(
    root: Path, *, order: int | None, source: str = "fpl_api"
) -> DataStore:
    store = DataStore(root)
    season = "2025-26"
    store.write(
        "gameweeks",
        _utc(
            pl.DataFrame(
                {
                    "season": [season],
                    "gw": [1],
                    "deadline_time": [_DEADLINE_GW1],
                    "finished": [False],
                    "data_checked": [False],
                    "available_at": [_START],
                    "source": ["fpl_api"],
                }
            )
        ),
    )
    store.write(
        "fixtures",
        _utc(
            pl.DataFrame(
                {
                    "season": [season],
                    "fixture_id": [1],
                    "gw": [1],
                    "team_h_key": [3],
                    "team_a_key": [7],
                    "kickoff_time": [_KICKOFF_GW1],
                    "team_h_score": [None],
                    "team_a_score": [None],
                    "finished": [False],
                    "available_at": [_START],
                    "source": ["fpl_api"],
                }
            )
        ),
    )
    store.write(
        "player_status",
        _utc(
            pl.DataFrame(
                {
                    "season": [season, season],
                    "player_key": [100, 200],
                    "status": ["a", "a"],
                    "news": ["", ""],
                    "price": [70, 80],
                    "ep_next": [3.0, 4.0],
                    "chance_of_playing": [100.0, 100.0],
                    "penalties_order": [order, None],
                    "available_at": [_START, _START],
                    "source": [source, source],
                }
            )
        ),
    )
    store.write(
        "player_fixture_stats",
        _utc(
            pl.DataFrame(
                {
                    "season": [season, season],
                    "gw": [1, 1],
                    "player_key": [100, 200],
                    "fixture_id": [1, 1],
                    "minutes": [90, 90],
                    "total_points": [5, 6],
                    "goals_scored": [0, 0],
                    "assists": [0, 0],
                    "clean_sheets": [0, 0],
                    "goals_conceded": [1, 1],
                    "saves": [0, 0],
                    "bonus": [0, 0],
                    "yellow_cards": [0, 0],
                    "red_cards": [0, 0],
                    "own_goals": [0, 0],
                    "penalties_saved": [0, 0],
                    "penalties_missed": [0, 0],
                    "starts": [1, 1],
                    "influence": [0.0, 0.0],
                    "creativity": [0.0, 0.0],
                    "threat": [0.0, 0.0],
                    "bps": [10, 12],
                    "clearances_blocks_interceptions": [0, 0],
                    "tackles": [0, 0],
                    "recoveries": [0, 0],
                    "defensive_contribution": [0, 0],
                    "available_at": [_START, _START],
                    "source": ["fpl_api", "fpl_api"],
                }
            )
        ),
    )
    return store


def _seed_odds_store(root: Path, *, available_at: datetime) -> DataStore:
    store = DataStore(root)
    season = "2024-25"
    deadline = datetime(2024, 8, 16, 17, tzinfo=UTC)
    kickoff = datetime(2024, 8, 17, 14, tzinfo=UTC)
    start = datetime(2024, 8, 1, tzinfo=UTC)
    store.write(
        "gameweeks",
        _utc(
            pl.DataFrame(
                {
                    "season": [season],
                    "gw": [1],
                    "deadline_time": [deadline],
                    "finished": [False],
                    "data_checked": [False],
                    "available_at": [start],
                    "source": ["fpl_api"],
                }
            )
        ),
    )
    store.write(
        "fixtures",
        _utc(
            pl.DataFrame(
                {
                    "season": [season],
                    "fixture_id": [10],
                    "gw": [1],
                    "team_h_key": [3],
                    "team_a_key": [7],
                    "kickoff_time": [kickoff],
                    "team_h_score": [None],
                    "team_a_score": [None],
                    "finished": [False],
                    "available_at": [start],
                    "source": ["fpl_api"],
                }
            )
        ),
    )
    store.write(
        "match_odds",
        _utc(
            pl.DataFrame(
                {
                    "season": [season],
                    "fixture_id": [10],
                    "team_h_key": [3],
                    "team_a_key": [7],
                    "kickoff_time": [kickoff],
                    "home_odds": [2.1],
                    "draw_odds": [3.4],
                    "away_odds": [3.5],
                    "over_25_odds": [1.9],
                    "under_25_odds": [1.9],
                    "odds_kind": ["closing_proxy"],
                    "available_at": [available_at],
                    "source": ["football_data"],
                }
            )
        ),
    )
    return store


def test_eval_spec_rejects_kind_lever_mismatch() -> None:
    with pytest.raises(ValidationError):
        EvalSpec.model_validate(
            _base_payload(
                kind="model",
                change={
                    "lever": "component_swap",
                    "slot": "minutes",
                    "diff": ["bad"],
                },
            )
        )


def test_eval_spec_rejects_blank_question() -> None:
    with pytest.raises(ValidationError):
        EvalSpec.model_validate(_base_payload(question="   "))


def test_detect_points_claim_phrase() -> None:
    assert detect_points_claim_phrase("better season points") == "season points"
    assert detect_points_claim_phrase("squad points under H=3") == "squad points"
    assert detect_points_claim_phrase("mean PPG gate") == "mean ppg"
    assert detect_points_claim_phrase("realised points share") == "realised points"
    assert detect_points_claim_phrase("Does RMSE improve?") is None


def test_claim_squad_points_requires_decisions() -> None:
    with pytest.raises(ValidationError, match="claim=squad_points"):
        EvalSpec.model_validate(
            _base_payload(
                kind="model",
                claim="squad_points",
                question="Does the fork earn more season points?",
                change={
                    "lever": "model_fork",
                    "orientation": "opt_in",
                    "slot": None,
                    "control_kind": "rolling_mean",
                    "treatment_kind": "lgbm_direct",
                    "diff": ["kind: rolling_mean → lgbm_direct"],
                },
                control="rolling_mean_v1",
                treatment="lgbm_direct_v1",
                suites={
                    "prediction": True,
                    "components": False,
                    "decisions": False,
                },
            )
        )


def test_forecast_prose_points_claim_rejected() -> None:
    with pytest.raises(ValidationError, match="season points"):
        EvalSpec.model_validate(
            _base_payload(
                question="Does the treatment earn more season points?",
                suites={
                    "prediction": True,
                    "components": False,
                    "decisions": False,
                },
            )
        )


def test_claim_squad_points_loads_when_decisions_on() -> None:
    spec = EvalSpec.model_validate(
        _base_payload(
            kind="model",
            claim="squad_points",
            question="Does the fork earn more season points?",
            change={
                "lever": "model_fork",
                "orientation": "opt_in",
                "slot": None,
                "control_kind": "rolling_mean",
                "treatment_kind": "lgbm_direct",
                "diff": ["kind: rolling_mean → lgbm_direct"],
            },
            control="rolling_mean_v1",
            treatment="lgbm_direct_v1",
            suites={
                "prediction": True,
                "components": False,
                "decisions": True,
                "decisions_horizon": 3,
            },
        )
    )
    assert spec.claim == "squad_points"
    assert spec.suites.decisions is True
    assert decisions_claim_failures(spec) == []


def test_tier_s1_fails_squad_points_claim(tmp_path: Path) -> None:
    _write_eval(
        tmp_path,
        "squad_claim",
        _base_payload(
            name="squad_claim",
            kind="model",
            claim="squad_points",
            question="Does the fork earn more season points?",
            change={
                "lever": "model_fork",
                "orientation": "opt_in",
                "slot": None,
                "control_kind": "rolling_mean",
                "treatment_kind": "lgbm_direct",
                "diff": ["kind: rolling_mean → lgbm_direct"],
            },
            control="rolling_mean_v1",
            treatment="lgbm_direct_v1",
            suites={
                "prediction": True,
                "components": False,
                "decisions": True,
                "decisions_horizon": 3,
            },
        ),
    )
    spec = load_eval_spec("squad_claim", directory=tmp_path)
    plan = plan_promote_eval(spec, dry_run=True, tier="S1")
    assert plan.suites.decisions is False
    fails = decisions_claim_failures(spec, decisions_enabled=plan.suites.decisions)
    assert fails and "claim=squad_points" in fails[0]
    with patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False):
        result = preflight_promote_eval(spec, plan, data_dir=tmp_path)
    assert not result.ok
    assert any("claim=squad_points" in f for f in result.failures)


def test_equivalence_rejects_squad_points_claim() -> None:
    with pytest.raises(ValidationError, match="equivalence"):
        EvalSpec.model_validate(
            {
                "name": "env_claim",
                "kind": "environment",
                "claim": "squad_points",
                "question": "Does the lock upgrade keep season points?",
                "change": {
                    "lever": "dependency_upgrade",
                    "packages": ["numpy"],
                    "control_ref": "main",
                    "treatment_ref": "lock-bump",
                    "diff": ["uv.lock"],
                },
                "control": "dc_layered_v2",
                "treatment": "dc_layered_v2",
                "suites": {"prediction": True, "decisions": True},
            }
        )


def test_component_swap_requires_slot() -> None:
    with pytest.raises(ValidationError):
        EvalSpec.model_validate(
            _base_payload(
                change={
                    "lever": "component_swap",
                    "slot": None,
                    "diff": ["x"],
                }
            )
        )


def test_model_fork_forbids_slot() -> None:
    with pytest.raises(ValidationError):
        EvalSpec.model_validate(
            _base_payload(
                name="fork",
                kind="model",
                change={
                    "lever": "model_fork",
                    "slot": "minutes",
                    "diff": ["kind swap"],
                },
                control="rolling_mean_v1",
                treatment="lgbm_direct_v1",
                suites={"prediction": True, "components": False},
            )
        )


def test_data_table_requires_tables() -> None:
    with pytest.raises(ValidationError):
        EvalSpec.model_validate(
            _base_payload(
                name="odds",
                kind="data",
                change={
                    "lever": "data_table",
                    "tables": [],
                    "diff": ["match_odds"],
                },
                treatment="dc_layered_odds_v1",
            )
        )


def test_load_and_plan_dry_run(tmp_path: Path) -> None:
    _write_eval(tmp_path, "demo_swap", _base_payload())
    spec = load_eval_spec("demo_swap", directory=tmp_path)
    plan = plan_promote_eval(spec, dry_run=True, tier="S1")
    assert plan.eval_name == "demo_swap"
    assert plan.lever == "component_swap"
    assert plan.to_gw == 10
    assert any("eval-components" in cmd for cmd in plan.commands)
    component_cmds = [c for c in plan.commands if "eval-components" in c]
    assert any("--skip-sims" in cmd for cmd in component_cmds)
    assert any("omitted arm-specific model flags" in w for w in plan.warnings), (
        plan.warnings
    )
    assert any("compare" in cmd and "dc_layered_v2" in cmd for cmd in plan.commands)
    assert any(
        "ModelSpec 'dc_layered_minutes_lgbm_v1' not found" in w for w in plan.warnings
    )
    text = format_plan(plan)
    assert "component_swap" in text
    assert "dry-run" in text
    payload = plan_to_json(plan)
    assert '"eval_name": "demo_swap"' in payload


def test_components_command_names_score_slot_specs() -> None:
    spec = load_eval_spec("example_data_odds_prior")
    plan = plan_promote_eval(spec, dry_run=True, to_gw=2)
    components = [cmd for cmd in plan.commands if "eval-components" in cmd]
    assert len(components) == 1
    cmd = components[0]
    assert "--score-models dixon_coles_cs_v1 dixon_coles_odds_v1" in cmd
    assert "--skip-sims" in cmd
    assert "dixon_coles_v1" not in cmd
    assert "odds_poisson_v1" not in cmd


def test_components_command_names_involvement_slot_specs() -> None:
    spec = load_eval_spec("pens_order_involvement_ab")
    plan = plan_promote_eval(spec, dry_run=True, to_gw=1)
    components = [cmd for cmd in plan.commands if "eval-components" in cmd]
    assert len(components) == 1
    cmd = components[0]
    assert "--involvement-model involvement_xg_nopens_v1 involvement_xg_v1" in cmd
    assert "--skip-sims" in cmd
    assert "involvement_shares_v1" not in cmd


def test_involvement_xg_vs_shares_eval_loads() -> None:
    spec = load_eval_spec("involvement_xg_vs_shares_ab")
    assert spec.kind == "component"
    assert spec.change.lever == "component_swap"
    assert spec.control == "dc_layered_shares_v2"
    assert spec.treatment == "dc_layered_v4"
    assert spec.change.slot == "involvement"
    assert spec.change.control_kind == "involvement_shares"
    assert spec.change.treatment_kind == "involvement_xg"
    assert spec.window.to_gw == 20
    plan = plan_promote_eval(spec, dry_run=True, to_gw=1)
    components = [cmd for cmd in plan.commands if "eval-components" in cmd]
    assert len(components) == 1
    cmd = components[0]
    assert "--involvement-model involvement_shares_v1 involvement_xg_v1" in cmd


def test_xg_blend_and_set_piece_evals_load() -> None:
    blend = load_eval_spec("xg_blend_ab")
    assert blend.control == "dc_layered_v4"
    assert blend.treatment == "dc_layered_xg_blend_1_v1"
    assert blend.change.lever == "hyperparam"
    assert blend.suites.components is True
    set_piece = load_eval_spec("set_piece_involvement_ab")
    assert set_piece.treatment == "dc_layered_setpieces_v1"
    assert set_piece.gates.coverage.metric == "set_piece_order_non_null"
    event = load_eval_spec("penalty_event_involvement_ab")
    assert event.treatment == "dc_layered_penalty_event_v1"
    assert event.change.diff[0].startswith(
        "components.involvement.params.use_penalty_event"
    )


def test_n_sims_lever_omits_skip_sims(tmp_path: Path) -> None:
    payload = _base_payload(
        name="n_sims_ab",
        kind="config",
        change={
            "lever": "hyperparam",
            "orientation": "opt_in",
            "slot": "score",
            "control_kind": "dixon_coles",
            "treatment_kind": "dixon_coles",
            "diff": ["params.n_sims: 1000 → 10000"],
            "allow_multi_lever": True,
        },
        control="dc_layered_v4",
        treatment="dc_layered_odds_v1",
        suites={"prediction": False, "components": True},
    )
    _write_eval(tmp_path, "n_sims_ab", payload)
    spec = load_eval_spec("n_sims_ab", directory=tmp_path)
    assert is_n_sims_lever(spec)
    plan = plan_promote_eval(spec, dry_run=True)
    components = [cmd for cmd in plan.commands if "eval-components" in cmd]
    assert components
    assert all("--skip-sims" not in cmd for cmd in components)


def test_scoreline_helpers() -> None:
    odds = load_eval_spec("example_data_odds_prior")
    pens = load_eval_spec("pens_order_involvement_ab")
    assert is_scoreline_lever(odds)
    assert not is_scoreline_lever(pens)
    from fplt.models.spec import load_model_specs

    specs = load_model_specs()
    names = slot_component_names(
        specs["dc_layered_v4"], specs["dc_layered_odds_v1"], "score"
    )
    assert names == ("dixon_coles_cs_v1", "dixon_coles_odds_v1")
    market = raw_de_vig_spec(coverage_read="strict_pit")
    assert market.name == "raw_de_vig_1x2"
    assert market.kind == "odds_1x2"
    assert market.params["use_closing_proxy"] is False
    proxy = raw_de_vig_spec(coverage_read="closing_proxy_labelled")
    assert proxy.params["use_closing_proxy"] is True


def test_scoreline_execute_writes_per_match(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_odds_store(tmp_path, available_at=_KICKOFF_GW1)
    # Finished result so scoreline metrics have a join target.
    store = DataStore(tmp_path)
    fixtures = store.read("fixtures").with_columns(
        pl.lit(2).alias("team_h_score"),
        pl.lit(1).alias("team_a_score"),
        pl.lit(True).alias("finished"),
    )
    store.write("fixtures", fixtures)
    payload = yaml.safe_load(
        Path("configs/evals/example_data_odds_prior.yaml").read_text(encoding="utf-8")
    )
    payload["name"] = "odds_board"
    payload["suites"] = {
        "prediction": False,
        "components": True,
        "decisions": False,
        "steps": 1,
    }
    payload["gates"]["allow_low_coverage"] = True
    payload["window"]["to_gw"] = 1
    _write_eval(tmp_path, "odds_board", payload)
    spec = load_eval_spec("odds_board", directory=tmp_path)
    out = tmp_path / "out"

    per_match = pl.DataFrame(
        {
            "season": ["2024-25"] * 6,
            "gw": [1] * 6,
            "fixture_id": [10] * 6,
            "arm": [
                "dixon_coles_cs_v1",
                "dixon_coles_cs_v1",
                "dixon_coles_odds_v1",
                "dixon_coles_odds_v1",
                "raw_de_vig_1x2",
                "raw_de_vig_1x2",
            ],
            "metric": ["rps", "logloss"] * 3,
            "value": [0.3, 1.0, 0.29, 0.98, 0.2, 0.9],
        }
    )
    summary = pl.DataFrame(
        {
            "component": ["score"] * 3,
            "metric": ["rps"] * 3,
            "model_name": [
                "dixon_coles_cs_v1",
                "dixon_coles_odds_v1",
                "raw_de_vig_1x2",
            ],
            "slice": ["overall"] * 3,
            "value": [0.3, 0.29, 0.2],
            "n": [1, 1, 1],
        }
    )
    fake = ComponentEvalResult(
        summary=summary,
        sims=pl.DataFrame(),
        output_dir=out / "components" / "2024-25",
        per_match=per_match,
    )

    def _fake_run(*_args: object, **kwargs: object) -> ComponentEvalResult:
        assert kwargs.get("sims") == []
        assert kwargs.get("write_per_match") is True
        score_models = kwargs.get("score_models")
        assert score_models == ["dixon_coles_cs_v1", "dixon_coles_odds_v1"]
        extra = kwargs.get("extra_score_specs")
        assert isinstance(extra, list) and len(extra) == 1
        assert extra[0].name == "raw_de_vig_1x2"
        assert extra[0].kind == "odds_1x2"
        assert "odds_poisson" not in str(extra[0].kind)
        return fake

    with (
        patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False),
        patch("fplt.backtest.promote_eval.run_component_eval", side_effect=_fake_run),
    ):
        result = execute_promote_eval(
            spec,
            data_dir=tmp_path,
            dry_run=False,
            tier="S1",
            to_gw=1,
            allow_dirty=False,
            output_dir=out,
        )
    assert result.exit_code == 0
    assert result.verdict == "inconclusive"
    assert (out / "per_match.parquet").is_file()
    board = pl.read_parquet(out / "per_match.parquet")
    assert set(board["arm"].unique().to_list()) == {
        "dixon_coles_cs_v1",
        "dixon_coles_odds_v1",
        "raw_de_vig_1x2",
    }
    assert "odds_poisson_v1" not in board["arm"].to_list()
    assert "raw_de_vig_1x2" in result.recommendation
    assert "not odds_poisson_v1" in result.recommendation


def test_tier_disables_decisions(tmp_path: Path) -> None:
    _write_eval(
        tmp_path,
        "demo_swap",
        _base_payload(
            suites={
                "prediction": True,
                "components": False,
                "decisions": True,
                "decisions_horizon": 3,
                "max_mean_ppg": 90,
            }
        ),
    )
    spec = load_eval_spec("demo_swap", directory=tmp_path)
    plan = plan_promote_eval(spec, dry_run=True, tier="S1")
    assert plan.suites.decisions is False
    assert any("disables decisions" in w for w in plan.warnings)
    assert not any("decisions" in cmd for cmd in plan.commands)


def test_repo_example_evals_load() -> None:
    for name in (
        "example_config_dc_v1_vs_v2",
        "example_component_minutes_v1_vs_v2",
        "example_model_lgbm_vs_rolling",
        "example_model_squad_points",
        "example_data_odds_prior",
        "example_environment_lock_upgrade",
        "example_refactor_no_behaviour",
        "pens_order_involvement_ab",
    ):
        spec = load_eval_spec(name)
        plan = plan_promote_eval(spec, dry_run=True)
        assert plan.eval_name == name
        assert plan.commands
        assert spec.question.strip()
        if name == "example_model_squad_points":
            assert spec.claim == "squad_points"
            assert spec.suites.decisions is True
            assert plan.suites.decisions is True
        else:
            assert spec.claim == "forecasts"


def test_cli_promote_eval_dry_run(capsys: pytest.CaptureFixture[str]) -> None:
    main(
        [
            "promote-eval",
            "--eval",
            "example_model_lgbm_vs_rolling",
            "--dry-run",
            "--tier",
            "S1",
        ]
    )
    out = capsys.readouterr().out
    assert "promote-eval example_model_lgbm_vs_rolling" in out
    assert "model_fork" in out
    assert "fplt compare" in out
    assert "dry-run" in out


def test_cli_promote_eval_missing_eval() -> None:
    with pytest.raises(SystemExit) as exc:
        main(["promote-eval", "--eval", "does_not_exist_zz"])
    assert exc.value.code == 2


def test_preflight_orientation_pens(tmp_path: Path) -> None:
    _seed_pens_store(tmp_path, order=1)
    spec = load_eval_spec("pens_order_involvement_ab")
    plan = plan_promote_eval(spec, dry_run=False, to_gw=1)
    with patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False):
        result = preflight_promote_eval(
            spec, plan, data_dir=tmp_path, allow_dirty=False
        )
    assert result.ok
    assert result.coverage is not None
    assert result.coverage.metric == "penalties_order_non_null"
    assert result.coverage.passed
    assert result.effective_flags["control.use_penalties_order"] is False
    assert result.effective_flags["treatment.use_penalties_order"] is True


def test_shared_market_cs_does_not_fail_odds_orientation(tmp_path: Path) -> None:
    """Production CS on both arms must not trip ``no contrast`` (FDB-267)."""
    before = datetime(2024, 8, 16, 12, tzinfo=UTC)
    _seed_odds_store(tmp_path, available_at=before)
    payload = _base_payload(
        name="odds_shared_cs",
        kind="config",
        question="odds prior with shared CS?",
        change={
            "lever": "hyperparam",
            "orientation": "opt_in",
            "slot": "score",
            "control_kind": "dixon_coles",
            "treatment_kind": "dixon_coles",
            "tables": ["match_odds"],
            "diff": ["market_prior_strength: 0 → 1.0"],
        },
        control="dc_layered_v4",
        treatment="dc_layered_odds_v1",
        window={
            "seasons": ["2024-25"],
            "from_gw": 1,
            "to_gw": 1,
            "modes": ["walk_forward"],
        },
        suites={"prediction": True, "components": False},
        gates={
            "min_coverage": 0.80,
            "coverage": {"metric": "odds_join", "read": "strict_pit"},
            "drop_if": {"coverage_below_min": True, "mean_ppg_gate_fail": True},
        },
    )
    _write_eval(tmp_path, "odds_shared_cs", payload)
    spec = load_eval_spec("odds_shared_cs", directory=tmp_path)
    plan = plan_promote_eval(spec, dry_run=False)
    with patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False):
        result = preflight_promote_eval(spec, plan, data_dir=tmp_path)
    assert result.ok
    assert not any("no contrast on market_cs_strength" in f for f in result.failures)
    assert "control.market_cs_strength" not in result.effective_flags
    assert "treatment.market_cs_strength" not in result.effective_flags
    assert result.effective_flags.get("treatment.market_prior_strength") == 1.0
    assert result.effective_flags.get("control.market_prior_strength") == 0.0


def test_preflight_pens_coverage_drop_on_all_null(tmp_path: Path) -> None:
    _seed_pens_store(tmp_path, order=None, source="vaastav")
    spec = load_eval_spec("pens_order_involvement_ab")
    plan = plan_promote_eval(spec, dry_run=False, to_gw=1)
    with patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False):
        result = preflight_promote_eval(spec, plan, data_dir=tmp_path)
    assert result.ok  # coverage miss is not a structural failure
    assert result.coverage is not None
    assert not result.coverage.passed
    assert result.coverage.rate == 0.0


def test_preflight_odds_strict_pit_misses_kickoff_rows(tmp_path: Path) -> None:
    kickoff = datetime(2024, 8, 17, 14, tzinfo=UTC)
    _seed_odds_store(tmp_path, available_at=kickoff)
    payload = _base_payload(
        name="odds_ab",
        kind="config",
        question="Does the odds prior help?",
        change={
            "lever": "hyperparam",
            "orientation": "opt_in",
            "slot": "score",
            "control_kind": "dixon_coles",
            "treatment_kind": "dixon_coles",
            "tables": ["match_odds"],
            "diff": ["market_prior_strength: 0 → 1.0"],
        },
        control="dc_layered_v4",
        treatment="dc_layered_odds_v1",
        window={
            "seasons": ["2024-25"],
            "from_gw": 1,
            "to_gw": 1,
            "modes": ["walk_forward"],
        },
        suites={"prediction": True, "components": False},
        gates={
            "min_coverage": 0.80,
            "coverage": {"metric": "odds_join", "read": "strict_pit"},
            "drop_if": {"coverage_below_min": True, "mean_ppg_gate_fail": True},
        },
    )
    _write_eval(tmp_path, "odds_ab", payload)
    spec = load_eval_spec("odds_ab", directory=tmp_path)
    plan = plan_promote_eval(spec, dry_run=False)
    with patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False):
        result = preflight_promote_eval(spec, plan, data_dir=tmp_path)
    assert result.coverage is not None
    assert result.coverage.rate == 0.0
    assert not result.coverage.passed
    assert result.pit_caveat is not None
    assert "Strict-PIT" in result.pit_caveat


def test_preflight_dc_season_eligible(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    for season, gw_deadline in (
        ("2024-25", datetime(2024, 8, 16, 17, tzinfo=UTC)),
        ("2025-26", _DEADLINE_GW1),
    ):
        store.write(
            "gameweeks",
            _utc(
                pl.DataFrame(
                    {
                        "season": [season],
                        "gw": [1],
                        "deadline_time": [gw_deadline],
                        "finished": [False],
                        "data_checked": [False],
                        "available_at": [_START],
                        "source": ["fpl_api"],
                    }
                )
            ),
        )
        store.write(
            "fixtures",
            _utc(
                pl.DataFrame(
                    {
                        "season": [season],
                        "fixture_id": [1 if season == "2024-25" else 2],
                        "gw": [1],
                        "team_h_key": [3],
                        "team_a_key": [7],
                        "kickoff_time": [gw_deadline + timedelta(days=1)],
                        "team_h_score": [None],
                        "team_a_score": [None],
                        "finished": [False],
                        "available_at": [_START],
                        "source": ["fpl_api"],
                    }
                )
            ),
        )
    payload = _base_payload(
        name="defcon_ab",
        kind="config",
        question="Does defensive contribution help on 2025-26?",
        change={
            "lever": "hyperparam",
            "orientation": "opt_out",
            "slot": "other",
            "control_kind": "other_player",
            "treatment_kind": "other_player",
            "diff": ["use_defensive_contribution: false → true"],
        },
        control="dc_layered_nodefcon_v1",
        treatment="dc_layered_v4",
        window={
            "seasons": ["2024-25", "2025-26"],
            "from_gw": 1,
            "to_gw": 1,
            "modes": ["walk_forward"],
        },
        suites={"prediction": True, "components": False},
        gates={
            "min_coverage": 1.0,
            "coverage": {"metric": "dc_season_eligible", "read": "strict_pit"},
            "drop_if": {"coverage_below_min": True, "mean_ppg_gate_fail": True},
        },
    )
    _write_eval(tmp_path, "defcon_ab", payload)
    spec = load_eval_spec("defcon_ab", directory=tmp_path)
    plan = plan_promote_eval(spec, dry_run=False)
    with patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False):
        result = preflight_promote_eval(spec, plan, data_dir=tmp_path)
    assert result.coverage is not None
    assert result.coverage.denominator == 2
    assert result.coverage.numerator == 1
    assert not result.coverage.passed


def test_preflight_dirty_tree_fails(tmp_path: Path) -> None:
    _seed_pens_store(tmp_path, order=1)
    spec = load_eval_spec("pens_order_involvement_ab")
    plan = plan_promote_eval(spec, dry_run=False, to_gw=1)
    with patch("fplt.backtest.promote_eval.git_is_dirty", return_value=True):
        result = preflight_promote_eval(
            spec, plan, data_dir=tmp_path, allow_dirty=False
        )
    assert not result.ok
    assert any("dirty" in f for f in result.failures)


def test_preflight_allow_dirty_forces_inconclusive(tmp_path: Path) -> None:
    _seed_pens_store(tmp_path, order=1)
    spec = load_eval_spec("pens_order_involvement_ab")
    plan = plan_promote_eval(spec, dry_run=False, to_gw=1)
    with patch("fplt.backtest.promote_eval.git_is_dirty", return_value=True):
        result = preflight_promote_eval(spec, plan, data_dir=tmp_path, allow_dirty=True)
    assert result.ok
    assert result.force_inconclusive
    assert result.git_dirty


def test_preflight_arms_inverted(tmp_path: Path) -> None:
    _seed_pens_store(tmp_path, order=1)
    payload = _base_payload(
        name="inverted",
        kind="config",
        question="inverted arms?",
        change={
            "lever": "hyperparam",
            "orientation": "opt_out",
            "slot": "involvement",
            "control_kind": "involvement_xg",
            "treatment_kind": "involvement_xg",
            "tables": ["player_status"],
            "diff": ["use_penalties_order swapped"],
        },
        # Swapped relative to opt_out: control should be off-fork.
        control="dc_layered_v4",
        treatment="dc_layered_nopens_v1",
        window={
            "seasons": ["2025-26"],
            "from_gw": 1,
            "to_gw": 1,
            "modes": ["walk_forward"],
        },
        suites={"prediction": True, "components": False},
        gates={
            "min_coverage": 0.01,
            "coverage": {
                "metric": "penalties_order_non_null",
                "read": "strict_pit",
            },
        },
    )
    _write_eval(tmp_path, "inverted", payload)
    spec = load_eval_spec("inverted", directory=tmp_path)
    plan = plan_promote_eval(spec, dry_run=False)
    with patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False):
        result = preflight_promote_eval(spec, plan, data_dir=tmp_path)
    assert not result.ok
    assert any("inverted" in f for f in result.failures)


def test_preflight_multi_lever_without_allow(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    store.write(
        "gameweeks",
        _utc(
            pl.DataFrame(
                {
                    "season": ["2024-25"],
                    "gw": [1],
                    "deadline_time": [datetime(2024, 8, 16, 17, tzinfo=UTC)],
                    "finished": [False],
                    "data_checked": [False],
                    "available_at": [_START],
                    "source": ["fpl_api"],
                }
            )
        ),
    )
    payload = _base_payload(
        name="multi",
        kind="config",
        question="multi lever?",
        change={
            "lever": "hyperparam",
            "slot": "involvement",
            "control_kind": "involvement_shares",
            "treatment_kind": "involvement_xg",
            "allow_multi_lever": False,
            "diff": ["v1 → v2"],
        },
        control="dc_layered_v1",
        treatment="dc_layered_v2",
        window={
            "seasons": ["2024-25"],
            "from_gw": 1,
            "to_gw": 1,
            "modes": ["walk_forward"],
        },
        suites={"prediction": True, "components": False},
    )
    _write_eval(tmp_path, "multi", payload)
    spec = load_eval_spec("multi", directory=tmp_path)
    plan = plan_promote_eval(spec, dry_run=False)
    with patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False):
        result = preflight_promote_eval(spec, plan, data_dir=tmp_path)
    assert not result.ok
    assert any("multi-lever" in f or "param family" in f for f in result.failures)


def test_execute_coverage_drop_writes_recommendation(tmp_path: Path) -> None:
    _seed_pens_store(tmp_path, order=None, source="vaastav")
    spec = load_eval_spec("pens_order_involvement_ab")
    out = tmp_path / "out"
    with patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False):
        result = execute_promote_eval(
            spec,
            data_dir=tmp_path,
            dry_run=False,
            to_gw=1,
            output_dir=out,
            argv=["promote-eval", "--execute", "--eval", "pens_order_involvement_ab"],
        )
    assert result.verdict == "drop"
    assert result.exit_code == 1
    rec = (out / "recommendation.md").read_text(encoding="utf-8")
    assert "question:" in rec
    assert "**drop**" in rec
    assert "Coverage" in rec
    audit = json.loads((out / "audit.json").read_text(encoding="utf-8"))
    assert audit["git_dirty"] is False
    assert "uv_lock_sha256" in audit
    assert "config" in audit
    assert result.experiment_id is not None
    assert (out / "experiment_id.txt").read_text(encoding="utf-8").strip() == (
        result.experiment_id
    )
    exp_root = tmp_path / "experiments" / f"exp={result.experiment_id}"
    assert (exp_root / "manifest.json").is_file()
    assert (exp_root / "results" / "verdict.json").is_file()
    frozen = json.loads((exp_root / "manifest.json").read_text(encoding="utf-8"))
    assert frozen["gates"] == spec.gates.model_dump(mode="json")
    assert frozen["eval_spec_hash"]


def test_execute_allow_dirty_inconclusive_skips_compare_on_s0(
    tmp_path: Path,
) -> None:
    _seed_pens_store(tmp_path, order=1)
    spec = load_eval_spec("pens_order_involvement_ab")
    out = tmp_path / "out"
    with (
        patch("fplt.backtest.promote_eval.git_is_dirty", return_value=True),
        patch("fplt.backtest.promote_eval.compare_models") as compare,
    ):
        result = execute_promote_eval(
            spec,
            data_dir=tmp_path,
            dry_run=False,
            tier="S0",
            to_gw=1,
            allow_dirty=True,
            output_dir=out,
        )
    compare.assert_not_called()
    assert result.verdict == "inconclusive"
    assert result.exit_code == 0
    rec = (out / "recommendation.md").read_text(encoding="utf-8")
    assert "--allow-dirty" in rec
    assert "inconclusive" in rec


def test_execute_runs_compare_and_stays_inconclusive(tmp_path: Path) -> None:
    _seed_pens_store(tmp_path, order=1)
    spec = load_eval_spec("pens_order_involvement_ab")
    out = tmp_path / "out"

    class _FakeCompare:
        summary = pl.DataFrame(
            {
                "model_name": ["dc_layered_nopens_v1", "dc_layered_v4"],
                "mode": ["walk_forward", "walk_forward"],
                "horizon": [1, 1],
                "mae": [1.0, 0.9],
                "rmse": [1.5, 1.4],
                "spearman": [0.1, 0.2],
                "top10_precision": [0.3, 0.4],
            }
        )
        detail = pl.DataFrame(
            {
                "model_name": ["dc_layered_nopens_v1", "dc_layered_v4"],
                "mode": ["walk_forward", "walk_forward"],
                "horizon": [1, 1],
                "target_gw": [1, 1],
                "mae": [1.0, 0.9],
            }
        )
        output_dir = out / "compare"
        runs: ClassVar[list[Any]] = []
        season = "2025-26"
        test_from_gw = 1
        test_to_gw = 1
        fit_gw = 1
        seasons_back = 4
        modes = ("walk_forward",)
        time_decay = None
        skipped_openfpl = None
        calibration = pl.DataFrame()

    with (
        patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False),
        patch(
            "fplt.backtest.promote_eval.compare_models",
            return_value=_FakeCompare(),
        ) as compare,
        patch(
            "fplt.backtest.promote_eval.read_predictions",
            return_value=pl.DataFrame(),
        ),
    ):
        result = execute_promote_eval(
            spec,
            data_dir=tmp_path,
            dry_run=False,
            to_gw=1,
            output_dir=out,
            argv=["promote-eval", "--execute"],
        )
    compare.assert_called_once()
    assert result.verdict == "inconclusive"
    assert result.exit_code == 0
    rec = (out / "recommendation.md").read_text(encoding="utf-8")
    assert "rmse_delta" in rec
    assert "mae_delta" in rec
    assert rec.index("rmse_delta") < rec.index("mae_delta")
    assert "headline" in rec
    assert "W9-style" in rec or "decision-relevant" in rec or "player pool" in rec
    assert "**inconclusive**" in rec
    assert "decisions: not run" in rec
    assert (out / "audit.json").is_file()
    assert (out / "summary.parquet").is_file()


def test_render_recommendation_order() -> None:
    spec = load_eval_spec("pens_order_involvement_ab")
    plan = plan_promote_eval(spec, dry_run=True, to_gw=1)
    from fplt.backtest.promote_eval import CoverageResult, PreflightResult

    preflight = PreflightResult(
        ok=True,
        coverage=CoverageResult(
            metric="penalties_order_non_null",
            numerator=1,
            denominator=100,
            rate=0.01,
            min_coverage=0.01,
            passed=True,
            extras={"penalties_order_eq_1_rate": 0.01},
        ),
        git_sha="abc",
        git_dirty=False,
        control_spec=ModelSpec(name="c", kind="dc_layered"),
        treatment_spec=ModelSpec(name="t", kind="dc_layered"),
        effective_flags={
            "control.use_penalties_order": False,
            "treatment.use_penalties_order": True,
        },
    )
    text = render_recommendation(
        spec,
        plan,
        preflight,
        verdict="inconclusive",
        allow_dirty=False,
        compare=None,
        deltas={
            "mae": 0.01,
            "per_gw_wins": {"control": 1, "treatment": 2, "ties": 0},
            "intervals": [
                {
                    "metric": "mae",
                    "position": None,
                    "mean": 0.01,
                    "hac_se": 0.02,
                    "hac_pvalue": 0.5,
                    "hac_ci": [-0.03, 0.05],
                    "bootstrap_ci": [-0.02, 0.04],
                }
            ],
        },
        prediction_ran=True,
    )
    assert "HAC_CI=" in text
    assert "bootstrap_CI=" in text
    assert "not a keep rule" in text
    # §9 order: title/question → arms → audit → window → coverage → join → caveat
    # → deltas → components → decisions → equivalence → call
    markers = [
        "question:",
        "## Arms",
        "## Audit",
        "## Window",
        "## Coverage",
        "## Join / available_at",
        "## PIT caveat",
        "## Prediction deltas",
        "## Components",
        "## Decisions",
        "## Equivalence",
        "## Call",
    ]
    positions = [text.index(marker) for marker in markers]
    assert positions == sorted(positions)


def test_cli_promote_eval_execute_coverage_drop(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], monkeypatch: pytest.MonkeyPatch
) -> None:
    _seed_pens_store(tmp_path, order=None)
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings = __import__("fplt.settings", fromlist=["get_settings"]).get_settings
    get_settings.cache_clear()
    with patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False):
        with pytest.raises(SystemExit) as exc:
            main(
                [
                    "promote-eval",
                    "--eval",
                    "pens_order_involvement_ab",
                    "--execute",
                    "--to-gw",
                    "1",
                    "--output-dir",
                    str(tmp_path / "cli-out"),
                ]
            )
    assert exc.value.code == 1
    out = capsys.readouterr().out
    assert "**drop**" in out
    get_settings.cache_clear()


def test_allow_low_coverage_forces_inconclusive(tmp_path: Path) -> None:
    _seed_pens_store(tmp_path, order=None, source="vaastav")
    payload = yaml.safe_load(
        (Path("configs/evals/pens_order_involvement_ab.yaml")).read_text(
            encoding="utf-8"
        )
    )
    payload["name"] = "pens_low"
    payload["gates"]["allow_low_coverage"] = True
    _write_eval(tmp_path, "pens_low", payload)
    spec = load_eval_spec("pens_low", directory=tmp_path)
    out = tmp_path / "out"
    with (
        patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False),
        patch("fplt.backtest.promote_eval.compare_models") as compare,
    ):
        result = execute_promote_eval(
            spec,
            data_dir=tmp_path,
            dry_run=False,
            tier="S0",
            to_gw=1,
            output_dir=out,
        )
    compare.assert_not_called()
    assert result.verdict == "inconclusive"
    assert result.exit_code == 0
    assert result.preflight.force_inconclusive
    assert result.preflight.coverage is not None
    assert not result.preflight.coverage.passed


def test_leak_fails_preflight(tmp_path: Path) -> None:
    _seed_pens_store(tmp_path, order=1)
    deadline = _DEADLINE_GW1
    feature_dir = tmp_path / "features" / "context" / "v=1" / "as_of=2025-26_gw01"
    feature_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "season": ["2025-26"],
            "fixture_id": [1],
            "player_key": [100],
            "available_at": [deadline + timedelta(hours=2)],
            "value": [1.0],
        }
    ).write_parquet(feature_dir / "part.parquet")
    payload = _base_payload(
        name="leak_demo",
        kind="feature",
        question="leak?",
        change={
            "lever": "feature_version",
            "orientation": "opt_in",
            "features": ["context@1"],
            "allow_multi_lever": True,
            "diff": ["context@1"],
        },
        control="rolling_mean_v1",
        treatment="lgbm_direct_v1",
        window={
            "seasons": ["2025-26"],
            "from_gw": 1,
            "to_gw": 1,
            "modes": ["walk_forward"],
        },
        suites={"prediction": True, "components": False},
    )
    _write_eval(tmp_path, "leak_demo", payload)
    spec = load_eval_spec("leak_demo", directory=tmp_path)
    out = tmp_path / "out"
    with patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False):
        result = execute_promote_eval(
            spec,
            data_dir=tmp_path,
            dry_run=False,
            output_dir=out,
        )
    assert result.verdict == "drop"
    assert result.exit_code == 2
    assert result.preflight.leak
    assert "leak" in (out / "recommendation.md").read_text(encoding="utf-8").lower()


def test_odds_join_passes_when_available_before_deadline(tmp_path: Path) -> None:
    before = datetime(2024, 8, 16, 12, tzinfo=UTC)  # before deadline 17:00
    _seed_odds_store(tmp_path, available_at=before)
    payload = _base_payload(
        name="odds_ok",
        kind="config",
        question="odds visible?",
        change={
            "lever": "hyperparam",
            "orientation": "opt_in",
            "slot": "score",
            "control_kind": "dixon_coles",
            "treatment_kind": "dixon_coles",
            "tables": ["match_odds"],
            "diff": ["market_prior_strength: 0 → 1.0"],
        },
        control="dc_layered_v4",
        treatment="dc_layered_odds_v1",
        window={
            "seasons": ["2024-25"],
            "from_gw": 1,
            "to_gw": 1,
            "modes": ["walk_forward"],
        },
        suites={"prediction": True, "components": False},
        gates={
            "min_coverage": 0.80,
            "coverage": {"metric": "odds_join", "read": "strict_pit"},
            "drop_if": {"coverage_below_min": True, "mean_ppg_gate_fail": True},
        },
    )
    _write_eval(tmp_path, "odds_ok", payload)
    spec = load_eval_spec("odds_ok", directory=tmp_path)
    plan = plan_promote_eval(spec, dry_run=False)
    with patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False):
        result = preflight_promote_eval(spec, plan, data_dir=tmp_path)
    assert result.ok
    assert result.coverage is not None
    assert result.coverage.passed
    assert result.coverage.rate == 1.0


def test_structural_preflight_fail_exits_2(tmp_path: Path) -> None:
    payload = _base_payload(
        name="missing_arm",
        kind="model",
        question="missing?",
        change={
            "lever": "model_fork",
            "control_kind": "rolling_mean",
            "treatment_kind": "lgbm_direct",
            "diff": ["kind"],
        },
        control="rolling_mean_v1",
        treatment="does_not_exist_model",
        suites={"prediction": True, "components": False},
    )
    _write_eval(tmp_path, "missing_arm", payload)
    spec = load_eval_spec("missing_arm", directory=tmp_path)
    out = tmp_path / "out"
    with patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False):
        result = execute_promote_eval(
            spec, data_dir=tmp_path, dry_run=False, output_dir=out
        )
    assert result.exit_code == 2
    assert not (out / "recommendation.md").exists()


def test_closing_proxy_read_requires_flag(tmp_path: Path) -> None:
    kickoff = datetime(2024, 8, 17, 14, tzinfo=UTC)
    _seed_odds_store(tmp_path, available_at=kickoff)
    payload = _base_payload(
        name="proxy_bad",
        kind="config",
        question="proxy?",
        change={
            "lever": "hyperparam",
            "orientation": "opt_in",
            "slot": "score",
            "control_kind": "dixon_coles",
            "treatment_kind": "dixon_coles",
            "tables": ["match_odds"],
            "diff": ["market_prior_strength"],
        },
        control="dc_layered_v4",
        treatment="dc_layered_odds_v1",
        window={
            "seasons": ["2024-25"],
            "from_gw": 1,
            "to_gw": 1,
            "modes": ["walk_forward"],
        },
        suites={"prediction": True, "components": False},
        gates={
            "min_coverage": 0.8,
            "coverage": {
                "metric": "odds_join",
                "read": "closing_proxy_labelled",
            },
        },
    )
    _write_eval(tmp_path, "proxy_bad", payload)
    spec = load_eval_spec("proxy_bad", directory=tmp_path)
    plan = plan_promote_eval(spec, dry_run=False)
    with patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False):
        result = preflight_promote_eval(spec, plan, data_dir=tmp_path)
    assert not result.ok
    assert any("closing_proxy_labelled" in f for f in result.failures)


def test_prematch_coverage_passes_for_cs_only_arm(tmp_path: Path) -> None:
    # Prematch collected Fri afternoon; FPL deadline Sat 11:00 → legal.
    available = datetime(2024, 8, 16, 16, tzinfo=UTC)
    store = _seed_odds_store(tmp_path, available_at=available)
    odds = store.read("match_odds").with_columns(
        pl.lit("prematch_scheduled").alias("odds_kind")
    )
    store.write("match_odds", _utc(odds))
    payload = _base_payload(
        name="cs_prematch_ok",
        kind="data",
        question="prematch CS coverage?",
        change={
            "lever": "data_table",
            "orientation": "opt_out",
            "slot": "score",
            "control_kind": "dixon_coles",
            "treatment_kind": "dixon_coles",
            "tables": ["match_odds"],
            "diff": ["market_cs_strength: 0 → 0.5"],
        },
        control="dc_layered_nocs_v1",
        treatment="dc_layered_v4",
        window={
            "seasons": ["2024-25"],
            "from_gw": 1,
            "to_gw": 1,
            "modes": ["walk_forward"],
        },
        suites={"prediction": False, "components": True},
        gates={
            "min_coverage": 0.80,
            "coverage": {"metric": "odds_join", "read": "prematch_scheduled"},
            "drop_if": {"coverage_below_min": True, "mean_ppg_gate_fail": True},
        },
    )
    _write_eval(tmp_path, "cs_prematch_ok", payload)
    spec = load_eval_spec("cs_prematch_ok", directory=tmp_path)
    plan = plan_promote_eval(spec, dry_run=False)
    with patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False):
        result = preflight_promote_eval(spec, plan, data_dir=tmp_path)
    assert result.ok
    assert result.coverage is not None
    assert result.coverage.passed
    assert result.coverage.rate == 1.0
    assert result.coverage.extras.get("read") == "prematch_scheduled"
    assert result.effective_flags.get("treatment.market_cs_strength") == 0.5


def test_prematch_read_rejects_closing_proxy_flag(tmp_path: Path) -> None:
    from fplt.backtest.promote_eval import _coverage_odds_join

    available = datetime(2024, 8, 16, 16, tzinfo=UTC)
    store = _seed_odds_store(tmp_path, available_at=available)
    payload = _base_payload(
        name="cs_proxy_bad",
        kind="data",
        question="proxy with prematch read?",
        change={
            "lever": "data_table",
            "orientation": "opt_out",
            "slot": "score",
            "control_kind": "dixon_coles",
            "treatment_kind": "dixon_coles",
            "tables": ["match_odds"],
            "diff": ["market_cs_strength"],
        },
        control="dc_layered_nocs_v1",
        treatment="dc_layered_v4",
        window={
            "seasons": ["2024-25"],
            "from_gw": 1,
            "to_gw": 1,
            "modes": ["walk_forward"],
        },
        suites={"prediction": False, "components": True},
        gates={
            "min_coverage": 0.8,
            "coverage": {"metric": "odds_join", "read": "prematch_scheduled"},
        },
    )
    _write_eval(tmp_path, "cs_proxy_bad", payload)
    spec = load_eval_spec("cs_proxy_bad", directory=tmp_path)
    plan = plan_promote_eval(spec, dry_run=False)
    bad_treatment = ModelSpec(
        name="dc_layered_cs_proxy",
        kind="dc_layered",
        components={
            "score": ModelSpec(
                name="dixon_coles_cs_proxy",
                kind="dixon_coles",
                params={"market_cs_strength": 0.5, "use_closing_proxy": True},
            )
        },
    )
    fixtures = store.read("fixtures")
    with pytest.raises(ValueError, match="prematch_scheduled"):
        _coverage_odds_join(spec, plan, store, fixtures, bad_treatment)


def test_cs_only_eval_spec_loads_from_configs() -> None:
    spec = load_eval_spec("cs_only_market_lambda_ab")
    assert spec.control == "dc_layered_nocs_v1"
    assert spec.treatment == "dc_layered_v4"
    assert spec.change.orientation == "opt_out"
    assert spec.gates.coverage.read == "prematch_scheduled"
    assert spec.gates.min_coverage == 0.80
    assert spec.suites.components is True
    assert spec.suites.prediction is False
    assert "brier_cs_home" in (spec.gates.keep_if.components_any or [])


def test_cs_only_player_rmse_eval_spec_loads_from_configs() -> None:
    """FDB-239: prediction+components CS-only revisit under protocol windows."""
    spec = load_eval_spec("cs_only_player_rmse_ab")
    assert spec.kind == "data"
    assert spec.control == "dc_layered_nocs_v1"
    assert spec.treatment == "dc_layered_v4"
    assert spec.change.orientation == "opt_out"
    assert spec.change.allow_multi_lever is False
    assert spec.claim == "forecasts"
    assert spec.gates.coverage.read == "prematch_scheduled"
    assert spec.gates.coverage.metric == "odds_join"
    assert spec.gates.min_coverage == 0.80
    assert spec.suites.prediction is True
    assert spec.suites.components is True
    assert spec.suites.decisions is False
    assert spec.window.from_gw == 1
    assert spec.window.to_gw == 20
    assert "Development" in spec.window.note or "development" in spec.window.note
    assert "rmse" in (spec.gates.keep_if.prediction_any or [])
    assert "brier_cs_home" in (spec.gates.keep_if.components_any or [])
    assert "brier_cs_away" in (spec.gates.keep_if.components_any or [])
    from fplt.models.spec import load_model_specs

    treatment = load_model_specs()[spec.treatment]
    score = treatment.components["score"]
    assert score.params["market_cs_strength"] == 0.5
    assert score.params.get("use_closing_proxy", False) is False
    assert "market_1x2_strength" not in score.params


def test_dynamic_covariate_ratings_eval_spec_loads_from_configs() -> None:
    """FDB-243: opt-in rest/congestion/form covariates; production unchanged."""
    spec = load_eval_spec("dynamic_covariate_ratings_ab")
    assert spec.kind == "config"
    assert spec.control == "dc_layered_v4"
    assert spec.treatment == "dc_layered_covariate_v1"
    assert spec.change.orientation == "opt_in"
    assert spec.change.allow_multi_lever is True
    assert spec.change.slot == "score"
    assert spec.claim == "forecasts"
    assert spec.suites.prediction is True
    assert spec.suites.components is True
    assert spec.suites.decisions is False
    assert spec.window.from_gw == 1
    assert spec.window.to_gw == 20
    assert "rps" in (spec.gates.keep_if.components_any or [])
    assert "logloss" in (spec.gates.keep_if.components_any or [])
    from fplt.models.spec import load_model_specs

    specs = load_model_specs()
    control = specs[spec.control].components["score"]
    treatment = specs[spec.treatment].components["score"]
    assert control.params.get("rest_coef", 0.0) == 0.0
    assert treatment.params["rest_coef"] == pytest.approx(0.05)
    assert treatment.params["congestion_coef"] == pytest.approx(0.05)
    assert treatment.params["form_window"] == 5
    assert treatment.params["form_strength"] == pytest.approx(0.2)
    assert treatment.params["market_cs_strength"] == pytest.approx(0.5)
    assert treatment.name == "dixon_coles_covariate_v1"
    plan = plan_promote_eval(spec, dry_run=True)
    assert plan.control == "dc_layered_v4"
    assert plan.treatment == "dc_layered_covariate_v1"
    assert any("eval-components" in cmd for cmd in plan.commands)


def test_prematch_1x2_eval_spec_loads_from_configs() -> None:
    spec = load_eval_spec("prematch_1x2_prior_ab")
    assert spec.control == "dc_layered_v4"
    assert spec.treatment == "dc_layered_1x2_prematch_v1"
    assert spec.change.orientation == "opt_in"
    assert spec.change.allow_multi_lever is True
    assert spec.gates.coverage.read == "prematch_scheduled"
    assert spec.gates.min_coverage == 0.80
    assert spec.suites.prediction is True
    assert spec.suites.components is True
    assert "rmse" in (spec.gates.keep_if.prediction_any or [])
    assert "rps" in (spec.gates.keep_if.components_any or [])
    from fplt.models.spec import load_model_specs

    treatment = load_model_specs()[spec.treatment]
    score = treatment.components["score"]
    assert score.params["market_1x2_strength"] == 1.0
    assert score.params["use_closing_proxy"] is False


def _seed_overlay_base(root: Path, *, status_source: str = "fpl_api") -> DataStore:
    """Minimal scored window for availability / injury / XI coverage."""
    store = _seed_pens_store(root, order=1, source=status_source)
    season = "2025-26"
    store.write(
        "players",
        _utc(
            pl.DataFrame(
                {
                    "season": [season, season],
                    "player_key": [100, 200],
                    "element_id": [1, 2],
                    "team_key": [3, 7],
                    "position": ["MID", "FWD"],
                    "web_name": ["A", "B"],
                    "available_at": [_START, _START],
                    "source": ["fpl_api", "fpl_api"],
                }
            )
        ),
    )
    return store


def test_overlay_eval_specs_register_coverage_and_window_note() -> None:
    injury = load_eval_spec("minutes_injury_overlay_ab")
    assert injury.gates.coverage.metric == "injury_overlay_mapped"
    assert injury.gates.min_coverage == 0.01
    assert "thin/S1" in injury.window.note or "thin" in injury.window.note.lower()

    xi = load_eval_spec("predicted_xi_ab")
    assert xi.gates.coverage.metric == "predicted_lineups_pit"
    assert xi.gates.min_coverage == 0.01
    assert "keep" in xi.window.note.lower()

    avail = load_eval_spec("fpl_availability_gate_ab")
    assert avail.gates.coverage.metric == "fpl_api_status_share"
    assert avail.gates.min_coverage == 0.01

    combine = load_eval_spec("forecast_combine_equal_ab")
    assert combine.gates.coverage.metric == "forecast_combine_member_join"
    assert combine.gates.min_coverage == 0.80

    probs_combine = load_eval_spec("lgbm_probs_combine_ab")
    assert probs_combine.treatment == "forecast_combine_probs_v1"
    assert probs_combine.control == "dc_layered_v4"
    assert probs_combine.gates.coverage.metric == "forecast_combine_member_join"
    assert probs_combine.gates.min_coverage == 0.80
    assert "spearman" in (probs_combine.gates.keep_if.prediction_any or [])
    assert probs_combine.report.store_doc == "lgbm-probs-combine.md"

    defcon = load_eval_spec("defcon_rate_ab")
    assert defcon.gates.coverage.metric == "dc_season_eligible"
    assert defcon.gates.min_coverage == 1.0
    assert defcon.window.note
    assert defcon.window.from_gw == 1
    assert defcon.window.to_gw == 19
    assert defcon.suites.decisions is False


def test_minutes_p1p6_keep_eval_loads() -> None:
    """FDB-257 package keep EvalSpec: opt-out, fpl_api coverage, no squad claim yet."""
    spec = load_eval_spec("minutes_p1p6_keep_ab")
    assert spec.kind == "config"
    assert spec.change.lever == "hyperparam"
    assert spec.change.orientation == "opt_out"
    assert spec.change.slot == "minutes"
    assert spec.change.allow_multi_lever is True
    assert spec.control == "dc_layered_minutes_pre_p1p6_v1"
    assert spec.treatment == "dc_layered_v4"
    assert spec.claim == "forecasts"
    assert spec.suites.decisions is False
    assert spec.gates.coverage.metric == "fpl_api_status_share"
    assert spec.gates.min_coverage == 0.01
    assert spec.window.from_gw == 1
    assert spec.window.to_gw == 20
    assert "2024-25" in spec.window.seasons
    assert "confirmation" in spec.window.note.lower()
    plan = plan_promote_eval(spec, dry_run=True)
    components = [cmd for cmd in plan.commands if "eval-components" in cmd]
    assert len(components) == 1
    assert "empirical_minutes_v2_pre_p1p6" in components[0]
    assert "empirical_minutes_v2" in components[0]


def test_preflight_injury_overlay_fails_when_empty(tmp_path: Path) -> None:
    _seed_overlay_base(tmp_path)
    # Empty injuries table: schema present, no rows → tables check fails.
    store = DataStore(tmp_path)
    store.write(
        "player_injuries",
        _utc(
            pl.DataFrame(
                {
                    "season": pl.Series([], dtype=pl.Utf8()),
                    "player_key": pl.Series([], dtype=pl.Int64()),
                    "api_player_id": pl.Series([], dtype=pl.Int64()),
                    "team_key": pl.Series([], dtype=pl.Int64()),
                    "api_team_id": pl.Series([], dtype=pl.Int64()),
                    "api_fixture_id": pl.Series([], dtype=pl.Int64()),
                    "fixture_id": pl.Series([], dtype=pl.Int64()),
                    "injury_type": pl.Series([], dtype=pl.Utf8()),
                    "reason": pl.Series([], dtype=pl.Utf8()),
                    "category": pl.Series([], dtype=pl.Utf8()),
                    "available_at": pl.Series([], dtype=pl.Datetime("us", "UTC")),
                    "source": pl.Series([], dtype=pl.Utf8()),
                }
            )
        ),
    )
    payload = _base_payload(
        name="injury_cov",
        kind="config",
        question="Does injury overlay fire?",
        change={
            "lever": "hyperparam",
            "orientation": "opt_in",
            "slot": "minutes",
            "control_kind": "empirical_minutes_v2",
            "treatment_kind": "empirical_minutes_v2",
            "tables": ["player_injuries"],
            "diff": ["use_injury_overlay: false → true"],
        },
        control="dc_layered_v4",
        treatment="dc_layered_injuries_v1",
        window={
            "seasons": ["2025-26"],
            "from_gw": 1,
            "to_gw": 1,
            "modes": ["walk_forward"],
            "note": "GW1–10 thin/S1 smoke only — never a keep window.",
        },
        suites={"prediction": True, "components": True},
        gates={
            "min_coverage": 0.01,
            "coverage": {"metric": "injury_overlay_mapped", "read": "strict_pit"},
            "drop_if": {"coverage_below_min": True, "mean_ppg_gate_fail": True},
        },
    )
    _write_eval(tmp_path, "injury_cov", payload)
    spec = load_eval_spec("injury_cov", directory=tmp_path)
    plan = plan_promote_eval(spec, dry_run=False)
    assert plan.window_note
    with patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False):
        # Empty table fails change.tables check before coverage.
        result = preflight_promote_eval(spec, plan, data_dir=tmp_path)
    assert not result.ok
    assert any("empty" in f for f in result.failures)


def test_preflight_injury_overlay_passes_with_mapped_row(tmp_path: Path) -> None:
    _seed_overlay_base(tmp_path)
    store = DataStore(tmp_path)
    store.write(
        "player_injuries",
        _utc(
            pl.DataFrame(
                {
                    "season": ["2025-26"],
                    "player_key": [100],
                    "api_player_id": [9],
                    "team_key": [3],
                    "api_team_id": [33],
                    "api_fixture_id": [99],
                    "fixture_id": [1],
                    "injury_type": ["Knock"],
                    "reason": ["Knock"],
                    "category": ["doubtful"],
                    "available_at": [_START],
                    "source": ["api_football"],
                }
            )
        ),
    )
    payload = _base_payload(
        name="injury_ok",
        kind="config",
        question="Does injury overlay fire?",
        change={
            "lever": "hyperparam",
            "orientation": "opt_in",
            "slot": "minutes",
            "control_kind": "empirical_minutes_v2",
            "treatment_kind": "empirical_minutes_v2",
            "tables": ["player_injuries"],
            "diff": ["use_injury_overlay: false → true"],
        },
        control="dc_layered_v4",
        treatment="dc_layered_injuries_v1",
        window={
            "seasons": ["2025-26"],
            "from_gw": 1,
            "to_gw": 1,
            "modes": ["walk_forward"],
        },
        suites={"prediction": True, "components": False},
        gates={
            "min_coverage": 0.01,
            "coverage": {"metric": "injury_overlay_mapped", "read": "strict_pit"},
            "drop_if": {"coverage_below_min": True, "mean_ppg_gate_fail": True},
        },
    )
    _write_eval(tmp_path, "injury_ok", payload)
    spec = load_eval_spec("injury_ok", directory=tmp_path)
    plan = plan_promote_eval(spec, dry_run=False)
    with patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False):
        result = preflight_promote_eval(spec, plan, data_dir=tmp_path)
    assert result.ok
    assert result.coverage is not None
    assert result.coverage.metric == "injury_overlay_mapped"
    assert result.coverage.passed
    assert result.coverage.rate >= 0.01


def test_preflight_predicted_lineups_fails_without_pit_rows(tmp_path: Path) -> None:
    _seed_overlay_base(tmp_path)
    store = DataStore(tmp_path)
    # Lineups only after deadline → rate 0.
    store.write(
        "predicted_lineups",
        _utc(
            pl.DataFrame(
                {
                    "season": ["2025-26"],
                    "team_key": [3],
                    "fixture_id": [1],
                    "player_key": [100],
                    "element_id": [1],
                    "role": ["start"],
                    "confidence": [0.9],
                    "available_at": [_DEADLINE_GW1 + timedelta(hours=1)],
                    "source": ["touchline_wire"],
                }
            )
        ),
    )
    payload = _base_payload(
        name="xi_miss",
        kind="config",
        question="Does predicted XI fire?",
        change={
            "lever": "hyperparam",
            "orientation": "opt_in",
            "slot": "minutes",
            "control_kind": "empirical_minutes_v2",
            "treatment_kind": "empirical_minutes_v2",
            "tables": ["predicted_lineups"],
            # Fork also sets window/strength; allow for coverage-unit focus.
            "allow_multi_lever": True,
            "diff": ["predicted_xi: false → true"],
        },
        control="dc_layered_v4",
        treatment="dc_layered_predicted_xi_v1",
        window={
            "seasons": ["2025-26"],
            "from_gw": 1,
            "to_gw": 1,
            "modes": ["walk_forward"],
        },
        suites={"prediction": True, "components": False},
        gates={
            "min_coverage": 0.01,
            "coverage": {"metric": "predicted_lineups_pit", "read": "strict_pit"},
            "drop_if": {"coverage_below_min": True, "mean_ppg_gate_fail": True},
        },
    )
    _write_eval(tmp_path, "xi_miss", payload)
    spec = load_eval_spec("xi_miss", directory=tmp_path)
    plan = plan_promote_eval(spec, dry_run=False)
    with patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False):
        result = preflight_promote_eval(spec, plan, data_dir=tmp_path)
    assert result.ok
    assert result.coverage is not None
    assert result.coverage.metric == "predicted_lineups_pit"
    assert not result.coverage.passed
    assert result.coverage.rate == 0.0


def test_preflight_fpl_api_status_fails_on_vaastav(tmp_path: Path) -> None:
    _seed_overlay_base(tmp_path, status_source="vaastav")
    payload = _base_payload(
        name="avail_vaastav",
        kind="config",
        question="Does the FPL gate fire on vaastav?",
        change={
            "lever": "hyperparam",
            "orientation": "opt_out",
            "slot": "minutes",
            "control_kind": "empirical_minutes_v2",
            "treatment_kind": "empirical_minutes_v2",
            "tables": ["player_status"],
            "diff": ["use_availability_gate: false → true"],
        },
        control="dc_layered_noavail_v1",
        treatment="dc_layered_v4",
        window={
            "seasons": ["2025-26"],
            "from_gw": 1,
            "to_gw": 1,
            "modes": ["walk_forward"],
        },
        suites={"prediction": True, "components": False},
        gates={
            "min_coverage": 0.01,
            "coverage": {"metric": "fpl_api_status_share", "read": "strict_pit"},
            "drop_if": {"coverage_below_min": True, "mean_ppg_gate_fail": True},
        },
    )
    _write_eval(tmp_path, "avail_vaastav", payload)
    spec = load_eval_spec("avail_vaastav", directory=tmp_path)
    plan = plan_promote_eval(spec, dry_run=False)
    with patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False):
        result = preflight_promote_eval(spec, plan, data_dir=tmp_path)
    assert result.ok
    assert result.coverage is not None
    assert result.coverage.metric == "fpl_api_status_share"
    assert not result.coverage.passed
    assert result.coverage.rate == 0.0


def test_preflight_forecast_combine_member_join_universe(tmp_path: Path) -> None:
    _seed_overlay_base(tmp_path)
    payload = _base_payload(
        name="combine_cov",
        kind="model",
        question="Does equal blend beat DC when both members cover the pool?",
        change={
            "lever": "model_fork",
            "orientation": "opt_in",
            "control_kind": "dc_layered",
            "treatment_kind": "forecast_combine",
            "diff": ["kind: dc_layered → forecast_combine"],
        },
        control="dc_layered_v4",
        treatment="forecast_combine_equal_v1",
        window={
            "seasons": ["2025-26"],
            "from_gw": 1,
            "to_gw": 1,
            "modes": ["walk_forward"],
            "note": "GW1–10 thin/S1 smoke only — never a keep window.",
        },
        suites={"prediction": True, "components": False},
        gates={
            "min_coverage": 0.80,
            "coverage": {
                "metric": "forecast_combine_member_join",
                "read": "strict_pit",
            },
            "drop_if": {"coverage_below_min": True, "mean_ppg_gate_fail": True},
        },
    )
    _write_eval(tmp_path, "combine_cov", payload)
    spec = load_eval_spec("combine_cov", directory=tmp_path)
    plan = plan_promote_eval(spec, dry_run=False)
    assert "thin" in plan.window_note.lower()
    with patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False):
        result = preflight_promote_eval(spec, plan, data_dir=tmp_path)
    assert result.ok
    assert result.coverage is not None
    assert result.coverage.metric == "forecast_combine_member_join"
    assert result.coverage.passed
    assert result.coverage.rate >= 0.80
    assert "dc_layered_v4" in result.coverage.extras["members"]
    assert "lgbm_direct_v1" in result.coverage.extras["members"]
    assert result.coverage.extras["member_missing"] == []


def test_preflight_forecast_combine_fails_when_member_predictions_miss(
    tmp_path: Path,
) -> None:
    _seed_overlay_base(tmp_path)
    from fplt.models.spec import load_model_specs

    specs = load_model_specs()
    dc = specs["dc_layered_v4"]
    lgbm = specs["lgbm_direct_v1"]
    # Archive DC predictions for the scored pool; LGBM archive exists but
    # covers none of the scored keys → member missing on the join.
    pred_dir = tmp_path / "predictions" / dc.model_id / "as_of=2025-26_gw01"
    pred_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "model_id": [dc.model_id, dc.model_id],
            "as_of_season": ["2025-26", "2025-26"],
            "as_of_gw": [1, 1],
            "target_season": ["2025-26", "2025-26"],
            "target_gw": [1, 1],
            "player_key": [100, 200],
            "fixture_id": [1, 1],
            "xpts": [4.0, 5.0],
            "xpts_sd": [1.0, 1.0],
            "p_played": [0.9, 0.9],
            "p_60": [0.8, 0.8],
            "p_goal": [0.2, 0.3],
            "p_assist": [0.1, 0.1],
            "p_cs": [0.0, 0.0],
            "p_haul": [0.1, 0.2],
        }
    ).write_parquet(pred_dir / "part.parquet")
    lgbm_dir = tmp_path / "predictions" / lgbm.model_id / "as_of=2025-26_gw01"
    lgbm_dir.mkdir(parents=True)
    pl.DataFrame(
        {
            "model_id": [lgbm.model_id],
            "as_of_season": ["2025-26"],
            "as_of_gw": [1],
            "target_season": ["2025-26"],
            "target_gw": [1],
            "player_key": [999],
            "fixture_id": [99],
            "xpts": [1.0],
            "xpts_sd": [0.5],
            "p_played": [None],
            "p_60": [None],
            "p_goal": [None],
            "p_assist": [None],
            "p_cs": [None],
            "p_haul": [None],
        }
    ).write_parquet(lgbm_dir / "part.parquet")

    payload = _base_payload(
        name="combine_miss",
        kind="model",
        question="Does a missing member fail coverage?",
        change={
            "lever": "model_fork",
            "orientation": "opt_in",
            "control_kind": "dc_layered",
            "treatment_kind": "forecast_combine",
            "diff": ["kind: dc_layered → forecast_combine"],
        },
        control="dc_layered_v4",
        treatment="forecast_combine_equal_v1",
        window={
            "seasons": ["2025-26"],
            "from_gw": 1,
            "to_gw": 1,
            "modes": ["walk_forward"],
        },
        suites={"prediction": True, "components": False},
        gates={
            "min_coverage": 0.80,
            "coverage": {
                "metric": "forecast_combine_member_join",
                "read": "strict_pit",
            },
            "drop_if": {"coverage_below_min": True, "mean_ppg_gate_fail": True},
        },
    )
    _write_eval(tmp_path, "combine_miss", payload)
    spec = load_eval_spec("combine_miss", directory=tmp_path)
    plan = plan_promote_eval(spec, dry_run=False)
    with patch("fplt.backtest.promote_eval.git_is_dirty", return_value=False):
        result = preflight_promote_eval(spec, plan, data_dir=tmp_path)
    assert result.coverage is not None
    assert result.coverage.metric == "forecast_combine_member_join"
    assert not result.coverage.passed
    assert result.coverage.rate == 0.0
    assert "lgbm_direct_v1" in result.coverage.extras["member_missing"]


# ---------------------------------------------------------------------------
# Pure helpers / error branches (coverage for promote_eval.py)
# ---------------------------------------------------------------------------


def test_detect_points_claim_phrase_edge_cases() -> None:
    assert detect_points_claim_phrase("") is None
    assert detect_points_claim_phrase("points only") is None
    assert detect_points_claim_phrase("Season Points rise") == "season points"
    assert detect_points_claim_phrase("SQUAD POINT total") == "squad points"
    assert detect_points_claim_phrase("mean-ppg improvement") == "mean ppg"
    assert detect_points_claim_phrase("mean_ppg gate") == "mean ppg"
    assert detect_points_claim_phrase("realised point delta") == "realised points"
    assert detect_points_claim_phrase("realiseed points") is None  # typo / US spelling
    # First matching pattern wins when several could match.
    assert (
        detect_points_claim_phrase("season points and squad points") == "season points"
    )


def test_decisions_claim_failures_enabled_override() -> None:
    # Valid squad_points spec (decisions on in YAML); override disables gate.
    spec = EvalSpec.model_validate(
        _base_payload(
            kind="model",
            claim="squad_points",
            question="Does the fork earn more season points?",
            change={
                "lever": "model_fork",
                "orientation": "opt_in",
                "slot": None,
                "control_kind": "rolling_mean",
                "treatment_kind": "lgbm_direct",
                "diff": ["kind: rolling_mean → lgbm_direct"],
            },
            control="rolling_mean_v1",
            treatment="lgbm_direct_v1",
            suites={
                "prediction": True,
                "components": False,
                "decisions": True,
                "decisions_horizon": 3,
            },
        )
    )
    assert decisions_claim_failures(spec) == []
    fails = decisions_claim_failures(spec, decisions_enabled=False)
    assert len(fails) == 1
    assert "claim=squad_points requires suites.decisions" in fails[0]

    forecast = EvalSpec.model_validate(
        _base_payload(
            question="Does RMSE improve?",
            description="season points mentioned in description",
            suites={"prediction": True, "components": False, "decisions": True},
        )
    )
    # decisions on → prose claim is fine even with forecasts claim.
    assert decisions_claim_failures(forecast) == []
    # Override off → prose points claim fails under forecasts.
    prose_fails = decisions_claim_failures(forecast, decisions_enabled=False)
    assert len(prose_fails) == 1
    assert "prose claims" in prose_fails[0]
    assert "season points" in prose_fails[0]


def test_decisions_claim_failures_forecasts_clean() -> None:
    spec = EvalSpec.model_validate(
        _base_payload(
            question="Does RMSE improve on the pool?",
            suites={"prediction": True, "components": False, "decisions": False},
        )
    )
    assert decisions_claim_failures(spec) == []
    assert decisions_claim_failures(spec, decisions_enabled=False) == []


def test_git_is_dirty_true_false_and_error(tmp_path: Path) -> None:
    with patch("fplt.backtest.promote_eval.subprocess.run") as run:
        run.return_value = type("R", (), {"stdout": " M file.py\n"})()
        assert git_is_dirty(cwd=tmp_path) is True
        run.return_value = type("R", (), {"stdout": "  \n"})()
        assert git_is_dirty(cwd=tmp_path) is False
        run.side_effect = OSError("no git")
        assert git_is_dirty(cwd=tmp_path) is False
        run.side_effect = subprocess.CalledProcessError(1, ["git"])
        assert git_is_dirty(cwd=tmp_path) is False


def test_effective_param_and_is_effectively_on() -> None:
    arm = ModelSpec(
        name="a",
        kind="dc_layered",
        params={"n_sims": 1000, "market_prior_strength": 0.5},
        components={
            "involvement": ModelSpec(
                name="inv",
                kind="involvement_xg",
                params={"use_penalties_order": False},
            )
        },
    )
    assert effective_param(arm, None, "n_sims") == 1000
    assert effective_param(arm, None, "market_prior_strength") == 0.5
    # Absent bool / float keys fall back to code defaults.
    assert effective_param(arm, None, "use_injury_overlay") is False
    assert effective_param(arm, None, "form_strength") == 0.0
    assert effective_param(arm, None, "unknown_key") is None
    # Slot params, missing slot → empty → defaults.
    assert effective_param(arm, "involvement", "use_penalties_order") is False
    assert effective_param(arm, "involvement", "use_availability_gate") is True
    assert effective_param(arm, "missing_slot", "use_penalties_order") is True

    assert is_effectively_on("use_penalties_order", True) is True
    assert is_effectively_on("use_penalties_order", False) is False
    assert is_effectively_on("market_prior_strength", 0.5) is True
    assert is_effectively_on("market_prior_strength", 0.0) is False
    assert is_effectively_on("custom_flag", "yes") is True
    assert is_effectively_on("custom_flag", "") is False


def test_dict_diff_paths_and_component_param_diffs() -> None:
    assert _dict_diff_paths(1, "x") == ["<root>"]
    assert _dict_diff_paths({"a": 1}, {"a": 1}) == []
    assert _dict_diff_paths({"a": 1}, {"a": 2}) == ["a"]
    assert _dict_diff_paths({"a": 1}, {"b": 1}) == ["a", "b"]
    # Nested name keys are ignored.
    assert _dict_diff_paths({"name": "x", "v": 1}, {"name": "y", "v": 1}) == []
    assert _dict_diff_paths({"nested": {"name": "a"}}, {"nested": {"name": "b"}}) == []
    assert _dict_diff_paths([1, 2], [1, 3]) == ["<root>"]
    assert _dict_diff_paths([1, 2], [1, 2]) == []
    assert _dict_diff_paths({"x": [1]}, {"x": [2]}) == ["x"]
    assert _dict_diff_paths(True, False, prefix="flag") == ["flag"]

    control = ModelSpec(
        name="c",
        kind="dc_layered",
        params={"n_sims": 100},
        components={
            "score": ModelSpec(name="s1", kind="dixon_coles", params={"rho": 0.1})
        },
    )
    treatment = ModelSpec(
        name="t",
        kind="dc_layered",
        params={"n_sims": 200},
        components={
            "score": ModelSpec(name="s2", kind="dixon_coles", params={"rho": 0.2})
        },
    )
    paths = _component_param_diffs(control, treatment)
    assert "params.n_sims" in paths
    assert "components.score.params.rho" in paths
    # Top-level / nested name must not appear.
    assert not any(p == "name" or p.endswith(".name") for p in paths)


def test_check_kinds_and_registered() -> None:
    from fplt.backtest.promote_eval import _ensure_model_registry

    _ensure_model_registry()
    spec = EvalSpec.model_validate(
        _base_payload(
            change={
                "lever": "component_swap",
                "orientation": "opt_in",
                "slot": "minutes",
                "control_kind": "empirical_minutes_v2",
                "treatment_kind": "lgbm_minutes",
                "allow_multi_lever": True,
                "diff": ["components.minutes.kind"],
            }
        )
    )
    control = ModelSpec(
        name="c",
        kind="dc_layered",
        components={
            "minutes": ModelSpec(name="m", kind="empirical_minutes_v2"),
        },
    )
    treatment = ModelSpec(
        name="t",
        kind="dc_layered",
        components={
            "minutes": ModelSpec(name="m", kind="lgbm_minutes"),
        },
    )
    assert _check_kinds(spec, control, treatment) == []

    # Missing slot / kind mismatch.
    bare = ModelSpec(name="bare", kind="dc_layered")
    fails = _check_kinds(spec, bare, treatment)
    assert any("control missing components.minutes" in f for f in fails)
    wrong = ModelSpec(
        name="w",
        kind="dc_layered",
        components={"minutes": ModelSpec(name="m", kind="wrong_kind")},
    )
    fails2 = _check_kinds(spec, wrong, treatment)
    assert any("control_kind" in f for f in fails2)

    # model_fork compares top-level kind.
    fork = EvalSpec.model_validate(
        _base_payload(
            kind="model",
            change={
                "lever": "model_fork",
                "orientation": "opt_in",
                "slot": None,
                "control_kind": "rolling_mean",
                "treatment_kind": "lgbm_direct",
                "diff": ["kind"],
            },
            control="rolling_mean_v1",
            treatment="lgbm_direct_v1",
        )
    )
    assert (
        _check_kinds(
            fork,
            ModelSpec(name="c", kind="rolling_mean"),
            ModelSpec(name="t", kind="lgbm_direct"),
        )
        == []
    )
    fork_fails = _check_kinds(
        fork,
        ModelSpec(name="c", kind="other"),
        ModelSpec(name="t", kind="lgbm_direct"),
    )
    assert any("control_kind" in f for f in fork_fails)

    with (
        patch(
            "fplt.backtest.promote_eval.registered_kinds",
            return_value=["dc_layered"],
        ),
        patch(
            "fplt.backtest.promote_eval.registered_components",
            return_value=["empirical_minutes_v2"],
        ),
    ):
        reg_ok = _check_registered(control, control)
        assert reg_ok == []
        bad = ModelSpec(
            name="bad",
            kind="unknown_kind",
            components={
                "minutes": ModelSpec(name="m", kind="unknown_comp"),
            },
        )
        reg_fail = _check_registered(bad, bad)
        assert any("not in registered_kinds" in f for f in reg_fail)
        assert any("not registered" in f for f in reg_fail)


def test_n_sims_and_production_arm() -> None:
    assert _n_sims(None) is None
    assert _n_sims(ModelSpec(name="a", kind="x")) is None
    assert _n_sims(ModelSpec(name="a", kind="x", params={"n_sims": 5000})) == 5000

    opt_in = EvalSpec.model_validate(_base_payload())
    assert _is_production_arm(opt_in, "control") is True
    assert _is_production_arm(opt_in, "treatment") is False

    opt_out = EvalSpec.model_validate(
        _base_payload(
            change={
                "lever": "component_swap",
                "orientation": "opt_out",
                "slot": "minutes",
                "control_kind": "lgbm_minutes",
                "treatment_kind": "empirical_minutes_v2",
                "allow_multi_lever": True,
                "diff": ["components.minutes.kind"],
            }
        )
    )
    assert _is_production_arm(opt_out, "treatment") is True
    assert _is_production_arm(opt_out, "control") is False

    no_orient = EvalSpec.model_validate(
        _base_payload(
            change={
                "lever": "component_swap",
                "orientation": None,
                "slot": "minutes",
                "control_kind": "empirical_minutes_v2",
                "treatment_kind": "lgbm_minutes",
                "allow_multi_lever": True,
                "diff": ["components.minutes.kind"],
            }
        )
    )
    assert _is_production_arm(no_orient, "control") is False

    equiv = load_eval_spec("example_refactor_no_behaviour")
    assert _is_production_arm(equiv, "control") is True
    assert _is_production_arm(equiv, "treatment") is False


def test_coverage_pass_and_lever_position_specific() -> None:
    assert _coverage_pass(0.5, None) is True
    assert _coverage_pass(0.5, 0.5) is True
    assert _coverage_pass(0.499999999999, 0.5) is True  # float epsilon
    assert _coverage_pass(0.4, 0.5) is False

    pens = load_eval_spec("pens_order_involvement_ab")
    assert _lever_is_position_specific(pens) is False
    synthetic = EvalSpec.model_validate(
        _base_payload(
            change={
                "lever": "component_swap",
                "orientation": "opt_in",
                "slot": "other",
                "allow_multi_lever": True,
                "diff": [
                    "components.other.params.use_defensive_contribution: false → true"
                ],
            }
        )
    )
    assert _lever_is_position_specific(synthetic) is True


def test_format_interval_lines() -> None:
    assert _format_interval_lines({}) == [
        "- HAC / bootstrap intervals: unavailable (no paired gameweeks)"
    ]
    assert _format_interval_lines({"intervals": []}) == [
        "- HAC / bootstrap intervals: unavailable (no paired gameweeks)"
    ]
    assert _format_interval_lines({"intervals": "bad"}) == [
        "- HAC / bootstrap intervals: unavailable (no paired gameweeks)"
    ]
    lines = _format_interval_lines(
        {
            "intervals": [
                "skip-me",
                {
                    "metric": "rmse",
                    "position": None,
                    "mean": 0.1,
                    "hac_se": 0.01,
                    "hac_pvalue": 0.2,
                    "hac_ci": [-0.1, 0.3],
                    "bootstrap_ci": [-0.05, 0.25],
                },
                {
                    "metric": "mae",
                    "position": "MID",
                    "mean": -0.2,
                    "hac_se": 0.03,
                    "hac_pvalue": 0.4,
                    "hac_ci": [-0.4, 0.0],
                    "bootstrap_ci": [-0.3, 0.1],
                },
            ]
        }
    )
    assert len(lines) == 2
    assert lines[0].startswith("- rmse:")
    assert "HAC_CI=" in lines[0]
    assert lines[1].startswith("- mae[MID]:")


def test_format_plan_and_plan_to_json_branches() -> None:
    plan = PromoteEvalPlan(
        eval_name="demo",
        kind="component",
        lever="component_swap",
        control="c",
        treatment="t",
        arms=["dose_a"],
        seasons=["2024-25"],
        from_gw=1,
        to_gw=5,
        seasons_back=4,
        modes=["walk_forward"],
        suites=EvalSpec.model_validate(_base_payload()).suites,
        commands=["fplt compare c t"],
        diff=["slot.kind"],
        notes="note text",
        window_note="thin window",
        warnings=["warn1"],
        dry_run=True,
    )
    text = format_plan(plan)
    assert "arms (dose): dose_a" in text
    assert "diff:" in text
    assert "notes: note text" in text
    assert "window.note: thin window" in text
    assert "warnings:" in text
    assert "warn1" in text
    assert "(dry-run: no fits executed)" in text
    payload = json.loads(plan_to_json(plan))
    assert payload["eval_name"] == "demo"
    assert payload["arms"] == ["dose_a"]
    assert payload["dry_run"] is True

    bare = PromoteEvalPlan(
        eval_name="bare",
        kind="config",
        lever="hyperparam",
        control="c",
        treatment="t",
        seasons=["2024-25"],
        from_gw=1,
        to_gw=2,
        seasons_back=2,
        modes=["walk_forward"],
        suites=EvalSpec.model_validate(_base_payload()).suites,
        commands=["cmd"],
        diff=[],
        notes="",
        dry_run=False,
    )
    bare_text = format_plan(bare)
    assert "arms (dose)" not in bare_text
    assert "notes:" not in bare_text
    assert "window.note:" not in bare_text
    assert "warnings:" not in bare_text
    assert "dry-run" not in bare_text


def test_check_one_lever_multi_diff_branches() -> None:
    control = ModelSpec(
        name="c",
        kind="dc_layered",
        components={
            "minutes": ModelSpec(
                name="m", kind="empirical_minutes_v2", params={"a": 1}
            ),
            "score": ModelSpec(name="s", kind="dixon_coles", params={"rho": 0.1}),
        },
    )
    treatment = ModelSpec(
        name="t",
        kind="dc_layered",
        components={
            "minutes": ModelSpec(name="m", kind="lgbm_minutes", params={"a": 2}),
            "score": ModelSpec(name="s", kind="dixon_coles", params={"rho": 0.2}),
        },
    )
    swap = EvalSpec.model_validate(
        _base_payload(
            change={
                "lever": "component_swap",
                "orientation": "opt_in",
                "slot": "minutes",
                "control_kind": "empirical_minutes_v2",
                "treatment_kind": "lgbm_minutes",
                "diff": ["components.minutes.kind"],
            }
        )
    )
    # Without allow_multi_lever, diffs outside minutes fail.
    fails = _check_one_lever(swap, control, treatment)
    assert any("multi-lever" in f for f in fails)

    # model_fork / allow_multi_lever / equivalence skip.
    fork = EvalSpec.model_validate(
        _base_payload(
            kind="model",
            change={
                "lever": "model_fork",
                "orientation": "opt_in",
                "slot": None,
                "diff": ["kind"],
            },
            control="rolling_mean_v1",
            treatment="lgbm_direct_v1",
        )
    )
    assert _check_one_lever(fork, control, treatment) == []
    assert (
        _check_one_lever(
            load_eval_spec("example_refactor_no_behaviour"), control, treatment
        )
        == []
    )

    hyper = EvalSpec.model_validate(
        _base_payload(
            kind="config",
            change={
                "lever": "hyperparam",
                "orientation": "opt_in",
                "slot": "score",
                "control_kind": "dixon_coles",
                "treatment_kind": "dixon_coles",
                "diff": ["components.score.params.rho"],
            },
            control="dc_layered_v2",
            treatment="dc_layered_odds_v1",
        )
    )
    # Diffs outside score + kind change under hyperparam.
    hyper_ctrl = ModelSpec(
        name="c",
        kind="dc_layered",
        components={
            "score": ModelSpec(
                name="s", kind="dixon_coles", params={"rho": 0.1, "xi": 1}
            ),
            "minutes": ModelSpec(name="m", kind="empirical_minutes_v2"),
        },
    )
    hyper_trt = ModelSpec(
        name="t",
        kind="dc_layered",
        components={
            "score": ModelSpec(
                name="s", kind="odds_poisson", params={"rho": 0.2, "xi": 2}
            ),
            "minutes": ModelSpec(name="m", kind="lgbm_minutes"),
        },
    )
    hyper_fails = _check_one_lever(hyper, hyper_ctrl, hyper_trt)
    assert any("outside slot" in f for f in hyper_fails)
    assert any("more than one param family" in f for f in hyper_fails)
    assert any("hyperparam lever changed" in f for f in hyper_fails)

    # Top-level slot=None hyperparam with multiple diffs.
    top = EvalSpec.model_validate(
        _base_payload(
            kind="config",
            change={
                "lever": "hyperparam",
                "orientation": "opt_in",
                "slot": None,
                "diff": ["params.n_sims"],
            },
            control="dc_layered_v2",
            treatment="dc_layered_odds_v1",
        )
    )
    top_c = ModelSpec(name="c", kind="dc_layered", params={"n_sims": 1, "seed_x": 1})
    top_t = ModelSpec(name="t", kind="dc_layered", params={"n_sims": 2, "seed_x": 2})
    top_fails = _check_one_lever(top, top_c, top_t)
    assert any("multiple diffs" in f for f in top_fails)


def test_git_show_file_success_and_failure(tmp_path: Path) -> None:
    with patch("fplt.backtest.promote_eval.subprocess.run") as run:
        run.return_value = type("R", (), {"stdout": "file-body"})()
        assert _git_show_file("HEAD", "uv.lock", cwd=tmp_path) == "file-body"
        run.side_effect = subprocess.CalledProcessError(128, ["git"])
        assert _git_show_file("HEAD", "missing", cwd=tmp_path) is None


def test_check_equivalence_preflight_helpers() -> None:
    control = ModelSpec(name="c", kind="dc_layered")
    treatment = ModelSpec(name="t", kind="dc_layered")

    # Lightweight stubs for early-return / mocked branches (frozen EvalSpec).
    class _Change:
        control_ref: str | None
        treatment_ref: str | None
        packages: list[str]
        allow_multi_lever: bool

        def __init__(
            self,
            *,
            control_ref: str | None,
            treatment_ref: str | None,
            packages: list[str] | None = None,
            allow_multi_lever: bool = False,
        ) -> None:
            self.control_ref = control_ref
            self.treatment_ref = treatment_ref
            self.packages = packages or []
            self.allow_multi_lever = allow_multi_lever

    class _Stub:
        kind: str
        change: _Change

        def __init__(self, kind: str, change: _Change) -> None:
            self.kind = kind
            self.change = change

    stub_missing = _Stub("refactor", _Change(control_ref=None, treatment_ref="b"))
    fails, warns, meta = _check_equivalence_preflight(
        stub_missing,
        control,
        treatment,  # type: ignore[arg-type]
    )
    assert fails and "require change.control_ref" in fails[0]
    assert warns == []
    assert meta == {}

    stub_same = _Stub("refactor", _Change(control_ref="main", treatment_ref="main"))
    with (
        patch(
            "fplt.backtest.promote_eval.resolve_git_sha",
            side_effect=["aaa", "aaa"],
        ),
        patch(
            "fplt.backtest.promote_eval._git_show_file",
            return_value='[[package]]\nname = "numpy"\nversion = "1.0.0"\n',
        ),
        patch(
            "fplt.backtest.promote_eval.resolved_spec_sha256",
            side_effect=["hash1", "hash1"],
        ),
    ):
        fails2, _, _ = _check_equivalence_preflight(
            stub_same,
            control,
            treatment,  # type: ignore[arg-type]
        )
    assert any("identical" in f for f in fails2)

    stub_env = _Stub(
        "environment",
        _Change(control_ref="a", treatment_ref="b", packages=["numpy"]),
    )
    with patch(
        "fplt.backtest.promote_eval.resolve_git_sha",
        side_effect=subprocess.CalledProcessError(128, ["git"]),
    ):
        fails3, _, _ = _check_equivalence_preflight(
            stub_env,
            control,
            treatment,  # type: ignore[arg-type]
        )
    assert any("could not resolve git refs" in f for f in fails3)

    with (
        patch(
            "fplt.backtest.promote_eval.resolve_git_sha",
            side_effect=["aaa", "bbb"],
        ),
        patch(
            "fplt.backtest.promote_eval._git_show_file",
            return_value=None,
        ),
        patch(
            "fplt.backtest.promote_eval.resolved_spec_sha256",
            side_effect=["h1", "h1"],
        ),
    ):
        fails4, _, _ = _check_equivalence_preflight(
            stub_env,
            control,
            treatment,  # type: ignore[arg-type]
        )
    assert any("uv.lock missing" in f for f in fails4)

    # Specs differ + lock differs without allow_multi_lever.
    lock_a = '[[package]]\nname = "numpy"\nversion = "1.0.0"\n'
    lock_b = '[[package]]\nname = "numpy"\nversion = "2.0.0"\n'
    with (
        patch(
            "fplt.backtest.promote_eval.resolve_git_sha",
            side_effect=["aaa", "bbb"],
        ),
        patch(
            "fplt.backtest.promote_eval._git_show_file",
            side_effect=[lock_a, lock_b],
        ),
        patch(
            "fplt.backtest.promote_eval.resolved_spec_sha256",
            side_effect=["h1", "h2"],
        ),
        patch(
            "fplt.backtest.promote_eval.diff_lockfiles",
            return_value=LockfileDiff(
                control_sha256="c",
                treatment_sha256="t",
                direct_changed=("numpy: 1.0.0 → 2.0.0",),
                transitive_changed=("foo: 1 → 2",),
                missing_packages=(),
                identical=False,
            ),
        ),
    ):
        fails5, warns5, meta5 = _check_equivalence_preflight(
            stub_env,
            control,
            treatment,  # type: ignore[arg-type]
        )
    assert any("two levers" in f for f in fails5)
    assert any("resolved spec hashes differ" in f for f in fails5)
    assert any("transitive" in w for w in warns5)
    assert meta5["control_ref_sha"] == "aaa"
    assert meta5["treatment_lock_sha256"] == "t"

    # refactor: specs differ + lock identical path
    stub_ref = _Stub("refactor", _Change(control_ref="a", treatment_ref="b"))
    with (
        patch(
            "fplt.backtest.promote_eval.resolve_git_sha",
            side_effect=["aaa", "bbb"],
        ),
        patch(
            "fplt.backtest.promote_eval._git_show_file",
            return_value=lock_a,
        ),
        patch(
            "fplt.backtest.promote_eval.resolved_spec_sha256",
            side_effect=["h1", "h2"],
        ),
        patch(
            "fplt.backtest.promote_eval.diff_lockfiles",
            return_value=LockfileDiff(
                control_sha256="c",
                treatment_sha256="c",
                direct_changed=(),
                transitive_changed=(),
                missing_packages=(),
                identical=True,
            ),
        ),
    ):
        fails6, _, _ = _check_equivalence_preflight(
            stub_ref,
            control,
            treatment,  # type: ignore[arg-type]
        )
    assert any("resolved specs differ" in f for f in fails6)


def test_render_recommendation_branches() -> None:
    spec = load_eval_spec("pens_order_involvement_ab")
    plan = plan_promote_eval(spec, dry_run=True, to_gw=1)
    cov = CoverageResult(
        metric="penalties_order_non_null",
        numerator=0,
        denominator=10,
        rate=0.0,
        min_coverage=0.01,
        passed=False,
        extras={"extra_key": "extra_val"},
    )
    preflight = PreflightResult(
        ok=False,
        coverage=cov,
        git_sha="deadbeef",
        git_dirty=True,
        pit_caveat="caveat text",
        control_spec=ModelSpec(name="c", kind="dc_layered", params={"n_sims": 100}),
        treatment_spec=ModelSpec(name="t", kind="dc_layered", params={"n_sims": 200}),
        effective_flags={
            "control.use_penalties_order": False,
            "treatment.use_penalties_order": True,
            "treatment.market_prior_source": "history",
            "treatment.market_prior_strength": 0.5,
            "treatment.market_cs_strength": 0.25,
        },
        control_lock_sha256="lockc",
        treatment_lock_sha256="lockt",
        control_ref="main",
        treatment_ref="feat",
    )
    per_match = pl.DataFrame(
        {
            "arm": ["raw_de_vig_1x2", "raw_de_vig_1x2"],
            "metric": ["rps", "rps"],
            "value": [0.1, 0.2],
        }
    )
    compare = CompareResult(
        season="2024-25",
        test_from_gw=1,
        test_to_gw=1,
        fit_gw=1,
        seasons_back=4,
        modes=(),
        runs=(),
        summary=pl.DataFrame(),
        detail=pl.DataFrame(),
        output_dir=Path("/tmp/compare-out"),
        calibration=None,
    )
    text = render_recommendation(
        spec,
        plan,
        preflight,
        verdict="drop",
        allow_dirty=True,
        compare=compare,
        deltas={
            "rmse": 0.1,
            "mae": 0.05,
            "spearman": 0.01,
            "top10_precision": 0.0,
            "brier_goal": 0.02,
            "log_assist": -0.01,
            "per_gw_wins": {"control": 1, "treatment": 2, "ties": 0},
            "per_gw_wins_mae": {"control": 2, "treatment": 1, "ties": 0},
            "mean_xpts": {"control": 3.0, "treatment": 3.1},
            "spa": {"pvalue_consistent": 0.1},
            "mcs": {"size": 2, "included": ["t"], "excluded": ["c"]},
            "intervals": [
                {
                    "metric": "rmse",
                    "position": None,
                    "mean": 0.1,
                    "hac_se": 0.01,
                    "hac_pvalue": 0.2,
                    "hac_ci": [0.0, 0.2],
                    "bootstrap_ci": [0.0, 0.2],
                }
            ],
        },
        prediction_ran=True,
        components_ran=True,
        component_deltas={"rps_delta": 0.01},
        decisions_ran=True,
        per_match=per_match,
    )
    assert "[--allow-dirty]" in text
    assert "(production)" in text
    assert "model_id=" in text
    assert "ref=main" in text
    assert "uv_lock_sha256=lockc" in text
    assert "extra_key: extra_val" in text
    assert "market_prior_source: history" in text
    assert "market_prior_strength" in text
    assert "market_cs_strength" in text
    assert "caveat text" in text
    assert "rmse_delta (headline" in text
    assert "per_gw_wins (RMSE)" in text
    assert "per_gw_wins (MAE" in text
    assert "mean_xpts:" in text
    assert "SPA" in text
    assert "MCS included" in text
    assert "decisions suite ran" in text
    assert "no event probabilities available for calibration" in text
    assert "(compare output:" in text
    assert "**drop**" in text
    # Coverage drop verdict paragraph.
    assert "Coverage 0.000 is below" in text

    # No coverage configured / prediction not run / components empty / decisions off.
    empty_pf = PreflightResult(ok=True, git_sha="x", coverage=None, pit_caveat=None)
    text2 = render_recommendation(
        spec,
        plan,
        empty_pf,
        verdict="inconclusive",
        allow_dirty=False,
        compare=None,
        deltas=None,
        prediction_ran=False,
        components_ran=True,
        component_deltas=None,
        decisions_ran=False,
    )
    assert "(no coverage metric configured)" in text2
    assert "prediction: not run" in text2
    assert "components suite ran (no pooled deltas)" in text2
    assert "decisions: not run" in text2
    assert "- (none)" in text2  # PIT caveat
    assert "n/a (improvement lever)" in text2

    # Equivalence keep / drop / inconclusive paragraphs via render.
    equiv = load_eval_spec("example_refactor_no_behaviour")
    equiv_plan = plan_promote_eval(equiv, dry_run=True, to_gw=1)
    diff_keep = EquivalenceDiff(
        identical=True,
        n_rows=10,
        n_beyond_tolerance=0,
        max_abs_diff=0.0,
        top_offenders=[],
    )
    pf_eq = PreflightResult(
        ok=True,
        git_sha="abc",
        control_lock_sha256="csha",
        treatment_lock_sha256="tsha",
        control_spec=ModelSpec(name="c", kind="dc_layered"),
        treatment_spec=ModelSpec(name="t", kind="dc_layered"),
    )
    keep_text = render_recommendation(
        equiv,
        equiv_plan,
        pf_eq,
        verdict="keep",
        allow_dirty=False,
        compare=None,
        deltas=None,
        prediction_ran=False,
        equivalence_diff=diff_keep,
    )
    assert "question_type: equivalence" in keep_text
    assert "orientation:" not in keep_text.split("## Arms")[0]
    assert "predictions compared: 10" in keep_text
    assert "top offenders: (none)" in keep_text
    assert "identical within tolerance" in keep_text
    assert "control=csha treatment=tsha" in keep_text

    diff_drop = EquivalenceDiff(
        identical=False,
        n_rows=5,
        n_beyond_tolerance=2,
        max_abs_diff=0.5,
        top_offenders=[{"player_key": 1, "abs_diff": 0.5}],
    )
    drop_text = render_recommendation(
        equiv,
        equiv_plan,
        pf_eq,
        verdict="drop",
        allow_dirty=False,
        compare=None,
        deltas=None,
        prediction_ran=False,
        equivalence_diff=diff_drop,
    )
    assert "beyond tolerance: 2" in drop_text
    assert "top offenders:" in drop_text
    assert "entirely on the bad side" in drop_text

    incon_text = render_recommendation(
        equiv,
        equiv_plan,
        pf_eq,
        verdict="inconclusive",
        allow_dirty=False,
        compare=None,
        deltas=None,
        prediction_ran=False,
        equivalence_diff=diff_drop,
    )
    assert "different, not worse" in incon_text

    no_diff_text = render_recommendation(
        equiv,
        equiv_plan,
        pf_eq,
        verdict="inconclusive",
        allow_dirty=False,
        compare=None,
        deltas=None,
        prediction_ran=False,
        equivalence_diff=None,
    )
    assert "equivalence suite did not run" in no_diff_text


def test_verdict_paragraph_branches() -> None:
    spec = load_eval_spec("pens_order_involvement_ab")
    leak_pf = PreflightResult(ok=False, leak=True, git_sha="x")
    assert "Leak:" in _verdict_paragraph(spec, "drop", leak_pf, allow_dirty=False)

    dirty_pf = PreflightResult(ok=True, git_dirty=True, git_sha="x")
    assert "Working tree was dirty" in _verdict_paragraph(
        spec, "inconclusive", dirty_pf, allow_dirty=True
    )

    low = EvalSpec.model_validate(
        _base_payload(
            gates={
                "min_coverage": 0.5,
                "allow_low_coverage": True,
                "coverage": {"metric": "penalties_order_non_null"},
            }
        )
    )
    low_pf = PreflightResult(
        ok=True, force_inconclusive=True, git_sha="x", coverage=None
    )
    assert "allow_low_coverage" in _verdict_paragraph(
        low, "inconclusive", low_pf, allow_dirty=False
    )

    # Improvement default paragraph.
    clean = PreflightResult(ok=True, git_sha="x")
    assert "auto-inconclusive" in _verdict_paragraph(
        spec, "inconclusive", clean, allow_dirty=False
    )


def _set_piece_status(
    store: DataStore,
    *,
    corners: int | None,
    freekicks: int | None,
    source: str = "fpl_api",
) -> None:
    """Overwrite player_status with explicit set-piece order columns."""
    season = "2025-26"
    store.write(
        "player_status",
        _utc(
            pl.DataFrame(
                {
                    "season": [season, season],
                    "player_key": [100, 200],
                    "status": ["a", "a"],
                    "news": ["", ""],
                    "price": [70, 80],
                    "ep_next": [3.0, 4.0],
                    "chance_of_playing": [100.0, 100.0],
                    "penalties_order": [1, None],
                    "corners_and_indirect_freekicks_order": [corners, None],
                    "direct_freekicks_order": [freekicks, None],
                    "available_at": [_START, _START],
                    "source": [source, source],
                }
            )
        ),
    )


def test_coverage_set_piece_order_empty_and_populated(tmp_path: Path) -> None:
    """Empty window + non-null orders cover both branches of the helper."""
    store = _seed_pens_store(tmp_path, order=1)
    spec = load_eval_spec("set_piece_involvement_ab")
    empty_plan = PromoteEvalPlan(
        eval_name=spec.name,
        kind=spec.kind,
        lever=spec.change.lever,
        control=spec.control,
        treatment=spec.treatment,
        arms=[],
        seasons=["2099-00"],
        from_gw=1,
        to_gw=1,
        seasons_back=4,
        modes=["walk_forward"],
        suites=spec.suites,
        commands=[],
        diff=list(spec.change.diff),
        notes="",
        dry_run=True,
    )
    empty = _coverage_set_piece_order(spec, empty_plan, store, pl.DataFrame())
    assert empty.metric == "set_piece_order_non_null"
    assert empty.numerator == 0
    assert empty.denominator == 0
    assert empty.rate == 0.0
    assert empty.passed is False
    assert empty.extras["corners_order_eq_1_rate"] == 0.0

    _set_piece_status(store, corners=1, freekicks=2)
    plan = plan_promote_eval(spec, dry_run=True, to_gw=1)
    filled = _coverage_set_piece_order(spec, plan, store, pl.DataFrame())
    assert filled.denominator == 2
    assert filled.numerator == 1
    assert filled.rate == pytest.approx(0.5)
    assert filled.passed is True
    assert filled.extras["corners_order_eq_1_rate"] == pytest.approx(0.5)
    assert filled.extras["direct_freekicks_order_eq_1_rate"] == 0.0
    assert filled.extras["source_fpl_api_share"] == pytest.approx(1.0)

    _set_piece_status(store, corners=None, freekicks=None, source="vaastav")
    nullish = _coverage_set_piece_order(spec, plan, store, pl.DataFrame())
    assert nullish.numerator == 0
    assert nullish.denominator == 2
    assert nullish.extras["source_fpl_api_share"] == 0.0


def test_coverage_set_piece_order_without_order_columns(tmp_path: Path) -> None:
    """When order columns are absent from status, numerator stays zero."""
    store = _seed_pens_store(tmp_path, order=1)
    spec = load_eval_spec("set_piece_involvement_ab")
    plan = plan_promote_eval(spec, dry_run=True, to_gw=1)
    with patch.object(
        store,
        "read",
        return_value=pl.DataFrame(
            {
                "season": ["2025-26", "2025-26"],
                "player_key": [100, 200],
                "available_at": [_START, _START],
                "source": ["fpl_api", "fpl_api"],
            }
        ),
    ):
        result = _coverage_set_piece_order(spec, plan, store, pl.DataFrame())
    assert result.numerator == 0
    assert result.denominator == 2
    assert result.extras["corners_order_eq_1_rate"] == 0.0


def _fake_compare(
    *,
    summary: pl.DataFrame,
    runs: tuple[CompareRun, ...] = (),
) -> CompareResult:
    return CompareResult(
        season="2024-25",
        test_from_gw=1,
        test_to_gw=1,
        fit_gw=1,
        seasons_back=4,
        modes=("walk_forward",),
        runs=runs,
        summary=summary,
        detail=pl.DataFrame(),
        output_dir=Path("/tmp/compare-out"),
    )


def test_prediction_deltas_early_exits_and_happy_path(tmp_path: Path) -> None:
    """_prediction_deltas covers empty / missing-id / ledger / SPA paths."""
    assert (
        _prediction_deltas(
            _fake_compare(summary=pl.DataFrame()),
            "c",
            "t",
            data_dir=tmp_path,
        )
        == {}
    )
    assert (
        _prediction_deltas(
            _fake_compare(summary=pl.DataFrame({"mae": [1.0]})),
            "c",
            "t",
            data_dir=tmp_path,
        )
        == {}
    )

    summary = pl.DataFrame(
        {
            "model_name": ["c", "t", "c", "t"],
            "mode": ["fixed", "fixed", "walk_forward", "walk_forward"],
            "horizon": [2, 2, 1, 1],
            "rmse": [2.0, 1.5, 1.2, 1.0],
            "mae": [1.5, 1.1, 0.9, 0.8],
            "spearman": [0.1, 0.2, 0.3, 0.4],
            "top10_precision": [0.2, 0.3, 0.4, 0.5],
            "brier_goal": [0.2, 0.1, 0.15, 0.1],
        }
    )
    missing_ids = _prediction_deltas(
        _fake_compare(summary=summary),
        "c",
        "t",
        data_dir=tmp_path,
    )
    assert missing_ids["rmse"] == pytest.approx(0.2)
    assert missing_ids["spearman"] == pytest.approx(0.1)
    assert missing_ids["per_gw_wins"] == {"control": 0, "treatment": 0, "ties": 0}

    control_spec = ModelSpec(name="c", kind="dc_layered", params={"tag": "c"})
    treatment_spec = ModelSpec(name="t", kind="dc_layered", params={"tag": "t"})
    dose_spec = ModelSpec(name="d", kind="dc_layered", params={"tag": "d"})
    cid, tid, did = (
        control_spec.model_id,
        treatment_spec.model_id,
        dose_spec.model_id,
    )
    runs = (
        CompareRun(model_name="c", mode="walk_forward", spec=control_spec),
        CompareRun(model_name="t", mode="walk_forward", spec=treatment_spec),
        CompareRun(model_name="d", mode="walk_forward", spec=dose_spec),
    )
    compare = _fake_compare(summary=summary, runs=runs)

    with patch(
        "fplt.backtest.promote_eval.read_predictions",
        side_effect=OSError("no ledger"),
    ):
        err = _prediction_deltas(compare, "c", "t", data_dir=tmp_path)
    assert err["per_gw_wins"]["ties"] == 0

    empty_ledger = pl.DataFrame(
        {
            "model_id": pl.Series([], dtype=pl.Utf8),
            "actual_pts": pl.Series([], dtype=pl.Float64),
        }
    )
    with patch(
        "fplt.backtest.promote_eval.read_predictions",
        return_value=empty_ledger,
    ):
        empty = _prediction_deltas(compare, "c", "t", data_dir=tmp_path)
    assert empty["per_gw_wins"] == {"control": 0, "treatment": 0, "ties": 0}

    scored = pl.DataFrame(
        {
            "model_id": [cid, tid, did] * 2,
            "actual_pts": [5.0, 6.0, 4.0, 7.0, 5.0, 6.0],
            "xpts": [4.0, 5.0, 3.0, 6.0, 5.5, 5.0],
            "target_season": ["2024-25"] * 6,
            "target_gw": [1, 1, 1, 2, 2, 2],
            "horizon": [1] * 6,
            "position": ["MID"] * 6,
        }
    )
    interval = PairedInterval(
        metric="rmse",
        n_gw=2,
        mean=0.1,
        hac_se=0.01,
        hac_pvalue=0.2,
        hac_ci_low=-0.1,
        hac_ci_high=0.3,
        bootstrap_ci_low=-0.05,
        bootstrap_ci_high=0.25,
        bootstrap_reps=100,
        bootstrap_seed=0,
    )
    spa = SpaResult(
        pvalue_consistent=0.1,
        pvalue_lower=0.05,
        pvalue_upper=0.2,
        n_gw=2,
        n_models=3,
    )
    mcs = McsResult(included=("t", "d"), excluded=("c",), size=0.9, n_gw=2)

    def _losses(ledger: pl.DataFrame, *, model_id: str, metric: str) -> pl.DataFrame:
        del ledger, metric
        base = {cid: 1.0, tid: 0.8, did: 0.9}
        return pl.DataFrame(
            {
                "target_season": ["2024-25", "2024-25"],
                "target_gw": [1, 2],
                "loss": [base[model_id], base[model_id] + 0.1],
            }
        )

    with (
        patch(
            "fplt.backtest.promote_eval.read_predictions",
            return_value=scored,
        ),
        patch(
            "fplt.backtest.promote_eval.filter_decision_pool",
            side_effect=RuntimeError("pool boom"),
        ),
        patch(
            "fplt.backtest.promote_eval.intervals_for_pair",
            return_value=[interval],
        ),
        patch(
            "fplt.backtest.promote_eval.equivalence_drop_from_intervals",
            return_value=False,
        ),
        patch(
            "fplt.backtest.promote_eval.per_gw_losses",
            side_effect=_losses,
        ),
        patch(
            "fplt.backtest.promote_eval.multi_arm_selection",
            return_value=(spa, mcs),
        ),
    ):
        happy = _prediction_deltas(
            compare, "c", "t", data_dir=tmp_path, position_specific=True
        )
    assert "intervals" in happy
    assert happy["spa"]["pvalue_consistent"] == 0.1
    assert happy["mcs"]["included"] == ["t", "d"]
    assert happy["per_gw_wins"]["treatment"] >= 0
    assert "per_gw_wins_mae" in happy

    with (
        patch(
            "fplt.backtest.promote_eval.read_predictions",
            return_value=scored,
        ),
        patch(
            "fplt.backtest.promote_eval.filter_decision_pool",
            return_value=scored.clear(),
        ),
    ):
        cleared = _prediction_deltas(compare, "c", "t", data_dir=tmp_path)
    assert cleared["per_gw_wins"] == {"control": 0, "treatment": 0, "ties": 0}

    only_c = summary.filter(pl.col("model_name") == "c")
    assert (
        _prediction_deltas(
            _fake_compare(summary=only_c, runs=runs),
            "c",
            "t",
            data_dir=tmp_path,
        )
        == {}
    )


def test_spa_mcs_from_ledger_branches() -> None:
    """_spa_mcs_from_ledger returns None on thin compares and payloads on success."""
    control = ModelSpec(name="c", kind="dc_layered", params={"tag": "c"})
    treatment = ModelSpec(name="t", kind="dc_layered", params={"tag": "d"})
    dose = ModelSpec(name="d", kind="dc_layered", params={"tag": "d2"})
    cid, tid, did = control.model_id, treatment.model_id, dose.model_id
    two = _fake_compare(
        summary=pl.DataFrame(),
        runs=(
            CompareRun(model_name="c", mode="walk_forward", spec=control),
            CompareRun(model_name="t", mode="walk_forward", spec=treatment),
        ),
    )
    scored = pl.DataFrame({"model_id": [cid], "xpts": [1.0]})
    assert (
        _spa_mcs_from_ledger(
            scored,
            compare=two,
            control="c",
            id_by_name={"c": cid, "t": tid},
        )
        is None
    )

    three = _fake_compare(
        summary=pl.DataFrame(),
        runs=(
            CompareRun(model_name="c", mode="walk_forward", spec=control),
            CompareRun(model_name="t", mode="walk_forward", spec=treatment),
            CompareRun(model_name="d", mode="walk_forward", spec=dose),
        ),
    )
    assert (
        _spa_mcs_from_ledger(
            scored,
            compare=three,
            control="missing",
            id_by_name={"c": cid, "t": tid, "d": did},
        )
        is None
    )
    with patch(
        "fplt.backtest.promote_eval.per_gw_losses",
        return_value=pl.DataFrame(
            {
                "target_season": ["2024-25"],
                "target_gw": [1],
                "loss": [1.0],
            }
        ),
    ):
        assert (
            _spa_mcs_from_ledger(
                scored,
                compare=three,
                control="c",
                id_by_name={"c": cid, "t": tid},
            )
            is None
        )

    empty_loss = pl.DataFrame(
        schema={
            "target_season": pl.Utf8(),
            "target_gw": pl.Int64(),
            "loss": pl.Float64(),
        }
    )
    with patch(
        "fplt.backtest.promote_eval.per_gw_losses",
        return_value=empty_loss,
    ):
        assert (
            _spa_mcs_from_ledger(
                scored,
                compare=three,
                control="c",
                id_by_name={"c": cid, "t": tid, "d": did},
            )
            is None
        )

    losses = pl.DataFrame(
        {
            "target_season": ["2024-25", "2024-25"],
            "target_gw": [1, 2],
            "loss": [1.0, 1.1],
        }
    )
    with (
        patch(
            "fplt.backtest.promote_eval.per_gw_losses",
            return_value=losses,
        ),
        patch(
            "fplt.backtest.promote_eval.multi_arm_selection",
            return_value=None,
        ),
    ):
        assert (
            _spa_mcs_from_ledger(
                scored,
                compare=three,
                control="c",
                id_by_name={"c": cid, "t": tid, "d": did},
            )
            is None
        )

    spa = SpaResult(0.2, 0.1, 0.3, 2, 3)
    mcs = McsResult(("c", "t"), ("d",), 0.75, 2)
    with (
        patch(
            "fplt.backtest.promote_eval.per_gw_losses",
            return_value=losses,
        ),
        patch(
            "fplt.backtest.promote_eval.multi_arm_selection",
            return_value=(spa, mcs),
        ),
    ):
        payload = _spa_mcs_from_ledger(
            scored,
            compare=three,
            control="c",
            id_by_name={"c": cid, "t": tid, "d": did},
        )
    assert payload is not None
    assert payload["spa"]["n_models"] == 3
    assert payload["mcs"]["excluded"] == ["d"]


def test_run_components_error_and_slot_paths(tmp_path: Path) -> None:
    """_run_components skips unresolved arms and runs mocked slot evals."""
    spec = EvalSpec.model_validate(_base_payload())
    plan = PromoteEvalPlan(
        eval_name=spec.name,
        kind=spec.kind,
        lever=spec.change.lever,
        control=spec.control,
        treatment=spec.treatment,
        arms=[],
        seasons=["2024-25"],
        from_gw=1,
        to_gw=1,
        seasons_back=4,
        modes=["walk_forward"],
        suites=spec.suites,
        commands=[],
        diff=list(spec.change.diff),
        notes="",
        dry_run=True,
    )
    bare = PreflightResult(ok=True, git_sha="x")
    with patch(
        "fplt.backtest.promote_eval.load_model_specs",
        return_value={},
    ):
        assert _run_components(
            spec,
            plan,
            bare,
            data_dir=tmp_path,
            output_dir=tmp_path / "c",
            models_dir=None,
        ) == (None, None, None)

    control = ModelSpec(name="c", kind="dc_layered")
    treatment = ModelSpec(name="t", kind="dc_layered")
    pf = PreflightResult(
        ok=True,
        git_sha="x",
        control_spec=control,
        treatment_spec=treatment,
    )
    with patch(
        "fplt.backtest.promote_eval.slot_component_names",
        return_value=None,
    ):
        assert _run_components(
            spec,
            plan,
            pf,
            data_dir=tmp_path,
            output_dir=tmp_path / "c",
            models_dir=None,
        ) == (None, None, None)

    minutes_control = ModelSpec(
        name="c",
        kind="dc_layered",
        components={"minutes": ModelSpec(name="mins_c", kind="empirical_minutes_v2")},
    )
    minutes_treatment = ModelSpec(
        name="t",
        kind="dc_layered",
        components={"minutes": ModelSpec(name="mins_t", kind="lgbm_minutes")},
    )
    pf_m = PreflightResult(
        ok=True,
        git_sha="x",
        control_spec=minutes_control,
        treatment_spec=minutes_treatment,
    )
    fake_result = ComponentEvalResult(
        summary=pl.DataFrame(
            {
                "model_name": ["mins_c", "mins_t"],
                "metric": ["mae", "mae"],
                "value": [1.0, 0.8],
            }
        ),
        sims=pl.DataFrame(),
        output_dir=tmp_path / "comp",
        per_match=pl.DataFrame({"arm": ["mins_c"], "metric": ["mae"], "value": [1.0]}),
    )
    with patch(
        "fplt.backtest.promote_eval.run_component_eval",
        return_value=fake_result,
    ) as runner:
        combined, per_match, deltas = _run_components(
            spec,
            plan,
            pf_m,
            data_dir=tmp_path,
            output_dir=tmp_path / "out",
            models_dir=None,
        )
    runner.assert_called()
    assert combined is not None
    assert per_match is not None
    assert deltas == {"mae_delta": pytest.approx(0.2)}

    score_spec = load_eval_spec("example_data_odds_prior")
    score_plan = plan_promote_eval(score_spec, dry_run=True, to_gw=1)
    pf_score = PreflightResult(
        ok=True,
        git_sha="x",
        control_spec=ModelSpec(name="c", kind="dc_layered"),
        treatment_spec=ModelSpec(name="t", kind="dc_layered"),
    )
    with (
        patch(
            "fplt.backtest.promote_eval.slot_component_names",
            return_value=None,
        ),
        patch(
            "fplt.backtest.promote_eval.run_component_eval",
            return_value=ComponentEvalResult(
                summary=pl.DataFrame(
                    {
                        "model_name": ["raw"],
                        "metric": ["rps"],
                        "value": [0.2],
                    }
                ),
                sims=pl.DataFrame(),
                output_dir=tmp_path / "s",
            ),
        ) as score_runner,
    ):
        combined_s, _, deltas_s = _run_components(
            score_spec,
            score_plan,
            pf_score,
            data_dir=tmp_path,
            output_dir=tmp_path / "score",
            models_dir=None,
        )
    assert combined_s is not None
    assert deltas_s is None
    kwargs = score_runner.call_args.kwargs
    assert kwargs["extra_score_specs"] is not None
    assert kwargs["write_per_match"] is True

    inv_spec = EvalSpec.model_validate(
        _base_payload(
            change={
                "lever": "component_swap",
                "orientation": "opt_in",
                "slot": "involvement",
                "allow_multi_lever": True,
                "diff": ["components.involvement.kind: a → b"],
            }
        )
    )
    inv_plan = PromoteEvalPlan(
        eval_name=inv_spec.name,
        kind=inv_spec.kind,
        lever=inv_spec.change.lever,
        control="c",
        treatment="t",
        arms=[],
        seasons=["2024-25"],
        from_gw=1,
        to_gw=1,
        seasons_back=4,
        modes=["walk_forward"],
        suites=inv_spec.suites,
        commands=[],
        diff=list(inv_spec.change.diff),
        notes="",
        dry_run=True,
    )
    pf_inv = PreflightResult(
        ok=True,
        git_sha="x",
        control_spec=ModelSpec(
            name="c",
            kind="dc_layered",
            components={"involvement": ModelSpec(name="inv_c", kind="involvement_xg")},
        ),
        treatment_spec=ModelSpec(
            name="t",
            kind="dc_layered",
            components={"involvement": ModelSpec(name="inv_t", kind="involvement_xg")},
        ),
    )
    with patch(
        "fplt.backtest.promote_eval.run_component_eval",
        return_value=ComponentEvalResult(
            summary=pl.DataFrame(),
            sims=pl.DataFrame(),
            output_dir=tmp_path / "i",
        ),
    ) as inv_runner:
        _run_components(
            inv_spec,
            inv_plan,
            pf_inv,
            data_dir=tmp_path,
            output_dir=tmp_path / "inv",
            models_dir=None,
        )
    assert inv_runner.call_args.kwargs["involvement_model"] == ["inv_c", "inv_t"]


def test_component_deltas_and_pooled_arm_metric() -> None:
    """_component_deltas / _pooled_arm_metric edge cases."""
    assert _component_deltas(pl.DataFrame(), ("a", "b")) == {}
    summary = pl.DataFrame(
        {
            "model_name": ["a", "b", "a", "b", "a"],
            "metric": ["rps", "rps", "mae", "mae", "flag"],
            "value": [0.4, 0.3, 1.0, None, True],
        }
    )
    deltas = _component_deltas(summary, ("a", "b"))
    assert deltas["rps_delta"] == pytest.approx(0.1)
    assert "mae_delta" not in deltas
    assert "flag_delta" not in deltas

    empty = pl.DataFrame(
        {"arm": pl.Series([], dtype=pl.Utf8), "metric": [], "value": []}
    )
    assert _pooled_arm_metric(empty, "x", "rps") is None
    frame = pl.DataFrame(
        {
            "arm": ["raw_de_vig_1x2", "raw_de_vig_1x2", "other"],
            "metric": ["rps", "rps", "rps"],
            "value": [0.2, 0.4, 0.9],
        }
    )
    assert _pooled_arm_metric(frame, "raw_de_vig_1x2", "rps") == pytest.approx(0.3)
    assert _pooled_arm_metric(frame, "missing", "rps") is None


def test_render_recommendation_scoreline_and_calibration() -> None:
    """Scoreline + calibration + window_note branches of render_recommendation."""
    from fplt.backtest.promote_eval import _RAW_DE_VIG_ARM

    spec = load_eval_spec("example_data_odds_prior")
    plan = plan_promote_eval(spec, dry_run=True, to_gw=1)
    plan = PromoteEvalPlan(
        eval_name=plan.eval_name,
        kind=plan.kind,
        lever=plan.lever,
        control=plan.control,
        treatment=plan.treatment,
        arms=plan.arms,
        seasons=plan.seasons,
        from_gw=plan.from_gw,
        to_gw=plan.to_gw,
        seasons_back=plan.seasons_back,
        modes=plan.modes,
        suites=plan.suites,
        commands=plan.commands,
        diff=plan.diff,
        notes=plan.notes,
        window_note="thin odds window",
        warnings=plan.warnings,
        dry_run=True,
    )
    preflight = PreflightResult(
        ok=True,
        git_sha="abc",
        control_spec=ModelSpec(name="c", kind="dc_layered", params={"n_sims": 50}),
        treatment_spec=ModelSpec(name="t", kind="dc_layered", params={"n_sims": 50}),
        effective_flags={"treatment.market_prior_source": "history"},
    )
    per_match = pl.DataFrame(
        {
            "arm": [_RAW_DE_VIG_ARM, _RAW_DE_VIG_ARM],
            "metric": ["rps", "rps"],
            "value": [0.1, 0.3],
        }
    )
    calibration = pl.DataFrame(
        {
            "event": ["goal"],
            "bucket": ["overall"],
            "n": [10],
            "mean_p": [0.2],
            "rate": [0.25],
        }
    )
    compare = CompareResult(
        season="2024-25",
        test_from_gw=1,
        test_to_gw=1,
        fit_gw=1,
        seasons_back=4,
        modes=("walk_forward",),
        runs=(),
        summary=pl.DataFrame(),
        detail=pl.DataFrame(),
        output_dir=Path("/tmp/compare-score"),
        calibration=calibration,
    )
    with patch(
        "fplt.backtest.promote_eval.format_calibration_markdown",
        return_value="|cal|",
    ):
        text = render_recommendation(
            spec,
            plan,
            preflight,
            verdict="inconclusive",
            allow_dirty=False,
            compare=compare,
            deltas={
                "rmse": 0.01,
                "brier_goal": 0.02,
                "spa": None,
                "mcs": None,
            },
            prediction_ran=True,
            components_ran=True,
            component_deltas={"rps_delta": 0.05},
            decisions_ran=False,
            per_match=per_match,
        )
    assert "thin odds window" in text
    assert "raw de-vig 1X2" in text
    assert "Event reliability" in text
    assert "|cal|" in text
    assert "market arm:" in text
    assert "per_match rows: 2" in text
    assert "brier_goal_delta" in text
