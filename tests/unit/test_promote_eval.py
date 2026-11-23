"""Tests for EvalSpec loading, preflight, coverage, and promote-eval execute."""

from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest.mock import patch

import polars as pl
import pytest
import yaml
from pydantic import ValidationError

from fplt.backtest.component_eval import ComponentEvalResult
from fplt.backtest.promote_eval import (
    EvalSpec,
    decisions_claim_failures,
    detect_points_claim_phrase,
    execute_promote_eval,
    format_plan,
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
        runs: list[Any] = []
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
