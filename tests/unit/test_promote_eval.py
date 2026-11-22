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
    assert "--score-models dixon_coles_v2 dixon_coles_odds_v1" in cmd
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
    assert spec.control == "dc_layered_shares_v2"
    assert spec.treatment == "dc_layered_v2"
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
        control="dc_layered_v2",
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
        specs["dc_layered_v2"], specs["dc_layered_odds_v1"], "score"
    )
    assert names == ("dixon_coles_v2", "dixon_coles_odds_v1")
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
                "dixon_coles_v2",
                "dixon_coles_v2",
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
                "dixon_coles_v2",
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
        assert score_models == ["dixon_coles_v2", "dixon_coles_odds_v1"]
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
        "dixon_coles_v2",
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
        control="dc_layered_v2",
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
        treatment="dc_layered_v2",
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
        control="dc_layered_v2",
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
                "model_name": ["dc_layered_nopens_v1", "dc_layered_v2"],
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
                "model_name": ["dc_layered_nopens_v1", "dc_layered_v2"],
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
    assert "mae_delta" in rec
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
        control="dc_layered_v2",
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
        control="dc_layered_v2",
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
