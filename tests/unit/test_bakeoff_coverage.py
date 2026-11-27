"""Extra bakeoff coverage: execute paths, ranking edges, selection guards."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import polars as pl
import yaml

from fplt.backtest.bakeoff import (
    BakeoffSpec,
    execute_bakeoff,
    plan_bakeoff,
    render_ranking,
    selection_payloads,
)
from fplt.backtest.compare import CompareResult, CompareRun


def _base_payload(**overrides: object) -> dict[str, object]:
    payload: dict[str, object] = {
        "name": "demo_championship",
        "question": "Which arm ranks best on the shared window?",
        "control": "rolling_mean_v1",
        "models": ["rolling_mean_v1", "dc_layered_v2", "lgbm_direct_v1"],
        "window": {
            "seasons": ["2024-25"],
            "from_gw": 1,
            "to_gw": 10,
            "modes": ["walk_forward"],
        },
        "suites": {"prediction": True, "components": False, "decisions": False},
    }
    payload.update(overrides)
    return payload


def _compare(tmp_path: Path, summary: pl.DataFrame) -> CompareResult:
    def _run(name: str, mid: str) -> CompareRun:
        return CompareRun(
            model_name=name,
            mode="walk_forward",
            spec=MagicMock(model_id=mid),
        )

    return CompareResult(
        season="2024-25",
        test_from_gw=1,
        test_to_gw=3,
        fit_gw=1,
        seasons_back=4,
        modes=("walk_forward",),
        runs=(
            _run("rolling_mean_v1", "rm"),
            _run("dc_layered_v2", "dc"),
            _run("lgbm_direct_v1", "lg"),
        ),
        summary=summary,
        detail=summary,
        output_dir=tmp_path / "compare",
    )


def _summary() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "model_name": ["rolling_mean_v1", "dc_layered_v2", "lgbm_direct_v1"],
            "mode": ["walk_forward"] * 3,
            "horizon": [1, 1, 1],
            "mae": [1.2, 1.0, 1.1],
            "rmse": [2.0, 1.8, 1.9],
            "spearman": [0.5, 0.6, 0.55],
            "n": [100, 100, 100],
            "model_id": ["rm", "dc", "lg"],
        }
    )


def test_execute_bakeoff_multi_season_compare_dirs(tmp_path: Path) -> None:
    spec = BakeoffSpec.model_validate(
        _base_payload(
            window={
                "seasons": ["2024-25", "2023-24"],
                "from_gw": 1,
                "to_gw": 3,
                "modes": ["walk_forward"],
            },
        )
    )
    compare = _compare(tmp_path, _summary())
    with (
        patch(
            "fplt.backtest.bakeoff.load_model_specs",
            return_value={
                "rolling_mean_v1": MagicMock(),
                "dc_layered_v2": MagicMock(),
                "lgbm_direct_v1": MagicMock(),
            },
        ),
        patch("fplt.backtest.bakeoff.compare_models", return_value=compare) as cmp,
        patch(
            "fplt.backtest.bakeoff.selection_payloads",
            return_value=(None, None),
        ),
    ):
        result = execute_bakeoff(spec, data_dir=tmp_path, dry_run=False)
    assert result.exit_code == 0
    assert cmp.call_count == 2
    out_dirs = [c.kwargs.get("output_dir") for c in cmp.call_args_list]
    assert any(p is not None and p.name == "2024-25" for p in out_dirs)
    assert any(p is not None and p.name == "2023-24" for p in out_dirs)


def test_execute_bakeoff_runs_pairwise_evals(tmp_path: Path) -> None:
    models = tmp_path / "models"
    models.mkdir()
    evals = tmp_path / "evals"
    evals.mkdir()
    (evals / "pair_a.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "pair_a",
                "kind": "config",
                "question": "Pair?",
                "change": {
                    "lever": "hyperparam",
                    "orientation": "opt_in",
                    "allow_multi_lever": True,
                    "diff": ["x"],
                },
                "control": "rolling_mean_v1",
                "treatment": "dc_layered_v2",
                "window": {"seasons": ["2024-25"], "from_gw": 1, "to_gw": 3},
            }
        ),
        encoding="utf-8",
    )
    spec = BakeoffSpec.model_validate(_base_payload(pairwise_evals=["pair_a"]))
    compare = _compare(tmp_path, _summary())
    pair_result = MagicMock(verdict="drop", exit_code=0)
    with (
        patch(
            "fplt.backtest.bakeoff.load_model_specs",
            return_value={
                "rolling_mean_v1": MagicMock(),
                "dc_layered_v2": MagicMock(),
                "lgbm_direct_v1": MagicMock(),
            },
        ),
        patch("fplt.backtest.bakeoff.compare_models", return_value=compare),
        patch(
            "fplt.backtest.bakeoff.selection_payloads",
            return_value=(
                {
                    "pvalue_consistent": 0.1,
                    "pvalue_lower": 0.05,
                    "pvalue_upper": 0.2,
                    "n_gw": 10,
                    "n_models": 2,
                    "skipped_empty": ["ghost"],
                },
                {
                    "included": ["dc_layered_v2", "lgbm_direct_v1", "rolling_mean_v1"],
                    "excluded": [],
                    "size": 0.05,
                    "n_gw": 10,
                },
            ),
        ),
        patch("fplt.backtest.bakeoff.load_eval_spec", return_value=MagicMock()),
        patch(
            "fplt.backtest.bakeoff.execute_promote_eval",
            return_value=pair_result,
        ) as pair_exec,
        patch("fplt.backtest.bakeoff.default_evals_dir", return_value=evals),
    ):
        result = execute_bakeoff(
            spec, data_dir=tmp_path, dry_run=False, models_dir=models, evals_dir=evals
        )
    assert result.exit_code == 0
    assert pair_exec.called
    ranking = (result.output_dir / "ranking.md").read_text(encoding="utf-8")
    assert "pair_a" in ranking
    assert "MCS excluded: (none)" in ranking
    assert "Skipped empty arms" in ranking
    assert "ghost" in ranking


def test_render_ranking_pairwise_and_mcs_edges() -> None:
    spec = BakeoffSpec.model_validate(_base_payload())
    with patch(
        "fplt.backtest.bakeoff.load_model_specs",
        return_value={
            "rolling_mean_v1": MagicMock(),
            "dc_layered_v2": MagicMock(),
            "lgbm_direct_v1": MagicMock(),
        },
    ):
        plan = plan_bakeoff(spec)
    text = render_ranking(
        spec,
        plan,
        summary=_summary(),
        spa={
            "pvalue_consistent": 0.2,
            "pvalue_lower": 0.1,
            "pvalue_upper": 0.3,
            "n_gw": 5,
            "n_models": 2,
            "skipped_empty": ["empty_arm"],
        },
        mcs={
            "included": ["dc_layered_v2"],
            "excluded": [],
            "size": 0.05,
            "n_gw": 5,
        },
        pairwise=["pair_a"],
    )
    assert "MCS excluded: (none)" in text
    assert "empty_arm" in text
    assert "pair_a" in text


def test_selection_payloads_read_and_empty_guards(tmp_path: Path) -> None:
    compare = CompareResult(
        season="2024-25",
        test_from_gw=1,
        test_to_gw=2,
        fit_gw=1,
        seasons_back=4,
        modes=("walk_forward",),
        runs=(
            CompareRun(
                model_name="a",
                mode="walk_forward",
                spec=MagicMock(model_id="a"),
            ),
            CompareRun(
                model_name="b",
                mode="walk_forward",
                spec=MagicMock(model_id="b"),
            ),
            CompareRun(
                model_name="c",
                mode="walk_forward",
                spec=MagicMock(model_id="c"),
            ),
        ),
        summary=pl.DataFrame(),
        detail=pl.DataFrame(),
        output_dir=tmp_path,
    )
    with patch(
        "fplt.backtest.bakeoff.read_predictions",
        side_effect=RuntimeError("no ledger"),
    ):
        spa, mcs = selection_payloads(
            data_dir=tmp_path,
            compare=compare,
            control="a",
            models=["a", "b", "c"],
        )
    assert spa is None and mcs is None

    empty = pl.DataFrame(schema={"model_id": pl.Utf8(), "actual_pts": pl.Float64()})
    with patch("fplt.backtest.bakeoff.read_predictions", return_value=empty):
        spa, mcs = selection_payloads(
            data_dir=tmp_path,
            compare=compare,
            control="a",
            models=["a", "b", "c"],
        )
    assert spa is None and mcs is None

    ledger = pl.DataFrame(
        {"model_id": ["a", "b", "c"], "actual_pts": [1.0, 1.0, 1.0]}
    )
    with (
        patch("fplt.backtest.bakeoff.read_predictions", return_value=ledger),
        patch(
            "fplt.backtest.bakeoff.filter_decision_pool",
            return_value=pl.DataFrame(
                schema={"model_id": pl.Utf8(), "actual_pts": pl.Float64()}
            ),
        ),
    ):
        spa, mcs = selection_payloads(
            data_dir=tmp_path,
            compare=compare,
            control="a",
            models=["a", "b", "c"],
        )
    assert spa is None and mcs is None

    with (
        patch("fplt.backtest.bakeoff.read_predictions", return_value=ledger),
        patch("fplt.backtest.bakeoff.filter_decision_pool", return_value=ledger),
    ):
        # Put the unknown arm first so mid is None before per_gw_losses runs.
        spa, mcs = selection_payloads(
            data_dir=tmp_path,
            compare=compare,
            control="a",
            models=["missing", "a", "b"],
        )
    assert spa is None and mcs is None


def test_selection_payloads_control_missing_and_thin_arms(tmp_path: Path) -> None:
    seasons = ["2024-25"] * 2
    gws = [1, 2]
    ledger = pl.DataFrame({"model_id": ["ctrl", "alt"], "actual_pts": [1.0, 1.0]})
    compare = CompareResult(
        season="2024-25",
        test_from_gw=1,
        test_to_gw=2,
        fit_gw=1,
        seasons_back=4,
        modes=("walk_forward",),
        runs=(
            CompareRun(
                model_name="control",
                mode="walk_forward",
                spec=MagicMock(model_id="ctrl"),
            ),
            CompareRun(
                model_name="challenger",
                mode="walk_forward",
                spec=MagicMock(model_id="alt"),
            ),
            CompareRun(
                model_name="empty",
                mode="walk_forward",
                spec=MagicMock(model_id="empty"),
            ),
        ),
        summary=pl.DataFrame(),
        detail=pl.DataFrame(),
        output_dir=tmp_path,
    )

    def fake_losses(
        _scored: pl.DataFrame, *, model_id: str, metric: str
    ) -> pl.DataFrame:
        del metric
        if model_id in {"empty", "ctrl"}:
            return pl.DataFrame(
                schema={
                    "target_season": pl.Utf8,
                    "target_gw": pl.Int64,
                    "loss": pl.Float64,
                }
            )
        return pl.DataFrame(
            {
                "target_season": seasons,
                "target_gw": gws,
                "loss": [0.9, 0.95],
            }
        )

    with (
        patch("fplt.backtest.bakeoff.read_predictions", return_value=ledger),
        patch("fplt.backtest.bakeoff.filter_decision_pool", return_value=ledger),
        patch("fplt.backtest.bakeoff.per_gw_losses", side_effect=fake_losses),
    ):
        spa, mcs = selection_payloads(
            data_dir=tmp_path,
            compare=compare,
            control="control",
            models=["control", "challenger", "empty"],
        )
    assert spa is None and mcs is None
