"""Multi-model eval resolves the core set and optional OpenFPL."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl
import pytest

from fplt.backtest.eval_set import _yaml_openfpl_model_dir_set
from fplt.models.direct.lgbm import LGBMDirectPredictor
from fplt.models.spec import ModelSpec, get_model
from fplt.orchestration import defs
from fplt.orchestration.eval_job import (
    CORE_MODELS,
    OPENFPL_MODEL,
    EvalResult,
    _openfpl_skip_note,
    _run_openfpl,
    resolve_eval_models,
    run_eval_board,
)
from fplt.settings import get_settings
from scripts.eval_board import build_parser, main


def test_default_models_are_the_core_set() -> None:
    assert resolve_eval_models("2024-25") == list(CORE_MODELS)


def test_with_openfpl_appends_openfpl() -> None:
    assert resolve_eval_models("2024-25", with_openfpl=True) == [
        *CORE_MODELS,
        OPENFPL_MODEL,
    ]


def test_openfpl_model_dir_implies_inclusion() -> None:
    assert OPENFPL_MODEL in resolve_eval_models(
        "2024-25", openfpl_model_dir="/models/openfpl"
    )


def test_yaml_model_dir_auto_includes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = ModelSpec(
        name=OPENFPL_MODEL,
        kind="openfpl",
        params={"model_dir": "/fitted", "compare_from": "2024-25"},
    )
    monkeypatch.setattr(
        "fplt.backtest.eval_set.load_model_specs",
        lambda: {OPENFPL_MODEL: spec},
    )
    assert OPENFPL_MODEL in resolve_eval_models("2025-26")


def test_openfpl_rejects_training_seasons() -> None:
    with pytest.raises(ValueError, match="2024-25"):
        resolve_eval_models("2023-24", with_openfpl=True)
    with pytest.raises(ValueError, match="2024-25"):
        resolve_eval_models("2023-24", models=[OPENFPL_MODEL])


def test_explicit_models_override_flags() -> None:
    assert resolve_eval_models(
        "2024-25",
        models=["rolling_mean_v1"],
        with_openfpl=True,
    ) == ["rolling_mean_v1"]


def test_empty_explicit_models_are_rejected() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        resolve_eval_models("2024-25", models=[])


def test_bad_season_label_is_rejected() -> None:
    with pytest.raises(ValueError, match="season must look like"):
        resolve_eval_models("2024", with_openfpl=True)


def test_run_eval_board_delegates_to_backtest(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    called: dict[str, Any] = {}

    def fake_backtest(
        data_dir: Path,
        season: str,
        from_gw: int,
        to_gw: int,
        models: list[str],
        *,
        refit: str | None = None,
        seasons_back: int | None = None,
        steps: int = 1,
    ) -> list[ModelSpec]:
        called["args"] = {
            "data_dir": data_dir,
            "season": season,
            "from_gw": from_gw,
            "to_gw": to_gw,
            "models": models,
            "refit": refit,
            "seasons_back": seasons_back,
            "steps": steps,
        }
        return [ModelSpec(name=name, kind="rolling_mean") for name in models]

    monkeypatch.setattr("fplt.orchestration.eval_job.run_backtest", fake_backtest)
    monkeypatch.setattr(
        "fplt.backtest.eval_set._yaml_openfpl_model_dir_set",
        lambda: False,
    )
    result = run_eval_board(
        tmp_path,
        "2024-25",
        1,
        2,
        models=["rolling_mean_v1", "ep_next_v1"],
        steps=2,
    )
    assert called["args"]["models"] == ["rolling_mean_v1", "ep_next_v1"]
    assert called["args"]["steps"] == 2
    assert [spec.name for spec in result.specs] == [
        "rolling_mean_v1",
        "ep_next_v1",
    ]
    assert result.skipped_openfpl is None


def test_run_eval_board_patches_openfpl_model_dir(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    openfpl_calls: list[str | None] = []

    def fake_backtest(
        data_dir: Path,
        season: str,
        from_gw: int,
        to_gw: int,
        models: list[str],
        **_: object,
    ) -> list[ModelSpec]:
        return [ModelSpec(name=name, kind="rolling_mean") for name in models]

    def fake_openfpl(
        data_dir: Path,
        season: str,
        from_gw: int,
        to_gw: int,
        *,
        model_dir: str | None,
        refit: object,
        seasons_back: object,
        steps: int,
    ) -> ModelSpec:
        del data_dir, season, from_gw, to_gw, refit, seasons_back, steps
        openfpl_calls.append(model_dir)
        return ModelSpec(name=OPENFPL_MODEL, kind="openfpl")

    monkeypatch.setattr("fplt.orchestration.eval_job.run_backtest", fake_backtest)
    monkeypatch.setattr("fplt.orchestration.eval_job._run_openfpl", fake_openfpl)
    monkeypatch.setattr("fplt.orchestration.eval_job.fill_actuals", lambda *_: None)
    result = run_eval_board(
        tmp_path,
        "2024-25",
        1,
        1,
        models=["rolling_mean_v1", OPENFPL_MODEL],
        openfpl_model_dir="/ensembles",
    )
    assert openfpl_calls == ["/ensembles"]
    assert [spec.name for spec in result.specs] == [
        "rolling_mean_v1",
        OPENFPL_MODEL,
    ]


def test_skip_note_when_openfpl_omitted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "fplt.orchestration.eval_job.run_backtest",
        lambda *a, **k: [ModelSpec(name="rolling_mean_v1", kind="rolling_mean")],
    )
    monkeypatch.setattr(
        "fplt.backtest.eval_set._yaml_openfpl_model_dir_set",
        lambda: False,
    )
    result = run_eval_board(tmp_path, "2024-25", 1, 1, models=None)
    assert result.skipped_openfpl is not None
    assert OPENFPL_MODEL in result.skipped_openfpl


def test_parser_defaults() -> None:
    args = build_parser().parse_args(["--season", "2024-25"])
    assert args.season == "2024-25"
    assert args.from_gw == 1
    assert args.to_gw == 38
    assert args.with_openfpl is False
    assert args.models is None


def test_script_main_prints_specs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    spec = ModelSpec(name="rolling_mean_v1", kind="rolling_mean")
    monkeypatch.setattr(
        "scripts.eval_board.run_eval_board",
        lambda *a, **k: EvalResult(specs=(spec,), skipped_openfpl="skipped note"),
    )
    assert main(["--season", "2024-25", "--models", "rolling_mean_v1"]) == 0
    out = capsys.readouterr()
    assert "rolling_mean_v1" in out.out
    assert "skipped note" in out.err
    get_settings.cache_clear()


def test_script_main_maps_errors(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()

    def boom(*_a: object, **_k: object) -> EvalResult:
        raise ValueError("bad season")

    monkeypatch.setattr("scripts.eval_board.run_eval_board", boom)
    assert main(["--season", "nope"]) == 2
    assert "bad season" in capsys.readouterr().err
    get_settings.cache_clear()


def test_eval_board_job_is_registered() -> None:
    assert "eval_board" in defs.get_repository_def().job_names


def test_eval_board_op_uses_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    seen: dict[str, object] = {}

    def fake_run(
        data_dir: Path,
        season: str,
        from_gw: int,
        to_gw: int,
        **kwargs: object,
    ) -> EvalResult:
        seen["data_dir"] = data_dir
        seen["season"] = season
        seen["kwargs"] = kwargs
        return EvalResult(
            specs=(ModelSpec(name="rolling_mean_v1", kind="rolling_mean"),)
        )

    monkeypatch.setattr("fplt.orchestration.eval_job.run_eval_board", fake_run)
    from dagster import RunConfig

    from fplt.orchestration.eval_job import EvalBoardConfig, eval_board_job

    result = eval_board_job.execute_in_process(
        run_config=RunConfig(ops={"eval_board_op": EvalBoardConfig(season="2024-25")})
    )
    assert result.success
    assert result.output_for_node("eval_board_op") == ["rolling_mean_v1"]
    assert seen["data_dir"] == tmp_path
    assert seen["season"] == "2024-25"
    get_settings.cache_clear()


def test_eval_job_registers_lgbm_direct() -> None:
    assert get_model("lgbm_direct") is LGBMDirectPredictor


def test_skip_note_for_early_default_season(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "fplt.orchestration.eval_job.run_backtest",
        lambda *a, **k: [ModelSpec(name="rolling_mean_v1", kind="rolling_mean")],
    )
    monkeypatch.setattr(
        "fplt.backtest.eval_set._yaml_openfpl_model_dir_set",
        lambda: False,
    )
    monkeypatch.setattr(
        "fplt.orchestration.eval_job.resolve_eval_models",
        lambda season, **_: list(CORE_MODELS),
    )
    result = run_eval_board(tmp_path, "2023-24", 1, 1)
    assert result.skipped_openfpl is not None
    assert "before" in result.skipped_openfpl


def test_default_checkout_does_not_auto_include(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fplt.models.external.openfpl import DEFAULT_MODEL_DIR

    spec = ModelSpec(
        name=OPENFPL_MODEL,
        kind="openfpl",
        params={"model_dir": DEFAULT_MODEL_DIR, "compare_from": "2024-25"},
    )
    monkeypatch.setattr(
        "fplt.backtest.eval_set.load_model_specs",
        lambda: {OPENFPL_MODEL: spec},
    )
    assert _yaml_openfpl_model_dir_set() is False
    assert resolve_eval_models("2024-25") == list(CORE_MODELS)


def test_yaml_model_dir_false_when_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "fplt.backtest.eval_set.load_model_specs",
        lambda: {},
    )
    assert _yaml_openfpl_model_dir_set() is False


def test_openfpl_skip_note_short_circuits() -> None:
    assert (
        _openfpl_skip_note(
            "2024-25",
            [OPENFPL_MODEL],
            with_openfpl=False,
            openfpl_model_dir=None,
        )
        is None
    )
    assert (
        _openfpl_skip_note(
            "2024-25",
            list(CORE_MODELS),
            with_openfpl=True,
            openfpl_model_dir=None,
        )
        is None
    )
    assert (
        _openfpl_skip_note(
            "2024-25",
            list(CORE_MODELS),
            with_openfpl=False,
            openfpl_model_dir="/x",
        )
        is None
    )


def test_run_openfpl_rejects_bad_ranges(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="steps"):
        _run_openfpl(
            tmp_path,
            "2024-25",
            1,
            1,
            model_dir=None,
            refit=None,
            seasons_back=None,
            steps=0,
        )
    with pytest.raises(ValueError, match="seasons_back"):
        _run_openfpl(
            tmp_path,
            "2024-25",
            1,
            1,
            model_dir=None,
            refit=None,
            seasons_back=0,
            steps=1,
        )
    with pytest.raises(ValueError, match="gameweeks"):
        _run_openfpl(
            tmp_path,
            "2024-25",
            3,
            1,
            model_dir=None,
            refit=None,
            seasons_back=None,
            steps=1,
        )


def test_run_openfpl_walks_forward(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = ModelSpec(
        name=OPENFPL_MODEL,
        kind="openfpl",
        params={"model_dir": "", "compare_from": "2024-25"},
        refit="never",
    )
    monkeypatch.setattr(
        "fplt.orchestration.eval_job.load_model_specs",
        lambda: {OPENFPL_MODEL: base},
    )

    class Stub:
        def __init__(self, spec: ModelSpec) -> None:
            self.spec = spec

        def fit(self, view: object) -> None:
            del view

        def predict(self, view: object, horizon: object) -> pl.DataFrame:
            del view, horizon
            return pl.DataFrame()

    monkeypatch.setattr("fplt.orchestration.eval_job.get_model", lambda _kind: Stub)
    monkeypatch.setattr(
        "fplt.orchestration.eval_job.walk_forward",
        lambda *_a, **_k: pl.DataFrame({"xpts": [1.0]}),
    )
    appended: list[Path] = []

    def fake_append(data_dir: Path, frame: pl.DataFrame) -> None:
        del frame
        appended.append(data_dir)

    monkeypatch.setattr("fplt.orchestration.eval_job.append_predictions", fake_append)
    monkeypatch.setattr("fplt.orchestration.eval_job.fill_actuals", lambda *_: None)
    monkeypatch.setattr("fplt.orchestration.eval_job.PITData", lambda store: store)
    monkeypatch.setattr("fplt.orchestration.eval_job.DataStore", lambda root: root)

    result = run_eval_board(
        tmp_path,
        "2024-25",
        1,
        1,
        models=[OPENFPL_MODEL],
        openfpl_model_dir="/ensembles",
        refit="never",
        seasons_back=2,
    )
    assert appended == [tmp_path]
    assert result.specs[0].params["model_dir"] == "/ensembles"
    assert result.specs[0].train_window.seasons_back == 2


def test_run_openfpl_warns_on_empty_dir(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
) -> None:
    base = ModelSpec(
        name=OPENFPL_MODEL,
        kind="openfpl",
        params={"model_dir": "", "compare_from": "2024-25"},
    )
    monkeypatch.setattr(
        "fplt.orchestration.eval_job.load_model_specs",
        lambda: {OPENFPL_MODEL: base},
    )
    monkeypatch.setattr(
        "fplt.orchestration.eval_job.get_model",
        lambda _kind: lambda spec: type("M", (), {"spec": spec})(),
    )
    monkeypatch.setattr(
        "fplt.orchestration.eval_job.walk_forward",
        lambda *_a, **_k: pl.DataFrame(),
    )
    monkeypatch.setattr(
        "fplt.orchestration.eval_job.append_predictions", lambda *_: None
    )
    monkeypatch.setattr("fplt.orchestration.eval_job.PITData", lambda store: store)
    monkeypatch.setattr("fplt.orchestration.eval_job.DataStore", lambda root: root)
    with caplog.at_level("WARNING"):
        _run_openfpl(
            tmp_path,
            "2024-25",
            1,
            1,
            model_dir=None,
            refit=None,
            seasons_back=None,
            steps=1,
        )
    assert "model_dir is empty" in caplog.text


def test_run_openfpl_missing_spec(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        "fplt.orchestration.eval_job.load_model_specs",
        lambda: {},
    )
    with pytest.raises(ValueError, match="unknown model spec"):
        _run_openfpl(
            tmp_path,
            "2024-25",
            1,
            1,
            model_dir="/x",
            refit=None,
            seasons_back=None,
            steps=1,
        )


def test_eval_board_op_logs_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    monkeypatch.setattr(
        "fplt.orchestration.eval_job.run_eval_board",
        lambda *a, **k: EvalResult(
            specs=(ModelSpec(name="rolling_mean_v1", kind="rolling_mean"),),
            skipped_openfpl="skipped openfpl_v1: demo",
        ),
    )
    from dagster import RunConfig

    from fplt.orchestration.eval_job import EvalBoardConfig, eval_board_job

    result = eval_board_job.execute_in_process(
        run_config=RunConfig(ops={"eval_board_op": EvalBoardConfig(season="2024-25")})
    )
    assert result.success
    get_settings.cache_clear()
