"""Toolkit 6 / §2.4: fplt exp component CLI + cache reuse + results schema."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import polars as pl
import pytest

import fplt.models.composite  # noqa: F401 — registers dc_layered
from fplt.backtest.component_experiment import (
    CONTROL_ARM,
    ParsedWindow,
    load_base_pipeline,
    measure_to_arm_metrics,
    model_for_pipeline,
    parse_arm_patch,
    parse_component_window,
    run_component_experiment,
    run_component_experiment_bench,
)
from fplt.backtest.measure import empty_measure_metrics
from fplt.cli import _exp, main
from fplt.data.store import DataStore
from fplt.models.component_cache import LAYERED_COMPONENT_SLOTS
from fplt.models.pipeline_spec import live_pipeline_defaults
from fplt.schemas.experiment_results import ArmMetricsSchema
from fplt.schemas.measure import MeasureMetricsSchema
from fplt.store.experiments import experiment_dir, load_experiment_manifest
from tests.builders import build_league

_LOCK = "a" * 64
_SHA = "deadbeef"
_TS = datetime(2026, 11, 29, 0, 0, 0, tzinfo=UTC)


def _league(tmp_path: Path) -> Path:
    root = tmp_path / "data"
    build_league(DataStore(root))
    return root


def test_parse_component_window_forms() -> None:
    full = parse_component_window("2024-25")
    assert full.season == "2024-25"
    assert full.from_gw == 1
    assert full.to_gw == 38
    assert full.label() == "2024-25"
    part = parse_component_window("2024-25:2-10")
    assert part.from_gw == 2
    assert part.to_gw == 10
    assert part.label() == "2024-25:2-10"
    assert part.as_ofs()[0].gw == 2
    with pytest.raises(ValueError, match="window must"):
        parse_component_window("2024")
    with pytest.raises(ValueError, match="from_gw"):
        parse_component_window("2024-25:5-1")
    with pytest.raises(ValueError, match="1..38"):
        parse_component_window("2024-25:0-1")
    with pytest.raises(ValueError, match="1..38"):
        parse_component_window("2024-25:1-99")


def test_parse_arm_patch_json_kv_and_spec(tmp_path: Path) -> None:
    assert parse_arm_patch(
        '{"model.components.minutes.params.logistic_blend": 0.8}',
        component="minutes",
    ) == {"model.components.minutes.params.logistic_blend": 0.8}
    assert parse_arm_patch(
        "model.components.minutes.params.logistic_blend=0.75",
        component="minutes",
    ) == {"model.components.minutes.params.logistic_blend": 0.75}
    assert parse_arm_patch(
        "model.components.minutes.params.note=plain",
        component="minutes",
    ) == {"model.components.minutes.params.note": "plain"}
    assert parse_arm_patch("empirical_minutes_v2", component="minutes") == {
        "model.components.minutes": "empirical_minutes_v2"
    }
    path = tmp_path / "arm.yaml"
    path.write_text(
        "model.components.minutes.params.logistic_blend: 0.9\n",
        encoding="utf-8",
    )
    assert parse_arm_patch(str(path), component="minutes")[
        "model.components.minutes.params.logistic_blend"
    ] == pytest.approx(0.9)
    json_path = tmp_path / "arm.json"
    json_path.write_text(
        '{"model.components.minutes.params.logistic_blend": 0.7}',
        encoding="utf-8",
    )
    assert parse_arm_patch(str(json_path), component="minutes")[
        "model.components.minutes.params.logistic_blend"
    ] == pytest.approx(0.7)
    with pytest.raises(ValueError, match="unrecognised"):
        parse_arm_patch("not_a_real_spec_zzz", component="minutes")
    with pytest.raises(ValueError, match="non-empty"):
        parse_arm_patch("   ", component="minutes")
    with pytest.raises(ValueError, match="object"):
        parse_arm_patch("[1, 2]", component="minutes")
    bad_yaml = tmp_path / "bad.yaml"
    bad_yaml.write_text("- just a list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        parse_arm_patch(str(bad_yaml), component="minutes")


def test_load_base_pipeline_and_model_for_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from fplt.models.spec import load_model_specs

    with pytest.raises(ValueError, match="non-empty"):
        load_base_pipeline("  ")
    with pytest.raises(KeyError, match="unknown base"):
        load_base_pipeline("no_such_model_zzz")
    pipe = load_base_pipeline("dc_layered_v4")
    assert pipe.model_name == "dc_layered_v4"
    model = model_for_pipeline(pipe)
    assert str(model.params.get("mc_backend", "python")) == str(pipe.env.mc_backend)
    # Already-matching backend (implicit python) returns without rewriting params.
    again = model_for_pipeline(
        live_pipeline_defaults(model="dc_layered_v4", mc_backend="python")
    )
    assert str(again.params.get("mc_backend", "python")) == "python"
    # Explicit env override rewrites model params.
    native_pipe = live_pipeline_defaults(model="dc_layered_v4", mc_backend="native")
    rewritten = model_for_pipeline(native_pipe)
    assert rewritten.params.get("mc_backend") == "native"
    fake = live_pipeline_defaults(model="dc_layered_v4", name="prod")
    monkeypatch.setattr(
        "fplt.automation.production.load_production_pipeline",
        lambda: fake,
    )
    assert load_base_pipeline("production").name == "prod"
    # Unknown mc_backend on the ModelSpec falls back to python.
    specs = load_model_specs()
    weird = specs["dc_layered_v4"].model_copy(
        update={"params": {**specs["dc_layered_v4"].params, "mc_backend": "cuda"}}
    )
    loaded = load_base_pipeline(
        "dc_layered_v4",
        specs={**specs, "dc_layered_v4": weird},
    )
    assert loaded.env.mc_backend == "python"


def test_measure_to_arm_metrics_schema() -> None:
    empty = measure_to_arm_metrics(empty_measure_metrics())
    ArmMetricsSchema.validate(empty)
    assert empty.is_empty()
    measure = MeasureMetricsSchema.validate(
        pl.DataFrame(
            {
                "arm": ["control", "control", "blend"],
                "window": ["2024-25:1-1"] * 3,
                "metric": ["brier_60", "brier_60", "mean_rps"],
                "value": [0.2, 0.25, 1.1],
                "se": [None, None, None],
                "n": [10, 4, 10],
                "stage": ["components", "components", "prediction"],
                "slice": ["overall", "source=fpl_api", "all"],
                "schema_version": [1, 1, 1],
            }
        )
    )
    arm_metrics = measure_to_arm_metrics(measure)
    ArmMetricsSchema.validate(arm_metrics)
    assert "slice" not in arm_metrics.columns
    # Prefer overall over source= for the same metric.
    control = arm_metrics.filter(
        (pl.col("arm") == "control") & (pl.col("metric") == "brier_60")
    )
    assert control.height == 1
    assert control["value"][0] == pytest.approx(0.2)
    # No preferred slices → fall back to all rows.
    other = MeasureMetricsSchema.validate(
        pl.DataFrame(
            {
                "arm": ["a"],
                "window": ["w"],
                "metric": ["m"],
                "value": [1.0],
                "se": [None],
                "n": [1],
                "stage": ["components"],
                "slice": ["position=MID"],
                "schema_version": [1],
            }
        )
    )
    fallback = measure_to_arm_metrics(other)
    assert fallback.height == 1


def test_run_rejects_bad_component_and_empty_arms(tmp_path: Path) -> None:
    data_dir = _league(tmp_path)
    with pytest.raises(ValueError, match="component must"):
        run_component_experiment(
            data_dir,
            base="dc_layered_v4",
            component="lineup",  # type: ignore[arg-type]
            arms=[{"model.components.minutes.params.logistic_blend": 0.8}],
            window="2024-25:1-1",
            n_sims_override=40,
        )
    with pytest.raises(ValueError, match="at least one"):
        run_component_experiment(
            data_dir,
            base="dc_layered_v4",
            component="minutes",
            arms=[],
            window=ParsedWindow("2024-25", 1, 1),
            n_sims_override=40,
        )


def test_run_component_experiment_bench(tmp_path: Path) -> None:
    data_dir = _league(tmp_path)
    phases = run_component_experiment_bench(data_dir, n_sims=40)
    assert "exp_component" in phases
    assert phases["exp_component"] >= 0.0


def test_score_component_arm_and_deadline_helpers(tmp_path: Path) -> None:
    """Non-minutes slot + ParsedWindow path still writes results."""
    from fplt.backtest import component_experiment as mod
    from fplt.types import GWKey

    data_dir = _league(tmp_path)
    result = run_component_experiment(
        data_dir,
        base="dc_layered_v4",
        component="score",
        arms=[{"model.components.score.params.xi": 0.001}],
        window=ParsedWindow("2024-25", 1, 1),
        code_sha=_SHA,
        lockfile_hash=_LOCK,
        build_id="score_build",
        experiment_id="exp_20261129T000000Z_score1",
        n_sims_override=40,
        host="test",
        submitted_by="test",
    )
    assert result.experiment_id.endswith("score1")
    key = GWKey("2024-25", 1)
    # Fallback deadline when gameweeks are empty.
    monkey_empty = MagicMock()
    monkey_empty.read.return_value = pl.DataFrame(
        {
            "season": pl.Series([], dtype=pl.Utf8),
            "gw": pl.Series([], dtype=pl.Int64),
            "deadline_time": pl.Series([], dtype=pl.Datetime("us", "UTC")),
            "available_at": pl.Series([], dtype=pl.Datetime("us", "UTC")),
        }
    )
    assert mod._deadline_utc(monkey_empty, key).year == 2024
    # Non-datetime deadline value falls back.
    monkey = MagicMock()
    monkey.read.return_value = pl.DataFrame(
        {
            "season": ["2024-25"],
            "gw": [1],
            "deadline_time": ["not-a-datetime"],
            "available_at": [datetime(2024, 8, 1, tzinfo=UTC)],
        }
    )
    assert mod._deadline_utc(monkey, key).year == 2024
    # Naive datetime gets UTC.
    naive = datetime(2024, 8, 16, 17, 0, 0)
    monkey_naive = MagicMock()
    monkey_naive.read.return_value = pl.DataFrame(
        {
            "season": ["2024-25"],
            "gw": [1],
            "deadline_time": [naive],
            "available_at": [naive],
        }
    )
    stamped = mod._deadline_utc(monkey_naive, key)
    assert stamped.tzinfo is not None
    # Aggregate / detail helpers on empty inputs.
    from fplt.schemas.component_eval import COMPONENT_EVAL_COLUMNS

    empty_detail = pl.DataFrame()
    assert mod._component_eval_from_detail(empty_detail, "a").is_empty()
    assert mod._aggregate_component_eval(
        pl.DataFrame(schema=COMPONENT_EVAL_COLUMNS)
    ).is_empty()
    assert mod._score_measures(
        component_summaries=[],
        forecast_scores=[],
        window_label="w",
    ).is_empty()
    # Weighted aggregate with null values / zero n.
    sparse = pl.DataFrame(
        {
            "component": ["minutes", "minutes"],
            "metric": ["brier_60", "brier_60"],
            "model_name": ["a", "a"],
            "slice": ["overall", "overall"],
            "value": [None, None],
            "n": [0, 0],
        },
        schema=COMPONENT_EVAL_COLUMNS,
    )
    agg = mod._aggregate_component_eval(sparse)
    assert agg.height == 1
    assert agg["value"][0] is None


def test_component_experiment_e2e_cache_and_schema(tmp_path: Path) -> None:
    """End-to-end on builder data: results schema + upstream cache hits."""
    data_dir = _league(tmp_path)
    patch = {"model.components.minutes.params.logistic_blend": 0.85}
    first = run_component_experiment(
        data_dir,
        base="dc_layered_v4",
        component="minutes",
        arms=[patch],
        window="2024-25:1-1",
        code_sha=_SHA,
        lockfile_hash=_LOCK,
        build_id="build_test",
        experiment_id="exp_20261129T000000Z_comp01",
        n_sims_override=40,
        host="test",
        submitted_by="test",
    )
    assert first.experiment_id == "exp_20261129T000000Z_comp01"
    assert CONTROL_ARM in first.arms
    assert len(first.arms) == 2
    MeasureMetricsSchema.validate(first.measure_metrics)
    ArmMetricsSchema.validate(first.arm_metrics)
    results = experiment_dir(data_dir, first.experiment_id) / "results"
    assert (results / "arm_metrics.parquet").is_file()
    on_disk = pl.read_parquet(results / "arm_metrics.parquet")
    ArmMetricsSchema.validate(on_disk)
    manifest = load_experiment_manifest(data_dir, first.experiment_id)
    assert manifest.flavour == "comparison"
    assert len(manifest.arms) == 2
    # Control miss-fills the cache; treatment reuses score/involvement/other.
    for slot in ("score", "involvement", "other"):
        assert first.cache.hits.get(slot, 0) >= 1
    assert first.cache.misses.get("minutes", 0) >= 1

    # Second run: every slot hits disk (no refit).
    second = run_component_experiment(
        data_dir,
        base="dc_layered_v4",
        component="minutes",
        arms=[patch],
        window="2024-25:1-1",
        code_sha=_SHA,
        lockfile_hash=_LOCK,
        build_id="build_test",
        experiment_id="exp_20261129T000000Z_comp02",
        n_sims_override=40,
        host="test",
        submitted_by="test",
    )
    assert sum(second.cache.misses.values()) == 0
    for slot in LAYERED_COMPONENT_SLOTS:
        assert second.cache.hits.get(slot, 0) >= 1


@pytest.mark.slow
def test_exp_component_cli_e2e(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """CLI surface on builder data writes an experiment and prints ids."""
    data_dir = _league(tmp_path)
    main(
        [
            "exp",
            "component",
            "--base",
            "dc_layered_v4",
            "--component",
            "minutes",
            "--arm",
            "model.components.minutes.params.logistic_blend=0.8",
            "--window",
            "2024-25:1-1",
            "--data-dir",
            str(data_dir),
            "--build-id",
            "cli_build",
            "--n-sims",
            "40",
        ]
    )
    out = capsys.readouterr().out
    assert "experiment_id=exp_" in out
    assert "cache_hits=" in out
    exp_ids = [
        path.name.removeprefix("exp=")
        for path in (data_dir / "experiments").glob("exp=*")
        if path.is_dir()
    ]
    assert exp_ids
    metrics_path = (
        data_dir
        / "experiments"
        / f"exp={exp_ids[0]}"
        / "results"
        / "arm_metrics.parquet"
    )
    assert metrics_path.is_file()
    ArmMetricsSchema.validate(pl.read_parquet(metrics_path))


def test_exp_component_handler_errors(tmp_path: Path) -> None:
    with pytest.raises(SystemExit) as exc:
        _exp(
            SimpleNamespace(
                exp_command="component",
                base="missing_model_zzz",
                component="minutes",
                arms=["model.components.minutes.params.logistic_blend=0.8"],
                window="2024-25:1-1",
                data_dir=tmp_path,
                build_id="x",
                n_sims=40,
            )
        )
    assert exc.value.code == 1
