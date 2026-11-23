"""Feature / model / prediction factories appear in defs and materialise."""

from pathlib import Path

import dagster as dg
import polars as pl
import pytest
from dagster import RunConfig

from fplt.data.store import DataStore
from fplt.models.predictions import predictions_path
from fplt.models.spec import artifact_dir, load_model_specs
from fplt.orchestration import defs
from fplt.orchestration.assets import staged_ingest
from fplt.orchestration.factories import (
    LineageConfig,
    feature_asset_name,
    feature_assets,
    fit_or_load_model,
    model_asset_name,
    model_assets,
    points_predictor_specs,
    predictions_asset_name,
    predictions_assets,
    write_feature_or_existing,
    write_predictions_or_existing,
)
from fplt.settings import get_settings
from fplt.types import GWKey
from tests.builders import build_league

_SEASON = "2024-25"
_AS_OF = GWKey(_SEASON, 2)


def _use_data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()


def _lineage_config() -> LineageConfig:
    return LineageConfig(season=_SEASON, gw=2, horizon=1)


def _asset_by_name(name: str) -> dg.AssetsDefinition:
    for asset in [*feature_assets, *model_assets, *predictions_assets]:
        if asset.key.path[-1] == name:
            return asset
    raise AssertionError(f"missing asset {name}")


def test_defs_registers_factory_assets_and_lineage_job() -> None:
    graph = defs.get_repository_def().asset_graph
    names = {key.path[-1] for key in graph.get_all_asset_keys()}
    assert feature_asset_name("context", 1) in names
    assert feature_asset_name("rolling_form", 1) in names
    assert feature_asset_name("understat_share", 1) in names
    assert model_asset_name("rolling_mean_v1") in names
    assert model_asset_name("lgbm_direct_v1") in names
    assert predictions_asset_name("rolling_mean_v1") in names
    assert "dixon_coles_v1" not in {
        name.removeprefix("model__") for name in names if name.startswith("model__")
    }
    assert "lineage_features_models" in defs.get_repository_def().job_names
    assert {spec.name for spec in points_predictor_specs()} >= {
        "rolling_mean_v1",
        "ep_next_v1",
        "dc_layered_v1",
        "dc_layered_v2",
        "dc_layered_v3",
        "dc_layered_v4",
        "dc_layered_odds_v1",
        "lgbm_direct_v1",
        "openfpl_v1",
    }


def test_helpers_write_feature_model_and_predictions(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    from fplt.data.pit import PITData
    from fplt.orchestration.factories import ensure_features_loaded

    features = {
        f"{item.name}@{item.version}": item for item in ensure_features_loaded()
    }
    view = PITData(store).view(_AS_OF)
    feature = features["context@1"]
    feature_path = write_feature_or_existing(view, feature, tmp_path)
    assert feature_path.exists()
    assert write_feature_or_existing(view, feature, tmp_path) == feature_path

    spec = load_model_specs()["rolling_mean_v1"]
    model = fit_or_load_model(tmp_path, spec, _AS_OF, view)
    predicted = model.predict(view, [_AS_OF])
    path = write_predictions_or_existing(tmp_path, spec, _AS_OF, predicted)
    assert path == predictions_path(tmp_path, spec, _AS_OF)
    assert path.exists()
    assert write_predictions_or_existing(tmp_path, spec, _AS_OF, predicted) == path
    assert (artifact_dir(tmp_path, spec, _AS_OF) / "spec.json").exists()


def test_materialize_rolling_mean_lineage(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _use_data_dir(tmp_path, monkeypatch)
    build_league(DataStore(tmp_path))
    config = _lineage_config()
    model_name = model_asset_name("rolling_mean_v1")
    predictions_name = predictions_asset_name("rolling_mean_v1")
    feature_name = feature_asset_name("context", 1)
    result = dg.materialize(
        [
            staged_ingest,
            _asset_by_name(feature_name),
            _asset_by_name(model_name),
            _asset_by_name(predictions_name),
        ],
        run_config=RunConfig(
            ops={
                feature_name: config,
                model_name: config,
                predictions_name: config,
            }
        ),
    )
    assert result.success
    spec = load_model_specs()["rolling_mean_v1"]
    pred_path = predictions_path(tmp_path, spec, _AS_OF)
    assert pred_path.exists()
    frame = pl.read_parquet(pred_path)
    assert frame.height == 2
    assert set(frame["target_gw"].to_list()) == {2}
    assert (artifact_dir(tmp_path, spec, _AS_OF) / "spec.json").exists()
    get_settings.cache_clear()
