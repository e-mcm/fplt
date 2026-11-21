"""The layered predictor is a points model driven by its YAML components."""

import logging
from pathlib import Path

import pytest

import fplt.models.composite  # noqa: F401 — registers the predictor
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.composite import CompositePredictor
from fplt.models.predictions import write_predictions
from fplt.models.spec import ModelSpec, get_model, load_model_specs
from fplt.schemas.predictions import PredictionSchema
from fplt.types import GWKey
from tests.builders import build_league


def test_the_spec_builds_the_layered_model() -> None:
    spec = load_model_specs()["dc_layered_v1"]
    assert get_model("dc_layered") is CompositePredictor
    assert spec.components["score"].kind == "dixon_coles"
    assert spec.params["n_sims"] == 5000


def test_dc_layered_v2_wires_improved_components() -> None:
    spec = load_model_specs()["dc_layered_v2"]
    assert spec.kind == "dc_layered"
    assert spec.params["n_sims"] == 10000
    assert spec.params["cs_red_void_prob"] == pytest.approx(0.02)
    assert spec.params["conceded_mode"] == "played"
    assert spec.params["sample_other"] is True
    assert spec.components["score"].name == "dixon_coles_v2"
    assert spec.components["minutes"].kind == "empirical_minutes_v2"
    assert spec.components["involvement"].kind == "involvement_xg"
    assert spec.components["other"].kind == "other_player"
    model = CompositePredictor(spec)
    assert model._n_sims == 10000
    assert model._cs_red_void_prob == pytest.approx(0.02)
    assert model._conceded_mode == "played"
    assert model._sample_other is True


def test_dc_layered_v2_scores_a_gameweek(tmp_path: Path) -> None:
    spec = load_model_specs()["dc_layered_v2"].model_copy(
        update={"params": {**load_model_specs()["dc_layered_v2"].params, "n_sims": 80}}
    )
    store = build_league(DataStore(tmp_path / "league"))
    view = PITData(store).view(GWKey("2024-25", 1))
    model = CompositePredictor(spec)
    model.fit(view)
    frame = model.predict(view, [GWKey("2024-25", 1)])
    checked = PredictionSchema.validate(frame)
    assert checked.height == 2
    assert checked["xpts"].is_not_null().all()


def test_a_gameweek_is_scored_and_written(
    tmp_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    spec = load_model_specs()["dc_layered_v1"].model_copy(
        update={"params": {"n_sims": 200, "own_goal_weight": 0.05, "p_assist": 0.75}}
    )
    store = build_league(DataStore(tmp_path / "league"))
    view = PITData(store).view(GWKey("2024-25", 1))
    model = CompositePredictor(spec)
    model.fit(view)
    with caplog.at_level(logging.INFO, logger="fplt.models.composite"):
        frame = model.predict(view, [GWKey("2024-25", 1), GWKey("2024-25", 2)])
    assert "fixture" in caplog.text
    assert "(1/" in caplog.text
    checked = PredictionSchema.validate(frame)
    assert checked.height == 4
    assert checked["xpts"].is_not_null().all()
    path = write_predictions(tmp_path, spec, view.as_of, frame)
    assert path.name == "part.parquet"


def test_bad_layered_specs_are_rejected() -> None:
    with pytest.raises(ValueError, match="n_sims"):
        CompositePredictor(_spec(n_sims=0))
    with pytest.raises(ValueError, match="cs_red_void_prob"):
        CompositePredictor(_spec(cs_red_void_prob=2.0))
    with pytest.raises(ValueError, match="conceded_mode"):
        CompositePredictor(_spec(conceded_mode="half"))
    with pytest.raises(ValueError, match="sample_other"):
        CompositePredictor(_spec(sample_other="yes"))
    with pytest.raises(ValueError, match="components"):
        CompositePredictor(ModelSpec(name="bare", kind="dc_layered"))


def test_fidelity_params_are_accepted_on_the_layered_spec() -> None:
    model = CompositePredictor(
        _spec(
            n_sims=50,
            cs_red_void_prob=0.02,
            conceded_mode="played",
            sample_other=True,
        )
    )
    assert model._cs_red_void_prob == pytest.approx(0.02)
    assert model._conceded_mode == "played"
    assert model._sample_other is True


def _spec(**params: object) -> ModelSpec:
    base = load_model_specs()["dc_layered_v1"]
    return base.model_copy(update={"params": {**base.params, **params}})
