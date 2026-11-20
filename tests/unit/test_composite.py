"""The layered predictor is a points model driven by its YAML components."""

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


def test_a_gameweek_is_scored_and_written(tmp_path: Path) -> None:
    spec = load_model_specs()["dc_layered_v1"].model_copy(
        update={"params": {"n_sims": 200, "own_goal_weight": 0.05, "p_assist": 0.75}}
    )
    store = build_league(DataStore(tmp_path / "league"))
    view = PITData(store).view(GWKey("2024-25", 1))
    model = CompositePredictor(spec)
    model.fit(view)
    frame = model.predict(view, [GWKey("2024-25", 1), GWKey("2024-25", 2)])
    checked = PredictionSchema.validate(frame)
    assert checked.height == 4
    assert checked["xpts"].is_not_null().all()
    path = write_predictions(tmp_path, spec, view.as_of, frame)
    assert path.name == "part.parquet"


def test_bad_layered_specs_are_rejected() -> None:
    with pytest.raises(ValueError, match="n_sims"):
        CompositePredictor(_spec(n_sims=0))
    with pytest.raises(ValueError, match="components"):
        CompositePredictor(ModelSpec(name="bare", kind="dc_layered"))


def _spec(**params: object) -> ModelSpec:
    base = load_model_specs()["dc_layered_v1"]
    return base.model_copy(update={"params": {**base.params, **params}})
