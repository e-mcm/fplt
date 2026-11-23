"""The layered predictor is a points model driven by its YAML components."""

import logging
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import fplt.models.composite  # noqa: F401 — registers the predictor
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.composite import CompositePredictor, _sample_minutes_crn
from fplt.models.distributions import MINUTE_BINS, MinutesDist
from fplt.models.predictions import write_predictions
from fplt.models.spec import ModelSpec, get_model, load_model_specs
from fplt.schemas.predictions import PredictionSchema
from fplt.types import GWKey
from fplt.utils.rng import paired_eval_seed, seed_for
from tests.builders import build_league


def test_the_spec_builds_the_layered_model() -> None:
    spec = load_model_specs()["dc_layered_v1"]
    assert get_model("dc_layered") is CompositePredictor
    assert spec.components["score"].kind == "dixon_coles"
    assert spec.params["n_sims"] == 5000


def test_dc_layered_odds_v1_wires_odds_score_component() -> None:
    spec = load_model_specs()["dc_layered_odds_v1"]
    assert spec.kind == "dc_layered"
    assert spec.components["score"].name == "dixon_coles_odds_v1"
    assert spec.components["score"].params["market_prior_strength"] == 1.0
    vanilla = load_model_specs()["dc_layered_v4"]
    assert "market_prior_strength" not in vanilla.components["score"].params


def test_dc_layered_history_v1_wires_history_score_component() -> None:
    spec = load_model_specs()["dc_layered_history_v1"]
    assert spec.kind == "dc_layered"
    assert spec.components["score"].name == "dixon_coles_history_v1"
    assert spec.components["score"].params["market_prior_source"] == "history"
    assert spec.components["score"].params["market_prior_strength"] == 1.0
    vanilla = load_model_specs()["dc_layered_v4"]
    assert "market_prior_source" not in vanilla.components["score"].params


def test_dc_layered_v2_keeps_mean_wire_identity() -> None:
    """Frozen pre-4d715b5 mean-wire compose (FDB-226)."""
    spec = load_model_specs()["dc_layered_v2"]
    assert spec.kind == "dc_layered"
    assert spec.params["n_sims"] == 10000
    assert spec.params["hard_starters"] is True
    assert spec.components["other"].name == "other_player_v1"
    assert "defcon_mode" not in spec.components["other"].params
    assert spec.model_id == "b12aad1575ff0d34"


def test_dc_layered_v4_wires_improved_components() -> None:
    spec = load_model_specs()["dc_layered_v4"]
    assert spec.kind == "dc_layered"
    assert spec.params["n_sims"] == 10000
    assert spec.params["cs_red_void_prob"] == pytest.approx(0.02)
    assert spec.params["conceded_mode"] == "played"
    assert spec.params["sample_other"] is True
    assert spec.params["hard_starters"] is True
    assert spec.params["target_starters"] == 11
    assert spec.components["score"].name == "dixon_coles_cs_v1"
    assert spec.components["score"].params["market_cs_strength"] == 0.5
    assert spec.components["minutes"].kind == "empirical_minutes_v2"
    assert spec.components["involvement"].kind == "involvement_xg"
    assert spec.components["other"].kind == "other_player"
    assert spec.components["other"].name == "other_player_defcon_rate_v1"
    assert spec.components["other"].params["defcon_mode"] == "rate"
    model = CompositePredictor(spec)
    assert model._n_sims == 10000
    assert model._cs_red_void_prob == pytest.approx(0.02)
    assert model._conceded_mode == "played"
    assert model._sample_other is True
    assert model._hard_starters is True
    assert model._target_starters == 11


def test_dc_layered_v4_scores_a_gameweek(tmp_path: Path) -> None:
    spec = load_model_specs()["dc_layered_v4"].model_copy(
        update={"params": {**load_model_specs()["dc_layered_v4"].params, "n_sims": 80}}
    )
    store = build_league(DataStore(tmp_path / "league"))
    view = PITData(store).view(GWKey("2024-25", 1))
    model = CompositePredictor(spec)
    model.fit(view)
    frame = model.predict(view, [GWKey("2024-25", 1)])
    checked = PredictionSchema.validate(frame)
    assert checked.height == 2
    assert checked["xpts"].is_not_null().all()
    for col in ("p_played", "p_60", "p_goal", "p_assist", "p_cs", "p_haul"):
        assert checked[col].is_not_null().all()
        assert checked[col].min() >= 0.0
        assert checked[col].max() <= 1.0


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
    with pytest.raises(ValueError, match="hard_starters"):
        CompositePredictor(_spec(hard_starters="yes"))
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


def test_paired_eval_predictions_are_bit_identical(tmp_path: Path) -> None:
    spec = _spec(n_sims=120)
    store = build_league(DataStore(tmp_path / "league"))
    view = PITData(store).view(GWKey("2024-25", 1))
    model = CompositePredictor(spec)
    model.fit(view)
    with paired_eval_seed(17):
        first = model.predict(view, [GWKey("2024-25", 1)]).sort(
            "player_key", "fixture_id"
        )
        second = model.predict(view, [GWKey("2024-25", 1)]).sort(
            "player_key", "fixture_id"
        )
    cols = [
        "xpts",
        "xpts_sd",
        "p_played",
        "p_60",
        "p_goal",
        "p_assist",
        "p_cs",
        "p_haul",
    ]
    assert first.select(cols).equals(second.select(cols))


def test_paired_eval_seed_pairs_across_model_ids(tmp_path: Path) -> None:
    base = _spec(n_sims=120)
    left_spec = base.model_copy(update={"name": "crn_left"})
    right_spec = base.model_copy(update={"name": "crn_right"})
    assert left_spec.model_id != right_spec.model_id
    as_of = GWKey("2024-25", 1)
    # Production streams diverge by model_id; CRN pairing must not.
    assert list(seed_for(left_spec.model_id, as_of, 1).random(4)) != list(
        seed_for(right_spec.model_id, as_of, 1).random(4)
    )
    store = build_league(DataStore(tmp_path / "league"))
    view = PITData(store).view(as_of)
    left = CompositePredictor(left_spec)
    right = CompositePredictor(right_spec)
    left.fit(view)
    right.fit(view)
    with paired_eval_seed(23):
        paired_left = left.predict(view, [as_of]).sort("player_key", "fixture_id")
        paired_right = right.predict(view, [as_of]).sort("player_key", "fixture_id")
    cols = [
        "xpts",
        "xpts_sd",
        "p_played",
        "p_60",
        "p_goal",
        "p_assist",
        "p_cs",
        "p_haul",
    ]
    assert paired_left.select(cols).equals(paired_right.select(cols))


def test_crn_minutes_ignore_an_extra_player() -> None:
    as_of = GWKey("2024-25", 1)
    fixture_id = 7
    seed = 5
    keys_small = pl.DataFrame({"player_key": [100, 200], "fixture_id": [7, 7]})
    keys_extra = pl.DataFrame({"player_key": [100, 150, 200], "fixture_id": [7, 7, 7]})
    pmf_small = np.array(
        [[0.1, 0.1, 0.1, 0.1, 0.2, 0.2, 0.2], [0.2, 0.2, 0.2, 0.1, 0.1, 0.1, 0.1]],
        dtype=np.float64,
    )
    pmf_extra = np.vstack(
        [
            pmf_small[0],
            [0.05, 0.05, 0.1, 0.1, 0.2, 0.25, 0.25],
            pmf_small[1],
        ]
    )
    small = MinutesDist(keys_small, MINUTE_BINS, pmf_small)
    extra = MinutesDist(keys_extra, MINUTE_BINS, pmf_extra)
    drawn_small = _sample_minutes_crn(small, seed, as_of, fixture_id, 40)
    drawn_extra = _sample_minutes_crn(extra, seed, as_of, fixture_id, 40)
    assert np.array_equal(drawn_small[0], drawn_extra[0])
    assert np.array_equal(drawn_small[1], drawn_extra[2])


def _spec(**params: object) -> ModelSpec:
    base = load_model_specs()["dc_layered_v1"]
    return base.model_copy(update={"params": {**base.params, **params}})
