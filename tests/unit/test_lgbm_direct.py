"""Direct LightGBM shares walk-forward slices with the layered model."""

from pathlib import Path

import numpy as np
import polars as pl
import pytest

import fplt.models.composite  # noqa: F401 — runs @register_model
import fplt.models.direct  # noqa: F401 — runs @register_model
from fplt.backtest.walk_forward import walk_forward
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.features.form import FEATURE_COLUMNS, training_frame
from fplt.features.registry import build_feature_matrix, get_feature
from fplt.models.direct import LGBMDirectPredictor
from fplt.models.direct.lgbm import _fit_binary_head, _predict_binary
from fplt.models.spec import get_model, load_model_specs
from fplt.types import GWKey
from tests.builders import build_league

_AS_OF = GWKey("2024-25", 2)
_KEYS = [
    "as_of_season",
    "as_of_gw",
    "target_season",
    "target_gw",
    "player_key",
    "fixture_id",
]


def test_registry_matches_the_yaml_spec() -> None:
    specs = load_model_specs()
    assert get_model(specs["lgbm_direct_v1"].kind) is LGBMDirectPredictor
    assert specs["lgbm_direct_v1"].features == [
        "context@1",
        "rolling_form@1",
        "opponent_strength@1",
    ]


def test_feature_matrix_carries_base_columns(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    view = PITData(store).view(_AS_OF)
    matrix = build_feature_matrix(
        view,
        ["context@1", "rolling_form@1", "opponent_strength@1"],
        [_AS_OF],
    )
    for column in FEATURE_COLUMNS:
        assert column in matrix.columns
    points = matrix.filter(pl.col("player_key") == 100)["total_points_rolling_5"]
    assert points.item() == 8.0
    assert get_feature("context", 1).grain == "player_fixture"


def test_training_frame_shifts_the_label_row(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    view = PITData(store).view(_AS_OF)
    train = training_frame(view)
    assert train.height == 2
    assert set(train.columns) == set(FEATURE_COLUMNS) | {
        "total_points",
        "minutes",
        "kickoff_time",
    }
    assert train["minutes"].null_count() == 0


def test_fit_predict_save_and_load(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path / "league"))
    spec = load_model_specs()["lgbm_direct_v1"]
    view = PITData(store).view(_AS_OF)
    model = LGBMDirectPredictor(spec)
    model.fit(view)
    assert model._booster is not None
    predicted = model.predict(view, [_AS_OF]).sort("player_key")
    assert predicted.height == 2
    assert predicted["xpts"].null_count() == 0
    dest = tmp_path / "artifact"
    model.save(dest)
    loaded = LGBMDirectPredictor.load(dest, spec)
    got = loaded.predict(view, [_AS_OF]).sort("player_key")
    assert got.equals(predicted)
    with pytest.raises(FileExistsError):
        model.save(dest)


def test_lgbm_probs_emits_bounded_event_heads(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path / "league"))
    specs = load_model_specs()
    spec = specs["lgbm_probs_v1"]
    assert spec.params.get("emit_probs") is True
    assert spec.kind == "lgbm_direct"
    view = PITData(store).view(_AS_OF)
    model = LGBMDirectPredictor(spec)
    model.fit(view)
    predicted = model.predict(view, [_AS_OF])
    assert predicted.height == 2
    for column in ("p_haul", "p_60", "p_played"):
        assert predicted[column].null_count() == 0
        values = predicted[column].to_list()
        assert all(0.0 <= float(value) <= 1.0 for value in values)
    # Tiny synthetic league is single-class on haul / 60+ → constant priors.
    assert predicted["p_haul"].to_list() == [0.0, 0.0]
    assert predicted["p_60"].to_list() == [1.0, 1.0]
    assert predicted["p_played"].to_list() == [1.0, 1.0]
    dest = tmp_path / "probs-artifact"
    model.save(dest)
    loaded = LGBMDirectPredictor.load(dest, spec)
    got = loaded.predict(view, [_AS_OF]).sort("player_key")
    assert got.select("p_haul", "p_60", "p_played").equals(
        predicted.sort("player_key").select("p_haul", "p_60", "p_played")
    )


def test_lgbm_direct_v1_still_leaves_probs_null(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path / "league"))
    spec = load_model_specs()["lgbm_direct_v1"]
    view = PITData(store).view(_AS_OF)
    model = LGBMDirectPredictor(spec)
    model.fit(view)
    predicted = model.predict(view, [_AS_OF])
    assert predicted["p_haul"].null_count() == predicted.height
    assert predicted["p_60"].null_count() == predicted.height
    assert predicted["p_played"].null_count() == predicted.height


def test_binary_head_trains_when_both_classes_present() -> None:
    """Regression: single-class prior path must not be the only covered branch."""
    rng = np.random.default_rng(0)
    n_features = len(FEATURE_COLUMNS)
    labels = np.array([0.0] * 20 + [1.0] * 20)
    matrix = rng.normal(size=(40, n_features))
    # Make class linearly separable on the first feature so the booster splits.
    matrix[:, 0] = labels * 3.0 + rng.normal(scale=0.05, size=40)
    weights = np.ones(40, dtype=float)
    spec = load_model_specs()["lgbm_probs_v1"]
    # Tiny trees so the synthetic signal is usable under parent YAML knobs.
    soft = spec.model_copy(
        update={
            "params": {
                **dict(spec.params),
                "num_leaves": 8,
                "min_child_samples": 2,
                "num_boost_round": 20,
            }
        }
    )
    booster, prior = _fit_binary_head(matrix, labels, weights, soft)
    assert booster is not None
    assert 0.4 <= prior <= 0.6
    scored = _predict_binary(booster, prior, matrix, matrix.shape[0])
    assert scored.shape == (40,)
    assert float(scored.min()) >= 0.0
    assert float(scored.max()) <= 1.0
    assert scored[labels == 1].mean() > scored[labels == 0].mean()


def test_layered_and_direct_share_walk_forward_slices(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    pit = PITData(store)
    as_ofs = [GWKey("2024-25", 1), GWKey("2024-25", 2)]
    specs = load_model_specs()
    layered = walk_forward(
        get_model("dc_layered")(specs["dc_layered_v1"]),
        pit,
        as_ofs,
        steps=2,
    )
    direct = walk_forward(
        get_model("lgbm_direct")(specs["lgbm_direct_v1"]),
        pit,
        as_ofs,
        steps=2,
    )
    left = layered.select(_KEYS).sort(_KEYS)
    right = direct.select(_KEYS).sort(_KEYS)
    assert left.equals(right)
    assert not left.is_empty()
