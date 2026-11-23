"""Unit tests for equal / position forecast combination."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest
import yaml

import fplt.models.baselines  # noqa: F401
import fplt.models.ensemble  # noqa: F401
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.spec import ModelSpec, get_model, load_model_specs
from fplt.types import GWKey
from tests.builders import build_league

_AS_OF = GWKey("2024-25", 1)
_HORIZON = [GWKey("2024-25", 1), GWKey("2024-25", 2)]


def _tiny_members(tmp_path: Path) -> dict[str, ModelSpec]:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "roll_a.yaml").write_text(
        yaml.safe_dump(
            {"name": "roll_a", "kind": "rolling_mean", "params": {"window": 3}}
        ),
        encoding="utf-8",
    )
    (tmp_path / "roll_b.yaml").write_text(
        yaml.safe_dump(
            {"name": "roll_b", "kind": "rolling_mean", "params": {"window": 5}}
        ),
        encoding="utf-8",
    )
    (tmp_path / "blend.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "blend_equal",
                "kind": "forecast_combine",
                "params": {"mode": "equal", "members": ["roll_a", "roll_b"]},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "blend_pos.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "blend_pos",
                "kind": "forecast_combine",
                "params": {
                    "mode": "position",
                    "members": ["roll_a", "roll_b"],
                    "position_weights": {
                        "MID": {"roll_a": 1.0, "roll_b": 0.0},
                        "FWD": {"roll_a": 0.0, "roll_b": 1.0},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    return load_model_specs(tmp_path)


def test_loader_expands_members_into_components(tmp_path: Path) -> None:
    specs = _tiny_members(tmp_path)
    blend = specs["blend_equal"]
    assert set(blend.components) == {"roll_a", "roll_b"}
    assert "members" not in blend.params
    assert blend.components["roll_a"].params["window"] == 3
    # model_id must move when a member changes.
    other = blend.model_copy(
        update={
            "components": {
                "roll_a": blend.components["roll_a"].model_copy(
                    update={"params": {"window": 99}}
                ),
                "roll_b": blend.components["roll_b"],
            }
        }
    )
    assert other.model_id != blend.model_id


def test_equal_blend_is_mean_of_members(tmp_path: Path) -> None:
    specs = _tiny_members(tmp_path / "specs")
    store = build_league(DataStore(tmp_path / "data"))
    view = PITData(store).view(_AS_OF)
    a = get_model("rolling_mean")(specs["roll_a"])
    b = get_model("rolling_mean")(specs["roll_b"])
    blend = get_model("forecast_combine")(specs["blend_equal"])
    a.fit(view)
    b.fit(view)
    blend.fit(view)
    pred_a = a.predict(view, _HORIZON).sort("player_key", "fixture_id")
    pred_b = b.predict(view, _HORIZON).sort("player_key", "fixture_id")
    pred = blend.predict(view, _HORIZON).sort("player_key", "fixture_id")
    expected = (pred_a["xpts"] + pred_b["xpts"]) / 2.0
    assert pred["xpts"].to_list() == pytest.approx(expected.to_list())


def test_position_blend_picks_member_by_position(tmp_path: Path) -> None:
    specs = _tiny_members(tmp_path / "specs")
    store = build_league(DataStore(tmp_path / "data"))
    view = PITData(store).view(_AS_OF)
    a = get_model("rolling_mean")(specs["roll_a"])
    b = get_model("rolling_mean")(specs["roll_b"])
    blend = get_model("forecast_combine")(specs["blend_pos"])
    a.fit(view)
    b.fit(view)
    blend.fit(view)
    pred_a = a.predict(view, _HORIZON)
    pred_b = b.predict(view, _HORIZON)
    pred = blend.predict(view, _HORIZON)
    # builders: player 100 = MID, player 200 = FWD
    mid = pred.filter(pl.col("player_key") == 100)["xpts"].to_list()
    fwd = pred.filter(pl.col("player_key") == 200)["xpts"].to_list()
    assert mid == pytest.approx(
        pred_a.filter(pl.col("player_key") == 100)["xpts"].to_list()
    )
    assert fwd == pytest.approx(
        pred_b.filter(pl.col("player_key") == 200)["xpts"].to_list()
    )


def test_save_load_round_trip(tmp_path: Path) -> None:
    specs = _tiny_members(tmp_path / "specs")
    store = build_league(DataStore(tmp_path / "data"))
    view = PITData(store).view(_AS_OF)
    model = get_model("forecast_combine")(specs["blend_equal"])
    model.fit(view)
    before = model.predict(view, _HORIZON).sort("player_key", "fixture_id")
    dest = tmp_path / "art"
    model.save(dest)
    loaded = get_model("forecast_combine").load(dest, specs["blend_equal"])
    after = loaded.predict(view, _HORIZON).sort("player_key", "fixture_id")
    assert after.equals(before)


def test_loader_rejects_unknown_member(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "blend.yaml").write_text(
        "name: blend\nkind: forecast_combine\nparams:\n  members: [missing]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unknown ModelSpec"):
        load_model_specs(tmp_path)


def test_loader_rejects_self_member(tmp_path: Path) -> None:
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "blend.yaml").write_text(
        "name: blend\nkind: forecast_combine\nparams:\n  members: [blend]\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="cannot list itself"):
        load_model_specs(tmp_path)


def test_rejects_bad_mode() -> None:
    spec = ModelSpec(
        name="bad",
        kind="forecast_combine",
        params={"mode": "magic"},
        components={
            "roll_a": ModelSpec(name="roll_a", kind="rolling_mean"),
        },
    )
    with pytest.raises(ValueError, match="mode must be one of"):
        get_model("forecast_combine")(spec)


def test_rejects_bad_position_weights() -> None:
    spec = ModelSpec(
        name="bad",
        kind="forecast_combine",
        params={
            "mode": "position",
            "position_weights": {"MID": {"nope": 1.0}},
        },
        components={
            "roll_a": ModelSpec(name="roll_a", kind="rolling_mean"),
        },
    )
    with pytest.raises(ValueError, match="unknown member"):
        get_model("forecast_combine")(spec)


def test_registered_production_yamls_expand() -> None:
    specs = load_model_specs()
    equal = specs["forecast_combine_equal_v1"]
    assert set(equal.components) == {"dc_layered_v3", "lgbm_direct_v1"}
    assert equal.params["mode"] == "equal"
    pos = specs["forecast_combine_pos_v1"]
    assert pos.params["mode"] == "position"
    assert "GKP" in pos.params["position_weights"]
