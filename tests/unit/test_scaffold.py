"""Scaffold contracts: legacy schemas, identity, and seeded randomness."""

from fplt.models.spec import ModelSpec, TrainWindow
from fplt.schemas import FixtureSchema
from fplt.types import GWKey
from fplt.utils.rng import seed_for


def test_legacy_schemas_remain_importable() -> None:
    assert FixtureSchema.__name__ == "FixtureSchema"


def test_model_id_changes_when_params_change() -> None:
    left = ModelSpec(name="baseline", kind="rolling_mean", params={"window": 5})
    right = ModelSpec(name="baseline", kind="rolling_mean", params={"window": 6})
    assert left.model_id == left.model_id
    assert left.model_id != right.model_id


def test_model_id_covers_nested_components() -> None:
    window = TrainWindow(seasons_back=3, time_decay=0.01)
    base = ModelSpec(name="dc", kind="dixon_coles", train_window=window)
    changed = ModelSpec(
        name="dc",
        kind="dixon_coles",
        train_window=TrainWindow(seasons_back=3, time_decay=0.02),
    )
    assert base.model_id != changed.model_id


def test_seed_for_is_stable_and_independent_per_fixture() -> None:
    as_of = GWKey("2024-25", 10)
    first = seed_for("abc", as_of, 1).random(4)
    again = seed_for("abc", as_of, 1).random(4)
    other = seed_for("abc", as_of, 2).random(4)
    assert list(first) == list(again)
    assert list(first) != list(other)


def test_gw_key_rejects_out_of_range() -> None:
    try:
        GWKey("2024-25", 0)
    except ValueError:
        return
    raise AssertionError("expected ValueError")
