"""Every registered predictor honours the schema, the horizon, and the cutoff."""

from datetime import timedelta
from pathlib import Path

import polars as pl
import pytest

import fplt.models.baselines  # noqa: F401 — runs @register_model
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.spec import get_model, load_model_specs, registered_kinds
from fplt.models.universe import player_fixtures
from fplt.schemas.predictions import PredictionSchema
from fplt.types import GWKey
from tests.builders import build_league

_AS_OF = GWKey("2024-25", 1)
_HORIZON = [GWKey("2024-25", 1), GWKey("2024-25", 2)]
_KEYS = ["target_season", "target_gw", "player_key", "fixture_id"]


def _spec(kind: str):
    matches = [spec for spec in load_model_specs().values() if spec.kind == kind]
    if len(matches) != 1:
        raise AssertionError(f"{kind} needs one YAML spec, found {len(matches)}")
    return matches[0]


def _predict(kind: str, store: DataStore) -> pl.DataFrame:
    model = get_model(kind)(_spec(kind))
    view = PITData(store).view(_AS_OF)
    model.fit(view)
    return model.predict(view, _HORIZON)


def _sorted(frame: pl.DataFrame) -> pl.DataFrame:
    return frame.sort(_KEYS)


@pytest.mark.parametrize("kind", registered_kinds())
def test_predictions_match_the_schema_and_cover_the_horizon(
    kind: str, tmp_path: Path
) -> None:
    store = build_league(DataStore(tmp_path))
    frame = _predict(kind, store)
    checked = PredictionSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    view = PITData(store).view(_AS_OF)
    expected = player_fixtures(view, _HORIZON).select(
        pl.col("season").alias("target_season"),
        pl.col("gw").alias("target_gw"),
        "player_key",
        "fixture_id",
    )
    got = checked.select(_KEYS)
    assert _sorted(got).equals(_sorted(expected))
    assert not got.is_empty()


@pytest.mark.parametrize("kind", registered_kinds())
def test_predictions_are_deterministic(kind: str, tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    first = _sorted(_predict(kind, store))
    second = _sorted(_predict(kind, store))
    assert first.equals(second)


@pytest.mark.parametrize("kind", registered_kinds())
def test_save_and_load_round_trip(kind: str, tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path / "league"))
    spec = _spec(kind)
    model = get_model(kind)(spec)
    view = PITData(store).view(_AS_OF)
    model.fit(view)
    expected = _sorted(model.predict(view, _HORIZON))
    dest = tmp_path / "artifact"
    model.save(dest)
    loaded = get_model(kind).load(dest, spec)
    assert _sorted(loaded.predict(view, _HORIZON)).equals(expected)
    with pytest.raises(FileExistsError):
        model.save(dest)


@pytest.mark.parametrize("kind", registered_kinds())
def test_rows_after_the_cutoff_do_not_change_predictions(
    kind: str, tmp_path: Path
) -> None:
    store = build_league(DataStore(tmp_path))
    cutoff = store.deadline(_AS_OF)
    later = cutoff + timedelta(days=1)
    status = store.read("player_status").with_columns(
        pl.lit(later).cast(pl.Datetime("us", "UTC")).alias("available_at"),
        (pl.col("ep_next") + 5).alias("ep_next"),
    )
    store.write("player_status", status)
    before = _sorted(_predict(kind, store))
    future_points = _future_values(
        store, "player_fixture_stats", "total_points", cutoff
    )
    _bump(store, "player_fixture_stats", "total_points", cutoff)
    _bump(store, "player_status", "ep_next", cutoff)
    assert _future_values(store, "player_fixture_stats", "total_points", cutoff) != (
        future_points
    )
    after = _sorted(_predict(kind, store))
    assert after.equals(before)


def _bump(store: DataStore, name: str, column: str, cutoff: object) -> None:
    frame = store.read(name)
    bumped = frame.with_columns(
        pl.when(pl.col("available_at") > cutoff)
        .then(pl.col(column) + 1)
        .otherwise(pl.col(column))
        .alias(column)
    )
    store.write(name, bumped)


def _future_values(store: DataStore, name: str, column: str, cutoff: object) -> list:
    frame = store.read(name).filter(pl.col("available_at") > cutoff)
    assert not frame.is_empty()
    return frame[column].sort().to_list()
