"""Rolling-mean and ep_next baselines stay inside the point-in-time view."""

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.baselines import EpNextPredictor, RollingMeanPredictor
from fplt.models.predictions import write_predictions
from fplt.models.spec import ModelSpec, get_model, load_model_specs
from fplt.types import GWKey
from tests.builders import build_league

_AS_OF = GWKey("2024-25", 2)
_LATER = datetime(2024, 8, 23, 16, tzinfo=UTC)


def _view(tmp_path: Path) -> tuple[DataStore, PITData]:
    store = build_league(DataStore(tmp_path))
    return store, PITData(store)


def _spec(kind: str, **params: object) -> ModelSpec:
    return ModelSpec(name=kind, kind=kind, params=params)


def test_registry_matches_the_yaml_specs() -> None:
    import fplt.models.baselines  # noqa: F401

    specs = load_model_specs()
    assert get_model(specs["rolling_mean_v1"].kind) is RollingMeanPredictor
    assert get_model(specs["ep_next_v1"].kind) is EpNextPredictor


def test_rolling_mean_uses_only_visible_appearances(tmp_path: Path) -> None:
    store, data = _view(tmp_path)
    store.write(
        "player_fixture_stats",
        store.read("player_fixture_stats")
        .head(1)
        .with_columns(
            pl.lit(0).alias("total_points"),
            pl.lit(datetime(2024, 8, 10, tzinfo=UTC)).alias("available_at"),
        ),
    )
    view = data.view(_AS_OF)
    wide = RollingMeanPredictor(_spec("rolling_mean", window=5))
    wide.fit(view)
    got = wide.predict(view, [_AS_OF]).sort("player_key")
    assert got["xpts"].to_list() == [4.0, 2.0]
    narrow = RollingMeanPredictor(_spec("rolling_mean", window=1))
    narrow.fit(view)
    latest = narrow.predict(view, [_AS_OF]).filter(pl.col("player_key") == 100)
    assert latest["xpts"].to_list() == [8.0]
    assert wide.predict(view, [GWKey("2024-25", 3)]).is_empty()


def test_double_gameweek_emits_two_rows(tmp_path: Path) -> None:
    store, data = _view(tmp_path)
    extra = (
        store.read("fixtures")
        .filter(pl.col("fixture_id") == 2)
        .with_columns(
            pl.lit(3).alias("fixture_id"),
            pl.lit(3).alias("team_h_key"),
            pl.lit(7).alias("team_a_key"),
        )
    )
    store.write("fixtures", extra)
    view = data.view(_AS_OF)
    model = RollingMeanPredictor(_spec("rolling_mean", window=5))
    model.fit(view)
    rows = model.predict(view, [_AS_OF]).filter(pl.col("player_key") == 100)
    assert sorted(rows["fixture_id"].to_list()) == [2, 3]


def test_ep_next_skips_vaastav_snapshots(tmp_path: Path) -> None:
    store, data = _view(tmp_path)
    view = data.view(_AS_OF)
    model = EpNextPredictor(_spec("ep_next"))
    model.fit(view)
    got = model.predict(view, [_AS_OF]).sort("player_key")
    assert got["xpts"].to_list() == [6.5, 4.0]
    assert got["xpts_sd"].to_list() == [None, None]

    degraded = store.read("player_status").with_columns(
        pl.lit("vaastav").alias("source"),
        pl.lit(_LATER).alias("available_at"),
    )
    store.write("player_status", degraded)
    hidden = data.view(_AS_OF)
    model.fit(hidden)
    assert model.predict(hidden, [_AS_OF]).is_empty()


def test_ep_next_skips_vaastav_fixtures(tmp_path: Path) -> None:
    store, data = _view(tmp_path)
    later = (
        store.read("fixtures")
        .filter(pl.col("fixture_id") == 2)
        .with_columns(
            pl.lit("vaastav").alias("source"),
            pl.lit(_LATER).alias("available_at"),
        )
    )
    store.write("fixtures", later)
    view = data.view(_AS_OF)
    model = EpNextPredictor(_spec("ep_next"))
    model.fit(view)
    assert model.predict(view, [_AS_OF]).is_empty()


def test_save_load_and_prediction_write_once(tmp_path: Path) -> None:
    _store, data = _view(tmp_path)
    view = data.view(_AS_OF)
    spec = _spec("rolling_mean", window=5)
    model = RollingMeanPredictor(spec)
    model.fit(view)
    expected = model.predict(view, [_AS_OF])
    artifact = tmp_path / "artifact"
    model.save(artifact)
    restored = RollingMeanPredictor.load(artifact, spec)
    keys = ["player_key", "fixture_id"]
    got = restored.predict(view, [_AS_OF]).sort(keys)
    assert got.equals(expected.sort(keys))
    with pytest.raises(FileExistsError):
        model.save(artifact)
    path = write_predictions(tmp_path, spec, _AS_OF, expected)
    assert path.name == "part.parquet"
    assert "predictions" in path.parts
    with pytest.raises(FileExistsError):
        write_predictions(tmp_path, spec, _AS_OF, expected)
