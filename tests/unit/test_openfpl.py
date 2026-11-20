"""OpenFPL adapter maps runner ids and refuses its training seasons."""

from __future__ import annotations

import sys
from pathlib import Path

import polars as pl
import pytest

import fplt.models.external.openfpl  # noqa: F401 — runs @register_model
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.external.openfpl import OpenFPLPredictor
from fplt.models.spec import ModelSpec, get_model, load_model_specs
from fplt.types import GWKey
from tests.builders import build_league

_ECHO = Path(__file__).parents[1] / "fixtures" / "openfpl_echo.py"
_AS_OF = GWKey("2024-25", 1)
_HORIZON = [GWKey("2024-25", 1)]


def _spec(**params: object) -> ModelSpec:
    base = load_model_specs()["openfpl_v1"]
    merged = dict(base.params)
    merged.update(params)
    return base.model_copy(update={"params": merged})


def _command(mode: str) -> list[str]:
    return [sys.executable, str(_ECHO), mode]


def _failing_command() -> list[str]:
    return [sys.executable, "-c", "raise SystemExit(1)"]


def _named(tmp_path: Path) -> tuple[DataStore, PITData]:
    store = build_league(DataStore(tmp_path))
    renamed = store.read("players").with_columns(
        pl.when(pl.col("player_key") == 100)
        .then(pl.lit("Alisson"))
        .otherwise(pl.lit("Saka"))
        .alias("web_name")
    )
    store.write("players", renamed)
    return store, PITData(store)


def test_registry_matches_the_yaml_spec() -> None:
    spec = load_model_specs()["openfpl_v1"]
    assert spec.kind == "openfpl"
    assert spec.params["compare_from"] == "2024-25"
    assert get_model("openfpl") is OpenFPLPredictor


def test_maps_runner_names_to_player_keys(tmp_path: Path) -> None:
    _store, data = _named(tmp_path)
    model = OpenFPLPredictor(_spec(command=_command("length")))
    view = data.view(_AS_OF)
    model.fit(view)
    got = model.predict(view, _HORIZON).sort("player_key")
    assert got["player_key"].to_list() == [100, 200]
    assert got["xpts"].to_list() == [7.0, 4.0]
    assert got["p_60"].to_list() == [None, None]


def test_one_name_covers_a_double_gameweek(tmp_path: Path) -> None:
    store, data = _named(tmp_path)
    extra = (
        store.read("fixtures")
        .filter(pl.col("fixture_id") == 1)
        .sort("available_at")
        .head(1)
        .with_columns(pl.lit(9).alias("fixture_id"))
    )
    store.write("fixtures", extra)
    model = OpenFPLPredictor(_spec(command=_command("length")))
    got = model.predict(data.view(_AS_OF), _HORIZON).filter(pl.col("player_key") == 100)
    assert sorted(got["fixture_id"].to_list()) == [1, 9]
    assert got["xpts"].to_list() == [7.0, 7.0]


def test_refuses_seasons_in_the_training_window(tmp_path: Path) -> None:
    _store, data = _named(tmp_path)
    model = OpenFPLPredictor(_spec(command=_failing_command()))
    view = data.view(_AS_OF)
    with pytest.raises(ValueError, match="2023-24"):
        model.predict(view, [GWKey("2023-24", 1)])


def test_ambiguous_web_name_raises(tmp_path: Path) -> None:
    store, data = _named(tmp_path)
    extra = (
        store.read("players")
        .filter(pl.col("player_key") == 100)
        .with_columns(
            pl.lit(300).alias("player_key"),
            pl.lit(30).alias("element_id"),
        )
    )
    store.write("players", extra)
    model = OpenFPLPredictor(_spec(command=_command("length")))
    with pytest.raises(ValueError, match="more than one player"):
        model.predict(data.view(_AS_OF), _HORIZON)


def test_unknown_player_id_raises(tmp_path: Path) -> None:
    _store, data = _named(tmp_path)
    model = OpenFPLPredictor(_spec(command=_command("unknown")))
    with pytest.raises(ValueError, match="not mapped"):
        model.predict(data.view(_AS_OF), _HORIZON)


def test_changed_player_id_raises(tmp_path: Path) -> None:
    _store, data = _named(tmp_path)
    model = OpenFPLPredictor(_spec(command=_command("rename")))
    with pytest.raises(ValueError, match="changed the player id"):
        model.predict(data.view(_AS_OF), _HORIZON)


def test_missing_runner_row_raises(tmp_path: Path) -> None:
    _store, data = _named(tmp_path)
    model = OpenFPLPredictor(_spec(command=_command("drop")))
    with pytest.raises(ValueError, match="one row per player-fixture"):
        model.predict(data.view(_AS_OF), _HORIZON)


def test_runner_failure_raises(tmp_path: Path) -> None:
    _store, data = _named(tmp_path)
    command = [
        sys.executable,
        "-c",
        "import sys; sys.stderr.write('boom'); sys.exit(1)",
    ]
    model = OpenFPLPredictor(_spec(command=command))
    with pytest.raises(RuntimeError, match="boom"):
        model.predict(data.view(_AS_OF), _HORIZON)


def test_non_json_output_raises(tmp_path: Path) -> None:
    _store, data = _named(tmp_path)
    model = OpenFPLPredictor(_spec(command=_command("badjson")))
    with pytest.raises(RuntimeError, match="did not return JSON"):
        model.predict(data.view(_AS_OF), _HORIZON)


def test_empty_horizon_does_not_call_the_runner(tmp_path: Path) -> None:
    _store, data = _named(tmp_path)
    model = OpenFPLPredictor(_spec(command=_failing_command()))
    got = model.predict(data.view(_AS_OF), [GWKey("2024-25", 3)])
    assert got.is_empty()


def test_missing_model_checkout_raises(tmp_path: Path) -> None:
    _store, data = _named(tmp_path)
    model = OpenFPLPredictor(_spec(model_dir=str(tmp_path / "openfpl")))
    with pytest.raises(RuntimeError, match="not found"):
        model.predict(data.view(_AS_OF), _HORIZON)


def test_unconfigured_runner_returns_zeros(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    model = OpenFPLPredictor(_spec())
    got = model.predict(PITData(store).view(_AS_OF), _HORIZON)
    assert got["xpts"].to_list() == [0.0, 0.0]
    assert set(got["player_key"].to_list()) == {100, 200}
