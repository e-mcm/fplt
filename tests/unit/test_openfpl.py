"""OpenFPL adapter maps runner ids and refuses its training seasons."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import polars as pl
import pytest

import fplt.models.external.openfpl  # noqa: F401 — runs @register_model
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.external.openfpl import DEFAULT_MODEL_DIR, OpenFPLPredictor
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
    assert spec.params["model_dir"] == DEFAULT_MODEL_DIR
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


def test_shared_web_name_keeps_separate_player_keys(tmp_path: Path) -> None:
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
    got = model.predict(data.view(_AS_OF), _HORIZON).sort("player_key")
    assert got["player_key"].to_list() == [100, 200, 300]
    assert got["xpts"].to_list() == [7.0, 4.0, 7.0]


def test_unknown_player_id_raises(tmp_path: Path) -> None:
    _store, data = _named(tmp_path)
    model = OpenFPLPredictor(_spec(command=_command("unknown")))
    with pytest.raises(ValueError, match="changed the player id"):
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
    with pytest.raises(FileNotFoundError, match="not found"):
        model.predict(data.view(_AS_OF), _HORIZON)


def test_payload_includes_openfpl_feature_matrix(tmp_path: Path) -> None:
    _store, data = _named(tmp_path)
    model = OpenFPLPredictor(_spec(command=_command("features")))
    got = model.predict(data.view(_AS_OF), _HORIZON).sort("player_key")
    assert got["xpts"].to_list() == [7.0, 7.0]


def test_empty_model_dir_raises(tmp_path: Path) -> None:
    store = build_league(DataStore(tmp_path))
    model = OpenFPLPredictor(_spec(model_dir=""))
    with pytest.raises(RuntimeError, match="model_dir is not set"):
        model.predict(PITData(store).view(_AS_OF), _HORIZON)


def test_custom_model_dir_must_already_contain_the_ensemble(tmp_path: Path) -> None:
    from fplt.models.external import openfpl as mod

    missing = tmp_path / "elsewhere"
    with pytest.raises(FileNotFoundError, match="not found"):
        mod._resolve_model_dir(_spec(model_dir=str(missing)))
    with pytest.raises(RuntimeError, match="not set"):
        mod._resolve_model_dir(_spec(model_dir="  "))
    with pytest.raises(RuntimeError, match="not set"):
        mod._resolve_model_dir(_spec(model_dir=None))
    ready = tmp_path / "ready"
    (ready / "models").mkdir(parents=True)
    (ready / "models" / "xscaler.save").write_bytes(b"x")
    assert mod._resolve_model_dir(_spec(model_dir=str(ready))) == ready


def test_existing_checkout_is_reused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.models.external import openfpl as mod

    dest = tmp_path / "OpenFPL"
    (dest / "models").mkdir(parents=True)
    (dest / "models" / "xscaler.save").write_bytes(b"x")

    def boom(*_args: object, **_kwargs: object) -> None:
        raise AssertionError("git must not run when the ensemble is present")

    monkeypatch.setattr(mod.subprocess, "run", boom)
    assert mod.ensure_openfpl_checkout(dest) == dest


def test_incomplete_checkout_raises(tmp_path: Path) -> None:
    from fplt.models.external import openfpl as mod

    dest = tmp_path / "OpenFPL"
    dest.mkdir()
    (dest / "README").write_text("partial", encoding="utf-8")
    with pytest.raises(FileNotFoundError, match="incomplete"):
        mod.ensure_openfpl_checkout(dest)


def test_fetch_writes_the_ensemble(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.models.external import openfpl as mod

    dest = tmp_path / "OpenFPL"
    dest.mkdir()

    def fake_run(
        argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if "clone" in argv:
            (dest / "models").mkdir(parents=True)
        if "sparse-checkout" in argv:
            (dest / "models" / "xscaler.save").write_bytes(b"x")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    assert mod.ensure_openfpl_checkout(dest) == dest
    assert (dest / "models" / "xscaler.save").is_file()


def test_fetch_failure_raises(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from fplt.models.external import openfpl as mod

    def fake_run(
        argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="could not fetch"):
        mod.ensure_openfpl_checkout(tmp_path / "OpenFPL")


def test_fetch_without_scaler_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.models.external import openfpl as mod

    dest = tmp_path / "OpenFPL"

    def fake_run(
        argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if "clone" in argv:
            dest.mkdir()
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    with pytest.raises(FileNotFoundError, match="not found"):
        mod.ensure_openfpl_checkout(dest)


def test_default_model_dir_fetches_before_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.models.external import openfpl as mod

    root = tmp_path / "repo"
    runner = root / "external" / "openfpl"
    runner.mkdir(parents=True)
    (runner / "predict.py").write_text("", encoding="utf-8")
    venv_python = runner / ".venv" / "bin" / "python"
    venv_python.parent.mkdir(parents=True)
    venv_python.write_text("", encoding="utf-8")
    monkeypatch.setattr(mod, "_repo_root", lambda: root)
    dest = root / DEFAULT_MODEL_DIR

    def fake_run(
        argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        if "clone" in argv:
            (dest / "models").mkdir(parents=True)
        if "sparse-checkout" in argv:
            (dest / "models" / "xscaler.save").write_bytes(b"x")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    argv = mod._command(_spec())
    assert argv[0] == str(venv_python)
    assert argv[-1] == str(dest)
    assert (dest / "models" / "xscaler.save").is_file()


def test_scoring_python_installs_the_runner_env(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.models.external import openfpl as mod

    root = tmp_path / "repo"
    runner = root / "external" / "openfpl"
    runner.mkdir(parents=True)
    monkeypatch.setattr(mod, "_repo_root", lambda: root)
    venv_python = runner / ".venv" / "bin" / "python"

    def fake_run(
        argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        assert argv == ["uv", "sync"]
        venv_python.parent.mkdir(parents=True)
        venv_python.write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    assert mod._scoring_python() == str(venv_python)


def test_scoring_python_install_failure_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.models.external import openfpl as mod

    root = tmp_path / "repo"
    (root / "external" / "openfpl").mkdir(parents=True)
    monkeypatch.setattr(mod, "_repo_root", lambda: root)

    def fake_run(
        argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, argv)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="could not install"):
        mod._scoring_python()


def test_scoring_python_requires_the_venv_binary(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    from fplt.models.external import openfpl as mod

    root = tmp_path / "repo"
    (root / "external" / "openfpl").mkdir(parents=True)
    monkeypatch.setattr(mod, "_repo_root", lambda: root)

    def fake_run(
        argv: list[str], **_kwargs: object
    ) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(argv, 0)

    monkeypatch.setattr(mod.subprocess, "run", fake_run)
    with pytest.raises(FileNotFoundError):
        mod._scoring_python()
