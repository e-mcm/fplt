"""Model specs load from YAML and record their identity once."""

import json
import subprocess
from pathlib import Path

import pytest

from fplt.models.spec import (
    ModelSpec,
    artifact_dir,
    git_sha,
    load_model_specs,
    save_spec,
)
from fplt.types import GWKey


def test_loaded_spec_matches_the_object_and_includes_package_version(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spec = load_model_specs()["rolling_mean_v1"]
    assert spec.kind == "rolling_mean"
    assert spec.params["window"] == 5
    built = ModelSpec(name="rolling_mean_v1", kind="rolling_mean", params={"window": 5})
    original = spec.model_id
    assert original == built.model_id
    monkeypatch.setattr("fplt.models.spec._package_version", lambda: "9.9.9")
    assert spec.model_id != original


def test_load_rejects_a_duplicate_name(tmp_path: Path) -> None:
    body = "name: same\nkind: rolling_mean\n"
    (tmp_path / "a.yaml").write_text(body, encoding="utf-8")
    (tmp_path / "b.yaml").write_text(body, encoding="utf-8")
    with pytest.raises(ValueError, match="duplicate"):
        load_model_specs(tmp_path)


def test_load_rejects_a_file_that_is_not_a_mapping(tmp_path: Path) -> None:
    (tmp_path / "notes.yaml").write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ValueError, match="mapping"):
        load_model_specs(tmp_path)


def test_load_missing_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_model_specs(tmp_path / "missing")


def test_nested_component_round_trips(tmp_path: Path) -> None:
    (tmp_path / "layered.yaml").write_text(
        "\n".join(
            [
                "name: layered",
                "kind: composite",
                "components:",
                "  score:",
                "    name: dc",
                "    kind: dixon_coles",
                "    params:",
                "      rho: -0.05",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    spec = load_model_specs(tmp_path)["layered"]
    assert spec.components["score"].params["rho"] == -0.05
    assert spec.components["score"].model_id != spec.model_id


def test_save_spec_writes_identity_once(tmp_path: Path) -> None:
    spec = ModelSpec(name="rolling_mean_v1", kind="rolling_mean", params={"window": 5})
    as_of = GWKey("2024-25", 2)
    dest = save_spec(tmp_path, spec, as_of)
    assert dest == artifact_dir(tmp_path, spec, as_of)
    assert dest.name == "as_of=2024-25_gw02"
    written = json.loads((dest / "spec.json").read_text(encoding="utf-8"))
    assert written["name"] == "rolling_mean_v1"
    assert (dest / "git_sha.txt").read_text(encoding="utf-8").strip() == git_sha()
    with pytest.raises(FileExistsError):
        save_spec(tmp_path, spec, as_of)


def test_git_sha_is_unknown_when_git_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    def _fail(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        raise subprocess.CalledProcessError(1, "git")

    monkeypatch.setattr("fplt.models.spec.subprocess.run", _fail)
    assert git_sha() == "unknown"


def test_git_sha_is_unknown_when_output_is_blank(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def _blank(*_args: object, **_kwargs: object) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(args=["git"], returncode=0, stdout="\n")

    monkeypatch.setattr("fplt.models.spec.subprocess.run", _blank)
    assert git_sha() == "unknown"
