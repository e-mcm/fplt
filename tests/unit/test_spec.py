"""Model specs load from YAML and record their identity once."""

import json
import subprocess
from pathlib import Path

import pytest
import yaml

from fplt.models.spec import (
    ModelSpec,
    artifact_dir,
    deep_merge,
    git_sha,
    load_model_specs,
    save_spec,
)
from fplt.types import GWKey

# Extends-fork model_ids for package 0.1.0 (parent dc_layered_v3 params).
# Update when intentional parent param changes land (e.g. logistic_blend).
# FDB-226: forks retargeted from rate-era v2 → v3; resolved hashes track parent.

_FORK_MODEL_IDS = {
    "dc_layered_nopens_v1": "21b0201ff51ec375",
    "dc_layered_nodefcon_v1": "c9e83926571f90b9",
    "dc_layered_odds_v1": "39f3faf71ef30a83",
}
_PRE_DEFCON_RATE_V2_MODEL_ID = "b12aad1575ff0d34"


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


def test_deep_merge_nested_dicts_and_replaces_lists() -> None:
    merged = deep_merge(
        {"a": 1, "nested": {"x": 1, "y": 2}, "items": [1, 2]},
        {"nested": {"y": 9, "z": 3}, "items": [3], "b": 2},
    )
    assert merged == {
        "a": 1,
        "nested": {"x": 1, "y": 9, "z": 3},
        "items": [3],
        "b": 2,
    }
    # Base is not mutated via shared nested refs from the override.
    override = {"nested": {"y": 0}}
    deep_merge({"nested": {"y": 1}}, override)
    assert override == {"nested": {"y": 0}}


def test_extends_forks_preserve_pre_migration_model_ids() -> None:
    specs = load_model_specs()
    for name, expected in _FORK_MODEL_IDS.items():
        assert specs[name].model_id == expected, name


def test_dc_layered_v2_keeps_pre_defcon_rate_model_id() -> None:
    """Boards from before 4d715b5 used this hash under the name dc_layered_v2."""
    specs = load_model_specs()
    assert specs["dc_layered_v2"].model_id == _PRE_DEFCON_RATE_V2_MODEL_ID
    assert specs["dc_layered_v3"].model_id != _PRE_DEFCON_RATE_V2_MODEL_ID
    assert specs["dc_layered_v3"].components["other"].params["defcon_mode"] == "rate"


def test_extends_forks_match_manual_full_merge() -> None:
    specs = load_model_specs()
    parent = specs["dc_layered_v3"].model_dump(mode="json")
    expected = {
        "dc_layered_nopens_v1": deep_merge(
            parent,
            {
                "name": "dc_layered_nopens_v1",
                "components": {
                    "involvement": {
                        "name": "involvement_xg_nopens_v1",
                        "params": {"use_penalties_order": False},
                    }
                },
            },
        ),
        "dc_layered_nodefcon_v1": deep_merge(
            parent,
            {
                "name": "dc_layered_nodefcon_v1",
                "components": {
                    "other": {
                        "name": "other_player_nodefcon_v1",
                        "params": {"use_defensive_contribution": False},
                    }
                },
            },
        ),
        "dc_layered_odds_v1": deep_merge(
            parent,
            {
                "name": "dc_layered_odds_v1",
                "components": {
                    "score": {
                        "name": "dixon_coles_odds_v1",
                        "params": {"market_prior_strength": 1.0},
                    }
                },
            },
        ),
    }
    for name, raw in expected.items():
        assert specs[name].model_dump(mode="json") == ModelSpec.model_validate(
            raw
        ).model_dump(mode="json")


def test_load_resolves_extends_in_a_custom_directory(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "base",
                "kind": "rolling_mean",
                "params": {"window": 5, "extra": 1},
            }
        ),
        encoding="utf-8",
    )
    (tmp_path / "fork.yaml").write_text(
        yaml.safe_dump(
            {
                "name": "fork",
                "extends": "base",
                "overrides": {"params": {"window": 7}},
            }
        ),
        encoding="utf-8",
    )
    specs = load_model_specs(tmp_path)
    assert specs["fork"].params == {"window": 7, "extra": 1}
    assert specs["fork"].kind == "rolling_mean"
    assert specs["base"].params["window"] == 5


def test_load_rejects_missing_extends_target(tmp_path: Path) -> None:
    (tmp_path / "fork.yaml").write_text(
        "name: fork\nextends: missing\noverrides: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="extends target"):
        load_model_specs(tmp_path)


def test_load_rejects_extends_chain(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        "name: base\nkind: rolling_mean\n",
        encoding="utf-8",
    )
    (tmp_path / "mid.yaml").write_text(
        "name: mid\nextends: base\noverrides: {}\n",
        encoding="utf-8",
    )
    (tmp_path / "leaf.yaml").write_text(
        "name: leaf\nextends: mid\noverrides: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="deeper than one"):
        load_model_specs(tmp_path)


def test_load_rejects_overrides_without_extends(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text(
        "name: bad\nkind: rolling_mean\noverrides:\n  params:\n    window: 1\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="overrides.*extends"):
        load_model_specs(tmp_path)


def test_load_rejects_unexpected_keys_on_extends_fork(tmp_path: Path) -> None:
    (tmp_path / "base.yaml").write_text(
        "name: base\nkind: rolling_mean\n",
        encoding="utf-8",
    )
    (tmp_path / "fork.yaml").write_text(
        "name: fork\nextends: base\nkind: rolling_mean\noverrides: {}\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="unexpected keys"):
        load_model_specs(tmp_path)


def test_load_rejects_empty_name(tmp_path: Path) -> None:
    (tmp_path / "bad.yaml").write_text(
        "name: ''\nkind: rolling_mean\n",
        encoding="utf-8",
    )
    with pytest.raises(ValueError, match="non-empty string name"):
        load_model_specs(tmp_path)


def test_loader_does_not_import_omegaconf() -> None:
    import fplt.models.spec as spec_mod

    assert not hasattr(spec_mod, "OmegaConf")
    assert "omegaconf" not in dir(spec_mod)
