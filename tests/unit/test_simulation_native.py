"""Unit tests for ``mc_backend`` resolution and native simulate wrapper."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from fplt.models.simulation_native import (
    FixtureSummary,
    native_kernel_available,
    resolve_mc_backend,
    simulate_match_native,
)
from fplt.models.spec import ModelSpec, write_identity
from fplt.scoring.rules import Rules

pytest.importorskip("fplt_mc")

_RULES = Rules.for_season("2024-25")


def _toy_inputs(n: int = 64) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    players = pl.DataFrame(
        {
            "player_key": [1, 2, 3, 4],
            "side": ["home", "home", "away", "away"],
            "position": ["DEF", "MID", "DEF", "FWD"],
            "goal_weight": [1.0, 1.0, 1.0, 1.0],
            "assist_weight": [1.0, 1.0, 1.0, 1.0],
        }
    )
    scorelines = np.column_stack(
        [np.full(n, 2, dtype=np.int64), np.full(n, 1, dtype=np.int64)]
    )
    minutes = np.full((4, n), 90.0)
    other = np.full((4, 3), 0.2)
    return players, scorelines, minutes, other


def test_native_kernel_available_when_wheel_installed() -> None:
    assert native_kernel_available() is True


def test_resolve_defaults_to_python() -> None:
    assert resolve_mc_backend(None, env={}) == "python"
    assert resolve_mc_backend("python", env={}) == "python"


def test_resolve_env_overrides_spec() -> None:
    assert resolve_mc_backend("python", env={"FPLT_MC_BACKEND": "native"}) == "native"
    assert resolve_mc_backend("native", env={"FPLT_MC_BACKEND": "python"}) == "python"


def test_resolve_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="mc_backend"):
        resolve_mc_backend("cuda", env={})
    with pytest.raises(ValueError, match="FPLT_MC_BACKEND"):
        resolve_mc_backend(None, env={"FPLT_MC_BACKEND": "cuda"})


def test_native_same_seed_is_deterministic() -> None:
    players, scorelines, minutes, other = _toy_inputs()
    a = simulate_match_native(
        scorelines,
        players,
        minutes,
        other,
        _RULES,
        kernel_seed=42,
        sample_other=True,
        cs_red_void_prob=0.1,
        return_samples=True,
    )
    b = simulate_match_native(
        scorelines,
        players,
        minutes,
        other,
        _RULES,
        kernel_seed=42,
        sample_other=True,
        cs_red_void_prob=0.1,
        return_samples=True,
    )
    np.testing.assert_array_equal(a.goals, b.goals)
    np.testing.assert_array_equal(a.assists, b.assists)
    np.testing.assert_allclose(a.points, b.points)
    c = simulate_match_native(
        scorelines,
        players,
        minutes,
        other,
        _RULES,
        kernel_seed=43,
        sample_other=True,
        cs_red_void_prob=0.1,
        return_samples=True,
    )
    assert not np.array_equal(a.points, c.points)


def test_summary_path_matches_sample_reduction() -> None:
    """K5 summary arrays must match Python reduce of the sample planes."""
    players, scorelines, minutes, other = _toy_inputs(n=256)
    summary = simulate_match_native(
        scorelines,
        players,
        minutes,
        other,
        _RULES,
        kernel_seed=7,
        sample_other=True,
        cs_red_void_prob=0.1,
        return_samples=False,
    )
    assert isinstance(summary, FixtureSummary)
    samples = simulate_match_native(
        scorelines,
        players,
        minutes,
        other,
        _RULES,
        kernel_seed=7,
        sample_other=True,
        cs_red_void_prob=0.1,
        return_samples=True,
    )
    np.testing.assert_allclose(summary.xpts, samples.points.mean(axis=1), atol=1e-12)
    np.testing.assert_allclose(summary.xpts_sd, samples.points.std(axis=1), atol=1e-12)
    np.testing.assert_allclose(summary.p_played, (minutes > 0).mean(axis=1), atol=1e-12)
    np.testing.assert_allclose(summary.p_60, (minutes >= 60).mean(axis=1), atol=1e-12)
    np.testing.assert_allclose(
        summary.p_goal, (samples.goals >= 1).mean(axis=1), atol=1e-12
    )
    np.testing.assert_allclose(
        summary.p_assist, (samples.assists >= 1).mean(axis=1), atol=1e-12
    )
    np.testing.assert_allclose(
        summary.p_cs, samples.clean_sheets.mean(axis=1), atol=1e-12
    )
    np.testing.assert_allclose(
        summary.p_haul, (samples.points >= 10).mean(axis=1), atol=1e-12
    )


def test_default_return_samples_is_summary() -> None:
    """Normal predict path default must not materialise sample planes."""
    players, scorelines, minutes, other = _toy_inputs()
    out = simulate_match_native(
        scorelines,
        players,
        minutes,
        other,
        _RULES,
        kernel_seed=11,
    )
    assert isinstance(out, FixtureSummary)
    assert out.xpts.shape == (4,)
    assert not hasattr(out, "points")


def test_write_identity_records_mc_provenance(tmp_path: Path) -> None:
    spec = ModelSpec(
        name="native_probe",
        kind="dc_layered",
        params={"mc_backend": "native", "n_sims": 10},
    )
    dest = tmp_path / "art"
    write_identity(dest, spec)
    prov = json.loads((dest / "mc_provenance.json").read_text(encoding="utf-8"))
    assert "fplt_mc" in prov
    assert prov["fplt_mc"]["version"]
    assert "kernel_api" in prov["fplt_mc"]


def test_dc_layered_v4_native_spec_loads() -> None:
    from fplt.models.spec import load_model_specs

    specs = load_model_specs()
    assert "dc_layered_v4_native" in specs
    native = specs["dc_layered_v4_native"]
    parent = specs["dc_layered_v4"]
    assert native.params["mc_backend"] == "native"
    assert native.params["n_sims"] == parent.params["n_sims"]
    assert native.model_id != parent.model_id
