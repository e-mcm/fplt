"""Bit-identical equivalence for PR0 NumPy quick wins (searchsorted + hard XI)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from hypothesis.extra.numpy import arrays

from fplt.models.composite import _sample_minutes_crn
from fplt.models.distributions import (
    MAX_GOALS,
    MINUTE_BINS,
    MinutesDist,
    ScorelineDist,
    _adjust_starter_count,
    _inverse_cdf_index,
    enforce_hard_starters,
)
from fplt.types import GWKey
from fplt.utils.rng import player_seed_for

_GOALS = MAX_GOALS + 1


def _legacy_inverse_cdf_index(cdf: np.ndarray, uniforms: np.ndarray) -> np.ndarray:
    """Former ``sum(cdf <= u)`` sampler index (pre-searchsorted)."""
    index = np.sum(cdf[:, None, :] <= uniforms[..., None], axis=-1)
    return np.minimum(index, cdf.shape[1] - 1)


def _legacy_adjust_starter_count(
    minutes: np.ndarray,
    indexes: np.ndarray,
    weights: np.ndarray,
    order_key: np.ndarray,
    *,
    want: int,
    starter_threshold: int,
    start_minutes: int,
    sub_minutes: int,
) -> None:
    """Former per-sim promote/demote loop (pre-vectorisation)."""
    if indexes.size == 0:
        return
    for sim in range(minutes.shape[1]):
        started = indexes[minutes[indexes, sim] >= starter_threshold]
        if started.size > want:
            ranked = started[np.argsort(order_key[started])]
            for index in ranked[: started.size - want]:
                minutes[int(index), sim] = sub_minutes
        elif started.size < want:
            pool = indexes[minutes[indexes, sim] < starter_threshold]
            pool = pool[weights[pool] > 0]
            if pool.size == 0:
                continue
            ranked = pool[np.argsort(-order_key[pool])]
            for index in ranked[: want - started.size]:
                minutes[int(index), sim] = start_minutes


def _legacy_enforce_hard_starters(
    minutes: np.ndarray,
    team_keys: np.ndarray,
    start_weight: np.ndarray,
    rng: np.random.Generator,
    *,
    positions: np.ndarray,
    target: int = 11,
    starter_threshold: int = 60,
    start_minutes: int = 90,
    sub_minutes: int = 15,
    gk_position: str = "GK",
) -> np.ndarray:
    """Former enforce_hard_starters body using the per-sim adjuster."""
    drawn = np.asarray(minutes, dtype=np.int64)
    teams = np.asarray(team_keys)
    weights = np.asarray(start_weight, dtype=np.float64)
    pos = np.asarray(positions)
    out = drawn.copy()
    noise = np.asarray(rng.random(int(drawn.shape[0])), dtype=np.float64)
    order_key = np.asarray(weights + 1e-9 * noise, dtype=np.float64)
    for team in np.unique(teams):
        indexes = np.flatnonzero(teams == team)
        if indexes.size == 0:
            continue
        is_gk = np.asarray([str(pos[i]) == gk_position for i in indexes])
        gk_idx = indexes[is_gk]
        out_idx = indexes[~is_gk]
        eligible_gk = gk_idx[weights[gk_idx] > 0]
        eligible_out = out_idx[weights[out_idx] > 0]
        want = min(target, int(eligible_gk.size + eligible_out.size))
        gk_want = 1 if eligible_gk.size > 0 and want >= 1 else 0
        out_want = min(want - gk_want, int(eligible_out.size))
        _legacy_adjust_starter_count(
            out,
            gk_idx,
            weights,
            order_key,
            want=gk_want,
            starter_threshold=starter_threshold,
            start_minutes=start_minutes,
            sub_minutes=sub_minutes,
        )
        _legacy_adjust_starter_count(
            out,
            out_idx,
            weights,
            order_key,
            want=out_want,
            starter_threshold=starter_threshold,
            start_minutes=start_minutes,
            sub_minutes=sub_minutes,
        )
    return out


def _legacy_sample_minutes_crn(
    minutes: MinutesDist,
    seed: int,
    as_of: GWKey,
    fixture_id: int,
    n_sims: int,
) -> np.ndarray:
    """Former CRN minutes sampler using boolean-tensor CDF inversion."""
    rows = minutes.keys.height
    if n_sims == 0 or rows == 0:
        return np.zeros((rows, n_sims), dtype=np.int64)
    drawn = np.empty((rows, n_sims), dtype=np.int64)
    bins = np.asarray(minutes.bins, dtype=np.int64)
    width = bins.shape[0]
    for index, player_key in enumerate(minutes.keys["player_key"].to_list()):
        rng = player_seed_for(seed, as_of, fixture_id, int(player_key))
        cdf = np.cumsum(minutes.pmf[index])
        cdf[-1] = 1.0
        uniforms = rng.random(n_sims)
        choice = np.sum(cdf[None, :] <= uniforms[:, None], axis=-1)
        choice = np.minimum(choice, width - 1)
        drawn[index] = bins[choice]
    return drawn


@given(
    masses=arrays(
        np.float64,
        (3, 11),
        elements=st.floats(0.01, 2.0, allow_nan=False, allow_infinity=False),
    ),
    uniforms=arrays(
        np.float64,
        (3, 17),
        elements=st.floats(0.0, 0.999999999999, allow_nan=False, allow_infinity=False),
    ),
)
@settings(max_examples=40)
def test_inverse_cdf_index_matches_boolean_tensor(
    masses: np.ndarray, uniforms: np.ndarray
) -> None:
    cdf = np.cumsum(masses, axis=1)
    cdf = cdf / cdf[:, -1:]
    cdf[:, -1] = 1.0
    np.testing.assert_array_equal(
        _inverse_cdf_index(cdf, uniforms),
        _legacy_inverse_cdf_index(cdf, uniforms),
    )


def test_scoreline_sample_matches_legacy_boolean_tensor() -> None:
    rng_pmf = np.random.default_rng(11)
    raw = rng_pmf.random((2, _GOALS, _GOALS))
    raw /= raw.sum(axis=(1, 2), keepdims=True)
    dist = ScorelineDist(np.array([1, 2]), raw)
    flat = dist.pmf.reshape(2, _GOALS * _GOALS)
    cdf = np.cumsum(flat, axis=1)
    cdf[:, -1] = 1.0
    draws = np.random.default_rng(12).random((2, 200))
    np.testing.assert_array_equal(
        _inverse_cdf_index(cdf, draws),
        _legacy_inverse_cdf_index(cdf, draws),
    )
    a = dist.sample(np.random.default_rng(13), 50)
    b = dist.sample(np.random.default_rng(13), 50)
    np.testing.assert_array_equal(a, b)


def test_minutes_sample_matches_legacy_boolean_tensor() -> None:
    pmf = np.random.default_rng(21).random((5, len(MINUTE_BINS)))
    pmf /= pmf.sum(axis=1, keepdims=True)
    dist = MinutesDist(
        pl.DataFrame(
            {"player_key": list(range(5)), "fixture_id": [1] * 5},
        ),
        MINUTE_BINS,
        pmf,
    )
    cdf = np.cumsum(dist.pmf, axis=1)
    cdf[:, -1] = 1.0
    draws = np.random.default_rng(22).random((5, 100))
    np.testing.assert_array_equal(
        _inverse_cdf_index(cdf, draws),
        _legacy_inverse_cdf_index(cdf, draws),
    )


def test_sample_minutes_crn_matches_legacy() -> None:
    pmf = np.random.default_rng(31).random((8, len(MINUTE_BINS)))
    pmf /= pmf.sum(axis=1, keepdims=True)
    minutes = MinutesDist(
        pl.DataFrame(
            {"player_key": list(range(100, 108)), "fixture_id": [9] * 8},
        ),
        MINUTE_BINS,
        pmf,
    )
    as_of = GWKey(season="2025-26", gw=3)
    new = _sample_minutes_crn(minutes, seed=7, as_of=as_of, fixture_id=9, n_sims=64)
    old = _legacy_sample_minutes_crn(
        minutes, seed=7, as_of=as_of, fixture_id=9, n_sims=64
    )
    np.testing.assert_array_equal(new, old)


def test_adjust_starter_count_matches_legacy_loop() -> None:
    rng = np.random.default_rng(41)
    n_players, n_sims = 14, 200
    minutes = rng.choice(MINUTE_BINS, size=(n_players, n_sims)).astype(np.int64)
    indexes = np.arange(n_players)
    weights = rng.random(n_players)
    order_key = weights + 1e-9 * rng.random(n_players)
    want = 11
    new_m = minutes.copy()
    old_m = minutes.copy()
    _adjust_starter_count(
        new_m,
        indexes,
        weights,
        order_key,
        want=want,
        starter_threshold=60,
        start_minutes=90,
        sub_minutes=15,
    )
    _legacy_adjust_starter_count(
        old_m,
        indexes,
        weights,
        order_key,
        want=want,
        starter_threshold=60,
        start_minutes=90,
        sub_minutes=15,
    )
    np.testing.assert_array_equal(new_m, old_m)


@pytest.mark.parametrize("seed", [0, 1, 2, 7, 99])
def test_enforce_hard_starters_matches_legacy(seed: int) -> None:
    rng_data = np.random.default_rng(seed)
    n_home, n_away, n_sims = 20, 20, 500
    n_players = n_home + n_away
    minutes = rng_data.choice(MINUTE_BINS, size=(n_players, n_sims)).astype(np.int64)
    team = np.array([0] * n_home + [1] * n_away, dtype=np.int64)
    weight = rng_data.random(n_players)
    # Prefer at least one eligible GK per side.
    weight[0] = max(weight[0], 0.1)
    weight[n_home] = max(weight[n_home], 0.1)
    positions = np.array(
        ["GK"] + ["MID"] * (n_home - 1) + ["GK"] + ["DEF"] * (n_away - 1),
        dtype=object,
    )
    new = enforce_hard_starters(
        minutes,
        team,
        weight,
        np.random.default_rng(seed + 1000),
        positions=positions,
    )
    old = _legacy_enforce_hard_starters(
        minutes,
        team,
        weight,
        np.random.default_rng(seed + 1000),
        positions=positions,
    )
    np.testing.assert_array_equal(new, old)


def test_u_equals_cdf_edge_matches_legacy() -> None:
    """``u`` landing exactly on a CDF knot must pick the same bin."""
    cdf = np.array([[0.25, 0.5, 0.75, 1.0]], dtype=np.float64)
    uniforms = np.array([[0.0, 0.25, 0.5, 0.75, 0.999999, 1.0]], dtype=np.float64)
    np.testing.assert_array_equal(
        _inverse_cdf_index(cdf, uniforms),
        _legacy_inverse_cdf_index(cdf, uniforms),
    )
