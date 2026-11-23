"""Hard eleven-starter constraint on minutes draws."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fplt.models.distributions import MINUTE_BINS, MinutesDist, enforce_hard_starters


def _positions(*labels: str) -> np.ndarray:
    return np.asarray(labels, dtype=object)


def test_enforce_hard_starters_exactly_eleven_when_eligible() -> None:
    # 1 GK + 13 outfield; all drawn as 90-minute starts.
    n_players = 14
    minutes = np.full((n_players, 5), 90, dtype=np.int64)
    team = np.zeros(n_players, dtype=np.int64)
    weight = np.linspace(1.0, 0.1, n_players)
    positions = _positions("GK", *["MID"] * 13)
    rng = np.random.default_rng(0)
    adjusted = enforce_hard_starters(
        minutes, team, weight, rng, positions=positions, target=11
    )
    for sim in range(5):
        starters = int(np.sum(adjusted[:, sim] >= 60))
        assert starters == 11
        assert int(np.sum((adjusted[:, sim] >= 60) & (positions == "GK"))) == 1


def test_enforce_hard_starters_promotes_when_understuffed() -> None:
    minutes = np.zeros((12, 3), dtype=np.int64)
    team = np.zeros(12, dtype=np.int64)
    weight = np.ones(12)
    positions = _positions("GK", *["DEF"] * 11)
    rng = np.random.default_rng(1)
    adjusted = enforce_hard_starters(
        minutes, team, weight, rng, positions=positions, target=11
    )
    for sim in range(3):
        assert int(np.sum(adjusted[:, sim] >= 60)) == 11
        assert int(np.sum((adjusted[:, sim] >= 60) & (positions == "GK"))) == 1


def test_ineligible_players_are_not_forced_to_start() -> None:
    minutes = np.zeros((5, 2), dtype=np.int64)
    team = np.zeros(5, dtype=np.int64)
    weight = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    positions = _positions("GK", "MID", "MID", "MID", "MID")
    rng = np.random.default_rng(2)
    adjusted = enforce_hard_starters(
        minutes, team, weight, rng, positions=positions, target=11
    )
    for sim in range(2):
        assert int(np.sum(adjusted[:, sim] >= 60)) == 2
        assert np.all(adjusted[2:, sim] == 0)
        assert adjusted[0, sim] >= 60  # the eligible GK starts


def test_two_teams_are_constrained_separately() -> None:
    minutes = np.full((8, 2), 90, dtype=np.int64)
    team = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    weight = np.ones(8)
    positions = _positions("GK", "MID", "MID", "MID", "GK", "DEF", "DEF", "DEF")
    rng = np.random.default_rng(3)
    adjusted = enforce_hard_starters(
        minutes, team, weight, rng, positions=positions, target=3
    )
    for sim in range(2):
        assert int(np.sum(adjusted[:4, sim] >= 60)) == 3
        assert int(np.sum(adjusted[4:, sim] >= 60)) == 3
        assert int(np.sum((adjusted[:4, sim] >= 60) & (positions[:4] == "GK"))) == 1
        assert int(np.sum((adjusted[4:, sim] >= 60) & (positions[4:] == "GK"))) == 1


def test_exactly_one_gk_when_both_drawn_as_starters() -> None:
    # Two GKs + ten outfield all at 90 → must demote one GK and keep 11.
    minutes = np.full((12, 4), 90, dtype=np.int64)
    team = np.zeros(12, dtype=np.int64)
    # First GK is preferred.
    weight = np.array([0.9, 0.4] + [0.8 - 0.05 * i for i in range(10)])
    positions = _positions("GK", "GK", *["MID"] * 10)
    rng = np.random.default_rng(4)
    adjusted = enforce_hard_starters(
        minutes, team, weight, rng, positions=positions, target=11
    )
    for sim in range(4):
        assert int(np.sum(adjusted[:, sim] >= 60)) == 11
        gk_starters = np.flatnonzero((adjusted[:, sim] >= 60) & (positions == "GK"))
        assert gk_starters.tolist() == [0]
        assert adjusted[1, sim] < 60


def test_promotes_gk_when_none_started() -> None:
    minutes = np.zeros((12, 3), dtype=np.int64)
    # Ten outfield already starting; GKs on the bench.
    minutes[2:, :] = 90
    team = np.zeros(12, dtype=np.int64)
    weight = np.linspace(1.0, 0.2, 12)
    positions = _positions("GK", "GK", *["FWD"] * 10)
    rng = np.random.default_rng(5)
    adjusted = enforce_hard_starters(
        minutes, team, weight, rng, positions=positions, target=11
    )
    for sim in range(3):
        assert int(np.sum(adjusted[:, sim] >= 60)) == 11
        gk_starters = np.flatnonzero((adjusted[:, sim] >= 60) & (positions == "GK"))
        assert gk_starters.tolist() == [0]


def test_no_eligible_gk_leaves_outfield_only() -> None:
    minutes = np.zeros((5, 2), dtype=np.int64)
    team = np.zeros(5, dtype=np.int64)
    weight = np.array([0.0, 1.0, 1.0, 1.0, 1.0])  # GK ineligible
    positions = _positions("GK", "DEF", "DEF", "MID", "MID")
    rng = np.random.default_rng(6)
    adjusted = enforce_hard_starters(
        minutes, team, weight, rng, positions=positions, target=11
    )
    for sim in range(2):
        assert int(np.sum(adjusted[:, sim] >= 60)) == 4
        assert adjusted[0, sim] == 0


def test_start_weights_match_p60() -> None:
    pmf = np.zeros((2, len(MINUTE_BINS)))
    pmf[0, -1] = 1.0  # 90
    pmf[1, 0] = 1.0  # 0
    dist = MinutesDist(
        pl.DataFrame({"player_key": [1, 2], "fixture_id": [1, 1]}),
        MINUTE_BINS,
        pmf,
    )
    weights = dist.start_weights()
    np.testing.assert_allclose(weights, [1.0, 0.0])


def test_enforce_hard_starters_rejects_bad_shapes() -> None:
    rng = np.random.default_rng(0)
    positions = _positions("GK", "MID")
    with pytest.raises(ValueError, match="2-D"):
        enforce_hard_starters(
            np.array([1, 2]),
            np.array([1, 1]),
            np.ones(2),
            rng,
            positions=positions,
        )
    with pytest.raises(ValueError, match="match"):
        enforce_hard_starters(
            np.zeros((2, 1), dtype=np.int64),
            np.array([1]),
            np.ones(2),
            rng,
            positions=positions,
        )
    with pytest.raises(ValueError, match="match"):
        enforce_hard_starters(
            np.zeros((2, 1), dtype=np.int64),
            np.array([1, 1]),
            np.ones(2),
            rng,
            positions=_positions("GK"),
        )
    with pytest.raises(ValueError, match="positive"):
        enforce_hard_starters(
            np.zeros((2, 1), dtype=np.int64),
            np.array([1, 1]),
            np.ones(2),
            rng,
            positions=positions,
            target=0,
        )
