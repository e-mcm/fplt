"""Hard eleven-starter constraint on minutes draws."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fplt.models.distributions import MINUTE_BINS, MinutesDist, enforce_hard_starters


def test_enforce_hard_starters_exactly_eleven_when_eligible() -> None:
    # 14 players on one team; first 14 overstuffed as 90-minute starts.
    n_players = 14
    minutes = np.full((n_players, 5), 90, dtype=np.int64)
    team = np.zeros(n_players, dtype=np.int64)
    weight = np.linspace(1.0, 0.1, n_players)
    rng = np.random.default_rng(0)
    adjusted = enforce_hard_starters(minutes, team, weight, rng, target=11)
    for sim in range(5):
        starters = int(np.sum(adjusted[:, sim] >= 60))
        assert starters == 11


def test_enforce_hard_starters_promotes_when_understuffed() -> None:
    minutes = np.zeros((12, 3), dtype=np.int64)
    team = np.zeros(12, dtype=np.int64)
    weight = np.ones(12)
    rng = np.random.default_rng(1)
    adjusted = enforce_hard_starters(minutes, team, weight, rng, target=11)
    for sim in range(3):
        assert int(np.sum(adjusted[:, sim] >= 60)) == 11


def test_ineligible_players_are_not_forced_to_start() -> None:
    minutes = np.zeros((5, 2), dtype=np.int64)
    team = np.zeros(5, dtype=np.int64)
    weight = np.array([1.0, 1.0, 0.0, 0.0, 0.0])
    rng = np.random.default_rng(2)
    adjusted = enforce_hard_starters(minutes, team, weight, rng, target=11)
    for sim in range(2):
        assert int(np.sum(adjusted[:, sim] >= 60)) == 2
        assert np.all(adjusted[2:, sim] == 0)


def test_two_teams_are_constrained_separately() -> None:
    minutes = np.full((8, 2), 90, dtype=np.int64)
    team = np.array([1, 1, 1, 1, 2, 2, 2, 2])
    weight = np.ones(8)
    rng = np.random.default_rng(3)
    adjusted = enforce_hard_starters(minutes, team, weight, rng, target=3)
    for sim in range(2):
        assert int(np.sum(adjusted[:4, sim] >= 60)) == 3
        assert int(np.sum(adjusted[4:, sim] >= 60)) == 3


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
    with pytest.raises(ValueError, match="2-D"):
        enforce_hard_starters(np.array([1, 2]), np.array([1, 1]), np.ones(2), rng)
    with pytest.raises(ValueError, match="match"):
        enforce_hard_starters(
            np.zeros((2, 1), dtype=np.int64), np.array([1]), np.ones(2), rng
        )
    with pytest.raises(ValueError, match="positive"):
        enforce_hard_starters(
            np.zeros((2, 1), dtype=np.int64),
            np.array([1, 1]),
            np.ones(2),
            rng,
            target=0,
        )
