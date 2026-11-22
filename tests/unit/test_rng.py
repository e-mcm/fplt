"""Production and promote-eval RNG streams."""

from __future__ import annotations

import numpy as np

from fplt.types import GWKey
from fplt.utils.rng import (
    active_eval_seed,
    eval_seed_for,
    paired_eval_seed,
    player_seed_for,
    seed_for,
)


def test_seed_for_is_stable_and_independent_per_fixture() -> None:
    as_of = GWKey("2024-25", 10)
    first = seed_for("abc", as_of, 1).random(4)
    again = seed_for("abc", as_of, 1).random(4)
    other = seed_for("abc", as_of, 2).random(4)
    assert list(first) == list(again)
    assert list(first) != list(other)


def test_seed_for_digest_is_unchanged() -> None:
    """Pin the production stream so CRN work cannot rewrite seed_for."""
    as_of = GWKey("2024-25", 10)
    drawn = seed_for("abc", as_of, 1).random(4)
    assert list(drawn) == [
        0.9600003520227198,
        0.6136028515773583,
        0.6691282607999509,
        0.6973863523793633,
    ]


def test_eval_seed_for_omits_model_id() -> None:
    as_of = GWKey("2024-25", 10)
    left = eval_seed_for(7, as_of, 1).random(4)
    right = eval_seed_for(7, as_of, 1).random(4)
    other_fixture = eval_seed_for(7, as_of, 2).random(4)
    other_seed = eval_seed_for(8, as_of, 1).random(4)
    assert list(left) == list(right)
    assert list(left) != list(other_fixture)
    assert list(left) != list(other_seed)
    # Distinct from the production stream for a typical model_id.
    assert list(left) != list(seed_for("abc", as_of, 1).random(4))


def test_player_seed_for_is_stable_per_player() -> None:
    as_of = GWKey("2024-25", 3)
    first = player_seed_for(11, as_of, 42, 100).random(5)
    again = player_seed_for(11, as_of, 42, 100).random(5)
    other = player_seed_for(11, as_of, 42, 200).random(5)
    assert list(first) == list(again)
    assert list(first) != list(other)


def test_extra_player_does_not_shift_other_player_draws() -> None:
    as_of = GWKey("2024-25", 3)
    seed = 19
    fixture_id = 55
    alone = player_seed_for(seed, as_of, fixture_id, 100).random(8)
    # Drawing another player's stream first must not change player 100.
    _ = player_seed_for(seed, as_of, fixture_id, 999).random(8)
    with_extra = player_seed_for(seed, as_of, fixture_id, 100).random(8)
    assert list(alone) == list(with_extra)


def test_paired_eval_seed_context_sets_active_seed() -> None:
    assert active_eval_seed() is None
    with paired_eval_seed(3):
        assert active_eval_seed() == 3
        with paired_eval_seed(9):
            assert active_eval_seed() == 9
        assert active_eval_seed() == 3
    assert active_eval_seed() is None


def test_player_streams_use_seed_sequence() -> None:
    as_of = GWKey("2023-24", 1)
    # Two players drawn in either order stay independent.
    a_then_b = np.concatenate(
        [
            player_seed_for(1, as_of, 10, 1).random(3),
            player_seed_for(1, as_of, 10, 2).random(3),
        ]
    )
    b_then_a = np.concatenate(
        [
            player_seed_for(1, as_of, 10, 2).random(3),
            player_seed_for(1, as_of, 10, 1).random(3),
        ]
    )
    assert list(a_then_b[:3]) == list(b_then_a[3:])
    assert list(a_then_b[3:]) == list(b_then_a[:3])
