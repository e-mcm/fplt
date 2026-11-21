"""De-vig and odds→λ helpers used by the DC market prior."""

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fplt.models.components.score.odds_lambda import (
    blend_lambda,
    de_vig,
    independent_poisson_markets,
    odds_to_lambda,
)


def test_de_vig_probs_sum_to_one() -> None:
    probs = de_vig(np.array([2.0, 3.5, 4.0]))
    assert probs.sum() == pytest.approx(1.0)
    assert np.all(probs > 0)


_ODDS = st.floats(min_value=1.01, max_value=20.0, allow_nan=False, allow_infinity=False)


@given(home=_ODDS, draw=_ODDS, away=_ODDS)
@settings(max_examples=40, deadline=None)
def test_de_vig_property_sums_to_one(home: float, draw: float, away: float) -> None:
    probs = de_vig(np.array([home, draw, away]))
    assert probs.sum() == pytest.approx(1.0, abs=1e-12)


def test_de_vig_rejects_invalid_odds() -> None:
    with pytest.raises(ValueError, match="> 1"):
        de_vig(np.array([1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="non-empty"):
        de_vig(np.array([]))


def test_odds_to_lambda_favours_the_shorter_price() -> None:
    # Strong home favourite → higher home λ than a balanced book.
    home_fav, away_dog = odds_to_lambda(1.40, 4.50, 8.00)
    balanced_h, balanced_a = odds_to_lambda(2.40, 3.30, 2.90)
    assert home_fav > away_dog
    assert home_fav > balanced_h
    assert abs(balanced_h - balanced_a) < abs(home_fav - away_dog)


def test_odds_to_lambda_with_over_under_moves_total() -> None:
    low_total_h, low_total_a = odds_to_lambda(
        2.50, 3.20, 2.90, over_25_odds=2.40, under_25_odds=1.55
    )
    high_total_h, high_total_a = odds_to_lambda(
        2.50, 3.20, 2.90, over_25_odds=1.50, under_25_odds=2.50
    )
    assert high_total_h + high_total_a > low_total_h + low_total_a


def test_odds_to_lambda_recovers_1x2_roughly() -> None:
    lambda_h, lambda_a = odds_to_lambda(1.80, 3.60, 4.50)
    p_home, p_draw, p_away, _ = independent_poisson_markets(lambda_h, lambda_a)
    target = de_vig(np.array([1.80, 3.60, 4.50]))
    assert abs(p_home - target[0]) < 0.05
    assert abs(p_draw - target[1]) < 0.05
    assert abs(p_away - target[2]) < 0.05


def test_blend_lambda_strength_zero_is_identity() -> None:
    assert blend_lambda(1.4, 2.0, 0.0) == pytest.approx(1.4)
    assert blend_lambda(1.4, None, 5.0) == pytest.approx(1.4)


def test_blend_lambda_moves_toward_market() -> None:
    mid = blend_lambda(1.0, 3.0, 1.0)
    assert mid == pytest.approx(2.0)
    strong = blend_lambda(1.0, 3.0, 9.0)
    assert abs(strong - 3.0) < abs(mid - 3.0)
