"""Oracle-gain error attribution (toolkit §1.1 / #401)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fplt.backtest.error_attribution import (
    attribute_errors,
    oracle_fixture_pmf,
    oracle_gains,
    points_pmf_from_simulation,
    price_band,
    realised_involvement_weights,
    replace_component_inputs,
    scores_from_ledger_and_pmf,
)
from fplt.backtest.points_scores import (
    discrete_rps,
    log_score_pmf,
    mse_mean_pmf,
    score_points_pmf,
)
from fplt.models.simulation import simulate_match
from fplt.schemas.error_attribution import (
    ERROR_ATTRIBUTION_COLUMNS,
    ORACLE_COMPONENTS,
    ErrorAttributionSchema,
)
from fplt.scoring.rules import Rules
from fplt.store.points_pmf import POINTS_PMF_MAX, POINTS_PMF_MIN
from fplt.utils.frames import require_dataframe

_RULES = Rules.for_season("2024-25")
_WIDTH = POINTS_PMF_MAX - POINTS_PMF_MIN + 1


def _delta_pmf(y: int) -> np.ndarray:
    dense = np.zeros(_WIDTH, dtype=np.float64)
    dense[_clip(y) - POINTS_PMF_MIN] = 1.0
    return dense


def _clip(y: int) -> int:
    return int(min(POINTS_PMF_MAX, max(POINTS_PMF_MIN, y)))


def _long_pmf(
    player_key: int,
    fixture_id: int,
    dense: np.ndarray,
) -> pl.DataFrame:
    rows = [
        {
            "player_key": player_key,
            "fixture_id": fixture_id,
            "points_bin": POINTS_PMF_MIN + i,
            "probability": float(p),
        }
        for i, p in enumerate(dense.tolist())
        if p > 0.0
    ]
    return pl.DataFrame(rows)


def _toy_sim_inputs(
    n: int = 32,
) -> tuple[pl.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    players = pl.DataFrame(
        {
            "player_key": [1, 2],
            "side": ["home", "away"],
            "position": ["MID", "FWD"],
            "goal_weight": [0.6, 0.4],
            "assist_weight": [0.5, 0.5],
        }
    )
    scorelines = np.column_stack(
        [np.full(n, 1, dtype=np.int64), np.full(n, 0, dtype=np.int64)]
    )
    minutes = np.full((2, n), 90, dtype=np.int64)
    other = np.zeros((2, 3), dtype=np.float64)
    other[:, 2] = 1.0
    return players, scorelines, minutes, other


def test_dense_pmf_ignores_oob_bins_and_zero_mass() -> None:
    from fplt.backtest.points_scores import _dense_pmf

    oob = _dense_pmf(
        np.array([POINTS_PMF_MIN - 5, 0], dtype=np.int64),
        np.array([0.5, 0.5], dtype=np.float64),
    )
    assert oob.sum() == pytest.approx(1.0)
    zero = _dense_pmf(
        np.array([0], dtype=np.int64),
        np.array([0.0], dtype=np.float64),
    )
    assert zero.sum() == pytest.approx(0.0)


def test_hand_computed_rps_log_mse() -> None:
    """RPS / log / MSE match a two-bin hand example (clipped into −5…30)."""
    dense = np.zeros(_WIDTH, dtype=np.float64)
    # P(0)=0.25, P(1)=0.75; y=1.
    dense[0 - POINTS_PMF_MIN] = 0.25
    dense[1 - POINTS_PMF_MIN] = 0.75
    y = 1
    # CDF vs indicator for y<=k: only bins before y contribute to RPS gap.
    # F(k)−1{y≤k} is non-zero for k < y when mass sits below y.
    expected_rps = 0.0
    cdf = 0.0
    for k in range(POINTS_PMF_MIN, POINTS_PMF_MAX + 1):
        cdf += dense[k - POINTS_PMF_MIN]
        observed = 1.0 if y <= k else 0.0
        expected_rps += (cdf - observed) ** 2
    assert discrete_rps(dense, y) == pytest.approx(expected_rps)
    assert log_score_pmf(dense, y) == pytest.approx(-np.log(0.75))
    assert mse_mean_pmf(dense, y) == pytest.approx((0.75 - 1.0) ** 2)


def test_perfect_component_oracle_gain_is_zero() -> None:
    """When control already equals the oracle PMF, every gain is 0."""
    y = 6
    dense = _delta_pmf(y)
    pmf = _long_pmf(10, 100, dense)
    outcomes = pl.DataFrame(
        {
            "player_key": [10],
            "fixture_id": [100],
            "pts": [y],
            "season": ["2024-25"],
            "gw": [1],
            "position": ["MID"],
            "price_band": ["mid"],
        }
    )
    control = score_points_pmf(pmf, outcomes)
    gains = oracle_gains(control, control, component="minutes")
    assert gains["gain_rps"][0] == pytest.approx(0.0)
    assert gains["gain_log_score"][0] == pytest.approx(0.0)
    assert gains["gain_mse_mean"][0] == pytest.approx(0.0)


@given(
    true_y=st.integers(min_value=0, max_value=12),
    noise=st.floats(min_value=0.05, max_value=0.45),
    data=st.data(),
)
@settings(max_examples=40, deadline=None)
def test_noisy_forecast_gain_nonnegative_in_expectation(
    true_y: int,
    noise: float,
    data: st.DataObject,
) -> None:
    """Proper scores: E[S(noisy)] ≥ E[S(true)]; mean oracle gain ≥ 0."""
    # Oracle = delta at true_y. Noisy = (1−ε)δ_y + ε·Uniform over a wrong bin.
    wrong = data.draw(
        st.integers(min_value=0, max_value=12).filter(lambda v: v != true_y)
    )
    oracle = _delta_pmf(true_y)
    noisy = (1.0 - noise) * oracle
    noisy[_clip(wrong) - POINTS_PMF_MIN] += noise

    # Draw outcomes from the true distribution (always true_y here).
    n = 80
    outcomes = pl.DataFrame(
        {
            "player_key": list(range(n)),
            "fixture_id": [1] * n,
            "pts": [true_y] * n,
        }
    )
    control_pmf = pl.concat(
        [_long_pmf(i, 1, noisy) for i in range(n)],
        how="vertical",
    )
    oracle_pmf = pl.concat(
        [_long_pmf(i, 1, oracle) for i in range(n)],
        how="vertical",
    )
    control = score_points_pmf(control_pmf, outcomes)
    oracle_scores = score_points_pmf(oracle_pmf, outcomes)
    gains = oracle_gains(control, oracle_scores, component="scoreline")
    assert float(gains["gain_rps"].mean()) >= -1e-12
    assert float(gains["gain_log_score"].mean()) >= -1e-12
    assert float(gains["gain_mse_mean"].mean()) >= -1e-12


def test_blank_gw_produces_no_scores() -> None:
    """Blank GW: PMF rows with no matching outcomes yield an empty score frame."""
    pmf = _long_pmf(1, 50, _delta_pmf(4))
    outcomes = pl.DataFrame(
        schema={
            "player_key": pl.Int64(),
            "fixture_id": pl.Int64(),
            "pts": pl.Int64(),
        }
    )
    scored = score_points_pmf(pmf, outcomes)
    assert scored.is_empty()


def test_dgw_scores_each_fixture_separately() -> None:
    """DGW: two fixtures for one player produce two scored rows."""
    pmf = pl.concat(
        [
            _long_pmf(7, 10, _delta_pmf(2)),
            _long_pmf(7, 11, _delta_pmf(5)),
        ]
    )
    outcomes = pl.DataFrame(
        {
            "player_key": [7, 7],
            "fixture_id": [10, 11],
            "pts": [2, 5],
            "season": ["2024-25", "2024-25"],
            "gw": [12, 12],
            "position": ["FWD", "FWD"],
            "price_band": ["premium", "premium"],
        }
    )
    scored = score_points_pmf(pmf, outcomes).sort("fixture_id")
    assert scored.height == 2
    assert scored["fixture_id"].to_list() == [10, 11]
    assert scored["rps"].to_list() == pytest.approx([0.0, 0.0])


def test_pit_safe_ignores_other_fixture_outcomes() -> None:
    """Outcomes for a different fixture_id must not score this PMF row."""
    pmf = _long_pmf(1, 100, _delta_pmf(3))
    outcomes = pl.DataFrame(
        {
            "player_key": [1, 1],
            "fixture_id": [999, 100],
            "pts": [99, 3],
        }
    )
    scored = score_points_pmf(pmf, outcomes)
    assert scored.height == 1
    assert scored["fixture_id"][0] == 100
    assert scored["rps"][0] == pytest.approx(0.0)
    # The bogus 99-point outcome for fixture 999 must not leak in.
    assert scored["mse_mean"][0] == pytest.approx(0.0)


def test_minutes_oracle_matches_broadcast_realised() -> None:
    """Replacing minutes with realised values changes only that plane."""
    players, scorelines, minutes, other = _toy_sim_inputs()
    realised = np.array([0, 90], dtype=np.int64)
    mins_o, scores_o, players_o, other_o = replace_component_inputs(
        component="minutes",
        minutes=minutes,
        scorelines=scorelines,
        sim_players=players,
        other_grid=other,
        realised_minutes=realised,
    )
    assert np.array_equal(scores_o, scorelines)
    assert players_o.equals(players)
    assert np.array_equal(other_o, other)
    assert np.array_equal(mins_o[:, 0], realised)
    assert np.all(mins_o[0] == 0)
    assert np.all(mins_o[1] == 90)


def test_oracle_fixture_pmf_perfect_minutes_gain_near_zero() -> None:
    """Control already at realised minutes ⇒ minutes oracle gain ≈ 0 (CRN)."""
    players, scorelines, minutes, other = _toy_sim_inputs(n=64)
    realised = np.array([90, 90], dtype=np.int64)
    rng_control = np.random.default_rng(0)
    control_sim = simulate_match(
        scorelines, players, minutes, other, _RULES, rng_control
    )
    control_pmf = points_pmf_from_simulation(
        control_sim.points, control_sim.player_key, fixture_id=1
    )
    rng_oracle = np.random.default_rng(0)
    oracle_pmf = oracle_fixture_pmf(
        component="minutes",
        minutes=minutes,
        scorelines=scorelines,
        sim_players=players,
        other_grid=other,
        rules=_RULES,
        rng=rng_oracle,
        fixture_id=1,
        realised_minutes=realised,
    )
    # Use control mean points as "outcomes" so both arms score the same y.
    y = np.rint(control_sim.points.mean(axis=1)).astype(np.int64)
    outcomes = pl.DataFrame(
        {
            "player_key": control_sim.player_key.tolist(),
            "fixture_id": [1, 1],
            "pts": y.tolist(),
            "season": ["2024-25", "2024-25"],
            "gw": [1, 1],
            "position": ["MID", "FWD"],
            "price_band": ["mid", "mid"],
        }
    )
    gains = oracle_gains(
        score_points_pmf(control_pmf, outcomes),
        score_points_pmf(oracle_pmf, outcomes),
        component="minutes",
    )
    assert float(gains["gain_rps"].abs().max()) == pytest.approx(0.0, abs=1e-12)


def test_attribute_errors_schema_and_bootstrap_ci() -> None:
    """Season-to-date rows validate and carry bootstrap CIs when n_gw ≥ 2."""
    rows: list[dict[str, object]] = []
    for gw in (1, 2, 3):
        for player in (1, 2):
            rows.append(
                {
                    "player_key": player,
                    "fixture_id": gw * 10 + player,
                    "season": "2024-25",
                    "gw": gw,
                    "position": "MID",
                    "price_band": "mid",
                    "component": "minutes",
                    "rps": 0.4,
                    "log_score": 1.0,
                    "mse_mean": 4.0,
                    "gain_rps": 0.1 * gw,
                    "gain_log_score": 0.05 * gw,
                    "gain_mse_mean": 0.2 * gw,
                }
            )
    gains = pl.DataFrame(rows)
    table = attribute_errors(
        gains,
        model_id="toy_v1",
        bootstrap_reps=200,
        bootstrap_seed=0,
        per_gw=True,
    )
    require_dataframe(ErrorAttributionSchema.validate(table))
    std = table.filter(pl.col("gw").is_null() & (pl.col("metric") == "rps"))
    assert std.height == 1
    assert std["ci_lo"][0] is not None
    assert std["ci_hi"][0] is not None
    assert std["n"][0] == 6
    per = table.filter(pl.col("gw").is_not_null())
    assert per.height == 9  # 3 GWs × 3 metrics
    std_only = attribute_errors(
        gains, model_id="toy_v1", bootstrap_reps=100, bootstrap_seed=1
    )
    assert std_only.filter(pl.col("gw").is_not_null()).is_empty()


def test_scores_from_ledger_and_pmf() -> None:
    """Ledger path (no forecast_scores) scores PMF against actual_pts."""
    pmf = _long_pmf(1, 10, _delta_pmf(4))
    ledger = pl.DataFrame(
        {
            "model_id": ["m"],
            "player_key": [1],
            "fixture_id": [10],
            "target_season": ["2024-25"],
            "target_gw": [5],
            "position": ["DEF"],
            "actual_pts": [4],
        }
    )
    scored = scores_from_ledger_and_pmf(pmf, ledger)
    assert scored.height == 1
    assert scored["season"][0] == "2024-25"
    assert scored["gw"][0] == 5
    assert scored["rps"][0] == pytest.approx(0.0)
    # Already-renamed season/gw columns skip the target_* rename branch.
    ledger2 = ledger.rename({"target_season": "season", "target_gw": "gw"})
    scored2 = scores_from_ledger_and_pmf(pmf, ledger2)
    assert scored2["season"][0] == "2024-25"


def test_realised_involvement_and_price_band() -> None:
    outcomes = pl.DataFrame(
        {
            "player_key": [1, 2, 3],
            "fixture_id": [1, 1, 1],
            "team_key": [10, 10, 20],
            "goals": [2, 0, 1],
            "assists": [0, 1, 0],
        }
    )
    weights = realised_involvement_weights(outcomes).sort("player_key")
    assert weights["goal_weight"].to_list() == pytest.approx([1.0, 0.0, 1.0])
    assert weights["assist_weight"].to_list() == pytest.approx([0.0, 1.0, 0.0])
    assert price_band(50) == "budget"
    assert price_band(60) == "mid"
    assert price_band(80) == "premium"


def test_error_attribution_schema_columns() -> None:
    empty = pl.DataFrame(schema=ERROR_ATTRIBUTION_COLUMNS)
    require_dataframe(ErrorAttributionSchema.validate(empty))
    assert set(ORACLE_COMPONENTS) == {
        "minutes",
        "scoreline",
        "involvement",
        "other",
    }


def test_score_helpers_reject_bad_inputs() -> None:
    bad = np.ones(3, dtype=np.float64)
    with pytest.raises(ValueError, match="length"):
        discrete_rps(bad, 0)
    with pytest.raises(ValueError, match="length"):
        log_score_pmf(bad, 0)
    with pytest.raises(ValueError, match="length"):
        mse_mean_pmf(bad, 0)
    with pytest.raises(ValueError, match="pts or actual_pts"):
        score_points_pmf(
            _long_pmf(1, 1, _delta_pmf(1)),
            pl.DataFrame({"player_key": [1], "fixture_id": [1]}),
        )
    with pytest.raises(ValueError, match="missing columns"):
        score_points_pmf(
            pl.DataFrame({"player_key": [1], "fixture_id": [1]}),
            pl.DataFrame({"player_key": [1], "fixture_id": [1], "pts": [1]}),
        )


def test_score_points_pmf_maps_target_keys_when_present() -> None:
    """Outcomes carrying only target_* still populate season/gw on scores."""
    pmf = _long_pmf(1, 10, _delta_pmf(2))
    outcomes = pl.DataFrame(
        {
            "player_key": [1],
            "fixture_id": [10],
            "pts": [2],
            "target_season": ["2024-25"],
            "target_gw": [3],
        }
    )
    scored = score_points_pmf(pmf, outcomes)
    assert scored["season"][0] == "2024-25"
    assert scored["gw"][0] == 3


def test_replace_component_error_and_oracle_paths() -> None:
    players, scorelines, minutes, other = _toy_sim_inputs(n=8)
    with pytest.raises(ValueError, match="unknown component"):
        replace_component_inputs(
            component="bonus",
            minutes=minutes,
            scorelines=scorelines,
            sim_players=players,
            other_grid=other,
        )
    with pytest.raises(ValueError, match="realised_minutes required"):
        replace_component_inputs(
            component="minutes",
            minutes=minutes,
            scorelines=scorelines,
            sim_players=players,
            other_grid=other,
        )
    with pytest.raises(ValueError, match="length must match"):
        replace_component_inputs(
            component="minutes",
            minutes=minutes,
            scorelines=scorelines,
            sim_players=players,
            other_grid=other,
            realised_minutes=np.array([90], dtype=np.int64),
        )
    with pytest.raises(ValueError, match="realised_scoreline required"):
        replace_component_inputs(
            component="scoreline",
            minutes=minutes,
            scorelines=scorelines,
            sim_players=players,
            other_grid=other,
        )
    mins_s, scores_s, players_s, other_s = replace_component_inputs(
        component="scoreline",
        minutes=minutes,
        scorelines=scorelines,
        sim_players=players,
        other_grid=other,
        realised_scoreline=(3, 1),
    )
    assert np.all(scores_s[:, 0] == 3)
    assert np.all(scores_s[:, 1] == 1)
    assert np.array_equal(mins_s, minutes)
    assert players_s.equals(players)
    assert np.array_equal(other_s, other)

    with pytest.raises(ValueError, match="realised_weights required"):
        replace_component_inputs(
            component="involvement",
            minutes=minutes,
            scorelines=scorelines,
            sim_players=players,
            other_grid=other,
        )
    weights = pl.DataFrame(
        {
            "player_key": [1, 2],
            "goal_weight": [1.0, 0.0],
            "assist_weight": [0.0, 1.0],
        }
    )
    _, _, players_i, _ = replace_component_inputs(
        component="involvement",
        minutes=minutes,
        scorelines=scorelines,
        sim_players=players,
        other_grid=other,
        realised_weights=weights,
    )
    assert players_i["goal_weight"].to_list() == pytest.approx([1.0, 0.0])

    with pytest.raises(ValueError, match="realised_other_total required"):
        replace_component_inputs(
            component="other",
            minutes=minutes,
            scorelines=scorelines,
            sim_players=players,
            other_grid=other,
        )
    with pytest.raises(ValueError, match="length must match"):
        replace_component_inputs(
            component="other",
            minutes=minutes,
            scorelines=scorelines,
            sim_players=players,
            other_grid=other,
            realised_other_total=np.array([1.0], dtype=np.float64),
        )
    # With realised minutes → bucket from those; without → mean control minutes.
    _, _, _, other_o = replace_component_inputs(
        component="other",
        minutes=minutes,
        scorelines=scorelines,
        sim_players=players,
        other_grid=other,
        realised_minutes=np.array([0, 45], dtype=np.int64),
        realised_other_total=np.array([2.0, 3.0], dtype=np.float64),
    )
    assert other_o[0, 0] == pytest.approx(2.0)
    assert other_o[1, 1] == pytest.approx(3.0)
    _, _, _, other_mean = replace_component_inputs(
        component="other",
        minutes=minutes,
        scorelines=scorelines,
        sim_players=players,
        other_grid=other,
        realised_other_total=np.array([1.5, 1.5], dtype=np.float64),
    )
    assert other_mean[0, 2] == pytest.approx(1.5)  # mean minutes 90 → 60+


def test_points_pmf_from_simulation_edges() -> None:
    with pytest.raises(ValueError, match="shape must be"):
        points_pmf_from_simulation(
            np.zeros(3), np.array([1, 2, 3], dtype=np.int64), fixture_id=1
        )
    empty = points_pmf_from_simulation(
        np.zeros((2, 0), dtype=np.float64),
        np.array([1, 2], dtype=np.int64),
        fixture_id=9,
    )
    assert empty.is_empty()


def test_oracle_gains_and_attribute_errors_edges() -> None:
    with pytest.raises(ValueError, match="unknown component"):
        oracle_gains(
            pl.DataFrame(
                {
                    "player_key": [1],
                    "fixture_id": [1],
                    "rps": [0.1],
                    "log_score": [0.2],
                    "mse_mean": [0.3],
                }
            ),
            pl.DataFrame(
                {
                    "player_key": [1],
                    "fixture_id": [1],
                    "rps": [0.0],
                    "log_score": [0.0],
                    "mse_mean": [0.0],
                }
            ),
            component="chips",
        )
    with pytest.raises(ValueError, match="missing columns"):
        attribute_errors(pl.DataFrame({"season": ["2024-25"]}), model_id="m")
    empty = attribute_errors(
        pl.DataFrame(
            schema={
                "season": pl.Utf8(),
                "gw": pl.Int64(),
                "position": pl.Utf8(),
                "price_band": pl.Utf8(),
                "component": pl.Utf8(),
                "gain_rps": pl.Float64(),
                "gain_log_score": pl.Float64(),
                "gain_mse_mean": pl.Float64(),
            }
        ),
        model_id="m",
    )
    assert empty.is_empty()


def test_realised_involvement_requires_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        realised_involvement_weights(
            pl.DataFrame({"player_key": [1], "fixture_id": [1]})
        )


def test_oracle_fixture_pmf_scoreline_component() -> None:
    players, scorelines, minutes, other = _toy_sim_inputs(n=16)
    pmf = oracle_fixture_pmf(
        component="scoreline",
        minutes=minutes,
        scorelines=scorelines,
        sim_players=players,
        other_grid=other,
        rules=_RULES,
        rng=np.random.default_rng(1),
        fixture_id=42,
        realised_scoreline=(0, 0),
    )
    assert not pmf.is_empty()
    assert set(pmf["fixture_id"].to_list()) == {42}
