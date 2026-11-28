"""Forecast evaluation: proper scores, PIT, climatology, aggregations (#395)."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from scipy import stats

from fplt.automation import post_gw as post_gw_mod
from fplt.automation.post_gw import (
    format_post_gw_ntfy,
    format_post_gw_summary,
    run_post_gw,
)
from fplt.automation.production import OfficialLine, ProductionOps
from fplt.backtest import forecast_scores as fs_mod
from fplt.backtest.forecast_scores import (
    aggregate_forecast_scores,
    compute_forecast_scores,
    empty_forecast_score_agg,
    empty_forecast_scores,
    fit_climatology,
    minutes_likely_bucket,
    selection_conditional_bias,
)
from fplt.backtest.points_scores import (
    LOG_SCORE_FLOOR,
    dense_pmf,
    discrete_rps,
    log_score_pmf,
    pmf_event_probs,
    randomised_pit,
)
from fplt.data.store import DataStore
from fplt.models.pipeline_spec import live_pipeline_defaults
from fplt.models.spec import load_model_specs
from fplt.schemas.manifests import BuildManifest, BuildTableEntry
from fplt.store.official import official_checkpoint_dir
from fplt.store.points_pmf import POINTS_PMF_MAX, POINTS_PMF_MIN
from fplt.store.runs import run_dir
from fplt.types import GWKey
from fplt.utils.rng import seed_for
from tests.unit.test_decision_scoring import _decision_gw

_WIDTH = POINTS_PMF_MAX - POINTS_PMF_MIN + 1
_TS = datetime(2024, 9, 1, 12, 0, tzinfo=UTC)
_DEADLINE = datetime(2024, 8, 16, 11, 0, tzinfo=UTC)
_SEASON = "2024-25"
_SPEC = "rolling_mean_v1"
_MODEL = load_model_specs()[_SPEC].model_id
_KEY = GWKey(_SEASON, 1)


def _delta(y: int) -> np.ndarray:
    dense = np.zeros(_WIDTH, dtype=np.float64)
    dense[y - POINTS_PMF_MIN] = 1.0
    return dense


def _long_pmf(player_key: int, fixture_id: int, dense: np.ndarray) -> pl.DataFrame:
    bins = np.arange(POINTS_PMF_MIN, POINTS_PMF_MAX + 1)
    mask = dense > 0
    return pl.DataFrame(
        {
            "player_key": [player_key] * int(mask.sum()),
            "fixture_id": [fixture_id] * int(mask.sum()),
            "points_bin": bins[mask].tolist(),
            "probability": dense[mask].tolist(),
            "target_season": [_SEASON] * int(mask.sum()),
            "target_gw": [1] * int(mask.sum()),
        }
    )


def test_hand_computed_rps_and_log_score() -> None:
    """RPS and log score match a two-bin hand example (Spec §2.3)."""
    dense = np.zeros(_WIDTH, dtype=np.float64)
    dense[0 - POINTS_PMF_MIN] = 0.25
    dense[1 - POINTS_PMF_MIN] = 0.75
    y = 1
    expected_rps = 0.0
    cdf = 0.0
    for k in range(POINTS_PMF_MIN, POINTS_PMF_MAX + 1):
        cdf += dense[k - POINTS_PMF_MIN]
        expected_rps += (cdf - (1.0 if y <= k else 0.0)) ** 2
    assert discrete_rps(dense, y) == pytest.approx(expected_rps)
    assert log_score_pmf(dense, y) == pytest.approx(-np.log(0.75))
    assert log_score_pmf(dense, 5) == pytest.approx(-np.log(LOG_SCORE_FLOOR))


def test_randomised_pit_reproducible_under_seed_for() -> None:
    """Same ``seed_for`` triple yields the same PIT draw."""
    dense = _delta(4)
    dense[4 - POINTS_PMF_MIN] = 0.5
    dense[5 - POINTS_PMF_MIN] = 0.5
    a = randomised_pit(dense, 4, seed_for(_MODEL, _KEY, fixture_id=99))
    b = randomised_pit(dense, 4, seed_for(_MODEL, _KEY, fixture_id=99))
    c = randomised_pit(dense, 4, seed_for(_MODEL, _KEY, fixture_id=100))
    assert a == pytest.approx(b)
    assert a != pytest.approx(c)


@pytest.mark.slow
def test_calibrated_pit_is_uniform_ks() -> None:
    """PIT of a calibrated synthetic forecast is uniform (KS, 10k draws)."""
    rng = np.random.default_rng(0)
    true = np.array([0.1, 0.2, 0.4, 0.2, 0.1], dtype=np.float64)
    support = np.array([0, 1, 2, 3, 4], dtype=np.int64)
    us: list[float] = []
    for i in range(10_000):
        y = int(rng.choice(support, p=true))
        dense = np.zeros(_WIDTH, dtype=np.float64)
        for bin_k, p in zip(support.tolist(), true.tolist(), strict=True):
            dense[bin_k - POINTS_PMF_MIN] = p
        us.append(randomised_pit(dense, y, np.random.default_rng(i)))
    stat = stats.kstest(us, "uniform")
    assert float(stat.pvalue) > 0.01


def test_overconfident_pit_is_u_shaped() -> None:
    """A deliberately overconfident forecast piles PIT mass at the edges."""
    rng = np.random.default_rng(1)
    true = np.full(5, 0.2, dtype=np.float64)
    support = np.arange(5, dtype=np.int64)
    # Forecast puts almost all mass on the mode of a flat truth → overconfident.
    forecast = np.array([0.02, 0.02, 0.92, 0.02, 0.02], dtype=np.float64)
    us: list[float] = []
    for i in range(2000):
        y = int(rng.choice(support, p=true))
        dense = np.zeros(_WIDTH, dtype=np.float64)
        for bin_k, p in zip(support.tolist(), forecast.tolist(), strict=True):
            dense[bin_k - POINTS_PMF_MIN] = p
        us.append(randomised_pit(dense, y, np.random.default_rng(10_000 + i)))
    counts, _ = np.histogram(us, bins=10, range=(0.0, 1.0))
    edge = float(counts[0] + counts[-1])
    mid = float(np.sum(counts[3:7]))
    assert edge > mid


def test_selection_bias_positive_on_optimiser_curse() -> None:
    """Picking top noisy μ̂ yields positive mean(μ̂ − y) on the selected set."""
    rng = np.random.default_rng(2)
    n = 200
    truth = rng.normal(5.0, 1.0, size=n)
    noise = rng.normal(0.0, 2.0, size=n)
    mu = truth + noise
    y = np.rint(truth).astype(np.int64)
    order = np.argsort(-mu)[:40]
    selected = set(int(i) for i in order.tolist())
    rows = []
    for i in range(n):
        rows.append(
            {
                "model_id": _MODEL,
                "season": _SEASON,
                "gw": 1 + (i % 5),
                "player_key": i,
                "fixture_id": 1000 + i,
                "position": "MID",
                "price_band": "mid",
                "minutes_bucket": "60+",
                "y": int(y[i]),
                "mu_hat": float(mu[i]),
                "rps": 0.1,
                "log_score": 0.1,
                "rps_clim": None,
                "pit_u": 0.5,
                "p_played": 0.9,
                "p_ge60": 0.8,
                "p_goal": 0.2,
                "p_assist": 0.2,
                "p_cs": 0.1,
                "p_haul": 0.1,
                "p_blank": 0.2,
                "y_played": 1,
                "y_ge60": 1,
                "y_goal": 0,
                "y_assist": 0,
                "y_cs": 0,
                "y_haul": 0,
                "y_blank": 0,
                "selected_lines": "primary" if i in selected else "",
                "schema_version": 1,
            }
        )
    scores = pl.DataFrame(rows)
    bias, delta = selection_conditional_bias(scores, line="primary")
    assert bias > 0.0
    assert delta > 0.0


def test_compute_forecast_scores_and_climatology() -> None:
    """End-to-end row scoring attaches events, PIT, clim RPS, and selection."""
    dense = np.zeros(_WIDTH, dtype=np.float64)
    dense[6 - POINTS_PMF_MIN] = 0.4
    dense[2 - POINTS_PMF_MIN] = 0.6
    pmf = _long_pmf(10, 100, dense)
    outcomes = pl.DataFrame(
        {
            "season": [_SEASON],
            "gw": [1],
            "player_key": [10],
            "fixture_id": [100],
            "pts": [6],
            "minutes": [90],
            "goals": [1],
            "assists": [0],
            "clean_sheets": [1],
            "position": ["FWD"],
            "price": [80],
        }
    )
    predictions = pl.DataFrame(
        {
            "model_id": [_MODEL],
            "target_season": [_SEASON],
            "target_gw": [1],
            "player_key": [10],
            "fixture_id": [100],
            "xpts": [4.2],
            "p_played": [0.95],
            "p_60": [0.9],
            "p_goal": [0.4],
            "p_assist": [0.2],
            "p_cs": [0.3],
            "p_haul": [0.15],
        }
    )
    clim = fit_climatology(
        pl.DataFrame(
            {
                "position": ["FWD", "FWD", "FWD"],
                "price_band": ["premium", "premium", "premium"],
                "minutes_bucket": ["60+", "60+", "60+"],
                "pts": [2, 6, 6],
            }
        )
    )
    scores = compute_forecast_scores(
        model_id=_MODEL,
        key=_KEY,
        points_pmf=pmf,
        predictions=predictions,
        outcomes=outcomes,
        selected_by_line={"primary": {10}},
        climatology=clim,
    )
    assert scores.height == 1
    assert scores["selected_lines"][0] == "primary"
    assert scores["y_goal"][0] == 1
    assert scores["y_haul"][0] == 0
    assert scores["rps_clim"][0] is not None
    assert 0.0 <= float(scores["pit_u"][0]) <= 1.0
    agg = aggregate_forecast_scores(scores, lines=("primary",))
    assert "mean_rps" in agg["metric"].to_list()
    assert "selection_bias" in agg["metric"].to_list()
    assert "pit_hist" in agg["metric"].to_list()


@given(
    y=st.integers(min_value=POINTS_PMF_MIN, max_value=POINTS_PMF_MAX),
    mass=st.floats(
        min_value=0.05, max_value=0.95, allow_nan=False, allow_infinity=False
    ),
)
@settings(max_examples=40, deadline=None)
def test_score_bounds_property(y: int, mass: float) -> None:
    """RPS ≥ 0, log score ≥ 0, PIT ∈ [0, 1] for any sparse two-bin PMF."""
    other = y + 1 if y < POINTS_PMF_MAX else y - 1
    dense = np.zeros(_WIDTH, dtype=np.float64)
    dense[y - POINTS_PMF_MIN] = mass
    dense[other - POINTS_PMF_MIN] = 1.0 - mass
    rps = discrete_rps(dense, y)
    log_s = log_score_pmf(dense, y)
    pit = randomised_pit(
        dense, y, np.random.default_rng(abs(hash((y, mass))) % (2**32))
    )
    assert rps >= 0.0
    assert log_s >= 0.0
    assert 0.0 <= pit <= 1.0


def test_minutes_likely_bucket_thresholds() -> None:
    """Minutes-likely strata align with climatology bucket labels."""
    assert minutes_likely_bucket(None) == "0"
    assert minutes_likely_bucket(0.1) == "0"
    assert minutes_likely_bucket(0.5) == "1-59"
    assert minutes_likely_bucket(0.9) == "60+"


def test_dense_pmf_normalises() -> None:
    """Sparse bins collapse and renormalise onto the official grid."""
    dense = dense_pmf(
        np.array([0, 0, 2], dtype=np.int64),
        np.array([1.0, 1.0, 2.0], dtype=np.float64),
    )
    assert dense.shape == (_WIDTH,)
    assert dense.sum() == pytest.approx(1.0)
    assert dense[0 - POINTS_PMF_MIN] == pytest.approx(0.5)


def test_empty_forecast_scores_schema() -> None:
    """Empty helper validates against the pandera schema."""
    assert empty_forecast_scores().is_empty()


def _ops() -> ProductionOps:
    pipeline = live_pipeline_defaults(model=_SPEC, horizon=3)
    return ProductionOps(
        models=[_SPEC],
        plan_horizon=3,
        pipeline=pipeline,
        lines=[
            OfficialLine(
                line="primary",
                spec=_SPEC,
                horizon=3,
                role="primary",
                squad_mode="real",
            )
        ],
    )


def _build() -> BuildManifest:
    return BuildManifest(
        build_id="b" * 64,
        created_at=_TS,
        staging_version=1,
        code_sha="abc1234",
        raw_manifest_hash="r" * 64,
        raw_cutoff=_TS,
        tables={
            "gameweeks": BuildTableEntry(sha="g" * 64, rows=1, schema_hash="s" * 64),
            "player_fixture_outcomes": BuildTableEntry(
                sha="o" * 64, rows=1, schema_hash="s" * 64
            ),
            "fixture_outcomes": BuildTableEntry(
                sha="f" * 64, rows=1, schema_hash="s" * 64
            ),
        },
    )


def test_run_post_gw_writes_forecast_scores(tmp_path: Path) -> None:
    """``run_post_gw`` persists forecast_scores + aggregations when PMF exists."""
    from fplt.data.store import DataStore
    from tests.unit.test_biggest_misses import (
        _fixtures,
        _outcomes,
        _seed_predict_run,
        _stats,
    )

    store = DataStore(tmp_path)
    store.write(
        "gameweeks",
        pl.DataFrame(
            [
                {
                    "season": _SEASON,
                    "gw": 1,
                    "deadline_time": _DEADLINE,
                    "finished": True,
                    "data_checked": True,
                    "available_at": _TS,
                    "source": "fpl_api",
                }
            ]
        ),
    )
    store.write("fixtures", _fixtures(n=10))
    store.write(
        "player_fixture_outcomes",
        _outcomes(
            [
                {
                    "player_key": 42,
                    "fixture_id": 1,
                    "team_key": 1,
                    "pts": 5,
                    "minutes": 90,
                    "goals": 0,
                    "assists": 0,
                    "clean_sheets": 1,
                }
            ]
        ),
    )
    store.write(
        "player_fixture_stats",
        _stats(
            [
                {
                    "player_key": 42,
                    "fixture_id": 1,
                    "minutes": 90,
                    "starts": 1,
                    "red_cards": 0,
                    "penalties_missed": 0,
                }
            ]
        ),
    )
    store.write(
        "players",
        pl.DataFrame(
            [
                {
                    "season": _SEASON,
                    "player_key": 42,
                    "element_id": 1,
                    "team_key": 1,
                    "position": "MID",
                    "web_name": "Test",
                    "available_at": _TS,
                    "source": "test",
                }
            ]
        ),
    )
    store.write(
        "player_status",
        pl.DataFrame(
            [
                {
                    "season": _SEASON,
                    "player_key": 42,
                    "status": "a",
                    "news": "",
                    "news_added": None,
                    "price": 70,
                    "ep_next": 4.0,
                    "chance_of_playing": 100.0,
                    "chance_of_playing_this_round": 100.0,
                    "penalties_order": None,
                    "corners_and_indirect_freekicks_order": None,
                    "direct_freekicks_order": None,
                    "available_at": _TS,
                    "source": "test",
                }
            ]
        ),
    )
    run_id = _seed_predict_run(tmp_path, player_key=42)
    dense = np.zeros(_WIDTH, dtype=np.float64)
    dense[5 - POINTS_PMF_MIN] = 0.7
    dense[2 - POINTS_PMF_MIN] = 0.3
    pred_dir = run_dir(tmp_path, run_id) / "predictions"
    pl.DataFrame(
        {
            "player_key": [42, 42],
            "fixture_id": [1, 1],
            "points_bin": [5, 2],
            "probability": [0.7, 0.3],
            "target_season": [_SEASON, _SEASON],
            "target_gw": [1, 1],
        }
    ).write_parquet(pred_dir / "points_pmf.parquet")
    del dense
    decisions = (
        _decision_gw(
            xi_pts=[2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 8],
            captain_index=10,
            n_fixtures=10,
        )
        .with_columns(
            pl.lit(1).cast(pl.Int64).alias("gw"),
            pl.lit(_SEASON).alias("season"),
            pl.lit(_MODEL).alias("model_id"),
        )
        .with_columns(
            pl.when(pl.col("player_key") == 1)
            .then(pl.lit(42))
            .otherwise(pl.col("player_key"))
            .alias("player_key")
        )
    )
    result = run_post_gw(
        tmp_path,
        _KEY,
        build=_build(),
        ops=_ops(),
        decisions=decisions,
        notify=False,
        push=False,
        rank_snapshot=None,
        n_fixtures=10,
    )
    assert result.skipped is False
    assert result.run_id is not None
    out = run_dir(tmp_path, result.run_id)
    scores = pl.read_parquet(out / "forecast_scores.parquet")
    agg = pl.read_parquet(out / "forecast_score_agg.parquet")
    assert scores.height >= 1
    assert scores["player_key"][0] == 42
    assert "primary" in str(scores["selected_lines"][0])
    assert not agg.is_empty()
    summary = (out / "summary.md").read_text(encoding="utf-8")
    assert "Forecast scores" in summary


def test_summary_and_ntfy_include_rps() -> None:
    """Summary / ntfy mention STD RPS when aggregations are present."""
    scorecard = pl.DataFrame(
        {
            "model_id": [_MODEL],
            "season": [_SEASON],
            "gw": [1],
            "gw_kind": ["normal"],
            "plan_ev": [50.0],
            "realised_points": [48],
            "realised_minus_plan_ev": [-2.0],
            "pmf_percentile": [0.4],
            "fpl_average": [None],
            "fpl_highest": [None],
            "rank_change": [None],
            "season_points_to_date": [48],
            "points_at_rank": [None],
            "percentile": [None],
            "realised_ex_chips": [48],
            "chip": [None],
        }
    )
    std = pl.DataFrame(
        {
            "model_id": [_MODEL],
            "through_gw": [1],
            "season_points": [48],
            "plan_ev_sum": [50.0],
        }
    )
    attribution = pl.DataFrame(
        schema={
            "model_id": pl.Utf8(),
            "season": pl.Utf8(),
            "gw": pl.Int64(),
            "gw_kind": pl.Utf8(),
            "plan_ev": pl.Float64(),
            "realised_points": pl.Int64(),
            "luck_vs_model": pl.Float64(),
            "captaincy_regret": pl.Int64(),
            "xi_bench_regret": pl.Int64(),
            "transfer_regret": pl.Int64(),
            "chip_timing_regret": pl.Int64(),
            "hindsight_best": pl.Int64(),
            "regret": pl.Int64(),
            "hit": pl.Float64(),
            "hit_payback_to_date": pl.Float64(),
            "chip_value": pl.Float64(),
            "intervention_delta": pl.Float64(),
        }
    )
    agg = pl.DataFrame(
        {
            "model_id": [_MODEL],
            "season": [_SEASON],
            "gw": [None],
            "slice": ["all"],
            "metric": ["mean_rps"],
            "value": [0.42],
            "ci_lo": [0.3],
            "ci_hi": [0.5],
            "n": [10],
            "schema_version": [1],
        }
    )
    text = format_post_gw_summary(scorecard, std, attribution, forecast_agg=agg)
    assert "RPS" in text
    ntfy = format_post_gw_ntfy(scorecard, std, forecast_agg=agg)
    assert "RPS 0.420" in ntfy


def test_pmf_helpers_reject_wrong_width() -> None:
    """Wrong-length dense PMFs raise ValueError on PIT and event probs."""
    bad = np.array([0.5, 0.5], dtype=np.float64)
    with pytest.raises(ValueError, match="length"):
        randomised_pit(bad, 0, np.random.default_rng(0))
    with pytest.raises(ValueError, match="length"):
        pmf_event_probs(bad)


def test_as_float_defaults() -> None:
    """``_as_float`` maps None / bool / non-numeric to the default."""
    assert fs_mod._as_float(None) == 0.0
    assert fs_mod._as_float(True, default=3.0) == 3.0
    assert fs_mod._as_float("x", default=1.5) == 1.5
    assert fs_mod._as_float(2) == 2.0


def test_fit_climatology_errors_and_empty() -> None:
    """Missing columns raise; empty history returns an empty long PMF."""
    with pytest.raises(ValueError, match="missing"):
        fit_climatology(pl.DataFrame({"position": ["MID"]}))
    empty = fit_climatology(
        pl.DataFrame(
            schema={
                "position": pl.Utf8(),
                "price_band": pl.Utf8(),
                "minutes_bucket": pl.Utf8(),
                "pts": pl.Int64(),
            }
        )
    )
    assert empty.is_empty()


def test_selection_bias_empty_paths() -> None:
    """Empty scores and no selected rows return zero / vs-all delta."""
    assert selection_conditional_bias(empty_forecast_scores()) == (0.0, 0.0)
    scores = pl.DataFrame(
        {
            "mu_hat": [5.0, 4.0],
            "y": [3, 4],
            "selected_lines": ["", ""],
        }
    )
    bias, delta = selection_conditional_bias(scores, line="primary")
    assert bias == 0.0
    assert delta == pytest.approx(0.0 - float(np.mean([2.0, 0.0])))


def test_compute_forecast_scores_empty_branches() -> None:
    """Empty PMF / outcomes / wrong GW / unmatched fixtures yield empty."""
    assert compute_forecast_scores(
        model_id=_MODEL,
        key=_KEY,
        points_pmf=pl.DataFrame(),
        predictions=pl.DataFrame(),
        outcomes=pl.DataFrame(),
    ).is_empty()
    outcomes = pl.DataFrame(
        {
            "season": ["2023-24"],
            "gw": [1],
            "player_key": [1],
            "fixture_id": [1],
            "pts": [1],
            "minutes": [90],
        }
    )
    assert compute_forecast_scores(
        model_id=_MODEL,
        key=_KEY,
        points_pmf=_long_pmf(1, 1, _delta(1)),
        predictions=pl.DataFrame(),
        outcomes=outcomes,
    ).is_empty()
    # PMF filtered out by season/gw aliases.
    pmf = pl.DataFrame(
        {
            "player_key": [1],
            "fixture_id": [1],
            "points_bin": [1],
            "probability": [1.0],
            "season": ["2023-24"],
            "gw": [2],
        }
    )
    outcomes2 = pl.DataFrame(
        {
            "season": [_SEASON],
            "gw": [1],
            "player_key": [1],
            "fixture_id": [1],
            "actual_pts": [1],
            "minutes": [0],
        }
    )
    assert compute_forecast_scores(
        model_id=_MODEL,
        key=_KEY,
        points_pmf=pmf,
        predictions=pl.DataFrame(),
        outcomes=outcomes2,
    ).is_empty()
    # Unmatched fixture skipped → empty rows.
    pmf2 = _long_pmf(99, 99, _delta(2))
    assert compute_forecast_scores(
        model_id=_MODEL,
        key=_KEY,
        points_pmf=pmf2,
        predictions=pl.DataFrame(),
        outcomes=outcomes2,
    ).is_empty()


def test_aggregate_empty_and_filters() -> None:
    """Empty / filtered-out frames return an empty aggregation."""
    assert aggregate_forecast_scores(empty_forecast_scores()).is_empty()
    one = pl.DataFrame(
        {
            "model_id": ["other"],
            "season": ["2023-24"],
            "gw": [1],
            "player_key": [1],
            "fixture_id": [1],
            "position": ["MID"],
            "price_band": ["mid"],
            "minutes_bucket": ["60+"],
            "y": [1],
            "mu_hat": [1.0],
            "rps": [0.1],
            "log_score": [0.1],
            "rps_clim": [None],
            "pit_u": [0.5],
            "p_played": [0.5],
            "p_ge60": [0.5],
            "p_goal": [0.1],
            "p_assist": [0.1],
            "p_cs": [0.1],
            "p_haul": [0.1],
            "p_blank": [0.2],
            "y_played": [1],
            "y_ge60": [1],
            "y_goal": [0],
            "y_assist": [0],
            "y_cs": [0],
            "y_haul": [0],
            "y_blank": [0],
            "selected_lines": [""],
            "schema_version": [1],
        }
    )
    assert aggregate_forecast_scores(
        one, model_id=_MODEL, season=_SEASON, lines=("shadow",)
    ).is_empty()


def test_attach_meta_defaults_and_prediction_aliases() -> None:
    """Meta defaults fill position/price_band; season/gw aliases filter preds."""
    outcomes = pl.DataFrame(
        {
            "season": [_SEASON],
            "gw": [1],
            "player_key": [7],
            "fixture_id": [8],
            "pts": [3],
            "minutes": [45],
            "goals": [0],
            "assists": [1],
            "clean_sheets": [0],
        }
    )
    pmf = _long_pmf(7, 8, _delta(3))
    preds = pl.DataFrame(
        {
            "season": [_SEASON],
            "gw": [1],
            "player_key": [7],
            "fixture_id": [8],
            "xpts": [2.5],
        }
    )
    scores = compute_forecast_scores(
        model_id=_MODEL,
        key=_KEY,
        points_pmf=pmf,
        predictions=preds,
        outcomes=outcomes,
        meta=pl.DataFrame({"player_key": [7], "position": ["DEF"], "price": [50]}),
    )
    assert scores["position"][0] == "DEF"
    assert scores["price_band"][0] == "budget"
    assert scores["y_assist"][0] == 1
    # Empty predictions path + no meta → defaults.
    scores2 = compute_forecast_scores(
        model_id=_MODEL,
        key=_KEY,
        points_pmf=pmf,
        predictions=pl.DataFrame(),
        outcomes=outcomes,
    )
    assert scores2["position"][0] == "MID"
    assert scores2["price_band"][0] == "mid"


def test_spearman_and_discrimination_edges() -> None:
    """Spearman degenerates; MZ skipped for short frames; top-k when n≥2."""
    assert np.isnan(fs_mod._spearman(np.array([1.0]), np.array([2.0])))
    assert np.isnan(fs_mod._spearman(np.array([1.0, 1.0]), np.array([3.0, 4.0])))
    rows = []
    for i, (mu, y) in enumerate([(1.0, 1), (2.0, 3), (3.0, 2), (4.0, 5)]):
        rows.append(
            {
                "model_id": _MODEL,
                "season": _SEASON,
                "gw": 1,
                "player_key": i,
                "fixture_id": 100 + i,
                "position": "MID",
                "price_band": "mid",
                "minutes_bucket": "60+",
                "y": y,
                "mu_hat": mu,
                "rps": 0.1,
                "log_score": 0.1,
                "rps_clim": 0.2,
                "pit_u": 0.5,
                "p_played": 0.9,
                "p_ge60": 0.8,
                "p_goal": 0.2,
                "p_assist": 0.2,
                "p_cs": 0.1,
                "p_haul": 0.1,
                "p_blank": 0.2,
                "y_played": 1,
                "y_ge60": 1,
                "y_goal": 0,
                "y_assist": 0,
                "y_cs": 0,
                "y_haul": 0,
                "y_blank": 0,
                "selected_lines": "",
                "schema_version": 1,
            }
        )
    frame = pl.DataFrame(rows)
    agg = aggregate_forecast_scores(frame, bootstrap_reps=50, bootstrap_seed=0)
    assert "spearman_within_pos" in agg["metric"].to_list()
    assert "mz_slope" in agg["metric"].to_list()
    assert "top5_hit_rate" in agg["metric"].to_list()
    # Short frame: MZ empty, discrimination skips singleton positions.
    short = frame.head(2).with_columns(
        pl.Series("position", ["GK", "DEF"]),
        pl.Series("player_key", [10, 11]),
    )
    assert (
        fs_mod._mincer_zarnowitz(
            short, model_id=_MODEL, season=_SEASON, slice_name="all"
        )
        == []
    )
    disc = fs_mod._discrimination(
        short, model_id=_MODEL, season=_SEASON, slice_name="all"
    )
    assert all(r["metric"] != "spearman_within_pos" for r in disc)


def test_event_probs_fallback_and_null_pred() -> None:
    """Missing / null prediction probs fall back to derived or 0."""
    derived = {"p_haul": 0.2, "p_blank": 0.3, "mu_hat": 4.0}
    empty = fs_mod._event_probs(pl.DataFrame(), derived)
    assert empty["p_haul"] == 0.2
    pred = pl.DataFrame(
        {
            "p_played": [None],
            "p_60": [0.55],
            "p_goal": [float("nan")],
            "p_assist": [0.1],
            "p_cs": [0.2],
            "p_haul": [None],
        }
    )
    probs = fs_mod._event_probs(pred, derived)
    assert probs["p_ge60"] == pytest.approx(0.55)
    assert probs["p_haul"] == pytest.approx(0.2)
    assert probs["p_goal"] == 0.0


def test_forecast_summary_lines_variants() -> None:
    """Summary lines cover CI-present, CI-absent, bias, and empty fallback."""
    with_ci = pl.DataFrame(
        {
            "slice": ["all", "all", "line=primary"],
            "metric": ["mean_rps", "pit_coverage_90", "selection_bias"],
            "value": [0.5, 0.9, 0.2],
            "ci_lo": [0.4, None, None],
            "ci_hi": [0.6, None, None],
        }
    )
    lines = post_gw_mod._forecast_summary_lines(with_ci)
    assert any("[" in line for line in lines)
    assert any("selection bias" in line for line in lines)
    assert post_gw_mod._forecast_summary_lines(empty_forecast_score_agg()) == [
        "_No forecast aggregations._"
    ]


def test_post_gw_forecast_helpers_edge_paths(tmp_path: Path) -> None:
    """Pointer / store / prior-load edge paths return empty frames."""
    key = _KEY
    # No pointer → empty PMF / early _forecast_eval return.
    scores, agg = post_gw_mod._forecast_eval(
        tmp_path,
        key,
        model_id=_MODEL,
        line="primary",
        decisions=pl.DataFrame({"gw": [1], "player_key": [1], "sold": [False]}),
    )
    assert scores.is_empty()
    assert agg.is_empty()
    assert post_gw_mod._official_points_pmf(tmp_path, key, line="primary").is_empty()

    base = official_checkpoint_dir(tmp_path, season=_SEASON, gw=1, checkpoint="T-90m")
    base.mkdir(parents=True)
    (base / "line=primary.json").write_text("{bad", encoding="utf-8")
    assert post_gw_mod._official_points_pmf(tmp_path, key, line="primary").is_empty()
    # Valid JSON but missing parquet under the predict run.
    (base / "line=primary.json").write_text(
        '{"predict_run_id":"' + ("r" * 16) + '"}', encoding="utf-8"
    )
    assert post_gw_mod._official_points_pmf(tmp_path, key, line="primary").is_empty()

    store = DataStore(tmp_path)
    meta = post_gw_mod._player_meta(store, _SEASON)
    assert meta.is_empty() or meta.height == 0
    store.write(
        "players",
        pl.DataFrame(
            [
                {
                    "season": _SEASON,
                    "player_key": 1,
                    "element_id": 1,
                    "team_key": 1,
                    "position": "MID",
                    "web_name": "A",
                    "available_at": _TS,
                    "source": "test",
                }
            ]
        ),
    )
    meta2 = post_gw_mod._player_meta(store, _SEASON)
    assert meta2["price"][0] == 60

    assert post_gw_mod._climatology_from_store(
        store, key, meta=meta2
    ).is_empty()  # no outcomes
    store.write(
        "player_fixture_outcomes",
        pl.DataFrame(
            [
                {
                    "season": _SEASON,
                    "gw": 1,
                    "fixture_id": 1,
                    "player_key": 1,
                    "team_key": 1,
                    "pts": 4,
                    "minutes": 90,
                    "goals": 0,
                    "assists": 0,
                    "clean_sheets": 0,
                    "available_at": _TS,
                    "schema_version": 1,
                    "source": "test",
                }
            ]
        ),
    )
    # Same-season only → prior empty.
    assert post_gw_mod._climatology_from_store(store, key, meta=meta2).is_empty()
    store.write(
        "player_fixture_outcomes",
        pl.DataFrame(
            [
                {
                    "season": "2023-24",
                    "gw": 1,
                    "fixture_id": 1,
                    "player_key": 1,
                    "team_key": 1,
                    "pts": 4,
                    "minutes": 90,
                    "goals": 0,
                    "assists": 0,
                    "clean_sheets": 0,
                    "available_at": _TS,
                    "schema_version": 1,
                    "source": "test",
                }
            ]
        ),
    )
    clim = post_gw_mod._climatology_from_store(store, key, meta=meta2)
    assert not clim.is_empty()

    assert post_gw_mod._prior_forecast_scores(tmp_path, key, model_id=_MODEL).is_empty()
    run = tmp_path / "runs" / "run=abc"
    run.mkdir(parents=True)
    empty_forecast_scores().write_parquet(run / "forecast_scores.parquet")
    assert post_gw_mod._prior_forecast_scores(tmp_path, key, model_id=_MODEL).is_empty()
    # Prior GW row should load for STD.
    prior_row = pl.DataFrame(
        {
            "model_id": [_MODEL],
            "season": [_SEASON],
            "gw": [1],
            "player_key": [1],
            "fixture_id": [1],
            "position": ["MID"],
            "price_band": ["mid"],
            "minutes_bucket": ["60+"],
            "y": [1],
            "mu_hat": [1.0],
            "rps": [0.1],
            "log_score": [0.1],
            "rps_clim": [None],
            "pit_u": [0.5],
            "p_played": [0.5],
            "p_ge60": [0.5],
            "p_goal": [0.0],
            "p_assist": [0.0],
            "p_cs": [0.0],
            "p_haul": [0.0],
            "p_blank": [0.5],
            "y_played": [1],
            "y_ge60": [1],
            "y_goal": [0],
            "y_assist": [0],
            "y_cs": [0],
            "y_haul": [0],
            "y_blank": [1],
            "selected_lines": [""],
            "schema_version": [1],
        }
    )
    prior_row.write_parquet(run / "forecast_scores.parquet")
    loaded = post_gw_mod._prior_forecast_scores(
        tmp_path, GWKey(_SEASON, 2), model_id=_MODEL
    )
    assert loaded.height == 1
    # OSError while reading a prior file is skipped.
    bad = tmp_path / "runs" / "run=bad"
    bad.mkdir()
    (bad / "forecast_scores.parquet").write_bytes(b"")
    from unittest.mock import patch

    real_read = pl.read_parquet

    def _read(path: object, *args: object, **kwargs: object) -> pl.DataFrame:
        if "run=bad" in str(path):
            raise OSError("boom")
        return real_read(path, *args, **kwargs)  # type: ignore[arg-type]

    with patch("fplt.automation.post_gw.pl.read_parquet", side_effect=_read):
        loaded2 = post_gw_mod._prior_forecast_scores(
            tmp_path, GWKey(_SEASON, 2), model_id=_MODEL
        )
    assert loaded2.height == 1


def test_forecast_eval_missing_outcomes(tmp_path: Path) -> None:
    """When outcomes cannot be read, ``_forecast_eval`` returns empties."""
    from unittest.mock import patch

    from tests.unit.test_biggest_misses import _seed_predict_run

    run_id = _seed_predict_run(tmp_path, player_key=42)
    pred_dir = run_dir(tmp_path, run_id) / "predictions"
    pl.DataFrame(
        {
            "player_key": [42],
            "fixture_id": [1],
            "points_bin": [5],
            "probability": [1.0],
            "target_season": [_SEASON],
            "target_gw": [1],
        }
    ).write_parquet(pred_dir / "points_pmf.parquet")
    with patch.object(DataStore, "read", side_effect=KeyError("missing")):
        scores, agg = post_gw_mod._forecast_eval(
            tmp_path,
            _KEY,
            model_id=_MODEL,
            line="primary",
            decisions=pl.DataFrame({"gw": [1], "player_key": [42]}),
        )
    assert scores.is_empty()
    assert agg.is_empty()


def test_mincer_zarnowitz_linalg_error() -> None:
    """LinAlgError in the MZ fit returns no rows."""
    from unittest.mock import patch

    frame = pl.DataFrame(
        {
            "y": [1.0, 2.0, 3.0],
            "mu_hat": [1.0, 2.0, 3.0],
        }
    )
    with patch(
        "fplt.backtest.forecast_scores.np.linalg.lstsq",
        side_effect=np.linalg.LinAlgError("boom"),
    ):
        assert (
            fs_mod._mincer_zarnowitz(
                frame, model_id=_MODEL, season=_SEASON, slice_name="all"
            )
            == []
        )
