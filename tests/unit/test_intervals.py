"""Unit tests for promote-eval paired intervals (§9)."""

from __future__ import annotations

import numpy as np
import polars as pl
import pytest

from fplt.backtest.intervals import (
    equivalence_drop_from_intervals,
    interval_entirely_bad,
    intervals_for_pair,
    loss_differential,
    mcs_set,
    multi_arm_selection,
    paired_interval,
    per_gw_losses,
    per_gw_wins_from_losses,
    spa_test,
)


def _ledger_rows(
    *,
    model_id: str,
    gws: list[int],
    xpts: list[float],
    actual: list[float],
    position: str = "MID",
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for i, gw in enumerate(gws):
        rows.append(
            {
                "model_id": model_id,
                "run_kind": "walk_forward",
                "as_of_season": "2024-25",
                "as_of_gw": gw,
                "target_season": "2024-25",
                "target_gw": gw,
                "horizon": 1,
                "player_key": 1000 + i,
                "position": position,
                "fixture_id": 10_000 + gw,
                "xpts": xpts[i],
                "xpts_sd": None,
                "p_haul": None,
                "p_60": None,
                "actual_pts": int(actual[i]),
                "actual_minutes": 90,
            }
        )
    return rows


def test_per_gw_losses_mae_and_differential() -> None:
    control = pl.DataFrame(
        _ledger_rows(
            model_id="c",
            gws=[1, 2, 3, 4],
            xpts=[2.0, 3.0, 4.0, 5.0],
            actual=[1.0, 1.0, 1.0, 1.0],
        )
    )
    treatment = pl.DataFrame(
        _ledger_rows(
            model_id="t",
            gws=[1, 2, 3, 4],
            xpts=[1.5, 2.0, 3.0, 4.0],
            actual=[1.0, 1.0, 1.0, 1.0],
        )
    )
    c_loss = per_gw_losses(control, model_id="c", metric="mae")
    t_loss = per_gw_losses(treatment, model_id="t", metric="mae")
    assert c_loss.height == 4
    d = loss_differential(c_loss, t_loss)
    # control abs errors 1,2,3,4 vs treatment 0.5,1,2,3 → d = 0.5,1,1,1
    assert d.shape == (4,)
    assert np.allclose(d, [0.5, 1.0, 1.0, 1.0])
    wins = per_gw_wins_from_losses(c_loss, t_loss)
    assert wins == {"control": 0, "treatment": 4, "ties": 0}


def test_paired_interval_positive_mean_and_seed_stable() -> None:
    d = np.array([0.2, 0.1, 0.15, 0.05, 0.25, 0.1, 0.2, 0.05], dtype=np.float64)
    a = paired_interval(d, metric="mae", bootstrap_reps=500, bootstrap_seed=7)
    b = paired_interval(d, metric="mae", bootstrap_reps=500, bootstrap_seed=7)
    assert a.n_gw == 8
    assert a.mean == pytest.approx(float(d.mean()))
    assert a.hac_se is not None and a.hac_se > 0
    assert a.hac_pvalue is not None
    assert a.hac_ci_low is not None and a.hac_ci_high is not None
    assert a.bootstrap_ci_low is not None and a.bootstrap_ci_high is not None
    assert a.hac_ci_low < a.mean < a.hac_ci_high
    assert a.bootstrap_ci_low == b.bootstrap_ci_low
    assert a.bootstrap_ci_high == b.bootstrap_ci_high
    assert not a.entirely_bad


def test_interval_entirely_bad_and_equivalence_drop() -> None:
    assert interval_entirely_bad(-0.5, -0.1, -0.4, -0.05) is True
    assert interval_entirely_bad(-0.5, 0.1, -0.4, -0.05) is False
    assert interval_entirely_bad(None, None, None, None) is False
    assert interval_entirely_bad(None, -0.1, None, None) is True

    bad = paired_interval(
        np.full(12, -0.3),
        metric="mae",
        bootstrap_reps=200,
        bootstrap_seed=0,
    )
    good = paired_interval(
        np.full(12, 0.3),
        metric="rmse",
        bootstrap_reps=200,
        bootstrap_seed=0,
    )
    assert bad.entirely_bad is True
    assert good.entirely_bad is False
    assert equivalence_drop_from_intervals([bad, good]) is True
    assert equivalence_drop_from_intervals([good]) is False


def test_intervals_for_pair_from_ledger() -> None:
    rows = _ledger_rows(
        model_id="c",
        gws=[1, 2, 3, 4, 5],
        xpts=[3.0, 3.0, 3.0, 3.0, 3.0],
        actual=[1.0, 1.0, 1.0, 1.0, 1.0],
    ) + _ledger_rows(
        model_id="t",
        gws=[1, 2, 3, 4, 5],
        xpts=[1.5, 1.5, 1.5, 1.5, 1.5],
        actual=[1.0, 1.0, 1.0, 1.0, 1.0],
    )
    ledger = pl.DataFrame(rows)
    intervals = intervals_for_pair(
        ledger,
        control_id="c",
        treatment_id="t",
        metrics=("mae",),
        bootstrap_reps=200,
        bootstrap_seed=1,
    )
    assert len(intervals) == 1
    assert intervals[0].metric == "mae"
    assert intervals[0].mean > 0
    assert intervals[0].n_gw == 5


def test_spearman_loss_is_negated() -> None:
    # Perfect rank agreement → spearman 1 → loss -1.
    rows = []
    for i, pts in enumerate([1, 2, 3, 4]):
        rows.append(
            {
                "model_id": "m",
                "run_kind": "walk_forward",
                "as_of_season": "2024-25",
                "as_of_gw": 1,
                "target_season": "2024-25",
                "target_gw": 1,
                "horizon": 1,
                "player_key": i,
                "position": "MID",
                "fixture_id": 1,
                "xpts": float(pts),
                "xpts_sd": None,
                "p_haul": None,
                "p_60": None,
                "actual_pts": pts,
                "actual_minutes": 90,
            }
        )
    losses = per_gw_losses(pl.DataFrame(rows), model_id="m", metric="spearman")
    assert losses.height == 1
    assert losses["loss"][0] == pytest.approx(-1.0)


def test_spa_and_mcs_multi_arm() -> None:
    rng = np.random.default_rng(0)
    t = 40
    # Control worse than two treatments on average.
    control = rng.normal(1.0, 0.2, size=t)
    better = control - 0.3 + rng.normal(0, 0.05, size=t)
    also = control - 0.25 + rng.normal(0, 0.05, size=t)
    spa = spa_test(control, np.column_stack([better, also]), reps=200, seed=0)
    assert 0.0 <= spa.pvalue_consistent <= 1.0
    assert spa.n_models == 2
    mcs = mcs_set(
        np.column_stack([control, better, also]),
        names=("control", "better", "also"),
        reps=200,
        seed=0,
    )
    assert set(mcs.included) | set(mcs.excluded) == {"control", "better", "also"}
    selection = multi_arm_selection(
        {"control": control, "better": better, "also": also},
        control="control",
        reps=200,
        seed=0,
    )
    assert selection is not None
    # Two arms only → no SPA/MCS.
    assert (
        multi_arm_selection(
            {"control": control, "better": better},
            control="control",
        )
        is None
    )


def test_spa_rejects_short_series() -> None:
    with pytest.raises(ValueError, match="T>=2"):
        spa_test([1.0], [[0.5]], reps=10, seed=0)


def test_empty_ledger_returns_empty_losses() -> None:
    empty = pl.DataFrame(
        schema={
            "model_id": pl.Utf8(),
            "horizon": pl.Int64(),
            "actual_pts": pl.Int64(),
            "target_season": pl.Utf8(),
            "target_gw": pl.Int64(),
            "xpts": pl.Float64(),
            "position": pl.Utf8(),
            "player_key": pl.Int64(),
            "p_haul": pl.Float64(),
            "p_60": pl.Float64(),
            "actual_minutes": pl.Int64(),
        }
    )
    losses = per_gw_losses(empty, model_id="missing", metric="mae")
    assert losses.is_empty()
    d = loss_differential(losses, losses)
    assert d.size == 0
    interval = paired_interval(d, metric="mae")
    assert interval.n_gw == 0
    assert interval.hac_se is None


def test_loss_differential_no_overlap() -> None:
    control = pl.DataFrame(
        {"target_season": ["2024-25"], "target_gw": [1], "loss": [1.0]}
    )
    treatment = pl.DataFrame(
        {"target_season": ["2024-25"], "target_gw": [2], "loss": [0.5]}
    )
    assert loss_differential(control, treatment).size == 0


def test_spa_1d_reshape_and_empty_alternatives() -> None:
    bench = np.array([1.0, 1.1, 0.9], dtype=np.float64)
    alt = np.array([0.8, 0.9, 0.7], dtype=np.float64)
    spa = spa_test(bench, alt, reps=50, seed=1)
    assert spa.n_models == 1
    with pytest.raises(ValueError, match="at least one alternative"):
        spa_test(bench, np.zeros((3, 0)), reps=10, seed=0)


def test_mcs_validation_errors() -> None:
    with pytest.raises(ValueError, match="T>=2"):
        mcs_set(np.array([[1.0, 2.0]]), names=("a", "b"), reps=10, seed=0)
    with pytest.raises(ValueError, match="columns must match"):
        mcs_set(
            np.array([[1.0, 2.0], [1.1, 2.1]]),
            names=("a",),
            reps=10,
            seed=0,
        )
    with pytest.raises(ValueError, match="at least two arms"):
        mcs_set(np.array([1.0, 1.1]), names=("a",), reps=10, seed=0)
    # 1-d with matching single name still fails the two-arm rule after reshape.
    with pytest.raises(ValueError, match="at least two arms"):
        mcs_set(np.array([1.0, 1.1, 1.2]), names=("only",), reps=10, seed=0)


def test_multi_arm_selection_errors() -> None:
    series = np.array([1.0, 1.1, 0.9], dtype=np.float64)
    with pytest.raises(ValueError, match="missing from loss_by_arm"):
        multi_arm_selection({"a": series, "b": series, "c": series}, control="z")
    with pytest.raises(ValueError, match="same length"):
        multi_arm_selection(
            {
                "control": series,
                "b": series,
                "c": np.array([1.0, 1.1], dtype=np.float64),
            },
            control="control",
        )


def test_per_gw_wins_control_ties_and_empty() -> None:
    empty = pl.DataFrame(
        schema={
            "target_season": pl.Utf8(),
            "target_gw": pl.Int64(),
            "loss": pl.Float64(),
        }
    )
    assert per_gw_wins_from_losses(empty, empty) == {
        "control": 0,
        "treatment": 0,
        "ties": 0,
    }
    control = pl.DataFrame(
        {
            "target_season": ["2024-25"] * 3,
            "target_gw": [1, 2, 3],
            "loss": [0.5, 1.0, 1.0],
        }
    )
    treatment = pl.DataFrame(
        {
            "target_season": ["2024-25"] * 3,
            "target_gw": [1, 2, 3],
            "loss": [1.0, 0.5, 1.0],
        }
    )
    wins = per_gw_wins_from_losses(control, treatment)
    assert wins == {"control": 1, "treatment": 1, "ties": 1}


def test_top10_brier_and_unknown_metric() -> None:
    from fplt.backtest.intervals import _gw_loss

    rows = []
    for i in range(12):
        rows.append(
            {
                "model_id": "m",
                "run_kind": "walk_forward",
                "as_of_season": "2024-25",
                "as_of_gw": 1,
                "target_season": "2024-25",
                "target_gw": 1,
                "horizon": 1,
                "player_key": i,
                "position": "MID",
                "fixture_id": 1,
                "xpts": float(12 - i),
                "xpts_sd": None,
                "p_haul": 0.2 if i < 5 else None,
                "p_60": 0.8 if i < 5 else None,
                "actual_pts": 12 - i,
                "actual_minutes": 90 if i < 5 else None,
            }
        )
    frame = pl.DataFrame(rows)
    top10 = per_gw_losses(frame, model_id="m", metric="top10_precision")
    assert top10.height == 1
    assert top10["loss"][0] == pytest.approx(-1.0)

    haul = per_gw_losses(frame, model_id="m", metric="brier_haul")
    assert haul.height == 1
    sixty = per_gw_losses(frame, model_id="m", metric="brier_60")
    assert sixty.height == 1

    # Empty probability columns → loss skipped → empty frame.
    no_prob = frame.with_columns(
        pl.lit(None).cast(pl.Float64).alias("p_haul"),
        pl.lit(None).cast(pl.Float64).alias("p_60"),
        pl.lit(None).cast(pl.Int64).alias("actual_minutes"),
    )
    assert per_gw_losses(no_prob, model_id="m", metric="brier_haul").is_empty()
    assert per_gw_losses(no_prob, model_id="m", metric="brier_60").is_empty()

    with pytest.raises(ValueError, match="unknown metric"):
        _gw_loss(frame, "not_a_metric")  # type: ignore[arg-type]

    assert _gw_loss(pl.DataFrame(), "mae") is None


def test_mean_num_non_numeric_fallback() -> None:
    from fplt.backtest.intervals import _mean_num

    assert _mean_num(pl.Series([None, None])) == 0.0
