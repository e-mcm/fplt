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
