"""Signal screen: PIT guard, BH-FDR, planted-signal ranking."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest

from fplt.backtest.signal_screen import (
    DERIVED_COVARIATES,
    apply_pit_guard,
    benjamini_hochberg,
    residuals_from_ledger,
    screen_signals,
    unused_staged_tables,
)
from fplt.schemas.ledger import LEDGER_COLUMNS
from fplt.schemas.signal_screen import SIGNAL_SCREEN_COLUMNS, SignalScreenSchema


def test_benjamini_hochberg_orders_and_caps() -> None:
    q, rejected = benjamini_hochberg([0.001, 0.01, 0.5, 0.9], alpha=0.10)
    assert q[0] <= q[1] <= 1.0
    assert bool(rejected[0]) is True
    assert bool(rejected[3]) is False


def test_benjamini_hochberg_handles_nan_and_empty() -> None:
    q, rejected = benjamini_hochberg([])
    assert q.size == 0
    assert rejected.size == 0
    q2, rejected2 = benjamini_hochberg([np.nan, 0.001])
    assert np.isnan(q2[0])
    assert bool(rejected2[0]) is False
    assert bool(rejected2[1]) is True


def test_bh_on_synthetic_nulls_keeps_fdr_at_most_10_percent() -> None:
    """Under a complete null, BH discoveries stay ≤ 10% (seeded)."""
    rng = np.random.default_rng(402)
    n_rows = 240
    n_null = 80
    residual = rng.normal(size=n_rows)
    mu_hat = rng.normal(size=n_rows)
    panel = _panel(residual=residual, mu_hat=mu_hat, seed=1)
    covariates = {
        f"null_{i}": pl.Series(f"null_{i}", rng.normal(size=n_rows))
        for i in range(n_null)
    }
    screened = screen_signals(panel, covariates, fdr_alpha=0.10)
    assert screened.height == n_null
    rate = screened["rejected"].mean()
    assert rate is not None
    assert float(rate) <= 0.10


def test_planted_signal_ranks_first() -> None:
    rng = np.random.default_rng(7)
    n_rows = 300
    signal = rng.normal(size=n_rows)
    noise = rng.normal(size=n_rows) * 0.25
    residual = 1.5 * signal + noise
    panel = _panel(residual=residual, mu_hat=rng.normal(size=n_rows), seed=2)
    covariates: dict[str, pl.Series] = {
        "planted": pl.Series("planted", signal),
    }
    for i in range(20):
        covariates[f"null_{i}"] = pl.Series(f"null_{i}", rng.normal(size=n_rows))
    screened = screen_signals(panel, covariates, fdr_alpha=0.10)
    assert screened["covariate"].to_list()[0] == "planted"
    assert screened["rank"].to_list()[0] == 1
    assert bool(screened["rejected"].to_list()[0]) is True


def test_pit_guard_nulls_values_after_cutoff() -> None:
    cutoff = datetime(2025, 1, 1, 12, 0, tzinfo=UTC)
    values = pl.Series("x", [1.0, 2.0, 3.0])
    available = pl.Series(
        "available_at",
        [
            cutoff - timedelta(hours=1),
            cutoff,
            cutoff + timedelta(minutes=1),
        ],
    )
    guarded = apply_pit_guard(values, available, cutoff)
    assert guarded.to_list() == [1.0, 2.0, None]


def test_pit_guard_per_row_cutoff_and_length_check() -> None:
    values = pl.Series("x", [10.0, 20.0])
    available = pl.Series(
        "available_at",
        [
            datetime(2025, 1, 1, tzinfo=UTC),
            datetime(2025, 1, 3, tzinfo=UTC),
        ],
    )
    cutoffs = pl.Series(
        "cutoff",
        [
            datetime(2025, 1, 2, tzinfo=UTC),
            datetime(2025, 1, 2, tzinfo=UTC),
        ],
    )
    guarded = apply_pit_guard(values, available, cutoffs)
    assert guarded.to_list() == [10.0, None]
    with pytest.raises(ValueError, match="same length"):
        apply_pit_guard(values, available[:1], cutoffs)


def test_pit_guard_stops_leaky_covariate_from_screening() -> None:
    """Correlation that exists only after the cut-off must not pass the screen."""
    rng = np.random.default_rng(11)
    n_rows = 160
    half = n_rows // 2
    residual = rng.normal(size=n_rows)
    panel = _panel(residual=residual, mu_hat=rng.normal(size=n_rows), seed=3)
    cutoff = datetime(2025, 6, 1, tzinfo=UTC)
    # Pre-cutoff: pure noise. Post-cutoff: identical to residual (would leak).
    leak = np.concatenate([rng.normal(size=half), residual[half:]])
    available = pl.Series(
        "available_at",
        [cutoff - timedelta(days=1)] * half + [cutoff + timedelta(days=1)] * half,
    )
    guarded = apply_pit_guard(pl.Series("leak", leak), available, cutoff)
    screened = screen_signals(panel, {"leak": guarded}, fdr_alpha=0.10)
    leak_row = screened.filter(pl.col("covariate") == "leak").row(0, named=True)
    assert leak_row["n"] == half
    assert bool(leak_row["rejected"]) is False

    unguarded = screen_signals(panel, {"leak": pl.Series("leak", leak)}, fdr_alpha=0.10)
    raw = unguarded.filter(pl.col("covariate") == "leak").row(0, named=True)
    assert bool(raw["rejected"]) is True
    assert abs(float(raw["effect_size"])) > abs(float(leak_row["effect_size"] or 0.0))


def test_residuals_from_ledger_points() -> None:
    ledger = pl.DataFrame(
        {
            "model_id": ["m", "m"],
            "run_kind": ["official", "official"],
            "as_of_season": ["2024-25", "2024-25"],
            "as_of_gw": [1, 1],
            "target_season": ["2024-25", "2024-25"],
            "target_gw": [2, 2],
            "horizon": [1, 1],
            "player_key": [1, 2],
            "position": ["MID", "FWD"],
            "fixture_id": [10, 11],
            "xpts": [4.0, 6.0],
            "xpts_sd": [1.0, 1.0],
            "p_played": [0.9, 0.8],
            "p_60": [0.7, 0.6],
            "p_goal": [0.2, 0.3],
            "p_assist": [0.1, 0.2],
            "p_cs": [0.3, 0.2],
            "p_haul": [0.1, 0.15],
            "actual_pts": [6, None],
            "actual_minutes": [90, None],
            "actual_goals": [1, None],
            "actual_assists": [0, None],
            "actual_cs": [0, None],
        },
        schema=LEDGER_COLUMNS,
    )
    panel = residuals_from_ledger(ledger)
    assert panel.height == 1
    assert panel["residual"].to_list() == [pytest.approx(2.0)]
    assert panel["mu_hat"].to_list() == [4.0]
    with pytest.raises(ValueError, match="points"):
        residuals_from_ledger(ledger, component="minutes")


def test_unused_staged_tables_and_derived_registry() -> None:
    unused = unused_staged_tables(
        {"players", "fixtures"},
        {"players", "fixtures", "match_odds"},
    )
    assert unused == {"match_odds"}
    assert "rest_days" in DERIVED_COVARIATES
    assert "minutes_last_3" in DERIVED_COVARIATES


def test_screen_signals_empty_covariates_and_schema() -> None:
    panel = _panel(residual=np.zeros(10), mu_hat=np.zeros(10), seed=0)
    empty = screen_signals(panel, {})
    SignalScreenSchema.validate(empty)
    assert list(empty.columns) == list(SIGNAL_SCREEN_COLUMNS)
    assert empty.is_empty()


def test_screen_signals_rejects_missing_panel_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        screen_signals(pl.DataFrame({"residual": [1.0]}), {"x": pl.Series("x", [1.0])})


def _panel(
    *,
    residual: np.ndarray,
    mu_hat: np.ndarray,
    seed: int,
) -> pl.DataFrame:
    n = residual.size
    rng = np.random.default_rng(seed)
    positions = np.asarray(["GKP", "DEF", "MID", "FWD"])[rng.integers(0, 4, size=n)]
    return pl.DataFrame(
        {
            "residual": residual,
            "mu_hat": mu_hat,
            "position": positions,
            "fixture_id": np.arange(n) % 30,
            "gw": (np.arange(n) % 12) + 1,
        }
    )
