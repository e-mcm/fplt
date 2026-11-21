"""Odds Poisson scoreline baseline reads match_odds via PIT (+ closing proxy)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import fplt.models.components.score  # noqa: F401 — registers components
from fplt.backtest.components import record_scorelines
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.components.score.odds_poisson import OddsPoissonScore
from fplt.models.spec import (
    ModelSpec,
    get_component,
    load_model_specs,
    registered_components,
    registered_kinds,
)
from fplt.types import GWKey

_SEASON = "2024-25"
_START = datetime(2024, 8, 1, tzinfo=UTC)


def test_odds_poisson_registers_as_component_not_predictor() -> None:
    specs = load_model_specs()
    assert get_component("odds_poisson") is OddsPoissonScore
    assert specs["odds_poisson_v1"].kind == "odds_poisson"
    assert specs["odds_poisson_v1"].params["use_closing_proxy"] is True
    assert "odds_poisson" in registered_components()
    assert "odds_poisson" not in registered_kinds()


def test_predict_uses_closing_proxy_before_kickoff(tmp_path: Path) -> None:
    store = _league(DataStore(tmp_path), with_odds=True)
    spec = load_model_specs()["odds_poisson_v1"]
    as_of = GWKey(_SEASON, 3)
    view = PITData(store).view(as_of)
    # Past GW1 kickoff is visible; GW3 kickoff-stamped odds are not yet.
    pit_odds = view.table("match_odds").collect()
    assert set(pit_odds["fixture_id"].to_list()) == {1}
    proxy = view.closing_proxy_match_odds().collect()
    assert set(proxy["fixture_id"].to_list()) == {1, 3}
    model = OddsPoissonScore(spec)
    model.fit(view)
    fixtures = (
        view.latest("fixtures", by=["season", "fixture_id"])
        .filter((pl.col("season") == _SEASON) & (pl.col("gw") == 3))
        .select("season", "gw", "fixture_id", "team_h_key", "team_a_key")
        .collect()
    )
    dist = model.predict(view, fixtures)
    assert dist.fixture_ids.tolist() == [3]
    # Short home price → higher home λ / home win mass than away.
    home_gt = np.tril(np.ones(dist.pmf.shape[1:]), k=-1)
    p_home = float((dist.pmf[0] * home_gt).sum())
    p_away = float((dist.pmf[0] * np.triu(np.ones(dist.pmf.shape[1:]), k=1)).sum())
    assert p_home > p_away
    assert float(dist.pmf[0].sum()) == pytest.approx(1.0)
    ledger = record_scorelines(model, view, fixtures)
    assert ledger.height == 1
    assert ledger["p_home"][0] > ledger["p_away"][0]


def test_missing_odds_uses_fallback_rates(tmp_path: Path) -> None:
    store = _league(DataStore(tmp_path), with_odds=False)
    spec = ModelSpec(
        name="odds_poisson",
        kind="odds_poisson",
        params={
            "use_closing_proxy": True,
            "fallback_lambda_home": 2.0,
            "fallback_lambda_away": 0.5,
        },
        refit="never",
    )
    as_of = GWKey(_SEASON, 3)
    view = PITData(store).view(as_of)
    fixtures = pl.DataFrame(
        {
            "season": [_SEASON],
            "gw": [3],
            "fixture_id": [3],
            "team_h_key": [1],
            "team_a_key": [2],
        }
    )
    dist = OddsPoissonScore(spec).predict(view, fixtures)
    goals = np.arange(dist.pmf.shape[1], dtype=np.float64)
    assert float(dist.marginal("home")[0] @ goals) == pytest.approx(2.0, abs=0.05)
    assert float(dist.marginal("away")[0] @ goals) == pytest.approx(0.5, abs=0.05)


def test_leakage_ignores_non_odds_and_other_fixture_odds(tmp_path: Path) -> None:
    store = _league(DataStore(tmp_path), with_odds=True)
    spec = load_model_specs()["odds_poisson_v1"]
    as_of = GWKey(_SEASON, 3)
    cutoff = store.deadline(as_of)
    model = OddsPoissonScore(spec)
    view = PITData(store).view(as_of)
    fixtures = (
        view.latest("fixtures", by=["season", "fixture_id"])
        .filter((pl.col("season") == _SEASON) & (pl.col("gw") == 3))
        .select("season", "gw", "fixture_id", "team_h_key", "team_a_key")
        .collect()
    )
    before = model.predict(view, fixtures).pmf.copy()
    # Bump finished GW1 odds prices — not the target fixture.
    odds = store.read("match_odds").with_columns(
        pl.when(pl.col("fixture_id") == 1)
        .then(pl.col("home_odds") + 5.0)
        .otherwise(pl.col("home_odds"))
        .alias("home_odds")
    )
    store.write("match_odds", odds)
    # Bump player_status after cutoff (classic leakage bait).
    later = cutoff + timedelta(days=1)
    status = store.read("player_status").with_columns(
        pl.lit(later).cast(pl.Datetime("us", "UTC")).alias("available_at"),
        (pl.col("ep_next") + 9).alias("ep_next"),
    )
    store.write("player_status", status)
    after = model.predict(PITData(store).view(as_of), fixtures).pmf
    np.testing.assert_allclose(before, after)


def test_disable_closing_proxy_falls_back_without_pit_odds(tmp_path: Path) -> None:
    store = _league(DataStore(tmp_path), with_odds=True)
    spec = ModelSpec(
        name="odds_poisson",
        kind="odds_poisson",
        params={
            "use_closing_proxy": False,
            "fallback_lambda_home": 1.8,
            "fallback_lambda_away": 0.9,
        },
        refit="never",
    )
    as_of = GWKey(_SEASON, 3)
    view = PITData(store).view(as_of)
    fixtures = pl.DataFrame(
        {
            "season": [_SEASON],
            "gw": [3],
            "fixture_id": [3],
            "team_h_key": [1],
            "team_a_key": [2],
        }
    )
    dist = OddsPoissonScore(spec).predict(view, fixtures)
    goals = np.arange(dist.pmf.shape[1], dtype=np.float64)
    assert float(dist.marginal("home")[0] @ goals) == pytest.approx(1.8, abs=0.05)


def _league(store: DataStore, *, with_odds: bool) -> DataStore:
    kickoffs = [_START + timedelta(days=7 * gw) for gw in range(1, 5)]
    deadlines = [kick - timedelta(hours=21) for kick in kickoffs]
    available = [kick + timedelta(hours=3) for kick in kickoffs]
    available[-1] = deadlines[-1] - timedelta(days=1)
    store.write(
        "teams",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON, _SEASON],
                    "team_key": [1, 2],
                    "team_id": [1, 2],
                    "name": ["Home", "Away"],
                    "short_name": ["HOM", "AWY"],
                    "available_at": [_START, _START],
                    "source": ["fpl_api", "fpl_api"],
                }
            )
        ),
    )
    store.write(
        "players",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * 2,
                    "player_key": [10, 20],
                    "element_id": [10, 20],
                    "team_key": [1, 2],
                    "position": ["MID", "MID"],
                    "web_name": ["A", "C"],
                    "available_at": [_START, _START],
                    "source": ["fpl_api", "fpl_api"],
                }
            )
        ),
    )
    store.write(
        "gameweeks",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * 4,
                    "gw": [1, 2, 3, 4],
                    "deadline_time": deadlines,
                    "finished": [True, True, True, False],
                    "data_checked": [True, True, True, False],
                    "available_at": [_START] * 4,
                    "source": ["fpl_api"] * 4,
                }
            )
        ),
    )
    schedule_at = [deadline - timedelta(days=1) for deadline in deadlines]
    store.write(
        "fixtures",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * 8,
                    "fixture_id": [1, 1, 2, 2, 3, 3, 4, 4],
                    "gw": [1, 1, 2, 2, 3, 3, 4, 4],
                    "team_h_key": [1, 1, 2, 2, 1, 1, 1, 1],
                    "team_a_key": [2, 2, 1, 1, 2, 2, 2, 2],
                    "kickoff_time": [
                        kickoffs[0],
                        kickoffs[0],
                        kickoffs[1],
                        kickoffs[1],
                        kickoffs[2],
                        kickoffs[2],
                        kickoffs[3],
                        kickoffs[3],
                    ],
                    "team_h_score": [None, 2, None, 0, None, 1, None, None],
                    "team_a_score": [None, 1, None, 1, None, 0, None, None],
                    "finished": [False, True, False, True, False, True, False, False],
                    "available_at": [
                        schedule_at[0],
                        available[0],
                        schedule_at[1],
                        available[1],
                        schedule_at[2],
                        available[2],
                        schedule_at[3],
                        schedule_at[3],
                    ],
                    "source": ["fpl_api"] * 8,
                }
            )
        ),
    )
    store.write(
        "player_status",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * 2,
                    "player_key": [10, 20],
                    "status": ["a", "a"],
                    "news": ["", ""],
                    "price": [70, 65],
                    "ep_next": [4.0, 3.0],
                    "chance_of_playing": [100.0, 100.0],
                    "available_at": [_START, _START],
                    "source": ["fpl_api", "fpl_api"],
                }
            )
        ),
    )
    if with_odds:
        store.write(
            "match_odds",
            _utc(
                pl.DataFrame(
                    {
                        "season": [_SEASON, _SEASON],
                        "fixture_id": [1, 3],
                        "team_h_key": [1, 1],
                        "team_a_key": [2, 2],
                        "kickoff_time": [kickoffs[0], kickoffs[2]],
                        "home_odds": [1.55, 1.45],
                        "draw_odds": [4.20, 4.40],
                        "away_odds": [6.00, 7.00],
                        "over_25_odds": [1.70, 1.65],
                        "under_25_odds": [2.20, 2.30],
                        "odds_kind": ["closing_proxy", "closing_proxy"],
                        "available_at": [kickoffs[0], kickoffs[2]],
                        "source": ["football_data", "football_data"],
                    }
                )
            ),
        )
    return store


def _utc(frame: pl.DataFrame) -> pl.DataFrame:
    columns = [
        pl.col(name).cast(pl.Datetime("us", "UTC"))
        for name, dtype in frame.schema.items()
        if isinstance(dtype, pl.Datetime)
    ]
    return frame.with_columns(columns) if columns else frame
