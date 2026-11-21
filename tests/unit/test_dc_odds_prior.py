"""Dixon–Coles market prior: strength 0 / missing odds ≡ vanilla."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import fplt.models.components.score  # noqa: F401 — registers the component
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.components.score.dixon_coles import MatchScoreModel, _align_odds
from fplt.models.components.score.odds_lambda import blend_lambda, odds_to_lambda
from fplt.models.spec import ModelSpec, load_model_specs
from fplt.types import GWKey

_SEASON = "2024-25"
_START = datetime(2024, 8, 1, tzinfo=UTC)


def test_odds_specs_are_registered_without_changing_vanilla() -> None:
    specs = load_model_specs()
    assert "dixon_coles_odds_v1" in specs
    assert "dc_layered_odds_v1" in specs
    vanilla_score = specs["dc_layered_v2"].components["score"].params
    assert "market_prior_strength" not in specs["dixon_coles_v2"].params
    assert "market_prior_strength" not in vanilla_score
    assert specs["dixon_coles_odds_v1"].params["market_prior_strength"] == 1.0
    odds_score = specs["dc_layered_odds_v1"].components["score"]
    assert odds_score.params["market_prior_strength"] == 1.0
    assert odds_score.name == "dixon_coles_odds_v1"


def test_market_prior_strength_zero_matches_vanilla(tmp_path: Path) -> None:
    store = _season_with_odds(DataStore(tmp_path), odds_before_deadline=True)
    view = PITData(store).view(GWKey(_SEASON, 4))
    fixtures = pl.DataFrame(
        {
            "season": [_SEASON],
            "fixture_id": [4],
            "team_h_key": [1],
            "team_a_key": [2],
        }
    )
    vanilla = MatchScoreModel(
        ModelSpec(
            name="dc_vanilla",
            kind="dixon_coles",
            params={"xi": 0.0, "home_advantage": 1.2, "rho": -0.05},
        )
    )
    zero_prior = MatchScoreModel(
        ModelSpec(
            name="dc_zero_prior",
            kind="dixon_coles",
            params={
                "xi": 0.0,
                "home_advantage": 1.2,
                "rho": -0.05,
                "market_prior_strength": 0.0,
            },
        )
    )
    vanilla.fit(view)
    zero_prior.fit(view)
    np.testing.assert_allclose(
        vanilla.predict(view, fixtures).pmf,
        zero_prior.predict(view, fixtures).pmf,
        rtol=1e-10,
        atol=1e-12,
    )


def test_missing_odds_matches_vanilla(tmp_path: Path) -> None:
    store = _season(DataStore(tmp_path))
    view = PITData(store).view(GWKey(_SEASON, 4))
    fixtures = pl.DataFrame(
        {
            "season": [_SEASON],
            "fixture_id": [4],
            "team_h_key": [1],
            "team_a_key": [2],
        }
    )
    base_params = {"xi": 0.0, "home_advantage": 1.2, "rho": -0.05}
    vanilla = MatchScoreModel(
        ModelSpec(name="dc_vanilla", kind="dixon_coles", params=base_params)
    )
    with_prior = MatchScoreModel(
        ModelSpec(
            name="dc_prior",
            kind="dixon_coles",
            params={**base_params, "market_prior_strength": 2.0},
        )
    )
    vanilla.fit(view)
    with_prior.fit(view)
    np.testing.assert_allclose(
        vanilla.predict(view, fixtures).pmf,
        with_prior.predict(view, fixtures).pmf,
        rtol=1e-10,
        atol=1e-12,
    )


def test_strength_moves_lambda_toward_market(tmp_path: Path) -> None:
    store = _season_with_odds(DataStore(tmp_path), odds_before_deadline=True)
    view = PITData(store).view(GWKey(_SEASON, 4))
    fixtures = pl.DataFrame(
        {
            "season": [_SEASON],
            "fixture_id": [4],
            "team_h_key": [1],
            "team_a_key": [2],
        }
    )
    base_params = {"xi": 0.0, "home_advantage": 1.0, "rho": 0.0}
    vanilla = MatchScoreModel(
        ModelSpec(name="dc_vanilla", kind="dixon_coles", params=base_params)
    )
    prior = MatchScoreModel(
        ModelSpec(
            name="dc_prior",
            kind="dixon_coles",
            params={**base_params, "market_prior_strength": 1.0},
        )
    )
    vanilla.fit(view)
    prior.fit(view)
    # Same fitted ratings → same DC λ before blend.
    dc_h, dc_a = vanilla._rates([1], [2])
    mkt_h, mkt_a = odds_to_lambda(1.25, 6.0, 12.0, 1.55, 2.40)
    expected_h = blend_lambda(float(dc_h[0]), mkt_h, 1.0)
    expected_a = blend_lambda(float(dc_a[0]), mkt_a, 1.0)
    vanilla_dist = vanilla.predict(view, fixtures)
    prior_dist = prior.predict(view, fixtures)
    assert not np.allclose(vanilla_dist.pmf, prior_dist.pmf)
    # Expected goals should move toward the short-priced home market.
    goals = np.arange(vanilla_dist.pmf.shape[1], dtype=np.float64)
    vanilla_eh = float(vanilla_dist.marginal("home")[0] @ goals)
    prior_eh = float(prior_dist.marginal("home")[0] @ goals)
    assert abs(prior_eh - expected_h) < abs(vanilla_eh - expected_h)
    assert prior_eh > vanilla_eh
    assert abs(prior_eh - expected_h) < 0.15
    assert abs(float(prior_dist.marginal("away")[0] @ goals) - expected_a) < 0.15


def test_odds_after_deadline_are_invisible(tmp_path: Path) -> None:
    store = _season_with_odds(DataStore(tmp_path), odds_before_deadline=False)
    view = PITData(store).view(GWKey(_SEASON, 4))
    fixtures = pl.DataFrame(
        {
            "season": [_SEASON],
            "fixture_id": [4],
            "team_h_key": [1],
            "team_a_key": [2],
        }
    )
    base_params = {"xi": 0.0, "home_advantage": 1.2, "rho": -0.05}
    vanilla = MatchScoreModel(
        ModelSpec(name="dc_vanilla", kind="dixon_coles", params=base_params)
    )
    prior = MatchScoreModel(
        ModelSpec(
            name="dc_prior",
            kind="dixon_coles",
            params={**base_params, "market_prior_strength": 5.0},
        )
    )
    vanilla.fit(view)
    prior.fit(view)
    np.testing.assert_allclose(
        vanilla.predict(view, fixtures).pmf,
        prior.predict(view, fixtures).pmf,
        rtol=1e-10,
        atol=1e-12,
    )


def test_negative_market_prior_strength_raises() -> None:
    with pytest.raises(ValueError, match="market_prior_strength"):
        MatchScoreModel(
            ModelSpec(
                name="bad",
                kind="dixon_coles",
                params={"market_prior_strength": -0.1},
            )
        )


def test_align_odds_scopes_fixture_id_by_season() -> None:
    """FPL fixture_ids restart each season — do not join cross-season odds."""
    fixtures = pl.DataFrame(
        {
            "season": ["2024-25"],
            "fixture_id": [1],
            "team_h_key": [10],
            "team_a_key": [20],
        }
    )
    odds = pl.DataFrame(
        {
            "season": ["2021-22", "2024-25"],
            "fixture_id": [1, 1],
            "team_h_key": [99, 10],
            "team_a_key": [88, 20],
            "kickoff_time": [
                datetime(2021, 8, 14, tzinfo=UTC),
                datetime(2024, 8, 16, tzinfo=UTC),
            ],
            "home_odds": [9.0, 1.50],
            "draw_odds": [5.0, 4.0],
            "away_odds": [1.20, 6.0],
            "over_25_odds": [None, None],
            "under_25_odds": [None, None],
            "available_at": [
                datetime(2021, 8, 14, tzinfo=UTC),
                datetime(2024, 8, 16, tzinfo=UTC),
            ],
        }
    )
    aligned = _align_odds(fixtures, odds)
    assert aligned["home_odds"][0] == pytest.approx(1.50)
    # Prior-season-only odds with the same fixture_id must not match.
    prior_only = odds.filter(pl.col("season") == "2021-22")
    empty = _align_odds(fixtures, prior_only)
    assert empty["home_odds"][0] is None


def test_dixon_coles_odds_yaml_predicts(tmp_path: Path) -> None:
    store = _season_with_odds(DataStore(tmp_path), odds_before_deadline=True)
    spec = load_model_specs()["dixon_coles_odds_v1"]
    model = MatchScoreModel(spec)
    view = PITData(store).view(GWKey(_SEASON, 4))
    model.fit(view)
    dist = model.predict(
        view,
        pl.DataFrame(
            {
                "season": [_SEASON],
                "fixture_id": [4],
                "team_h_key": [1],
                "team_a_key": [2],
            }
        ),
    )
    assert dist.pmf.shape == (1, 33, 33)
    assert dist.pmf.sum() == pytest.approx(1.0)


def _season(store: DataStore) -> DataStore:
    kickoffs = [_START + timedelta(days=7 * gw) for gw in range(1, 5)]
    deadlines = [kick - timedelta(hours=21) for kick in kickoffs]
    results = [(1, 0), (0, 1), (1, 1), (None, None)]
    homes = [1, 2, 1, 1]
    aways = [2, 1, 2, 2]
    finished = [True, True, True, False]
    available = [kick + timedelta(hours=3) for kick in kickoffs]
    available[-1] = deadlines[-1] - timedelta(days=1)
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
    store.write(
        "fixtures",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * 4,
                    "fixture_id": [1, 2, 3, 4],
                    "gw": [1, 2, 3, 4],
                    "team_h_key": homes,
                    "team_a_key": aways,
                    "kickoff_time": kickoffs,
                    "team_h_score": [score[0] for score in results],
                    "team_a_score": [score[1] for score in results],
                    "finished": finished,
                    "available_at": available,
                    "source": ["fpl_api"] * 4,
                }
            )
        ),
    )
    return store


def _season_with_odds(store: DataStore, *, odds_before_deadline: bool) -> DataStore:
    store = _season(store)
    fixtures = store.read("fixtures")
    kick = fixtures.filter(pl.col("fixture_id") == 4)["kickoff_time"][0]
    deadline = store.deadline(GWKey(_SEASON, 4))
    available_at = deadline - timedelta(hours=1) if odds_before_deadline else kick
    store.write(
        "match_odds",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON],
                    "fixture_id": [4],
                    "team_h_key": [1],
                    "team_a_key": [2],
                    "kickoff_time": [kick],
                    "home_odds": [1.25],
                    "draw_odds": [6.0],
                    "away_odds": [12.0],
                    "over_25_odds": [1.55],
                    "under_25_odds": [2.40],
                    "odds_kind": ["closing_proxy"],
                    "available_at": [available_at],
                    "source": ["football_data"],
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
