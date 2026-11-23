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
from fplt.models.components.score.odds_lambda import (
    blend_lambda,
    cs_probs_from_lambda,
    de_vig,
    odds_to_lambda,
    outcome_probs_from_joint,
)
from fplt.models.spec import ModelSpec, load_model_specs
from fplt.types import GWKey

_SEASON = "2024-25"
_START = datetime(2024, 8, 1, tzinfo=UTC)


def test_odds_specs_are_registered_without_changing_vanilla() -> None:
    specs = load_model_specs()
    assert "dixon_coles_odds_v1" in specs
    assert "dc_layered_odds_v1" in specs
    vanilla_score = specs["dc_layered_v4"].components["score"].params
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
        rtol=1e-8,
        atol=1e-10,
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
        rtol=1e-8,
        atol=1e-10,
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


def test_market_1x2_strength_zero_matches_vanilla(tmp_path: Path) -> None:
    store = _season_with_odds(DataStore(tmp_path), odds_before_deadline=True)
    view = PITData(store).view(GWKey(_SEASON, 4))
    fixtures = pl.DataFrame(
        {
            "season": [_SEASON],
            "fixture_id": [4],
            "team_h_key": [1],
            "team_a_key": [2],
            "kickoff_time": [
                store.read("fixtures").filter(pl.col("fixture_id") == 4)[
                    "kickoff_time"
                ][0]
            ],
        }
    )
    base = {"xi": 0.0, "home_advantage": 1.2, "rho": -0.05}
    vanilla = MatchScoreModel(ModelSpec(name="v", kind="dixon_coles", params=base))
    zero = MatchScoreModel(
        ModelSpec(
            name="z",
            kind="dixon_coles",
            params={**base, "market_1x2_strength": 0.0, "use_closing_proxy": True},
        )
    )
    vanilla.fit(view)
    zero.fit(view)
    np.testing.assert_allclose(
        vanilla.predict(view, fixtures).pmf,
        zero.predict(view, fixtures).pmf,
        rtol=1e-10,
        atol=1e-12,
    )


def test_market_1x2_strength_moves_margins_toward_market(tmp_path: Path) -> None:
    store = _season_with_odds(DataStore(tmp_path), odds_before_deadline=True)
    view = PITData(store).view(GWKey(_SEASON, 4))
    kick = store.read("fixtures").filter(pl.col("fixture_id") == 4)["kickoff_time"][0]
    fixtures = pl.DataFrame(
        {
            "season": [_SEASON],
            "fixture_id": [4],
            "team_h_key": [1],
            "team_a_key": [2],
            "kickoff_time": [kick],
        }
    )
    base = {"xi": 0.0, "home_advantage": 1.0, "rho": 0.0}
    vanilla = MatchScoreModel(ModelSpec(name="v", kind="dixon_coles", params=base))
    blended = MatchScoreModel(
        ModelSpec(
            name="b",
            kind="dixon_coles",
            params={
                **base,
                "market_1x2_strength": 1.0,
                "use_closing_proxy": True,
            },
        )
    )
    vanilla.fit(view)
    blended.fit(view)
    from fplt.backtest.components import _outcome_probs

    v_h, _, v_a = _outcome_probs(vanilla.predict(view, fixtures))
    b_h, _, b_a = _outcome_probs(blended.predict(view, fixtures))
    market = de_vig(np.array([1.25, 6.0, 12.0]))
    # Short home price → blend should raise p_home vs vanilla toward market.
    assert float(b_h[0]) > float(v_h[0])
    assert abs(float(b_h[0]) - float(market[0])) < abs(float(v_h[0]) - float(market[0]))
    assert float(b_a[0]) < float(v_a[0])


def test_dixon_coles_1x2_and_layered_specs_registered() -> None:
    specs = load_model_specs()
    assert specs["dixon_coles_1x2_v1"].params["market_1x2_strength"] == 1.0
    assert specs["dixon_coles_1x2_v1"].params["use_closing_proxy"] is True
    assert "market_prior_strength" not in specs["dixon_coles_1x2_v1"].params
    layered = specs["dc_layered_1x2_v1"]
    assert layered.components["score"].params["market_1x2_strength"] == 1.0
    assert layered.components["score"].params["use_closing_proxy"] is True
    assert (
        "market_prior_strength" not in specs["dc_layered_v4"].components["score"].params
    )


def test_dixon_coles_1x2_prematch_and_layered_specs_registered() -> None:
    """Prematch 1X2 fork pairs strength with use_closing_proxy=false."""
    specs = load_model_specs()
    score = specs["dixon_coles_1x2_prematch_v1"]
    assert score.params["market_1x2_strength"] == 1.0
    assert score.params["use_closing_proxy"] is False
    assert "market_prior_strength" not in score.params
    assert "market_cs_strength" not in score.params
    layered = specs["dc_layered_1x2_prematch_v1"]
    nested = layered.components["score"]
    assert nested.name == "dixon_coles_1x2_prematch_v1"
    assert nested.params["market_1x2_strength"] == 1.0
    # Explicit false overrides MatchScoreModel's default (true when 1x2 on).
    assert nested.params["use_closing_proxy"] is False
    assert (
        "market_1x2_strength" not in specs["dc_layered_v4"].components["score"].params
    )


def test_market_1x2_prematch_defaults_closing_proxy_off_when_set() -> None:
    """Constructor keeps use_closing_proxy=false when the YAML sets it."""
    model = MatchScoreModel(
        ModelSpec(
            name="prematch",
            kind="dixon_coles",
            params={
                "xi": 0.0,
                "market_1x2_strength": 1.0,
                "use_closing_proxy": False,
            },
        )
    )
    assert model._market_1x2 == 1.0
    assert model._use_closing_proxy is False


def test_dixon_coles_cs_and_layered_specs_registered() -> None:
    specs = load_model_specs()
    cs = specs["dixon_coles_cs_v1"]
    assert cs.params["market_cs_strength"] == 0.5
    assert cs.params["use_closing_proxy"] is False
    assert "market_prior_strength" not in cs.params
    assert "market_1x2_strength" not in cs.params
    layered = specs["dc_layered_cs_v1"]
    score = layered.components["score"]
    assert score.name == "dixon_coles_cs_v1"
    assert score.params["market_cs_strength"] == 0.5
    # Layered fork omits use_closing_proxy so one-lever diffs stay on
    # market_cs_strength alone; MatchScoreModel still defaults it off.
    assert "use_closing_proxy" not in score.params
    # Production v4 composes CS after FDB-267; frozen v3 stays off.
    v4_score = specs["dc_layered_v4"].components["score"]
    assert v4_score.params["market_cs_strength"] == 0.5
    v3_score = specs["dc_layered_v3"].components["score"]
    assert "market_cs_strength" not in v3_score.params
    nocs = specs["dc_layered_nocs_v1"].components["score"]
    assert nocs.name == "dixon_coles_v2"
    assert nocs.params["market_cs_strength"] == 0.0


def test_market_cs_strength_zero_matches_vanilla(tmp_path: Path) -> None:
    store = _season_with_odds(DataStore(tmp_path), odds_before_deadline=True)
    view = PITData(store).view(GWKey(_SEASON, 4))
    fixtures = pl.DataFrame(
        {
            "season": [_SEASON],
            "fixture_id": [4],
            "team_h_key": [1],
            "team_a_key": [2],
            "kickoff_time": [
                store.read("fixtures").filter(pl.col("fixture_id") == 4)[
                    "kickoff_time"
                ][0]
            ],
        }
    )
    base = {"xi": 0.0, "home_advantage": 1.2, "rho": -0.05}
    vanilla = MatchScoreModel(ModelSpec(name="v", kind="dixon_coles", params=base))
    zero = MatchScoreModel(
        ModelSpec(
            name="z",
            kind="dixon_coles",
            params={**base, "market_cs_strength": 0.0, "use_closing_proxy": False},
        )
    )
    vanilla.fit(view)
    zero.fit(view)
    # Fit noise can differ by ~1e-8 across independent L-BFGS runs; match the
    # market_prior_strength_zero tolerance rather than bit-identity.
    np.testing.assert_allclose(
        vanilla.predict(view, fixtures).pmf,
        zero.predict(view, fixtures).pmf,
        rtol=1e-8,
        atol=1e-10,
    )


def test_market_cs_strength_moves_cs_keeps_1x2(tmp_path: Path) -> None:
    store = _season_with_odds(DataStore(tmp_path), odds_before_deadline=True)
    view = PITData(store).view(GWKey(_SEASON, 4))
    kick = store.read("fixtures").filter(pl.col("fixture_id") == 4)["kickoff_time"][0]
    fixtures = pl.DataFrame(
        {
            "season": [_SEASON],
            "fixture_id": [4],
            "team_h_key": [1],
            "team_a_key": [2],
            "kickoff_time": [kick],
        }
    )
    base = {"xi": 0.0, "home_advantage": 1.0, "rho": 0.0}
    vanilla = MatchScoreModel(ModelSpec(name="v", kind="dixon_coles", params=base))
    blended = MatchScoreModel(
        ModelSpec(
            name="b",
            kind="dixon_coles",
            params={
                **base,
                "market_cs_strength": 0.5,
                "use_closing_proxy": False,
            },
        )
    )
    vanilla.fit(view)
    blended.fit(view)
    v_dist = vanilla.predict(view, fixtures)
    b_dist = blended.predict(view, fixtures)
    v_h, v_d, v_a = outcome_probs_from_joint(v_dist.pmf)
    b_h, b_d, b_a = outcome_probs_from_joint(b_dist.pmf)
    # DC 1X2 preserved exactly after CS blend + restore.
    assert float(b_h[0]) == pytest.approx(float(v_h[0]), abs=1e-9)
    assert float(b_d[0]) == pytest.approx(float(v_d[0]), abs=1e-9)
    assert float(b_a[0]) == pytest.approx(float(v_a[0]), abs=1e-9)
    mkt_h, mkt_a = odds_to_lambda(1.25, 6.0, 12.0, 1.55, 2.40)
    mkt_cs_home, mkt_cs_away = cs_probs_from_lambda(mkt_h, mkt_a)
    # Short home price → low away λ → high home CS; blend should move toward it.
    v_cs_h = float(v_dist.p_clean_sheet("home")[0])
    b_cs_h = float(b_dist.p_clean_sheet("home")[0])
    assert abs(b_cs_h - mkt_cs_home) < abs(v_cs_h - mkt_cs_home)
    v_cs_a = float(v_dist.p_clean_sheet("away")[0])
    b_cs_a = float(b_dist.p_clean_sheet("away")[0])
    assert abs(b_cs_a - mkt_cs_away) <= abs(v_cs_a - mkt_cs_away) + 1e-9


def test_negative_market_cs_strength_raises() -> None:
    with pytest.raises(ValueError, match="market_cs_strength"):
        MatchScoreModel(
            ModelSpec(
                name="bad",
                kind="dixon_coles",
                params={"market_cs_strength": -0.1},
            )
        )


def test_odds_after_deadline_leave_cs_blend_as_vanilla(tmp_path: Path) -> None:
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
    base = {"xi": 0.0, "home_advantage": 1.2, "rho": -0.05}
    vanilla = MatchScoreModel(ModelSpec(name="v", kind="dixon_coles", params=base))
    cs = MatchScoreModel(
        ModelSpec(
            name="cs",
            kind="dixon_coles",
            params={**base, "market_cs_strength": 0.5, "use_closing_proxy": False},
        )
    )
    vanilla.fit(view)
    cs.fit(view)
    np.testing.assert_allclose(
        vanilla.predict(view, fixtures).pmf,
        cs.predict(view, fixtures).pmf,
        rtol=1e-10,
        atol=1e-12,
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
