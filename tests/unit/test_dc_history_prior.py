"""Dixon–Coles history market prior: Gaussian (a0, d0) from past closes."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import fplt.models.components.score  # noqa: F401 — registers the component
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.components.score.dixon_coles import (
    MatchScoreModel,
    _finished,
    _fit_history_market_ratings,
)
from fplt.models.spec import ModelSpec, load_model_specs
from fplt.types import GWKey

_SEASON = "2024-25"
_START = datetime(2024, 8, 1, tzinfo=UTC)
# Independent L-BFGS fits of the same data can differ by ~1e-8 on the PMF
# (seen on CI for missing-past-odds / strength-zero cases; max rel ~4e-8).
# Real history priors move mass by O(0.1), so this band stays well clear.
_VANILLA_PMF_RTOL = 1e-7
_VANILLA_PMF_ATOL = 1e-8


def _assert_pmf_matches_vanilla(left: np.ndarray, right: np.ndarray) -> None:
    """Assert two PMFs match within L-BFGS fit noise (vanilla-equivalence)."""
    np.testing.assert_allclose(
        left, right, rtol=_VANILLA_PMF_RTOL, atol=_VANILLA_PMF_ATOL
    )


def test_history_specs_are_registered_without_changing_vanilla() -> None:
    specs = load_model_specs()
    assert "dixon_coles_history_v1" in specs
    assert "dc_layered_history_v1" in specs
    history = specs["dixon_coles_history_v1"].params
    assert history["market_prior_source"] == "history"
    assert history["market_prior_strength"] == 1.0
    assert "market_prior_source" not in specs["dixon_coles_v2"].params
    assert "market_prior_source" not in specs["dixon_coles_odds_v1"].params
    layered = specs["dc_layered_history_v1"].components["score"]
    assert layered.name == "dixon_coles_history_v1"
    assert layered.params["market_prior_source"] == "history"
    assert (
        "market_prior_source" not in specs["dc_layered_v4"].components["score"].params
    )


def test_absent_source_keeps_predict_time_lambda_blend(tmp_path: Path) -> None:
    store = _season_with_target_odds(DataStore(tmp_path), odds_before_deadline=True)
    view = PITData(store).view(GWKey(_SEASON, 4))
    fixtures = pl.DataFrame(
        {
            "season": [_SEASON],
            "fixture_id": [4],
            "team_h_key": [1],
            "team_a_key": [2],
        }
    )
    base = {"xi": 0.0, "home_advantage": 1.0, "rho": 0.0}
    blend = MatchScoreModel(
        ModelSpec(
            name="blend",
            kind="dixon_coles",
            params={**base, "market_prior_strength": 1.0},
        )
    )
    history = MatchScoreModel(
        ModelSpec(
            name="history",
            kind="dixon_coles",
            params={
                **base,
                "market_prior_strength": 1.0,
                "market_prior_source": "history",
            },
        )
    )
    blend.fit(view)
    history.fit(view)
    # Target-fixture odds move the absent-source blend; history mode ignores them.
    assert not np.allclose(
        blend.predict(view, fixtures).pmf, history.predict(view, fixtures).pmf
    )


def test_history_strength_zero_matches_vanilla(tmp_path: Path) -> None:
    store = _season_with_past_closes(DataStore(tmp_path))
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
    vanilla = MatchScoreModel(
        ModelSpec(name="vanilla", kind="dixon_coles", params=base)
    )
    zero = MatchScoreModel(
        ModelSpec(
            name="zero",
            kind="dixon_coles",
            params={
                **base,
                "market_prior_strength": 0.0,
                "market_prior_source": "history",
            },
        )
    )
    vanilla.fit(view)
    zero.fit(view)
    _assert_pmf_matches_vanilla(
        vanilla.predict(view, fixtures).pmf,
        zero.predict(view, fixtures).pmf,
    )


def test_history_prior_moves_ratings_toward_market(tmp_path: Path) -> None:
    store = _season_with_past_closes(DataStore(tmp_path))
    view = PITData(store).view(GWKey(_SEASON, 4))
    fixtures = pl.DataFrame(
        {
            "season": [_SEASON],
            "fixture_id": [4],
            "team_h_key": [1],
            "team_a_key": [2],
        }
    )
    base = {"xi": 0.0, "home_advantage": 1.0, "rho": 0.0}
    vanilla = MatchScoreModel(
        ModelSpec(name="vanilla", kind="dixon_coles", params=base)
    )
    history = MatchScoreModel(
        ModelSpec(
            name="history",
            kind="dixon_coles",
            params={
                **base,
                "market_prior_strength": 50.0,
                "market_prior_source": "history",
            },
        )
    )
    vanilla.fit(view)
    history.fit(view)
    assert not np.allclose(
        vanilla.predict(view, fixtures).pmf, history.predict(view, fixtures).pmf
    )
    played = _finished(view)
    teams = sorted(
        set(played["team_h_key"].to_list()) | set(played["team_a_key"].to_list())
    )
    lookup = pl.DataFrame({"team_key": teams, "idx": list(range(len(teams)))})
    indexed = played.join(
        lookup, left_on="team_h_key", right_on="team_key", how="left"
    ).join(lookup, left_on="team_a_key", right_on="team_key", how="left")
    weights = np.ones(indexed.height, dtype=np.float64)
    a0, d0 = _fit_history_market_ratings(
        view,
        indexed,
        indexed["idx"].to_numpy(),
        indexed["idx_right"].to_numpy(),
        weights,
        len(teams),
        1.0,
    )
    assert a0 is not None
    assert d0 is not None
    vanilla_a = np.array([vanilla._attack[team] for team in teams])
    history_a = np.array([history._attack[team] for team in teams])
    # Strong Gaussian pull should land nearer the market ratings than vanilla.
    assert float(np.sum((history_a - a0) ** 2)) < float(np.sum((vanilla_a - a0) ** 2))


def test_history_missing_past_odds_matches_vanilla(tmp_path: Path) -> None:
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
    base = {"xi": 0.0, "home_advantage": 1.2, "rho": -0.05}
    vanilla = MatchScoreModel(
        ModelSpec(name="vanilla", kind="dixon_coles", params=base)
    )
    history = MatchScoreModel(
        ModelSpec(
            name="history",
            kind="dixon_coles",
            params={
                **base,
                "market_prior_strength": 5.0,
                "market_prior_source": "history",
            },
        )
    )
    vanilla.fit(view)
    history.fit(view)
    _assert_pmf_matches_vanilla(
        vanilla.predict(view, fixtures).pmf,
        history.predict(view, fixtures).pmf,
    )


def test_history_leakage_ignores_odds_after_deadline(tmp_path: Path) -> None:
    """Post-deadline closes must not enter the history-prior fit."""
    store = _season(DataStore(tmp_path))
    as_of = GWKey(_SEASON, 4)
    cutoff = store.deadline(as_of)
    fixtures = pl.DataFrame(
        {
            "season": [_SEASON],
            "fixture_id": [4],
            "team_h_key": [1],
            "team_a_key": [2],
        }
    )
    params = {
        "xi": 0.0,
        "home_advantage": 1.0,
        "rho": 0.0,
        "market_prior_strength": 20.0,
        "market_prior_source": "history",
    }
    vanilla = MatchScoreModel(
        ModelSpec(
            name="vanilla",
            kind="dixon_coles",
            params={"xi": 0.0, "home_advantage": 1.0, "rho": 0.0},
        )
    )
    vanilla.fit(PITData(store).view(as_of))
    before = vanilla.predict(PITData(store).view(as_of), fixtures).pmf.copy()

    # Stamp finished-match closes and the target fixture after the cutoff.
    # Store keys collapse one row per match, so late available_at is the only
    # copy — PIT must hide every row.
    late = cutoff + timedelta(hours=1)
    finished = store.read("fixtures").filter(pl.col("finished"))
    rows: dict[str, list[object]] = {
        "season": [],
        "fixture_id": [],
        "team_h_key": [],
        "team_a_key": [],
        "kickoff_time": [],
        "home_odds": [],
        "draw_odds": [],
        "away_odds": [],
        "over_25_odds": [],
        "under_25_odds": [],
        "odds_kind": [],
        "available_at": [],
        "source": [],
    }
    for row in finished.iter_rows(named=True):
        rows["season"].append(_SEASON)
        rows["fixture_id"].append(int(row["fixture_id"]))
        rows["team_h_key"].append(int(row["team_h_key"]))
        rows["team_a_key"].append(int(row["team_a_key"]))
        rows["kickoff_time"].append(row["kickoff_time"])
        rows["home_odds"].append(1.20)
        rows["draw_odds"].append(6.0)
        rows["away_odds"].append(15.0)
        rows["over_25_odds"].append(1.55)
        rows["under_25_odds"].append(2.40)
        rows["odds_kind"].append("closing_proxy")
        rows["available_at"].append(late)
        rows["source"].append("football_data")
    kick4 = store.read("fixtures").filter(pl.col("fixture_id") == 4)["kickoff_time"][0]
    rows["season"].append(_SEASON)
    rows["fixture_id"].append(4)
    rows["team_h_key"].append(1)
    rows["team_a_key"].append(2)
    rows["kickoff_time"].append(kick4)
    rows["home_odds"].append(1.05)
    rows["draw_odds"].append(12.0)
    rows["away_odds"].append(25.0)
    rows["over_25_odds"].append(1.40)
    rows["under_25_odds"].append(2.80)
    rows["odds_kind"].append("closing_proxy")
    rows["available_at"].append(late)
    rows["source"].append("football_data")
    store.write("match_odds", _utc(pl.DataFrame(rows)))

    history = MatchScoreModel(
        ModelSpec(name="history", kind="dixon_coles", params=params)
    )
    history.fit(PITData(store).view(as_of))
    after = history.predict(PITData(store).view(as_of), fixtures).pmf
    np.testing.assert_allclose(before, after, rtol=1e-8, atol=1e-8)
    assert history._attack == vanilla._attack


def test_history_ignores_target_fixture_odds_at_predict(tmp_path: Path) -> None:
    """History mode must not blend λ from the fixtures being scored."""
    store = _season_with_past_closes(DataStore(tmp_path))
    as_of = GWKey(_SEASON, 4)
    cutoff = store.deadline(as_of)
    fixtures = pl.DataFrame(
        {
            "season": [_SEASON],
            "fixture_id": [4],
            "team_h_key": [1],
            "team_a_key": [2],
        }
    )
    params = {
        "xi": 0.0,
        "home_advantage": 1.0,
        "rho": 0.0,
        "market_prior_strength": 20.0,
        "market_prior_source": "history",
    }
    model = MatchScoreModel(
        ModelSpec(name="history", kind="dixon_coles", params=params)
    )
    model.fit(PITData(store).view(as_of))
    before = model.predict(PITData(store).view(as_of), fixtures).pmf.copy()
    kick4 = store.read("fixtures").filter(pl.col("fixture_id") == 4)["kickoff_time"][0]
    # Deadline-visible target line would move the absent-source blend; history
    # predict must ignore it (past closes for the fit stay in the store).
    store.write(
        "match_odds",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON],
                    "fixture_id": [4],
                    "team_h_key": [1],
                    "team_a_key": [2],
                    "kickoff_time": [kick4],
                    "home_odds": [1.05],
                    "draw_odds": [12.0],
                    "away_odds": [25.0],
                    "over_25_odds": [1.40],
                    "under_25_odds": [2.80],
                    "odds_kind": ["closing_proxy"],
                    "available_at": [cutoff - timedelta(hours=1)],
                    "source": ["football_data"],
                }
            )
        ),
    )
    after = model.predict(PITData(store).view(as_of), fixtures).pmf
    np.testing.assert_allclose(before, after, rtol=1e-10, atol=1e-12)


def test_invalid_market_prior_source_raises() -> None:
    with pytest.raises(ValueError, match="market_prior_source"):
        MatchScoreModel(
            ModelSpec(
                name="bad",
                kind="dixon_coles",
                params={"market_prior_source": "as_of"},
            )
        )


def test_dixon_coles_history_yaml_predicts(tmp_path: Path) -> None:
    store = _season_with_past_closes(DataStore(tmp_path))
    spec = load_model_specs()["dixon_coles_history_v1"]
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


def _season_with_past_closes(store: DataStore) -> DataStore:
    """Closing lines on finished matches (available_at = kickoff)."""
    store = _season(store)
    fixtures = store.read("fixtures").filter(pl.col("finished"))
    # Short home prices whenever team 1 is home → strong attack prior for team 1.
    rows = {
        "season": [],
        "fixture_id": [],
        "team_h_key": [],
        "team_a_key": [],
        "kickoff_time": [],
        "home_odds": [],
        "draw_odds": [],
        "away_odds": [],
        "over_25_odds": [],
        "under_25_odds": [],
        "odds_kind": [],
        "available_at": [],
        "source": [],
    }
    for row in fixtures.iter_rows(named=True):
        home = int(row["team_h_key"])
        if home == 1:
            home_odds, away_odds = 1.30, 10.0
        else:
            home_odds, away_odds = 3.50, 2.10
        rows["season"].append(_SEASON)
        rows["fixture_id"].append(int(row["fixture_id"]))
        rows["team_h_key"].append(home)
        rows["team_a_key"].append(int(row["team_a_key"]))
        rows["kickoff_time"].append(row["kickoff_time"])
        rows["home_odds"].append(home_odds)
        rows["draw_odds"].append(4.50)
        rows["away_odds"].append(away_odds)
        rows["over_25_odds"].append(1.80)
        rows["under_25_odds"].append(2.00)
        rows["odds_kind"].append("closing_proxy")
        rows["available_at"].append(row["kickoff_time"])
        rows["source"].append("football_data")
    store.write("match_odds", _utc(pl.DataFrame(rows)))
    return store


def _season_with_target_odds(
    store: DataStore, *, odds_before_deadline: bool
) -> DataStore:
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
