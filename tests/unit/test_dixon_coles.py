"""Dixon-Coles reads only finished fixtures visible at the deadline."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import fplt.models.components.score  # noqa: F401 — registers the component
from fplt.backtest.components import (
    append_components,
    fill_scoreline_actuals,
    ranked_probability_score,
    read_components,
    record_scorelines,
)
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.components.score.dixon_coles import (
    MatchScoreModel,
    _dc_correction,
    _joint,
    _match_counts,
    _promoted_teams,
    _shrink_ratings,
)
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


def test_low_score_correction_matches_the_four_cases() -> None:
    assert _dc_correction(0, 0, 1.0, 1.0, -0.1) == pytest.approx(1.1)
    assert _dc_correction(0, 1, 1.0, 1.0, -0.1) == pytest.approx(0.9)
    assert _dc_correction(1, 0, 1.0, 1.0, -0.1) == pytest.approx(0.9)
    assert _dc_correction(1, 1, 1.0, 1.0, -0.1) == pytest.approx(1.1)
    assert _dc_correction(2, 2, 1.0, 1.0, -0.1) == pytest.approx(1.0)


def test_ranked_probability_score_is_zero_for_a_certain_hit() -> None:
    certain = ranked_probability_score(
        np.array([1.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([2.0]),
        np.array([0.0]),
    )
    wrong = ranked_probability_score(
        np.array([1.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([0.0]),
        np.array([2.0]),
    )
    assert certain[0] == pytest.approx(0.0)
    assert wrong[0] == pytest.approx(1.0)


def test_specs_register_as_components_not_predictors() -> None:
    specs = load_model_specs()
    assert get_component("dixon_coles") is MatchScoreModel
    assert get_component("poisson_score") is MatchScoreModel
    assert specs["dixon_coles_v1"].params["rho"] == -0.1
    assert specs["poisson_score_v1"].params["rho"] == 0.0
    assert specs["dixon_coles_v2"].params["fit_rho"] is True
    assert specs["dixon_coles_v2"].params["fit_home_advantage"] is True
    assert specs["dixon_coles_v2"].kind == "dixon_coles"
    assert {"dixon_coles", "poisson_score"} <= set(registered_components())
    assert "dixon_coles" not in registered_kinds()


def test_fit_ignores_a_result_published_after_the_deadline(tmp_path: Path) -> None:
    store = _season(DataStore(tmp_path))
    late = (
        store.read("fixtures")
        .filter(pl.col("fixture_id") == 1)
        .with_columns(
            pl.lit(9).alias("team_h_score"),
            pl.lit(datetime(2024, 9, 1, tzinfo=UTC)).alias("available_at"),
        )
    )
    store.write("fixtures", late)
    spec = ModelSpec(
        name="dixon_coles",
        kind="dixon_coles",
        params={"xi": 0.0, "home_advantage": 1.2, "rho": -0.05},
    )
    as_of = GWKey(_SEASON, 4)
    first = _predict(store, spec, as_of)
    bumped = store.read("fixtures").filter(
        pl.col("available_at") > store.deadline(as_of)
    )
    revised = bumped.with_columns(
        pl.lit(0).alias("team_h_score"),
        pl.lit(8).alias("team_a_score"),
    )
    store.write("fixtures", revised)
    second = _predict(store, spec, as_of)
    # L-BFGS can differ in the last bits when the visible rows are the same.
    np.testing.assert_allclose(first.pmf, second.pmf, rtol=1e-5, atol=1e-8)
    assert bumped.height > 0


def test_scoreline_ledger_waits_for_data_checked(tmp_path: Path) -> None:
    store = _season(DataStore(tmp_path))
    view = PITData(store).view(GWKey(_SEASON, 4))
    specs = load_model_specs()
    fixtures = (
        view.latest("fixtures", by=["season", "fixture_id"])
        .select("season", "gw", "fixture_id", "team_h_key", "team_a_key")
        .collect()
    )
    for name in ("dixon_coles_v1", "poisson_score_v1"):
        model = MatchScoreModel(specs[name])
        model.fit(view)
        append_components(tmp_path, record_scorelines(model, view, fixtures))
    fill_scoreline_actuals(tmp_path, store)
    filled = read_components(tmp_path)
    checked = filled.filter(pl.col("target_gw") < 4)
    pending = filled.filter(pl.col("target_gw") == 4)
    assert set(filled["model_id"].to_list()) == {
        specs["dixon_coles_v1"].model_id,
        specs["poisson_score_v1"].model_id,
    }
    assert checked["rps"].null_count() == 0
    assert pending["actual_home"].null_count() == pending.height
    totals = filled.select(
        (pl.col("p_home") + pl.col("p_draw") + pl.col("p_away")).alias("p")
    )
    assert totals["p"].min() == pytest.approx(1.0, abs=1e-9)


def test_joint_with_zero_rho_is_independent() -> None:
    joint = _joint(np.array([1.2]), np.array([0.8]), 0.0)
    outer = joint[0, :11, :11]
    home = outer.sum(axis=1)
    away = outer.sum(axis=0)
    rebuilt = np.outer(home, away)
    rebuilt /= rebuilt.sum()
    np.testing.assert_allclose(outer / outer.sum(), rebuilt, atol=1e-8)


def test_v2_estimates_negative_rho_when_low_scores_cluster(tmp_path: Path) -> None:
    store = _clustered_low_scores(DataStore(tmp_path))
    spec = ModelSpec(
        name="dixon_coles_v2_rho",
        kind="dixon_coles",
        params={
            "xi": 0.0,
            "home_advantage": 1.0,
            "rho": 0.0,
            "fit_rho": True,
            "fit_home_advantage": False,
        },
    )
    model = MatchScoreModel(spec)
    model.fit(PITData(store).view(GWKey(_SEASON, 6)))
    assert model._rho < -0.01


def test_v2_estimates_home_advantage_above_one(tmp_path: Path) -> None:
    store = _home_heavy(DataStore(tmp_path))
    spec = ModelSpec(
        name="dixon_coles_v2_ha",
        kind="dixon_coles",
        params={
            "xi": 0.0,
            "home_advantage": 1.0,
            "rho": 0.0,
            "fit_rho": False,
            "fit_home_advantage": True,
        },
    )
    model = MatchScoreModel(spec)
    model.fit(PITData(store).view(GWKey(_SEASON, 8)))
    assert model._home > 1.05


def test_shrinkage_pulls_sparse_ratings_toward_neutral() -> None:
    attack = np.array([2.0, 0.5])
    defence = np.array([0.4, 1.6])
    counts = np.array([1.0, 20.0])
    shrunk_a, shrunk_d = _shrink_ratings(
        attack, defence, counts, [1, 2], set(), new_prior=9.0, promoted_prior=0.0
    )
    # One match + strong prior → almost fully pulled to 1.
    assert abs(shrunk_a[0] - 1.0) < abs(attack[0] - 1.0)
    assert abs(shrunk_d[0] - 1.0) < abs(defence[0] - 1.0)
    # Twenty matches move much less than the sparse team.
    assert abs(shrunk_a[1] - attack[1]) < abs(shrunk_a[0] - attack[0])


def test_promoted_prior_is_stronger_than_new_team_prior() -> None:
    attack = np.array([2.5])
    defence = np.array([0.3])
    counts = np.array([2.0])
    base_a, _ = _shrink_ratings(
        attack, defence, counts, [99], set(), new_prior=3.0, promoted_prior=0.0
    )
    promo_a, _ = _shrink_ratings(
        attack, defence, counts, [99], {99}, new_prior=3.0, promoted_prior=8.0
    )
    assert abs(promo_a[0] - 1.0) < abs(base_a[0] - 1.0)


def test_promoted_teams_are_those_missing_from_earlier_seasons(tmp_path: Path) -> None:
    store = _two_season_promotion(DataStore(tmp_path))
    view = PITData(store).view(GWKey(_SEASON, 2))
    promoted = _promoted_teams(view, [1, 2, 3])
    assert promoted == {3}


def test_match_counts_sum_home_and_away_rows() -> None:
    played = pl.DataFrame(
        {"team_h_key": [1, 2, 1], "team_a_key": [2, 1, 3], "team_h_score": [1, 0, 2]}
    )
    counts = _match_counts(played, [1, 2, 3])
    np.testing.assert_array_equal(counts, np.array([3.0, 2.0, 1.0]))


def test_v2_yaml_fit_predicts_valid_scorelines(tmp_path: Path) -> None:
    store = _season(DataStore(tmp_path))
    spec = load_model_specs()["dixon_coles_v2"]
    model = MatchScoreModel(spec)
    view = PITData(store).view(GWKey(_SEASON, 4))
    model.fit(view)
    dist = model.predict(
        view, pl.DataFrame({"fixture_id": [4], "team_h_key": [1], "team_a_key": [2]})
    )
    assert dist.pmf.shape == (1, 33, 33)
    assert dist.pmf.sum() == pytest.approx(1.0)


def _predict(store: DataStore, spec: ModelSpec, as_of: GWKey):
    model = MatchScoreModel(spec)
    view = PITData(store).view(as_of)
    model.fit(view)
    upcoming = pl.DataFrame({"fixture_id": [4], "team_h_key": [1], "team_a_key": [2]})
    return model.predict(view, upcoming)


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


def _utc(frame: pl.DataFrame) -> pl.DataFrame:
    columns = [
        pl.col(name).cast(pl.Datetime("us", "UTC"))
        for name, dtype in frame.schema.items()
        if isinstance(dtype, pl.Datetime)
    ]
    return frame.with_columns(columns) if columns else frame


def _write_gameweeks(store: DataStore, n: int, season: str = _SEASON) -> list[datetime]:
    kickoffs = [_START + timedelta(days=7 * gw) for gw in range(1, n + 1)]
    deadlines = [kick - timedelta(hours=21) for kick in kickoffs]
    store.write(
        "gameweeks",
        _utc(
            pl.DataFrame(
                {
                    "season": [season] * n,
                    "gw": list(range(1, n + 1)),
                    "deadline_time": deadlines,
                    "finished": [True] * (n - 1) + [False],
                    "data_checked": [True] * (n - 1) + [False],
                    "available_at": [_START] * n,
                    "source": ["fpl_api"] * n,
                }
            )
        ),
    )
    return kickoffs


def _clustered_low_scores(store: DataStore) -> DataStore:
    """Many 0-0 and 1-1 results so MLE ρ should go negative."""
    n = 6
    kickoffs = _write_gameweeks(store, n)
    scores = [(0, 0), (1, 1), (0, 0), (1, 1), (0, 0), (None, None)]
    homes = [1, 2, 1, 2, 1, 1]
    aways = [2, 1, 2, 1, 2, 2]
    available = [kick + timedelta(hours=3) for kick in kickoffs]
    available[-1] = kickoffs[-1] - timedelta(days=1)
    store.write(
        "fixtures",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * n,
                    "fixture_id": list(range(1, n + 1)),
                    "gw": list(range(1, n + 1)),
                    "team_h_key": homes,
                    "team_a_key": aways,
                    "kickoff_time": kickoffs,
                    "team_h_score": [score[0] for score in scores],
                    "team_a_score": [score[1] for score in scores],
                    "finished": [True] * (n - 1) + [False],
                    "available_at": available,
                    "source": ["fpl_api"] * n,
                }
            )
        ),
    )
    return store


def _home_heavy(store: DataStore) -> DataStore:
    """Home side always scores more so estimated HA should exceed 1."""
    n = 8
    kickoffs = _write_gameweeks(store, n)
    scores = [(2, 0), (3, 1), (2, 1), (1, 0), (2, 0), (3, 0), (2, 1), (None, None)]
    homes = [1, 2, 1, 2, 1, 2, 1, 1]
    aways = [2, 1, 2, 1, 2, 1, 2, 2]
    available = [kick + timedelta(hours=3) for kick in kickoffs]
    available[-1] = kickoffs[-1] - timedelta(days=1)
    store.write(
        "fixtures",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * n,
                    "fixture_id": list(range(1, n + 1)),
                    "gw": list(range(1, n + 1)),
                    "team_h_key": homes,
                    "team_a_key": aways,
                    "kickoff_time": kickoffs,
                    "team_h_score": [score[0] for score in scores],
                    "team_a_score": [score[1] for score in scores],
                    "finished": [True] * (n - 1) + [False],
                    "available_at": available,
                    "source": ["fpl_api"] * n,
                }
            )
        ),
    )
    return store


def _two_season_promotion(store: DataStore) -> DataStore:
    """Teams 1–2 played 2023-24; team 3 appears only in 2024-25."""
    prior = "2023-24"
    prior_start = datetime(2023, 8, 1, tzinfo=UTC)
    store.write(
        "gameweeks",
        _utc(
            pl.DataFrame(
                {
                    "season": [prior, prior, _SEASON, _SEASON],
                    "gw": [1, 2, 1, 2],
                    "deadline_time": [
                        prior_start,
                        prior_start + timedelta(days=7),
                        _START,
                        _START + timedelta(days=7),
                    ],
                    "finished": [True, True, True, False],
                    "data_checked": [True, True, True, False],
                    "available_at": [prior_start, prior_start, _START, _START],
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
                    "season": [prior, prior, _SEASON, _SEASON],
                    "fixture_id": [1, 2, 3, 4],
                    "gw": [1, 2, 1, 2],
                    "team_h_key": [1, 2, 1, 3],
                    "team_a_key": [2, 1, 3, 2],
                    "kickoff_time": [
                        prior_start + timedelta(hours=21),
                        prior_start + timedelta(days=7, hours=21),
                        _START + timedelta(hours=21),
                        _START + timedelta(days=7, hours=21),
                    ],
                    "team_h_score": [1, 0, 2, None],
                    "team_a_score": [0, 1, 1, None],
                    "finished": [True, True, True, False],
                    "available_at": [
                        prior_start + timedelta(days=1),
                        prior_start + timedelta(days=8),
                        _START + timedelta(days=1),
                        _START + timedelta(days=6),
                    ],
                    "source": ["fpl_api"] * 4,
                }
            )
        ),
    )
    return store
