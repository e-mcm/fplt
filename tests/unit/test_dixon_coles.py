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
    assert set(registered_components()) == {"dixon_coles", "poisson_score"}
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
    np.testing.assert_allclose(first.pmf, second.pmf)
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
