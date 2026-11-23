"""Dixon–Coles mid-season rest / congestion / form covariates (FDB-243)."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import fplt.models.components.score  # noqa: F401 — registers the component
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.components.score.covariates import (
    form_rating_multipliers,
    schedule_rate_multipliers,
)
from fplt.models.components.score.dixon_coles import MatchScoreModel
from fplt.models.spec import ModelSpec, load_model_specs
from fplt.types import GWKey

_SEASON = "2024-25"
_START = datetime(2024, 8, 1, tzinfo=UTC)
_VANILLA_PMF_RTOL = 1e-7
_VANILLA_PMF_ATOL = 1e-8


def test_covariate_specs_registered_without_changing_production() -> None:
    """Standalone + layered forks exist; production stays covariate-off."""
    specs = load_model_specs()
    assert "dixon_coles_covariate_v1" in specs
    assert "dc_layered_covariate_v1" in specs
    cov = specs["dixon_coles_covariate_v1"].params
    assert cov["rest_coef"] == pytest.approx(0.05)
    assert cov["congestion_coef"] == pytest.approx(0.05)
    assert cov["form_window"] == 5
    assert cov["form_strength"] == pytest.approx(0.2)
    assert cov["market_cs_strength"] == pytest.approx(0.5)
    prod = specs["dc_layered_v4"].components["score"].params
    assert prod.get("rest_coef", 0.0) == 0.0
    assert prod.get("congestion_coef", 0.0) == 0.0
    assert prod.get("form_strength", 0.0) == 0.0
    layered = specs["dc_layered_covariate_v1"].components["score"]
    assert layered.name == "dixon_coles_covariate_v1"
    assert layered.params["rest_coef"] == pytest.approx(0.05)
    assert layered.params["market_cs_strength"] == pytest.approx(0.5)


def test_zero_covariate_strengths_match_vanilla(tmp_path: Path) -> None:
    """Absent / zero rest, congestion, and form recover vanilla PMFs."""
    store = _season(DataStore(tmp_path))
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
    vanilla = MatchScoreModel(
        ModelSpec(name="dc_vanilla", kind="dixon_coles", params=base)
    )
    zero = MatchScoreModel(
        ModelSpec(
            name="dc_zero_cov",
            kind="dixon_coles",
            params={
                **base,
                "rest_coef": 0.0,
                "congestion_coef": 0.0,
                "form_window": 5,
                "form_strength": 0.0,
            },
        )
    )
    vanilla.fit(view)
    zero.fit(view)
    np.testing.assert_allclose(
        vanilla.predict(view, fixtures).pmf,
        zero.predict(view, fixtures).pmf,
        rtol=_VANILLA_PMF_RTOL,
        atol=_VANILLA_PMF_ATOL,
    )


def test_negative_covariate_coefs_raise() -> None:
    """Covariate strengths must be non-negative."""
    with pytest.raises(ValueError, match="rest_coef"):
        MatchScoreModel(
            ModelSpec(
                name="bad_rest",
                kind="dixon_coles",
                params={"rest_coef": -0.1},
            )
        )
    with pytest.raises(ValueError, match="congestion_coef"):
        MatchScoreModel(
            ModelSpec(
                name="bad_cong",
                kind="dixon_coles",
                params={"congestion_coef": -0.1},
            )
        )
    with pytest.raises(ValueError, match="form_strength"):
        MatchScoreModel(
            ModelSpec(
                name="bad_form",
                kind="dixon_coles",
                params={"form_strength": -0.1},
            )
        )


def test_rest_coef_raises_lambda_after_long_rest(tmp_path: Path) -> None:
    """Positive rest_coef increases home λ when home has more than a week off."""
    store = _season(DataStore(tmp_path), stretch_last_home_rest=True)
    view = PITData(store).view(GWKey(_SEASON, 4))
    fixtures = (
        store.read("fixtures")
        .filter(pl.col("fixture_id") == 4)
        .select("season", "fixture_id", "team_h_key", "team_a_key", "kickoff_time")
    )
    base = {"xi": 0.0, "home_advantage": 1.0, "rho": 0.0}
    vanilla = MatchScoreModel(ModelSpec(name="v", kind="dixon_coles", params=base))
    rested = MatchScoreModel(
        ModelSpec(
            name="r",
            kind="dixon_coles",
            params={**base, "rest_coef": 0.5, "congestion_coef": 0.0},
        )
    )
    vanilla.fit(view)
    rested.fit(view)
    v_pmf = vanilla.predict(view, fixtures).pmf[0]
    r_pmf = rested.predict(view, fixtures).pmf[0]
    # Higher home λ → more mass on high home goals (row sums of joint).
    v_home = v_pmf.sum(axis=1)
    r_home = r_pmf.sum(axis=1)
    assert float(np.arange(len(r_home)) @ r_home) > float(
        np.arange(len(v_home)) @ v_home
    )


def test_form_strength_moves_attack_after_high_scoring_run(tmp_path: Path) -> None:
    """Positive form_strength raises attack after a team over-scores."""
    store = _season(DataStore(tmp_path), blowout_home=True)
    view = PITData(store).view(GWKey(_SEASON, 4))
    fixtures = (
        store.read("fixtures")
        .filter(pl.col("fixture_id") == 4)
        .select("season", "fixture_id", "team_h_key", "team_a_key", "kickoff_time")
    )
    base = {"xi": 0.0, "home_advantage": 1.0, "rho": 0.0}
    vanilla = MatchScoreModel(ModelSpec(name="v", kind="dixon_coles", params=base))
    form = MatchScoreModel(
        ModelSpec(
            name="f",
            kind="dixon_coles",
            params={
                **base,
                "form_window": 3,
                "form_strength": 0.5,
            },
        )
    )
    vanilla.fit(view)
    form.fit(view)
    assert form._attack_form.get(1, 1.0) > 1.0
    v_pmf = vanilla.predict(view, fixtures).pmf[0]
    f_pmf = form.predict(view, fixtures).pmf[0]
    v_home = v_pmf.sum(axis=1)
    f_home = f_pmf.sum(axis=1)
    assert float(np.arange(len(f_home)) @ f_home) > float(
        np.arange(len(v_home)) @ v_home
    )


def test_schedule_multipliers_ones_when_coefs_zero(tmp_path: Path) -> None:
    """Schedule helper returns ones when both coefs are off."""
    store = _season(DataStore(tmp_path))
    view = PITData(store).view(GWKey(_SEASON, 4))
    fixtures = store.read("fixtures").filter(pl.col("fixture_id") == 4)
    home, away = schedule_rate_multipliers(
        view, fixtures, rest_coef=0.0, congestion_coef=0.0
    )
    np.testing.assert_allclose(home, [1.0])
    np.testing.assert_allclose(away, [1.0])


def test_form_multipliers_ones_when_disabled() -> None:
    """Form helper is a no-op when window or strength is off."""
    played = pl.DataFrame(
        {
            "season": [_SEASON],
            "kickoff_time": [_START],
            "team_h_key": [1],
            "team_a_key": [2],
            "team_h_score": [2],
            "team_a_score": [0],
        }
    )
    attack = {1: 1.2, 2: 0.9}
    defence = {1: 0.8, 2: 1.1}
    a_mult, d_mult = form_rating_multipliers(
        played,
        [1, 2],
        attack,
        defence,
        home_advantage=1.0,
        form_window=0,
        form_strength=0.5,
        season=_SEASON,
    )
    assert a_mult == {1: 1.0, 2: 1.0}
    assert d_mult == {1: 1.0, 2: 1.0}


def _season(
    store: DataStore,
    *,
    stretch_last_home_rest: bool = False,
    blowout_home: bool = False,
) -> DataStore:
    """Four GWs; optional long rest before GW4 or blowout results for form."""
    kickoffs = [_START + timedelta(days=7 * gw) for gw in range(1, 5)]
    if stretch_last_home_rest:
        # Team 1 plays GW1 + GW3 (~day 21), then GW4 after another ~21 days
        # so rest_weeks ≈ 3 (above the neutral 1-week baseline).
        kickoffs[3] = kickoffs[2] + timedelta(days=21)
    deadlines = [kick - timedelta(hours=21) for kick in kickoffs]
    if blowout_home:
        results: list[tuple[int | None, int | None]] = [
            (4, 0),
            (0, 1),
            (5, 0),
            (None, None),
        ]
    else:
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
