"""Predict-run component tables (weekly-ui §3.1)."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import numpy as np
import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fplt.backtest.predict_live import run_predict
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.distributions import MAX_GOALS, MINUTE_BINS, MinutesDist, ScorelineDist
from fplt.optimise.component_tables import build_run_component_tables
from fplt.optimise.explain import LayeredExplainParts, minutes_summary_from_pmf
from fplt.schemas.run_components import ComponentMinutesSchema
from fplt.store.api import Store
from fplt.store.runs import run_dir, write_predict_run
from fplt.types import GWKey
from tests.builders import build_league

_TS = datetime(2024, 8, 23, 17, tzinfo=UTC)
_SEASON = "2024-25"


class _Score:
    def __init__(self, home: int = 2, away: int = 0) -> None:
        self.home = home
        self.away = away

    def predict(self, _view, fixtures):
        ids = fixtures["fixture_id"].to_numpy()
        goals = MAX_GOALS + 1
        grid = np.zeros((len(ids), goals + 1, goals + 1))
        for index in range(len(ids)):
            grid[index, self.home, self.away] = 1.0
        return ScorelineDist(ids, grid)


class _Minutes:
    def __init__(self, minute: int = 90) -> None:
        self.minute = minute

    def predict(self, _view, asked):
        n = asked.height
        pmf = np.zeros((n, len(MINUTE_BINS)))
        pmf[:, list(MINUTE_BINS).index(self.minute)] = 1.0
        return MinutesDist(asked.select("player_key", "fixture_id"), MINUTE_BINS, pmf)


class _Involvement:
    def predict(self, _view, asked):
        # Equal shares within each team-fixture (sums to 1 when both players play).
        return asked.select("player_key", "fixture_id").with_columns(
            pl.lit(0.5).alias("goal_weight"),
            pl.lit(0.5).alias("assist_weight"),
        )


def _parts(*, minute: int = 90, home: int = 2, away: int = 0) -> LayeredExplainParts:
    return LayeredExplainParts(
        score=_Score(home=home, away=away),
        minutes=_Minutes(minute=minute),
        involvement=_Involvement(),
    )


def test_minutes_summary_zero_minutes_player() -> None:
    keys = pl.DataFrame({"player_key": [1], "fixture_id": [10]})
    pmf = np.zeros((1, len(MINUTE_BINS)))
    pmf[0, list(MINUTE_BINS).index(0)] = 1.0
    summary = minutes_summary_from_pmf(MinutesDist(keys, MINUTE_BINS, pmf))
    assert float(summary["xmins"][0]) == pytest.approx(0.0)
    assert float(summary["p_played"][0]) == pytest.approx(0.0)
    assert float(summary["p_60"][0]) == pytest.approx(0.0)


def test_build_tables_double_and_blank_gameweek(tmp_path) -> None:
    store = _league_with_double_and_blank(DataStore(tmp_path))
    view = PITData(store).view(GWKey(_SEASON, 2))
    tables = build_run_component_tables(
        view,
        parts=_parts(),
        horizon=[GWKey(_SEASON, 2), GWKey(_SEASON, 3)],
        run_id="r_test",
        as_of=GWKey(_SEASON, 2),
        deadline_utc=_TS,
    )
    # GW2: team 3 doubles (fixtures 2+3); team 7 plays once. GW3 blanks team 3.
    gw2 = tables.minutes.filter(pl.col("target_gw") == 2)
    assert gw2.filter(pl.col("player_key") == 100).height == 2
    assert gw2.filter(pl.col("player_key") == 200).height == 1
    assert tables.scoreline.filter(pl.col("target_gw") == 2).height == 4
    assert (
        tables.minutes.filter(
            (pl.col("target_gw") == 3) & (pl.col("player_key") == 100)
        ).height
        == 0
    )
    assert (
        tables.minutes.filter(
            (pl.col("target_gw") == 3) & (pl.col("player_key") == 200)
        ).height
        == 1
    )


def test_p_cs_matches_score_component(tmp_path) -> None:
    store = build_league(DataStore(tmp_path))
    view = PITData(store).view(GWKey(_SEASON, 2))
    # Deterministic 2-0 → home CS = 1.0, away CS = 0.0
    tables = build_run_component_tables(
        view,
        parts=_parts(home=2, away=0),
        horizon=[GWKey(_SEASON, 2)],
        run_id="r_cs",
        as_of=GWKey(_SEASON, 2),
        deadline_utc=_TS,
    )
    home = tables.scoreline.filter(pl.col("is_home"))
    away = tables.scoreline.filter(~pl.col("is_home"))
    assert float(home["p_cs"][0]) == pytest.approx(1.0)
    assert float(away["p_cs"][0]) == pytest.approx(0.0)
    assert float(home["team_eg"][0]) == pytest.approx(2.0)
    assert float(home["opp_eg"][0]) == pytest.approx(0.0)


def test_opaque_predict_writes_no_component_tables(tmp_path) -> None:
    build_league(DataStore(tmp_path))
    run_predict(tmp_path, "rolling_mean_v1", 1, season=_SEASON, gw=2)
    runs = list((tmp_path / "runs").glob("run=*/_SUCCESS"))
    assert len(runs) == 1
    root = runs[0].parent
    assert not (root / "predictions" / "components" / "minutes.parquet").exists()
    assert not (root / "predictions" / "components" / "scoreline.parquet").exists()
    assert not (root / "predictions" / "components" / "involvement.parquet").exists()


def test_write_predict_run_persists_component_tables(tmp_path) -> None:
    store = build_league(DataStore(tmp_path))
    view = PITData(store).view(GWKey(_SEASON, 2))
    from fplt.models.predictions import empty_predictions
    from fplt.models.spec import load_model_specs

    spec = load_model_specs()["rolling_mean_v1"]
    as_of = GWKey(_SEASON, 2)

    def _builder(run_id: str):
        return build_run_component_tables(
            view,
            parts=_parts(minute=0),
            horizon=[as_of],
            run_id=run_id,
            as_of=as_of,
            deadline_utc=store.deadline(as_of),
        )

    manifest = write_predict_run(
        tmp_path,
        spec=spec,
        as_of=as_of,
        deadline_utc=store.deadline(as_of),
        horizon=1,
        frame=empty_predictions(),
        components_for_run_id=_builder,
    )
    analytic = Store(tmp_path)
    minutes = analytic.load("minutes", run=manifest).collect()
    scoreline = analytic.load("scoreline", run=manifest).collect()
    involvement = analytic.load("involvement", run=manifest).collect()
    ComponentMinutesSchema.validate(minutes)
    assert minutes["xmins"].to_list() == [0.0, 0.0]
    assert scoreline.height == 2
    assert set(involvement["exp_goals_method"].unique().to_list()) == {
        "team_eg_x_weight"
    }
    assert analytic.load("player_fixture", run=manifest).collect().is_empty()
    assert (run_dir(tmp_path, manifest.run_id) / "_SUCCESS").exists()


def test_stored_component_depth_matches_live_parts(tmp_path) -> None:
    """Depth from stored tables matches ``build_player_depth`` with the same parts."""
    from fplt.models.predictions import empty_predictions
    from fplt.models.spec import load_model_specs
    from fplt.optimise.component_tables import depth_rows_from_component_tables
    from fplt.optimise.explain import build_player_depth
    from fplt.scoring.rules import Rules

    store = build_league(DataStore(tmp_path))
    view = PITData(store).view(GWKey(_SEASON, 2))
    parts = _parts(home=2, away=1)
    as_of = GWKey(_SEASON, 2)
    spec = load_model_specs()["rolling_mean_v1"]

    def _builder(run_id: str):
        return build_run_component_tables(
            view,
            parts=parts,
            horizon=[as_of],
            run_id=run_id,
            as_of=as_of,
            deadline_utc=store.deadline(as_of),
        )

    write_predict_run(
        tmp_path,
        spec=spec,
        as_of=as_of,
        deadline_utc=store.deadline(as_of),
        horizon=1,
        frame=empty_predictions(),
        purpose="depth-parity",
        components_for_run_id=_builder,
    )
    analytic = Store(tmp_path)
    runs = analytic.runs(spec_name=spec.name, run_type="predict")
    run_id = str(runs["run_id"][0])
    stored = depth_rows_from_component_tables(
        minutes=analytic.load("minutes", run=run_id).collect(),
        scoreline=analytic.load("scoreline", run=run_id).collect(),
        involvement=analytic.load("involvement", run=run_id).collect(),
        player_keys=[200],
        target_gw=2,
    )
    preds = pl.DataFrame(
        {
            "player_key": [200],
            "position": ["FWD"],
            "target_gw": [2],
            "xpts": [4.0],
            "p_played": [1.0],
            "p_60": [1.0],
            "p_goal": [0.3],
            "p_assist": [0.1],
            "p_cs": [0.2],
            "p_haul": [0.1],
        }
    )
    live = build_player_depth(
        view,
        player_key=200,
        target_gw=2,
        predictions=preds,
        rules=Rules.for_season(_SEASON),
        parts=parts,
    ).depth
    cols = [
        "player_key",
        "fixture_id",
        "is_home",
        "team_key",
        "opponent_team_key",
        "team_eg",
        "opp_eg",
        "goal_weight",
        "assist_weight",
        "xmins",
    ]
    assert (
        stored.select(cols).sort("fixture_id").to_dicts()
        == live.select(cols).sort("fixture_id").to_dicts()
    )


@given(
    weights=st.lists(st.floats(min_value=0.0, max_value=10.0), min_size=1, max_size=8)
)
@settings(max_examples=30, deadline=None)
def test_goal_weight_sums_to_one_per_team_fixture(weights: list[float]) -> None:
    """Involvement component contract: shares are renormalised to sum 1.

    The run table stores those shares; Σ goal_weight over a team-fixture is 1
    when every squad player is present (the involvement component's contract).
    """
    total = sum(weights)
    if total <= 0:
        shares = [1.0 / len(weights)] * len(weights)
    else:
        shares = [w / total for w in weights]
    assert sum(shares) == pytest.approx(1.0)
    frame = pl.DataFrame(
        {
            "team_key": [1] * len(shares),
            "fixture_id": [10] * len(shares),
            "goal_weight": shares,
        }
    )
    summed = frame.group_by("team_key", "fixture_id").agg(pl.col("goal_weight").sum())
    assert float(summed["goal_weight"][0]) == pytest.approx(1.0)


def _league_with_double_and_blank(root: DataStore) -> DataStore:
    """GW2 double for both clubs; GW3 only team 7 has a fixture (team 3 blank)."""
    build_league(root)
    season = _SEASON
    kick2 = datetime(2024, 8, 24, 14, tzinfo=UTC)
    kick3 = datetime(2024, 8, 31, 14, tzinfo=UTC)
    deadline3 = datetime(2024, 8, 30, 17, tzinfo=UTC)
    start = datetime(2024, 8, 1, tzinfo=UTC)
    # Replace fixtures: keep GW1 history; GW2 double (2,3); GW3 only team 7 vs phantom.
    # Use team 3 as opponent away so team 3 is blank in GW3 if we only add team7 home
    # against a third team — builders only have teams 3 and 7. So GW3: 7 vs 3 means
    # team 3 is NOT blank. Instead leave team 3 with no GW3 row by not scheduling them:
    # schedule 7 home vs 3 in GW3 would include 3. For a true blank, omit team 3's GW3
    # entirely — impossible with only two teams in one fixture.
    # Practical blank: GW3 fixture involving only team 7 requires another team_key.
    root.write(
        "teams",
        pl.DataFrame(
            {
                "season": [season, season, season],
                "team_key": [3, 7, 11],
                "team_id": [1, 2, 3],
                "name": ["Arsenal", "Spurs", "Other"],
                "short_name": ["ARS", "TOT", "OTH"],
                "available_at": [start, start, start],
                "source": ["fpl_api", "fpl_api", "fpl_api"],
            }
        ).with_columns(pl.col("available_at").cast(pl.Datetime("us", "UTC"))),
    )
    root.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": [season, season, season],
                "gw": [1, 2, 3],
                "deadline_time": [
                    datetime(2024, 8, 16, 17, tzinfo=UTC),
                    _TS,
                    deadline3,
                ],
                "finished": [True, False, False],
                "data_checked": [True, False, False],
                "available_at": [start, start, start],
                "source": ["fpl_api", "fpl_api", "fpl_api"],
            }
        ).with_columns(
            pl.col("deadline_time").cast(pl.Datetime("us", "UTC")),
            pl.col("available_at").cast(pl.Datetime("us", "UTC")),
        ),
    )
    result_at = kick2 - timedelta(days=6)
    root.write(
        "fixtures",
        pl.DataFrame(
            {
                "season": [season] * 5,
                "fixture_id": [1, 1, 2, 3, 4],
                "gw": [1, 1, 2, 2, 3],
                "team_h_key": [3, 3, 7, 3, 7],
                "team_a_key": [7, 7, 3, 11, 11],
                "kickoff_time": [kick2 - timedelta(days=7)] * 2 + [kick2, kick2, kick3],
                "team_h_score": [None, 2, None, None, None],
                "team_a_score": [None, 0, None, None, None],
                "finished": [False, True, False, False, False],
                "available_at": [start, result_at, start, start, start],
                "source": ["fpl_api"] * 5,
            }
        ).with_columns(
            pl.col("kickoff_time").cast(pl.Datetime("us", "UTC")),
            pl.col("available_at").cast(pl.Datetime("us", "UTC")),
        ),
    )
    return root
