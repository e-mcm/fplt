"""Raw de-vig 1X2 scoreline baseline (no odds_to_lambda)."""

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
from fplt.models.components.score.odds_1x2 import Odds1X2Score
from fplt.models.components.score.odds_lambda import de_vig, rescale_joint_to_1x2
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


def test_odds_1x2_registers_as_component_not_predictor() -> None:
    specs = load_model_specs()
    assert get_component("odds_1x2") is Odds1X2Score
    assert specs["odds_1x2_v1"].kind == "odds_1x2"
    assert specs["odds_1x2_v1"].params["use_closing_proxy"] is True
    assert "odds_1x2" in registered_components()
    assert "odds_1x2" not in registered_kinds()


def test_predict_1x2_margins_match_de_vig(tmp_path: Path) -> None:
    store = _league(DataStore(tmp_path), with_odds=True)
    spec = load_model_specs()["odds_1x2_v1"]
    as_of = GWKey(_SEASON, 3)
    view = PITData(store).view(as_of)
    fixtures = (
        view.latest("fixtures", by=["season", "fixture_id"])
        .filter((pl.col("season") == _SEASON) & (pl.col("gw") == 3))
        .select(
            "season", "gw", "fixture_id", "team_h_key", "team_a_key", "kickoff_time"
        )
        .collect()
    )
    model = Odds1X2Score(spec)
    model.fit(view)
    dist = model.predict(view, fixtures)
    ledger = record_scorelines(model, view, fixtures)
    expected = de_vig(np.array([1.40, 4.50, 7.50]))
    assert ledger["p_home"][0] == pytest.approx(float(expected[0]), rel=1e-6)
    assert ledger["p_draw"][0] == pytest.approx(float(expected[1]), rel=1e-6)
    assert ledger["p_away"][0] == pytest.approx(float(expected[2]), rel=1e-6)
    assert float(dist.pmf[0].sum()) == pytest.approx(1.0)


def test_missing_odds_uses_fallback_1x2(tmp_path: Path) -> None:
    store = _league(DataStore(tmp_path), with_odds=False)
    spec = ModelSpec(
        name="odds_1x2",
        kind="odds_1x2",
        params={
            "use_closing_proxy": True,
            "fallback_p_home": 0.5,
            "fallback_p_draw": 0.3,
            "fallback_p_away": 0.2,
        },
        refit="never",
    )
    view = PITData(store).view(GWKey(_SEASON, 3))
    fixtures = pl.DataFrame(
        {
            "season": [_SEASON],
            "gw": [3],
            "fixture_id": [3],
            "team_h_key": [1],
            "team_a_key": [2],
        }
    )
    ledger = record_scorelines(Odds1X2Score(spec), view, fixtures)
    assert ledger["p_home"][0] == pytest.approx(0.5)
    assert ledger["p_draw"][0] == pytest.approx(0.3)
    assert ledger["p_away"][0] == pytest.approx(0.2)


def test_rescale_joint_to_1x2_preserves_targets() -> None:
    goals = np.arange(8)
    shape = np.outer(np.exp(-goals / 2), np.exp(-goals / 3))
    shape /= shape.sum()
    out = rescale_joint_to_1x2(shape, 0.55, 0.25, 0.20)
    home_gt = goals[:, None] > goals[None, :]
    home_eq = goals[:, None] == goals[None, :]
    home_lt = goals[:, None] < goals[None, :]
    assert float(out[home_gt].sum()) == pytest.approx(0.55)
    assert float(out[home_eq].sum()) == pytest.approx(0.25)
    assert float(out[home_lt].sum()) == pytest.approx(0.20)


def _league(store: DataStore, *, with_odds: bool) -> DataStore:
    kickoffs = [_START + timedelta(days=7 * gw) for gw in (1, 2, 3)]
    deadlines = [kick - timedelta(hours=21) for kick in kickoffs]
    store.write(
        "gameweeks",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * 3,
                    "gw": [1, 2, 3],
                    "deadline_time": deadlines,
                    "finished": [True, True, False],
                    "data_checked": [True, True, False],
                    "available_at": [_START] * 3,
                    "source": ["fpl_api"] * 3,
                }
            )
        ),
    )
    store.write(
        "fixtures",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * 3,
                    "fixture_id": [1, 2, 3],
                    "gw": [1, 2, 3],
                    "team_h_key": [1, 2, 1],
                    "team_a_key": [2, 1, 2],
                    "kickoff_time": kickoffs,
                    "team_h_score": [1, 0, None],
                    "team_a_score": [0, 1, None],
                    "finished": [True, True, False],
                    "available_at": [
                        kickoffs[0] + timedelta(hours=3),
                        kickoffs[1] + timedelta(hours=3),
                        deadlines[2] - timedelta(days=1),
                    ],
                    "source": ["fpl_api"] * 3,
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
                        "home_odds": [2.10, 1.40],
                        "draw_odds": [3.40, 4.50],
                        "away_odds": [3.50, 7.50],
                        "over_25_odds": [1.80, 1.70],
                        "under_25_odds": [2.00, 2.10],
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
