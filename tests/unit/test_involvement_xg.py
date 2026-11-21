"""Understat xG involvement blends with FPL rates and calibrates sim knobs."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

import fplt.models.components.involvement  # noqa: F401 — registers the component
import fplt.models.composite  # noqa: F401 — registers the layered predictor
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.components.involvement.rates import (
    current_teams,
    position_goal_assist_rates,
)
from fplt.models.components.involvement.xg import InvolvementXg, _blend_rate
from fplt.models.composite import CompositePredictor, _apply_involvement_sim_params
from fplt.models.spec import ModelSpec, get_component, load_model_specs
from fplt.types import GWKey

_SEASON = "2024-25"
_START = datetime(2024, 8, 1, tzinfo=UTC)
_ASK = pl.DataFrame({"player_key": [1, 2], "fixture_id": [9, 9]})


def _xg_spec(**params: object) -> ModelSpec:
    return ModelSpec(
        name="involvement_xg",
        kind="involvement_xg",
        params={
            "window": 10,
            "shrinkage": 0,
            "penalty_bonus": 0.3,
            "xg_blend": 1.0,
            **params,
        },
    )


def test_spec_registers_as_a_component() -> None:
    specs = load_model_specs()
    assert get_component("involvement_xg") is InvolvementXg
    assert specs["involvement_xg_v1"].params["xg_blend"] == 0.7


def test_pure_xg_gives_the_high_xg_player_the_share(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        fpl=[_row(1, "FWD", 1, goals=0), _row(2, "FWD", 1, goals=2)],
        understat=[
            _ust(1, 1, xg=1.0, xa=0.0),
            _ust(2, 1, xg=0.0, xa=0.5),
        ],
    )
    frame = _predict(store)
    by_player = {row["player_key"]: row for row in frame.iter_rows(named=True)}
    assert by_player[1]["goal_weight"] == pytest.approx(1.0)
    assert by_player[2]["goal_weight"] == pytest.approx(0.0)
    assert by_player[1]["assist_weight"] == pytest.approx(0.0)
    assert by_player[2]["assist_weight"] == pytest.approx(1.0)


def test_missing_understat_falls_back_to_fpl(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        fpl=[_row(1, "FWD", 1, goals=2), _row(2, "DEF", 1, assists=1)],
        understat=[],
    )
    frame = _predict(store, xg_blend=0.7)
    by_player = {row["player_key"]: row for row in frame.iter_rows(named=True)}
    assert by_player[1]["goal_weight"] == pytest.approx(1.0)
    assert by_player[2]["assist_weight"] == pytest.approx(1.0)


def test_blend_mixes_xg_and_goals(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        fpl=[_row(1, "FWD", 1, goals=1), _row(2, "FWD", 1, goals=0)],
        understat=[
            _ust(1, 1, xg=0.0),
            _ust(2, 1, xg=1.0),
        ],
    )
    frame = _predict(store, xg_blend=0.5, shrinkage=0)
    weights = dict(
        zip(frame["player_key"].to_list(), frame["goal_weight"].to_list(), strict=True)
    )
    # rates: p1 = 0.5*0 + 0.5*1 = 0.5; p2 = 0.5*1 + 0.5*0 = 0.5
    assert weights[1] == pytest.approx(0.5)
    assert weights[2] == pytest.approx(0.5)


def test_penalty_miss_still_boosts_goals(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        fpl=[_row(1, "FWD", 1, pens_missed=1), _row(2, "FWD", 1)],
        understat=[_ust(1, 1, xg=0.0), _ust(2, 1, xg=0.0)],
    )
    frame = _predict(store, xg_blend=1.0)
    weights = dict(
        zip(frame["player_key"].to_list(), frame["goal_weight"].to_list(), strict=True)
    )
    assert weights[1] == pytest.approx(1.0)
    assert weights[2] == pytest.approx(0.0)


def test_sim_params_come_from_history(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        fpl=[
            _row(1, "FWD", 1, goals=2, assists=1, own_goals=0),
            _row(2, "DEF", 1, goals=0, assists=0, own_goals=1),
        ],
        understat=[],
    )
    view = PITData(store).view(GWKey(_SEASON, 3))
    model = InvolvementXg(_xg_spec())
    model.fit(view)
    # own / (goals + own) = 1/3 → weight = (1/3)/(2/3) = 0.5
    assert model.own_goal_weight == pytest.approx(0.5)
    assert model.p_assist == pytest.approx(0.5)


def test_layered_model_picks_up_fitted_sim_params(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        fpl=[
            _row(1, "FWD", 1, goals=2, assists=1),
            _row(2, "DEF", 1, own_goals=1),
        ],
        understat=[],
    )
    view = PITData(store).view(GWKey(_SEASON, 3))
    involvement = InvolvementXg(_xg_spec())
    involvement.fit(view)
    base = load_model_specs()["dc_layered_v1"]
    involvement_spec = load_model_specs()["involvement_xg_v1"]
    spec = base.model_copy(
        update={
            "params": {"n_sims": 50, "own_goal_weight": 0.05, "p_assist": 0.75},
            "components": {**base.components, "involvement": involvement_spec},
        }
    )
    model = CompositePredictor(spec)
    model._involvement = involvement
    _apply_involvement_sim_params(model)
    assert model._own_goal == pytest.approx(0.5)
    assert model._p_assist == pytest.approx(0.5)
    model._fitted = True
    artifact = tmp_path / "artifact"
    model.save(artifact)
    loaded = CompositePredictor.load(artifact, spec)
    assert loaded._own_goal == pytest.approx(0.5)
    assert loaded._p_assist == pytest.approx(0.5)


def test_xg_params_are_checked() -> None:
    with pytest.raises(ValueError, match="window"):
        InvolvementXg(_xg_spec(window=0))
    with pytest.raises(ValueError, match="xg_blend"):
        InvolvementXg(_xg_spec(xg_blend=1.5))
    with pytest.raises(ValueError, match="shrinkage"):
        InvolvementXg(_xg_spec(shrinkage=-1))
    with pytest.raises(ValueError, match="penalty_bonus"):
        InvolvementXg(_xg_spec(penalty_bonus=-0.1))


def test_rejects_predict_before_fit(tmp_path: Path) -> None:
    store = _store(tmp_path, fpl=[_row(1, "FWD", 1)], understat=[])
    view = PITData(store).view(GWKey(_SEASON, 3))
    model = InvolvementXg(_xg_spec())
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(view, _ASK)
    model.fit(view)
    with pytest.raises(ValueError, match="fixture_id"):
        model.predict(view, pl.DataFrame({"player_key": [1]}))
    assert model.predict(view, _ASK.clear()).is_empty()


def test_understat_only_player_uses_xg_rate(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        fpl=[
            _row(1, "FWD", 1, goals=0, minutes=0),
            _row(2, "FWD", 1, goals=0, minutes=0),
        ],
        understat=[_ust(1, 1, xg=2.0), _ust(2, 1, xg=0.0)],
    )
    # Zero-minute FPL rows still create history; wipe stats so rates come from xG.
    store.path("player_fixture_stats").unlink()
    frame = _predict(store, xg_blend=0.7)
    weights = dict(
        zip(frame["player_key"].to_list(), frame["goal_weight"].to_list(), strict=True)
    )
    assert weights[1] == pytest.approx(1.0)
    assert weights[2] == pytest.approx(0.0)


def test_all_own_goals_set_unit_weight(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        fpl=[_row(1, "DEF", 1, goals=0, own_goals=2)],
        understat=[],
    )
    view = PITData(store).view(GWKey(_SEASON, 3))
    model = InvolvementXg(_xg_spec())
    model.fit(view)
    assert model.own_goal_weight == pytest.approx(1.0)
    assert model.p_assist == pytest.approx(0.75)


def test_zero_minute_understat_skips_position_prior(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        fpl=[_row(1, "FWD", 1, goals=1), _row(2, "FWD", 1, goals=0)],
        understat=[
            _ust(1, 1, xg=0.5, minutes=0),
            _ust(2, 1, xg=0.5, minutes=0),
        ],
    )
    frame = _predict(store, xg_blend=1.0, shrinkage=0)
    # Both have zero minutes in Understat → rates 0; FPL unused at blend 1.
    assert frame["goal_weight"].to_list() == pytest.approx([0.5, 0.5])


def test_blend_rate_edges() -> None:
    assert _blend_rate(None, None, 0.7) == 0.0
    assert _blend_rate(1.0, None, 0.7) == 1.0
    assert _blend_rate(None, 2.0, 0.7) == 2.0


def test_rate_helpers_handle_empty_frames(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    store.write(
        "gameweeks",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON],
                    "gw": [3],
                    "deadline_time": [_deadline(3)],
                    "finished": [False],
                    "data_checked": [False],
                    "available_at": [_START],
                    "source": ["fpl_api"],
                }
            )
        ),
    )
    view = PITData(store).view(GWKey(_SEASON, 3))
    assert current_teams(view).is_empty()
    blank = pl.DataFrame(
        {
            "minutes": [0],
            "position": ["FWD"],
            "goals_scored": [0],
            "assists": [0],
        }
    )
    assert position_goal_assist_rates(blank) == {}


def _predict(store: DataStore, **params: object) -> pl.DataFrame:
    view = PITData(store).view(GWKey(_SEASON, 3))
    model = InvolvementXg(_xg_spec(**params))
    model.fit(view)
    return model.predict(view, _ASK)


def _row(
    player: int,
    position: str,
    team: int,
    *,
    gw: int = 1,
    minutes: int = 90,
    goals: int = 0,
    assists: int = 0,
    pens_missed: int = 0,
    own_goals: int = 0,
) -> dict[str, object]:
    return {
        "player": player,
        "position": position,
        "team": team,
        "gw": gw,
        "minutes": minutes,
        "goals": goals,
        "assists": assists,
        "pens_missed": pens_missed,
        "own_goals": own_goals,
    }


def _ust(
    player: int,
    team: int,
    *,
    match_id: int = 1,
    xg: float = 0.0,
    xa: float = 0.0,
    minutes: int = 90,
) -> dict[str, object]:
    return {
        "player": player,
        "team": team,
        "match_id": match_id,
        "xg": xg,
        "xa": xa,
        "minutes": minutes,
    }


def _store(
    root: Path,
    *,
    fpl: list[dict[str, object]],
    understat: list[dict[str, object]],
) -> DataStore:
    store = DataStore(root)
    players = list({(row["player"], row["position"], row["team"]) for row in fpl})
    store.write(
        "players",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * len(players),
                    "player_key": [player for player, _, _ in players],
                    "element_id": [player for player, _, _ in players],
                    "team_key": [team for _, _, team in players],
                    "position": [position for _, position, _ in players],
                    "web_name": [str(player) for player, _, _ in players],
                    "available_at": [_START] * len(players),
                    "source": ["fpl_api"] * len(players),
                }
            )
        ),
    )
    store.write(
        "gameweeks",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * 3,
                    "gw": [1, 2, 3],
                    "deadline_time": [_deadline(gw) for gw in (1, 2, 3)],
                    "finished": [True, True, False],
                    "data_checked": [True, True, False],
                    "available_at": [_START] * 3,
                    "source": ["fpl_api"] * 3,
                }
            )
        ),
    )
    when = [_kick(int(row["gw"])) + timedelta(hours=3) for row in fpl]
    store.write(
        "player_fixture_stats",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * len(fpl),
                    "gw": [row["gw"] for row in fpl],
                    "player_key": [row["player"] for row in fpl],
                    "fixture_id": [row["gw"] for row in fpl],
                    "minutes": [row["minutes"] for row in fpl],
                    "total_points": [0] * len(fpl),
                    "goals_scored": [row["goals"] for row in fpl],
                    "assists": [row["assists"] for row in fpl],
                    "clean_sheets": [0] * len(fpl),
                    "goals_conceded": [0] * len(fpl),
                    "saves": [0] * len(fpl),
                    "bonus": [0] * len(fpl),
                    "yellow_cards": [0] * len(fpl),
                    "red_cards": [0] * len(fpl),
                    "own_goals": [row["own_goals"] for row in fpl],
                    "penalties_saved": [0] * len(fpl),
                    "penalties_missed": [row["pens_missed"] for row in fpl],
                    "starts": [1] * len(fpl),
                    "available_at": when,
                    "source": ["fpl_api"] * len(fpl),
                }
            )
        ),
    )
    if understat:
        store.write(
            "understat_player_matches",
            _utc(
                pl.DataFrame(
                    {
                        "season": [_SEASON] * len(understat),
                        "match_id": [row["match_id"] for row in understat],
                        "understat_player_id": [row["player"] for row in understat],
                        "player_key": [row["player"] for row in understat],
                        "team_key": [row["team"] for row in understat],
                        "xg": [row["xg"] for row in understat],
                        "xa": [row["xa"] for row in understat],
                        "shots": [0] * len(understat),
                        "key_passes": [0] * len(understat),
                        "xg_chain": [0.0] * len(understat),
                        "xg_buildup": [0.0] * len(understat),
                        "minutes": [row["minutes"] for row in understat],
                        "available_at": [
                            _kick(1) + timedelta(hours=3) for _ in understat
                        ],
                        "source": ["understat"] * len(understat),
                    }
                )
            ),
        )
    return store


def _kick(gw: int) -> datetime:
    return datetime(2024, 8, 10, 15, tzinfo=UTC) + timedelta(days=7 * (gw - 1))


def _deadline(gw: int) -> datetime:
    return _kick(gw) - timedelta(hours=21)


def _utc(frame: pl.DataFrame) -> pl.DataFrame:
    columns = [
        pl.col(name).cast(pl.Datetime("us", "UTC"))
        for name, dtype in frame.schema.items()
        if isinstance(dtype, pl.Datetime)
    ]
    return frame.with_columns(columns) if columns else frame
