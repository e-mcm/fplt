"""Involvement shares and other-points priors use only the visible history."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

import fplt.models.components.involvement  # noqa: F401 — registers the component
import fplt.models.components.other  # noqa: F401 — registers the component
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.components.involvement.rates import (
    appearance_history,
    penalty_takers,
)
from fplt.models.components.involvement.shares import InvolvementShares
from fplt.models.components.other.priors import OtherPointsPriors
from fplt.models.spec import (
    ModelSpec,
    get_component,
    load_model_specs,
    registered_kinds,
)
from fplt.types import GWKey

_SEASON = "2024-25"
_START = datetime(2024, 8, 1, tzinfo=UTC)
_ASK = pl.DataFrame({"player_key": [1, 2], "fixture_id": [9, 9]})


def _share_spec(**params: object) -> ModelSpec:
    return ModelSpec(
        name="involvement_shares",
        kind="involvement_shares",
        params={"window": 10, "shrinkage": 0, "penalty_bonus": 0.3, **params},
    )


def _prior_spec() -> ModelSpec:
    return ModelSpec(name="other_priors", kind="other_priors", params={})


def test_specs_register_as_components() -> None:
    specs = load_model_specs()
    assert get_component("involvement_shares") is InvolvementShares
    assert get_component("other_priors") is OtherPointsPriors
    assert specs["involvement_shares_v1"].params["window"] == 10
    assert specs["other_priors_v1"].kind == "other_priors"
    assert "involvement_shares" not in registered_kinds()
    assert "other_priors" not in registered_kinds()


def test_the_scorer_takes_the_teams_goal_share(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [
            _row(1, "FWD", 1, goals=2),
            _row(2, "DEF", 1, assists=1),
        ],
    )
    frame = _shares(store)
    by_player = {row["player_key"]: row for row in frame.iter_rows(named=True)}
    assert by_player[1]["goal_weight"] == pytest.approx(1.0)
    assert by_player[2]["goal_weight"] == pytest.approx(0.0)
    assert by_player[1]["assist_weight"] == pytest.approx(0.0)
    assert by_player[2]["assist_weight"] == pytest.approx(1.0)
    assert frame["goal_weight"].sum() == pytest.approx(1.0)


def test_shrinkage_pulls_a_blank_forward_toward_the_position(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [_row(1, "FWD", 1, goals=2), _row(2, "FWD", 1, goals=0)],
    )
    frame = _shares(store, shrinkage=5)
    weights = _weights(frame)
    assert weights[1] == pytest.approx(7 / 12)
    assert weights[2] == pytest.approx(5 / 12)


def test_a_penalty_miss_adds_goals_per_ninety(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [_row(1, "FWD", 1, pens_missed=1), _row(2, "FWD", 1)],
    )
    frame = _shares(store)
    weights = _weights(frame)
    assert weights[1] == pytest.approx(1.0)
    assert weights[2] == pytest.approx(0.0)


def test_penalties_order_one_marks_the_taker(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [_row(1, "FWD", 1), _row(2, "FWD", 1)],
        status=[_status(1, order=1), _status(2, order=2)],
    )
    frame = _shares(store)
    weights = _weights(frame)
    assert weights[1] == pytest.approx(1.0)
    assert weights[2] == pytest.approx(0.0)


def test_penalties_order_gates_off_a_miss(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [_row(1, "FWD", 1, pens_missed=1), _row(2, "FWD", 1, pens_missed=1)],
        status=[_status(1, order=2), _status(2, order=1)],
    )
    frame = _shares(store)
    weights = _weights(frame)
    assert weights[2] == pytest.approx(1.0)
    assert weights[1] == pytest.approx(0.0)


def test_null_penalties_order_falls_back_to_miss(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [_row(1, "FWD", 1, pens_missed=1), _row(2, "FWD", 1)],
        status=[_status(1, order=None), _status(2, order=None)],
    )
    frame = _shares(store)
    weights = _weights(frame)
    assert weights[1] == pytest.approx(1.0)
    assert weights[2] == pytest.approx(0.0)


def test_use_penalties_order_false_ignores_order(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [_row(1, "FWD", 1), _row(2, "FWD", 1, pens_missed=1)],
        status=[_status(1, order=1), _status(2, order=2)],
    )
    with_order = _weights(_shares(store))
    without = _weights(_shares(store, use_penalties_order=False))
    assert with_order[1] == pytest.approx(1.0)
    assert with_order[2] == pytest.approx(0.0)
    assert without[2] == pytest.approx(1.0)
    assert without[1] == pytest.approx(0.0)


def test_rejects_non_bool_use_penalties_order() -> None:
    with pytest.raises(ValueError, match="use_penalties_order"):
        InvolvementShares(_share_spec(use_penalties_order="yes"))


def test_nopens_control_spec_loads() -> None:
    specs = load_model_specs()
    control = specs["involvement_shares_nopens_v1"].params["use_penalties_order"]
    treatment = specs["involvement_shares_v1"].params.get("use_penalties_order", True)
    assert control is False
    assert treatment is True


def test_penalty_takers_respect_pit_cutoff(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [_row(1, "FWD", 1, pens_missed=0), _row(2, "FWD", 1)],
        status=[_status(1, order=2), _status(2, order=2)],
    )
    late = store.read("player_status").filter(pl.col("player_key") == 1)
    late = late.with_columns(
        pl.lit(1).alias("penalties_order"),
        pl.lit(_deadline(3) + timedelta(hours=1)).alias("available_at"),
    )
    store.write("player_status", late)
    view = PITData(store).view(GWKey(_SEASON, 3))
    history = appearance_history(view)
    assert penalty_takers(view, history) == set()


def test_a_team_with_no_rates_splits_equally(tmp_path: Path) -> None:
    store = _store(
        tmp_path, [_row(1, "DEF", 1, minutes=0), _row(2, "DEF", 1, minutes=0)]
    )
    frame = _shares(store)
    assert frame["goal_weight"].to_list() == pytest.approx([0.5, 0.5])
    assert frame["assist_weight"].to_list() == pytest.approx([0.5, 0.5])


def test_no_history_and_an_unknown_player_still_share(tmp_path: Path) -> None:
    store = _store(tmp_path, [_row(1, "DEF", 1, minutes=0)])
    store.path("player_fixture_stats").unlink()
    view = PITData(store).view(GWKey(_SEASON, 3))
    model = InvolvementShares(_share_spec())
    model.fit(view)
    asked = pl.DataFrame({"player_key": [1, 9], "fixture_id": [9, 9]})
    frame = model.predict(view, asked)
    assert frame["goal_weight"].to_list() == pytest.approx([1.0, 1.0])


def test_clubs_do_not_share_a_fixture(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [_row(1, "FWD", 1, goals=2), _row(2, "FWD", 2, goals=0)],
    )
    frame = _shares(store)
    assert frame["goal_weight"].to_list() == pytest.approx([1.0, 1.0])


def test_only_the_latest_window_sets_the_rate(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [
            _row(1, "FWD", 1, gw=1, goals=4),
            _row(1, "FWD", 1, gw=2, goals=0),
            _row(2, "FWD", 1, gw=1, goals=0),
        ],
    )
    frame = _shares(store, window=1)
    weights = _weights(frame)
    assert weights[1] == pytest.approx(0.5)
    assert weights[2] == pytest.approx(0.5)


def test_a_result_after_the_deadline_does_not_change_shares(tmp_path: Path) -> None:
    store = _store(tmp_path, [_row(1, "FWD", 1, goals=1), _row(2, "DEF", 1)])
    first = _shares(store)
    late = store.read("player_fixture_stats").filter(pl.col("player_key") == 2)
    late = late.with_columns(
        pl.lit(5).alias("goals_scored"),
        pl.lit(_deadline(3) + timedelta(hours=1)).alias("available_at"),
        pl.lit(3).alias("gw"),
        pl.lit(3).alias("fixture_id"),
    )
    store.write("player_fixture_stats", late)
    assert _shares(store).equals(first)


def test_shares_reject_a_bad_request(tmp_path: Path) -> None:
    store = _store(tmp_path, [_row(1, "FWD", 1, goals=1)])
    view = PITData(store).view(GWKey(_SEASON, 3))
    model = InvolvementShares(_share_spec())
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(view, _ASK)
    model.fit(view)
    with pytest.raises(ValueError, match="fixture_id"):
        model.predict(view, pl.DataFrame({"player_key": [1]}))
    assert model.predict(view, _ASK.clear()).is_empty()


def test_share_params_are_checked() -> None:
    with pytest.raises(ValueError, match="window"):
        InvolvementShares(_share_spec(window=0))
    with pytest.raises(ValueError, match="shrinkage"):
        InvolvementShares(_share_spec(shrinkage=-1))
    with pytest.raises(ValueError, match="penalty_bonus"):
        InvolvementShares(_share_spec(penalty_bonus=-0.1))


def test_priors_follow_position_and_minutes_bucket(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [
            _row(1, "GK", 1, minutes=90, saves=6, bonus=1),
            _row(2, "DEF", 1, minutes=90, yellow=1),
            _row(2, "DEF", 1, gw=2, minutes=20, bonus=1),
        ],
    )
    frame = _priors(store, pl.DataFrame({"player_key": [1, 2], "fixture_id": [9, 9]}))
    gk = _cell(frame, 1, "60+")
    defender = _cell(frame, 2, "60+")
    cameo = _cell(frame, 2, "1-59")
    bench = _cell(frame, 2, "0")
    assert gk["saves"] == pytest.approx(2.0)
    assert gk["bonus"] == pytest.approx(1.0)
    assert gk["defensive_contribution"] == pytest.approx(0.0)
    assert defender["cards"] == pytest.approx(-1.0)
    assert defender["saves"] == pytest.approx(0.0)
    assert cameo["bonus"] == pytest.approx(1.0)
    assert bench["bonus"] == pytest.approx(0.0)
    assert bench["cards"] == pytest.approx(0.0)


def test_two_keepers_average_their_save_points(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [
            _row(1, "GK", 1, minutes=90, saves=6),
            _row(2, "GK", 1, gw=2, minutes=90, saves=0),
        ],
    )
    frame = _priors(store, pl.DataFrame({"player_key": [1], "fixture_id": [9]}))
    assert _cell(frame, 1, "60+")["saves"] == pytest.approx(1.0)


def test_an_unknown_position_gets_zeros(tmp_path: Path) -> None:
    store = _store(tmp_path, [_row(1, "UNK", 1, minutes=90, bonus=3, saves=6)])
    frame = _priors(store, pl.DataFrame({"player_key": [1], "fixture_id": [9]}))
    assert frame.select("bonus", "saves", "cards", "other").to_dicts() == [
        {"bonus": 0.0, "saves": 0.0, "cards": 0.0, "other": 0.0},
        {"bonus": 0.0, "saves": 0.0, "cards": 0.0, "other": 0.0},
        {"bonus": 0.0, "saves": 0.0, "cards": 0.0, "other": 0.0},
    ]


def test_a_result_after_the_deadline_does_not_change_priors(tmp_path: Path) -> None:
    store = _store(tmp_path, [_row(1, "GK", 1, minutes=90, saves=3)])
    ask = pl.DataFrame({"player_key": [1], "fixture_id": [9]})
    first = _priors(store, ask)
    late = store.read("player_fixture_stats").with_columns(
        pl.lit(12).alias("saves"),
        pl.lit(_deadline(3) + timedelta(hours=1)).alias("available_at"),
        pl.lit(3).alias("gw"),
        pl.lit(3).alias("fixture_id"),
    )
    store.write("player_fixture_stats", late)
    assert _priors(store, ask).equals(first)


def test_own_goals_and_penalties_land_in_other(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [_row(1, "DEF", 1, minutes=90, own_goals=1, pens_saved=1)],
    )
    frame = _priors(store, pl.DataFrame({"player_key": [1], "fixture_id": [9]}))
    assert _cell(frame, 1, "60+")["other"] == pytest.approx(3.0)


def test_priors_reject_a_bad_request(tmp_path: Path) -> None:
    store = _store(tmp_path, [_row(1, "GK", 1, minutes=90, saves=3)])
    view = PITData(store).view(GWKey(_SEASON, 3))
    model = OtherPointsPriors(_prior_spec())
    ask = pl.DataFrame({"player_key": [1], "fixture_id": [9]})
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(view, ask)
    model.fit(view)
    with pytest.raises(ValueError, match="fixture_id"):
        model.predict(view, pl.DataFrame({"player_key": [1]}))
    assert model.predict(view, ask.clear()).is_empty()


def test_a_season_without_rules_raises(tmp_path: Path) -> None:
    store = _store(tmp_path, [_row(1, "GK", 1, minutes=90, saves=3)], season="1999-00")
    view = PITData(store).view(GWKey("1999-00", 3))
    with pytest.raises(FileNotFoundError, match="1999-00"):
        OtherPointsPriors(_prior_spec()).fit(view)


def _weights(frame: pl.DataFrame) -> dict[int, float]:
    return dict(
        zip(
            frame["player_key"].to_list(),
            frame["goal_weight"].to_list(),
            strict=True,
        )
    )


def _shares(store: DataStore, **params: object) -> pl.DataFrame:
    view = PITData(store).view(GWKey(_SEASON, 3))
    model = InvolvementShares(_share_spec(**params))
    model.fit(view)
    return model.predict(view, _ASK)


def _priors(store: DataStore, asked: pl.DataFrame) -> pl.DataFrame:
    view = PITData(store).view(GWKey(store.read("gameweeks")["season"][0], 3))
    model = OtherPointsPriors(_prior_spec())
    model.fit(view)
    return model.predict(view, asked)


def _cell(frame: pl.DataFrame, player: int, bucket: str) -> dict[str, object]:
    rows = frame.filter(
        (pl.col("player_key") == player) & (pl.col("minutes_bucket") == bucket)
    )
    return rows.row(0, named=True)


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
    pens_saved: int = 0,
    own_goals: int = 0,
    saves: int = 0,
    yellow: int = 0,
    bonus: int = 0,
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
        "pens_saved": pens_saved,
        "own_goals": own_goals,
        "saves": saves,
        "yellow": yellow,
        "bonus": bonus,
    }


def _status(player: int, *, order: int | None) -> dict[str, object]:
    return {"player": player, "order": order}


def _store(
    root: Path,
    rows: list[dict[str, object]],
    *,
    season: str = _SEASON,
    status: list[dict[str, object]] | None = None,
) -> DataStore:
    store = DataStore(root)
    players = list({(row["player"], row["position"], row["team"]) for row in rows})
    store.write(
        "players",
        _utc(
            pl.DataFrame(
                {
                    "season": [season] * len(players),
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
                    "season": [season] * 3,
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
    when = [_kick(int(row["gw"])) + timedelta(hours=3) for row in rows]
    store.write(
        "player_fixture_stats",
        _utc(
            pl.DataFrame(
                {
                    "season": [season] * len(rows),
                    "gw": [row["gw"] for row in rows],
                    "player_key": [row["player"] for row in rows],
                    "fixture_id": [row["gw"] for row in rows],
                    "minutes": [row["minutes"] for row in rows],
                    "total_points": [0] * len(rows),
                    "goals_scored": [row["goals"] for row in rows],
                    "assists": [row["assists"] for row in rows],
                    "clean_sheets": [0] * len(rows),
                    "goals_conceded": [0] * len(rows),
                    "saves": [row["saves"] for row in rows],
                    "bonus": [row["bonus"] for row in rows],
                    "yellow_cards": [row["yellow"] for row in rows],
                    "red_cards": [0] * len(rows),
                    "own_goals": [row["own_goals"] for row in rows],
                    "penalties_saved": [row["pens_saved"] for row in rows],
                    "penalties_missed": [row["pens_missed"] for row in rows],
                    "starts": [1] * len(rows),
                    "available_at": when,
                    "source": ["fpl_api"] * len(rows),
                }
            )
        ),
    )
    if status is not None:
        store.write(
            "player_status",
            _utc(
                pl.DataFrame(
                    {
                        "season": [season] * len(status),
                        "player_key": [row["player"] for row in status],
                        "status": ["a"] * len(status),
                        "news": [""] * len(status),
                        "price": [100] * len(status),
                        "ep_next": [4.0] * len(status),
                        "chance_of_playing": [100.0] * len(status),
                        "penalties_order": [row["order"] for row in status],
                        "available_at": [_START] * len(status),
                        "source": ["fpl_api"] * len(status),
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
