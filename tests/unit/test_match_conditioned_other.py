"""Match-conditioned other offsets lift with goals / CS / scored."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl
import pytest

import fplt.models.components.other  # noqa: F401 — registers components
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.components.other.match_conditioned import OtherMatchConditioned
from fplt.models.simulation import simulate_match
from fplt.models.spec import ModelSpec, get_component, load_model_specs
from fplt.scoring.rules import Rules
from fplt.types import GWKey

_SEASON = "2024-25"
_START = datetime(2024, 8, 1, tzinfo=UTC)
_RULES = Rules.for_season(_SEASON)


def _spec(**params: object) -> ModelSpec:
    return ModelSpec(
        name="other_match_conditioned",
        kind="other_match_conditioned",
        params={
            "window": 10,
            "shrinkage": 0,
            "bps_bonus_weight": 0.0,
            "match_shrinkage": 0.0,
            **params,
        },
    )


def test_spec_registers_and_fork_extends_v3() -> None:
    specs = load_model_specs()
    assert get_component("other_match_conditioned") is OtherMatchConditioned
    assert specs["other_match_conditioned_v1"].params["defcon_mode"] == "rate"
    assert specs["other_match_conditioned_v1"].params["match_shrinkage"] == 20
    layered = specs["dc_layered_match_other_v1"]
    assert layered.components["other"].kind == "other_match_conditioned"
    assert layered.components["other"].name == "other_match_conditioned_v1"
    assert specs["dc_layered_v3"].components["other"].kind == "other_player"


def test_match_delta_broadcasts_across_sims(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [
            _row(1, "MID", 3, minutes=90, bonus=3, goals=1, team_goals=2, cs=0),
            _row(1, "MID", 3, gw=2, minutes=90, bonus=0, goals=0, team_goals=0, cs=0),
        ],
    )
    model = _fit(store)
    delta = model.match_delta(
        positions=np.array(["MID", "FWD"]),
        minutes=np.full((2, 5), 90.0),
        team_goals=np.zeros((2, 5), dtype=np.int64),
        clean_sheet=np.zeros((2, 5), dtype=bool),
        player_goals=np.zeros((2, 5), dtype=np.int64),
    )
    assert delta.shape == (2, 5)


def test_deltas_zero_when_minutes_are_zero(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [
            _row(1, "MID", 3, minutes=90, bonus=3, goals=1, team_goals=2, cs=0),
            _row(1, "MID", 3, gw=2, minutes=90, bonus=0, goals=0, team_goals=0, cs=0),
        ],
    )
    model = _fit(store)
    delta = model.match_delta(
        positions=np.array(["MID"]),
        minutes=np.array([[0.0, 90.0]]),
        team_goals=np.array([[2, 2]]),
        clean_sheet=np.array([[False, False]]),
        player_goals=np.array([[1, 1]]),
    )
    assert delta[0, 0] == pytest.approx(0.0)
    assert delta[0, 1] != pytest.approx(0.0)


def test_scored_and_cs_lift_relative_to_blank(tmp_path: Path) -> None:
    # Same player across contexts so the minutes-bucket baseline is the mean
    # and residuals carry the match structure (not absorbed into the prior).
    store = _store(
        tmp_path,
        [
            _row(1, "MID", 3, minutes=90, bonus=3, goals=1, team_goals=2, cs=0),
            _row(1, "MID", 3, gw=2, minutes=90, bonus=2, goals=0, team_goals=0, cs=1),
            _row(1, "MID", 3, gw=3, minutes=90, bonus=0, goals=0, team_goals=0, cs=0),
        ],
    )
    model = _fit(store, as_of_gw=4)
    blank = model.match_delta(
        positions=np.array(["MID"]),
        minutes=np.array([[90.0]]),
        team_goals=np.array([[0]]),
        clean_sheet=np.array([[False]]),
        player_goals=np.array([[0]]),
    )[0, 0]
    scored = model.match_delta(
        positions=np.array(["MID"]),
        minutes=np.array([[90.0]]),
        team_goals=np.array([[2]]),
        clean_sheet=np.array([[False]]),
        player_goals=np.array([[1]]),
    )[0, 0]
    clean = model.match_delta(
        positions=np.array(["MID"]),
        minutes=np.array([[90.0]]),
        team_goals=np.array([[0]]),
        clean_sheet=np.array([[True]]),
        player_goals=np.array([[0]]),
    )[0, 0]
    assert scored > blank
    assert clean > blank


def test_predict_matches_baseline_schema(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [_row(1, "GK", 3, minutes=90, saves=3, bonus=0, goals=0, team_goals=1, cs=0)],
    )
    model = _fit(store)
    frame = model.predict(
        PITData(store).view(GWKey(_SEASON, 3)),
        pl.DataFrame({"player_key": [1], "fixture_id": [9]}),
    )
    assert set(frame.columns) >= {
        "player_key",
        "fixture_id",
        "minutes_bucket",
        "bonus",
        "saves",
        "cards",
        "defensive_contribution",
        "other",
    }
    assert frame.filter(pl.col("minutes_bucket") == "60+").height == 1


def test_simulate_match_applies_offsets() -> None:
    class _Lift:
        def match_delta(self, **_: object) -> np.ndarray:
            return np.array([[5.0, 5.0]])

    players = pl.DataFrame(
        {
            "player_key": [1],
            "side": ["home"],
            "position": ["MID"],
            "goal_weight": [0.0],
            "assist_weight": [0.0],
        }
    )
    plain = simulate_match(
        np.array([[0, 0], [0, 0]]),
        players,
        np.full((1, 2), 90.0),
        np.array([[0.0, 0.0, 1.0]]),
        _RULES,
        np.random.default_rng(1),
        own_goal_weight=0.0,
        p_assist=0.0,
        sample_other=False,
    )
    lifted = simulate_match(
        np.array([[0, 0], [0, 0]]),
        players,
        np.full((1, 2), 90.0),
        np.array([[0.0, 0.0, 1.0]]),
        _RULES,
        np.random.default_rng(1),
        own_goal_weight=0.0,
        p_assist=0.0,
        sample_other=False,
        match_other=_Lift(),
    )
    assert lifted.points[0].tolist() == pytest.approx((plain.points[0] + 5.0).tolist())


def test_rejects_bad_match_shrinkage() -> None:
    with pytest.raises(ValueError, match="match_shrinkage"):
        OtherMatchConditioned(_spec(match_shrinkage=-1))


def _fit(store: DataStore, *, as_of_gw: int = 3) -> OtherMatchConditioned:
    view = PITData(store).view(GWKey(_SEASON, as_of_gw))
    model = OtherMatchConditioned(_spec())
    model.fit(view)
    return model


def _row(
    player: int,
    position: str,
    team: int,
    *,
    gw: int = 1,
    minutes: int = 90,
    saves: int = 0,
    bonus: int = 0,
    goals: int = 0,
    team_goals: int = 0,
    cs: int = 0,
) -> dict[str, object]:
    return {
        "player": player,
        "position": position,
        "team": team,
        "gw": gw,
        "minutes": minutes,
        "saves": saves,
        "bonus": bonus,
        "goals": goals,
        "team_goals": team_goals,
        "cs": cs,
    }


def _store(root: Path, rows: list[dict[str, object]]) -> DataStore:
    store = DataStore(root)
    players = list({(row["player"], row["position"], row["team"]) for row in rows})
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
    max_gw = max(int(row["gw"]) for row in rows)
    # Always keep deadlines through GW4 so default as_of=3 is valid.
    gws = list(range(1, max(max_gw + 2, 5)))
    store.write(
        "gameweeks",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * len(gws),
                    "gw": gws,
                    "deadline_time": [_deadline(gw) for gw in gws],
                    "finished": [gw <= max_gw for gw in gws],
                    "data_checked": [gw <= max_gw for gw in gws],
                    "available_at": [_START] * len(gws),
                    "source": ["fpl_api"] * len(gws),
                }
            )
        ),
    )
    fixture_ids = sorted({int(row["gw"]) for row in rows})
    # Home team 3 vs away team 7; score from the first row that claims each side.
    home_scores: dict[int, int] = {}
    away_scores: dict[int, int] = {}
    for row in rows:
        fixture = int(row["gw"])
        if int(row["team"]) == 3:
            home_scores[fixture] = int(row["team_goals"])
            away_scores.setdefault(fixture, 0 if int(row["cs"]) else 1)
        else:
            away_scores[fixture] = int(row["team_goals"])
            home_scores.setdefault(fixture, 0 if int(row["cs"]) else 1)
        if int(row["cs"]) and int(row["team"]) == 3:
            away_scores[fixture] = 0
        if int(row["cs"]) and int(row["team"]) == 7:
            home_scores[fixture] = 0
    store.write(
        "fixtures",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * len(fixture_ids),
                    "fixture_id": fixture_ids,
                    "gw": fixture_ids,
                    "kickoff_time": [_kick(fid) for fid in fixture_ids],
                    "team_h_key": [3] * len(fixture_ids),
                    "team_a_key": [7] * len(fixture_ids),
                    "team_h_score": [home_scores[fid] for fid in fixture_ids],
                    "team_a_score": [away_scores[fid] for fid in fixture_ids],
                    "finished": [True] * len(fixture_ids),
                    "available_at": [
                        _kick(fid) + timedelta(hours=3) for fid in fixture_ids
                    ],
                    "source": ["fpl_api"] * len(fixture_ids),
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
                    "season": [_SEASON] * len(rows),
                    "gw": [row["gw"] for row in rows],
                    "player_key": [row["player"] for row in rows],
                    "fixture_id": [row["gw"] for row in rows],
                    "minutes": [row["minutes"] for row in rows],
                    "total_points": [0] * len(rows),
                    "goals_scored": [row["goals"] for row in rows],
                    "assists": [0] * len(rows),
                    "clean_sheets": [row["cs"] for row in rows],
                    "goals_conceded": [0] * len(rows),
                    "saves": [row["saves"] for row in rows],
                    "bonus": [row["bonus"] for row in rows],
                    "yellow_cards": [0] * len(rows),
                    "red_cards": [0] * len(rows),
                    "own_goals": [0] * len(rows),
                    "penalties_saved": [0] * len(rows),
                    "penalties_missed": [0] * len(rows),
                    "starts": [1] * len(rows),
                    "bps": [0] * len(rows),
                    "clearances_blocks_interceptions": [0] * len(rows),
                    "tackles": [0] * len(rows),
                    "recoveries": [0] * len(rows),
                    "defensive_contribution": [0] * len(rows),
                    "available_at": when,
                    "source": ["fpl_api"] * len(rows),
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
    exprs = [
        pl.col(name).cast(pl.Datetime("us", "UTC"))
        for name, dtype in frame.schema.items()
        if isinstance(dtype, pl.Datetime)
    ]
    return frame.with_columns(exprs) if exprs else frame
