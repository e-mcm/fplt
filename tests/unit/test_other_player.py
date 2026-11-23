"""Player-level other-points priors shrink toward position × bucket."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

import fplt.models.components.other  # noqa: F401 — registers the component
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.components.other.player import OtherPlayerPriors
from fplt.models.spec import ModelSpec, get_component, load_model_specs
from fplt.types import GWKey

_SEASON = "2024-25"
_START = datetime(2024, 8, 1, tzinfo=UTC)


def _spec(**params: object) -> ModelSpec:
    return ModelSpec(
        name="other_player",
        kind="other_player",
        params={"window": 10, "shrinkage": 0, "bps_bonus_weight": 0.0, **params},
    )


def test_spec_registers_as_component() -> None:
    specs = load_model_specs()
    assert get_component("other_player") is OtherPlayerPriors
    assert specs["other_player_v1"].params["window"] == 10
    assert specs["other_player_v1"].params["bps_bonus_weight"] == 0.5


def test_expected_total_sums_bucket_channels(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [_row(1, "GK", 1, minutes=90, saves=6, bonus=1)],
    )
    view = PITData(store).view(GWKey(_SEASON, 3))
    model = OtherPlayerPriors(_spec())
    model.fit(view)
    frame = model.predict(view, pl.DataFrame({"player_key": [1], "fixture_id": [9]}))
    row = _cell(frame, 1, "60+")
    assert model.expected_total(1, "GK", "60+") == pytest.approx(
        float(row["bonus"])
        + float(row["saves"])
        + float(row["cards"])
        + float(row["defensive_contribution"])
        + float(row["other"])
    )


def test_a_high_save_keeper_keeps_his_own_rate(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [
            _row(1, "GK", 1, minutes=90, saves=6),
            _row(2, "GK", 1, gw=2, minutes=90, saves=0),
        ],
    )
    frame = _predict(store, pl.DataFrame({"player_key": [1, 2], "fixture_id": [9, 9]}))
    assert _cell(frame, 1, "60+")["saves"] == pytest.approx(2.0)
    assert _cell(frame, 2, "60+")["saves"] == pytest.approx(0.0)


def test_shrinkage_pulls_a_blank_mid_toward_the_position(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [
            _row(1, "MID", 1, minutes=90, bonus=2),
            _row(2, "MID", 1, gw=2, minutes=90, bonus=0),
        ],
    )
    frame = _predict(
        store,
        pl.DataFrame({"player_key": [2], "fixture_id": [9]}),
        shrinkage=1,
    )
    # Position mean bonus is 1; one blank sample → (0 + 1*1) / (1+1) = 0.5.
    assert _cell(frame, 2, "60+")["bonus"] == pytest.approx(0.5)


def test_bps_proxy_lifts_bonus_when_weight_is_on(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [
            _row(1, "MID", 1, minutes=90, bonus=3, bps=40),
            _row(2, "MID", 1, gw=2, minutes=90, bonus=0, bps=40),
        ],
    )
    without = _predict(
        store,
        pl.DataFrame({"player_key": [2], "fixture_id": [9]}),
        shrinkage=1,
        bps_bonus_weight=0.0,
    )
    with_proxy = _predict(
        store,
        pl.DataFrame({"player_key": [2], "fixture_id": [9]}),
        shrinkage=1,
        bps_bonus_weight=1.0,
    )
    assert _cell(with_proxy, 2, "60+")["bonus"] > _cell(without, 2, "60+")["bonus"]


def test_defensive_contribution_uses_staged_count(tmp_path: Path) -> None:
    # Action columns alone would miss the threshold; the staged aggregate is 10.
    store = _store(
        tmp_path,
        [
            _row(
                1,
                "DEF",
                1,
                minutes=90,
                clearances=0,
                tackles=0,
                recoveries=0,
                defensive_contribution=10,
            )
        ],
        season="2025-26",
    )
    frame = _predict(
        store,
        pl.DataFrame({"player_key": [1], "fixture_id": [9]}),
        season="2025-26",
    )
    assert _cell(frame, 1, "60+")["defensive_contribution"] == pytest.approx(2.0)


def test_defensive_contribution_below_threshold_is_zero(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [_row(1, "DEF", 1, minutes=90, defensive_contribution=9)],
        season="2025-26",
    )
    frame = _predict(
        store,
        pl.DataFrame({"player_key": [1], "fixture_id": [9]}),
        season="2025-26",
    )
    assert _cell(frame, 1, "60+")["defensive_contribution"] == pytest.approx(0.0)


def test_older_seasons_keep_defensive_at_zero(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [
            _row(
                1,
                "DEF",
                1,
                minutes=90,
                clearances=20,
                tackles=20,
                recoveries=20,
                defensive_contribution=20,
            )
        ],
    )
    frame = _predict(store, pl.DataFrame({"player_key": [1], "fixture_id": [9]}))
    assert _cell(frame, 1, "60+")["defensive_contribution"] == pytest.approx(0.0)


def test_use_defensive_contribution_false_zeros_channel(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [_row(1, "DEF", 1, minutes=90, defensive_contribution=10)],
        season="2025-26",
    )
    frame = _predict(
        store,
        pl.DataFrame({"player_key": [1], "fixture_id": [9]}),
        season="2025-26",
        use_defensive_contribution=False,
    )
    assert _cell(frame, 1, "60+")["defensive_contribution"] == pytest.approx(0.0)


def test_rejects_non_bool_use_defensive_contribution() -> None:
    with pytest.raises(ValueError, match="use_defensive_contribution"):
        OtherPlayerPriors(_spec(use_defensive_contribution="yes"))


def test_rejects_bad_defcon_mode() -> None:
    with pytest.raises(ValueError, match="defcon_mode"):
        OtherPlayerPriors(_spec(defcon_mode="gamma"))


def test_an_unknown_player_gets_zeros(tmp_path: Path) -> None:
    store = _store(tmp_path, [_row(1, "FWD", 1, minutes=90, bonus=3)])
    frame = _predict(store, pl.DataFrame({"player_key": [99], "fixture_id": [9]}))
    # Player 99 is missing from the roster → empty position → zeros.
    assert _cell(frame, 99, "60+")["bonus"] == pytest.approx(0.0)


def test_no_history_still_predicts_zeros(tmp_path: Path) -> None:
    store = _store(tmp_path, [_row(1, "MID", 1, minutes=90, bonus=1)])
    store.path("player_fixture_stats").unlink()
    frame = _predict(store, pl.DataFrame({"player_key": [1], "fixture_id": [9]}))
    assert frame.select("bonus", "saves", "cards", "other").to_dicts() == [
        {"bonus": 0.0, "saves": 0.0, "cards": 0.0, "other": 0.0},
        {"bonus": 0.0, "saves": 0.0, "cards": 0.0, "other": 0.0},
        {"bonus": 0.0, "saves": 0.0, "cards": 0.0, "other": 0.0},
    ]


def test_same_position_teammate_gets_the_prior(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [
            _row(1, "FWD", 1, minutes=90, bonus=2),
            _row(2, "FWD", 1, minutes=0),
        ],
    )
    # Player 2 has a 0-minute row only, so 60+ falls back to the FWD prior.
    frame = _predict(store, pl.DataFrame({"player_key": [2], "fixture_id": [9]}))
    assert _cell(frame, 2, "60+")["bonus"] == pytest.approx(2.0)


def test_rejects_bad_params_and_requests(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="window"):
        OtherPlayerPriors(_spec(window=0))
    with pytest.raises(ValueError, match="shrinkage"):
        OtherPlayerPriors(_spec(shrinkage=-1))
    with pytest.raises(ValueError, match="bps_bonus_weight"):
        OtherPlayerPriors(_spec(bps_bonus_weight=1.5))
    store = _store(tmp_path, [_row(1, "GK", 1, minutes=90, saves=3)])
    view = PITData(store).view(GWKey(_SEASON, 3))
    model = OtherPlayerPriors(_spec())
    ask = pl.DataFrame({"player_key": [1], "fixture_id": [9]})
    with pytest.raises(RuntimeError, match="fit"):
        model.predict(view, ask)
    model.fit(view)
    with pytest.raises(ValueError, match="fixture_id"):
        model.predict(view, pl.DataFrame({"player_key": [1]}))
    assert model.predict(view, ask.clear()).is_empty()


def test_a_result_after_the_deadline_does_not_change_priors(tmp_path: Path) -> None:
    store = _store(tmp_path, [_row(1, "GK", 1, minutes=90, saves=3)])
    ask = pl.DataFrame({"player_key": [1], "fixture_id": [9]})
    first = _predict(store, ask)
    late = store.read("player_fixture_stats").with_columns(
        pl.lit(12).alias("saves"),
        pl.lit(_deadline(3) + timedelta(hours=1)).alias("available_at"),
        pl.lit(3).alias("gw"),
        pl.lit(3).alias("fixture_id"),
    )
    store.write("player_fixture_stats", late)
    assert _predict(store, ask).equals(first)


def _predict(
    store: DataStore,
    asked: pl.DataFrame,
    *,
    season: str | None = None,
    **params: object,
) -> pl.DataFrame:
    label = season or store.read("gameweeks")["season"][0]
    view = PITData(store).view(GWKey(str(label), 3))
    model = OtherPlayerPriors(_spec(**params))
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
    saves: int = 0,
    yellow: int = 0,
    bonus: int = 0,
    bps: int = 0,
    clearances: int = 0,
    tackles: int = 0,
    recoveries: int = 0,
    defensive_contribution: int = 0,
) -> dict[str, object]:
    return {
        "player": player,
        "position": position,
        "team": team,
        "gw": gw,
        "minutes": minutes,
        "saves": saves,
        "yellow": yellow,
        "bonus": bonus,
        "bps": bps,
        "clearances": clearances,
        "tackles": tackles,
        "recoveries": recoveries,
        "defensive_contribution": defensive_contribution,
    }


def _store(
    root: Path, rows: list[dict[str, object]], *, season: str = _SEASON
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
                    "goals_scored": [0] * len(rows),
                    "assists": [0] * len(rows),
                    "clean_sheets": [0] * len(rows),
                    "goals_conceded": [0] * len(rows),
                    "saves": [row["saves"] for row in rows],
                    "bonus": [row["bonus"] for row in rows],
                    "yellow_cards": [row["yellow"] for row in rows],
                    "red_cards": [0] * len(rows),
                    "own_goals": [0] * len(rows),
                    "penalties_saved": [0] * len(rows),
                    "penalties_missed": [0] * len(rows),
                    "starts": [1] * len(rows),
                    "bps": [row["bps"] for row in rows],
                    "clearances_blocks_interceptions": [
                        row["clearances"] for row in rows
                    ],
                    "tackles": [row["tackles"] for row in rows],
                    "recoveries": [row["recoveries"] for row in rows],
                    "defensive_contribution": [
                        row["defensive_contribution"] for row in rows
                    ],
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
    columns = [
        pl.col(name).cast(pl.Datetime("us", "UTC"))
        for name, dtype in frame.schema.items()
        if isinstance(dtype, pl.Datetime)
    ]
    return frame.with_columns(columns) if columns else frame
