"""Unit tests for Poisson defensive-contribution rate helpers."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import fplt.models.components.other  # noqa: F401 — registers the component
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.components.other.defcon_rate import (
    expected_defcon_points,
    p_award,
    parse_bucket_minutes,
    parse_defcon_mode,
)
from fplt.models.components.other.player import OtherPlayerPriors
from fplt.models.spec import ModelSpec, load_model_specs
from fplt.types import GWKey

_SEASON = "2025-26"
_START = datetime(2024, 8, 1, tzinfo=UTC)


def test_defcon_rate_specs_load() -> None:
    specs = load_model_specs()
    other = specs["other_player_defcon_rate_v1"]
    assert other.params["defcon_mode"] == "rate"
    assert "defcon_bucket_minutes" not in other.params
    layered = specs["dc_layered_defcon_rate_v1"]
    assert layered.components["other"].name == "other_player_defcon_rate_v1"
    assert layered.components["other"].params["defcon_mode"] == "rate"
    assert "defcon_bucket_minutes" not in layered.components["other"].params
    # Released v1 stays on the mean wire.
    assert "defcon_mode" not in specs["other_player_v1"].params


def test_defcon_rate_ab_is_one_param_family() -> None:
    from fplt.backtest.promote_eval import _check_one_lever, load_eval_spec

    specs = load_model_specs()
    spec = load_eval_spec("defcon_rate_ab")
    assert spec.control == "dc_layered_defcon_mean_v1"
    assert spec.treatment == "dc_layered_v2"
    assert specs["dc_layered_v2"].components["other"].params["defcon_mode"] == "rate"
    assert (
        specs["dc_layered_defcon_mean_v1"].components["other"].params["defcon_mode"]
        == "mean"
    )
    assert (
        _check_one_lever(
            spec, specs["dc_layered_defcon_mean_v1"], specs["dc_layered_v2"]
        )
        == []
    )


def test_parse_defcon_mode_rejects_unknown() -> None:
    with pytest.raises(ValueError, match="defcon_mode"):
        parse_defcon_mode("poisson")


def test_parse_bucket_minutes_requires_all_buckets() -> None:
    with pytest.raises(ValueError, match="missing bucket"):
        parse_bucket_minutes({"0": 0, "60+": 75})


def test_rate_mode_uses_poisson_p_award(tmp_path: Path) -> None:
    # Ten actions in 90 minutes → rate 10 per 90.
    # At 75 minutes λ≈8.33; P(X≥10)*2 is well below the realised 2.0 mean.
    store = _store(
        tmp_path,
        [_row(1, "DEF", 1, minutes=90, defensive_contribution=10)],
    )
    mean = _predict(store, defcon_mode="mean")
    rate = _predict(store, defcon_mode="rate")
    assert _cell(mean, 1, "60+")["defensive_contribution"] == pytest.approx(2.0)
    expected = expected_defcon_points(
        rate_per_90=10.0, minutes=75.0, threshold=10, points=2
    )
    assert _cell(rate, 1, "60+")["defensive_contribution"] == pytest.approx(expected)
    assert expected < 2.0
    assert _cell(rate, 1, "0")["defensive_contribution"] == pytest.approx(0.0)


def test_rate_mode_respects_use_defensive_false(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [_row(1, "DEF", 1, minutes=90, defensive_contribution=20)],
    )
    frame = _predict(store, defcon_mode="rate", use_defensive_contribution=False)
    assert _cell(frame, 1, "60+")["defensive_contribution"] == pytest.approx(0.0)


def test_rate_mode_zeros_pre_2025(tmp_path: Path) -> None:
    store = _store(
        tmp_path,
        [_row(1, "DEF", 1, minutes=90, defensive_contribution=20)],
        season="2024-25",
    )
    frame = _predict(store, defcon_mode="rate", season="2024-25")
    assert _cell(frame, 1, "60+")["defensive_contribution"] == pytest.approx(0.0)


def test_higher_rate_raises_p_award() -> None:
    low = p_award(rate_per_90=5.0, minutes=90.0, threshold=10)
    high = p_award(rate_per_90=15.0, minutes=90.0, threshold=10)
    assert 0.0 <= low < high <= 1.0


def test_more_minutes_raises_p_award() -> None:
    short = p_award(rate_per_90=12.0, minutes=45.0, threshold=10)
    long = p_award(rate_per_90=12.0, minutes=90.0, threshold=10)
    assert 0.0 <= short < long <= 1.0


@given(
    rate=st.floats(min_value=0.0, max_value=40.0, allow_nan=False),
    minutes=st.floats(min_value=0.0, max_value=90.0, allow_nan=False),
    threshold=st.integers(min_value=1, max_value=20),
    points=st.integers(min_value=0, max_value=5),
)
@settings(max_examples=40)
def test_expected_defcon_bounds(
    rate: float, minutes: float, threshold: int, points: int
) -> None:
    value = expected_defcon_points(
        rate_per_90=rate, minutes=minutes, threshold=threshold, points=points
    )
    assert 0.0 <= value <= float(points)


def _predict(
    store: DataStore,
    *,
    season: str | None = None,
    **params: object,
) -> pl.DataFrame:
    label = season or store.read("gameweeks")["season"][0]
    view = PITData(store).view(GWKey(str(label), 3))
    model = OtherPlayerPriors(
        ModelSpec(
            name="other_player",
            kind="other_player",
            params={
                "window": 10,
                "shrinkage": 0,
                "bps_bonus_weight": 0.0,
                **params,
            },
        )
    )
    model.fit(view)
    return model.predict(view, pl.DataFrame({"player_key": [1], "fixture_id": [9]}))


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
    defensive_contribution: int = 0,
) -> dict[str, object]:
    return {
        "player": player,
        "position": position,
        "team": team,
        "gw": gw,
        "minutes": minutes,
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
                    "saves": [0] * len(rows),
                    "bonus": [0] * len(rows),
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
