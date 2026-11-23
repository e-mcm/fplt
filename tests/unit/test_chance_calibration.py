"""P1/P2 chance-flag calibration from snapshot player_status."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

import fplt.models.components.minutes  # noqa: F401 — registers the component
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.components.minutes.calibrate import (
    ChanceCalibration,
    calibrated_long_factor,
    calibrated_play_gate,
    fit_chance_calibration,
    snap_chance_bucket,
)
from fplt.models.components.minutes.common import chance
from fplt.models.components.minutes.empirical_v2 import EmpiricalMinutesV2
from fplt.models.distributions import MINUTE_BINS
from fplt.models.spec import ModelSpec, load_model_specs
from fplt.types import GWKey

_SEASON = "2024-25"
_START = datetime(2024, 8, 1, tzinfo=UTC)
_ZERO = {
    "goals_scored": 0,
    "assists": 0,
    "clean_sheets": 0,
    "goals_conceded": 0,
    "saves": 0,
    "bonus": 0,
    "yellow_cards": 0,
    "red_cards": 0,
    "own_goals": 0,
    "penalties_saved": 0,
    "penalties_missed": 0,
    "total_points": 0,
}


def test_snap_chance_bucket_nearest_flag() -> None:
    assert snap_chance_bucket(None) is None
    assert snap_chance_bucket(0.0) == 0
    assert snap_chance_bucket(25.0) == 25
    assert snap_chance_bucket(70.0) == 75
    assert snap_chance_bucket(100.0) == 100


def test_calibrated_play_gate_falls_back_to_face() -> None:
    empty = ChanceCalibration.identity()
    assert calibrated_play_gate(75.0, 0.75, empty) == 0.75
    mapped = ChanceCalibration(play_gate={75: 0.4}, long_factor={})
    assert calibrated_play_gate(75.0, 0.75, mapped) == 0.4


def test_calibrated_long_factor_full_availability_is_one() -> None:
    cal = ChanceCalibration(play_gate={}, long_factor={75: 0.5})
    assert calibrated_long_factor(100.0, cal) == 1.0
    assert calibrated_long_factor(None, cal) == 1.0
    assert calibrated_long_factor(75.0, cal) == 0.5


def test_chance_uses_calibration_when_provided() -> None:
    cal = ChanceCalibration(play_gate={50: 0.3}, long_factor={})
    assert chance("a", 50.0, 0.5) == 0.5
    assert chance("a", 50.0, 0.5, calibration=cal) == 0.3


def test_fit_maps_flagged_underplay_and_early_sub(tmp_path: Path) -> None:
    """Snapshot 75% blanks lower the play gate; early-subs lower the ≥60 factor."""
    under = _calibration_store(tmp_path / "under", mode="underplay")
    early = _calibration_store(tmp_path / "early", mode="early_sub")
    under_view = PITData(under).view(GWKey(_SEASON, 21))
    early_view = PITData(early).view(GWKey(_SEASON, 21))
    under_stats = under_view.latest(
        "player_fixture_stats", ["season", "player_key", "fixture_id"]
    ).collect()
    early_stats = early_view.latest(
        "player_fixture_stats", ["season", "player_key", "fixture_id"]
    ).collect()
    under_cal = fit_chance_calibration(under_view, under_stats, min_count=10)
    early_cal = fit_chance_calibration(early_view, early_stats, min_count=10)
    assert under_cal.play_gate[75] < 0.3
    assert under_cal.play_gate[100] == pytest.approx(1.0, abs=0.05)
    assert early_cal.long_factor[75] < 0.5


def test_v2_replaces_face_value_gate_with_snapshot_map(tmp_path: Path) -> None:
    """P1: calibrated 75% gate benches a nailed starter that face-value still plays."""
    store = _calibration_store(tmp_path, mode="underplay")
    face = EmpiricalMinutesV2(
        _spec(
            calibrate_chance_flags=False,
            soft_squad=False,
            chance_cal_min_count=5,
            return_ramp_games=0.0,
        )
    )
    calibrated = EmpiricalMinutesV2(
        _spec(
            calibrate_chance_flags=True,
            soft_squad=False,
            chance_cal_min_count=5,
            return_ramp_games=0.0,
        )
    )
    view = PITData(store).view(GWKey(_SEASON, 21))
    face.fit(view)
    calibrated.fit(view)
    keys = pl.DataFrame({"player_key": [1], "fixture_id": [21]})
    face_dist = face.predict(view, keys)
    cal_dist = calibrated.predict(view, keys)
    # Healthy start shares survive absence blanks; face 0.75 still plays.
    assert float(face_dist.pmf[0, 0]) < 0.5
    assert float(cal_dist.pmf[0, 0]) > 0.7


def test_v2_scales_p_long_for_flagged_players(tmp_path: Path) -> None:
    """P2: flagged starters get a lower P(≥60|start) than the face-value path."""
    store = _calibration_store(tmp_path, mode="early_sub")
    face = EmpiricalMinutesV2(
        _spec(
            calibrate_chance_flags=False,
            soft_squad=False,
            chance_cal_min_count=5,
            return_ramp_games=0.0,
        )
    )
    calibrated = EmpiricalMinutesV2(
        _spec(
            calibrate_chance_flags=True,
            soft_squad=False,
            chance_cal_min_count=5,
            return_ramp_games=0.0,
        )
    )
    view = PITData(store).view(GWKey(_SEASON, 21))
    face.fit(view)
    calibrated.fit(view)
    keys = pl.DataFrame({"player_key": [1], "fixture_id": [21]})
    face_p60 = float(face.predict(view, keys).pmf[0, MINUTE_BINS >= 60].sum())
    cal_p60 = float(calibrated.predict(view, keys).pmf[0, MINUTE_BINS >= 60].sum())
    assert cal_p60 < face_p60


def test_spec_defaults_enable_calibration() -> None:
    specs = load_model_specs()
    params = specs["empirical_minutes_v2"].params
    assert params["calibrate_chance_flags"] is True
    assert params["chance_cal_min_count"] == 20
    assert specs["dc_layered_v4"].components["minutes"].params["calibrate_chance_flags"]


def _spec(**params: object) -> ModelSpec:
    return ModelSpec(
        name="empirical_minutes_v2",
        kind="empirical_minutes_v2",
        params={
            "windows": [3, 5, 10],
            "shrinkage": 5,
            "doubtful_chance": 0.5,
            "soft_squad": False,
            "target_starters": 11.0,
            "soft_squad_strength": 0.5,
            "logistic_blend": 0.5,
            "l2": 1.0,
            "return_ramp_games": 2.0,
            **params,
        },
    )


def _calibration_store(tmp_path: Path, *, mode: str) -> DataStore:
    """Nailed starter (#1) then flagged 75%; companion (#2) always 100%.

    ``underplay``: flagged GWs are doubtful blanks (absence-tagged so start
    shares stay high; play_gate[75] near 0).
    ``early_sub``: flagged GWs still start but only 45 minutes (long_factor
    drops; play_gate stays near 1).
    """
    store = DataStore(tmp_path)
    healthy = 10
    flagged = 10
    max_gw = healthy + flagged + 1
    store.write(
        "players",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON, _SEASON],
                    "player_key": [1, 2],
                    "element_id": [1, 2],
                    "team_key": [1, 1],
                    "position": ["MID", "MID"],
                    "available_at": [_START, _START],
                    "source": ["fpl_api", "fpl_api"],
                }
            )
        ),
    )
    store.write(
        "teams",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON, _SEASON],
                    "team_key": [1, 2],
                    "team_id": [1, 2],
                    "name": ["T1", "T2"],
                    "short_name": ["T1", "T2"],
                    "available_at": [_START, _START],
                    "source": ["fpl_api", "fpl_api"],
                }
            )
        ),
    )
    deadlines = [_kick(gw) - timedelta(hours=21) for gw in range(1, max_gw + 1)]
    store.write(
        "gameweeks",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * max_gw,
                    "gw": list(range(1, max_gw + 1)),
                    "deadline_time": deadlines,
                    "finished": [gw < max_gw for gw in range(1, max_gw + 1)],
                    "data_checked": [gw < max_gw for gw in range(1, max_gw + 1)],
                    "available_at": [_START] * max_gw,
                    "source": ["fpl_api"] * max_gw,
                }
            )
        ),
    )
    fixture_ids = list(range(1, max_gw + 1))
    store.write(
        "fixtures",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * len(fixture_ids),
                    "fixture_id": fixture_ids,
                    "gw": fixture_ids,
                    "team_h_key": [1] * len(fixture_ids),
                    "team_a_key": [2] * len(fixture_ids),
                    "kickoff_time": [_kick(gw) for gw in fixture_ids],
                    "team_h_score": [1 if gw < max_gw else None for gw in fixture_ids],
                    "team_a_score": [0 if gw < max_gw else None for gw in fixture_ids],
                    "finished": [gw < max_gw for gw in fixture_ids],
                    "available_at": [
                        _kick(gw) + timedelta(hours=3)
                        if gw < max_gw
                        else _kick(gw) - timedelta(days=14)
                        for gw in fixture_ids
                    ],
                    "source": ["fpl_api"] * len(fixture_ids),
                }
            )
        ),
    )
    status_rows: list[dict[str, object]] = []
    stats_rows: list[dict[str, object]] = []
    for gw in range(1, healthy + flagged + 1):
        flagged_gw = gw > healthy
        if flagged_gw and mode == "underplay":
            minutes, starts, chance_pct, status = 0, 0, 75.0, "d"
        elif flagged_gw and mode == "early_sub":
            minutes, starts, chance_pct, status = 45, 1, 75.0, "d"
        else:
            minutes, starts, chance_pct, status = 90, 1, 100.0, "a"
        status_rows.append(
            {
                "season": _SEASON,
                "player_key": 1,
                "status": status,
                "news": "Knock" if flagged_gw else "",
                "price": 70,
                "ep_next": None,
                "chance_of_playing": chance_pct,
                "chance_of_playing_this_round": chance_pct,
                "available_at": deadlines[gw - 1] - timedelta(minutes=5),
                "source": "fpl_api",
            }
        )
        stats_rows.append(
            {
                "season": _SEASON,
                "gw": gw,
                "player_key": 1,
                "fixture_id": gw,
                "minutes": minutes,
                "starts": starts,
                "available_at": _kick(gw) + timedelta(hours=3),
                "source": "fpl_api",
                **_ZERO,
            }
        )
        status_rows.append(
            {
                "season": _SEASON,
                "player_key": 2,
                "status": "a",
                "news": "",
                "price": 70,
                "ep_next": None,
                "chance_of_playing": 100.0,
                "chance_of_playing_this_round": 100.0,
                "available_at": deadlines[gw - 1] - timedelta(minutes=5),
                "source": "fpl_api",
            }
        )
        stats_rows.append(
            {
                "season": _SEASON,
                "gw": gw,
                "player_key": 2,
                "fixture_id": gw,
                "minutes": 90,
                "starts": 1,
                "available_at": _kick(gw) + timedelta(hours=3),
                "source": "fpl_api",
                **_ZERO,
            }
        )
    # Predict-time status: player 1 still flagged at 75%.
    status_rows.append(
        {
            "season": _SEASON,
            "player_key": 1,
            "status": "d",
            "news": "Knock",
            "price": 70,
            "ep_next": None,
            "chance_of_playing": 75.0,
            "chance_of_playing_this_round": 75.0,
            "available_at": deadlines[max_gw - 1] - timedelta(minutes=5),
            "source": "fpl_api",
        }
    )
    status_rows.append(
        {
            "season": _SEASON,
            "player_key": 2,
            "status": "a",
            "news": "",
            "price": 70,
            "ep_next": None,
            "chance_of_playing": 100.0,
            "chance_of_playing_this_round": 100.0,
            "available_at": deadlines[max_gw - 1] - timedelta(minutes=5),
            "source": "fpl_api",
        }
    )
    store.write("player_status", _utc(pl.DataFrame(status_rows)))
    store.write("player_fixture_stats", _utc(pl.DataFrame(stats_rows)))
    return store


def _kick(gw: int) -> datetime:
    return datetime(2024, 8, 10, 15, tzinfo=UTC) + timedelta(days=7 * (gw - 1))


def _utc(frame: pl.DataFrame) -> pl.DataFrame:
    columns = [
        pl.col(name).cast(pl.Datetime("us", "UTC"))
        for name, dtype in frame.schema.items()
        if isinstance(dtype, pl.Datetime)
    ]
    return frame.with_columns(columns) if columns else frame
