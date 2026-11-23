"""Absence-aware shares and return-ramp helpers (minutes P4)."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl
import pytest

import fplt.models.components.minutes  # noqa: F401 — registers the component
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.components.minutes.absence import (
    annotate_absences,
    available_rolling_shares,
    return_metrics,
    return_ramp_factor,
    scale_return_feature,
)
from fplt.models.components.minutes.empirical_v2 import EmpiricalMinutesV2
from fplt.models.spec import ModelSpec
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


def test_return_metrics_tracks_streak_and_recovery() -> None:
    assert return_metrics([]) == (0.0, 0.0, False)
    assert return_metrics([False, False, False]) == (3.0, 0.0, False)
    assert return_metrics([True, True, True]) == (0.0, 3.0, False)
    assert return_metrics([False, True, True, False, False]) == (2.0, 2.0, False)


def test_return_metrics_marks_suspension_streak() -> None:
    gsr, alen, from_ban = return_metrics(
        [False, True, False],
        [False, True, False],
    )
    assert (gsr, alen, from_ban) == (1.0, 1.0, True)
    # Mixed injury then suspension → not suspension-only.
    _, _, mixed = return_metrics(
        [True, True, False],
        [False, True, False],
    )
    assert mixed is False


def test_return_ramp_factor_scales_first_games_back() -> None:
    assert return_ramp_factor(0.0, 0.0, 2.0) == 1.0
    assert return_ramp_factor(0.0, 3.0, 2.0) == pytest.approx(1.0 / 3.0)
    assert return_ramp_factor(1.0, 3.0, 2.0) == pytest.approx(2.0 / 3.0)
    assert return_ramp_factor(2.0, 3.0, 2.0) == 1.0
    assert return_ramp_factor(0.0, 3.0, 0.0) == 1.0
    assert return_ramp_factor(0.0, 1.0, 2.0, from_suspension=True) == 1.0


def test_scale_return_feature_caps_at_ten() -> None:
    assert scale_return_feature(0.0) == 0.0
    assert scale_return_feature(5.0) == 0.5
    assert scale_return_feature(99.0) == 1.0


def test_available_rolling_shares_skip_injury_zeros() -> None:
    group = pl.DataFrame(
        {
            "starts": [1, 1, 1, 0, 0, 0, 1],
            "minutes": [90, 90, 90, 0, 0, 0, 90],
            "is_absence": [False, False, False, True, True, True, False],
        }
    )
    features = available_rolling_shares(group, (3, 5, 10))
    # Last three *available* appearances are all starts — not diluted by zeros.
    assert features["starts_share_3"] == pytest.approx(1.0)
    assert features["starts_share_5"] == pytest.approx(1.0)
    assert features["games_since_return"] == pytest.approx(1.0)
    assert features["absence_length"] == pytest.approx(3.0)


def test_annotate_absences_uses_status_at_deadline(tmp_path: Path) -> None:
    store = _store_with_injury_gap(tmp_path)
    view = PITData(store).view(GWKey(_SEASON, 8))
    stats = view.latest(
        "player_fixture_stats", ["season", "player_key", "fixture_id"]
    ).collect()
    tagged = annotate_absences(view, stats).sort("gw")
    by_gw = {
        int(row["gw"]): bool(row["is_absence"]) for row in tagged.iter_rows(named=True)
    }
    assert by_gw[1] is False
    assert by_gw[2] is False
    assert by_gw[3] is True
    assert by_gw[4] is True
    assert by_gw[5] is True
    assert by_gw[6] is False


def test_injury_zeros_do_not_kill_start_share(tmp_path: Path) -> None:
    """Regression for P4: returning starter keeps high share after injury blanks."""
    store = _store_with_injury_gap(tmp_path)
    model = EmpiricalMinutesV2(
        ModelSpec(
            name="empirical_minutes_v2",
            kind="empirical_minutes_v2",
            params={
                "windows": [3, 5, 10],
                "shrinkage": 5,
                "doubtful_chance": 0.5,
                "soft_squad": False,
                "logistic_blend": 0.0,
                "l2": 1.0,
                "return_ramp_games": 0.0,
                "calibrate_chance_flags": False,
            },
        )
    )
    view = PITData(store).view(GWKey(_SEASON, 8))
    model.fit(view)
    features = model._player_features[1]
    assert features["starts_share_3"] == pytest.approx(1.0)
    assert features["starts_share_5"] == pytest.approx(1.0)
    assert features["games_since_return"] == pytest.approx(1.0)
    assert features["absence_length"] == pytest.approx(3.0)
    dist = model.predict(view, pl.DataFrame({"player_key": [1], "fixture_id": [8]}))
    # Empirical blend=0 + absence-aware share → still expected to play.
    assert float(1.0 - dist.pmf[0, 0]) > 0.7


def test_return_ramp_dampens_first_game_back(tmp_path: Path) -> None:
    store = _store_with_injury_gap(tmp_path)
    ask = pl.DataFrame({"player_key": [1], "fixture_id": [8]})
    full = _predict_ramp(store, ask, ramp=0.0)
    ramped = _predict_ramp(store, ask, ramp=2.0)
    p_full = float(1.0 - full.pmf[0, 0])
    p_ramped = float(1.0 - ramped.pmf[0, 0])
    assert p_ramped < p_full
    # gsr=1, ramp_games=2 → factor (1+1)/(2+1) = 2/3.
    assert p_ramped == pytest.approx(p_full * (2.0 / 3.0), rel=0.15)


def test_suspension_absence_skips_return_ramp(tmp_path: Path) -> None:
    """One-match ban must not look like an injury recovery under the ramp."""
    store = _store_with_suspension_gap(tmp_path)
    view = PITData(store).view(GWKey(_SEASON, 8))
    stats = view.latest(
        "player_fixture_stats", ["season", "player_key", "fixture_id"]
    ).collect()
    tagged = annotate_absences(view, stats).sort("gw")
    by_gw = {
        int(row["gw"]): (bool(row["is_absence"]), bool(row["is_suspension"]))
        for row in tagged.iter_rows(named=True)
    }
    assert by_gw[3] == (True, True)
    assert by_gw[4] == (True, True)
    assert by_gw[5] == (True, True)
    ask = pl.DataFrame({"player_key": [1], "fixture_id": [8]})
    full = _predict_ramp(store, ask, ramp=0.0)
    ramped = _predict_ramp(store, ask, ramp=2.0)
    assert float(1.0 - ramped.pmf[0, 0]) == pytest.approx(
        float(1.0 - full.pmf[0, 0]), rel=1e-6
    )


def test_bad_return_ramp_refused() -> None:
    with pytest.raises(ValueError, match="return_ramp_games"):
        EmpiricalMinutesV2(
            ModelSpec(
                name="empirical_minutes_v2",
                kind="empirical_minutes_v2",
                params={"windows": [3], "return_ramp_games": -1},
            )
        )


def _predict_ramp(store: DataStore, ask: pl.DataFrame, *, ramp: float):
    model = EmpiricalMinutesV2(
        ModelSpec(
            name="empirical_minutes_v2",
            kind="empirical_minutes_v2",
            params={
                "windows": [3, 5, 10],
                "shrinkage": 5,
                "doubtful_chance": 0.5,
                "soft_squad": False,
                "logistic_blend": 0.0,
                "l2": 1.0,
                "return_ramp_games": ramp,
                "calibrate_chance_flags": False,
            },
        )
    )
    view = PITData(store).view(GWKey(_SEASON, 8))
    model.fit(view)
    return model.predict(view, ask)


def _store_with_injury_gap(root: Path) -> DataStore:
    """Regular starter, three injured blanks, one game back — predict GW8."""
    return _store_with_absence_gap(root, absent_status="i")


def _store_with_suspension_gap(root: Path) -> DataStore:
    """Same timeline as the injury gap, but status ``s`` for the blanks."""
    return _store_with_absence_gap(root, absent_status="s")


def _store_with_absence_gap(root: Path, *, absent_status: str) -> DataStore:
    """Regular starter, three unavailable blanks, one game back — predict GW8."""
    store = DataStore(root)
    max_gw = 8
    # GW1–2 start; GW3–5 unavailable zeros; GW6 one game back (predict as-of GW8).
    stats = [
        (1, 90, 1),
        (2, 90, 1),
        (3, 0, 0),
        (4, 0, 0),
        (5, 0, 0),
        (6, 90, 1),
    ]
    store.write(
        "players",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON],
                    "player_key": [1],
                    "element_id": [1],
                    "team_key": [1],
                    "position": ["MID"],
                    "web_name": ["1"],
                    "available_at": [_START],
                    "source": ["fpl_api"],
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
    status_rows = [
        (_START, "a", 100.0),
        (deadlines[2] - timedelta(hours=1), absent_status, 0.0),
        (deadlines[5] - timedelta(hours=1), "a", 100.0),
    ]
    store.write(
        "player_status",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * len(status_rows),
                    "player_key": [1] * len(status_rows),
                    "status": [row[1] for row in status_rows],
                    "news": [""] * len(status_rows),
                    "price": [70] * len(status_rows),
                    "ep_next": pl.Series(
                        "ep_next", [None] * len(status_rows), dtype=pl.Float64
                    ),
                    "chance_of_playing": pl.Series(
                        "chance_of_playing",
                        [row[2] for row in status_rows],
                        dtype=pl.Float64,
                    ),
                    "available_at": [row[0] for row in status_rows],
                    "source": ["fpl_api"] * len(status_rows),
                }
            )
        ),
    )
    store.write(
        "player_fixture_stats",
        _utc(
            pl.DataFrame(
                {
                    "season": [_SEASON] * len(stats),
                    "gw": [gw for gw, _m, _s in stats],
                    "player_key": [1] * len(stats),
                    "fixture_id": [gw for gw, _m, _s in stats],
                    "minutes": [minutes for _gw, minutes, _s in stats],
                    "starts": [starts for _gw, _m, starts in stats],
                    "available_at": [
                        _kick(gw) + timedelta(hours=3) for gw, _m, _s in stats
                    ],
                    "source": ["fpl_api"] * len(stats),
                    **{name: [0] * len(stats) for name in _ZERO},
                }
            )
        ),
    )
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
