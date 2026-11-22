"""Season fixture PIT: long-horizon schedule visibility without rearrangement leaks."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import polars as pl

from fplt.data.pit import PITData
from fplt.data.staging import stage_bootstrap
from fplt.data.store import DataStore
from fplt.data.vaastav import import_season, season_start
from fplt.types import GWKey

_SEASON = "2024-25"
_START = season_start(_SEASON)


def _kick(gw: int) -> datetime:
    """Saturday 14:00 UTC roughly ``gw`` weeks after mid-August."""
    return datetime(2024, 8, 17, 14, tzinfo=UTC) + timedelta(weeks=gw - 1)


def _deadline(gw: int) -> datetime:
    return _kick(gw) - timedelta(hours=2)


def _write_long_vaastav(root: Path, *, n_gw: int = 12) -> Path:
    season = root / _SEASON
    (season / "gws").mkdir(parents=True)
    pl.DataFrame(
        {
            "id": [1],
            "code": [100],
            "element_type": [3],
            "web_name": ["A"],
            "team": [1],
        }
    ).write_csv(season / "players_raw.csv")
    pl.DataFrame(
        {"id": [1, 2], "code": [3, 7], "name": ["A", "B"], "short_name": ["A", "B"]}
    ).write_csv(season / "teams.csv")
    gws = list(range(1, n_gw + 1))
    pl.DataFrame(
        {
            "id": [100 + gw for gw in gws],
            "event": gws,
            "team_h": [1 if gw % 2 else 2 for gw in gws],
            "team_a": [2 if gw % 2 else 1 for gw in gws],
            "kickoff_time": [_kick(gw).strftime("%Y-%m-%dT%H:%M:%SZ") for gw in gws],
            "finished": [gw <= 2 for gw in gws],
            "team_h_score": [1 if gw <= 2 else None for gw in gws],
            "team_a_score": [0 if gw <= 2 else None for gw in gws],
        }
    ).write_csv(season / "fixtures.csv")
    pl.DataFrame(
        {
            "name": ["A"] * n_gw,
            "element": [1] * n_gw,
            "fixture": [100 + gw for gw in gws],
            "GW": gws,
            "was_home": [gw % 2 == 1 for gw in gws],
            "team": [1] * n_gw,
            "opponent_team": [2] * n_gw,
            "minutes": [90] * n_gw,
            "total_points": [2] * n_gw,
            "goals_scored": [0] * n_gw,
            "assists": [0] * n_gw,
            "clean_sheets": [0] * n_gw,
            "goals_conceded": [1] * n_gw,
            "saves": [0] * n_gw,
            "bonus": [0] * n_gw,
            "yellow_cards": [0] * n_gw,
            "red_cards": [0] * n_gw,
            "own_goals": [0] * n_gw,
            "penalties_saved": [0] * n_gw,
            "penalties_missed": [0] * n_gw,
            "starts": [1] * n_gw,
            "value": [50] * n_gw,
            "kickoff_time": [_kick(gw).strftime("%Y-%m-%dT%H:%M:%SZ") for gw in gws],
            "position": ["MID"] * n_gw,
        }
    ).write_csv(season / "gws" / "merged_gw.csv")
    return season


def test_vaastav_mid_season_deadline_sees_eight_upcoming_gws(tmp_path: Path) -> None:
    """Season-start schedule stamps expose H≥8 fixtures at a mid-season deadline."""
    store = DataStore(tmp_path / "store")
    import_season(store, _write_long_vaastav(tmp_path), _SEASON)
    staged = store.read("fixtures").filter(pl.col("team_h_score").is_null())
    assert staged["available_at"].unique().to_list() == [_START]

    as_of = GWKey(_SEASON, 3)
    view = PITData(store).view(as_of)
    visible = (
        view.latest("fixtures", by=["season", "fixture_id"])
        .collect()
        .filter(pl.col("gw") >= as_of.gw)
    )
    upcoming = sorted(visible["gw"].unique().to_list())
    assert upcoming[:8] == list(range(3, 11))
    assert len(upcoming) >= 8


def test_api_rearrangement_after_available_at_stays_hidden(tmp_path: Path) -> None:
    """A later snapshot's rearranged gw/kickoff is invisible before its capture."""
    teams = [
        {"id": 1, "code": 3, "name": "Arsenal", "short_name": "ARS"},
        {"id": 2, "code": 7, "name": "Spurs", "short_name": "TOT"},
    ]
    elements = [
        {
            "id": 10,
            "code": 100,
            "team": 1,
            "element_type": 3,
            "web_name": "Saka",
            "status": "a",
            "news": "",
            "now_cost": 100,
            "ep_next": "6.0",
            "chance_of_playing_next_round": 100,
        }
    ]
    early_events = [
        {
            "id": gw,
            "deadline_time": _deadline(gw).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished": False,
            "data_checked": False,
        }
        for gw in range(1, 12)
    ]
    early_bootstrap = {
        "teams": teams,
        "elements": elements,
        "events": early_events,
    }
    # Original: fixture 50 sits in GW 10.
    early_fixtures = [
        {
            "id": 50,
            "event": 10,
            "team_h": 1,
            "team_a": 2,
            "kickoff_time": _kick(10).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished": False,
            "team_h_score": None,
            "team_a_score": None,
        }
    ]
    # Plus a full upcoming slate so H=8 is covered from the early capture.
    for gw in range(3, 12):
        if gw == 10:
            continue
        early_fixtures.append(
            {
                "id": 100 + gw,
                "event": gw,
                "team_h": 1 if gw % 2 else 2,
                "team_a": 2 if gw % 2 else 1,
                "kickoff_time": _kick(gw).strftime("%Y-%m-%dT%H:%M:%SZ"),
                "finished": False,
                "team_h_score": None,
                "team_a_score": None,
            }
        )

    early_at = datetime(2024, 8, 1, tzinfo=UTC)
    rearrange_at = datetime(2024, 10, 1, tzinfo=UTC)
    # Mid-season deadline after the early schedule, before the rearrangement.
    mid_deadline = _deadline(3)
    assert early_at < mid_deadline < rearrange_at

    late_bootstrap = {
        "teams": teams,
        "elements": elements,
        "events": [
            {
                **event,
                "finished": event["id"] < 3,
                "data_checked": event["id"] < 3,
            }
            for event in early_events
        ],
    }
    # Rearrangement: same fixture_id moves to GW 20 (outside the early slate).
    late_fixtures = [
        {
            "id": 50,
            "event": 20,
            "team_h": 1,
            "team_a": 2,
            "kickoff_time": _kick(20).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished": False,
            "team_h_score": None,
            "team_a_score": None,
        },
        *[fixture for fixture in early_fixtures if int(fixture["id"]) != 50],
    ]

    store = DataStore(tmp_path)
    for frames in (
        stage_bootstrap(early_bootstrap, early_fixtures, early_at, _SEASON),
        stage_bootstrap(late_bootstrap, late_fixtures, rearrange_at, _SEASON),
    ):
        for name, frame in frames.items():
            if not frame.is_empty():
                store.write(name, frame)

    # Override GW3 deadline so the view cutoff sits between the two captures.
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": [_SEASON],
                "gw": [3],
                "deadline_time": [mid_deadline],
                "finished": [False],
                "data_checked": [False],
                "available_at": [early_at],
                "source": ["fpl_api"],
            }
        ).with_columns(
            pl.col("deadline_time").cast(pl.Datetime("us", "UTC")),
            pl.col("available_at").cast(pl.Datetime("us", "UTC")),
        ),
    )

    view = PITData(store).view(GWKey(_SEASON, 3))
    latest = view.latest("fixtures", by=["season", "fixture_id"]).collect()
    moved = latest.filter(pl.col("fixture_id") == 50)
    assert moved.height == 1
    assert moved["gw"].to_list() == [10]
    assert moved["kickoff_time"].to_list() == [_kick(10)]
    assert 20 not in latest["gw"].to_list()

    upcoming = sorted(latest.filter(pl.col("gw") >= 3)["gw"].unique().to_list())
    assert len(upcoming) >= 8

    # Push cutoff past the rearrangement by appending a later deadline row.
    store.write(
        "gameweeks",
        pl.DataFrame(
            {
                "season": [_SEASON],
                "gw": [3],
                "deadline_time": [rearrange_at + timedelta(hours=1)],
                "finished": [False],
                "data_checked": [False],
                "available_at": [rearrange_at],
                "source": ["fpl_api"],
            }
        ).with_columns(
            pl.col("deadline_time").cast(pl.Datetime("us", "UTC")),
            pl.col("available_at").cast(pl.Datetime("us", "UTC")),
        ),
    )
    later = PITData(store).view(GWKey(_SEASON, 3))
    rearranged = (
        later.latest("fixtures", by=["season", "fixture_id"])
        .collect()
        .filter(pl.col("fixture_id") == 50)
    )
    assert rearranged["gw"].to_list() == [20]


def test_checked_snapshot_does_not_backdate_schedule(tmp_path: Path) -> None:
    """Post-result captures must not invent kickoff−14d schedule stamps."""
    kickoff = datetime(2024, 8, 17, 14, tzinfo=UTC)
    captured = datetime(2024, 8, 18, 12, tzinfo=UTC)
    bootstrap = {
        "teams": [
            {"id": 1, "code": 3, "name": "Arsenal", "short_name": "ARS"},
            {"id": 2, "code": 7, "name": "Spurs", "short_name": "TOT"},
        ],
        "elements": [
            {
                "id": 10,
                "code": 100,
                "team": 1,
                "element_type": 3,
                "web_name": "Saka",
                "status": "a",
                "news": "",
                "now_cost": 100,
                "ep_next": "6.0",
                "chance_of_playing_next_round": 100,
            }
        ],
        "events": [
            {
                "id": 1,
                "deadline_time": "2024-08-16T17:30:00Z",
                "finished": True,
                "data_checked": True,
            }
        ],
    }
    fixtures = [
        {
            "id": 50,
            "event": 1,
            "team_h": 1,
            "team_a": 2,
            "kickoff_time": kickoff.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "finished": True,
            "team_h_score": 2,
            "team_a_score": 0,
        }
    ]
    frame = stage_bootstrap(bootstrap, fixtures, captured, "2024-25")["fixtures"]
    assert frame.height == 1
    assert frame["available_at"].to_list() == [captured]
    assert frame["team_h_score"].to_list() == [2]
    del tmp_path
