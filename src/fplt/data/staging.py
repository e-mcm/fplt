"""Turn raw FPL API payloads into staged frames.

Schedule rows are published at ``captured_at``. Result rows are published at
the capture where ``data_checked`` is true, which is the moment those scores
become visible to a later point-in-time view.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any, cast

import polars as pl

_POSITION = {1: "GK", 2: "DEF", 3: "MID", 4: "FWD"}


def _utc(value: str | datetime) -> datetime:
    if isinstance(value, datetime):
        return value.astimezone(UTC)
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)


def _team_code(teams: list[dict[str, Any]]) -> dict[int, int]:
    return {int(team["id"]): int(team["code"]) for team in teams}


def stage_bootstrap(
    bootstrap: dict[str, Any],
    fixtures: list[dict[str, Any]],
    captured_at: datetime,
    season: str,
    source: str = "fpl_api",
) -> dict[str, pl.DataFrame]:
    """Stage one bootstrap snapshot and its fixture list."""
    moment = captured_at.astimezone(UTC)
    codes = _team_code(bootstrap.get("teams", []))
    players = _players(bootstrap, codes, moment, season, source)
    teams = _teams(bootstrap, moment, season, source)
    gameweeks = _gameweeks(bootstrap, moment, season, source)
    status = _status(bootstrap, moment, season, source)
    fixture_frame = _fixtures(fixtures, codes, moment, season, source, bootstrap)
    return {
        "players": players,
        "teams": teams,
        "gameweeks": gameweeks,
        "player_status": status,
        "fixtures": fixture_frame,
    }


def stage_event_live(
    payload: dict[str, Any],
    season: str,
    gw: int,
    available_at: datetime,
    element_to_player: dict[int, int],
    source: str = "fpl_api",
) -> pl.DataFrame:
    """Stage per-player fixture stats from ``event/{gw}/live``."""
    rows: list[dict[str, Any]] = []
    for element in payload.get("elements", []):
        element_id = int(element["id"])
        player_key = element_to_player.get(element_id, element_id)
        stats = element.get("stats", {})
        explains = cast(list[dict[str, Any]], element.get("explain") or [])
        fixture_ids = [int(item["fixture"]) for item in explains] or [0]
        for fixture_id in fixture_ids:
            rows.append(
                {
                    "season": season,
                    "gw": gw,
                    "player_key": player_key,
                    "fixture_id": fixture_id,
                    "minutes": int(stats.get("minutes", 0)),
                    "total_points": int(stats.get("total_points", 0)),
                    "goals_scored": int(stats.get("goals_scored", 0)),
                    "assists": int(stats.get("assists", 0)),
                    "clean_sheets": int(stats.get("clean_sheets", 0)),
                    "goals_conceded": int(stats.get("goals_conceded", 0)),
                    "saves": int(stats.get("saves", 0)),
                    "bonus": int(stats.get("bonus", 0)),
                    "yellow_cards": int(stats.get("yellow_cards", 0)),
                    "red_cards": int(stats.get("red_cards", 0)),
                    "own_goals": int(stats.get("own_goals", 0)),
                    "penalties_saved": int(stats.get("penalties_saved", 0)),
                    "penalties_missed": int(stats.get("penalties_missed", 0)),
                    "starts": int(stats.get("starts", 0)),
                    "influence": _stat_float(stats, "influence"),
                    "creativity": _stat_float(stats, "creativity"),
                    "threat": _stat_float(stats, "threat"),
                    "bps": int(float(stats.get("bps") or 0)),
                    "available_at": available_at.astimezone(UTC),
                    "source": source,
                }
            )
    if not rows:
        from fplt.schemas.staged import POLARS_SCHEMAS

        return pl.DataFrame(schema=POLARS_SCHEMAS["player_fixture_stats"])
    return pl.DataFrame(rows).with_columns(
        pl.col("available_at").cast(pl.Datetime("us", "UTC"))
    )


def _players(
    bootstrap: dict[str, Any],
    codes: dict[int, int],
    moment: datetime,
    season: str,
    source: str,
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for element in bootstrap.get("elements", []):
        rows.append(
            {
                "season": season,
                "player_key": int(element["code"]),
                "element_id": int(element["id"]),
                "team_key": codes[int(element["team"])],
                "position": _POSITION[int(element["element_type"])],
                "web_name": str(element.get("web_name", "")),
                "available_at": moment,
                "source": source,
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("available_at").cast(pl.Datetime("us", "UTC"))
    )


def _teams(
    bootstrap: dict[str, Any], moment: datetime, season: str, source: str
) -> pl.DataFrame:
    rows = [
        {
            "season": season,
            "team_key": int(team["code"]),
            "team_id": int(team["id"]),
            "name": str(team["name"]),
            "short_name": str(team.get("short_name", "")),
            "available_at": moment,
            "source": source,
        }
        for team in bootstrap.get("teams", [])
    ]
    return pl.DataFrame(rows).with_columns(
        pl.col("available_at").cast(pl.Datetime("us", "UTC"))
    )


def _gameweeks(
    bootstrap: dict[str, Any], moment: datetime, season: str, source: str
) -> pl.DataFrame:
    rows = [
        {
            "season": season,
            "gw": int(event["id"]),
            "deadline_time": _utc(str(event["deadline_time"])),
            "finished": bool(event.get("finished", False)),
            "data_checked": bool(event.get("data_checked", False)),
            "available_at": moment,
            "source": source,
        }
        for event in bootstrap.get("events", [])
    ]
    return pl.DataFrame(rows).with_columns(
        pl.col("deadline_time").cast(pl.Datetime("us", "UTC")),
        pl.col("available_at").cast(pl.Datetime("us", "UTC")),
    )


def _status(
    bootstrap: dict[str, Any], moment: datetime, season: str, source: str
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for element in bootstrap.get("elements", []):
        chance = element.get("chance_of_playing_next_round")
        ep = element.get("ep_next")
        rows.append(
            {
                "season": season,
                "player_key": int(element["code"]),
                "status": str(element.get("status", "a")),
                "news": str(element.get("news") or ""),
                "price": int(element.get("now_cost", 0)),
                "ep_next": float(ep) if ep not in (None, "") else None,
                "chance_of_playing": float(chance) if chance is not None else None,
                "available_at": moment,
                "source": source,
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("available_at").cast(pl.Datetime("us", "UTC"))
    )


def _fixtures(
    fixtures: list[dict[str, Any]],
    codes: dict[int, int],
    moment: datetime,
    season: str,
    source: str,
    bootstrap: dict[str, Any],
) -> pl.DataFrame:
    checked = {
        int(event["id"]): bool(event.get("data_checked", False))
        for event in bootstrap.get("events", [])
    }
    rows: list[dict[str, Any]] = []
    for fixture in fixtures:
        gw = fixture.get("event")
        if gw is None:
            continue
        kickoff = _utc(str(fixture["kickoff_time"]))
        finished = bool(fixture.get("finished", False))
        # Scores stay null until the gameweek is data-checked, so a view taken
        # at the deadline cannot see the result from this same snapshot.
        scores_visible = checked.get(int(gw), False)
        home = fixture.get("team_h_score")
        away = fixture.get("team_a_score")
        home_score = int(home) if scores_visible and home is not None else None
        away_score = int(away) if scores_visible and away is not None else None
        rows.append(
            {
                "season": season,
                "fixture_id": int(fixture["id"]),
                "gw": int(gw),
                "team_h_key": codes[int(fixture["team_h"])],
                "team_a_key": codes[int(fixture["team_a"])],
                "kickoff_time": kickoff,
                "team_h_score": home_score,
                "team_a_score": away_score,
                "finished": finished and scores_visible,
                "available_at": moment,
                "source": source,
            }
        )
        if scores_visible:
            # Also publish the pre-match schedule at kickoff minus 14 days so a
            # view before the result still sees the fixture.
            rows.append(
                {
                    **rows[-1],
                    "team_h_score": None,
                    "team_a_score": None,
                    "finished": False,
                    "available_at": kickoff - timedelta(days=14),
                }
            )
    frame = pl.DataFrame(rows)
    return frame.with_columns(
        pl.col("kickoff_time").cast(pl.Datetime("us", "UTC")),
        pl.col("available_at").cast(pl.Datetime("us", "UTC")),
        pl.col("team_h_score").cast(pl.Int64),
        pl.col("team_a_score").cast(pl.Int64),
    )


def _stat_float(stats: dict[str, Any], key: str) -> float:
    """Parse an FPL live ``stats`` field that may be a string or number."""
    value = stats.get(key, 0)
    if value is None or value == "":
        return 0.0
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
