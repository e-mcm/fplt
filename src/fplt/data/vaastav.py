"""Import vaastav historical seasons into the staged parquet tables.

Availability is not observed per deadline in this source, so every row is
flagged ``source = "vaastav"``. Result rows become visible at kickoff + 3h.
Status is a single degraded row per player: no price, news, or chance.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import polars as pl

from fplt.data.store import DataStore
from fplt.schemas.staged import POLARS_SCHEMAS

_RESULT_LAG = timedelta(hours=3)
_MIN_SEASON = "2021-22"
_POSITIONS = {"GK", "DEF", "MID", "FWD"}
_UTC = pl.Datetime("us", "UTC")


def season_start(season: str) -> datetime:
    """1 July of the first year in ``season``, used as the schedule timestamp."""
    year = int(season.split("-")[0])
    return datetime(year, 7, 1, tzinfo=UTC)


def import_season(store: DataStore, season_dir: Path, season: str) -> None:
    """Stage one vaastav season directory if it is 2021-22 or later.

    Args:
        store: Destination for the staged parquet tables.
        season_dir: Directory containing ``players_raw.csv``, ``teams.csv``,
            ``fixtures.csv``, and ``gws/merged_gw.csv``.
        season: Season label such as ``"2021-22"``. Earlier seasons are ignored.
    """
    if season < _MIN_SEASON:
        return
    gw_path = season_dir / "gws" / "merged_gw.csv"
    if not gw_path.exists():
        return
    players = _read_optional(season_dir / "players_raw.csv")
    teams = _read_optional(season_dir / "teams.csv")
    fixtures = _read_optional(season_dir / "fixtures.csv")
    gameweeks = _read_gw(gw_path)
    code_by_element = _element_codes(players)
    team_code = _team_codes(teams)
    start = season_start(season)
    store.write(
        "players",
        _players(gameweeks, code_by_element, _team_lookup(teams), season, start),
    )
    if teams is not None:
        store.write("teams", _teams(teams, season, start))
    if fixtures is not None:
        store.write("fixtures", _fixtures(fixtures, team_code, season))
        store.write("gameweeks", _gameweeks(fixtures, season, start))
    store.write("player_status", _status(gameweeks, code_by_element, season, start))
    store.write(
        "player_fixture_stats",
        _stats(gameweeks, code_by_element, fixtures, season),
    )


def import_available(store: DataStore, data_dir: Path) -> list[str]:
    """Import every season directory under ``data_dir`` that has a merged gameweek file.

    Seasons before 2021-22 are skipped. Returns the season labels that were imported.
    """
    if not data_dir.is_dir():
        return []
    imported: list[str] = []
    for path in sorted(item for item in data_dir.iterdir() if item.is_dir()):
        if path.name < _MIN_SEASON or not (path / "gws" / "merged_gw.csv").exists():
            continue
        import_season(store, path, path.name)
        imported.append(path.name)
    return imported


def _read_optional(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        return None
    return pl.read_csv(path, infer_schema_length=1000)


def _read_gw(path: Path) -> pl.DataFrame:
    try:
        return pl.read_csv(path, infer_schema_length=5000)
    except Exception:
        return pl.read_csv(path, infer_schema_length=5000, encoding="latin1")


def _element_codes(players: pl.DataFrame | None) -> dict[int, int]:
    if players is None or "code" not in players.columns or "id" not in players.columns:
        return {}
    return {
        int(row["id"]): int(row["code"])
        for row in players.select(["id", "code"]).iter_rows(named=True)
    }


def _team_codes(teams: pl.DataFrame | None) -> dict[int, int]:
    if teams is None or "code" not in teams.columns or "id" not in teams.columns:
        return {}
    return {
        int(row["id"]): int(row["code"])
        for row in teams.select(["id", "code"]).iter_rows(named=True)
        if row["id"] is not None and row["code"] is not None
    }


def _team_lookup(teams: pl.DataFrame | None) -> dict[str, int]:
    """Map a team id or name onto the stable team code.

    Merged gameweeks name the club (``Arsenal``). Fixture files use the
    numeric team id. Both have to meet the code stored on ``teams``.
    """
    lookup: dict[str, int] = {}
    for team_id, code in _team_codes(teams).items():
        lookup[str(team_id)] = code
    if teams is None:
        return lookup
    for row in teams.iter_rows(named=True):
        raw_code = row.get("code")
        if raw_code is None:
            continue
        code = int(raw_code)
        for key in ("name", "short_name"):
            label = row.get(key)
            if label:
                lookup[str(label)] = code
    return lookup


def _team_key(value: object, lookup: dict[str, int]) -> int:
    label = "" if value is None else str(value)
    if label in lookup:
        return lookup[label]
    return _as_int(value)


def _kickoff(value: object) -> datetime:
    parsed = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _as_int(value: object, default: int = 0) -> int:
    try:
        return int(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _players(
    gw: pl.DataFrame,
    codes: dict[int, int],
    teams: dict[str, int],
    season: str,
    start: datetime,
) -> pl.DataFrame:
    if gw.is_empty():
        return pl.DataFrame(schema=POLARS_SCHEMAS["players"])
    latest = gw.sort("GW").unique(subset=["element"], keep="last")
    rows: list[dict[str, Any]] = []
    for row in latest.iter_rows(named=True):
        element = _as_int(row["element"])
        position = str(row.get("position") or "MID")
        rows.append(
            {
                "season": season,
                "player_key": codes.get(element, element),
                "element_id": element,
                "team_key": _team_key(row.get("team"), teams),
                "position": position if position in _POSITIONS else "MID",
                "web_name": str(row.get("name") or ""),
                "available_at": start,
                "source": "vaastav",
            }
        )
    return pl.DataFrame(rows).with_columns(pl.col("available_at").cast(_UTC))


def _teams(teams: pl.DataFrame, season: str, start: datetime) -> pl.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "season": season,
            "team_key": int(row["code"]),
            "team_id": int(row["id"]),
            "name": str(row["name"]),
            "short_name": str(row.get("short_name") or ""),
            "available_at": start,
            "source": "vaastav",
        }
        for row in teams.iter_rows(named=True)
        if row.get("code") is not None and row.get("id") is not None
    ]
    if not rows:
        return pl.DataFrame(schema=POLARS_SCHEMAS["teams"])
    return pl.DataFrame(rows).with_columns(pl.col("available_at").cast(_UTC))


def _fixtures(
    fixtures: pl.DataFrame, team_code: dict[int, int], season: str
) -> pl.DataFrame:
    rows: list[dict[str, Any]] = []
    for row in fixtures.iter_rows(named=True):
        if row.get("event") is None:
            continue
        kickoff = _kickoff(row["kickoff_time"])
        home_id = _as_int(row["team_h"])
        away_id = _as_int(row["team_a"])
        base = {
            "season": season,
            "fixture_id": _as_int(row["id"]),
            "gw": _as_int(row["event"]),
            "team_h_key": team_code.get(home_id, home_id),
            "team_a_key": team_code.get(away_id, away_id),
            "kickoff_time": kickoff,
            "source": "vaastav",
        }
        rows.append(
            {
                **base,
                "team_h_score": None,
                "team_a_score": None,
                "finished": False,
                "available_at": kickoff - timedelta(days=14),
            }
        )
        if row.get("finished") and row.get("team_h_score") is not None:
            rows.append(
                {
                    **base,
                    "team_h_score": _as_int(row["team_h_score"]),
                    "team_a_score": _as_int(row["team_a_score"]),
                    "finished": True,
                    "available_at": kickoff + _RESULT_LAG,
                }
            )
    empty = pl.DataFrame(schema=POLARS_SCHEMAS["fixtures"])
    frame = pl.DataFrame(rows) if rows else empty
    return frame.with_columns(
        pl.col("kickoff_time").cast(pl.Datetime("us", "UTC")),
        pl.col("available_at").cast(pl.Datetime("us", "UTC")),
        pl.col("team_h_score").cast(pl.Int64),
        pl.col("team_a_score").cast(pl.Int64),
    )


def _gameweeks(fixtures: pl.DataFrame, season: str, start: datetime) -> pl.DataFrame:
    if "event" not in fixtures.columns:
        return pl.DataFrame(schema=POLARS_SCHEMAS["gameweeks"])
    played = fixtures.filter(pl.col("event").is_not_null())
    if played.is_empty():
        return pl.DataFrame(schema=POLARS_SCHEMAS["gameweeks"])
    grouped = played.group_by("event").agg(
        pl.col("kickoff_time").min().alias("kickoff"),
        pl.col("finished").cast(pl.Boolean).all().alias("checked"),
    )
    rows: list[dict[str, Any]] = []
    for row in grouped.iter_rows(named=True):
        kickoff = _kickoff(row["kickoff"])
        checked = bool(row["checked"])
        rows.append(
            {
                "season": season,
                "gw": _as_int(row["event"]),
                "deadline_time": kickoff - timedelta(hours=2),
                "finished": checked,
                "data_checked": checked,
                "available_at": start,
                "source": "vaastav",
            }
        )
    return pl.DataFrame(rows).with_columns(
        pl.col("deadline_time").cast(pl.Datetime("us", "UTC")),
        pl.col("available_at").cast(pl.Datetime("us", "UTC")),
    )


def _status(
    gw: pl.DataFrame, codes: dict[int, int], season: str, start: datetime
) -> pl.DataFrame:
    if gw.is_empty() or "element" not in gw.columns:
        return pl.DataFrame(schema=POLARS_SCHEMAS["player_status"])
    rows: list[dict[str, Any]] = [
        {
            "season": season,
            "player_key": codes.get(_as_int(row["element"]), _as_int(row["element"])),
            "status": "a",
            "news": "",
            "price": 0,
            "ep_next": None,
            "chance_of_playing": None,
            "available_at": start,
            "source": "vaastav",
        }
        for row in gw.select("element").unique().iter_rows(named=True)
    ]
    return pl.DataFrame(rows).with_columns(
        pl.col("available_at").cast(pl.Datetime("us", "UTC")),
        pl.col("ep_next").cast(pl.Float64),
        pl.col("chance_of_playing").cast(pl.Float64),
    )


def _stats(
    gw: pl.DataFrame,
    codes: dict[int, int],
    fixtures: pl.DataFrame | None,
    season: str,
) -> pl.DataFrame:
    fixture_by_team_gw = _fixture_index(fixtures)
    rows: list[dict[str, Any]] = []
    for row in gw.iter_rows(named=True):
        team = _as_int(row.get("team"))
        was_home = bool(row.get("was_home"))
        found = fixture_by_team_gw.get((team, _as_int(row.get("GW")), was_home))
        if found is None:
            fixture_id = _as_int(row.get("fixture"))
            raw_kick = row.get("kickoff_time")
            kickoff = _kickoff(raw_kick) if raw_kick else season_start(season)
        else:
            fixture_id, kickoff = found
        minutes = _as_int(row.get("minutes"))
        raw_starts = row.get("starts")
        starts = _as_int(raw_starts) if raw_starts is not None else int(minutes >= 60)
        element = _as_int(row.get("element"))
        rows.append(
            {
                "season": season,
                "gw": _as_int(row.get("GW")),
                "player_key": codes.get(element, element),
                "fixture_id": fixture_id,
                "minutes": minutes,
                "total_points": _as_int(row.get("total_points")),
                "goals_scored": _as_int(row.get("goals_scored")),
                "assists": _as_int(row.get("assists")),
                "clean_sheets": _as_int(row.get("clean_sheets")),
                "goals_conceded": _as_int(row.get("goals_conceded")),
                "saves": _as_int(row.get("saves")),
                "bonus": _as_int(row.get("bonus")),
                "yellow_cards": _as_int(row.get("yellow_cards")),
                "red_cards": _as_int(row.get("red_cards")),
                "own_goals": _as_int(row.get("own_goals")),
                "penalties_saved": _as_int(row.get("penalties_saved")),
                "penalties_missed": _as_int(row.get("penalties_missed")),
                "starts": starts,
                "available_at": kickoff + _RESULT_LAG,
                "source": "vaastav",
            }
        )
    if not rows:
        return pl.DataFrame(schema=POLARS_SCHEMAS["player_fixture_stats"])
    return pl.DataFrame(rows).with_columns(pl.col("available_at").cast(_UTC))


def _fixture_index(
    fixtures: pl.DataFrame | None,
) -> dict[tuple[int, int, bool], tuple[int, datetime]]:
    index: dict[tuple[int, int, bool], tuple[int, datetime]] = {}
    if fixtures is None:
        return index
    for row in fixtures.iter_rows(named=True):
        if row.get("event") is None or row.get("kickoff_time") is None:
            continue
        kickoff = _kickoff(row["kickoff_time"])
        gw_id = _as_int(row["event"])
        fixture_id = _as_int(row["id"])
        index[(int(row["team_h"]), gw_id, True)] = (fixture_id, kickoff)
        index[(int(row["team_a"]), gw_id, False)] = (fixture_id, kickoff)
    return index
