"""Import vaastav historical seasons into the staged parquet tables.

Availability is not observed per deadline in this source, so every row is
flagged ``source = "vaastav"``. Fixture **schedule** rows use
``season_start`` (1 July) as the earliest-known proxy so mid-season
deadlines can see the rest of the season (honest H=8–10 planning). Vaastav
ships only the final ``fixtures.csv``, so late rearrangements present only
in that file can appear earlier than they were published — prefer FPL API
snapshot fixtures when they exist for the season. Result rows become
visible at kickoff + 3h. Status still lacks news, chance-of-playing, and
``penalties_order``, but per-gameweek ``value`` from ``merged_gw.csv`` is
staged as ``player_status.price`` at that GW's deadline so historical
transfer budgets bind. Opta ``expected_goals`` / ``expected_assists`` /
``expected_goal_involvements`` / ``expected_goals_conceded`` are staged when
present (2022-23 onwards); earlier seasons leave them null.
"""

from __future__ import annotations

import re
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
_CAP_RE = re.compile(r"^captured_at=(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)$")
_SEASON_RE = re.compile(r"^\d{4}-\d{2}$")


def season_start(season: str) -> datetime:
    """1 July of the first year in ``season``, used as the schedule timestamp."""
    year = int(season.split("-")[0])
    return datetime(year, 7, 1, tzinfo=UTC)


def iter_vaastav_captures(raw_root: Path) -> list[Path]:
    """Return ``raw_root/vaastav/captured_at=*`` dirs sorted by stamp ascending."""
    base = raw_root / "vaastav"
    if not base.is_dir():
        return []
    found: list[tuple[datetime, Path]] = []
    for path in base.iterdir():
        match = _CAP_RE.match(path.name)
        if match is None or not path.is_dir():
            continue
        stamp = datetime.strptime(match.group(1), "%Y-%m-%dT%H-%M-%SZ").replace(
            tzinfo=UTC
        )
        found.append((stamp, path))
    return [path for _, path in sorted(found)]


def latest_season_dirs(data_dir: Path) -> dict[str, Path]:
    """Map season label → latest ``…/data/<season>`` under ``raw/vaastav/``.

    ``captured_at`` is provenance only: a 2026 capture of 2022-23 is still the
    source for a 2022-23 backtest. Callers apply row-level ``available_at`` via
    ``PITView``; do not filter seasons by capture stamp.

    Args:
        data_dir: ``$FPLT_DATA_DIR`` (contains ``raw/``).

    Returns:
        Season folders from the newest capture that includes each season.
    """
    by_season: dict[str, Path] = {}
    for capture in iter_vaastav_captures(data_dir / "raw"):
        data = capture / "data"
        if not data.is_dir():
            continue
        for path in data.iterdir():
            if (
                path.is_dir()
                and _SEASON_RE.match(path.name)
                and (path / "gws" / "merged_gw.csv").is_file()
            ):
                by_season[path.name] = path
    return by_season


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
    deadlines: dict[int, datetime] = {}
    if fixtures is not None:
        store.write("fixtures", _fixtures(fixtures, team_code, season, start))
        gw_meta = _gameweeks(fixtures, season, start)
        store.write("gameweeks", gw_meta)
        deadlines = {
            int(row["gw"]): row["deadline_time"]
            for row in gw_meta.iter_rows(named=True)
        }
    store.write(
        "player_status",
        _status(gameweeks, code_by_element, season, start, deadlines),
    )
    store.write(
        "player_fixture_stats",
        _stats(gameweeks, code_by_element, fixtures, season),
    )


def import_available(store: DataStore, data_dir: Path) -> list[str]:
    """Import vaastav seasons from the latest ``raw/vaastav/captured_at=…`` trees.

    Reads ``data_dir/raw/vaastav/`` only (not ``data_dir/<season>/``). For each
    season, uses the newest capture that contains it. Seasons before 2021-22
    are skipped at import time; raw may still hold earlier seasons forever.

    Args:
        store: Destination for staged parquet tables.
        data_dir: ``$FPLT_DATA_DIR`` root containing ``raw/``.

    Returns:
        Season labels that were imported, sorted.
    """
    imported: list[str] = []
    for season, path in sorted(latest_season_dirs(data_dir).items()):
        if season < _MIN_SEASON:
            continue
        import_season(store, path, season)
        imported.append(season)
    return imported


def _read_optional(path: Path) -> pl.DataFrame | None:
    if not path.exists():
        return None
    return pl.read_csv(path, infer_schema_length=1000)


def _read_gw(path: Path) -> pl.DataFrame:
    try:
        return pl.read_csv(path, infer_schema_length=5000)
    except pl.exceptions.PolarsError:
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


def _as_float(value: object, default: float = 0.0) -> float:
    try:
        return float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return default


def _optional_float(row: dict[str, Any], key: str) -> float | None:
    """Parse an optional float column, or ``None`` when the field is absent.

    Vaastav shipped Opta ``expected_*`` on merged gameweeks from 2022-23
    onwards; earlier seasons omit the columns entirely.
    """
    if key not in row:
        return None
    value = row[key]
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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
    fixtures: pl.DataFrame,
    team_code: dict[int, int],
    season: str,
    start: datetime,
) -> pl.DataFrame:
    """Stage schedule at ``start`` and results at kickoff + 3h.

    Args:
        fixtures: Vaastav ``fixtures.csv`` frame.
        team_code: FPL team id → ``team_key``.
        season: Season label.
        start: Earliest-known schedule stamp (``season_start``).
    """
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
                "available_at": start,
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
    gw: pl.DataFrame,
    codes: dict[int, int],
    season: str,
    start: datetime,
    deadlines: dict[int, datetime],
) -> pl.DataFrame:
    """Stage per-GW prices from vaastav ``value``; news/chance/pens stay unknown."""
    if gw.is_empty() or "element" not in gw.columns:
        return pl.DataFrame(schema=POLARS_SCHEMAS["player_status"])
    has_value = "value" in gw.columns
    has_gw = "GW" in gw.columns
    rows: list[dict[str, Any]] = []
    seen: set[tuple[int, int]] = set()
    for row in gw.iter_rows(named=True):
        element = _as_int(row["element"])
        gameweek = _as_int(row.get("GW"), 0) if has_gw else 0
        if gameweek < 1:
            key = (element, 0)
            if key in seen:
                continue
            seen.add(key)
            rows.append(
                _status_row(
                    season,
                    codes.get(element, element),
                    price=0,
                    available_at=start,
                )
            )
            continue
        key = (element, gameweek)
        if key in seen:
            continue
        seen.add(key)
        price = _as_int(row.get("value"), 0) if has_value else 0
        available = deadlines.get(gameweek)
        if available is None:
            raw_kick = row.get("kickoff_time")
            available = _kickoff(raw_kick) - timedelta(hours=2) if raw_kick else start
        rows.append(
            _status_row(
                season,
                codes.get(element, element),
                price=price,
                available_at=available,
            )
        )
    if not rows:
        return pl.DataFrame(schema=POLARS_SCHEMAS["player_status"])
    return pl.DataFrame(rows).with_columns(
        pl.col("available_at").cast(pl.Datetime("us", "UTC")),
        pl.col("news_added").cast(pl.Datetime("us", "UTC")),
        pl.col("ep_next").cast(pl.Float64),
        pl.col("chance_of_playing").cast(pl.Float64),
        pl.col("chance_of_playing_this_round").cast(pl.Float64),
        pl.col("penalties_order").cast(pl.Int64),
        pl.col("corners_and_indirect_freekicks_order").cast(pl.Int64),
        pl.col("direct_freekicks_order").cast(pl.Int64),
    )


def _status_row(
    season: str, player_key: int, *, price: int, available_at: datetime
) -> dict[str, Any]:
    return {
        "season": season,
        "player_key": player_key,
        "status": "a",
        "news": "",
        "news_added": None,
        "price": price,
        "ep_next": None,
        "chance_of_playing": None,
        "chance_of_playing_this_round": None,
        "penalties_order": None,
        "corners_and_indirect_freekicks_order": None,
        "direct_freekicks_order": None,
        "available_at": available_at,
        "source": "vaastav",
    }


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
                "influence": _as_float(row.get("influence")),
                "creativity": _as_float(row.get("creativity")),
                "threat": _as_float(row.get("threat")),
                "bps": _as_int(row.get("bps")),
                "clearances_blocks_interceptions": _as_int(
                    row.get("clearances_blocks_interceptions")
                ),
                "tackles": _as_int(row.get("tackles")),
                "recoveries": _as_int(row.get("recoveries")),
                "defensive_contribution": _as_int(row.get("defensive_contribution")),
                "expected_goals": _optional_float(row, "expected_goals"),
                "expected_assists": _optional_float(row, "expected_assists"),
                "expected_goal_involvements": _optional_float(
                    row, "expected_goal_involvements"
                ),
                "expected_goals_conceded": _optional_float(
                    row, "expected_goals_conceded"
                ),
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
