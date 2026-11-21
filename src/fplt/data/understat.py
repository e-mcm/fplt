"""Stage Understat match rosters. The caller supplies the id map and the clock."""

from __future__ import annotations

import gzip
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast

import polars as pl
import yaml

from fplt.data.store import DataStore
from fplt.schemas.staged import POLARS_SCHEMAS

_MATCH_LAG = timedelta(hours=3)
_ID_MAPS = (
    Path(__file__).resolve().parents[3] / "configs" / "understat" / "id_maps.yaml"
)


def default_id_maps_path() -> Path:
    """Repo-root path to ``configs/understat/id_maps.yaml``."""
    return _ID_MAPS


def load_id_maps(path: Path) -> tuple[dict[int, int], dict[int, int]]:
    """Load Understat→FPL player and team code maps from YAML.

    Args:
        path: YAML file with ``players`` and ``teams`` integer maps.

    Returns:
        ``(player_keys, team_keys)`` dictionaries. Missing sections become
        empty maps.

    Raises:
        ValueError: The file is not a mapping or a value is not an int.
    """
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError(f"id maps at {path} must be a mapping")
    raw = cast(dict[str, Any], loaded)
    return (
        _int_map(raw.get("players"), "players"),
        _int_map(raw.get("teams"), "teams"),
    )


def season_label_from_understat_year(year: int | str) -> str:
    """Convert Understat's season start year (e.g. ``2024``) to ``2024-25``."""
    start = int(year)
    return f"{start}-{(start + 1) % 100:02d}"


def understat_year_from_season(season: str) -> str:
    """Convert ``2024-25`` to Understat's ``2024`` season token."""
    return season.split("-", 1)[0]


def match_raw_path(root: Path, season: str, match_id: int) -> Path:
    """Write-once path for one Understat ``getMatchData`` payload."""
    return (
        root
        / "understat"
        / f"season={season}"
        / f"match_id={match_id}"
        / "match.json.gz"
    )


def dates_raw_path(root: Path, season: str, captured_at: datetime) -> Path:
    """Write-once path for one Understat league-dates snapshot."""
    stamp = captured_at.astimezone(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    return (
        root
        / "understat"
        / f"season={season}"
        / "dates"
        / f"captured_at={stamp}"
        / "dates.json.gz"
    )


def read_json_gz(path: Path) -> Any:
    """Read one gzipped JSON file."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def available_at_from_match(payload: dict[str, Any]) -> datetime:
    """Kickoff plus three hours — the historical results proxy from the design.

    Args:
        payload: Raw ``getMatchData`` JSON (uses ``tmpl.date``) or a normalised
            match that already carries ``datetime`` / ``available_at``.

    Returns:
        UTC timestamp when the roster is treated as visible.

    Raises:
        KeyError: No usable datetime field is present.
        ValueError: The datetime string cannot be parsed.
    """
    if "available_at" in payload:
        value = payload["available_at"]
        if isinstance(value, datetime):
            return value.astimezone(UTC) if value.tzinfo else value.replace(tzinfo=UTC)
        return _parse_understat_datetime(str(cast(object, value)))
    if "datetime" in payload:
        return (
            _parse_understat_datetime(str(cast(object, payload["datetime"])))
            + _MATCH_LAG
        )
    tmpl = payload.get("tmpl")
    if isinstance(tmpl, dict) and "date" in tmpl:
        return _parse_understat_datetime(str(cast(object, tmpl["date"]))) + _MATCH_LAG
    raise KeyError("match payload has no tmpl.date, datetime, or available_at")


def normalize_match(match_id: int, payload: dict[str, Any]) -> dict[str, Any]:
    """Turn a ``getMatchData`` payload into the ``stage_understat`` match shape.

    Already-normalised matches (``team_h`` plus ``h`` / ``a`` player lists) are
    returned with ``id`` forced to ``match_id``.

    Args:
        match_id: Understat match id (also the directory name on disk).
        payload: Raw AJAX JSON or a normalised match dict.

    Returns:
        Dict with ``id``, ``team_h``, ``team_a``, ``h``, and ``a``.

    Raises:
        KeyError: Required roster / tmpl fields are missing.
        ValueError: A side's roster is neither a dict nor a list.
    """
    if _is_normalised(payload):
        return {
            "id": match_id,
            "team_h": int(payload["team_h"]),
            "team_a": int(payload["team_a"]),
            "h": list(cast(list[Any], payload["h"])),
            "a": list(cast(list[Any], payload["a"])),
        }
    tmpl = cast(dict[str, Any], payload["tmpl"])
    rosters = cast(dict[str, Any], payload["rosters"])
    return {
        "id": match_id,
        "team_h": int(tmpl["h"]),
        "team_a": int(tmpl["a"]),
        "h": _side_players(rosters, "h"),
        "a": _side_players(rosters, "a"),
    }


def iter_raw_matches(
    raw_root: Path, seasons: list[str] | None = None
) -> list[tuple[Path, str, int]]:
    """List write-once Understat match files under ``raw_root/understat``.

    Args:
        raw_root: Directory that corresponds to ``$FPLT_DATA_DIR/raw``.
        seasons: Optional season labels to include; ``None`` means all.

    Returns:
        ``(path, season, match_id)`` triples sorted by season then match id.
    """
    base = raw_root / "understat"
    if not base.exists():
        return []
    found: list[tuple[Path, str, int]] = []
    for path in sorted(base.glob("season=*/match_id=*/match.json.gz")):
        season = path.parents[1].name.removeprefix("season=")
        if seasons is not None and season not in seasons:
            continue
        match_id = int(path.parent.name.removeprefix("match_id="))
        found.append((path, season, match_id))
    return found


def ingest_understat(
    store: DataStore,
    raw_root: Path,
    player_keys: dict[int, int],
    team_keys: dict[int, int],
    *,
    seasons: list[str] | None = None,
) -> int:
    """Stage every raw Understat match file. Return how many matches were staged.

    Args:
        store: Destination ``DataStore``.
        raw_root: ``$FPLT_DATA_DIR/raw``.
        player_keys: Understat player id → ``player_key``.
        team_keys: Understat team id → ``team_key``.
        seasons: Optional season filter.

    Returns:
        Number of match files consumed.
    """
    count = 0
    for path, season, match_id in iter_raw_matches(raw_root, seasons):
        payload = cast(dict[str, Any], read_json_gz(path))
        match = normalize_match(match_id, payload)
        available_at = available_at_from_match(payload)
        stage_understat(store, [match], season, available_at, player_keys, team_keys)
        count += 1
    return count


def stage_understat(
    store: DataStore,
    matches: list[dict[str, Any]],
    season: str,
    available_at: datetime,
    player_keys: dict[int, int],
    team_keys: dict[int, int],
) -> pl.DataFrame:
    """Append player-match xG and xA.

    Args:
        store: Destination for ``understat_player_matches``.
        matches: Understat match rosters. Each has ``id``, ``team_h``,
            ``team_a``, and ``h`` / ``a`` player lists with ``id``, ``xg``,
            ``xa``, ``time``, and optionally ``shots``, ``key_passes``,
            ``xg_chain``, ``xg_buildup``.
        season: Season label stored on every row.
        available_at: When this roster became visible. Later deadlines can
            see it; earlier ones cannot.
        player_keys: Understat player id to ``player_key``. Missing ids are
            stored with a null ``player_key``.
        team_keys: Understat team id to ``team_key``.

    Returns:
        The staged table after the append.

    Raises:
        KeyError: A side's team id has no ``team_key``.
    """
    rows: list[dict[str, object]] = []
    for match in matches:
        match_id = int(match["id"])
        for side, team_field in (("h", "team_h"), ("a", "team_a")):
            team_id = int(match[team_field])
            if team_id not in team_keys:
                raise KeyError(f"understat team {team_id} has no team_key")
            players = match[side]
            if not isinstance(players, list):
                raise ValueError(f"match {match_id} side {side} must be a list")
            for player in cast(list[dict[str, Any]], players):
                understat_id = int(player["id"])
                rows.append(
                    {
                        "season": season,
                        "match_id": match_id,
                        "understat_player_id": understat_id,
                        "player_key": player_keys.get(understat_id),
                        "team_key": team_keys[team_id],
                        "xg": float(player["xg"]),
                        "xa": float(player["xa"]),
                        "shots": int(player.get("shots", 0) or 0),
                        "key_passes": int(player.get("key_passes", 0) or 0),
                        "xg_chain": float(player.get("xg_chain", 0) or 0),
                        "xg_buildup": float(player.get("xg_buildup", 0) or 0),
                        "minutes": int(player["time"]),
                        "available_at": available_at,
                        "source": "understat",
                    }
                )
    frame = pl.DataFrame(rows) if rows else _empty()
    return store.write(
        "understat_player_matches",
        frame.with_columns(pl.col("player_key").cast(pl.Int64())),
    )


def _int_map(value: object, label: str) -> dict[int, int]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"{label} map must be a mapping")
    mapping = cast(dict[object, object], value)
    out: dict[int, int] = {}
    for key, mapped in mapping.items():
        out[int(cast(Any, key))] = int(cast(Any, mapped))
    return out


def _parse_understat_datetime(value: str) -> datetime:
    cleaned = value.strip().replace("Z", "+00:00")
    if "T" in cleaned or "+" in cleaned:
        parsed = datetime.fromisoformat(cleaned)
    else:
        parsed = datetime.strptime(cleaned, "%Y-%m-%d %H:%M:%S")
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def _is_normalised(payload: dict[str, Any]) -> bool:
    return "team_h" in payload and "h" in payload and "a" in payload


def _side_players(rosters: dict[str, Any], side: str) -> list[dict[str, Any]]:
    entries: object = rosters.get(side) or {}
    raw_players: list[object]
    if isinstance(entries, dict):
        raw_players = list(cast(dict[object, object], entries).values())
    elif isinstance(entries, list):
        raw_players = list(cast(list[object], entries))
    else:
        raise ValueError(f"rosters.{side} must be a dict or list")
    rows: list[dict[str, Any]] = []
    for item in raw_players:
        player = cast(dict[str, Any], item)
        rows.append(
            {
                "id": int(player["player_id"]),
                "xg": float(player["xG"]),
                "xa": float(player["xA"]),
                "shots": int(_roster_number(player, "shots")),
                "key_passes": int(_roster_number(player, "key_passes")),
                "xg_chain": float(
                    _roster_number(player, "xGChain", "xgChain", "xg_chain")
                ),
                "xg_buildup": float(
                    _roster_number(player, "xGBuildup", "xgBuildup", "xg_buildup")
                ),
                "time": int(player["time"]),
            }
        )
    return rows


def _roster_number(player: dict[str, Any], *keys: str) -> float:
    """Read a numeric roster field under any of the known Understat key spellings."""
    for key in keys:
        if key in player and player[key] is not None and player[key] != "":
            return float(player[key])
    return 0.0


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=POLARS_SCHEMAS["understat_player_matches"])
