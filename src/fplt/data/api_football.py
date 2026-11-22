"""Stage API-Football injuries. The caller supplies id maps."""

from __future__ import annotations

import gzip
import json
import logging
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl
import yaml

from fplt.data.store import DataStore
from fplt.schemas.staged import POLARS_SCHEMAS

_ID_MAPS = (
    Path(__file__).resolve().parents[3] / "configs" / "api_football" / "id_maps.yaml"
)
_LOG = logging.getLogger(__name__)
_CAPTURED_RE = re.compile(r"captured_at=(\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}Z)")
_SOURCE = "api_football"


def default_id_maps_path() -> Path:
    """Repo-root path to ``configs/api_football/id_maps.yaml``."""
    return _ID_MAPS


def load_id_maps(path: Path) -> tuple[dict[int, int], dict[int, int]]:
    """Load API-Football→FPL player and team code maps from YAML.

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


def season_label_from_api_year(year: int | str) -> str:
    """Convert API-Football season start year (e.g. ``2024``) to ``2024-25``."""
    start = int(year)
    return f"{start}-{(start + 1) % 100:02d}"


def api_year_from_season(season: str) -> int:
    """Convert ``2024-25`` to API-Football's ``2024`` season year."""
    return int(season.split("-", 1)[0])


def injuries_raw_path(root: Path, season: str, captured_at: datetime) -> Path:
    """Write-once path for one API-Football injuries snapshot."""
    stamp = captured_at.astimezone(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    return (
        root
        / "api_football"
        / f"season={season}"
        / "injuries"
        / f"captured_at={stamp}"
        / "injuries.json.gz"
    )


def read_json_gz(path: Path) -> Any:
    """Read one gzipped JSON file."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def available_at_from_path(path: Path) -> datetime:
    """Parse ``available_at`` from a write-once injuries path.

    Args:
        path: ``.../captured_at=YYYY-MM-DDTHH-MM-SSZ/injuries.json.gz``.

    Returns:
        UTC capture timestamp used as PIT ``available_at``.

    Raises:
        ValueError: The path has no parseable ``captured_at`` stamp.
    """
    match = _CAPTURED_RE.search(path.as_posix())
    if match is None:
        raise ValueError(f"no captured_at stamp in {path}")
    parts = match.group(1)
    iso = f"{parts[0:10]}T{parts[11:13]}:{parts[14:16]}:{parts[17:19]}+00:00"
    return datetime.fromisoformat(iso).astimezone(UTC)


def categorize_injury(injury_type: str, reason: str) -> str:
    """Map API type/reason text onto a gate category.

    ``out`` forces a zero minutes gate (suspensions / missing fixture).
    ``doubtful`` soft-downgrades toward the minutes ``doubtful_chance`` when
    FPL tags still look available.
    """
    combined = f"{injury_type} {reason}".strip().lower()
    if not combined:
        return "doubtful"
    hard = (
        "suspen",
        "missing",
        "red card",
        "banned",
        "ineligible",
    )
    if any(token in combined for token in hard):
        return "out"
    return "doubtful"


def iter_raw_injuries(
    raw_root: Path, seasons: list[str] | None = None
) -> list[tuple[Path, str]]:
    """List write-once injuries snapshots under ``raw_root/api_football``.

    Args:
        raw_root: Directory that corresponds to ``$FPLT_DATA_DIR/raw``.
        seasons: Optional season labels to include; ``None`` means all.

    Returns:
        ``(path, season)`` pairs sorted by season then capture stamp.
    """
    base = raw_root / "api_football"
    if not base.exists():
        return []
    found: list[tuple[Path, str]] = []
    for path in sorted(base.glob("season=*/injuries/captured_at=*/injuries.json.gz")):
        season = path.parents[2].name.removeprefix("season=")
        if seasons is not None and season not in seasons:
            continue
        found.append((path, season))
    return found


def ingest_api_football(
    store: DataStore,
    raw_root: Path,
    player_keys: dict[int, int],
    team_keys: dict[int, int],
    *,
    seasons: list[str] | None = None,
) -> int:
    """Stage raw API-Football injuries snapshots into ``player_injuries``.

    Args:
        store: Destination ``DataStore``.
        raw_root: ``$FPLT_DATA_DIR/raw``.
        player_keys: API-Football player id → ``player_key``.
        team_keys: API-Football team id → ``team_key``.
        seasons: Optional season filter.

    Returns:
        Number of snapshot files consumed.
    """
    count = 0
    unmapped_players: set[int] = set()
    unmapped_teams: set[int] = set()
    for path, season in iter_raw_injuries(raw_root, seasons):
        payload = cast(dict[str, Any], read_json_gz(path))
        available_at = available_at_from_path(path)
        _, players, teams = stage_player_injuries(
            store,
            payload,
            season,
            available_at,
            player_keys,
            team_keys,
        )
        unmapped_players.update(players)
        unmapped_teams.update(teams)
        count += 1
    if unmapped_players:
        sample = sorted(unmapped_players)[:20]
        _LOG.warning(
            "api_football: %d unmapped player id(s); sample=%s "
            "(see configs/api_football/id_maps.yaml)",
            len(unmapped_players),
            sample,
        )
    if unmapped_teams:
        sample = sorted(unmapped_teams)[:20]
        _LOG.warning(
            "api_football: %d unmapped team id(s); sample=%s "
            "(see configs/api_football/id_maps.yaml)",
            len(unmapped_teams),
            sample,
        )
    return count


def stage_player_injuries(
    store: DataStore,
    payload: dict[str, Any],
    season: str,
    available_at: datetime,
    player_keys: dict[int, int],
    team_keys: dict[int, int],
) -> tuple[pl.DataFrame, set[int], set[int]]:
    """Append one injuries snapshot to ``player_injuries``.

    Args:
        store: Destination for ``player_injuries``.
        payload: Raw ``/injuries`` JSON (``response`` list) or a bare list.
        season: Season label stored on every row.
        available_at: Capture time — PIT visibility stamp.
        player_keys: API player id → ``player_key``. Missing ids stage with
            a null ``player_key``.
        team_keys: API team id → ``team_key``. Missing ids stage with a null
            ``team_key`` (unlike Understat teams, which reject).

    Returns:
        ``(staged_frame, unmapped_player_ids, unmapped_team_ids)``.
    """
    rows, unmapped_players, unmapped_teams = rows_from_injuries_payload(
        payload,
        season=season,
        available_at=available_at,
        player_keys=player_keys,
        team_keys=team_keys,
    )
    frame = pl.DataFrame(rows) if rows else _empty()
    if not rows:
        return store.write("player_injuries", frame), unmapped_players, unmapped_teams
    typed = frame.with_columns(
        pl.col("player_key").cast(pl.Int64()),
        pl.col("team_key").cast(pl.Int64()),
        pl.col("fixture_id").cast(pl.Int64()),
    )
    return store.write("player_injuries", typed), unmapped_players, unmapped_teams


def rows_from_injuries_payload(
    payload: dict[str, Any] | list[Any],
    *,
    season: str,
    available_at: datetime,
    player_keys: dict[int, int],
    team_keys: dict[int, int],
) -> tuple[list[dict[str, object]], set[int], set[int]]:
    """Normalise an ``/injuries`` payload into staged row dicts.

    Args:
        payload: Full API envelope or a bare ``response`` list.
        season: FPL season label.
        available_at: Capture UTC timestamp.
        player_keys: API→FPL player map.
        team_keys: API→FPL team map.

    Returns:
        Row dicts plus the sets of unmapped API player and team ids.
    """
    entries = _response_entries(payload)
    rows: list[dict[str, object]] = []
    unmapped_players: set[int] = set()
    unmapped_teams: set[int] = set()
    for raw_entry in entries:
        if not isinstance(raw_entry, dict):
            continue
        entry = cast(dict[str, Any], raw_entry)
        player = cast(dict[str, Any], entry.get("player") or {})
        team = cast(dict[str, Any], entry.get("team") or {})
        fixture = cast(dict[str, Any], entry.get("fixture") or {})
        if "id" not in player:
            continue
        api_player_id = int(player["id"])
        api_team_id = int(team["id"]) if team.get("id") is not None else None
        api_fixture_id = int(fixture["id"]) if fixture.get("id") is not None else 0
        player_key = player_keys.get(api_player_id)
        if player_key is None:
            unmapped_players.add(api_player_id)
        team_key: int | None = None
        if api_team_id is not None:
            team_key = team_keys.get(api_team_id)
            if team_key is None:
                unmapped_teams.add(api_team_id)
        injury_type = str(cast(object, entry.get("type") or ""))
        reason = str(cast(object, entry.get("reason") or ""))
        rows.append(
            {
                "season": season,
                "player_key": player_key,
                "api_player_id": api_player_id,
                "team_key": team_key,
                "api_team_id": 0 if api_team_id is None else api_team_id,
                "api_fixture_id": api_fixture_id,
                "fixture_id": None,
                "injury_type": injury_type,
                "reason": reason,
                "category": categorize_injury(injury_type, reason),
                "available_at": available_at,
                "source": _SOURCE,
            }
        )
    return rows, unmapped_players, unmapped_teams


def _response_entries(payload: dict[str, Any] | list[Any]) -> list[object]:
    if isinstance(payload, list):
        return cast(list[object], payload)
    response = payload.get("response", [])
    if isinstance(response, list):
        return cast(list[object], response)
    raise ValueError("injuries payload response must be a list")


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


def _empty() -> pl.DataFrame:
    return pl.DataFrame(schema=POLARS_SCHEMAS["player_injuries"])
