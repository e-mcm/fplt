"""Predicted XI ingest and staging (Touchline Wire; optional API-Football).

Near-deadline lineup overrides land in staged ``predicted_lineups`` with
``available_at = capture time``. Models read only via ``PITView``.

Touchline Wire is experimental (zero-auth third party; MIT-with-attribution).
Live fetch is gated by ``FPLT_TOUCHLINE_LINEUPS_ENABLED`` and rate-limited.
Do not scrape Fantasy Football Scout HTML or Twitter/X.
"""

from __future__ import annotations

import gzip
import json
import logging
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import httpx
import polars as pl

from fplt.data.store import DataStore
from fplt.schemas.staged import POLARS_SCHEMAS

_LOG = logging.getLogger(__name__)

SOURCE_TOUCHLINE = "touchline_wire"
SOURCE_API_FOOTBALL = "api_football"
ROLE_START = "start"
ROLE_BENCH = "bench"
ROLE_OUT = "out"
_ROLES = frozenset({ROLE_START, ROLE_BENCH, ROLE_OUT})

TOUCHLINE_LINEUPS_URL = "https://touchlinewire.com/api/v1/lineups"
TOUCHLINE_DOCS_URL = "https://touchlinewire.com/docs"
TOUCHLINE_USER_AGENT = (
    "fplt/0.1 (+https://github.com/e-mcm/fplt; OSS; attribution: Touchline Wire)"
)
# Conservative default: one full-slate pull per five minutes.
DEFAULT_MIN_INTERVAL_SECONDS = 300

_last_touchline_fetch_monotonic: float | None = None


class PredictedLineupsDisabled(RuntimeError):
    """Live Touchline fetch refused because the feature flag is off."""


class PredictedLineupsRateLimit(RuntimeError):
    """Live Touchline fetch refused because the min interval has not elapsed."""


def touchline_raw_path(raw_root: Path, season: str, captured_at: datetime) -> Path:
    """Write-once path for one Touchline Wire lineups snapshot."""
    return (
        raw_root
        / "touchline"
        / f"season={season}"
        / f"captured_at={_stamp(captured_at)}"
        / "lineups.json.gz"
    )


def api_football_lineups_raw_path(
    raw_root: Path, season: str, captured_at: datetime
) -> Path:
    """Write-once path for one API-Football fixtures/lineups snapshot.

    Package B (injuries) may share ``raw/api_football/``; lineups stay under
    ``lineups/`` so both PRs can land without colliding.
    """
    return (
        raw_root
        / "api_football"
        / f"season={season}"
        / "lineups"
        / f"captured_at={_stamp(captured_at)}"
        / "lineups.json.gz"
    )


def write_json_gz(path: Path, payload: object) -> bool:
    """Write gzipped JSON when ``path`` does not already exist.

    Args:
        path: Destination under ``data/raw/``.
        payload: JSON-serialisable object.

    Returns:
        True when a new file was written.
    """
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(payload, handle, separators=(",", ":"))
    return True


def read_json_gz(path: Path) -> Any:
    """Read one gzipped JSON file."""
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        return json.load(handle)


def fetch_touchline_lineups(
    *,
    client: httpx.Client | None = None,
    enabled: bool = False,
    min_interval_seconds: int = DEFAULT_MIN_INTERVAL_SECONDS,
    url: str = TOUCHLINE_LINEUPS_URL,
) -> dict[str, Any]:
    """GET Touchline Wire ``/api/v1/lineups`` with attribution and rate limit.

    Args:
        client: Optional shared ``httpx.Client``.
        enabled: Must be true (``FPLT_TOUCHLINE_LINEUPS_ENABLED``).
        min_interval_seconds: Minimum gap between live fetches in this process.
        url: Override for tests.

    Returns:
        Parsed JSON object with ``meta`` and ``data``.

    Raises:
        PredictedLineupsDisabled: ``enabled`` is false.
        PredictedLineupsRateLimit: Called again too soon.
        httpx.HTTPError: Transport or HTTP failure.
        ValueError: Payload is not a JSON object.
    """
    global _last_touchline_fetch_monotonic
    if not enabled:
        raise PredictedLineupsDisabled(
            "Touchline Wire lineups fetch is disabled; set "
            "FPLT_TOUCHLINE_LINEUPS_ENABLED=1 after reviewing ToS/accuracy risk"
        )
    now = time.monotonic()
    last = _last_touchline_fetch_monotonic
    if last is not None and (now - last) < min_interval_seconds:
        wait = min_interval_seconds - (now - last)
        raise PredictedLineupsRateLimit(
            f"Touchline Wire rate limit: wait {wait:.0f}s "
            f"(min interval {min_interval_seconds}s; {TOUCHLINE_DOCS_URL})"
        )
    headers = {
        "User-Agent": TOUCHLINE_USER_AGENT,
        "Accept": "application/json",
    }
    owns_client = client is None
    http = client or httpx.Client(timeout=30.0, headers=headers)
    try:
        if not owns_client:
            http.headers.update(headers)
        response = http.get(url)
        response.raise_for_status()
        payload = response.json()
    finally:
        if owns_client:
            http.close()
    if not isinstance(payload, dict):
        raise ValueError("Touchline Wire lineups payload must be a JSON object")
    _last_touchline_fetch_monotonic = time.monotonic()
    typed = cast(dict[str, Any], payload)
    meta_raw = typed.get("meta")
    if isinstance(meta_raw, dict):
        meta = cast(dict[str, Any], meta_raw)
        _LOG.info(
            "fetched Touchline Wire lineups count=%s license=%s attribution=%s",
            meta.get("count"),
            meta.get("license"),
            meta.get("attribution_url") or TOUCHLINE_DOCS_URL,
        )
    return typed


def capture_touchline_lineups(
    raw_root: Path,
    season: str,
    *,
    enabled: bool = False,
    min_interval_seconds: int = DEFAULT_MIN_INTERVAL_SECONDS,
    captured_at: datetime | None = None,
    client: httpx.Client | None = None,
    url: str = TOUCHLINE_LINEUPS_URL,
) -> tuple[Path, bool]:
    """Fetch Touchline Wire lineups and write them write-once under ``raw_root``.

    Args:
        raw_root: ``$FPLT_DATA_DIR/raw``.
        season: FPL season label for the path.
        enabled: Feature flag for live HTTP.
        min_interval_seconds: Process-local rate limit.
        captured_at: Capture stamp (default: now UTC).
        client: Optional shared HTTP client.
        url: Override for tests.

    Returns:
        ``(path, written)`` where ``written`` is False if the path existed.
    """
    stamp = captured_at or datetime.now(tz=UTC)
    if stamp.tzinfo is None:
        stamp = stamp.replace(tzinfo=UTC)
    else:
        stamp = stamp.astimezone(UTC)
    path = touchline_raw_path(raw_root, season, stamp)
    if path.exists():
        return path, False
    payload = fetch_touchline_lineups(
        client=client,
        enabled=enabled,
        min_interval_seconds=min_interval_seconds,
        url=url,
    )
    written = write_json_gz(path, payload)
    return path, written


def iter_touchline_raw(
    raw_root: Path, seasons: list[str] | None = None
) -> list[tuple[Path, str, datetime]]:
    """List write-once Touchline lineups files, oldest first."""
    base = raw_root / "touchline"
    if not base.exists():
        return []
    found: list[tuple[Path, str, datetime]] = []
    for path in sorted(base.glob("season=*/captured_at=*/lineups.json.gz")):
        season = path.parents[1].name.removeprefix("season=")
        if seasons is not None and season not in seasons:
            continue
        captured = _parse_stamp(path.parent.name.removeprefix("captured_at="))
        found.append((path, season, captured))
    return found


def iter_api_football_lineups_raw(
    raw_root: Path, seasons: list[str] | None = None
) -> list[tuple[Path, str, datetime]]:
    """List write-once API-Football lineups files, oldest first."""
    base = raw_root / "api_football"
    if not base.exists():
        return []
    found: list[tuple[Path, str, datetime]] = []
    for path in sorted(base.glob("season=*/lineups/captured_at=*/lineups.json.gz")):
        season = path.parents[2].name.removeprefix("season=")
        if seasons is not None and season not in seasons:
            continue
        captured = _parse_stamp(path.parent.name.removeprefix("captured_at="))
        found.append((path, season, captured))
    return found


def stage_touchline_payload(
    store: DataStore,
    payload: dict[str, Any],
    season: str,
    available_at: datetime,
    *,
    source: str = SOURCE_TOUCHLINE,
) -> pl.DataFrame:
    """Parse a Touchline Wire lineups payload into ``predicted_lineups``.

    Maps ``team_id`` → ``team_key`` and element ``id`` → ``player_key`` via
    staged ``teams`` / ``players``. Attaches each club's next unfinished
    fixture id(s) when fixtures are staged.

    Args:
        store: Destination store (also read for id maps + fixtures).
        payload: Touchline JSON object.
        season: FPL season label.
        available_at: PIT stamp (capture time).
        source: Provenance string.

    Returns:
        Rows written (may be empty).
    """
    clubs = payload.get("data")
    if clubs is None:
        return pl.DataFrame(schema=POLARS_SCHEMAS["predicted_lineups"])
    club_rows = _object_list(clubs, label="Touchline Wire data")

    team_id_to_key = _team_id_map(store, season)
    element_to_player = _element_id_map(store, season)
    upcoming = _upcoming_fixtures_by_team(store, season, available_at)

    rows: list[dict[str, object]] = []
    for club in club_rows:
        team_id = club.get("team_id")
        if team_id is None:
            continue
        team_key = team_id_to_key.get(int(cast(int | str, team_id)))
        if team_key is None:
            _LOG.warning("skip Touchline club team_id=%s: no team_key", team_id)
            continue
        fixture_ids = upcoming.get(team_key, [None])
        club_confidence = _as_unit_confidence(club.get("tactical_confidence_rate"))
        for role, key in ((ROLE_START, "starters"), (ROLE_BENCH, "bench")):
            players = _dict_list(club.get(key))
            for player in players:
                element_raw = player.get("id")
                if element_raw is None:
                    continue
                element_id = int(cast(int | str, element_raw))
                player_key = element_to_player.get(element_id)
                likelihood = player.get("likelihood")
                if isinstance(likelihood, int | float) and not isinstance(
                    likelihood, bool
                ):
                    confidence: float | None = float(likelihood) / 100.0
                else:
                    confidence = club_confidence
                for fixture_id in fixture_ids:
                    rows.append(
                        {
                            "season": season,
                            "team_key": team_key,
                            "fixture_id": fixture_id,
                            "player_key": player_key,
                            "element_id": element_id,
                            "role": role,
                            "confidence": confidence,
                            "available_at": available_at,
                            "source": source,
                        }
                    )
    if not rows:
        return pl.DataFrame(schema=POLARS_SCHEMAS["predicted_lineups"])
    # Drop rows with no player_key — surname-only join is forbidden.
    frame = pl.DataFrame(rows).filter(pl.col("player_key").is_not_null())
    if frame.is_empty():
        return pl.DataFrame(schema=POLARS_SCHEMAS["predicted_lineups"])
    return store.write("predicted_lineups", frame)


def stage_api_football_lineups_payload(
    store: DataStore,
    payload: dict[str, Any] | list[dict[str, Any]],
    season: str,
    available_at: datetime,
    *,
    player_id_map: dict[int, int],
    team_id_map: dict[int, int],
    fixture_id_map: dict[int, int] | None = None,
    source: str = SOURCE_API_FOOTBALL,
) -> pl.DataFrame:
    """Stage API-Football ``/fixtures/lineups``-shaped JSON into the same table.

    Kept as a pure staging adapter so Package B (injuries) and this PR can
    share ``raw/api_football/`` without coupling HTTP clients. Caller supplies
    id maps (API-Football id → FPL ``player_key`` / ``team_key`` / fixture).

    Args:
        store: Destination store.
        payload: One response object or a list of fixture lineup objects.
        season: FPL season label.
        available_at: PIT stamp.
        player_id_map: API-Football player id → ``player_key``.
        team_id_map: API-Football team id → ``team_key``.
        fixture_id_map: API-Football fixture id → FPL ``fixture_id``.
        source: Provenance string.

    Returns:
        Rows written (may be empty).
    """
    entries = _api_football_entries(payload)
    rows: list[dict[str, object]] = []
    for entry in entries:
        fixture_block = _as_dict(entry.get("fixture"))
        api_fixture = entry.get("fixture_id")
        if api_fixture is None and fixture_block is not None:
            api_fixture = fixture_block.get("id")
        fpl_fixture = None
        if api_fixture is not None and fixture_id_map is not None:
            fpl_fixture = fixture_id_map.get(int(cast(int | str, api_fixture)))
        team_block = _as_dict(entry.get("team"))
        if team_block is None:
            continue
        api_team = team_block.get("id")
        if api_team is None:
            continue
        team_key = team_id_map.get(int(cast(int | str, api_team)))
        if team_key is None:
            continue
        for role, key in ((ROLE_START, "startXI"), (ROLE_BENCH, "substitutes")):
            for item in _dict_list(entry.get(key)):
                nested = _as_dict(item.get("player"))
                player = nested if nested is not None else item
                api_player = player.get("id")
                if api_player is None:
                    continue
                player_key = player_id_map.get(int(cast(int | str, api_player)))
                if player_key is None:
                    continue
                rows.append(
                    {
                        "season": season,
                        "team_key": team_key,
                        "fixture_id": fpl_fixture,
                        "player_key": player_key,
                        "element_id": None,
                        "role": role,
                        "confidence": None,
                        "available_at": available_at,
                        "source": source,
                    }
                )
    if not rows:
        return pl.DataFrame(schema=POLARS_SCHEMAS["predicted_lineups"])
    return store.write("predicted_lineups", pl.DataFrame(rows))


def ingest_predicted_lineups(
    store: DataStore,
    raw_root: Path,
    *,
    seasons: list[str] | None = None,
) -> int:
    """Stage all raw Touchline (+ API-Football lineups) snapshots.

    Args:
        store: Destination store.
        raw_root: ``$FPLT_DATA_DIR/raw``.
        seasons: Optional season filter.

    Returns:
        Number of raw files consumed.
    """
    count = 0
    for path, season, captured in iter_touchline_raw(raw_root, seasons):
        loaded = read_json_gz(path)
        if not isinstance(loaded, dict):
            raise ValueError(f"{path} must contain a JSON object")
        stage_touchline_payload(store, cast(dict[str, Any], loaded), season, captured)
        count += 1
    for path, season, captured in iter_api_football_lineups_raw(raw_root, seasons):
        loaded = read_json_gz(path)
        # API-Football staging needs maps; skip silently when maps are empty
        # so Package B can land maps later without breaking this ingest.
        if isinstance(loaded, list):
            payload: dict[str, Any] | list[dict[str, Any]] = _dict_list(
                cast(list[object], loaded)
            )
        elif isinstance(loaded, dict):
            payload = cast(dict[str, Any], loaded)
        else:
            raise ValueError(f"{path} must contain a JSON object or list")
        stage_api_football_lineups_payload(
            store,
            payload,
            season,
            captured,
            player_id_map={},
            team_id_map={},
        )
        count += 1
    return count


def score_predicted_starts(
    predicted: pl.DataFrame, actual_stats: pl.DataFrame
) -> pl.DataFrame:
    """Compare predicted ``start`` roles to official ``starts`` labels.

    Args:
        predicted: ``PredictedLineupsSchema`` rows (any capture).
        actual_stats: ``player_fixture_stats`` with ``starts`` and keys
            ``season``, ``player_key``, ``fixture_id``.

    Returns:
        One row per predicted start with ``actual_started`` and ``correct``.
        Empty when either input is empty or ``fixture_id`` cannot join.
    """
    if predicted.is_empty() or actual_stats.is_empty():
        return pl.DataFrame(
            schema={
                "season": pl.Utf8(),
                "fixture_id": pl.Int64(),
                "player_key": pl.Int64(),
                "source": pl.Utf8(),
                "actual_started": pl.Boolean(),
                "correct": pl.Boolean(),
            }
        )
    starts = predicted.filter(
        (pl.col("role") == ROLE_START)
        & pl.col("fixture_id").is_not_null()
        & pl.col("player_key").is_not_null()
    )
    if starts.is_empty():
        return pl.DataFrame(
            schema={
                "season": pl.Utf8(),
                "fixture_id": pl.Int64(),
                "player_key": pl.Int64(),
                "source": pl.Utf8(),
                "actual_started": pl.Boolean(),
                "correct": pl.Boolean(),
            }
        )
    label_cols = ["season", "player_key", "fixture_id", "starts"]
    if "available_at" in actual_stats.columns:
        labels = (
            actual_stats.sort("available_at")
            .unique(subset=["season", "player_key", "fixture_id"], keep="last")
            .select(label_cols)
        )
    else:
        labels = actual_stats.select(label_cols).unique(
            subset=["season", "player_key", "fixture_id"], keep="last"
        )
    labels = labels.with_columns((pl.col("starts") > 0).alias("actual_started"))
    return (
        starts.join(labels, on=["season", "player_key", "fixture_id"], how="inner")
        .with_columns(pl.col("actual_started").alias("correct"))
        .select(
            "season",
            "fixture_id",
            "player_key",
            "source",
            "actual_started",
            "correct",
        )
    )


def reset_touchline_rate_limit() -> None:
    """Clear the process-local Touchline fetch timestamp (tests only)."""
    global _last_touchline_fetch_monotonic
    _last_touchline_fetch_monotonic = None


def _stamp(value: datetime) -> str:
    stamp = value.astimezone(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")
    return stamp


def _parse_stamp(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%dT%H-%M-%SZ").replace(tzinfo=UTC)


def _as_unit_confidence(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        return None
    number = float(value)
    if number > 1.0:
        number = number / 100.0
    if not 0.0 <= number <= 1.0:
        return None
    return number


def _team_id_map(store: DataStore, season: str) -> dict[int, int]:
    teams = (
        store.table("teams")
        .filter(pl.col("season") == season)
        .sort("available_at")
        .unique(subset=["team_id"], keep="last")
        .collect()
    )
    return {
        int(team_id): int(team_key)
        for team_id, team_key in zip(
            teams["team_id"].to_list(), teams["team_key"].to_list(), strict=True
        )
    }


def _element_id_map(store: DataStore, season: str) -> dict[int, int]:
    players = (
        store.table("players")
        .filter(pl.col("season") == season)
        .sort("available_at")
        .unique(subset=["element_id"], keep="last")
        .collect()
    )
    return {
        int(element_id): int(player_key)
        for element_id, player_key in zip(
            players["element_id"].to_list(),
            players["player_key"].to_list(),
            strict=True,
        )
    }


def _upcoming_fixtures_by_team(
    store: DataStore, season: str, as_of: datetime
) -> dict[int, list[int | None]]:
    """Next unfinished fixture id(s) per team at ``as_of`` (DGW → several)."""
    fixtures = (
        store.table("fixtures")
        .filter(
            (pl.col("season") == season)
            & (pl.col("available_at") <= as_of)
            & (pl.col("finished") == False)  # noqa: E712
            & pl.col("team_h_score").is_null()
        )
        .sort("available_at")
        .unique(subset=["fixture_id"], keep="last")
        .collect()
    )
    if fixtures.is_empty():
        return {}
    # Prefer the earliest kickoff still ahead of as_of; fall back to any unfinished.
    future = fixtures.filter(pl.col("kickoff_time") >= as_of)
    pool = future if not future.is_empty() else fixtures
    home = pool.select(
        pl.col("team_h_key").alias("team_key"),
        "fixture_id",
        "kickoff_time",
    )
    away = pool.select(
        pl.col("team_a_key").alias("team_key"),
        "fixture_id",
        "kickoff_time",
    )
    combined = pl.concat([home, away]).sort(["team_key", "kickoff_time", "fixture_id"])
    # Keep fixtures that share the earliest kickoff per team (DGW same day).
    first = combined.group_by("team_key").agg(
        pl.col("kickoff_time").min().alias("min_ko")
    )
    linked = (
        combined.join(first, on="team_key", how="inner")
        .filter(pl.col("kickoff_time") == pl.col("min_ko"))
        .select("team_key", "fixture_id")
    )
    out: dict[int, list[int | None]] = {}
    for row in linked.iter_rows(named=True):
        out.setdefault(int(row["team_key"]), []).append(int(row["fixture_id"]))
    return out


def _api_football_entries(
    payload: dict[str, Any] | list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if isinstance(payload, list):
        return list(payload)
    response = payload.get("response")
    if isinstance(response, list):
        return _dict_list(cast(list[object], response))
    data = payload.get("data")
    if isinstance(data, list):
        return _dict_list(cast(list[object], data))
    return [payload]


def _as_dict(value: object) -> dict[str, Any] | None:
    if isinstance(value, dict):
        return cast(dict[str, Any], value)
    return None


def _dict_list(value: object) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    items = cast(list[object], value)
    return [cast(dict[str, Any], row) for row in items if isinstance(row, dict)]


def _object_list(value: object, *, label: str) -> list[dict[str, Any]]:
    if isinstance(value, dict):
        return [cast(dict[str, Any], value)]
    if isinstance(value, list):
        return _dict_list(cast(list[object], value))
    raise ValueError(f"{label} must be a list or object")
