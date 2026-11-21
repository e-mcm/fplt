"""Stage football-data.co.uk E0 closing odds into ``match_odds``.

Closing averages (``AvgCH`` / ``AvgCD`` / ``AvgCA``, plus O/U when present)
are labelled ``odds_kind=closing_proxy`` with ``available_at`` at kickoff.
That is a pre-kickoff proxy, **not** an FPL deadline snapshot.
"""

from __future__ import annotations

import logging
from datetime import UTC, date, datetime, time
from pathlib import Path
from typing import Any, cast
from zoneinfo import ZoneInfo

import polars as pl
import yaml

from fplt.data.store import DataStore
from fplt.schemas.staged import POLARS_SCHEMAS

_LONDON = ZoneInfo("Europe/London")
_TEAM_MAP = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "football_data"
    / "team_names.yaml"
)
_LOG = logging.getLogger(__name__)
_ODDS_KIND = "closing_proxy"
_SOURCE = "football_data"
_DEFAULT_KICKOFF = time(15, 0)

# Prefer closing market averages; fall back to Bet365 closing, then opening avg.
_HOME_COLS = ("AvgCH", "B365CH", "AvgH", "B365H")
_DRAW_COLS = ("AvgCD", "B365CD", "AvgD", "B365D")
_AWAY_COLS = ("AvgCA", "B365CA", "AvgA", "B365A")
# O/U stays closing-only so nullable absence means "no closing total", not open.
_OVER_COLS = ("AvgC>2.5", "B365C>2.5")
_UNDER_COLS = ("AvgC<2.5", "B365C<2.5")


def default_team_map_path() -> Path:
    """Repo-root path to ``configs/football_data/team_names.yaml``."""
    return _TEAM_MAP


def load_team_names(path: Path) -> dict[str, int]:
    """Load football-data team name → FPL ``team_key`` map from YAML.

    Args:
        path: YAML file with a ``teams`` mapping of name → integer code.

    Returns:
        Name → ``team_key`` dictionary. Missing ``teams`` becomes empty.

    Raises:
        ValueError: The file is not a mapping or a value is not an int.
    """
    loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    if loaded is None:
        loaded = {}
    if not isinstance(loaded, dict):
        raise ValueError(f"team map at {path} must be a mapping")
    raw = cast(dict[str, Any], loaded).get("teams", {})
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ValueError(f"teams section in {path} must be a mapping")
    out: dict[str, int] = {}
    for name, code in cast(dict[Any, Any], raw).items():
        if not isinstance(name, str):
            raise ValueError(f"team name keys must be strings, got {name!r}")
        try:
            out[name] = int(code)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"team_key for {name!r} must be an int") from exc
    return out


def season_token(season: str) -> str:
    """Convert ``2024-25`` to football-data's ``2425`` path token."""
    start, end = season.split("-", 1)
    return f"{int(start) % 100:02d}{int(end) % 100:02d}"


def season_label_from_token(token: str) -> str:
    """Convert football-data ``2425`` to ``2024-25``.

    Args:
        token: Four-digit season path token.

    Returns:
        FPL season label.

    Raises:
        ValueError: ``token`` is not four digits.
    """
    if len(token) != 4 or not token.isdigit():
        raise ValueError(
            f"football-data season token must be four digits, got {token!r}"
        )
    start = 2000 + int(token[:2])
    end = int(token[2:])
    return f"{start}-{end:02d}"


def e0_raw_path(raw_root: Path, season: str) -> Path:
    """Write-once path for one E0 CSV under ``raw/football_data``."""
    return raw_root / "football_data" / f"season={season}" / "E0.csv"


def iter_raw_e0(
    raw_root: Path, seasons: list[str] | None = None
) -> list[tuple[Path, str]]:
    """List write-once E0 CSV files under ``raw_root/football_data``."""
    base = raw_root / "football_data"
    if not base.exists():
        return []
    found: list[tuple[Path, str]] = []
    for path in sorted(base.glob("season=*/E0.csv")):
        season = path.parent.name.removeprefix("season=")
        if seasons is not None and season not in seasons:
            continue
        found.append((path, season))
    return found


def parse_kickoff(date_text: str, time_text: str | None) -> datetime:
    """Parse football-data ``Date`` + optional ``Time`` as Europe/London → UTC.

    When ``Time`` is missing, use 15:00 local (traditional English kickoff) so
    ``available_at`` still lands on match day.

    Args:
        date_text: ``DD/MM/YYYY`` or ``DD/MM/YY``.
        time_text: ``HH:MM`` or blank.

    Returns:
        Aware UTC kickoff.

    Raises:
        ValueError: Date or time cannot be parsed.
    """
    match_date = _parse_date(date_text)
    kick = _DEFAULT_KICKOFF
    if time_text is not None and str(time_text).strip():
        parts = str(time_text).strip().split(":")
        if len(parts) != 2:
            raise ValueError(f"bad football-data Time {time_text!r}")
        kick = time(int(parts[0]), int(parts[1]))
    local = datetime.combine(match_date, kick, tzinfo=_LONDON)
    return local.astimezone(UTC)


def parse_e0_csv(path: Path, season: str, team_keys: dict[str, int]) -> pl.DataFrame:
    """Parse one E0 CSV into a ``MatchOddsSchema`` frame (no store write).

    Rows without closing/opening 1X2 odds are skipped. Unknown team names raise.

    Args:
        path: Local E0 CSV.
        season: FPL season label (e.g. ``2024-25``).
        team_keys: football-data ``HomeTeam``/``AwayTeam`` → ``team_key``.

    Returns:
        Typed odds frame ready for ``DataStore.write("match_odds", ...)``.

    Raises:
        KeyError: A side has no entry in ``team_keys``.
    """
    frame = pl.read_csv(path, infer_schema_length=10_000)
    if frame.is_empty():
        return pl.DataFrame(schema=POLARS_SCHEMAS["match_odds"])
    rows: list[dict[str, Any]] = []
    for row in frame.iter_rows(named=True):
        home_name = str(row.get("HomeTeam") or "").strip()
        away_name = str(row.get("AwayTeam") or "").strip()
        if not home_name or not away_name:
            continue
        if home_name not in team_keys:
            raise KeyError(f"football-data team {home_name!r} has no team_key")
        if away_name not in team_keys:
            raise KeyError(f"football-data team {away_name!r} has no team_key")
        home = _first_odds(row, _HOME_COLS)
        draw = _first_odds(row, _DRAW_COLS)
        away = _first_odds(row, _AWAY_COLS)
        if home is None or draw is None or away is None:
            _LOG.debug("skip %s vs %s: missing 1X2 odds", home_name, away_name)
            continue
        date_text = str(row.get("Date") or "").strip()
        time_raw = row.get("Time")
        time_text = None if time_raw is None else str(time_raw)
        kickoff = parse_kickoff(date_text, time_text)
        rows.append(
            {
                "season": season,
                "fixture_id": None,
                "team_h_key": team_keys[home_name],
                "team_a_key": team_keys[away_name],
                "kickoff_time": kickoff,
                "home_odds": home,
                "draw_odds": draw,
                "away_odds": away,
                "over_25_odds": _first_odds(row, _OVER_COLS),
                "under_25_odds": _first_odds(row, _UNDER_COLS),
                "odds_kind": _ODDS_KIND,
                "available_at": kickoff,
                "source": _SOURCE,
            }
        )
    if not rows:
        return pl.DataFrame(schema=POLARS_SCHEMAS["match_odds"])
    return pl.DataFrame(rows, schema=POLARS_SCHEMAS["match_odds"])


def attach_fixture_ids(odds: pl.DataFrame, fixtures: pl.DataFrame) -> pl.DataFrame:
    """Fill nullable ``fixture_id`` from staged FPL fixtures on date + teams.

    Args:
        odds: ``match_odds`` rows (``MatchOddsSchema``).
        fixtures: Staged ``fixtures`` rows.

    Returns:
        Odds with ``fixture_id`` set when a unique date+teams match exists.
    """
    if odds.is_empty() or fixtures.is_empty():
        return odds
    wanted = fixtures.select(
        "season",
        "fixture_id",
        "team_h_key",
        "team_a_key",
        pl.col("kickoff_time").dt.date().alias("match_date"),
    ).unique(
        subset=["season", "team_h_key", "team_a_key", "match_date"],
        keep="last",
    )
    joined = (
        odds.with_columns(pl.col("kickoff_time").dt.date().alias("match_date"))
        .join(
            wanted,
            on=["season", "team_h_key", "team_a_key", "match_date"],
            how="left",
            suffix="_fix",
        )
        .with_columns(
            pl.coalesce(pl.col("fixture_id_fix"), pl.col("fixture_id")).alias(
                "fixture_id"
            )
        )
        .drop("match_date", "fixture_id_fix")
        .select(list(POLARS_SCHEMAS["match_odds"]))
    )
    return joined


def stage_match_odds(
    store: DataStore,
    path: Path,
    season: str,
    team_keys: dict[str, int],
    *,
    join_fixtures: bool = True,
) -> pl.DataFrame:
    """Parse one E0 CSV and append it to ``match_odds``.

    Args:
        store: Destination ``DataStore``.
        path: Local write-once E0 CSV.
        season: FPL season label.
        team_keys: Name → ``team_key`` map.
        join_fixtures: When true, attach ``fixture_id`` from staged fixtures.

    Returns:
        The frame written (``MatchOddsSchema``).
    """
    frame = parse_e0_csv(path, season, team_keys)
    if join_fixtures and not frame.is_empty():
        frame = attach_fixture_ids(frame, store.read("fixtures"))
    if frame.is_empty():
        return frame
    return store.write("match_odds", frame)


def ingest_football_data(
    store: DataStore,
    raw_root: Path,
    team_keys: dict[str, int],
    *,
    seasons: list[str] | None = None,
    join_fixtures: bool = True,
) -> int:
    """Stage all raw E0 CSVs under ``raw_root/football_data``.

    Args:
        store: Destination ``DataStore``.
        raw_root: ``$FPLT_DATA_DIR/raw``.
        team_keys: Name → ``team_key`` map.
        seasons: Optional season filter.
        join_fixtures: When true, attach ``fixture_id`` from staged fixtures.

    Returns:
        Number of E0 CSV files consumed.
    """
    count = 0
    for path, season in iter_raw_e0(raw_root, seasons):
        stage_match_odds(store, path, season, team_keys, join_fixtures=join_fixtures)
        count += 1
    return count


def _parse_date(date_text: str) -> date:
    text = date_text.strip()
    for fmt in ("%d/%m/%Y", "%d/%m/%y"):
        try:
            return datetime.strptime(text, fmt).date()
        except ValueError:
            continue
    raise ValueError(f"bad football-data Date {date_text!r}")


def _first_odds(row: dict[str, Any], columns: tuple[str, ...]) -> float | None:
    for name in columns:
        if name not in row:
            continue
        value = row[name]
        if value is None:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        try:
            odds = float(value)
        except (TypeError, ValueError):
            continue
        if odds > 1.0:
            return odds
    return None
