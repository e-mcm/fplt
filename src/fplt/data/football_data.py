"""Stage football-data.co.uk E0 odds into ``match_odds``.

Two ``odds_kind`` values are written from the same E0 CSV:

- ``closing_proxy`` — closing averages (``AvgCH`` / ``AvgCD`` / ``AvgCA``, plus
  O/U when present), falling back to opening averages when close is absent.
  ``available_at`` is kickoff. Pre-kickoff proxy, **not** an FPL deadline
  snapshot.
- ``prematch_scheduled`` — non-closing columns only (``AvgH`` / ``AvgD`` /
  ``AvgA``, ``B365H`` / ``B365D`` / ``B365A``, plus O/U ``Avg>2.5`` etc.).
  ``available_at`` is a **conservative** football-data collection estimate
  (see ``estimated_prematch_available_at``). Closing columns are never
  relabelled as prematch.

Deadline coverage: any fixture whose FPL deadline falls before the estimated
collection time must drop out of prematch boards — use
``check_prematch_deadline_coverage`` / ``require_prematch_deadline_coverage``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
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
_CLOSING_KIND = "closing_proxy"
_PREMATCH_KIND = "prematch_scheduled"
_SOURCE = "football_data"
_DEFAULT_KICKOFF = time(15, 0)
# Conservative "afternoon" stamp for football-data's published collection window.
_PREMATCH_COLLECTION_LOCAL = time(17, 0)

# Prefer closing market averages; fall back to Bet365 closing, then opening avg.
_CLOSING_HOME_COLS = ("AvgCH", "B365CH", "AvgH", "B365H")
_CLOSING_DRAW_COLS = ("AvgCD", "B365CD", "AvgD", "B365D")
_CLOSING_AWAY_COLS = ("AvgCA", "B365CA", "AvgA", "B365A")
# O/U stays closing-only so nullable absence means "no closing total", not open.
_CLOSING_OVER_COLS = ("AvgC>2.5", "B365C>2.5")
_CLOSING_UNDER_COLS = ("AvgC<2.5", "B365C<2.5")

# Non-closing columns only — never read AvgCH / B365CH into prematch.
_PREMATCH_HOME_COLS = ("AvgH", "B365H")
_PREMATCH_DRAW_COLS = ("AvgD", "B365D")
_PREMATCH_AWAY_COLS = ("AvgA", "B365A")
_PREMATCH_OVER_COLS = ("Avg>2.5", "B365>2.5")
_PREMATCH_UNDER_COLS = ("Avg<2.5", "B365<2.5")


class PrematchCoverageError(ValueError):
    """Prematch deadline coverage failed for the requested board fixtures."""


@dataclass(frozen=True)
class PrematchCoverageResult:
    """Deadline-legal coverage of ``prematch_scheduled`` odds over fixtures.

    A fixture is covered only when a ``prematch_scheduled`` row joins and
    ``available_at <= deadline_time``. Fixtures whose FPL deadline falls before
    the estimated collection time are listed in ``dropped_fixture_ids``.
    """

    numerator: int
    denominator: int
    rate: float
    min_coverage: float | None
    passed: bool
    dropped_fixture_ids: tuple[int, ...]
    message: str


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


def estimated_prematch_available_at(kickoff: datetime) -> datetime:
    """Conservative football-data collection time for non-closing odds.

    football-data states that pre-match odds for weekend games are collected
    on Friday afternoons, and midweek games on Tuesday afternoons. This helper
    stamps that window late so early FPL deadlines fail the coverage gate
    rather than leaking post-deadline lines.

    Rule (Europe/London kickoff calendar day):

    - **Fri / Sat / Sun** → that week's **Friday 17:00** London.
    - **Mon / Tue / Wed / Thu** → that week's **Tuesday 17:00** London
      (Monday fixtures therefore stamp *after* kickoff and drop under the
      deadline coverage gate).

    Args:
        kickoff: Aware match kickoff (any zone; converted to London).

    Returns:
        Aware UTC ``available_at`` for ``odds_kind=prematch_scheduled``.
    """
    local = kickoff.astimezone(_LONDON)
    weekday = local.weekday()  # Mon=0 … Sun=6
    if weekday >= 4:
        # Friday of the same week (Fri=0 days back, Sat=1, Sun=2).
        collection_date = local.date() - timedelta(days=weekday - 4)
    else:
        # Tuesday of the same week (Mon → +1 day, Tue → 0, Wed/Thu → back).
        collection_date = local.date() - timedelta(days=weekday - 1)
    stamped = datetime.combine(
        collection_date, _PREMATCH_COLLECTION_LOCAL, tzinfo=_LONDON
    )
    return stamped.astimezone(UTC)


def parse_e0_csv(path: Path, season: str, team_keys: dict[str, int]) -> pl.DataFrame:
    """Parse one E0 CSV into a ``MatchOddsSchema`` frame (no store write).

    Emits up to two rows per match: ``closing_proxy`` (closing columns with
    opening fallback) and ``prematch_scheduled`` (non-closing columns only).
    Rows without usable 1X2 odds for a kind are skipped for that kind. Unknown
    team names raise.

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
        date_text = str(row.get("Date") or "").strip()
        time_raw = row.get("Time")
        time_text = None if time_raw is None else str(time_raw)
        kickoff = parse_kickoff(date_text, time_text)
        base = {
            "season": season,
            "fixture_id": None,
            "team_h_key": team_keys[home_name],
            "team_a_key": team_keys[away_name],
            "kickoff_time": kickoff,
            "source": _SOURCE,
        }
        closing = _odds_triple(
            row, _CLOSING_HOME_COLS, _CLOSING_DRAW_COLS, _CLOSING_AWAY_COLS
        )
        if closing is not None:
            home, draw, away = closing
            rows.append(
                {
                    **base,
                    "home_odds": home,
                    "draw_odds": draw,
                    "away_odds": away,
                    "over_25_odds": _first_odds(row, _CLOSING_OVER_COLS),
                    "under_25_odds": _first_odds(row, _CLOSING_UNDER_COLS),
                    "odds_kind": _CLOSING_KIND,
                    "available_at": kickoff,
                }
            )
        prematch = _odds_triple(
            row, _PREMATCH_HOME_COLS, _PREMATCH_DRAW_COLS, _PREMATCH_AWAY_COLS
        )
        if prematch is not None:
            home, draw, away = prematch
            rows.append(
                {
                    **base,
                    "home_odds": home,
                    "draw_odds": draw,
                    "away_odds": away,
                    "over_25_odds": _first_odds(row, _PREMATCH_OVER_COLS),
                    "under_25_odds": _first_odds(row, _PREMATCH_UNDER_COLS),
                    "odds_kind": _PREMATCH_KIND,
                    "available_at": estimated_prematch_available_at(kickoff),
                }
            )
        if closing is None and prematch is None:
            _LOG.debug("skip %s vs %s: missing 1X2 odds", home_name, away_name)
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


def prematch_board_odds(odds: pl.DataFrame, fixtures: pl.DataFrame) -> pl.DataFrame:
    """Deadline-legal ``prematch_scheduled`` rows for board fixtures.

    Drops any joined fixture whose FPL ``deadline_time`` is strictly before the
    row's ``available_at`` (estimated collection time). Fixtures need
    ``deadline_time`` (join gameweeks first when staging boards).

    Args:
        odds: Staged ``match_odds`` (any kinds; non-prematch ignored).
        fixtures: Board fixtures with ``season``, ``fixture_id``, and
            ``deadline_time``.

    Returns:
        Prematch rows that survive the deadline coverage gate, typed as
        ``MatchOddsSchema`` columns.
    """
    if odds.is_empty() or fixtures.is_empty():
        return pl.DataFrame(schema=POLARS_SCHEMAS["match_odds"])
    if "deadline_time" not in fixtures.columns:
        raise ValueError(
            "fixtures must include deadline_time for prematch board filtering"
        )
    prematch = odds.filter(pl.col("odds_kind") == _PREMATCH_KIND)
    if prematch.is_empty():
        return pl.DataFrame(schema=POLARS_SCHEMAS["match_odds"])
    targets = fixtures.select(["season", "fixture_id", "deadline_time"]).unique(
        subset=["season", "fixture_id"], keep="last"
    )
    joined = (
        prematch.filter(pl.col("fixture_id").is_not_null())
        .join(targets, on=["season", "fixture_id"], how="inner")
        .filter(pl.col("available_at") <= pl.col("deadline_time"))
        .drop("deadline_time")
        .select(list(POLARS_SCHEMAS["match_odds"]))
    )
    return joined


def check_prematch_deadline_coverage(
    fixtures: pl.DataFrame,
    odds: pl.DataFrame,
    *,
    min_coverage: float | None = 1.0,
) -> PrematchCoverageResult:
    """Measure deadline-legal ``prematch_scheduled`` coverage over fixtures.

    A fixture is covered when a ``prematch_scheduled`` row joins on
    ``(season, fixture_id)`` and ``available_at <= deadline_time``. Otherwise
    it is listed in ``dropped_fixture_ids`` (missing odds or deadline before
    estimated collection).

    Args:
        fixtures: Rows with ``season``, ``fixture_id``, ``deadline_time``.
        odds: Staged ``match_odds``.
        min_coverage: Required covered fraction; ``None`` skips the pass/fail
            threshold (``passed`` stays true).

    Returns:
        ``PrematchCoverageResult`` with rate, dropped ids, and a clear message.

    Raises:
        ValueError: ``fixtures`` lacks ``deadline_time``.
    """
    if "deadline_time" not in fixtures.columns:
        raise ValueError(
            "fixtures must include deadline_time for prematch coverage checks"
        )
    targets = fixtures.select(["season", "fixture_id", "deadline_time"]).unique(
        subset=["season", "fixture_id"], keep="last"
    )
    denom = targets.height
    if denom == 0:
        return PrematchCoverageResult(
            numerator=0,
            denominator=0,
            rate=0.0,
            min_coverage=min_coverage,
            passed=min_coverage is None or min_coverage == 0.0,
            dropped_fixture_ids=(),
            message="no fixtures to cover for prematch_scheduled",
        )
    legal = prematch_board_odds(odds, targets)
    covered_ids: set[int] = set()
    if not legal.is_empty():
        covered_ids = {int(fid) for fid in legal["fixture_id"].drop_nulls().to_list()}
    all_ids = targets["fixture_id"].to_list()
    dropped = tuple(
        sorted(
            int(fid)
            for fid in all_ids
            if fid is not None and int(fid) not in covered_ids
        )
    )
    numerator = denom - len(dropped)
    rate = numerator / denom
    passed = True if min_coverage is None else rate + 1e-12 >= min_coverage
    if passed:
        message = (
            f"prematch_scheduled deadline coverage {rate:.1%} "
            f"({numerator}/{denom}); min_coverage={min_coverage}"
        )
    else:
        message = (
            "prematch_scheduled deadline coverage failed: "
            f"{rate:.1%} ({numerator}/{denom}) < min_coverage={min_coverage}. "
            "Fixtures whose FPL deadline falls before the estimated football-data "
            f"collection time (or lack prematch odds) dropped: {list(dropped)}"
        )
    return PrematchCoverageResult(
        numerator=numerator,
        denominator=denom,
        rate=rate,
        min_coverage=min_coverage,
        passed=passed,
        dropped_fixture_ids=dropped,
        message=message,
    )


def require_prematch_deadline_coverage(
    fixtures: pl.DataFrame,
    odds: pl.DataFrame,
    *,
    min_coverage: float = 1.0,
) -> PrematchCoverageResult:
    """Like ``check_prematch_deadline_coverage`` but raises on failure.

    Args:
        fixtures: Board fixtures with ``deadline_time``.
        odds: Staged ``match_odds``.
        min_coverage: Required covered fraction.

    Returns:
        Passing ``PrematchCoverageResult``.

    Raises:
        PrematchCoverageError: Coverage is below ``min_coverage``.
    """
    result = check_prematch_deadline_coverage(fixtures, odds, min_coverage=min_coverage)
    if not result.passed:
        raise PrematchCoverageError(result.message)
    return result


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


def _odds_triple(
    row: dict[str, Any],
    home_cols: tuple[str, ...],
    draw_cols: tuple[str, ...],
    away_cols: tuple[str, ...],
) -> tuple[float, float, float] | None:
    home = _first_odds(row, home_cols)
    draw = _first_odds(row, draw_cols)
    away = _first_odds(row, away_cols)
    if home is None or draw is None or away is None:
        return None
    return home, draw, away


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
