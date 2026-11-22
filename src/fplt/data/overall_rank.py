"""Ingest and cache FPL overall-league points totals at key ranks.

The classic Overall league (id ``314``) exposes standings 50 managers per page.
Fetching a handful of pages around ranks such as 10k and 100k is cheap and is
enough to map a season points total onto an equivalent rank / percentile
(see ``fplt.backtest.live_percentile`` and evaluation-north-star §3).

Raw caches are write-once under
``raw/fpl_api/season=.../overall_rank/captured_at=.../points_at_rank.json.gz``.
"""

from __future__ import annotations

import gzip
import json
import logging
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import polars as pl

from fplt.data.fpl_client import FPLAPIError, FPLClient
from fplt.schemas.overall_rank import OverallRankLadderSchema

logger = logging.getLogger(__name__)

OVERALL_LEAGUE_ID = 314
STANDINGS_PAGE_SIZE = 50
DEFAULT_TARGET_RANKS: tuple[int, ...] = (
    1,
    1_000,
    10_000,
    100_000,
    500_000,
    1_000_000,
)


@dataclass(frozen=True)
class PointsAtRank:
    """Points total observed near one target overall rank."""

    target_rank: int
    observed_rank: int
    points: int
    page: int


@dataclass(frozen=True)
class OverallRankSnapshot:
    """Cached ladder of overall-league points at selected ranks.

    Args:
        season: FPL season label (e.g. ``2026-27``).
        captured_at: UTC time the standings pages were fetched.
        league_id: Classic league id (Overall is ``314``).
        total_players: Field size from ``bootstrap-static``.
        ranks: Points observed near each requested target rank.
    """

    season: str
    captured_at: datetime
    league_id: int
    total_players: int
    ranks: tuple[PointsAtRank, ...]

    def to_payload(self) -> dict[str, Any]:
        """Serialise for write-once gzipped JSON."""
        return {
            "season": self.season,
            "captured_at": _stamp(self.captured_at),
            "league_id": self.league_id,
            "total_players": self.total_players,
            "ranks": [asdict(row) for row in self.ranks],
        }

    def to_frame(self) -> pl.DataFrame:
        """Return a validated ladder frame (``OverallRankLadderSchema``)."""
        stamp = _stamp(self.captured_at)
        frame = pl.DataFrame(
            {
                "season": [self.season] * len(self.ranks),
                "captured_at": [stamp] * len(self.ranks),
                "league_id": [self.league_id] * len(self.ranks),
                "total_players": [self.total_players] * len(self.ranks),
                "target_rank": [row.target_rank for row in self.ranks],
                "observed_rank": [row.observed_rank for row in self.ranks],
                "points": [row.points for row in self.ranks],
                "page": [row.page for row in self.ranks],
            }
        )
        return OverallRankLadderSchema.validate(frame)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> OverallRankSnapshot:
        """Parse a cached ``points_at_rank`` JSON object.

        Args:
            payload: Object as written by ``to_payload``.

        Returns:
            Typed snapshot.

        Raises:
            FPLAPIError: Required fields are missing or malformed.
        """
        try:
            season = str(payload["season"])
            captured_raw = str(payload["captured_at"])
            league_id = int(payload["league_id"])
            total_players = int(payload["total_players"])
            ranks_raw = payload["ranks"]
        except (KeyError, TypeError, ValueError) as exc:
            raise FPLAPIError("overall-rank payload missing required fields") from exc
        if not isinstance(ranks_raw, list) or not ranks_raw:
            raise FPLAPIError("overall-rank payload needs a non-empty ranks list")
        ranks: list[PointsAtRank] = []
        for row_raw in cast(list[Any], ranks_raw):
            if not isinstance(row_raw, dict):
                raise FPLAPIError("overall-rank ranks entries must be objects")
            row = cast(dict[str, Any], row_raw)
            try:
                ranks.append(
                    PointsAtRank(
                        target_rank=int(row["target_rank"]),
                        observed_rank=int(row["observed_rank"]),
                        points=int(row["points"]),
                        page=int(row["page"]),
                    )
                )
            except (KeyError, TypeError, ValueError) as exc:
                raise FPLAPIError("overall-rank rank row malformed") from exc
        return cls(
            season=season,
            captured_at=_parse_stamp(captured_raw),
            league_id=league_id,
            total_players=total_players,
            ranks=tuple(ranks),
        )


def season_label(moment: datetime) -> str:
    """FPL season containing ``moment``. A season starts in July."""
    year = moment.year if moment.month >= 7 else moment.year - 1
    return f"{year}-{(year + 1) % 100:02d}"


def page_for_rank(rank: int) -> int:
    """1-based standings page that contains ``rank`` (50 managers per page).

    Args:
        rank: Overall rank (1 = leader).

    Returns:
        Page index suitable for ``page_standings``.

    Raises:
        ValueError: ``rank`` is less than 1.
    """
    if rank < 1:
        raise ValueError(f"rank must be >= 1, got {rank}")
    return (rank + STANDINGS_PAGE_SIZE - 1) // STANDINGS_PAGE_SIZE


def points_near_rank(
    standings_payload: dict[str, Any], target_rank: int
) -> PointsAtRank:
    """Pick the standings row closest to ``target_rank`` from one page.

    Ties share a rank in the FPL API, so the page for ``ceil(R/50)`` may not
    contain ``R`` exactly; the nearest observed rank is stored instead.

    Args:
        standings_payload: Body from ``classic_league_standings``.
        target_rank: Desired overall rank.

    Returns:
        ``PointsAtRank`` for the closest row on the page.

    Raises:
        FPLAPIError: The page has no ``standings.results``.
    """
    standings_raw = standings_payload.get("standings")
    if not isinstance(standings_raw, dict):
        raise FPLAPIError("standings payload needs standings object")
    standings = cast(dict[str, Any], standings_raw)
    results_raw = standings.get("results")
    if not isinstance(results_raw, list) or not results_raw:
        raise FPLAPIError("standings payload needs non-empty results")
    results = cast(list[Any], results_raw)
    page_raw = standings.get("page")
    page = int(page_raw) if page_raw is not None else page_for_rank(target_rank)
    best: dict[str, Any] | None = None
    best_dist: int | None = None
    for row_raw in results:
        if not isinstance(row_raw, dict):
            continue
        row = cast(dict[str, Any], row_raw)
        try:
            observed = int(row["rank"])
            points = int(row["total"])
        except (KeyError, TypeError, ValueError) as exc:
            raise FPLAPIError("standings row missing rank/total") from exc
        dist = abs(observed - target_rank)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best = {"observed_rank": observed, "points": points}
    if best is None:
        raise FPLAPIError("standings payload needs usable results")
    return PointsAtRank(
        target_rank=target_rank,
        observed_rank=int(best["observed_rank"]),
        points=int(best["points"]),
        page=page,
    )


def fetch_overall_rank_snapshot(
    client: FPLClient,
    *,
    season: str | None = None,
    captured_at: datetime | None = None,
    target_ranks: Sequence[int] = DEFAULT_TARGET_RANKS,
    league_id: int = OVERALL_LEAGUE_ID,
) -> OverallRankSnapshot:
    """Fetch bootstrap field size and points near each target rank.

    Args:
        client: FPL HTTP client (public endpoints only).
        season: Season label; defaults from ``captured_at``.
        captured_at: Snapshot time (UTC); defaults to now.
        target_ranks: Ranks to sample (at least 10k and 100k recommended).
        league_id: Classic league id (Overall ``314``).

    Returns:
        ``OverallRankSnapshot`` ready to cache.

    Raises:
        ValueError: ``target_ranks`` is empty.
        FPLAPIError: API payloads are missing required fields.
    """
    if not target_ranks:
        raise ValueError("target_ranks must be non-empty")
    moment = captured_at or datetime.now(UTC)
    if moment.tzinfo is None:
        moment = moment.replace(tzinfo=UTC)
    else:
        moment = moment.astimezone(UTC)
    label = season or season_label(moment)
    bootstrap = client.bootstrap_static()
    try:
        total_players = int(bootstrap["total_players"])
    except (KeyError, TypeError, ValueError) as exc:
        raise FPLAPIError("bootstrap-static needs total_players") from exc
    ranks: list[PointsAtRank] = []
    # One request per distinct page even when targets share a page.
    pages: dict[int, dict[str, Any]] = {}
    for target in sorted(set(int(r) for r in target_ranks)):
        page = page_for_rank(target)
        if page not in pages:
            logger.info(
                "fetching overall standings league=%s page=%s (target_rank=%s)",
                league_id,
                page,
                target,
            )
            pages[page] = client.classic_league_standings(
                league_id, page_standings=page
            )
        ranks.append(points_near_rank(pages[page], target))
    return OverallRankSnapshot(
        season=label,
        captured_at=moment,
        league_id=league_id,
        total_players=total_players,
        ranks=tuple(ranks),
    )


def overall_rank_path(root: Path, season: str, captured_at: datetime) -> Path:
    """Write-once path for one overall-rank ladder snapshot."""
    return (
        root
        / "fpl_api"
        / f"season={season}"
        / "overall_rank"
        / f"captured_at={_stamp(captured_at)}"
        / "points_at_rank.json.gz"
    )


def write_overall_rank_snapshot(root: Path, snapshot: OverallRankSnapshot) -> bool:
    """Write ``snapshot`` gzipped JSON if the path does not exist.

    Args:
        root: Raw data root (usually ``$FPLT_DATA_DIR/raw``).
        snapshot: Ladder to persist.

    Returns:
        True when a new file was written.
    """
    path = overall_rank_path(root, snapshot.season, snapshot.captured_at)
    if path.exists():
        return False
    path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(path, "wt", encoding="utf-8") as handle:
        json.dump(snapshot.to_payload(), handle, separators=(",", ":"))
    return True


def load_overall_rank_snapshot(path: Path) -> OverallRankSnapshot:
    """Load one cached overall-rank ladder.

    Args:
        path: Gzipped ``points_at_rank.json.gz``.

    Returns:
        Typed ``OverallRankSnapshot``.
    """
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise FPLAPIError("overall-rank cache must be a JSON object")
    return OverallRankSnapshot.from_payload(cast(dict[str, Any], payload))


def iter_overall_rank_snapshots(root: Path, season: str | None = None) -> list[Path]:
    """List cached overall-rank ladder paths, oldest first.

    Args:
        root: Raw data root.
        season: When set, only that season's captures.

    Returns:
        Paths to ``points_at_rank.json.gz`` files.
    """
    base = root / "fpl_api"
    if season is not None:
        pattern = f"season={season}/overall_rank/captured_at=*/points_at_rank.json.gz"
    else:
        pattern = "season=*/overall_rank/captured_at=*/points_at_rank.json.gz"
    return sorted(base.glob(pattern))


def load_latest_overall_rank(root: Path, season: str) -> OverallRankSnapshot | None:
    """Load the most recent overall-rank capture for ``season``, if any.

    Args:
        root: Raw data root.
        season: FPL season label.

    Returns:
        Latest snapshot, or ``None`` when none are cached.
    """
    paths = iter_overall_rank_snapshots(root, season)
    if not paths:
        return None
    return load_overall_rank_snapshot(paths[-1])


def capture_overall_rank(
    client: FPLClient,
    root: Path,
    *,
    season: str | None = None,
    captured_at: datetime | None = None,
    target_ranks: Iterable[int] = DEFAULT_TARGET_RANKS,
    league_id: int = OVERALL_LEAGUE_ID,
) -> tuple[OverallRankSnapshot, bool]:
    """Fetch the ladder and write it write-once under ``root``.

    Args:
        client: FPL HTTP client.
        root: Raw data root.
        season: Optional season override.
        captured_at: Optional capture time.
        target_ranks: Ranks to sample.
        league_id: Classic league id.

    Returns:
        ``(snapshot, wrote)`` where ``wrote`` is False if the path existed.
    """
    snapshot = fetch_overall_rank_snapshot(
        client,
        season=season,
        captured_at=captured_at,
        target_ranks=tuple(target_ranks),
        league_id=league_id,
    )
    wrote = write_overall_rank_snapshot(root, snapshot)
    return snapshot, wrote


def _stamp(moment: datetime) -> str:
    return moment.astimezone(UTC).strftime("%Y-%m-%dT%H-%M-%SZ")


def _parse_stamp(value: str) -> datetime:
    if value.endswith("Z") and "T" in value and "-" in value[value.index("T") :]:
        # Filesystem stamp: 2026-11-22T12-00-00Z
        try:
            return datetime.strptime(value, "%Y-%m-%dT%H-%M-%SZ").replace(tzinfo=UTC)
        except ValueError:
            pass
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(UTC)
