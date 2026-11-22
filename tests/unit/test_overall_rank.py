"""Overall-league points-at-rank ingest and write-once cache."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

import httpx
import pytest

from fplt.data.fpl_client import FPLAPIError, FPLClient
from fplt.data.overall_rank import (
    DEFAULT_TARGET_RANKS,
    OverallRankSnapshot,
    PointsAtRank,
    capture_overall_rank,
    fetch_overall_rank_snapshot,
    load_latest_overall_rank,
    load_overall_rank_snapshot,
    overall_rank_path,
    page_for_rank,
    points_near_rank,
    season_label,
    write_overall_rank_snapshot,
)

_FIXTURES = Path(__file__).parents[1] / "fixtures" / "fpl_api"
_API = "https://fantasy.premierleague.com/api"
_NOW = datetime(2025, 5, 25, 18, 0, tzinfo=UTC)


def _load(name: str) -> dict:
    return json.loads((_FIXTURES / name).read_text(encoding="utf-8"))


def _client(handler: httpx.MockTransport) -> tuple[FPLClient, httpx.Client]:
    http = httpx.Client(transport=handler, base_url=_API)
    return FPLClient(http, base_url=_API), http


def test_page_for_rank_uses_fifty_per_page() -> None:
    assert page_for_rank(1) == 1
    assert page_for_rank(50) == 1
    assert page_for_rank(51) == 2
    assert page_for_rank(10_000) == 200
    assert page_for_rank(100_000) == 2000


def test_page_for_rank_rejects_non_positive() -> None:
    with pytest.raises(ValueError, match="rank must be >= 1"):
        page_for_rank(0)


def test_season_label_starts_in_july() -> None:
    assert season_label(datetime(2025, 6, 30, tzinfo=UTC)) == "2024-25"
    assert season_label(datetime(2025, 7, 1, tzinfo=UTC)) == "2025-26"


def test_points_near_rank_picks_closest_row() -> None:
    payload = _load("standings_page_200.json")
    row = points_near_rank(payload, 10_000)
    assert row == PointsAtRank(
        target_rank=10_000,
        observed_rank=10_000,
        points=2195,
        page=200,
    )


def test_points_near_rank_requires_results() -> None:
    with pytest.raises(FPLAPIError, match="non-empty results"):
        points_near_rank({"standings": {"page": 1, "results": []}}, 1)


def test_fetch_snapshot_uses_fixtures_no_network() -> None:
    bootstrap = _load("bootstrap_static_total_players.json")
    pages = {
        1: _load("standings_page_1.json"),
        200: _load("standings_page_200.json"),
        2000: _load("standings_page_2000.json"),
    }

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/bootstrap-static/"):
            return httpx.Response(200, json=bootstrap)
        if "/leagues-classic/314/standings/" in request.url.path:
            page = int(request.url.params["page_standings"])
            return httpx.Response(200, json=pages[page])
        raise AssertionError(request.url)

    api, http = _client(httpx.MockTransport(handler))
    try:
        snap = fetch_overall_rank_snapshot(
            api,
            season="2024-25",
            captured_at=_NOW,
            target_ranks=(1, 10_000, 100_000),
        )
    finally:
        api.close()
        http.close()

    assert snap.season == "2024-25"
    assert snap.total_players == 10_000_000
    assert snap.league_id == 314
    assert [row.target_rank for row in snap.ranks] == [1, 10_000, 100_000]
    assert snap.ranks[1].points == 2195
    assert snap.ranks[2].points == 2040
    frame = snap.to_frame()
    assert frame.height == 3
    assert "points" in frame.columns


def test_write_once_and_load_latest(tmp_path: Path) -> None:
    payload = _load("points_at_rank.json")
    snap = OverallRankSnapshot.from_payload(payload)
    root = tmp_path / "raw"
    assert write_overall_rank_snapshot(root, snap)
    assert not write_overall_rank_snapshot(root, snap)
    path = overall_rank_path(root, snap.season, snap.captured_at)
    assert path.exists()
    loaded = load_overall_rank_snapshot(path)
    assert loaded.total_players == snap.total_players
    assert loaded.ranks[1].points == 2195
    latest = load_latest_overall_rank(root, "2024-25")
    assert latest is not None
    assert latest.season == "2024-25"
    assert load_latest_overall_rank(root, "2099-00") is None


def test_capture_overall_rank_write_once(tmp_path: Path) -> None:
    bootstrap = _load("bootstrap_static_total_players.json")
    page = _load("standings_page_200.json")

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path.endswith("/bootstrap-static/"):
            return httpx.Response(200, json=bootstrap)
        if "/leagues-classic/314/standings/" in request.url.path:
            return httpx.Response(200, json=page)
        raise AssertionError(request.url)

    api, http = _client(httpx.MockTransport(handler))
    try:
        first, wrote = capture_overall_rank(
            api,
            tmp_path / "raw",
            season="2024-25",
            captured_at=_NOW,
            target_ranks=(10_000,),
        )
        second, wrote_again = capture_overall_rank(
            api,
            tmp_path / "raw",
            season="2024-25",
            captured_at=_NOW,
            target_ranks=(10_000,),
        )
    finally:
        api.close()
        http.close()

    assert wrote
    assert not wrote_again
    assert first.ranks[0].points == second.ranks[0].points == 2195


def test_default_target_ranks_include_10k_and_100k() -> None:
    assert 10_000 in DEFAULT_TARGET_RANKS
    assert 100_000 in DEFAULT_TARGET_RANKS


def test_from_payload_rejects_empty_ranks() -> None:
    with pytest.raises(FPLAPIError, match="non-empty ranks"):
        OverallRankSnapshot.from_payload(
            {
                "season": "2024-25",
                "captured_at": "2025-05-25T18-00-00Z",
                "league_id": 314,
                "total_players": 1,
                "ranks": [],
            }
        )


def test_classic_league_standings_on_client() -> None:
    payload = _load("standings_page_1.json")

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.url.path.endswith("/leagues-classic/314/standings/")
        assert request.url.params["page_standings"] == "1"
        return httpx.Response(200, json=payload)

    api, http = _client(httpx.MockTransport(handler))
    try:
        body = api.classic_league_standings(314, page_standings=1)
    finally:
        api.close()
        http.close()
    assert body["league"]["name"] == "Overall"
