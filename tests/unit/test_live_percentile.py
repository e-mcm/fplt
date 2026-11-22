"""Season points → overall rank / percentile mapping and multi-season rollup."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from fplt.backtest.live_percentile import (
    SeasonPercentile,
    equivalent_percentile,
    equivalent_rank,
    mean_percentile,
    season_percentiles_from_points,
)
from fplt.data.overall_rank import OverallRankSnapshot

_FIXTURES = Path(__file__).parents[1] / "fixtures" / "fpl_api"


def _snapshot() -> OverallRankSnapshot:
    payload = json.loads(
        (_FIXTURES / "points_at_rank.json").read_text(encoding="utf-8")
    )
    return OverallRankSnapshot.from_payload(payload)


def test_equivalent_rank_hits_ladder_rungs() -> None:
    snap = _snapshot()
    assert equivalent_rank(2400, snap) == 1.0
    assert equivalent_rank(2195, snap) == 10_000.0
    assert equivalent_rank(2040, snap) == 100_000.0


def test_equivalent_rank_interpolates_between_rungs() -> None:
    snap = _snapshot()
    # Midway in points between 10k (2195) and 100k (2040).
    mid_points = (2195 + 2040) // 2
    rank = equivalent_rank(mid_points, snap)
    assert 10_000 < rank < 100_000
    # Closer to equal spacing in points → near midpoint of ranks.
    assert rank == pytest.approx(55_000.0, rel=0.05)


def test_equivalent_rank_clamps_outside_ladder() -> None:
    snap = _snapshot()
    assert equivalent_rank(3000, snap) == 1.0
    assert equivalent_rank(1000, snap) == 1_000_000.0


def test_equivalent_percentile_is_rank_over_field() -> None:
    snap = _snapshot()
    eq = equivalent_percentile(2195, snap)
    assert eq.season == "2024-25"
    assert eq.season_points == 2195
    assert eq.rank == 10_000.0
    assert eq.percentile == pytest.approx(10_000 / 11_500_000)
    assert eq.total_players == 11_500_000


def test_mean_percentile_averages_across_seasons() -> None:
    rows = [
        SeasonPercentile(
            season="2023-24", season_points=2100, percentile=0.02, rank=200_000
        ),
        SeasonPercentile(
            season="2024-25", season_points=2200, percentile=0.01, rank=100_000
        ),
    ]
    assert mean_percentile(rows) == pytest.approx(0.015)


def test_mean_percentile_rejects_empty() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        mean_percentile([])


def test_season_percentiles_from_points_rollup() -> None:
    snap = _snapshot()
    # Re-use the same ladder under two season keys for the helper contract.
    snapshots = {"2023-24": snap, "2024-25": snap}
    rows = season_percentiles_from_points(
        [("2023-24", 2195), ("2024-25", 2040)],
        snapshots,
    )
    assert [row.season for row in rows] == ["2023-24", "2024-25"]
    assert rows[0].rank == 10_000.0
    assert rows[1].rank == 100_000.0
    mean = mean_percentile(rows)
    expected = (10_000 / 11_500_000 + 100_000 / 11_500_000) / 2
    assert mean == pytest.approx(expected)


def test_mean_percentile_doc_is_not_promote_gate() -> None:
    """Guard the north-star note: single-season live rank is not a promote gate."""
    assert "promote gate" in mean_percentile.__doc__.lower()
    assert "single" in mean_percentile.__doc__.lower()
