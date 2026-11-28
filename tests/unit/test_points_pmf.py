"""Streaming / chunked points_pmf builders (#372)."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime

import polars as pl
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from fplt.store.points_pmf import (
    narrow_pmf_draws,
    points_pmf_from_draws,
    points_pmf_tables_from_sample,
    prepare_pmf_draws,
    run_points_pmf_memory_bench,
)
from fplt.types import GWKey

_TS = datetime(2026, 8, 15, 11, 0, tzinfo=UTC)


def _multi_gw_draws() -> pl.DataFrame:
    """Two GWs including a DGW (two fixtures in GW12) for equivalence tests."""
    return pl.DataFrame(
        {
            "player_key": [7, 7, 7, 7, 7, 7],
            "team_key": [1, 1, 1, 1, 1, 1],
            "fixture_id": [10, 10, 11, 11, 20, 20],
            "target_season": ["2026-27"] * 6,
            "target_gw": [12, 12, 12, 12, 13, 13],
            "horizon": [0, 0, 0, 0, 1, 1],
            "draw_id": [0, 1, 0, 1, 0, 1],
            "points": [2, 4, 3, 1, 5, 8],
        }
    )


def test_chunked_pmf_equals_all_at_once_both_grains() -> None:
    """Per-GW binning matches materialising all draws at once (DGW-safe)."""
    draws = _multi_gw_draws()
    all_fx = points_pmf_from_draws(
        draws,
        run_id="r",
        as_of_season="2026-27",
        as_of_gw=12,
        deadline_utc=_TS,
        grain="fixture",
    ).sort(["target_gw", "fixture_id", "points_bin"])
    all_gw = points_pmf_from_draws(
        draws,
        run_id="r",
        as_of_season="2026-27",
        as_of_gw=12,
        deadline_utc=_TS,
        grain="gameweek",
    ).sort(["target_gw", "points_bin"])

    fx_parts: list[pl.DataFrame] = []
    gw_parts: list[pl.DataFrame] = []
    for gw in sorted(draws["target_gw"].unique().to_list()):
        chunk = draws.filter(pl.col("target_gw") == gw)
        fx_parts.append(
            points_pmf_from_draws(
                chunk,
                run_id="r",
                as_of_season="2026-27",
                as_of_gw=12,
                deadline_utc=_TS,
                grain="fixture",
            )
        )
        gw_parts.append(
            points_pmf_from_draws(
                chunk,
                run_id="r",
                as_of_season="2026-27",
                as_of_gw=12,
                deadline_utc=_TS,
                grain="gameweek",
            )
        )
    chunk_fx = pl.concat(fx_parts).sort(["target_gw", "fixture_id", "points_bin"])
    chunk_gw = pl.concat(gw_parts).sort(["target_gw", "points_bin"])
    assert chunk_fx.equals(all_fx)
    assert chunk_gw.equals(all_gw)
    # DGW: draw0 totals 5, draw1 totals 5 → single bin at GW grain.
    dgw = all_gw.filter(pl.col("target_gw") == 12)
    assert dgw.height == 1
    assert dgw["points_bin"].to_list() == [5]
    assert dgw["probability"].to_list() == [pytest.approx(1.0)]


def test_streaming_sample_called_per_gw() -> None:
    """Builder calls sample() once per GW with only that GW's keys."""
    as_of = GWKey("2026-27", 12)
    keys = [GWKey("2026-27", 12), GWKey("2026-27", 13)]
    players = pl.DataFrame({"player_key": [7], "team_key": [1]})
    calls: list[list[GWKey]] = []

    def _sample(chunk: Sequence[GWKey]) -> pl.DataFrame:
        calls.append(list(chunk))
        assert len(chunk) == 1
        key = chunk[0]
        if key.gw == 12:
            return pl.DataFrame(
                {
                    "player_key": [7, 7, 7, 7],
                    "fixture_id": [10, 10, 11, 11],
                    "target_season": ["2026-27"] * 4,
                    "target_gw": [12, 12, 12, 12],
                    "sim": [0, 1, 0, 1],
                    "points": [2.0, 4.0, 3.0, 1.0],
                    "minutes": [90, 90, 90, 90],
                }
            )
        return pl.DataFrame(
            {
                "player_key": [7, 7],
                "fixture_id": [20, 20],
                "target_season": ["2026-27", "2026-27"],
                "target_gw": [13, 13],
                "sim": [0, 1],
                "points": [5.0, 8.0],
                "minutes": [90, 90],
            }
        )

    fx, gw = points_pmf_tables_from_sample(
        _sample,
        horizon_keys=keys,
        players=players,
        run_id="stream",
        as_of=as_of,
        deadline_utc=_TS,
    )
    assert [c[0] for c in calls] == keys
    assert all(len(c) == 1 for c in calls)
    assert set(fx["target_gw"].to_list()) == {12, 13}
    assert set(gw["target_gw"].to_list()) == {12, 13}


def test_prepare_pmf_draws_narrows_and_rejects_missing_team() -> None:
    samples = pl.DataFrame(
        {
            "player_key": [1, 1],
            "fixture_id": [9, 9],
            "target_season": ["2026-27", "2026-27"],
            "target_gw": [1, 1],
            "sim": [0, 1],
            "points": [4.0, 6.0],
            "minutes": [90, 90],
        }
    )
    players = pl.DataFrame({"player_key": [1], "team_key": [3]})
    draws = prepare_pmf_draws(samples, players, as_of=GWKey("2026-27", 1))
    assert draws["points"].dtype == pl.Int16
    assert draws["player_key"].dtype == pl.Int32
    assert draws["draw_id"].dtype == pl.Int32
    with pytest.raises(ValueError, match="missing team_key"):
        prepare_pmf_draws(
            samples.with_columns(pl.lit(999).alias("player_key")),
            players,
            as_of=GWKey("2026-27", 1),
        )


def test_streaming_raises_when_all_gws_empty() -> None:
    def _empty(_chunk: Sequence[GWKey]) -> pl.DataFrame:
        return pl.DataFrame(
            schema={
                "player_key": pl.Int64,
                "fixture_id": pl.Int64,
                "target_season": pl.Utf8,
                "target_gw": pl.Int64,
                "sim": pl.Int64,
                "points": pl.Float64,
                "minutes": pl.Int64,
            }
        )

    with pytest.raises(ValueError, match="returned no rows"):
        points_pmf_tables_from_sample(
            _empty,
            horizon_keys=[GWKey("2026-27", 1)],
            players=pl.DataFrame({"player_key": [1], "team_key": [1]}),
            run_id="empty",
            as_of=GWKey("2026-27", 1),
            deadline_utc=_TS,
        )


@given(points=st.lists(st.integers(min_value=-5, max_value=30), min_size=2, max_size=8))
@settings(max_examples=25, deadline=None)
def test_pmf_probabilities_sum_to_one(points: list[int]) -> None:
    n = len(points)
    draws = pl.DataFrame(
        {
            "player_key": [1] * n,
            "team_key": [1] * n,
            "fixture_id": [9] * n,
            "target_season": ["2026-27"] * n,
            "target_gw": [1] * n,
            "horizon": [0] * n,
            "draw_id": list(range(n)),
            "points": points,
        }
    )
    fx = points_pmf_from_draws(
        draws,
        run_id="prop",
        as_of_season="2026-27",
        as_of_gw=1,
        deadline_utc=_TS,
        grain="fixture",
    )
    gw = points_pmf_from_draws(
        draws,
        run_id="prop",
        as_of_season="2026-27",
        as_of_gw=1,
        deadline_utc=_TS,
        grain="gameweek",
    )
    assert fx["probability"].sum() == pytest.approx(1.0)
    assert gw["probability"].sum() == pytest.approx(1.0)


def test_points_pmf_memory_bench_small() -> None:
    phases = run_points_pmf_memory_bench(
        n_gws=2, fixtures_per_gw=2, n_players=4, n_sims=5
    )
    assert "points_pmf" in phases
    assert phases["points_pmf"] > 0


def test_narrow_pmf_draws_noop_without_known_columns() -> None:
    frame = pl.DataFrame({"other": [1, 2]})
    assert narrow_pmf_draws(frame).equals(frame)
