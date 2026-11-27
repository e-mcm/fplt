"""Extra preflight coverage for store helpers and dry-run GW1 failure."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from fplt.data.store import DataStore
from fplt.ops.preflight import (
    CheckResult,
    _optional_status,
    _table,
    check_dry_run_plan,
)
from tests.unit.test_preflight import _SEASON, _TS, _season_frames


def test_structural_dry_run_fails_when_gw1_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    frames = _season_frames(include_unmapped=False)
    monkeypatch.setattr(
        "fplt.ops.preflight.check_model_specs",
        lambda *_a, **_k: CheckResult("model_specs", True, "ok"),
    )
    no_gw1 = frames["gameweeks"].with_columns(pl.lit(2).alias("gw"))
    result = check_dry_run_plan(
        _SEASON,
        gameweeks=no_gw1,
        players=frames["players"],
        fixtures=frames["fixtures"],
        plan_probe=None,
    )
    assert result.ok is False
    assert "GW1" in result.detail


def test_table_and_optional_status_read_existing(tmp_path: Path) -> None:
    store = DataStore(tmp_path)
    frames = _season_frames(include_unmapped=False)
    store.write("teams", frames["teams"])
    status = pl.DataFrame(
        {
            "season": [_SEASON, _SEASON],
            "player_key": [100, 200],
            "price": [50, 60],
            "status": ["a", "a"],
            "news": ["", ""],
            "news_added": [_TS, _TS],
            "ep_next": [None, None],
            "chance_of_playing": [100.0, 100.0],
            "chance_of_playing_this_round": [100.0, 100.0],
            "penalties_order": [None, None],
            "corners_and_indirect_freekicks_order": [None, None],
            "direct_freekicks_order": [None, None],
            "available_at": [_TS, _TS],
            "source": ["fpl_api", "fpl_api"],
        }
    ).with_columns(
        pl.col("ep_next").cast(pl.Float64),
        pl.col("penalties_order").cast(pl.Int64),
        pl.col("corners_and_indirect_freekicks_order").cast(pl.Int64),
        pl.col("direct_freekicks_order").cast(pl.Int64),
    )
    store.write("player_status", status)

    teams = _table(store, "teams", None)
    assert teams is not None
    assert teams.height == 2
    loaded = _optional_status(store)
    assert loaded is not None
    assert loaded.height == 2
