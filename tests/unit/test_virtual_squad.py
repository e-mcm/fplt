"""B12c: virtual squad chain and official points_pmf binning."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import polars as pl
import pytest

from fplt.optimise.decisions import advance_squad_from_plan
from fplt.optimise.plan import GameweekPlan
from fplt.optimise.state import ChipInventory, OwnedPlayer, SquadState
from fplt.schemas.manifests import AsOfDeadline
from fplt.scoring.chips import empty_remaining
from fplt.store.points_pmf import points_pmf_from_draws
from fplt.store.runs import RunWriter
from fplt.store.virtual_squad import (
    chip_remaining,
    frame_to_squad_state,
    resolve_virtual_squad,
    squad_state_to_frame,
    write_squad_state_pair,
)
from fplt.types import GWKey

_TS = datetime(2026, 11, 26, 12, 0, tzinfo=UTC)
_AS_OF = GWKey("2026-27", 12)


def _squad(*, bank: int = 50, ft: int = 1, bboost: int = 1) -> SquadState:
    remaining = empty_remaining()
    remaining["bboost"] = bboost
    return SquadState(
        bank=bank,
        free_transfers=ft,
        ft_bank_cap=5,
        hit_cost=4,
        decay=0.9,
        bench_weight=0.0,
        owned=[
            OwnedPlayer(player_key=1, selling_price=45, purchase_price=45),
            OwnedPlayer(player_key=2, selling_price=50, purchase_price=50),
        ],
        chips=ChipInventory(remaining=remaining, played=[]),
    )


def _plan_sell_buy(*, chip: str | None = None) -> GameweekPlan:
    # Keep player 1, sell 2, buy 3 at 55; optional chip on step 0.
    picks = pl.DataFrame(
        {
            "step": [0, 0, 0],
            "player_key": [1, 2, 3],
            "price": [45, 50, 55],
            "bought": [False, False, True],
            "sold": [False, True, False],
            "in_xi": [True, False, True],
            "captain": [False, False, False],
            "vice": [False, False, False],
            "xpts": [1.0, 0.0, 1.0],
        }
    )
    return GameweekPlan(
        picks=picks,
        hits=(0.0,),
        free_transfers=(1,),
        points=1.0,
        chips=(chip,),
    )


def test_round_trip_and_midseason_init(tmp_path: Path) -> None:
    real = _squad(bank=80, ft=2)
    assert resolve_virtual_squad(tmp_path, prev_run_id=None, real_squad=real) == real
    frame = squad_state_to_frame(real, run_id="r_x", as_of=_AS_OF, deadline_utc=_TS)
    assert frame_to_squad_state(frame).bank == 80


def test_chain_bank_ft_and_chip_consumption(tmp_path: Path) -> None:
    state = _squad(bank=50, ft=1, bboost=1)
    plan = _plan_sell_buy(chip="bboost")
    advanced = advance_squad_from_plan(
        state, plan, pl.DataFrame({"player_key": [1, 2, 3]}), gameweek=12, rules=None
    )
    # sold 2 (+50) bought 3 (−55) → bank 45; bboost consumed; FT regenerates to 1
    assert advanced.bank == 45
    assert chip_remaining(advanced, "bboost") == 0
    assert advanced.free_transfers == 1
    assert {p.player_key for p in advanced.owned} == {1, 3}

    with RunWriter(
        tmp_path,
        kind="official",
        mode="live",
        run_type="plan",
        spec_hash="a" * 64,
        spec_label="demo",
        model_id="deadbeef",
        as_of=[AsOfDeadline(season="2026-27", gw=12, deadline_utc=_TS)],
        horizon=5,
        plan_horizon=5,
        code_sha="abc",
        code_dirty=False,
        lockfile_hash="b" * 64,
        host="test",
        raw_manifest_hash="c" * 64,
        staged_digests={},
        started_at=_TS,
        run_id="r_20261126T120000Z_virt01",
        line="v4_h5",
        squad_mode="virtual",
        checkpoint="T-90m",
    ) as writer:
        write_squad_state_pair(
            writer,
            state_in=state,
            state_out=advanced,
            as_of=_AS_OF,
            deadline_utc=_TS,
        )

    nxt = resolve_virtual_squad(
        tmp_path, prev_run_id="r_20261126T120000Z_virt01", real_squad=_squad()
    )
    assert nxt.bank == 45
    assert chip_remaining(nxt, "bboost") == 0
    # Chip used on the virtual line stays unavailable later.
    assert chip_remaining(nxt, "bboost") == 0
    assert any(play.name == "bboost" and play.gw == 12 for play in nxt.chips.played)


def test_points_pmf_fixture_and_dgw_vs_bgw() -> None:
    # Two fixtures in GW12 (DGW) for one player; BGWless GW13 absent.
    draws = pl.DataFrame(
        {
            "player_key": [7, 7, 7, 7],
            "team_key": [1, 1, 1, 1],
            "fixture_id": [10, 10, 11, 11],
            "target_season": ["2026-27"] * 4,
            "target_gw": [12, 12, 12, 12],
            "horizon": [0, 0, 0, 0],
            "draw_id": [0, 1, 0, 1],
            "points": [2, 4, 3, 1],  # draw0 total 5, draw1 total 5
        }
    )
    fx = points_pmf_from_draws(
        draws,
        run_id="r_pmf",
        as_of_season="2026-27",
        as_of_gw=12,
        deadline_utc=_TS,
        grain="fixture",
    )
    assert set(fx["fixture_id"].to_list()) == {10, 11}
    gw = points_pmf_from_draws(
        draws,
        run_id="r_pmf",
        as_of_season="2026-27",
        as_of_gw=12,
        deadline_utc=_TS,
        grain="gameweek",
    )
    # Both draws sum to 5 → single bin with probability 1
    assert gw.height == 1
    assert gw["points_bin"].to_list() == [5]
    assert gw["probability"].to_list() == [pytest.approx(1.0)]
    # Blank GW: no rows for target_gw=13
    assert gw.filter(pl.col("target_gw") == 13).is_empty()


def test_points_pmf_clips_tails() -> None:
    draws = pl.DataFrame(
        {
            "player_key": [1, 1],
            "team_key": [1, 1],
            "fixture_id": [9, 9],
            "target_season": ["2026-27", "2026-27"],
            "target_gw": [1, 1],
            "horizon": [0, 0],
            "draw_id": [0, 1],
            "points": [-9, 40],
        }
    )
    fx = points_pmf_from_draws(
        draws,
        run_id="r_pmf",
        as_of_season="2026-27",
        as_of_gw=1,
        deadline_utc=_TS,
        grain="fixture",
    )
    assert set(fx["points_bin"].to_list()) == {-5, 30}
