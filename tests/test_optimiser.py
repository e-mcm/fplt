"""Tests for the ILP team selector and transfer recommender."""

import numpy as np
import pandas as pd
import pytest

from fplt.optimisation.team_selector import SelectionResult, select_team
from fplt.optimisation.transfers import recommend_transfers

# ── Helpers ───────────────────────────────────────────────────────────────────


def _make_players(
    n_gk: int = 4,
    n_def: int = 10,
    n_mid: int = 10,
    n_fwd: int = 6,
    price: int = 60,
    rng_seed: int = 0,
) -> pd.DataFrame:
    """Generate a synthetic player pool with realistic values."""
    rng = np.random.default_rng(rng_seed)
    rows = []
    eid = 1
    for pos, count in [("GK", n_gk), ("DEF", n_def), ("MID", n_mid), ("FWD", n_fwd)]:
        for i in range(count):
            rows.append(
                {
                    "element": eid,
                    "name": f"{pos}_{i}",
                    "position": pos,
                    "team": f"club_{(eid % 20) + 1}",
                    "value": price + rng.integers(-10, 20),
                    "xpts": float(rng.uniform(2.0, 8.0)),
                }
            )
            eid += 1
    return pd.DataFrame(rows)


# ── Team selector ─────────────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def player_pool() -> pd.DataFrame:
    return _make_players()


@pytest.fixture(scope="module")
def selection(player_pool: pd.DataFrame) -> SelectionResult:
    return select_team(player_pool)


def test_squad_size(selection: SelectionResult) -> None:
    assert len(selection.squad) == 15


def test_xi_size(selection: SelectionResult) -> None:
    assert len(selection.xi) == 11


def test_captain_is_in_xi(selection: SelectionResult) -> None:
    xi_ids = set(selection.xi["element"].tolist())
    assert selection.captain["element"] in xi_ids


def test_budget_not_exceeded(selection: SelectionResult) -> None:
    total_value = selection.squad["value"].sum()
    assert total_value <= 1000


def test_squad_position_quotas(selection: SelectionResult) -> None:
    counts = selection.squad["position"].value_counts()
    assert counts.get("GK", 0) == 2
    assert counts.get("DEF", 0) == 5
    assert counts.get("MID", 0) == 5
    assert counts.get("FWD", 0) == 3


def test_xi_formation_valid(selection: SelectionResult) -> None:
    counts = selection.xi["position"].value_counts()
    assert counts.get("GK", 0) == 1
    assert counts.get("DEF", 0) >= 3
    assert counts.get("MID", 0) >= 2
    assert counts.get("FWD", 0) >= 1


def test_max_three_per_club(selection: SelectionResult) -> None:
    club_counts = selection.squad["team"].value_counts()
    assert (club_counts <= 3).all()


def test_total_xpts_positive(selection: SelectionResult) -> None:
    assert selection.total_xpts > 0


# ── Transfer recommender ───────────────────────────────────────────────────────


@pytest.fixture(scope="module")
def squad_and_pool(
    player_pool: pd.DataFrame, selection: SelectionResult
) -> tuple[pd.DataFrame, pd.DataFrame]:
    squad_ids = set(selection.squad["element"].tolist())
    pool = player_pool[~player_pool["element"].isin(squad_ids)].copy()
    return selection.squad.copy(), pool


def test_single_transfer_returns_one_transfer(squad_and_pool: tuple) -> None:
    squad, pool = squad_and_pool
    rec = recommend_transfers(squad, pool, free_transfers=1, n_transfers=1)
    assert len(rec.transfers) == 1


def test_double_transfer_returns_two_transfers(squad_and_pool: tuple) -> None:
    squad, pool = squad_and_pool
    rec = recommend_transfers(squad, pool, free_transfers=1, n_transfers=2)
    assert len(rec.transfers) == 2


def test_transfer_preserves_position(squad_and_pool: tuple) -> None:
    squad, pool = squad_and_pool
    rec = recommend_transfers(squad, pool, free_transfers=1, n_transfers=1)
    t = rec.transfers[0]
    assert t.player_out["position"] == t.player_in["position"]


def test_net_gain_equals_gross_minus_hit(squad_and_pool: tuple) -> None:
    squad, pool = squad_and_pool
    rec = recommend_transfers(squad, pool, free_transfers=0, n_transfers=1)
    assert abs(rec.net_gain - (rec.gross_gain - rec.hit_cost)) < 1e-9


def test_invalid_n_transfers_raises(squad_and_pool: tuple) -> None:
    squad, pool = squad_and_pool
    with pytest.raises(ValueError):
        recommend_transfers(squad, pool, n_transfers=3)
