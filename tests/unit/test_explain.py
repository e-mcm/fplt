"""Forecast board decomposition and plan/transfer decision deltas."""

from __future__ import annotations

import polars as pl
import pytest

from fplt.optimise.explain import (
    append_explain_sections,
    build_decision_explain,
    build_forecast_board,
    format_decision_explain,
    format_forecast_board,
)
from fplt.optimise.plan import plan_squad
from fplt.optimise.state import OwnedPlayer, SquadState
from fplt.scoring.rules import Rules


def test_forecast_board_components_sum_to_xpts() -> None:
    rules = Rules.for_season("2024-25")
    preds = pl.DataFrame(
        {
            "player_key": [1, 2],
            "position": ["MID", "DEF"],
            "target_gw": [1, 1],
            "fixture_id": [10, 10],
            "xpts": [6.0, 5.0],
            "xpts_sd": [1.0, 1.0],
            "p_played": [1.0, 0.9],
            "p_60": [0.8, 0.7],
            "p_goal": [0.2, 0.1],
            "p_assist": [0.3, 0.05],
            "p_cs": [0.1, 0.4],
            "p_haul": [0.05, 0.02],
        }
    )
    board = build_forecast_board(preds, season="2024-25", rules=rules, target_gw=1)
    assert board.height == 2
    for row in board.iter_rows(named=True):
        parts = (
            float(row["appearance_xpts"])
            + float(row["goals_xpts"])
            + float(row["assists_xpts"])
            + float(row["cs_xpts"])
            + float(row["residual_xpts"])
        )
        assert parts == pytest.approx(float(row["xpts"]))
    mid = board.filter(pl.col("player_key") == 1).row(0, named=True)
    # p_short=0.2 → 0.2*1; p_long=0.8 → 0.8*2; goals 0.2*5; assists 0.3*3; cs 0.1*1
    assert float(mid["appearance_xpts"]) == pytest.approx(0.2 * 1 + 0.8 * 2)
    assert float(mid["goals_xpts"]) == pytest.approx(0.2 * 5)
    assert float(mid["assists_xpts"]) == pytest.approx(0.3 * 3)
    assert float(mid["cs_xpts"]) == pytest.approx(0.1 * 1)
    text = format_forecast_board(board, names={1: "Salah", 2: "Saliba"})
    assert "Salah" in text
    assert "App≈" in text


def test_forecast_board_null_components_without_probs() -> None:
    rules = Rules.for_season("2024-25")
    preds = pl.DataFrame(
        {
            "player_key": [1],
            "position": ["FWD"],
            "target_gw": [3],
            "xpts": [4.5],
        }
    )
    board = build_forecast_board(preds, season="2024-25", rules=rules)
    row = board.row(0, named=True)
    assert row["appearance_xpts"] is None
    assert row["residual_xpts"] is None
    assert float(row["xpts"]) == pytest.approx(4.5)


def test_forecast_board_sums_double_gameweek_fixtures() -> None:
    rules = Rules.for_season("2024-25")
    preds = pl.DataFrame(
        {
            "player_key": [1, 1],
            "position": ["MID", "MID"],
            "target_gw": [7, 7],
            "fixture_id": [1, 2],
            "xpts": [3.0, 2.0],
            "p_played": [1.0, 1.0],
            "p_60": [1.0, 1.0],
            "p_goal": [0.0, 0.0],
            "p_assist": [0.0, 0.0],
            "p_cs": [0.0, 0.0],
            "p_haul": [0.0, 0.0],
        }
    )
    board = build_forecast_board(preds, season="2024-25", rules=rules, top_n=1)
    assert board.height == 1
    assert float(board["xpts"][0]) == pytest.approx(5.0)
    # Two long appearances: 2 + 2
    assert float(board["appearance_xpts"][0]) == pytest.approx(4.0)


def test_decision_explain_transfer_deltas_and_net() -> None:
    pool = _pool()
    owned = [
        OwnedPlayer(player_key=key, selling_price=price)
        for key, price in (
            (1, 45),
            (2, 45),
            (3, 50),
            (4, 50),
            (5, 50),
            (6, 50),
            (7, 50),
            (8, 60),
            (9, 60),
            (10, 60),
            (11, 60),
            (12, 40),  # weak mid to sell
            (13, 70),
            (14, 70),
            (15, 70),
        )
    ]
    state = SquadState.model_validate(
        {
            "bank": 30,
            "free_transfers": 1,
            "ft_bank_cap": 2,
            "hit_cost": 4,
            "decay": 1.0,
            "bench_weight": 0.0,
            "owned": [player.model_dump() for player in owned],
        }
    )
    # Strong replacement mid 16 with high xPts; mid 12 is weak.
    xpts = [2.0] * 16
    xpts[11] = 0.5  # player 12
    xpts[15] = 9.0  # player 16
    points = pl.DataFrame(
        {
            "player_key": list(range(1, 17)),
            "step": [0] * 16,
            "xpts": xpts,
        }
    )
    candidates = pool.drop("xpts")
    plan = plan_squad(
        candidates,
        points,
        state,
        initial=False,
        horizon=1,
    )
    explain = build_decision_explain(plan, state, candidates)
    assert explain.is_draft is False
    assert explain.baseline_xi_xpts is not None
    assert explain.net_delta is not None
    assert explain.transfers_count >= 1
    assert explain.hit in (0.0, 4.0)
    # Planned XI should beat keep-squad baseline after accounting for hits.
    assert explain.planned_xi_xpts >= explain.baseline_xi_xpts
    assert explain.net_delta == pytest.approx(
        explain.planned_xi_xpts - explain.baseline_xi_xpts - explain.hit
    )
    if not explain.transfers.is_empty():
        row = explain.transfers.row(0, named=True)
        assert float(row["delta_xpts"]) == pytest.approx(
            float(row["xpts_in"]) - float(row["xpts_out"])
        )
    text = format_decision_explain(explain, names={16: "Haul", 15: "Weak"})
    assert "Decision explain" in text
    assert "Constraints touched" in text
    assert "Net delta" in text


def test_decision_explain_draft_has_no_baseline() -> None:
    pool = _pool()
    points = pool.select("player_key", pl.lit(0).alias("step"), "xpts")
    state = SquadState.model_validate(
        {
            "bank": 1000,
            "free_transfers": 1,
            "ft_bank_cap": 2,
            "hit_cost": 4,
            "decay": 1.0,
            "bench_weight": 0.0,
            "owned": [],
        }
    )
    plan = plan_squad(pool.drop("xpts"), points, state, initial=True, horizon=1)
    explain = build_decision_explain(plan, state, pool.drop("xpts"))
    assert explain.is_draft is True
    assert explain.baseline_xi_xpts is None
    assert explain.net_delta is None
    assert explain.transfers.is_empty()
    assert "draft" in format_decision_explain(explain).lower()


def test_append_explain_sections_joins_markdown() -> None:
    out = append_explain_sections(
        "# season\n",
        decision="### Decision explain\n\nok\n",
        forecast="### Forecast board\n\nok\n",
    )
    assert out.startswith("# season")
    assert "Decision explain" in out
    assert "Forecast board" in out
    assert out.endswith("\n")


def test_build_forecast_board_rejects_bad_top_n() -> None:
    rules = Rules.for_season("2024-25")
    preds = pl.DataFrame(
        {
            "player_key": [1],
            "position": ["MID"],
            "target_gw": [1],
            "xpts": [1.0],
        }
    )
    with pytest.raises(ValueError, match="top_n"):
        build_forecast_board(preds, season="2024-25", rules=rules, top_n=0)


def _pool() -> pl.DataFrame:
    """Fifteen squad players plus one haul mid (16) as a transfer target."""
    rows: list[dict[str, object]] = []
    # 2 GK, 5 DEF, 5 MID, 3 FWD = 15; plus mid 16.
    positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3 + ["MID"]
    for index, position in enumerate(positions, start=1):
        price = 45
        xpts = 2.0
        if index == 12:
            price = 40
            xpts = 0.5
        elif index == 16:
            price = 70
            xpts = 9.0
        rows.append(
            {
                "player_key": index,
                "position": position,
                "team_key": ((index - 1) % 10) + 1,
                "price": price,
                "xpts": xpts,
            }
        )
    return pl.DataFrame(rows)
