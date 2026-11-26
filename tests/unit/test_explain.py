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


def test_decision_explain_soft_fails_when_baseline_budget_breaks() -> None:
    """Keep-squad list prices over £100m must not kill transfers / planned XI."""
    pool = _expensive_owned_pool()
    # Selling prices fit the plan; candidate list prices blow the baseline.
    owned = [OwnedPlayer(player_key=key, selling_price=45) for key in range(1, 16)]
    state = SquadState.model_validate(
        {
            "bank": 0,
            "free_transfers": 1,
            "ft_bank_cap": 2,
            "hit_cost": 4,
            "decay": 1.0,
            "bench_weight": 0.0,
            "owned": [player.model_dump() for player in owned],
        }
    )
    xpts = [2.0] * 16
    xpts[11] = 0.5
    xpts[15] = 9.0
    points = pl.DataFrame(
        {
            "player_key": list(range(1, 17)),
            "step": [0] * 16,
            "xpts": xpts,
        }
    )
    # Owned list prices 15×70 = 1050 > 1000 → baseline select_squad fails.
    expensive = pool.drop("xpts").with_columns(
        pl.when(pl.col("player_key") <= 15)
        .then(pl.lit(70))
        .otherwise(pl.col("price"))
        .alias("price")
    )
    plan = plan_squad(
        expensive,
        points,
        state,
        initial=False,
        horizon=1,
    )
    explain = build_decision_explain(plan, state, expensive)
    assert explain.is_draft is False
    assert explain.baseline_xi_xpts is None
    assert explain.net_delta is None
    assert any("baseline unavailable" in note for note in explain.constraints)
    assert explain.planned_xi_xpts > 0
    text = format_decision_explain(explain)
    assert "baseline unavailable" in text.lower() or "—" in text


def test_explain_plan_markdown_emits_board_when_baseline_fails(
    tmp_path,
) -> None:
    from fplt.data.pit import PITData
    from fplt.data.store import DataStore
    from fplt.optimise.explain import explain_plan_markdown
    from fplt.types import GWKey
    from tests.builders import build_league

    store = build_league(DataStore(tmp_path))
    view = PITData(store).view(GWKey("2024-25", 2))
    rules = Rules.for_season("2024-25")
    pool = _expensive_owned_pool()
    owned = [OwnedPlayer(player_key=key, selling_price=45) for key in range(1, 16)]
    state = SquadState.model_validate(
        {
            "bank": 0,
            "free_transfers": 1,
            "ft_bank_cap": 2,
            "hit_cost": 4,
            "decay": 1.0,
            "bench_weight": 0.0,
            "owned": [player.model_dump() for player in owned],
        }
    )
    xpts = [2.0] * 16
    xpts[11] = 0.5
    xpts[15] = 9.0
    candidates = pool.drop("xpts").with_columns(
        pl.when(pl.col("player_key") <= 15)
        .then(pl.lit(70))
        .otherwise(pl.col("price"))
        .alias("price")
    )
    points = pl.DataFrame(
        {
            "player_key": list(range(1, 17)),
            "step": [0] * 16,
            "xpts": xpts,
        }
    )
    plan = plan_squad(candidates, points, state, initial=False, horizon=1)
    preds = pl.DataFrame(
        {
            "player_key": list(range(1, 17)),
            "position": pool["position"].to_list(),
            "target_gw": [2] * 16,
            "xpts": xpts,
            "p_played": [1.0] * 16,
            "p_60": [0.8] * 16,
            "p_goal": [0.1] * 16,
            "p_assist": [0.1] * 16,
            "p_cs": [0.2] * 16,
            "p_haul": [0.05] * 16,
        }
    )
    text = explain_plan_markdown(
        plan,
        state,
        candidates,
        preds,
        season="2024-25",
        gameweek=2,
        rules=rules,
        view=view,
        parts=None,
    )
    assert "Forecast board" in text
    assert "baseline unavailable" in text.lower() or "Baseline XI" in text
    assert "Why (fixture" in text or "Starting XI depth" in text


def test_player_depth_fixture_labels_without_parts(tmp_path) -> None:
    from fplt.data.pit import PITData
    from fplt.data.store import DataStore
    from fplt.optimise.explain import build_player_depth, format_player_depth
    from fplt.types import GWKey
    from tests.builders import build_league

    store = build_league(DataStore(tmp_path))
    view = PITData(store).view(GWKey("2024-25", 2))
    rules = Rules.for_season("2024-25")
    preds = pl.DataFrame(
        {
            "player_key": [100],
            "position": ["MID"],
            "target_gw": [2],
            "xpts": [5.0],
            "p_played": [1.0],
            "p_60": [0.9],
            "p_goal": [0.2],
            "p_assist": [0.1],
            "p_cs": [0.3],
            "p_haul": [0.05],
        }
    )
    depth = build_player_depth(
        view,
        player_key=100,
        target_gw=2,
        predictions=preds,
        rules=rules,
        parts=None,
    )
    assert depth.depth.height == 1
    row = depth.depth.row(0, named=True)
    assert int(row["fixture_id"]) == 2
    assert row["is_home"] is False  # team 3 away at team 7 in GW2 builder
    assert row["team_eg"] is None
    assert row["xmins"] is None
    text = format_player_depth(depth, names={100: "A"})
    assert "Player depth" in text
    assert "A" in text


def test_player_depth_fills_layered_fields(tmp_path) -> None:
    import numpy as np

    from fplt.data.pit import PITData
    from fplt.data.store import DataStore
    from fplt.models.distributions import (
        MAX_GOALS,
        MINUTE_BINS,
        MinutesDist,
        ScorelineDist,
    )
    from fplt.optimise.explain import (
        LayeredExplainParts,
        build_player_depth,
        xmins_from_minutes_pmf,
    )
    from fplt.types import GWKey
    from tests.builders import build_league

    store = build_league(DataStore(tmp_path))
    view = PITData(store).view(GWKey("2024-25", 2))
    rules = Rules.for_season("2024-25")
    goals = MAX_GOALS + 1
    pmf = np.zeros((1, goals + 1, goals + 1))
    pmf[0, 2, 1] = 1.0

    class _Score:
        def predict(self, _view, fixtures):
            ids = fixtures["fixture_id"].to_numpy()
            grid = np.zeros((len(ids), goals + 1, goals + 1))
            for index in range(len(ids)):
                grid[index, 2, 1] = 1.0
            return ScorelineDist(ids, grid)

    class _Minutes:
        def predict(self, _view, asked):
            n = asked.height
            pmf_m = np.zeros((n, len(MINUTE_BINS)))
            pmf_m[:, list(MINUTE_BINS).index(90)] = 1.0
            keys = asked.select("player_key", "fixture_id")
            return MinutesDist(keys, MINUTE_BINS, pmf_m)

    class _Involvement:
        def predict(self, _view, asked):
            return asked.select("player_key", "fixture_id").with_columns(
                pl.lit(0.4).alias("goal_weight"),
                pl.lit(0.25).alias("assist_weight"),
            )

    parts = LayeredExplainParts(
        score=_Score(), minutes=_Minutes(), involvement=_Involvement()
    )
    preds = pl.DataFrame(
        {
            "player_key": [200],
            "position": ["FWD"],
            "target_gw": [2],
            "xpts": [4.0],
            "p_played": [1.0],
            "p_60": [1.0],
            "p_goal": [0.3],
            "p_assist": [0.1],
            "p_cs": [0.2],
            "p_haul": [0.1],
        }
    )
    depth = build_player_depth(
        view,
        player_key=200,
        target_gw=2,
        predictions=preds,
        rules=rules,
        parts=parts,
    )
    row = depth.depth.row(0, named=True)
    assert float(row["goal_weight"]) == pytest.approx(0.4)
    assert float(row["assist_weight"]) == pytest.approx(0.25)
    assert float(row["xmins"]) == pytest.approx(90.0)
    # Player 200 is team 7, home in GW2 fixture 2 vs team 3.
    assert row["is_home"] is True
    assert float(row["team_eg"]) == pytest.approx(2.0)
    assert float(row["opp_eg"]) == pytest.approx(1.0)
    xmins = xmins_from_minutes_pmf(
        _Minutes().predict(view, depth.depth.select("player_key", "fixture_id"))
    )
    assert float(xmins["xmins"][0]) == pytest.approx(90.0)


def test_layered_explain_parts_none_for_opaque() -> None:
    from fplt.optimise.explain import layered_explain_parts

    assert layered_explain_parts(object()) is None


def test_transfer_pair_depth_markdown(tmp_path) -> None:
    from fplt.data.pit import PITData
    from fplt.data.store import DataStore
    from fplt.optimise.explain import (
        build_transfer_pair_depth,
        format_transfer_pair_depth,
    )
    from fplt.types import GWKey
    from tests.builders import build_league

    store = build_league(DataStore(tmp_path))
    view = PITData(store).view(GWKey("2024-25", 2))
    rules = Rules.for_season("2024-25")
    preds = pl.DataFrame(
        {
            "player_key": [100, 200],
            "position": ["MID", "FWD"],
            "target_gw": [2, 2],
            "xpts": [4.0, 5.0],
            "p_played": [1.0, 1.0],
            "p_60": [0.8, 0.9],
            "p_goal": [0.1, 0.2],
            "p_assist": [0.1, 0.1],
            "p_cs": [0.2, 0.1],
            "p_haul": [0.05, 0.1],
        }
    )
    pair = build_transfer_pair_depth(
        view,
        player_out=100,
        player_in=200,
        target_gw=2,
        predictions=preds,
        rules=rules,
        parts=None,
    )
    text = format_transfer_pair_depth(pair, names={100: "A", 200: "B"})
    assert "A → B" in text
    assert "Forecast board" in text


def _expensive_owned_pool() -> pl.DataFrame:
    """Fifteen owned players plus cheap haul mid 16."""
    rows: list[dict[str, object]] = []
    positions = ["GK"] * 2 + ["DEF"] * 5 + ["MID"] * 5 + ["FWD"] * 3 + ["MID"]
    for index, position in enumerate(positions, start=1):
        price = 45 if index == 16 else 70
        xpts = 9.0 if index == 16 else (0.5 if index == 12 else 2.0)
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
