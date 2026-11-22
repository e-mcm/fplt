"""Three-arm chip decision board on a tiny synthetic season."""

from __future__ import annotations

import polars as pl

from fplt.optimise.chip_windows import ChipUseWindows, rule_based_chip_windows
from fplt.optimise.decisions import (
    format_chip_decision_scoreboard,
    simulate_decisions,
    summarise_chip_decisions,
)
from fplt.optimise.state import ChipInventory, SquadState
from fplt.scoring.chips import ChipRules


def test_three_arms_produce_distinct_ledger_ids_and_labels() -> None:
    candidates, predictions, actuals, fixtures = _toy_season()
    state = _draft()
    gameweeks = [1, 2, 3]
    frames: list[pl.DataFrame] = []
    for arm, model_id in (
        ("off", "toy::chip_off"),
        ("rule_based", "toy::chip_rule_based"),
        ("optimised", "toy::chip_optimised"),
    ):
        # Use explicit model ids; simulate_decisions does not rewrite them.
        windows = None
        if arm == "rule_based":
            windows = rule_based_chip_windows(
                fixtures,
                season="2024-25",
                rules=ChipRules.for_season("2025-26"),
                predictions=predictions.rename({"target_gw": "gw"}),
                gameweeks=gameweeks,
            )
        frame = simulate_decisions(
            candidates,
            predictions,
            actuals,
            state,
            model_id=model_id if arm != "off" else "toy",
            season="2024-25",
            gameweeks=gameweeks,
            horizon=1,
            chip_arm=arm,  # type: ignore[arg-type]
            chip_rules_season="2025-26",
            chip_windows=windows,
            fixtures=fixtures,
        )
        frames.append(frame)
        if arm == "off":
            assert set(frame["chip"].to_list()) == {None}
        else:
            # At least the chip column is present; forced/optimised may or may
            # not fire on this tiny horizon depending on window start (GW2+).
            assert "chip" in frame.columns

    board = summarise_chip_decisions(
        pl.concat(frames),
        season="2024-25",
        chip_rules_season="2025-26",
    )
    assert board.height == 3
    assert set(board["chip_arm"].to_list()) == {"off", "rule_based", "optimised"}
    assert board["counterfactual"].to_list() == [True, True, True]
    assert "2024-25-under-2025/26-rules" in board["rules_label"].to_list()
    table = format_chip_decision_scoreboard(board)
    assert "counterfactual" in table
    assert "rule_based" in table


def test_free_hit_advance_keeps_permanent_squad() -> None:
    candidates, predictions, actuals, _fixtures = _toy_season()
    # Force Free Hit in GW2 via windows; start owned after GW1 draft path.
    state = _draft()
    # Draft GW1 chips-off, then play GW2 with forced FH under 2025-26 rules.
    first = simulate_decisions(
        candidates,
        predictions,
        actuals,
        state,
        model_id="fh_test",
        season="2024-25",
        gameweeks=[1],
        horizon=1,
        chip_arm="off",
    )
    owned_keys = set(
        first.filter((pl.col("gw") == 1) & ~pl.col("sold"))["player_key"].to_list()
    )
    assert len(owned_keys) == 15
    from fplt.optimise.state import OwnedPlayer

    owned = [
        OwnedPlayer(player_key=key, selling_price=45) for key in sorted(owned_keys)
    ]
    after = SquadState(
        bank=100,
        free_transfers=1,
        ft_bank_cap=2,
        hit_cost=4,
        decay=1.0,
        bench_weight=0.0,
        owned=owned,
        chips=ChipInventory(
            remaining={"wildcard": 0, "freehit": 1, "bboost": 0, "3xc": 0},
            played=[],
        ),
    )
    windows = ChipUseWindows(
        allowed={"freehit": frozenset({2})},
        forced={"freehit": frozenset({2})},
    )
    second = simulate_decisions(
        candidates,
        predictions,
        actuals,
        after,
        model_id="fh_test",
        season="2024-25",
        gameweeks=[2],
        horizon=2,
        chip_arm="optimised",
        chip_rules_season="2025-26",
        chip_windows=windows,
    )
    assert second.filter(pl.col("gw") == 2)["chip"][0] == "freehit"


def _draft() -> SquadState:
    return SquadState(
        bank=1000,
        free_transfers=1,
        ft_bank_cap=2,
        hit_cost=4,
        decay=1.0,
        bench_weight=0.0,
        owned=[],
        chips=ChipInventory(
            remaining={"wildcard": 0, "freehit": 0, "bboost": 0, "3xc": 0},
            played=[],
        ),
    )


def _toy_season() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Three gameweeks, six clubs, enough players for a legal draft."""
    cand_rows: list[dict[str, object]] = []
    pred_rows: list[dict[str, object]] = []
    act_rows: list[dict[str, object]] = []
    player = 1
    for club in range(1, 7):
        for position in ("GK", "DEF", "DEF", "MID", "FWD"):
            for gw in (1, 2, 3):
                cand_rows.append(
                    {
                        "gw": gw,
                        "player_key": player,
                        "position": position,
                        "team_key": club,
                        "price": 40 + club,
                    }
                )
                xpts = float(club + gw)
                pred_rows.append(
                    {
                        "as_of_gw": gw,
                        "target_gw": gw,
                        "player_key": player,
                        "xpts": xpts,
                    }
                )
                # Horizon look-ahead stubs for GW2→3 when horizon=2.
                if gw < 3:
                    pred_rows.append(
                        {
                            "as_of_gw": gw,
                            "target_gw": gw + 1,
                            "player_key": player,
                            "xpts": xpts * 0.9,
                        }
                    )
                act_rows.append(
                    {
                        "gw": gw,
                        "player_key": player,
                        "actual_pts": int(xpts),
                        "actual_minutes": 90,
                    }
                )
            player += 1
    fixtures = pl.DataFrame(
        {
            "season": ["2024-25", "2024-25", "2024-25", "2024-25"],
            "gw": [2, 2, 3, 3],
            "team_h_key": [1, 1, 2, 3],
            "team_a_key": [2, 3, 4, 5],
        }
    )
    return (
        pl.DataFrame(cand_rows),
        pl.DataFrame(pred_rows),
        pl.DataFrame(act_rows),
        fixtures,
    )
