"""Three-arm chip decision board on a tiny synthetic season."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import polars as pl
import pytest

from fplt.optimise.chip_windows import ChipUseWindows, rule_based_chip_windows
from fplt.optimise.decisions import (
    format_chip_decision_scoreboard,
    simulate_decisions,
    summarise_chip_decisions,
)
from fplt.optimise.plan import plan_squad
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


def test_decision_rows_accept_chip_after_leading_nulls() -> None:
    """Null chips early must not freeze the column before a later chip name."""
    from fplt.optimise.decisions import _empty_decisions

    schema = _empty_decisions().schema
    rows: list[dict[str, object]] = []
    for index in range(120):
        rows.append(
            {
                "model_id": "m",
                "season": "2025-26",
                "gw": 1 + index // 15,
                "player_key": index,
                "position": "MID",
                "in_xi": True,
                "captain": False,
                "vice": False,
                "bought": False,
                "sold": False,
                "xpts": 1.0,
                "actual_pts": 1,
                "actual_minutes": 90,
                "hit": 0.0,
                "free_transfers": 1,
                "chip": None,
            }
        )
    rows.append({**rows[-1], "gw": 20, "player_key": 999, "chip": "bboost"})
    frame = pl.DataFrame(rows, schema=schema)
    assert frame.filter(pl.col("chip") == "bboost").height == 1


def test_simulate_decisions_forwards_lineup_reopt(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """``lineup_reopt=False`` reaches ``plan_squad`` (MILP-only ablation)."""
    candidates, predictions, actuals, _fixtures = _toy_season()
    seen: dict[str, object] = {}
    real = plan_squad

    def spy_plan(*args: object, **kwargs: object):
        seen.update(kwargs)
        return real(*args, **kwargs)

    monkeypatch.setattr("fplt.optimise.decisions.plan_squad", spy_plan)
    simulate_decisions(
        candidates,
        predictions,
        actuals,
        _draft(),
        model_id="reopt_test",
        season="2024-25",
        gameweeks=[1],
        horizon=1,
        chip_arm="off",
        lineup_reopt=False,
    )
    assert seen["lineup_reopt"] is False


def test_simulate_decisions_forwards_board_highs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Chip-board HiGHS options reach ``plan_squad`` (FDB-228)."""
    from fplt.optimise.highs_options import HighsSolveOptions
    from fplt.optimise.plan import GameweekPlan

    candidates, predictions, actuals, _fixtures = _toy_season()
    seen: dict[str, object] = {}
    options = HighsSolveOptions.board_deterministic()
    empty_picks = pl.DataFrame(
        schema={
            "player_key": pl.Int64,
            "position": pl.Utf8,
            "team_key": pl.Int64,
            "price": pl.Int64,
            "step": pl.Int64,
            "in_xi": pl.Boolean,
            "captain": pl.Boolean,
            "vice": pl.Boolean,
            "bought": pl.Boolean,
            "sold": pl.Boolean,
            "xpts": pl.Float64,
        }
    )

    def stub_plan(*_args: object, **kwargs: object) -> GameweekPlan:
        seen.update(kwargs)
        return GameweekPlan(
            picks=empty_picks,
            hits=(0.0,),
            free_transfers=(1,),
            points=0.0,
            chips=(None,),
        )

    monkeypatch.setattr("fplt.optimise.decisions.plan_squad", stub_plan)
    monkeypatch.setattr(
        "fplt.optimise.decisions._decision_rows",
        lambda *_a, **_k: [],
    )
    monkeypatch.setattr(
        "fplt.optimise.decisions._advance",
        lambda state, *_a, **_k: state,
    )
    frame = simulate_decisions(
        candidates,
        predictions,
        actuals,
        _draft(),
        model_id="highs_test",
        season="2024-25",
        gameweeks=[1],
        horizon=1,
        chip_arm="off",
        highs_options=options,
    )
    assert seen["highs_options"] == options
    assert frame.is_empty()


def test_chip_board_defaults_to_deterministic_highs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``run_chip_decision_board`` passes board_deterministic into run_decisions."""
    from fplt.optimise.decisions import run_chip_decision_board
    from fplt.optimise.highs_options import HighsSolveOptions

    captured: dict[str, object] = {}

    def fake_run_decisions(*_args: object, **kwargs: object) -> Path:
        captured.update(kwargs)
        return tmp_path / "decisions.parquet"

    monkeypatch.setattr("fplt.optimise.decisions.run_decisions", fake_run_decisions)
    monkeypatch.setattr(
        "fplt.optimise.decisions.load_model_specs",
        lambda: {"rolling_mean_v1": type("S", (), {"model_id": "m"})()},
    )
    monkeypatch.setattr(
        "fplt.optimise.decisions.read_decisions",
        lambda _root: __import__("polars").DataFrame(
            schema={
                "model_id": pl.Utf8,
                "season": pl.Utf8,
                "gw": pl.Int64,
                "player_key": pl.Int64,
                "position": pl.Utf8,
                "in_xi": pl.Boolean,
                "captain": pl.Boolean,
                "vice": pl.Boolean,
                "bought": pl.Boolean,
                "sold": pl.Boolean,
                "xpts": pl.Float64,
                "actual_pts": pl.Float64,
                "actual_minutes": pl.Float64,
                "hit": pl.Float64,
                "free_transfers": pl.Int64,
                "chip": pl.Utf8,
            }
        ),
    )
    monkeypatch.setattr(
        "fplt.optimise.decisions.summarise_chip_decisions",
        lambda *_a, **_k: pl.DataFrame(
            {
                "model_id": ["m::chip_optimised"],
                "base_model_id": ["m"],
                "season": ["2025-26"],
                "chip_arm": ["optimised"],
                "rules_label": ["2025-26"],
                "counterfactual": [False],
                "total_points": [0],
                "gameweeks": [0],
                "per_gw": [[]],
            }
        ),
    )
    run_chip_decision_board(
        tmp_path,
        ["rolling_mean_v1"],
        "2025-26",
        1,
        1,
        3,
        arms=["optimised"],
        simulate=True,
    )
    assert captured["highs_options"] == HighsSolveOptions.board_deterministic()


def test_chip_board_cli_passes_no_lineup_reopt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``fplt decisions --chip-board --no-lineup-reopt`` reaches the board."""
    from fplt.cli import main
    from fplt.settings import get_settings

    monkeypatch.setenv("FPLT_DATA_DIR", str(tmp_path))
    get_settings.cache_clear()
    fake = pl.DataFrame(
        {
            "model_id": ["m::chip_optimised"],
            "base_model_id": ["m"],
            "season": ["2025-26"],
            "chip_arm": ["optimised"],
            "rules_label": ["2025-26"],
            "counterfactual": [False],
            "total_points": [100],
            "gameweeks": [2],
            "per_gw": [[50, 50]],
        }
    )
    with patch("fplt.cli.run_chip_decision_board", return_value=fake) as run_mock:
        try:
            main(
                [
                    "decisions",
                    "--chip-board",
                    "--season",
                    "2025-26",
                    "--horizon",
                    "3",
                    "--models",
                    "rolling_mean_v1",
                    "--from-gw",
                    "1",
                    "--to-gw",
                    "2",
                    "--chip-arms",
                    "optimised",
                    "--summarise-only",
                    "--no-lineup-reopt",
                ]
            )
        finally:
            get_settings.cache_clear()
    assert run_mock.call_args.kwargs["lineup_reopt"] is False


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
