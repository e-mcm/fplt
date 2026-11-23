"""Constrained hindsight oracle, attainable share, and component regret."""

from __future__ import annotations

import polars as pl
import pytest

from fplt.optimise.attainable import (
    ORACLE_MODEL_ID,
    attainable_share,
    chip_timing_regret,
    component_regret,
    enrich_decision_summary,
    foresight_predictions,
    format_attainable_scoreboard,
    is_oracle_model_id,
    oracle_ledger_model_id,
    simulate_hindsight_oracle,
    summarise_attainable,
)
from fplt.optimise.decisions import simulate_decisions
from fplt.optimise.realised import format_decision_scoreboard, summarise_decisions
from fplt.optimise.regret_board import format_regret_report
from fplt.optimise.state import SquadState


def test_attainable_share_is_realised_over_oracle() -> None:
    assert attainable_share(80, 100) == 0.8
    assert attainable_share(10, 0) == 0.0
    assert attainable_share(0, 50) == 0.0


def test_chip_timing_hook_is_chips_off_by_default() -> None:
    placeholder = chip_timing_regret()
    assert placeholder.status == "chips_off"
    assert placeholder.regret is None
    assert placeholder.chip is None
    computed = chip_timing_regret(chips_enabled=True, chip="bboost", regret=12)
    assert computed.status == "computed"
    assert computed.regret == 12
    assert computed.chip == "bboost"


def test_foresight_predictions_expose_future_actuals() -> None:
    actuals = pl.DataFrame(
        {
            "gw": [1, 1, 2, 2],
            "player_key": [1, 2, 1, 2],
            "actual_pts": [3, 5, 7, 1],
        }
    )
    preds = foresight_predictions(actuals, [1, 2])
    assert preds.filter(pl.col("as_of_gw") == 1).height == 4
    assert preds.filter(pl.col("as_of_gw") == 2).height == 2
    gw2 = preds.filter((pl.col("as_of_gw") == 1) & (pl.col("target_gw") == 2))
    assert set(gw2["xpts"].to_list()) == {7.0, 1.0}
    assert foresight_predictions(actuals.clear(), [1]).is_empty()
    with pytest.raises(ValueError, match="missing columns"):
        foresight_predictions(pl.DataFrame({"gw": [1]}), [1])


def test_oracle_ledger_model_id_encodes_chip_arm() -> None:
    assert oracle_ledger_model_id("off") == ORACLE_MODEL_ID
    assert oracle_ledger_model_id("optimised") == f"{ORACLE_MODEL_ID}::chip_optimised"
    assert oracle_ledger_model_id("rule_based") == f"{ORACLE_MODEL_ID}::chip_rule_based"
    assert is_oracle_model_id(ORACLE_MODEL_ID)
    assert is_oracle_model_id(oracle_ledger_model_id("optimised"))
    assert not is_oracle_model_id("dc_layered_v2::chip_optimised")


def test_chip_timing_regret_when_tc_beats_chosen_captain_chip() -> None:
    """Bench Boost on a haul-heavy bench beats no scoring chip."""
    picks = _decision_gw(
        xi_pts=[2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 2],
        captain_index=1,
        bench_mid_pts=9,
    ).with_columns(pl.lit(None).cast(pl.Utf8).alias("chip"))
    # Mark season as chips-on by playing TC on a weak captain week elsewhere —
    # use BB here so the local opt prefers Bench Boost.
    picks = picks.with_columns(pl.lit("3xc").alias("chip"))
    # Captain has only 3 pts; BB adds the 9-pt bench mid → better than TC (+6).
    regret = component_regret(picks)
    assert regret["chip_timing_regret"][0] is not None
    assert int(regret["chip_timing_regret"][0]) >= 3


def test_summarise_attainable_accepts_chips_on_oracle_id() -> None:
    gw1 = _decision_gw(xi_pts=[2] * 11, captain_index=0)
    decisions = gw1.with_columns(pl.lit(None).cast(pl.Utf8).alias("chip"))
    oracle = decisions.with_columns(
        pl.lit(oracle_ledger_model_id("optimised")).alias("model_id"),
        pl.when(pl.col("captain"))
        .then(pl.lit(20))
        .otherwise(pl.col("actual_pts"))
        .alias("actual_pts"),
    )
    full = summarise_attainable(decisions, oracle)
    assert full.height == 1
    assert full["attainable_points"][0] > full["total_points"][0]


def test_format_regret_report_lists_components() -> None:
    gw1 = _decision_gw(xi_pts=[2] * 11, captain_index=0)
    decisions = gw1.with_columns(pl.lit("bboost").alias("chip"))
    oracle = decisions.with_columns(pl.lit(ORACLE_MODEL_ID).alias("model_id"))
    summary = summarise_attainable(decisions, oracle)
    per_gw = component_regret(decisions)
    text = format_regret_report(
        summary,
        per_gw,
        meta={"season": "2024-25", "policy_points": 10, "oracle_points": 20},
    )
    assert "Captaincy" in text
    assert "Chip timing" in text


def test_hindsight_oracle_beats_or_matches_blind_policy() -> None:
    pool = _pool()
    candidates = pl.concat([pool.with_columns(pl.lit(gw).alias("gw")) for gw in (1, 2)])
    predictions = pl.DataFrame(
        {
            "as_of_gw": [gw for gw in (1, 2) for _ in range(pool.height)],
            "target_gw": [gw for gw in (1, 2) for _ in range(pool.height)],
            "player_key": pool["player_key"].to_list() * 2,
            "xpts": [1.0] * (pool.height * 2),
        }
    )
    haul_key = int(pool.filter(pl.col("position") == "FWD")["player_key"][0])
    actual_pts = [
        12 if int(key) == haul_key else 1 for key in pool["player_key"].to_list()
    ]
    actuals = pl.DataFrame(
        {
            "gw": [gw for gw in (1, 2) for _ in range(pool.height)],
            "player_key": pool["player_key"].to_list() * 2,
            "actual_pts": actual_pts * 2,
            "actual_minutes": [90] * (pool.height * 2),
        }
    )
    policy = simulate_decisions(
        candidates,
        predictions,
        actuals,
        _state(),
        model_id="blind",
        season="2024-25",
        gameweeks=[1, 2],
        horizon=1,
    )
    oracle = simulate_hindsight_oracle(
        candidates,
        actuals,
        _state(),
        season="2024-25",
        gameweeks=[1, 2],
        horizon=1,
    )
    assert oracle["model_id"].unique().to_list() == [ORACLE_MODEL_ID]
    policy_total = int(summarise_decisions(policy)["total_points"][0])
    oracle_total = int(summarise_decisions(oracle)["total_points"][0])
    assert oracle_total >= policy_total
    assert 0.0 <= attainable_share(policy_total, oracle_total) <= 1.0


def test_hindsight_oracle_optimised_chips_beats_chips_off() -> None:
    """Optimised chips-on hindsight should not score below chips-off."""
    candidates, actuals, fixtures = _chip_toy()
    off = simulate_hindsight_oracle(
        candidates,
        actuals,
        _state(),
        season="2024-25",
        gameweeks=[1, 2, 3],
        horizon=2,
        chip_arm="off",
    )
    on = simulate_hindsight_oracle(
        candidates,
        actuals,
        _state(),
        season="2024-25",
        gameweeks=[1, 2, 3],
        horizon=2,
        chip_arm="optimised",
        fixtures=fixtures,
        force_play=False,
    )
    assert on["model_id"].unique().to_list() == [oracle_ledger_model_id("optimised")]
    chips_played = [c for c in on["chip"].unique().to_list() if c is not None]
    assert chips_played, "optimised hindsight should play at least one chip"
    off_total = int(summarise_decisions(off)["total_points"][0])
    on_total = int(summarise_decisions(on)["total_points"][0])
    assert on_total >= off_total


def test_captaincy_regret_counts_best_armband_in_xi() -> None:
    picks = _decision_gw(
        xi_pts=[2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 8],
        captain_index=0,
    )
    regret = component_regret(picks)
    assert regret.height == 1
    assert int(regret["captaincy_regret"][0]) == 6
    assert regret["chip_timing_regret"][0] is None


def test_xi_bench_regret_when_haul_is_benched() -> None:
    picks = _decision_gw(
        xi_pts=[2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 2],
        captain_index=1,
        bench_mid_pts=9,
    )
    regret = component_regret(picks)
    assert int(regret["xi_bench_regret"][0]) >= 8


def test_enrich_and_format_scoreboard_show_share() -> None:
    gw1 = _decision_gw(xi_pts=[2] * 11, captain_index=0)
    gw2 = gw1.with_columns(pl.lit(2).cast(pl.Int64).alias("gw"))
    decisions = pl.concat([gw1, gw2])
    oracle = decisions.with_columns(
        pl.lit(ORACLE_MODEL_ID).alias("model_id"),
        pl.when(pl.col("captain"))
        .then(pl.lit(20))
        .otherwise(pl.col("actual_pts"))
        .alias("actual_pts"),
    )
    summary = summarise_decisions(decisions)
    enriched = enrich_decision_summary(summary, oracle)
    assert enriched["attainable_points"][0] > enriched["total_points"][0]
    table = format_decision_scoreboard(enriched)
    assert "Attainable" in table
    assert "Share" in table

    full = summarise_attainable(decisions, oracle)
    assert full["share_of_attainable"][0] <= 1.0
    assert "captaincy_regret" in full.columns
    board = format_attainable_scoreboard(full)
    assert "Cap regret" in board
    assert format_attainable_scoreboard(full.clear()) == "_No decision rows to score._"


def test_component_regret_rejects_missing_columns() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        component_regret(pl.DataFrame({"model_id": ["a"]}))


def _state(**overrides: object) -> SquadState:
    base: dict[str, object] = {
        "bank": 1000,
        "free_transfers": 1,
        "ft_bank_cap": 5,
        "hit_cost": 4,
        "decay": 1.0,
        "bench_weight": 0.0,
        "owned": [],
    }
    base.update(overrides)
    return SquadState.model_validate(base)


def _pool() -> pl.DataFrame:
    """Candidate pool matching ``tests/unit/test_plan.py``."""
    rows: list[dict[str, object]] = []
    player = 1
    for club in range(1, 7):
        for position in ("GK", "DEF", "DEF", "MID", "FWD"):
            rows.append(
                {
                    "player_key": player,
                    "position": position,
                    "team_key": club,
                    "price": 40 + club,
                }
            )
            player += 1
    return pl.DataFrame(rows)


def _chip_toy() -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Three-GW toy season with a DGW so the chip scheduler has a landmark."""
    cand_rows: list[dict[str, object]] = []
    act_rows: list[dict[str, object]] = []
    player = 1
    for club in range(1, 7):
        for position in ("GK", "DEF", "DEF", "MID", "FWD"):
            for gw in (1, 2, 3):
                # Haul in GW2 so TC/BB has something to amplify under foresight.
                pts = 15 if gw == 2 and position == "FWD" and club == 6 else 2 + gw
                cand_rows.append(
                    {
                        "gw": gw,
                        "player_key": player,
                        "position": position,
                        "team_key": club,
                        "price": 40 + club,
                    }
                )
                act_rows.append(
                    {
                        "gw": gw,
                        "player_key": player,
                        "actual_pts": pts,
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
    return pl.DataFrame(cand_rows), pl.DataFrame(act_rows), fixtures


def _decision_gw(
    *,
    xi_pts: list[int],
    captain_index: int,
    bench_mid_pts: int = 0,
    hit: float = 0.0,
) -> pl.DataFrame:
    """Build one model × GW decision frame with a valid 2/5/5/3 fifteen."""
    # XI 1/3/4/3 + bench GK/DEF/DEF/MID → squad 2/5/5/3.
    positions = [
        "GK",
        "DEF",
        "DEF",
        "DEF",
        "MID",
        "MID",
        "MID",
        "MID",
        "FWD",
        "FWD",
        "FWD",
    ]
    rows: list[dict[str, object]] = []
    for index, (pts, position) in enumerate(zip(xi_pts, positions, strict=True)):
        rows.append(
            _row(
                key=index + 1,
                position=position,
                in_xi=True,
                captain=index == captain_index,
                vice=index == (captain_index + 1) % 11,
                pts=pts,
                mins=90,
                hit=hit,
                team_key=(index % 7) + 1,
            )
        )
    rows.extend(
        [
            _row(
                key=12,
                position="GK",
                in_xi=False,
                captain=False,
                vice=False,
                pts=0,
                mins=0,
                hit=hit,
                team_key=8,
            ),
            _row(
                key=13,
                position="DEF",
                in_xi=False,
                captain=False,
                vice=False,
                pts=0,
                mins=0,
                hit=hit,
                team_key=8,
            ),
            _row(
                key=14,
                position="DEF",
                in_xi=False,
                captain=False,
                vice=False,
                pts=0,
                mins=0,
                hit=hit,
                team_key=9,
            ),
            _row(
                key=15,
                position="MID",
                in_xi=False,
                captain=False,
                vice=False,
                pts=bench_mid_pts,
                mins=90 if bench_mid_pts else 0,
                hit=hit,
                team_key=10,
            ),
        ]
    )
    return pl.DataFrame(rows)


def _row(
    *,
    key: int,
    position: str,
    in_xi: bool,
    captain: bool,
    vice: bool,
    pts: int,
    mins: int,
    hit: float,
    team_key: int,
) -> dict[str, object]:
    return {
        "model_id": "policy",
        "season": "2024-25",
        "gw": 1,
        "player_key": key,
        "position": position,
        "in_xi": in_xi,
        "captain": captain,
        "vice": vice,
        "bought": False,
        "sold": False,
        "xpts": float(pts),
        "actual_pts": pts,
        "actual_minutes": mins,
        "hit": hit,
        "free_transfers": 1,
        "team_key": team_key,
        "price": 50,
    }
