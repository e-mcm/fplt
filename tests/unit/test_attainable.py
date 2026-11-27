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


def test_oracle_pin_rejects_bad_fields() -> None:
    from fplt.optimise.attainable import OraclePin

    with pytest.raises(ValueError, match="horizon"):
        OraclePin(horizon=0, decay=1.0)
    with pytest.raises(ValueError, match="decay"):
        OraclePin(horizon=1, decay=0.0)


def test_resolve_oracle_pin_defaults_and_requires_explicit_on_divergence() -> None:
    from fplt.optimise.attainable import OraclePin, resolve_oracle_pin

    state = _state(decay=0.9)
    pin = resolve_oracle_pin(horizon=3, state=state, policy_horizons=[None, None])
    assert pin == OraclePin(horizon=3, decay=0.9, chip_arm="off", force_play=True)
    explicit = OraclePin(horizon=8, decay=1.0, chip_arm="optimised", force_play=False)
    assert (
        resolve_oracle_pin(
            horizon=3,
            state=state,
            policy_horizons=[3, 8],
            pin=explicit,
        )
        is explicit
    )
    with pytest.raises(ValueError, match="divergent horizons"):
        resolve_oracle_pin(horizon=3, state=state, policy_horizons=[3, 8])


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


def test_resolve_oracle_pin_empty_policy_horizons() -> None:
    from fplt.optimise.attainable import OraclePin, resolve_oracle_pin

    pin = resolve_oracle_pin(horizon=5, state=_state(decay=0.85), policy_horizons=())
    assert pin == OraclePin(horizon=5, decay=0.85)


def test_actuals_from_staged_filters_and_empty() -> None:
    from fplt.optimise.attainable import actuals_from_staged

    class _Store:
        def __init__(self, frame: pl.DataFrame) -> None:
            self._frame = frame

        def read(self, _name: str) -> pl.DataFrame:
            return self._frame

    assert actuals_from_staged(_Store(pl.DataFrame()), "2024-25").is_empty()
    stats = pl.DataFrame(
        {
            "season": ["2023-24", "2024-25", "2024-25"],
            "gw": [1, 1, 1],
            "player_key": [1, 1, 2],
            "total_points": [9, 5, 3],
            "minutes": [90, 90, 45],
        }
    )
    assert actuals_from_staged(_Store(stats), "2099-00").is_empty()
    got = actuals_from_staged(_Store(stats), "2024-25").sort("player_key")
    assert got["actual_pts"].to_list() == [5, 3]
    assert got["actual_minutes"].to_list() == [90, 45]


def test_component_regret_empty_and_no_xi() -> None:
    from fplt.optimise import attainable as mod

    empty = pl.DataFrame(
        schema={
            "model_id": pl.Utf8(),
            "season": pl.Utf8(),
            "gw": pl.Int64(),
            "player_key": pl.Int64(),
            "position": pl.Utf8(),
            "in_xi": pl.Boolean(),
            "captain": pl.Boolean(),
            "vice": pl.Boolean(),
            "bought": pl.Boolean(),
            "sold": pl.Boolean(),
            "actual_pts": pl.Int64(),
            "actual_minutes": pl.Int64(),
            "hit": pl.Float64(),
            "free_transfers": pl.Int64(),
        }
    )
    assert component_regret(empty).is_empty()
    # All sold / no XI — score_gameweek rejects; helpers return zero regret.
    picks = _decision_gw(xi_pts=[2] * 11, captain_index=0).with_columns(
        pl.lit(True).alias("sold"),
        pl.lit(False).alias("in_xi"),
    )
    with pytest.raises(ValueError, match="at least one in_xi"):
        component_regret(picks)
    assert mod._captaincy_regret(picks) == 0
    assert mod._xi_bench_regret(picks) == 0


def test_summarise_attainable_empty_policy_and_missing_oracle_season() -> None:
    gw1 = _decision_gw(xi_pts=[2] * 11, captain_index=0)
    oracle = gw1.with_columns(pl.lit(ORACLE_MODEL_ID).alias("model_id"))
    # Policy empty after oracle-id filter:
    assert summarise_attainable(oracle, oracle).is_empty()
    with pytest.raises(ValueError, match="oracle decisions are empty"):
        summarise_attainable(gw1, oracle.clear())
    wrong_season = oracle.with_columns(pl.lit("2099-00").alias("season"))
    with pytest.raises(ValueError, match="oracle has no total"):
        summarise_attainable(gw1, wrong_season)


def test_enrich_decision_summary_empty_and_pre_summarised_oracle() -> None:
    empty = summarise_decisions(_decision_gw(xi_pts=[2] * 11, captain_index=0).clear())
    enriched = enrich_decision_summary(
        empty, pl.DataFrame({"season": [], "total_points": []})
    )
    assert "attainable_points" in enriched.columns
    summary = summarise_decisions(_decision_gw(xi_pts=[2] * 11, captain_index=0))
    pre = pl.DataFrame({"season": ["2024-25"], "total_points": [100]})
    out = enrich_decision_summary(summary, pre)
    assert out["attainable_points"][0] == 100
    assert out["share_of_attainable"][0] is not None


def test_format_attainable_scoreboard_share_only_and_nulls() -> None:
    board = pl.DataFrame(
        {
            "model_id": ["a"],
            "season": ["2024-25"],
            "total_points": [50],
            "gameweeks": [5],
            "attainable_points": [None],
            "share_of_attainable": [None],
        }
    )
    text = format_attainable_scoreboard(board)
    assert "Attainable" in text
    assert "Share" in text
    assert "-" in text
    # Zero gameweeks → mean 0.0
    zero_gw = board.with_columns(
        pl.lit(0).alias("gameweeks"),
        pl.lit(80).cast(pl.Int64).alias("attainable_points"),
        pl.lit(0.625).alias("share_of_attainable"),
    )
    assert "0.0" in format_attainable_scoreboard(zero_gw)


def test_transfer_regret_with_market_and_bought_sold() -> None:
    from fplt.optimise.attainable import component_regret as regret_fn

    # Start from a full XV, mark one mid sold and one bought so pre-transfer rebuilds.
    picks = _decision_gw(xi_pts=[2, 3, 4, 5, 1, 1, 1, 1, 1, 1, 2], captain_index=1)
    # Sell key 15 (bench mid), buy key 99 as MID with low points (policy hurt itself).
    sold_row = picks.filter(pl.col("player_key") == 15).with_columns(
        pl.lit(True).alias("sold"),
        pl.lit(False).alias("bought"),
        pl.lit(False).alias("in_xi"),
        pl.lit(8).alias("actual_pts"),
        pl.lit(90).alias("actual_minutes"),
    )
    bought = pl.DataFrame(
        [
            {
                "model_id": "policy",
                "season": "2024-25",
                "gw": 1,
                "player_key": 99,
                "position": "MID",
                "in_xi": False,
                "captain": False,
                "vice": False,
                "bought": True,
                "sold": False,
                "xpts": 0.0,
                "actual_pts": 0,
                "actual_minutes": 0,
                "hit": 0.0,
                "free_transfers": 1,
                "team_key": 11,
                "price": 50,
            }
        ]
    )
    kept = picks.filter(pl.col("player_key") != 15)
    part = pl.concat([kept, sold_row, bought], how="diagonal_relaxed")
    market = pl.concat(
        [
            part.select(
                "gw",
                "player_key",
                "position",
                "team_key",
                "price",
                "actual_pts",
                "actual_minutes",
            ),
            pl.DataFrame(
                {
                    "gw": [1],
                    "player_key": [100],
                    "position": ["MID"],
                    "team_key": [12],
                    "price": [45],
                    "actual_pts": [15],
                    "actual_minutes": [90],
                }
            ),
        ],
        how="diagonal_relaxed",
    )
    out = regret_fn(part, market=market)
    assert out.height == 1
    assert int(out["transfer_regret"][0]) >= 0


def test_transfer_regret_without_price_returns_zero() -> None:
    picks = _decision_gw(xi_pts=[2] * 11, captain_index=0).drop("price", "team_key")
    regret = component_regret(picks)
    assert int(regret["transfer_regret"][0]) == 0


def test_chip_timing_empty_xi_and_wildcard_as_none() -> None:
    from fplt.optimise import attainable as mod

    picks = _decision_gw(xi_pts=[2] * 11, captain_index=0).with_columns(
        pl.lit(False).alias("in_xi"),
        pl.lit("wildcard").alias("chip"),
    )
    # Season has a chip played → chips_on True, but no XI → regret 0.
    assert mod._frame_played_chips(picks) is True
    hook = mod._chip_timing_regret_gw(picks)
    assert hook.regret == 0


def test_market_for_gw_helpers_and_attach() -> None:
    from fplt.optimise import attainable as mod

    assert mod._market_for_gw(None, 1) is None
    assert mod._market_for_gw(pl.DataFrame({"player_key": [1]}).clear(), 1) is None
    bare = pl.DataFrame({"player_key": [1], "price": [50]})
    assert mod._market_for_gw(bare, 1) is bare
    gw_m = pl.DataFrame({"gw": [1, 2], "player_key": [1, 2], "price": [40, 50]})
    assert mod._market_for_gw(gw_m, 9) is None
    part = mod._market_for_gw(gw_m, 1)
    assert part is not None
    assert part.height == 1
    # Attach market fields when part lacks them.
    lean = pl.DataFrame({"player_key": [1], "actual_pts": [3]})
    market = pl.DataFrame(
        {"player_key": [1], "price": [55], "team_key": [3], "position": ["MID"]}
    )
    attached = mod._attach_market_fields(lean, market)
    assert "price" in attached.columns
    assert int(attached["price"][0]) == 55
    assert mod._attach_market_fields(lean, lean) is lean or mod._attach_market_fields(
        lean, pl.DataFrame({"player_key": [1]})
    ).equals(lean)


def test_actual_maps_and_team_key_expr() -> None:
    from fplt.optimise import attainable as mod

    part = pl.DataFrame(
        {
            "player_key": [1],
            "actual_pts": [None],
            "actual_minutes": [None],
        }
    )
    market = pl.DataFrame({"player_key": [2], "actual_pts": [7]})
    pts, mins = mod._actual_maps(part, market)
    assert pts[1] == 0
    assert pts[2] == 7
    assert mins[1] == 0
    frame = pl.DataFrame({"player_key": [1001, 2002]})
    expr = mod._team_key_expr(frame)
    out = frame.select(expr)
    assert out.columns[0] == "team_key"
    assert set(out["team_key"].to_list()) == {2, 3}


def test_pre_transfer_state_bought_and_sold() -> None:
    from fplt.optimise import attainable as mod

    part = pl.DataFrame(
        {
            "player_key": [1, 2, 3],
            "sold": [False, False, True],
            "bought": [False, True, False],
            "price": [40, 50, 45],
            "position": ["MID", "MID", "MID"],
            "in_xi": [True, True, False],
            "captain": [True, False, False],
            "vice": [False, True, False],
            "actual_pts": [5, 1, 8],
            "actual_minutes": [90, 90, 90],
            "hit": [0.0, 0.0, 0.0],
            "free_transfers": [1, 1, 1],
            "model_id": ["p"] * 3,
            "season": ["2024-25"] * 3,
            "gw": [1, 1, 1],
            "xpts": [5.0, 1.0, 8.0],
            "team_key": [1, 2, 3],
        }
    )
    owned, bank = mod._pre_transfer_state(part)
    keys = {p.player_key for p in owned}
    assert 1 in keys
    assert 3 in keys
    assert 2 not in keys
    assert bank == 0


def test_resolve_oracle_actuals_prefer_staged_and_ledger(tmp_path, monkeypatch) -> None:
    from fplt.data.store import DataStore
    from fplt.optimise import attainable as mod

    store = DataStore(tmp_path)
    staged = pl.DataFrame(
        {
            "gw": [1],
            "player_key": [1],
            "actual_pts": [4],
            "actual_minutes": [90],
        }
    )
    monkeypatch.setattr(mod, "actuals_from_staged", lambda *_a, **_k: staged)
    got = mod._resolve_oracle_actuals(
        tmp_path, store, "2024-25", actuals_model=None, prefer_staged=True
    )
    assert got["actual_pts"].to_list() == [4]

    ledger = pl.DataFrame(
        {
            "model_id": ["m1", "m1"],
            "target_season": ["2024-25", "2024-25"],
            "target_gw": [1, 1],
            "player_key": [1, 2],
            "actual_pts": [6, 7],
            "actual_minutes": [90, 90],
            "as_of_gw": [1, 1],
            "horizon": [1, 1],
            "fixture_id": [10, 11],
        }
    )
    monkeypatch.setattr(mod, "read_predictions", lambda _p: ledger)
    monkeypatch.setattr(mod, "actuals_from_staged", lambda *_a, **_k: staged.clear())
    from_ledger = mod._resolve_oracle_actuals(
        tmp_path, store, "2024-25", actuals_model=None, prefer_staged=False
    )
    assert set(from_ledger["actual_pts"].to_list()) == {6, 7}


def test_resolve_actuals_model_id_and_actual_rows(monkeypatch) -> None:
    from fplt.models.spec import ModelSpec
    from fplt.optimise import attainable as mod

    ledger = pl.DataFrame(
        {
            "model_id": ["rolling_mean_v1"],
            "target_season": ["2024-25"],
            "target_gw": [1],
            "player_key": [1],
            "actual_pts": [3],
            "as_of_gw": [1],
            "horizon": [1],
        }
    )
    with pytest.raises(ValueError, match="empty"):
        mod._resolve_actuals_model_id(ledger.clear(), "2024-25", None)
    with pytest.raises(ValueError, match="no ledger rows"):
        mod._resolve_actuals_model_id(ledger, "2099-00", None)
    assert mod._resolve_actuals_model_id(ledger, "2024-25", None) == "rolling_mean_v1"
    monkeypatch.setattr(
        mod,
        "load_model_specs",
        lambda: {"rm": ModelSpec(name="rm", kind="rolling_mean", seed=0)},
    )
    # ModelSpec.model_id property — check what it returns
    specs = {"rm": ModelSpec(name="rm", kind="rolling_mean", seed=0)}
    monkeypatch.setattr(mod, "load_model_specs", lambda: specs)
    assert (
        mod._resolve_actuals_model_id(ledger, "2024-25", "rm") == specs["rm"].model_id
    )

    empty = mod._actual_rows_for_oracle(ledger.clear(), "m", "2024-25")
    assert empty.is_empty()
    assert mod._actual_rows_for_oracle(ledger, "missing", "2024-25").is_empty()
    null_pts = ledger.with_columns(pl.lit(None).cast(pl.Int64).alias("actual_pts"))
    assert mod._actual_rows_for_oracle(
        null_pts, "rolling_mean_v1", "2024-25"
    ).is_empty()
    # Without actual_minutes → null minutes column
    no_mins = (
        ledger.drop("actual_minutes") if "actual_minutes" in ledger.columns else ledger
    )
    rows = mod._actual_rows_for_oracle(no_mins, "rolling_mean_v1", "2024-25")
    assert rows.height == 1
    assert rows["actual_minutes"][0] is None
    # With fixture_id unique keep last
    multi = pl.DataFrame(
        {
            "model_id": ["m"] * 2,
            "target_season": ["2024-25"] * 2,
            "target_gw": [1, 1],
            "player_key": [1, 1],
            "actual_pts": [2, 5],
            "actual_minutes": [45, 90],
            "as_of_gw": [1, 2],
            "horizon": [1, 1],
            "fixture_id": [1, 1],
        }
    )
    dedup = mod._actual_rows_for_oracle(multi, "m", "2024-25")
    assert int(dedup["actual_pts"][0]) == 5


def test_run_hindsight_oracle_staged_path(tmp_path, monkeypatch) -> None:
    from fplt.optimise import attainable as mod

    pool = _pool().with_columns(pl.lit(1).alias("gw"))
    actuals = pl.DataFrame(
        {
            "gw": [1] * pool.height,
            "player_key": pool["player_key"].to_list(),
            "actual_pts": [2] * pool.height,
            "actual_minutes": [90] * pool.height,
        }
    )
    # DecisionSchema has no team_key / price — drop market columns from pool.
    drop_cols = [c for c in ("team_key", "price", "gw") if c in pool.columns]
    base = pool.drop(drop_cols) if drop_cols else pool
    keys = base["player_key"].to_list()[:15]
    decisions = base.with_columns(
        [
            pl.lit(ORACLE_MODEL_ID).alias("model_id"),
            pl.lit("2024-25").alias("season"),
            pl.lit(1).alias("gw"),
            pl.col("player_key").is_in(keys[:11]).alias("in_xi"),
            (pl.col("player_key") == keys[0]).alias("captain"),
            (pl.col("player_key") == keys[1]).alias("vice"),
            pl.lit(False).alias("bought"),
            pl.lit(False).alias("sold"),
            pl.lit(1.0).alias("xpts"),
            pl.lit(2).alias("actual_pts"),
            pl.lit(90).alias("actual_minutes"),
            pl.lit(0.0).alias("hit"),
            pl.lit(1).alias("free_transfers"),
            pl.lit(None).cast(pl.Utf8).alias("chip"),
        ]
    )

    monkeypatch.setattr(mod, "candidates_as_of", lambda _view: pool.drop("gw"))
    monkeypatch.setattr(
        mod,
        "_resolve_oracle_actuals",
        lambda *_a, **_k: actuals,
    )
    monkeypatch.setattr(
        mod,
        "simulate_hindsight_oracle",
        lambda *_a, **_k: decisions,
    )

    class _Pit:
        def view(self, _key: object) -> None:
            return None

    monkeypatch.setattr(mod, "PITData", lambda _store: _Pit())
    path = mod.run_hindsight_oracle(
        tmp_path, "2024-25", 1, 1, horizon=1, prefer_staged_actuals=True
    )
    assert path.exists()

    monkeypatch.setattr(
        mod,
        "_resolve_oracle_actuals",
        lambda *_a, **_k: actuals.clear(),
    )
    with pytest.raises(ValueError, match="no oracle actuals"):
        mod.run_hindsight_oracle(tmp_path, "2024-25", 1, 1, horizon=1)

    monkeypatch.setattr(
        mod,
        "_resolve_oracle_actuals",
        lambda *_a, **_k: actuals,
    )

    class _Store:
        def __init__(self, *_a: object, **_k: object) -> None:
            pass

        def read(self, _name: str) -> pl.DataFrame:
            return pl.DataFrame(
                schema={
                    "season": pl.Utf8(),
                    "gw": pl.Int64(),
                    "team_h_key": pl.Int64(),
                    "team_a_key": pl.Int64(),
                }
            )

    monkeypatch.setattr(mod, "DataStore", _Store)
    with pytest.raises(ValueError, match="requires staged fixtures"):
        mod.run_hindsight_oracle(
            tmp_path, "2024-25", 1, 1, horizon=1, chip_arm="optimised"
        )


def test_optimal_lineup_select_squad_fallback() -> None:
    from unittest.mock import patch

    from fplt.optimise import attainable as mod

    picks = _decision_gw(xi_pts=[2] * 11, captain_index=0).drop("team_key")
    with patch("fplt.optimise.attainable.select_squad", side_effect=ValueError("bad")):
        score = mod._optimal_lineup_score(picks)
    assert score == mod._best_captain_score(picks)
    no_xi = picks.with_columns(pl.lit(False).alias("in_xi"))
    with pytest.raises(ValueError, match="at least one in_xi"):
        mod._best_captain_score(no_xi)


def test_captaincy_regret_empty_xi_after_sold() -> None:
    from fplt.optimise import attainable as mod

    picks = _decision_gw(xi_pts=[2] * 11, captain_index=0).with_columns(
        pl.lit(True).alias("sold")
    )
    assert mod._captaincy_regret(picks) == 0


def test_transfer_regret_draft_and_plan_failure() -> None:
    from unittest.mock import patch

    from fplt.optimise import attainable as mod

    # GW1 draft: no pre-owned → 0
    draft = _decision_gw(xi_pts=[2] * 11, captain_index=0).with_columns(
        pl.lit(True).alias("bought")
    )
    assert mod._transfer_regret(draft, hit=0.0, free_transfers=1, market=draft) == 0
    # plan_squad failure → 0
    picks = _decision_gw(xi_pts=[2] * 11, captain_index=0)
    with patch("fplt.optimise.attainable.plan_squad", side_effect=ValueError("x")):
        assert mod._transfer_regret(picks, hit=0.0, free_transfers=1, market=picks) == 0
    # Market player without actuals gets zero xpts padding
    market = picks.select(
        "gw",
        "player_key",
        "position",
        "team_key",
        "price",
        "actual_pts",
        "actual_minutes",
    )
    extra = pl.DataFrame(
        {
            "gw": [1],
            "player_key": [777],
            "position": ["MID"],
            "team_key": [9],
            "price": [40],
            "actual_pts": [None],
            "actual_minutes": [None],
        }
    )
    market = pl.concat([market, extra], how="diagonal_relaxed")
    # Keep owned so not draft; regret still non-negative
    assert mod._transfer_regret(picks, hit=0.0, free_transfers=1, market=market) >= 0


def test_resolve_oracle_actuals_ledger_miss_falls_to_staged(
    tmp_path, monkeypatch
) -> None:
    from fplt.data.store import DataStore
    from fplt.optimise import attainable as mod

    store = DataStore(tmp_path)
    staged = pl.DataFrame(
        {
            "gw": [1],
            "player_key": [1],
            "actual_pts": [9],
            "actual_minutes": [90],
        }
    )
    # prefer_staged but empty → fall through; ledger empty → staged
    monkeypatch.setattr(mod, "actuals_from_staged", lambda *_a, **_k: staged)
    monkeypatch.setattr(mod, "read_predictions", lambda _p: pl.DataFrame())
    got = mod._resolve_oracle_actuals(
        tmp_path, store, "2024-25", actuals_model=None, prefer_staged=True
    )
    assert got["actual_pts"].to_list() == [9]

    # Ledger present but resolve model raises → staged
    monkeypatch.setattr(
        mod,
        "read_predictions",
        lambda _p: pl.DataFrame({"model_id": ["m"], "target_season": ["other"]}),
    )
    got2 = mod._resolve_oracle_actuals(
        tmp_path, store, "2024-25", actuals_model=None, prefer_staged=False
    )
    assert got2["actual_pts"].to_list() == [9]

    # Horizon==1 filter path (no fixture_id)
    ledger = pl.DataFrame(
        {
            "model_id": ["m", "m"],
            "target_season": ["2024-25", "2024-25"],
            "target_gw": [1, 1],
            "player_key": [1, 1],
            "actual_pts": [2, 4],
            "as_of_gw": [1, 1],
            "horizon": [1, 2],
        }
    )
    rows = mod._actual_rows_for_oracle(ledger, "m", "2024-25")
    assert rows.height == 1
    assert int(rows["actual_pts"][0]) == 2
