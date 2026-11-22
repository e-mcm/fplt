"""Ensemble sensitivity: shared-state week-1 move-frequency board."""

from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

from fplt.cli import main
from fplt.optimise.decisions import advance_squad_from_plan
from fplt.optimise.ensemble_sensitivity import (
    agreement_flags,
    format_ensemble_sensitivity_board,
    run_ensemble_sensitivity,
    summarise_ensemble_sensitivity,
    transfer_in_symmetric_diff,
    week1_actions_from_decision_rows,
    write_ensemble_sensitivity_report,
)
from fplt.optimise.plan import plan_squad
from fplt.optimise.pruning_loss import Week1Actions
from fplt.optimise.squad import select_squad
from fplt.optimise.state import OwnedPlayer, SquadState
from fplt.schemas.ensemble_sensitivity import (
    EnsembleSensitivityRowSchema,
    EnsembleSensitivitySummarySchema,
)


def _state(**overrides: object) -> SquadState:
    values: dict[str, object] = {
        "bank": 1000,
        "free_transfers": 1,
        "ft_bank_cap": 2,
        "hit_cost": 4,
        "decay": 1.0,
        "bench_weight": 0.0,
        "owned": [],
    }
    values.update(overrides)
    return SquadState.model_validate(values)


def _pool() -> pl.DataFrame:
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
                    "xpts": float(club + (1 if position == "DEF" else 0)),
                }
            )
            player += 1
    return pl.DataFrame(rows)


def _predictions(
    model_id: str,
    candidates: pl.DataFrame,
    *,
    as_of: int,
    target: int,
    xpts_by_key: dict[int, float] | None = None,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    for row in candidates.iter_rows(named=True):
        key = int(row["player_key"])
        xpts = float(row["xpts"]) if xpts_by_key is None else float(xpts_by_key[key])
        rows.append(
            {
                "as_of_gw": as_of,
                "target_gw": target,
                "player_key": key,
                "xpts": xpts,
            }
        )
    del model_id  # predictions are keyed externally by model_id
    return pl.DataFrame(rows)


def test_advance_squad_from_plan_applies_draft_buys() -> None:
    """Public advance helper owns the drafted fifteen after an initial solve."""
    pool = _pool()
    points = pool.select("player_key", pl.lit(0).alias("step"), "xpts")
    plan = plan_squad(pool.drop("xpts"), points, _state(), initial=True, horizon=1)
    advanced = advance_squad_from_plan(
        _state(), plan, pool.drop("xpts"), gameweek=1, rules=None
    )
    assert len(advanced.owned) == 15
    kept = set(
        int(key) for key in plan.picks.filter(~pl.col("sold"))["player_key"].to_list()
    )
    assert {player.player_key for player in advanced.owned} == kept


def test_agreement_flags_and_symmetric_diff() -> None:
    """Transfer mismatch flips transfer/week1 flags and counts the Δ set."""
    left = Week1Actions(
        transfers_in=(1, 2),
        transfers_out=(3,),
        xi=(1, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13),
        captain=1,
        vice=4,
    )
    right = Week1Actions(
        transfers_in=(1, 9),
        transfers_out=(3,),
        xi=left.xi,
        captain=1,
        vice=4,
    )
    flags = agreement_flags(left, right)
    assert flags["week1"] is False
    assert flags["transfer"] is False
    assert flags["xi"] is True
    assert flags["captain"] is True
    assert flags["vice"] is True
    assert transfer_in_symmetric_diff(left, right) == 2


def test_week1_actions_from_decision_rows_initial_ignores_buys() -> None:
    """Draft buys are not transfers when ``initial`` is True."""
    frame = pl.DataFrame(
        {
            "model_id": ["a"] * 3,
            "season": ["2024-25"] * 3,
            "gw": [1, 1, 1],
            "player_key": [1, 2, 3],
            "position": ["GK", "DEF", "MID"],
            "in_xi": [True, True, False],
            "captain": [True, False, False],
            "vice": [False, True, False],
            "bought": [True, True, True],
            "sold": [False, False, False],
            "xpts": [1.0, 1.0, 0.5],
            "actual_pts": [None, None, None],
            "actual_minutes": [None, None, None],
            "hit": [0.0, 0.0, 0.0],
            "free_transfers": [0, 0, 0],
            "chip": [None, None, None],
        }
    )
    actions = week1_actions_from_decision_rows(frame, initial=True)
    assert actions.transfers_in == ()
    assert actions.transfers_out == ()
    assert actions.xi == (1, 2)
    assert actions.captain == 1
    assert actions.vice == 2


def test_identical_forecasts_agree_on_draft() -> None:
    """Same xPts → week-1 agree; move frequency is zero."""
    pool = _pool()
    candidates = pool.select(
        "player_key", "position", "team_key", "price"
    ).with_columns(pl.lit(1).alias("gw"))
    base = _predictions("control", pool, as_of=1, target=1)
    report = run_ensemble_sensitivity(
        candidates,
        {"control": base, "challenger": base.clone()},
        _state(),
        season="2024-25",
        gameweeks=[1],
        horizon=1,
        control_model_id="control",
        challenger_model_ids=["challenger"],
    )
    checked = EnsembleSensitivityRowSchema.validate(report.rows)
    assert isinstance(checked, pl.DataFrame)
    assert checked.height == 1
    assert checked["week1_agree"][0] is True
    assert checked["transfer_agree"][0] is True
    summary = EnsembleSensitivitySummarySchema.validate(report.summary)
    assert isinstance(summary, pl.DataFrame)
    assert float(summary["move_frequency"][0]) == pytest.approx(0.0)
    assert float(summary["week1_agree_rate"][0]) == pytest.approx(1.0)


def test_disagreement_on_captain_and_transfers() -> None:
    """Perturbed forecasts change captain and/or transfers from a shared squad."""
    pool = _pool()
    selected = select_squad(pool).players
    owned = [
        OwnedPlayer(player_key=int(key), selling_price=int(price))
        for key, price in selected.select("player_key", "price").iter_rows()
    ]
    bank = 1000 - int(selected["price"].sum())
    # Add cheap mirrors so a transfer is available.
    mirrors = selected.head(2).with_columns(
        (pl.col("player_key") + 1000).alias("player_key")
    )
    candidates_base = pl.concat(
        [
            pool.select("player_key", "position", "team_key", "price", "xpts"),
            mirrors.select("player_key", "position", "team_key", "price").with_columns(
                pl.lit(0.5).alias("xpts")
            ),
        ]
    )
    candidates = candidates_base.drop("xpts").with_columns(pl.lit(2).alias("gw"))
    owned_keys = set(int(k) for k in selected["player_key"].to_list())
    mirror_keys = [int(k) for k in mirrors["player_key"].to_list()]

    control_xpts = {
        int(row["player_key"]): (
            2.0
            if int(row["player_key"]) in owned_keys
            else 0.1
            if int(row["player_key"]) in mirror_keys
            else float(row["xpts"])
        )
        for row in candidates_base.iter_rows(named=True)
    }
    # Challenger loves the first mirror enough to transfer, and boosts a
    # different owned player for captaincy.
    challenger_xpts = dict(control_xpts)
    first_owned = int(selected["player_key"][0])
    second_owned = int(selected["player_key"][1])
    challenger_xpts[mirror_keys[0]] = 8.0
    challenger_xpts[first_owned] = 0.5
    challenger_xpts[second_owned] = 7.0
    # Keep the sold player low so the hit is worthwhile.
    sold_template = int(selected.head(1)["player_key"][0])
    # The mirror replaces the same-position owned player when possible.
    for key in owned_keys:
        row = selected.filter(pl.col("player_key") == key)
        if row["position"][0] == mirrors["position"][0] and key != second_owned:
            challenger_xpts[key] = 0.2
            sold_template = key
            break
    del sold_template

    control_pred = _predictions(
        "control", candidates_base, as_of=2, target=2, xpts_by_key=control_xpts
    )
    challenger_pred = _predictions(
        "challenger",
        candidates_base,
        as_of=2,
        target=2,
        xpts_by_key=challenger_xpts,
    )
    report = run_ensemble_sensitivity(
        candidates,
        {"control": control_pred, "challenger": challenger_pred},
        _state(bank=bank, owned=owned, free_transfers=1),
        season="2024-25",
        gameweeks=[2],
        horizon=1,
        control_model_id="control",
        challenger_model_ids=["challenger"],
    )
    row = report.rows.row(0, named=True)
    assert row["week1_agree"] is False
    # At least one of transfer / captain should diverge under the perturbation.
    assert (not row["transfer_agree"]) or (not row["captain_agree"])
    summary = report.summary
    assert float(summary["move_frequency"][0]) == pytest.approx(1.0)


def test_shared_state_advances_on_control_across_two_gws() -> None:
    """GW2 uses the squad the control drafted at GW1 (shared path)."""
    pool = _pool()
    cand1 = pool.select("player_key", "position", "team_key", "price").with_columns(
        pl.lit(1).alias("gw")
    )
    cand2 = cand1.with_columns(pl.lit(2).alias("gw"))
    candidates = pl.concat([cand1, cand2])
    # Identical forecasts → agree both weeks.
    pred_c = pl.concat(
        [
            _predictions("c", pool, as_of=1, target=1),
            _predictions("c", pool, as_of=2, target=2),
        ]
    )
    pred_h = pred_c.clone()
    report = run_ensemble_sensitivity(
        candidates,
        {"c": pred_c, "h": pred_h},
        _state(),
        season="2024-25",
        gameweeks=[1, 2],
        horizon=1,
        control_model_id="c",
        challenger_model_ids=["h"],
    )
    assert report.rows.height == 2
    assert report.rows["week1_agree"].all()
    assert int(report.summary["n_gw"][0]) == 2
    assert float(report.summary["move_frequency"][0]) == pytest.approx(0.0)


def test_rejects_empty_challengers_and_control_in_list() -> None:
    """Invalid member lists raise ValueError."""
    pool = _pool()
    candidates = pool.select(
        "player_key", "position", "team_key", "price"
    ).with_columns(pl.lit(1).alias("gw"))
    pred = _predictions("c", pool, as_of=1, target=1)
    with pytest.raises(ValueError, match="challenger_model_ids"):
        run_ensemble_sensitivity(
            candidates,
            {"c": pred},
            _state(),
            season="2024-25",
            gameweeks=[1],
            horizon=1,
            control_model_id="c",
            challenger_model_ids=[],
        )
    with pytest.raises(ValueError, match="must not include the control"):
        run_ensemble_sensitivity(
            candidates,
            {"c": pred, "h": pred},
            _state(),
            season="2024-25",
            gameweeks=[1],
            horizon=1,
            control_model_id="c",
            challenger_model_ids=["c"],
        )


def test_summarise_empty_and_report_io(tmp_path: Path) -> None:
    """Empty summarise validates; write/format round-trip."""
    empty_rows = pl.DataFrame(
        schema={
            "season": pl.String(),
            "gw": pl.Int64(),
            "control_model_id": pl.String(),
            "challenger_model_id": pl.String(),
            "week1_agree": pl.Boolean(),
            "transfer_agree": pl.Boolean(),
            "xi_agree": pl.Boolean(),
            "captain_agree": pl.Boolean(),
            "vice_agree": pl.Boolean(),
            "n_transfers_control": pl.Int64(),
            "n_transfers_challenger": pl.Int64(),
            "n_transfer_in_symmetric_diff": pl.Int64(),
            "hit_control": pl.Float64(),
            "hit_challenger": pl.Float64(),
            "horizon": pl.Int64(),
        }
    )
    summary = summarise_ensemble_sensitivity(empty_rows)
    assert summary.is_empty()
    assert format_ensemble_sensitivity_board(summary).startswith("_No ensemble")

    pool = _pool()
    candidates = pool.select(
        "player_key", "position", "team_key", "price"
    ).with_columns(pl.lit(1).alias("gw"))
    pred = _predictions("c", pool, as_of=1, target=1)
    report = run_ensemble_sensitivity(
        candidates,
        {"c": pred, "h": pred.clone()},
        _state(),
        season="2024-25",
        gameweeks=[1],
        horizon=1,
        control_model_id="c",
        challenger_model_ids=["h"],
    )
    text = format_ensemble_sensitivity_board(report.summary)
    assert "Move freq" in text
    assert "c" in text
    paths = write_ensemble_sensitivity_report(report, tmp_path)
    assert paths["rows"].is_file()
    assert paths["summary"].is_file()
    assert "move_frequency" in paths["json"].read_text()
    assert "forecast-driven thrashing" in paths["json"].read_text()


def test_cli_ensemble_sensitivity_help() -> None:
    """CLI advertises the ensemble-sensitivity subcommand."""
    with pytest.raises(SystemExit) as exc:
        main(["ensemble-sensitivity", "--help"])
    assert exc.value.code == 0
