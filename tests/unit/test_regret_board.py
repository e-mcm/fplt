"""Regret-by-component board report helpers."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from fplt.optimise.attainable import (
    ORACLE_MODEL_ID,
    component_regret,
    summarise_attainable,
)
from fplt.optimise.decisions import write_decisions
from fplt.optimise.regret_board import (
    RegretBoardResult,
    format_regret_report,
    run_regret_board,
    write_regret_board,
)


def test_write_regret_board_emits_artifacts(tmp_path: Path) -> None:
    policy = _gw(model_id="policy", chip="bboost")
    oracle = _gw(model_id=ORACLE_MODEL_ID, chip=None).with_columns(
        pl.when(pl.col("captain"))
        .then(pl.lit(15))
        .otherwise(pl.col("actual_pts"))
        .alias("actual_pts")
    )
    summary = summarise_attainable(policy, oracle)
    per_gw = component_regret(policy)
    result = RegretBoardResult(
        summary=summary,
        per_gw=per_gw,
        policy=policy,
        oracle=oracle,
        report_md=format_regret_report(
            summary,
            per_gw,
            meta={
                "season": "2024-25",
                "from_gw": 1,
                "to_gw": 1,
                "horizon": 3,
                "model": "dc_layered_v2",
                "policy_model_id": "policy",
                "oracle_model_id": ORACLE_MODEL_ID,
                "chip_arm": "optimised",
                "oracle_chip_arm": "optimised",
                "force_play": True,
                "oracle_force_play": True,
                "lineup_reopt": True,
                "price_forecast_method": "mean_gw_delta",
                "policy_points": int(summary["total_points"][0]),
                "oracle_points": int(summary["attainable_points"][0]),
            },
        ),
        meta={"season": "2024-25"},
    )
    out = tmp_path / "board"
    path = write_regret_board(result, out)
    assert path == out / "report.md"
    assert (out / "summary.parquet").exists()
    assert (out / "per_gw.parquet").exists()
    assert (out / "policy_decisions.parquet").exists()
    assert (out / "oracle_decisions.parquet").exists()
    assert (out / "meta.json").exists()
    assert "Component totals" in path.read_text(encoding="utf-8")


def test_run_regret_board_summarise_only_from_ledger(tmp_path: Path) -> None:
    policy = _gw(model_id="abc::chip_optimised", chip="3xc")
    oracle = _gw(
        model_id=f"{ORACLE_MODEL_ID}::chip_optimised", chip="bboost"
    ).with_columns(
        pl.when(pl.col("captain"))
        .then(pl.lit(18))
        .otherwise(pl.col("actual_pts"))
        .alias("actual_pts")
    )
    write_decisions(tmp_path, pl.concat([policy, oracle]))
    result = run_regret_board(
        tmp_path,
        season="2024-25",
        from_gw=1,
        to_gw=1,
        horizon=1,
        model="dc_layered_v2",
        chip_arm="optimised",
        simulate_policy=False,
        simulate_oracle=False,
        policy_parquet=None,
    )
    assert result.summary.height == 1
    assert result.per_gw.height == 1
    assert result.meta["policy_points"] <= result.meta["oracle_points"]
    assert "Captaincy" in result.report_md


def test_run_regret_board_loads_policy_parquet(tmp_path: Path) -> None:
    policy = _gw(model_id="tiphash::chip_optimised", chip="bboost")
    oracle = _gw(model_id=f"{ORACLE_MODEL_ID}::chip_optimised", chip=None).with_columns(
        pl.lit(12).alias("actual_pts")
    )
    parquet = tmp_path / "policy.parquet"
    policy.write_parquet(parquet)
    write_decisions(tmp_path, oracle)
    result = run_regret_board(
        tmp_path,
        season="2024-25",
        from_gw=1,
        to_gw=1,
        horizon=1,
        model="dc_layered_v2",
        chip_arm="optimised",
        simulate_policy=False,
        simulate_oracle=False,
        policy_parquet=parquet,
    )
    assert result.policy["model_id"][0] == "tiphash::chip_optimised"
    assert result.summary["share_of_attainable"][0] >= 0.0


def _gw(*, model_id: str, chip: str | None) -> pl.DataFrame:
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
    for index, position in enumerate(positions):
        rows.append(
            {
                "model_id": model_id,
                "season": "2024-25",
                "gw": 1,
                "player_key": index + 1,
                "position": position,
                "in_xi": True,
                "captain": index == 0,
                "vice": index == 1,
                "bought": False,
                "sold": False,
                "xpts": 2.0,
                "actual_pts": 2 + (index % 3),
                "actual_minutes": 90,
                "hit": 0.0,
                "free_transfers": 1,
                "chip": chip,
            }
        )
    for index, position in enumerate(["GK", "DEF", "DEF", "MID"]):
        rows.append(
            {
                "model_id": model_id,
                "season": "2024-25",
                "gw": 1,
                "player_key": 12 + index,
                "position": position,
                "in_xi": False,
                "captain": False,
                "vice": False,
                "bought": False,
                "sold": False,
                "xpts": 1.0,
                "actual_pts": 4 if position == "MID" else 0,
                "actual_minutes": 90 if position == "MID" else 0,
                "hit": 0.0,
                "free_transfers": 1,
                "chip": chip,
            }
        )
    return pl.DataFrame(rows)
