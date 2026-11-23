"""Regret-by-component decision board (evaluation north star §2).

Runs (or reuses) a live-stack policy season, a matching constrained hindsight
oracle, and reports captaincy / XI-bench / transfer / chip-timing regret plus
share of attainable points. Does not flip ``PRODUCTION_MODELS``.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import polars as pl

from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.spec import load_model_specs
from fplt.optimise.attainable import (
    ORACLE_MODEL_ID,
    actuals_from_staged,
    component_regret,
    format_attainable_scoreboard,
    is_oracle_model_id,
    oracle_ledger_model_id,
    run_hindsight_oracle,
    summarise_attainable,
)
from fplt.optimise.chip_windows import (
    ChipArm,
    chip_arm_model_id,
    parse_chip_arm_model_id,
)
from fplt.optimise.decisions import (
    DEFAULT_LIVE_CHIP_ARM,
    DEFAULT_LIVE_PRICE_FORECAST_METHOD,
    candidates_as_of,
    read_decisions,
    run_decisions,
    write_decisions,
)
from fplt.optimise.plan import DEFAULT_PLAN_HORIZON
from fplt.optimise.price_forecast import PriceForecastMethod
from fplt.optimise.realised import summarise_decisions
from fplt.types import GWKey

logger = logging.getLogger(__name__)

__all__ = [
    "RegretBoardResult",
    "format_regret_report",
    "run_regret_board",
    "write_regret_board",
]


@dataclass(frozen=True)
class RegretBoardResult:
    """Season summary, per-GW regret, ledgers, and rendered markdown."""

    summary: pl.DataFrame
    per_gw: pl.DataFrame
    policy: pl.DataFrame
    oracle: pl.DataFrame
    report_md: str
    meta: dict[str, Any]


def run_regret_board(
    data_dir: Path,
    *,
    season: str,
    from_gw: int = 1,
    to_gw: int = 38,
    horizon: int = DEFAULT_PLAN_HORIZON,
    model: str = "dc_layered_v2",
    chip_arm: ChipArm = DEFAULT_LIVE_CHIP_ARM,
    chip_rules_season: str | None = None,
    force_play: bool = True,
    lineup_reopt: bool = True,
    price_forecast_method: PriceForecastMethod | None = (
        DEFAULT_LIVE_PRICE_FORECAST_METHOD
    ),
    oracle_chip_arm: ChipArm | None = None,
    oracle_force_play: bool | None = None,
    simulate_policy: bool = True,
    simulate_oracle: bool = True,
    policy_parquet: Path | None = None,
    prefer_staged_actuals: bool = True,
) -> RegretBoardResult:
    """Build the regret-by-component board for one season.

    Args:
        data_dir: Root with ``staged/`` and ``ledger/``.
        season: Fixture / decision season.
        from_gw: First gameweek.
        to_gw: Last gameweek, inclusive.
        horizon: Planner look-ahead (policy and oracle).
        model: YAML spec name for the policy arm.
        chip_arm: Policy chip policy (live default ``optimised``).
        chip_rules_season: Chip-rules year (default ``season``).
        force_play: Policy optimised-arm force-pack (live default true).
        lineup_reopt: Policy post-MILP lineup re-opt (live default true).
        price_forecast_method: Policy within-horizon price forecast (live
            default ``mean_gw_delta``).
        oracle_chip_arm: Oracle chip arm (default same as policy).
        oracle_force_play: Oracle force-play (default same as policy).
        simulate_policy: When true, run ``run_decisions`` for the policy.
        simulate_oracle: When true, run ``run_hindsight_oracle``.
        policy_parquet: Optional decision parquet to use as the policy ledger
            (skips ``simulate_policy`` when set).
        prefer_staged_actuals: Oracle actuals from staged stats (default true).

    Returns:
        ``RegretBoardResult`` with summary, per-GW regret, and markdown.

    Raises:
        KeyError: Unknown ``model`` YAML name.
        ValueError: Missing policy/oracle rows or empty markets.
    """
    rules_season = chip_rules_season if chip_rules_season is not None else season
    o_arm: ChipArm = oracle_chip_arm if oracle_chip_arm is not None else chip_arm
    o_force = force_play if oracle_force_play is None else oracle_force_play
    spec = load_model_specs()[model]
    if chip_arm == "off":
        policy_ledger_id = spec.model_id
    else:
        policy_ledger_id = chip_arm_model_id(spec.model_id, chip_arm)
    oracle_id = oracle_ledger_model_id(o_arm)

    if policy_parquet is not None:
        policy = pl.read_parquet(policy_parquet)
        write_decisions(data_dir, policy)
        simulate_policy = False
    elif simulate_policy:
        logger.info(
            "regret board policy model=%s arm=%s season=%s H=%s force_play=%s "
            "lineup_reopt=%s price_forecast=%s",
            model,
            chip_arm,
            season,
            horizon,
            force_play,
            lineup_reopt,
            price_forecast_method,
        )
        run_decisions(
            data_dir,
            model,
            season,
            from_gw,
            to_gw,
            horizon,
            chip_arm=chip_arm,
            chip_rules_season=rules_season,
            force_play=force_play,
            lineup_reopt=lineup_reopt,
            price_forecast_method=price_forecast_method,
        )

    if simulate_oracle:
        logger.info(
            "regret board oracle arm=%s season=%s H=%s force_play=%s",
            o_arm,
            season,
            horizon,
            o_force,
        )
        run_hindsight_oracle(
            data_dir,
            season,
            from_gw,
            to_gw,
            horizon,
            actuals_model=model,
            chip_arm=o_arm,
            chip_rules_season=rules_season,
            force_play=o_force,
            prefer_staged_actuals=prefer_staged_actuals,
            lineup_reopt=lineup_reopt,
        )

    ledger = read_decisions(data_dir)
    if ledger.is_empty():
        raise ValueError("decision ledger is empty")
    window = (pl.col("gw") >= from_gw) & (pl.col("gw") <= to_gw)
    season_rows = ledger.filter((pl.col("season") == season) & window)
    policy = _select_policy(season_rows, policy_ledger_id, policy_parquet is not None)
    oracle = season_rows.filter(pl.col("model_id") == oracle_id)
    if oracle.is_empty():
        # Chips-off oracle id fallback when arm is off.
        oracle = season_rows.filter(pl.col("model_id") == ORACLE_MODEL_ID)
    if policy.is_empty():
        raise ValueError(
            f"no policy decision rows for model_id≈{policy_ledger_id!r} season={season}"
        )
    if oracle.is_empty():
        raise ValueError(
            f"no oracle decision rows for model_id={oracle_id!r} season={season}"
        )

    store = DataStore(data_dir)
    market = _market_frame(store, season, from_gw, to_gw)
    per_gw = component_regret(policy, market=market)
    summary = summarise_attainable(policy, oracle, market=market)
    meta: dict[str, Any] = {
        "season": season,
        "from_gw": from_gw,
        "to_gw": to_gw,
        "horizon": horizon,
        "model": model,
        "policy_model_id": str(policy["model_id"][0]),
        "oracle_model_id": str(oracle["model_id"][0]),
        "chip_arm": chip_arm,
        "oracle_chip_arm": o_arm,
        "force_play": force_play,
        "oracle_force_play": o_force,
        "lineup_reopt": lineup_reopt,
        "price_forecast_method": price_forecast_method,
        "chip_rules_season": rules_season,
        "policy_points": int(summarise_decisions(policy)["total_points"][0]),
        "oracle_points": int(summarise_decisions(oracle)["total_points"][0]),
    }
    report = format_regret_report(summary, per_gw, meta=meta)
    return RegretBoardResult(
        summary=summary,
        per_gw=per_gw,
        policy=policy,
        oracle=oracle,
        report_md=report,
        meta=meta,
    )


def format_regret_report(
    summary: pl.DataFrame,
    per_gw: pl.DataFrame,
    *,
    meta: dict[str, Any] | None = None,
    top_n: int = 10,
) -> str:
    """Render a markdown regret board with season totals and top GW rows."""
    lines: list[str] = ["# Regret-by-component board", ""]
    if meta:
        lines.extend(
            [
                "| Knob | Value |",
                "|---|---|",
                (
                    f"| Season | {meta.get('season')} "
                    f"GW{meta.get('from_gw')}–{meta.get('to_gw')} |"
                ),
                f"| Horizon | H={meta.get('horizon')} |",
                (f"| Policy | `{meta.get('policy_model_id')}` ({meta.get('model')}) |"),
                (
                    f"| Chip arm | {meta.get('chip_arm')} "
                    f"(oracle {meta.get('oracle_chip_arm')}) |"
                ),
                (
                    f"| Force-play | {meta.get('force_play')} "
                    f"(oracle {meta.get('oracle_force_play')}) |"
                ),
                f"| Lineup reopt | {meta.get('lineup_reopt')} |",
                f"| Price forecast | {meta.get('price_forecast_method')} |",
                f"| Policy points | {meta.get('policy_points')} |",
                f"| Attainable (oracle) | {meta.get('oracle_points')} |",
                "",
            ]
        )
    lines.extend(
        [
            "## Season summary",
            "",
            format_attainable_scoreboard(summary),
            "",
        ]
    )
    if summary.height:
        row = summary.row(0, named=True)
        total_regret = (
            int(row.get("captaincy_regret") or 0)
            + int(row.get("xi_bench_regret") or 0)
            + int(row.get("transfer_regret") or 0)
            + int(row.get("chip_timing_regret") or 0)
        )
        lines.extend(
            [
                "## Component totals",
                "",
                "| Component | Regret |",
                "|---|---:|",
                f"| Captaincy | {int(row.get('captaincy_regret') or 0)} |",
                f"| XI / bench | {int(row.get('xi_bench_regret') or 0)} |",
                f"| Transfers | {int(row.get('transfer_regret') or 0)} |",
                f"| Chip timing | {_fmt_chip(row.get('chip_timing_regret'))} |",
                f"| Sum (components) | {total_regret} |",
                "",
                "_Component regrets are local (per-GW) and can overlap; they do "
                "not partition the oracle gap._",
                "",
            ]
        )
    if not per_gw.is_empty():
        ranked = (
            per_gw.with_columns(
                (
                    pl.col("captaincy_regret")
                    + pl.col("xi_bench_regret")
                    + pl.col("transfer_regret")
                    + pl.col("chip_timing_regret").fill_null(0)
                ).alias("total_regret")
            )
            .sort("total_regret", descending=True)
            .head(top_n)
        )
        lines.extend(
            [
                f"## Top {ranked.height} gameweeks by component regret",
                "",
                "| GW | Realised | Cap | XI | Xfer | Chip | Total |",
                "|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for gw_row in ranked.iter_rows(named=True):
            lines.append(
                "| {gw} | {realised} | {cap} | {xi} | {xfer} | {chip} | {tot} |".format(
                    gw=int(gw_row["gw"]),
                    realised=int(gw_row["realised_points"]),
                    cap=int(gw_row["captaincy_regret"]),
                    xi=int(gw_row["xi_bench_regret"]),
                    xfer=int(gw_row["transfer_regret"]),
                    chip=_fmt_chip(gw_row.get("chip_timing_regret")),
                    tot=int(gw_row["total_regret"]),
                )
            )
        lines.append("")
    return "\n".join(lines)


def write_regret_board(result: RegretBoardResult, output_dir: Path) -> Path:
    """Write summary parquet/JSON, per-GW parquet, and ``report.md``."""
    output_dir.mkdir(parents=True, exist_ok=True)
    result.summary.write_parquet(output_dir / "summary.parquet")
    result.per_gw.write_parquet(output_dir / "per_gw.parquet")
    result.policy.write_parquet(output_dir / "policy_decisions.parquet")
    result.oracle.write_parquet(output_dir / "oracle_decisions.parquet")
    (output_dir / "meta.json").write_text(
        json.dumps(result.meta, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    report_path = output_dir / "report.md"
    report_path.write_text(result.report_md + "\n", encoding="utf-8")
    return report_path


def _select_policy(
    season_rows: pl.DataFrame, policy_ledger_id: str, from_parquet: bool
) -> pl.DataFrame:
    """Pick policy rows, tolerating tip-hash remaps on parquet / summarise-only."""
    del from_parquet  # retained for call-site clarity; fuzzy match always applies
    exact = season_rows.filter(pl.col("model_id") == policy_ledger_id)
    if not exact.is_empty():
        return exact
    non_oracle = season_rows.filter(
        ~pl.col("model_id").map_elements(is_oracle_model_id, return_dtype=pl.Boolean)
    )
    if non_oracle.is_empty():
        return exact
    if non_oracle["model_id"].n_unique() == 1:
        return non_oracle
    # Prefer chip-arm suffix matching the requested ledger id.
    _, want_arm = parse_chip_arm_model_id(policy_ledger_id)
    if want_arm is not None:
        matched = [
            mid
            for mid in non_oracle["model_id"].unique().to_list()
            if parse_chip_arm_model_id(str(mid))[1] == want_arm
        ]
        if len(matched) == 1:
            return non_oracle.filter(pl.col("model_id") == matched[0])
    return exact


def _market_frame(
    store: DataStore, season: str, from_gw: int, to_gw: int
) -> pl.DataFrame:
    """Build gw × player market + staged actuals for transfer regret.

    Returns an empty frame when staged gameweeks / candidates are missing so
    summarise-only boards still run (transfer regret stays 0).
    """
    pit = PITData(store)
    pools: list[pl.DataFrame] = []
    for gw in range(from_gw, to_gw + 1):
        try:
            pools.append(
                candidates_as_of(pit.view(GWKey(season, gw))).with_columns(
                    pl.lit(gw).cast(pl.Int64).alias("gw")
                )
            )
        except KeyError:
            logger.info("regret board: no staged candidates for %s gw%s", season, gw)
            continue
    candidates = pl.concat(pools) if pools else pl.DataFrame()
    if candidates.is_empty():
        return candidates
    actuals = actuals_from_staged(store, season)
    if actuals.is_empty():
        return candidates.with_columns(
            pl.lit(None).cast(pl.Int64).alias("actual_pts"),
            pl.lit(None).cast(pl.Int64).alias("actual_minutes"),
        )
    return candidates.join(actuals, on=["gw", "player_key"], how="left")


def _fmt_chip(value: int | float | None) -> str:
    if value is None:
        return "-"
    return str(int(value))
