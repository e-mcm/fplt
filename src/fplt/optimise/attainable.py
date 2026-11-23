"""Constrained hindsight oracle, share of attainable points, and component regret.

The oracle is **not** unbounded perfect foresight. It reuses the decision
simulator (``simulate_decisions`` / ``plan_squad``) with realised points as
the planner's ``xpts``, under the same free-transfer banking, hit cost,
budget, squad composition, and FPL 50% sell-on selling prices (as-of staged
``player_status`` prices; within-horizon forecasts off by default) as the
decision bakeoff. Default chip policy is ``off`` (N5 board / ``--with-oracle``);
pass ``chip_arm="optimised"`` (and fixtures) for a chips-on attainable ceiling.
``chip_timing_regret`` remains a placeholder while chips are off.

Share of attainable points = realised season total ÷ oracle season total.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import polars as pl

from fplt.backtest.ledger import read_predictions
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.spec import load_model_specs
from fplt.optimise.chip_windows import ChipArm, chip_arm_model_id
from fplt.optimise.decisions import (
    candidates_as_of,
    simulate_decisions,
    write_decisions,
)
from fplt.optimise.plan import plan_squad
from fplt.optimise.realised import score_gameweek, summarise_decisions
from fplt.optimise.squad import select_squad
from fplt.optimise.state import (
    ChipInventory,
    OwnedPlayer,
    SquadState,
    load_squad_state,
)
from fplt.schemas.attainable import AttainableShareSchema, ComponentRegretSchema
from fplt.scoring.chips import ChipRules, empty_remaining
from fplt.types import GWKey

ORACLE_MODEL_ID = "hindsight_oracle_v1"

__all__ = [
    "ORACLE_MODEL_ID",
    "ChipTimingRegret",
    "actuals_from_staged",
    "attainable_share",
    "chip_timing_regret",
    "component_regret",
    "enrich_decision_summary",
    "foresight_predictions",
    "format_attainable_scoreboard",
    "oracle_ledger_model_id",
    "run_hindsight_oracle",
    "simulate_hindsight_oracle",
    "summarise_attainable",
]


@dataclass(frozen=True)
class ChipTimingRegret:
    """Placeholder for Wave C chip-timing regret.

    While chips are off in the decision sim, ``regret`` is ``None`` and
    ``status`` is ``\"chips_off\"``. Callers still receive a stable hook.
    """

    chip: str | None
    regret: int | None
    status: str


def foresight_predictions(
    actuals: pl.DataFrame,
    gameweeks: list[int],
) -> pl.DataFrame:
    """Build planner inputs where every deadline sees realised points ahead.

    For each ``as_of`` in ``gameweeks`` and each ``target`` >= ``as_of`` that
    appears in ``actuals``, emit ``xpts = actual_pts``. Missing future
    targets score zero in ``plan_squad`` (same as the decision bakeoff).

    Args:
        actuals: ``gw``, ``player_key``, ``actual_pts`` (minutes optional).
        gameweeks: Deadlines that will be played, in order.

    Returns:
        ``as_of_gw``, ``target_gw``, ``player_key``, ``xpts``.
    """
    empty = pl.DataFrame(
        schema={
            "as_of_gw": pl.Int64(),
            "target_gw": pl.Int64(),
            "player_key": pl.Int64(),
            "xpts": pl.Float64(),
        }
    )
    if actuals.is_empty() or not gameweeks:
        return empty
    required = {"gw", "player_key", "actual_pts"}
    missing = required - set(actuals.columns)
    if missing:
        raise ValueError(f"actuals missing columns: {sorted(missing)}")
    scored = (
        actuals.group_by("gw", "player_key")
        .agg(pl.col("actual_pts").sum())
        .with_columns(pl.col("actual_pts").cast(pl.Float64).alias("xpts"))
        .select("gw", "player_key", "xpts")
    )
    as_ofs = pl.DataFrame({"as_of_gw": gameweeks})
    return (
        as_ofs.join(scored, how="cross")
        .filter(pl.col("gw") >= pl.col("as_of_gw"))
        .rename({"gw": "target_gw"})
        .select("as_of_gw", "target_gw", "player_key", "xpts")
    )


def oracle_ledger_model_id(
    chip_arm: ChipArm = "off",
    *,
    base_model_id: str = ORACLE_MODEL_ID,
) -> str:
    """Return the decision-ledger ``model_id`` for a hindsight oracle arm.

    Chips-off keeps the historical ``hindsight_oracle_v1`` id so existing
    boards stay stable. Other arms use ``chip_arm_model_id``.
    """
    if chip_arm == "off":
        return base_model_id
    return chip_arm_model_id(base_model_id, chip_arm)


def actuals_from_staged(store: DataStore, season: str) -> pl.DataFrame:
    """Build oracle actuals from staged ``player_fixture_stats``.

    Sums ``total_points`` / ``minutes`` per player–gameweek (double GWs
    contribute both fixtures). Prefer this when the prediction ledger is
    empty or when the oracle should not depend on a model backtest.

    Args:
        store: Staged tables including ``player_fixture_stats``.
        season: Season label.

    Returns:
        ``gw``, ``player_key``, ``actual_pts``, ``actual_minutes``.
    """
    empty = pl.DataFrame(
        schema={
            "gw": pl.Int64(),
            "player_key": pl.Int64(),
            "actual_pts": pl.Int64(),
            "actual_minutes": pl.Int64(),
        }
    )
    stats = store.read("player_fixture_stats")
    if stats.is_empty():
        return empty
    chosen = stats.filter(pl.col("season") == season)
    if chosen.is_empty():
        return empty
    return (
        chosen.group_by("gw", "player_key")
        .agg(
            pl.col("total_points").sum().alias("actual_pts"),
            pl.col("minutes").sum().alias("actual_minutes"),
        )
        .select("gw", "player_key", "actual_pts", "actual_minutes")
    )


def simulate_hindsight_oracle(
    candidates: pl.DataFrame,
    actuals: pl.DataFrame,
    state: SquadState,
    *,
    season: str,
    gameweeks: list[int],
    horizon: int,
    model_id: str | None = None,
    chip_arm: ChipArm = "off",
    chip_rules_season: str | None = None,
    fixtures: pl.DataFrame | None = None,
    force_play: bool = True,
) -> pl.DataFrame:
    """Play a constrained hindsight season via ``simulate_decisions``.

    Args:
        candidates: ``gw``, ``player_key``, ``position``, ``team_key``, ``price``.
        actuals: Realised points (and optional minutes) per player-gameweek.
        state: Starting bank / FT / ownership (empty owned = GW1 draft).
        season: Season label written onto rows.
        gameweeks: Deadlines to play, in order.
        horizon: Planner look-ahead (same meaning as the decision bakeoff).
        model_id: Written onto every row. Defaults via ``oracle_ledger_model_id``.
        chip_arm: ``off`` (default N5), ``rule_based``, or ``optimised``.
        chip_rules_season: Season whose ``configs/chips`` file to load when
            chips are on. Defaults to ``season``.
        fixtures: Staged fixtures required for ``rule_based`` / ``optimised``.
        force_play: Forwarded to the optimised chip scheduler (default true).

    Returns:
        A frame matching ``DecisionSchema``.

    Raises:
        ValueError: Propagated from ``simulate_decisions`` when a week has no
            candidates or the squad rules cannot be met.
    """
    ledger_id = model_id if model_id is not None else oracle_ledger_model_id(chip_arm)
    predictions = foresight_predictions(actuals, gameweeks)
    return simulate_decisions(
        candidates,
        predictions,
        actuals,
        state,
        model_id=ledger_id,
        season=season,
        gameweeks=gameweeks,
        horizon=horizon,
        chip_arm=chip_arm,
        chip_rules_season=chip_rules_season,
        fixtures=fixtures,
        force_play=force_play,
    )


def run_hindsight_oracle(
    data_dir: Path,
    season: str,
    from_gw: int,
    to_gw: int,
    horizon: int,
    *,
    actuals_model: str | None = None,
    chip_arm: ChipArm = "off",
    chip_rules_season: str | None = None,
    force_play: bool = True,
    prefer_staged_actuals: bool = False,
) -> Path:
    """Simulate the constrained oracle from staged data and append the ledger.

    Uses the same draft rules as ``run_decisions``: 1000 budget, one free
    transfer, YAML decay / banking cap / hit cost. Actuals default to the
    prediction ledger (deduplicated like the decision bakeoff); set
    ``prefer_staged_actuals`` (or leave the ledger empty) to score from
    staged ``player_fixture_stats`` instead.

    Args:
        data_dir: Root with ``staged/`` and ``ledger/``.
        season: Season to play.
        from_gw: First gameweek.
        to_gw: Last gameweek, inclusive.
        horizon: Planner horizon.
        actuals_model: YAML spec whose ledger actuals to use. Defaults to the
            first available model_id in the prediction ledger for ``season``.
        chip_arm: Chip policy (default chips off for N5 board continuity).
        chip_rules_season: Chip-rules season when chips are on.
        force_play: Optimised-arm half-window packing (default true).
        prefer_staged_actuals: Prefer staged stats over the prediction ledger.

    Returns:
        The decision parquet path.

    Raises:
        KeyError: Unknown ``actuals_model``.
        ValueError: No actuals for the season, or chips on without fixtures.
    """
    rules = load_squad_state()
    draft = rules.model_copy(update={"owned": [], "bank": 1000, "free_transfers": 1})
    rules_season = chip_rules_season if chip_rules_season is not None else season
    if chip_arm != "off":
        counts = empty_remaining()
        counts.update(ChipRules.for_season(rules_season).remaining_counts())
        draft = draft.model_copy(
            update={"chips": ChipInventory(remaining=counts, played=[])}
        )
    store = DataStore(data_dir)
    gameweeks = list(range(from_gw, to_gw + 1))
    pools = [
        candidates_as_of(PITData(store).view(GWKey(season, gw))).with_columns(
            pl.lit(gw).alias("gw")
        )
        for gw in gameweeks
    ]
    actuals = _resolve_oracle_actuals(
        data_dir,
        store,
        season,
        actuals_model=actuals_model,
        prefer_staged=prefer_staged_actuals,
    )
    if actuals.is_empty():
        raise ValueError(f"no oracle actuals for season={season}")
    fixtures = (
        store.read("fixtures").filter(pl.col("season") == season)
        if chip_arm != "off"
        else None
    )
    if chip_arm != "off" and (fixtures is None or fixtures.is_empty()):
        raise ValueError(f"chip_arm={chip_arm!r} requires staged fixtures for {season}")
    frame = simulate_hindsight_oracle(
        pl.concat(pools),
        actuals,
        draft,
        season=season,
        gameweeks=gameweeks,
        horizon=horizon,
        chip_arm=chip_arm,
        chip_rules_season=chip_rules_season,
        fixtures=fixtures,
        force_play=force_play,
    )
    return write_decisions(data_dir, frame)


def attainable_share(realised: int, attainable: int) -> float:
    """Return realised ÷ attainable; ``0.0`` when attainable is non-positive."""
    if attainable <= 0:
        return 0.0
    return float(realised) / float(attainable)


def chip_timing_regret(
    *,
    chips_enabled: bool = False,
    chip: str | None = None,
    regret: int | None = None,
) -> ChipTimingRegret:
    """Return chip-timing regret, or a chips-off placeholder.

    Args:
        chips_enabled: When false (decision-sim v1), returns ``status='chips_off'``.
        chip: Optional chip name once Wave C wires inventory.
        regret: Points left on the table vs hindsight chip timing.

    Returns:
        A ``ChipTimingRegret`` value object.
    """
    if not chips_enabled:
        return ChipTimingRegret(chip=None, regret=None, status="chips_off")
    return ChipTimingRegret(chip=chip, regret=regret, status="computed")


def component_regret(decisions: pl.DataFrame) -> pl.DataFrame:
    """Compute captaincy, XI/bench, transfer, and chip-timing regret per GW.

    Definitions (all in realised points after autosub / vice-on-blank):

    - **Captaincy**: best armband contribution inside the chosen XI minus the
      contribution from the chosen captain/vice path.
    - **XI/bench**: best XI+captain from the owned fifteen (via
      ``select_squad`` on actuals) minus the chosen XI with an optimal
      captain — isolates lineup order from captaincy.
    - **Transfers**: hindsight-optimal single-GW action from the pre-transfer
      squad (H=1, actuals as ``xpts``, same FT/hit/bank) net of hits, minus
      the policy's realised net. Clamped at zero.
    - **Chip timing**: ``null`` while chips are off (``chip_timing_regret``).

    Args:
        decisions: Rows matching ``DecisionSchema`` for one or more models.

    Returns:
        A frame matching ``ComponentRegretSchema``.

    Raises:
        ValueError: Required columns are missing.
    """
    required = {
        "model_id",
        "season",
        "gw",
        "player_key",
        "position",
        "in_xi",
        "captain",
        "vice",
        "bought",
        "sold",
        "actual_pts",
        "actual_minutes",
        "hit",
        "free_transfers",
    }
    missing = required - set(decisions.columns)
    if missing:
        raise ValueError(f"decisions missing columns: {sorted(missing)}")
    if decisions.is_empty():
        return _empty_regret()

    # Transfer regret needs real candidate market fields; DecisionSchema omits
    # them, so ledger-only frames leave transfer_regret at 0.
    has_market = "price" in decisions.columns and "team_key" in decisions.columns
    chip = chip_timing_regret()
    rows: list[dict[str, object]] = []
    keys = (
        decisions.select("model_id", "season", "gw")
        .unique()
        .sort("model_id", "season", "gw")
    )
    for model_id, season, gameweek in keys.iter_rows():
        part = decisions.filter(
            (pl.col("model_id") == model_id)
            & (pl.col("season") == season)
            & (pl.col("gw") == gameweek)
        )
        hit = float(part["hit"][0]) if part.height else 0.0
        free = int(part["free_transfers"][0]) if part.height else 1
        realised = score_gameweek(part, hit=hit)
        cap_regret = _captaincy_regret(part)
        xi_regret = _xi_bench_regret(part)
        xfer_regret = (
            _transfer_regret(part, hit=hit, free_transfers=free) if has_market else 0
        )
        rows.append(
            {
                "model_id": str(model_id),
                "season": str(season),
                "gw": int(gameweek),
                "realised_points": realised,
                "captaincy_regret": cap_regret,
                "xi_bench_regret": xi_regret,
                "transfer_regret": xfer_regret,
                "chip_timing_regret": chip.regret,
            }
        )
    return _regret_frame(pl.DataFrame(rows))


def summarise_attainable(
    decisions: pl.DataFrame,
    oracle: pl.DataFrame,
) -> pl.DataFrame:
    """Join policy season totals to the oracle and attach component regret.

    Args:
        decisions: Policy decision rows (exclude the oracle model id).
        oracle: Constrained hindsight rows (``ORACLE_MODEL_ID``).

    Returns:
        A frame matching ``AttainableShareSchema``.

    Raises:
        ValueError: Oracle has no rows for a policy season, or columns missing.
    """
    policy = decisions.filter(pl.col("model_id") != ORACLE_MODEL_ID)
    summary = summarise_decisions(policy)
    if summary.is_empty():
        return _empty_share()
    oracle_summary = summarise_decisions(oracle)
    if oracle_summary.is_empty():
        raise ValueError("oracle decisions are empty")
    attainable_by_season = {
        str(row["season"]): int(row["total_points"])
        for row in oracle_summary.iter_rows(named=True)
    }
    regret = component_regret(policy)
    regret_tot = (
        regret.group_by("model_id", "season").agg(
            pl.col("captaincy_regret").sum(),
            pl.col("xi_bench_regret").sum(),
            pl.col("transfer_regret").sum(),
            pl.col("chip_timing_regret").sum(),
        )
        if not regret.is_empty()
        else pl.DataFrame(
            schema={
                "model_id": pl.Utf8(),
                "season": pl.Utf8(),
                "captaincy_regret": pl.Int64(),
                "xi_bench_regret": pl.Int64(),
                "transfer_regret": pl.Int64(),
                "chip_timing_regret": pl.Int64(),
            }
        )
    )
    joined = summary.join(regret_tot, on=["model_id", "season"], how="left")
    out_rows: list[dict[str, object]] = []
    for row in joined.iter_rows(named=True):
        season = str(row["season"])
        if season not in attainable_by_season:
            raise ValueError(f"oracle has no total for season={season}")
        attainable = attainable_by_season[season]
        total = int(row["total_points"])
        out_rows.append(
            {
                "model_id": str(row["model_id"]),
                "season": season,
                "total_points": total,
                "attainable_points": attainable,
                "share_of_attainable": attainable_share(total, attainable),
                "gameweeks": int(row["gameweeks"]),
                "captaincy_regret": int(row.get("captaincy_regret") or 0),
                "xi_bench_regret": int(row.get("xi_bench_regret") or 0),
                "transfer_regret": int(row.get("transfer_regret") or 0),
                "chip_timing_regret": row.get("chip_timing_regret"),
            }
        )
    return _share_frame(pl.DataFrame(out_rows))


def enrich_decision_summary(
    summary: pl.DataFrame,
    oracle: pl.DataFrame,
) -> pl.DataFrame:
    """Attach ``attainable_points`` and ``share_of_attainable`` to a scoreboard.

    Thin board hook: leaves component regret to ``summarise_attainable``.

    Args:
        summary: Output of ``summarise_decisions``.
        oracle: Oracle decision rows (or already summarised oracle totals with
            ``season`` and ``total_points``).

    Returns:
        ``summary`` plus attainable columns. Rows whose season is missing from
        the oracle keep null shares.
    """
    if summary.is_empty():
        return summary.with_columns(
            pl.lit(None).cast(pl.Int64).alias("attainable_points"),
            pl.lit(None).cast(pl.Float64).alias("share_of_attainable"),
        )
    if "gw" in oracle.columns:
        oracle_summary = summarise_decisions(oracle)
    else:
        oracle_summary = oracle
    by_season = {
        str(row["season"]): int(row["total_points"])
        for row in oracle_summary.iter_rows(named=True)
    }
    attainable = [by_season.get(str(season)) for season in summary["season"].to_list()]
    shares = [
        None if value is None else attainable_share(int(total), int(value))
        for total, value in zip(
            summary["total_points"].to_list(), attainable, strict=True
        )
    ]
    return summary.with_columns(
        pl.Series("attainable_points", attainable, dtype=pl.Int64),
        pl.Series("share_of_attainable", shares, dtype=pl.Float64),
    )


def format_attainable_scoreboard(summary: pl.DataFrame) -> str:
    """Render season totals with share of attainable and regret columns."""
    if summary.is_empty():
        return "_No decision rows to score._"
    has_share = "share_of_attainable" in summary.columns
    has_regret = "captaincy_regret" in summary.columns
    headers = ["Rank", "Model", "Season", "Points", "GWs", "Mean/GW"]
    if has_share:
        headers.extend(["Attainable", "Share"])
    if has_regret:
        headers.extend(["Cap regret", "XI regret", "Xfer regret", "Chip regret"])
    lines = [
        "| " + " | ".join(headers) + " |",
        "| "
        + " | ".join(["---:"] + ["---"] * 2 + ["---:"] * (len(headers) - 3))
        + " |",
    ]
    for index, row in enumerate(summary.iter_rows(named=True), start=1):
        gameweeks = int(row["gameweeks"])
        mean = float(row["total_points"]) / gameweeks if gameweeks else 0.0
        cells = [
            str(index),
            str(row["model_id"]),
            str(row["season"]),
            str(row["total_points"]),
            str(gameweeks),
            f"{mean:.1f}",
        ]
        if has_share:
            att = row.get("attainable_points")
            share = row.get("share_of_attainable")
            cells.append("-" if att is None else str(int(att)))
            cells.append("-" if share is None else f"{float(share):.3f}")
        if has_regret:
            chip = row.get("chip_timing_regret")
            cells.extend(
                [
                    str(int(row.get("captaincy_regret") or 0)),
                    str(int(row.get("xi_bench_regret") or 0)),
                    str(int(row.get("transfer_regret") or 0)),
                    "-" if chip is None else str(int(chip)),
                ]
            )
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _captaincy_regret(part: pl.DataFrame) -> int:
    kept = part.filter(~pl.col("sold")) if "sold" in part.columns else part
    xi = kept.filter(pl.col("in_xi"))
    if xi.is_empty():
        return 0
    hit = 0.0
    chosen = score_gameweek(kept, hit=hit)
    # Best captain in the chosen XI: try each XI player as captain / vice pair.
    best = chosen
    keys = xi["player_key"].to_list()
    for captain_key in keys:
        for vice_key in keys:
            if vice_key == captain_key:
                continue
            trial = kept.with_columns(
                (pl.col("player_key") == captain_key).alias("captain"),
                (pl.col("player_key") == vice_key).alias("vice"),
            )
            best = max(best, score_gameweek(trial, hit=hit))
    return max(0, best - chosen)


def _xi_bench_regret(part: pl.DataFrame) -> int:
    kept = part.filter(~pl.col("sold")) if "sold" in part.columns else part
    if kept.height < 15:
        # Partial squads (sold rows only) — skip.
        owned = kept
    else:
        owned = kept
    if owned.filter(pl.col("in_xi")).is_empty():
        return 0
    # Chosen XI with optimal captain (strip captaincy from the gap).
    chosen_opt = _best_captain_score(owned)
    optimal = _optimal_lineup_score(owned)
    return max(0, optimal - chosen_opt)


def _optimal_lineup_score(owned: pl.DataFrame) -> int:
    """Best XI+captain from a fixed fifteen using actual points as xpts."""
    players = owned.select(
        "player_key",
        "position",
        pl.lit(0).alias("price"),
        pl.when(pl.col("actual_pts").is_null())
        .then(0)
        .otherwise(pl.col("actual_pts"))
        .cast(pl.Float64)
        .alias("xpts"),
        _team_key_expr(owned),
    )
    # select_squad needs team_key; synthesise if absent.
    try:
        selection = select_squad(players)
    except ValueError:
        return _best_captain_score(owned)
    picks = selection.players
    lined = owned.join(
        picks.select("player_key", "in_xi", "captain", "vice"),
        on="player_key",
        how="left",
        suffix="_opt",
    ).with_columns(
        pl.col("in_xi_opt").fill_null(False).alias("in_xi"),
        pl.col("captain_opt").fill_null(False).alias("captain"),
        pl.col("vice_opt").fill_null(False).alias("vice"),
    )
    return score_gameweek(lined, hit=0.0)


def _best_captain_score(owned: pl.DataFrame) -> int:
    kept = owned
    xi_keys = kept.filter(pl.col("in_xi"))["player_key"].to_list()
    if not xi_keys:
        return score_gameweek(kept, hit=0.0)
    best = score_gameweek(kept, hit=0.0)
    for captain_key in xi_keys:
        for vice_key in xi_keys:
            if vice_key == captain_key:
                continue
            trial = kept.with_columns(
                (pl.col("player_key") == captain_key).alias("captain"),
                (pl.col("player_key") == vice_key).alias("vice"),
            )
            best = max(best, score_gameweek(trial, hit=0.0))
    return best


def _transfer_regret(
    part: pl.DataFrame,
    *,
    hit: float,
    free_transfers: int,
) -> int:
    """Hindsight H=1 action from the pre-transfer squad minus policy net."""
    if "price" not in part.columns or "team_key" not in part.columns:
        # Decision ledger lacks candidate fields — transfer regret needs them.
        return 0
    policy_net = score_gameweek(part, hit=hit)
    pre_owned, bank = _pre_transfer_state(part)
    if not pre_owned:
        # GW1 draft: transfers are not the lever; skip.
        return 0
    pool = part.select(
        "player_key",
        "position",
        "team_key",
        "price",
    ).unique(subset=["player_key"])
    # Candidate pool for a single GW must include everyone we might buy.
    # When the ledger only has kept+sold rows, the pool is incomplete — then
    # the hindsight action can only reshuffle within those players.
    points = (
        part.group_by("player_key")
        .agg(pl.col("actual_pts").sum())
        .with_columns(
            pl.when(pl.col("actual_pts").is_null())
            .then(0.0)
            .otherwise(pl.col("actual_pts").cast(pl.Float64))
            .alias("xpts"),
            pl.lit(0).alias("step"),
        )
        .select("player_key", "step", "xpts")
    )
    state = SquadState(
        bank=max(bank, 0),
        free_transfers=max(free_transfers, 1),
        ft_bank_cap=max(free_transfers, 5),
        hit_cost=4,
        owned=pre_owned,
    )
    try:
        plan = plan_squad(pool, points, state, initial=False, horizon=1)
    except ValueError:
        return 0
    # Build a decision-like frame for the oracle action and score it.
    step = plan.picks.filter(pl.col("step") == 0)
    actual_by = {
        int(key): (0 if pts is None else int(pts))
        for key, pts in part.select("player_key", "actual_pts").iter_rows()
    }
    minutes_by = {
        int(key): (0 if mins is None else int(mins))
        for key, mins in part.select("player_key", "actual_minutes").iter_rows()
    }
    rows: list[dict[str, object]] = []
    for row in step.iter_rows(named=True):
        key = int(row["player_key"])
        rows.append(
            {
                "player_key": key,
                "position": str(row["position"]),
                "in_xi": bool(row["in_xi"]),
                "captain": bool(row["captain"]),
                "vice": bool(row["vice"]),
                "sold": bool(row["sold"]),
                "actual_pts": None if row["sold"] else actual_by.get(key, 0),
                "actual_minutes": None if row["sold"] else minutes_by.get(key, 0),
            }
        )
    best_net = score_gameweek(pl.DataFrame(rows), hit=float(plan.hits[0]))
    return max(0, best_net - policy_net)


def _pre_transfer_state(
    part: pl.DataFrame,
) -> tuple[list[OwnedPlayer], int]:
    kept = part.filter(~pl.col("sold"))
    sold = part.filter(pl.col("sold"))
    bought_keys = set(
        int(key) for key in kept.filter(pl.col("bought"))["player_key"].to_list()
    )
    pre: list[OwnedPlayer] = []
    # Players kept but not bought were already owned.
    for row in kept.iter_rows(named=True):
        key = int(row["player_key"])
        if key in bought_keys:
            continue
        price = int(row["price"]) if "price" in part.columns else 0
        pre.append(
            OwnedPlayer(player_key=key, selling_price=price, purchase_price=price)
        )
    for row in sold.iter_rows(named=True):
        price = int(row["price"]) if "price" in part.columns else 0
        pre.append(
            OwnedPlayer(
                player_key=int(row["player_key"]),
                selling_price=price,
                purchase_price=price,
            )
        )
    # Approximate bank: we do not store bank on the ledger; use 0 so the
    # hindsight plan can only buy with sale proceeds + zero cash (conservative).
    return pre, 0


def _team_key_expr(frame: pl.DataFrame) -> pl.Expr:
    if "team_key" in frame.columns:
        return pl.col("team_key")
    # Distinct synthetic clubs so max-3-per-club never binds.
    return (pl.col("player_key") % 1000 + 1).alias("team_key")


def _resolve_oracle_actuals(
    data_dir: Path,
    store: DataStore,
    season: str,
    *,
    actuals_model: str | None,
    prefer_staged: bool,
) -> pl.DataFrame:
    """Pick staged or ledger actuals for the oracle."""
    if prefer_staged:
        staged = actuals_from_staged(store, season)
        if not staged.is_empty():
            return staged
    ledger = read_predictions(data_dir)
    if not ledger.is_empty():
        try:
            model_id = _resolve_actuals_model_id(ledger, season, actuals_model)
        except ValueError:
            model_id = None
        if model_id is not None:
            from_ledger = _actual_rows_for_oracle(ledger, model_id, season)
            if not from_ledger.is_empty():
                return from_ledger
    return actuals_from_staged(store, season)


def _resolve_actuals_model_id(
    ledger: pl.DataFrame,
    season: str,
    actuals_model: str | None,
) -> str:
    if actuals_model is not None:
        return load_model_specs()[actuals_model].model_id
    if ledger.is_empty():
        raise ValueError("prediction ledger is empty")
    chosen = ledger.filter(pl.col("target_season") == season)
    if chosen.is_empty():
        raise ValueError(f"no ledger rows for season={season}")
    return str(chosen["model_id"][0])


def _actual_rows_for_oracle(
    ledger: pl.DataFrame, model_id: str, season: str
) -> pl.DataFrame:
    """Collapse ledger actuals the same way as ``decisions._actual_rows``."""
    empty = pl.DataFrame(
        schema={
            "gw": pl.Int64(),
            "player_key": pl.Int64(),
            "actual_pts": pl.Int64(),
            "actual_minutes": pl.Int64(),
        }
    )
    if ledger.is_empty():
        return empty
    chosen = ledger.filter(
        (pl.col("model_id") == model_id) & (pl.col("target_season") == season)
    )
    if chosen.is_empty():
        return empty
    scored = chosen.filter(pl.col("actual_pts").is_not_null())
    if scored.is_empty():
        return empty
    if "fixture_id" in scored.columns:
        scored = scored.sort("as_of_gw", "horizon").unique(
            subset=["target_gw", "player_key", "fixture_id"],
            keep="last",
        )
    elif "horizon" in scored.columns:
        scored = scored.filter(pl.col("horizon") == 1)
    if "actual_minutes" in scored.columns:
        return (
            scored.group_by("target_gw", "player_key")
            .agg(pl.col("actual_pts").sum(), pl.col("actual_minutes").sum())
            .rename({"target_gw": "gw"})
        )
    return (
        scored.group_by("target_gw", "player_key")
        .agg(pl.col("actual_pts").sum())
        .rename({"target_gw": "gw"})
        .with_columns(pl.lit(None).cast(pl.Int64).alias("actual_minutes"))
    )


def _empty_regret() -> pl.DataFrame:
    return _regret_frame(
        pl.DataFrame(
            schema={
                "model_id": pl.Utf8(),
                "season": pl.Utf8(),
                "gw": pl.Int64(),
                "realised_points": pl.Int64(),
                "captaincy_regret": pl.Int64(),
                "xi_bench_regret": pl.Int64(),
                "transfer_regret": pl.Int64(),
                "chip_timing_regret": pl.Int64(),
            }
        )
    )


def _empty_share() -> pl.DataFrame:
    return _share_frame(
        pl.DataFrame(
            schema={
                "model_id": pl.Utf8(),
                "season": pl.Utf8(),
                "total_points": pl.Int64(),
                "attainable_points": pl.Int64(),
                "share_of_attainable": pl.Float64(),
                "gameweeks": pl.Int64(),
                "captaincy_regret": pl.Int64(),
                "xi_bench_regret": pl.Int64(),
                "transfer_regret": pl.Int64(),
                "chip_timing_regret": pl.Int64(),
            }
        )
    )


def _regret_frame(frame: pl.DataFrame) -> pl.DataFrame:
    checked = ComponentRegretSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked


def _share_frame(frame: pl.DataFrame) -> pl.DataFrame:
    checked = AttainableShareSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked
