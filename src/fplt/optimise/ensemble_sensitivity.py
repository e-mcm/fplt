"""Ensemble sensitivity: DC vs LGBM (or bakeoff pair) move-frequency board.

Solves the same pre-deadline squad once per forecast member, compares week-1
actions (transfers, XI, captain, vice), then advances ownership with the
**control** plan. Disagreement is therefore forecast-driven, not an artifact
of diverging independent decision sims.

Default bakeoff pair: ``dc_layered_v3`` (control) vs ``lgbm_direct_v1``.
Does not change ``PRODUCTION_MODELS`` or the MILP. Chips stay off for this
board (Wave D2 reporting only).
"""

from __future__ import annotations

import json
import logging
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from fplt.backtest.ledger import read_predictions
from fplt.data.pit import PITData
from fplt.data.store import DataStore
from fplt.models.spec import load_model_specs
from fplt.optimise.decisions import (
    advance_squad_from_plan,
    candidates_as_of,
    prediction_rows_from_ledger,
)
from fplt.optimise.plan import GameweekPlan, plan_squad
from fplt.optimise.prices import apply_market_prices, market_prices_from_candidates
from fplt.optimise.pruning_loss import (
    Week1Actions,
    week1_actions_agree,
    week1_actions_from_plan,
)
from fplt.optimise.state import SquadState, load_squad_state
from fplt.schemas.ensemble_sensitivity import (
    EnsembleSensitivityRowSchema,
    EnsembleSensitivitySummarySchema,
)
from fplt.types import GWKey

logger = logging.getLogger(__name__)

# Current forecast bakeoff pair (Wave E1 / D2). Control = live layered default.
DEFAULT_ENSEMBLE_CONTROL = "dc_layered_v3"
DEFAULT_ENSEMBLE_CHALLENGERS: tuple[str, ...] = ("lgbm_direct_v1",)


@dataclass(frozen=True)
class EnsembleSensitivityReport:
    """Per-GW rows, pair summary, and the member ids used."""

    rows: pl.DataFrame
    summary: pl.DataFrame
    control_model_id: str
    challenger_model_ids: tuple[str, ...]
    season: str
    horizon: int


def agreement_flags(left: Week1Actions, right: Week1Actions) -> dict[str, bool]:
    """Boolean agreement flags for each week-1 action field group.

    Args:
        left: Control (or reference) week-1 actions.
        right: Challenger week-1 actions.

    Returns:
        Keys ``week1``, ``transfer``, ``xi``, ``captain``, ``vice``.
    """
    return {
        "week1": week1_actions_agree(left, right),
        "transfer": (
            left.transfers_in == right.transfers_in
            and left.transfers_out == right.transfers_out
        ),
        "xi": left.xi == right.xi,
        "captain": left.captain == right.captain,
        "vice": left.vice == right.vice,
    }


def transfer_in_symmetric_diff(left: Week1Actions, right: Week1Actions) -> int:
    """Cardinality of the symmetric difference of transfers-in sets."""
    return len(set(left.transfers_in) ^ set(right.transfers_in))


def week1_actions_from_decision_rows(
    frame: pl.DataFrame,
    *,
    initial: bool,
) -> Week1Actions:
    """Extract week-1 actions from ``DecisionSchema`` rows for one model × GW.

    Args:
        frame: Decision rows for a single model and gameweek.
        initial: True when the gameweek was an empty-squad draft (buys are not
            transfers).

    Returns:
        Sorted transfer / XI keys and armbands.

    Raises:
        ValueError: ``frame`` is empty.
    """
    if frame.is_empty():
        raise ValueError("decision frame is empty")
    if initial:
        transfers_in: tuple[int, ...] = ()
        transfers_out: tuple[int, ...] = ()
    else:
        transfers_in = _sorted_keys(frame.filter(pl.col("bought")))
        transfers_out = _sorted_keys(frame.filter(pl.col("sold")))
    xi = _sorted_keys(frame.filter(pl.col("in_xi")))
    captains = _sorted_keys(frame.filter(pl.col("captain")))
    vices = _sorted_keys(frame.filter(pl.col("vice")))
    return Week1Actions(
        transfers_in=transfers_in,
        transfers_out=transfers_out,
        xi=xi,
        captain=captains[0] if captains else None,
        vice=vices[0] if vices else None,
    )


def run_ensemble_sensitivity(
    candidates: pl.DataFrame,
    predictions_by_model: Mapping[str, pl.DataFrame],
    state: SquadState,
    *,
    season: str,
    gameweeks: Sequence[int],
    horizon: int,
    control_model_id: str,
    challenger_model_ids: Sequence[str],
) -> EnsembleSensitivityReport:
    """Shared-state walk: solve each member, compare, advance on control.

    Args:
        candidates: ``gw``, ``player_key``, ``position``, ``team_key``, ``price``.
        predictions_by_model: ``as_of_gw``, ``target_gw``, ``player_key``,
            ``xpts`` keyed by content-addressed ``model_id``.
        state: Starting squad (empty owned = GW1 draft).
        season: Season label written onto every row.
        gameweeks: Deadlines to play, in order.
        horizon: Planner look-ahead (capped at gameweek 38).
        control_model_id: Member whose plan advances ownership.
        challenger_model_ids: Members compared against control each GW.

    Returns:
        Validated rows + summary for every control × challenger pair.

    Raises:
        ValueError: Empty gameweeks / challengers, unknown model id, or a
            missing candidate pool.
        KeyError: A requested ``model_id`` is absent from ``predictions_by_model``.
    """
    if not gameweeks:
        raise ValueError("gameweeks must be non-empty")
    if not challenger_model_ids:
        raise ValueError("challenger_model_ids must be non-empty")
    if control_model_id not in predictions_by_model:
        raise KeyError(f"control model_id {control_model_id!r} missing predictions")
    for model_id in challenger_model_ids:
        if model_id not in predictions_by_model:
            raise KeyError(f"challenger model_id {model_id!r} missing predictions")
        if model_id == control_model_id:
            raise ValueError("challenger_model_ids must not include the control")

    current = state
    rows: list[dict[str, object]] = []
    for gameweek in gameweeks:
        window = [
            gameweek + offset for offset in range(horizon) if gameweek + offset <= 38
        ]
        pool = candidates.filter(pl.col("gw") == gameweek).drop("gw")
        if pool.is_empty():
            raise ValueError(f"no candidates for gw {gameweek}")
        current = apply_market_prices(current, market_prices_from_candidates(pool))
        initial = not current.owned
        control_plan = _solve_member(
            pool,
            predictions_by_model[control_model_id],
            current,
            as_of=gameweek,
            window=window,
            initial=initial,
        )
        control_actions = week1_actions_from_plan(control_plan, initial=initial)
        for challenger_id in challenger_model_ids:
            challenger_plan = _solve_member(
                pool,
                predictions_by_model[challenger_id],
                current,
                as_of=gameweek,
                window=window,
                initial=initial,
            )
            challenger_actions = week1_actions_from_plan(
                challenger_plan, initial=initial
            )
            flags = agreement_flags(control_actions, challenger_actions)
            rows.append(
                {
                    "season": season,
                    "gw": int(gameweek),
                    "control_model_id": control_model_id,
                    "challenger_model_id": challenger_id,
                    "week1_agree": flags["week1"],
                    "transfer_agree": flags["transfer"],
                    "xi_agree": flags["xi"],
                    "captain_agree": flags["captain"],
                    "vice_agree": flags["vice"],
                    "n_transfers_control": len(control_actions.transfers_in),
                    "n_transfers_challenger": len(challenger_actions.transfers_in),
                    "n_transfer_in_symmetric_diff": transfer_in_symmetric_diff(
                        control_actions, challenger_actions
                    ),
                    "hit_control": float(control_plan.hits[0]),
                    "hit_challenger": float(challenger_plan.hits[0]),
                    "horizon": len(window),
                }
            )
            logger.info(
                "ensemble-sensitivity gw=%s control=%s challenger=%s "
                "week1_agree=%s transfer_agree=%s captain_agree=%s",
                gameweek,
                control_model_id,
                challenger_id,
                flags["week1"],
                flags["transfer"],
                flags["captain"],
            )
        current = advance_squad_from_plan(
            current, control_plan, pool, gameweek=gameweek, rules=None
        )

    frame = pl.DataFrame(rows)
    checked = EnsembleSensitivityRowSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    summary = summarise_ensemble_sensitivity(checked)
    return EnsembleSensitivityReport(
        rows=checked,
        summary=summary,
        control_model_id=control_model_id,
        challenger_model_ids=tuple(challenger_model_ids),
        season=season,
        horizon=horizon,
    )


def run_ensemble_sensitivity_from_data_dir(
    data_dir: Path,
    *,
    season: str,
    from_gw: int,
    to_gw: int,
    horizon: int,
    control: str = DEFAULT_ENSEMBLE_CONTROL,
    challengers: Sequence[str] | None = None,
) -> EnsembleSensitivityReport:
    """Load ledger predictions + staged candidates and run the shared-state board.

    Args:
        data_dir: Root with ``staged/`` and ``ledger/``.
        season: Season label.
        from_gw: First gameweek.
        to_gw: Last gameweek, inclusive.
        horizon: Planner horizon (must match backtest ``--steps``).
        control: YAML ModelSpec name for the ownership path (default
            ``dc_layered_v3``).
        challengers: YAML ModelSpec names to compare (default
            ``lgbm_direct_v1``).

    Returns:
        Report from :func:`run_ensemble_sensitivity`.

    Raises:
        KeyError: Unknown ModelSpec name.
        ValueError: Empty challenger list or overlapping control.
    """
    specs = load_model_specs()
    control_spec = specs[control]
    names = (
        list(challengers)
        if challengers is not None
        else list(DEFAULT_ENSEMBLE_CHALLENGERS)
    )
    if not names:
        raise ValueError("challengers must be non-empty")
    if control in names:
        raise ValueError("challengers must not include the control model name")
    challenger_specs = [specs[name] for name in names]

    squad_rules = load_squad_state()
    draft = squad_rules.model_copy(
        update={"owned": [], "bank": 1000, "free_transfers": 1}
    )
    store = DataStore(data_dir)
    gameweeks = list(range(from_gw, to_gw + 1))
    pools = [
        candidates_as_of(PITData(store).view(GWKey(season, gw))).with_columns(
            pl.lit(gw).alias("gw")
        )
        for gw in gameweeks
    ]
    ledger = read_predictions(data_dir)
    predictions_by_model: dict[str, pl.DataFrame] = {
        control_spec.model_id: prediction_rows_from_ledger(
            ledger, control_spec.model_id, season
        )
    }
    for spec in challenger_specs:
        predictions_by_model[spec.model_id] = prediction_rows_from_ledger(
            ledger, spec.model_id, season
        )
    return run_ensemble_sensitivity(
        pl.concat(pools),
        predictions_by_model,
        draft,
        season=season,
        gameweeks=gameweeks,
        horizon=horizon,
        control_model_id=control_spec.model_id,
        challenger_model_ids=[spec.model_id for spec in challenger_specs],
    )


def summarise_ensemble_sensitivity(rows: pl.DataFrame) -> pl.DataFrame:
    """Aggregate per-GW flags into move-frequency rates per pair.

    Args:
        rows: Frame matching ``EnsembleSensitivityRowSchema``.

    Returns:
        One row per control × challenger with agree rates and move frequencies.
    """
    checked = EnsembleSensitivityRowSchema.validate(rows)
    assert isinstance(checked, pl.DataFrame)
    if checked.is_empty():
        empty = pl.DataFrame(
            schema={
                "control_model_id": pl.String(),
                "challenger_model_id": pl.String(),
                "n_gw": pl.Int64(),
                "week1_agree_rate": pl.Float64(),
                "transfer_agree_rate": pl.Float64(),
                "xi_agree_rate": pl.Float64(),
                "captain_agree_rate": pl.Float64(),
                "vice_agree_rate": pl.Float64(),
                "move_frequency": pl.Float64(),
                "transfer_move_frequency": pl.Float64(),
                "mean_transfer_in_symmetric_diff": pl.Float64(),
                "mean_abs_hit_diff": pl.Float64(),
            }
        )
        validated_empty = EnsembleSensitivitySummarySchema.validate(empty)
        assert isinstance(validated_empty, pl.DataFrame)
        return validated_empty
    summary = (
        checked.group_by("control_model_id", "challenger_model_id", maintain_order=True)
        .agg(
            pl.len().alias("n_gw"),
            pl.col("week1_agree").mean().alias("week1_agree_rate"),
            pl.col("transfer_agree").mean().alias("transfer_agree_rate"),
            pl.col("xi_agree").mean().alias("xi_agree_rate"),
            pl.col("captain_agree").mean().alias("captain_agree_rate"),
            pl.col("vice_agree").mean().alias("vice_agree_rate"),
            pl.col("n_transfer_in_symmetric_diff")
            .mean()
            .alias("mean_transfer_in_symmetric_diff"),
            (pl.col("hit_control") - pl.col("hit_challenger"))
            .abs()
            .mean()
            .alias("mean_abs_hit_diff"),
        )
        .with_columns(
            (1.0 - pl.col("week1_agree_rate")).alias("move_frequency"),
            (1.0 - pl.col("transfer_agree_rate")).alias("transfer_move_frequency"),
        )
        .select(
            "control_model_id",
            "challenger_model_id",
            "n_gw",
            "week1_agree_rate",
            "transfer_agree_rate",
            "xi_agree_rate",
            "captain_agree_rate",
            "vice_agree_rate",
            "move_frequency",
            "transfer_move_frequency",
            "mean_transfer_in_symmetric_diff",
            "mean_abs_hit_diff",
        )
        .sort("challenger_model_id")
    )
    validated = EnsembleSensitivitySummarySchema.validate(summary)
    assert isinstance(validated, pl.DataFrame)
    return validated


def format_ensemble_sensitivity_board(summary: pl.DataFrame) -> str:
    """Render the move-frequency board as a markdown table."""
    if summary.is_empty():
        return "_No ensemble-sensitivity rows to score._"
    lines = [
        "| Control | Challenger | GWs | Week-1 agree | Transfer agree | "
        "XI agree | Captain agree | Move freq | Transfer move freq |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary.iter_rows(named=True):
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["control_model_id"]),
                    str(row["challenger_model_id"]),
                    str(int(row["n_gw"])),
                    f"{float(row['week1_agree_rate']):.3f}",
                    f"{float(row['transfer_agree_rate']):.3f}",
                    f"{float(row['xi_agree_rate']):.3f}",
                    f"{float(row['captain_agree_rate']):.3f}",
                    f"{float(row['move_frequency']):.3f}",
                    f"{float(row['transfer_move_frequency']):.3f}",
                ]
            )
            + " |"
        )
    return "\n".join(lines)


def write_ensemble_sensitivity_report(
    report: EnsembleSensitivityReport,
    output_dir: Path,
) -> dict[str, Path]:
    """Write parquet + JSON summary under ``output_dir``.

    Args:
        report: Harness result.
        output_dir: Destination directory (created if missing).

    Returns:
        Paths for ``rows.parquet``, ``summary.parquet``, and ``summary.json``.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    rows_path = output_dir / "rows.parquet"
    summary_path = output_dir / "summary.parquet"
    json_path = output_dir / "summary.json"
    report.rows.write_parquet(rows_path)
    report.summary.write_parquet(summary_path)
    payload = {
        "season": report.season,
        "horizon": report.horizon,
        "control_model_id": report.control_model_id,
        "challenger_model_ids": list(report.challenger_model_ids),
        "n_gw_rows": report.rows.height,
        "summary": report.summary.to_dicts(),
        "metric_notes": {
            "week1_agree": (
                "identical transfers_in/out, XI, captain, and vice "
                "(shared pre-deadline squad)"
            ),
            "move_frequency": "1 - week1_agree_rate (forecast-driven thrashing)",
            "transfer_move_frequency": "1 - transfer_agree_rate",
            "ownership_path": "control plan advances the shared squad each GW",
            "chips": "off",
        },
    }
    json_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    return {"rows": rows_path, "summary": summary_path, "json": json_path}


def _solve_member(
    pool: pl.DataFrame,
    predictions: pl.DataFrame,
    state: SquadState,
    *,
    as_of: int,
    window: list[int],
    initial: bool,
) -> GameweekPlan:
    return plan_squad(
        pool,
        _horizon_points(predictions, as_of, window),
        state,
        initial=initial,
        horizon=len(window),
    )


def _horizon_points(
    predictions: pl.DataFrame, as_of: int, window: list[int]
) -> pl.DataFrame:
    """Map ``as_of`` / ``target_gw`` predictions onto planner ``step`` grain."""
    empty = pl.DataFrame(
        schema={"player_key": pl.Int64(), "step": pl.Int64(), "xpts": pl.Float64()}
    )
    if predictions.is_empty():
        return empty
    chosen = predictions.filter(
        (pl.col("as_of_gw") == as_of) & pl.col("target_gw").is_in(window)
    )
    if chosen.is_empty():
        return empty
    step_of = {gameweek: step for step, gameweek in enumerate(window)}
    return chosen.with_columns(
        pl.col("target_gw").replace_strict(step_of).alias("step")
    ).select("player_key", "step", "xpts")


def _sorted_keys(frame: pl.DataFrame) -> tuple[int, ...]:
    if frame.is_empty():
        return ()
    return tuple(int(key) for key in frame["player_key"].sort().to_list())
