"""Other-points measurement: haul Brier, top-decile cal, match residual.

Diagnostic board for Rank-3 (FDB-240): bonus / saves / defcon / residual other
are currently **independent of the simulated match** (goals / CS). This module
walks a layered points model, reports haul Brier + top xPts-decile haul
calibration + a pool captain-regret proxy, and measures the residual of the
other-points component against realised match context.

Measurement only — no keep A/B and no production flip.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl

import fplt.models.composite  # noqa: F401  # pyright: ignore[reportUnusedImport]
from fplt.backtest.decision_pool import (
    POOL_FILTER_DESCRIPTION,
    decision_pool_keys,
)
from fplt.backtest.walk_forward import allowed_seasons, should_refit, walk_forward
from fplt.data.pit import PITData, PITView
from fplt.data.store import DataStore
from fplt.models.spec import ModelSpec, get_component, get_model, load_model_specs
from fplt.schemas.other_points_measurement import (
    OTHER_POINTS_CAPTAIN_REGRET_COLUMNS,
    OTHER_POINTS_MEASUREMENT_SUMMARY_COLUMNS,
    OtherPointsCaptainRegretSchema,
    OtherPointsMeasurementSummarySchema,
)
from fplt.scoring.rules import MatchStats, Rules
from fplt.types import GWKey
from fplt.utils.frames import require_dataframe

logger = logging.getLogger(__name__)

_EPS = 1e-15
_DEFAULT_POINTS = "dc_layered_v4"
_HAUL_THRESHOLD = 10


@dataclass(frozen=True)
class OtherPointsMeasurementResult:
    """Summary metrics, captain-regret table, detail rows, and output path."""

    summary: pl.DataFrame
    captain_regret: pl.DataFrame
    detail: pl.DataFrame
    output_dir: Path


def run_other_points_measurement(
    data_dir: Path,
    season: str,
    *,
    from_gw: int = 1,
    to_gw: int = 10,
    points_model: str = _DEFAULT_POINTS,
    n_sims: int | None = None,
    output_dir: Path | None = None,
) -> OtherPointsMeasurementResult:
    """Walk-forward other-points / haul diagnostics for one season window.

    Args:
        data_dir: Root with ``staged/``.
        season: Evaluation season such as ``"2025-26"``.
        from_gw: First as-of gameweek scored (horizon-1).
        to_gw: Last as-of gameweek, inclusive.
        points_model: Layered points YAML name (default production stack).
        n_sims: Optional Monte Carlo override for faster smoke runs.
        output_dir: Where to write parquet / JSON / markdown.

    Returns:
        Validated summary / captain-regret frames and the output path.

    Raises:
        ValueError: Empty window, unknown YAML name, or invalid ``n_sims``.
        KeyError: Points model kind is not registered.
    """
    if from_gw > to_gw:
        raise ValueError(f"from_gw ({from_gw}) must be <= to_gw ({to_gw})")
    if n_sims is not None and n_sims < 1:
        raise ValueError(f"n_sims must be a positive int, got {n_sims!r}")
    specs = load_model_specs()
    if points_model not in specs:
        raise ValueError(f"unknown model spec {points_model!r}")
    spec = specs[points_model]
    if n_sims is not None:
        params = {**spec.params, "n_sims": n_sims}
        spec = spec.model_copy(update={"params": params})
    if "other" not in spec.components:
        raise ValueError(
            f"{points_model!r} has no other component; need a layered points model"
        )
    store = DataStore(data_dir)
    pit = PITData(store)
    as_ofs = [GWKey(season, gw) for gw in range(from_gw, to_gw + 1)]
    detail = _collect_detail(pit, store, as_ofs, spec)
    captain = captain_regret_by_gw(detail)
    summary = _summarise(detail, captain, model_name=spec.name)
    out = output_dir or (
        data_dir / "other_points_measurement" / f"{season}_gw{from_gw}-{to_gw}"
    )
    out.mkdir(parents=True, exist_ok=True)
    summary.write_parquet(out / "summary.parquet")
    captain.write_parquet(out / "captain_regret.parquet")
    detail.write_parquet(out / "detail.parquet")
    regret_md = format_captain_regret_markdown(captain) + "\n"
    (out / "captain_regret.md").write_text(regret_md)
    payload = {
        "season": season,
        "from_gw": from_gw,
        "to_gw": to_gw,
        "points_model": spec.name,
        "n_sims": int(spec.params.get("n_sims", 0)),
        "other_component": spec.components["other"].name,
        "decision_pool": POOL_FILTER_DESCRIPTION,
        "note": (
            "Measurement only. Other points remain independent of simulated "
            "goals/CS until a later match-conditioned keep A/B."
        ),
        "summary": summary.to_dicts(),
        "captain_regret": captain.to_dicts(),
    }
    (out / "summary.json").write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("wrote other-points measurement under %s", out)
    return OtherPointsMeasurementResult(
        summary=summary,
        captain_regret=captain,
        detail=detail,
        output_dir=out,
    )


def minutes_bucket(minutes: int) -> str:
    """Map realised minutes to the other-points bucket labels."""
    if minutes <= 0:
        return "0"
    if minutes < 60:
        return "1-59"
    return "60+"


def team_goals_band(goals: int) -> str:
    """Coarse team-goals slice for independence diagnostics."""
    if goals <= 0:
        return "0"
    if goals == 1:
        return "1"
    return "2+"


def brier_score(probs: np.ndarray, outcomes: np.ndarray) -> float | None:
    """Mean Bernoulli Brier score, or ``None`` when empty."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    if p.size == 0:
        return None
    return float(np.mean((p - y) ** 2))


def log_score(probs: np.ndarray, outcomes: np.ndarray) -> float | None:
    """Mean Bernoulli log loss (lower is better), or ``None`` when empty."""
    p = np.asarray(probs, dtype=np.float64)
    y = np.asarray(outcomes, dtype=np.float64)
    if p.size == 0:
        return None
    clipped = np.clip(p, _EPS, 1.0 - _EPS)
    return float(-np.mean(y * np.log(clipped) + (1.0 - y) * np.log(1.0 - clipped)))


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float | None:
    """Pearson correlation, or ``None`` when undefined."""
    xx = np.asarray(x, dtype=np.float64)
    yy = np.asarray(y, dtype=np.float64)
    if xx.size < 2 or yy.size < 2 or xx.shape != yy.shape:
        return None
    if not (np.isfinite(xx).all() and np.isfinite(yy).all()):
        mask = np.isfinite(xx) & np.isfinite(yy)
        xx = xx[mask]
        yy = yy[mask]
    if xx.size < 2:
        return None
    if float(np.std(xx)) == 0.0 or float(np.std(yy)) == 0.0:
        return None
    return float(np.corrcoef(xx, yy)[0, 1])


def captain_regret_by_gw(detail: pl.DataFrame) -> pl.DataFrame:
    """Unconstrained pool captain regret: max-xPts pick vs max-actual oracle.

    This is a forecast-calibration proxy (not the squad-constrained regret
    board). Rows must be in the decision pool with non-null actuals.

    Args:
        detail: Per player–fixture measurement rows.

    Returns:
        Validated ``OtherPointsCaptainRegretSchema`` rows, one per GW.
    """
    empty = pl.DataFrame(schema=OTHER_POINTS_CAPTAIN_REGRET_COLUMNS)
    if detail.is_empty():
        return _validated_captain(empty)
    pool = detail.filter(
        pl.col("in_decision_pool")
        & pl.col("actual_pts").is_not_null()
        & pl.col("xpts").is_not_null()
    )
    if pool.is_empty():
        return _validated_captain(empty)
    rows: list[dict[str, object]] = []
    for part in pool.partition_by(["season", "gw"], maintain_order=True):
        chosen = part.sort("xpts", descending=True).row(0, named=True)
        oracle = part.sort("actual_pts", descending=True).row(0, named=True)
        chosen_actual = float(chosen["actual_pts"])
        oracle_actual = float(oracle["actual_pts"])
        rows.append(
            {
                "season": str(chosen["season"]),
                "gw": int(chosen["gw"]),
                "chosen_player_key": int(chosen["player_key"]),
                "oracle_player_key": int(oracle["player_key"]),
                "chosen_xpts": float(chosen["xpts"]),
                "chosen_actual": chosen_actual,
                "oracle_actual": oracle_actual,
                "regret": max(0.0, oracle_actual - chosen_actual),
                "n_pool": part.height,
            }
        )
    return _validated_captain(
        pl.DataFrame(rows, schema=OTHER_POINTS_CAPTAIN_REGRET_COLUMNS)
    )


def format_captain_regret_markdown(captain: pl.DataFrame) -> str:
    """Render the per-GW captain-regret proxy as Markdown."""
    if captain.is_empty():
        return (
            "# Captain regret proxy\n\n"
            "_No decision-pool rows with actuals "
            "(unconstrained max-xPts vs max-actual)._"
        )
    lines = [
        "# Captain regret proxy",
        "",
        "Unconstrained decision-pool max-`xPts` captain vs max-`actual_pts` "
        "oracle (not the squad-constrained `fplt regret-board`).",
        "",
        "| GW | chosen | oracle | xPts | chosen pts | oracle pts | regret | n |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in captain.sort("gw").iter_rows(named=True):
        lines.append(
            "| {gw} | {chosen} | {oracle} | {xpts:.2f} | {chosen_pts:.0f} | "
            "{oracle_pts:.0f} | {regret:.1f} | {n} |".format(
                gw=row["gw"],
                chosen=row["chosen_player_key"],
                oracle=row["oracle_player_key"],
                xpts=row["chosen_xpts"],
                chosen_pts=row["chosen_actual"],
                oracle_pts=row["oracle_actual"],
                regret=row["regret"],
                n=row["n_pool"],
            )
        )
    return "\n".join(lines)


def format_summary_table(summary: pl.DataFrame) -> str:
    """Pretty-print the measurement summary for the CLI."""
    if summary.is_empty():
        return "(no other-points measurement rows)"
    return str(
        summary.sort(["metric", "slice"]).select(
            "model_name", "metric", "slice", "value", "n"
        )
    )


def _collect_detail(
    pit: PITData,
    store: DataStore,
    as_ofs: list[GWKey],
    spec: ModelSpec,
) -> pl.DataFrame:
    model = get_model(spec.kind)(spec)
    other_spec = spec.components["other"]
    other_model = get_component(other_spec.kind)(other_spec)
    pool_keys = decision_pool_keys(
        store,
        seasons=sorted({as_of.season for as_of in as_ofs}),
    )
    pool_set: set[tuple[str, int, int]] = set()
    if not pool_keys.is_empty():
        pool_set = {
            (str(season), int(gw), int(player))
            for season, gw, player in pool_keys.select(
                "as_of_season", "as_of_gw", "player_key"
            ).iter_rows()
        }
    preds = walk_forward(model, pit, as_ofs, steps=1, run_kind="backtest")
    if preds.is_empty():
        return _empty_detail()
    scored = _attach_actuals(preds, store)
    scored = scored.filter(pl.col("actual_pts").is_not_null())
    if scored.is_empty():
        return _empty_detail()

    rows: list[dict[str, object]] = []
    for index, as_of in enumerate(as_ofs):
        view = _view(pit, as_of, spec)
        if should_refit(spec, index) or index == 0:
            other_model.fit(view)
        part = scored.filter(
            (pl.col("as_of_season") == as_of.season)
            & (pl.col("as_of_gw") == as_of.gw)
            & (pl.col("horizon") == 1)
        )
        if part.is_empty():
            continue
        asked = part.select("player_key", "fixture_id").unique(maintain_order=True)
        other_pred = other_model.predict(view, asked)
        other_lookup = {
            (int(r["player_key"]), int(r["fixture_id"]), str(r["minutes_bucket"])): r
            for r in other_pred.iter_rows(named=True)
        }
        context = _match_context(store, as_of)
        stats = _stats_rows(store, as_of)
        rules = Rules.for_season(as_of.season)
        for row in part.iter_rows(named=True):
            player = int(row["player_key"])
            fixture = int(row["fixture_id"])
            key = (as_of.season, as_of.gw, player)
            stat = stats.get((player, fixture))
            if stat is None:
                continue
            minutes = int(stat["minutes"])
            bucket = minutes_bucket(minutes)
            predicted = other_lookup.get((player, fixture, bucket))
            pred_bonus = float(predicted["bonus"]) if predicted else 0.0
            pred_saves = float(predicted["saves"]) if predicted else 0.0
            pred_cards = float(predicted["cards"]) if predicted else 0.0
            pred_defcon = (
                float(predicted["defensive_contribution"]) if predicted else 0.0
            )
            pred_other_res = float(predicted["other"]) if predicted else 0.0
            pred_other = (
                pred_bonus + pred_saves + pred_cards + pred_defcon + pred_other_res
            )
            act_bonus, act_saves, act_cards, act_defcon, act_other_res = (
                _other_components(rules, str(row["position"]), stat)
            )
            actual_other = (
                act_bonus + act_saves + act_cards + act_defcon + act_other_res
            )
            ctx = context.get(fixture, {"team_goals_home": 0, "team_goals_away": 0})
            side = _player_side(store, as_of, player, fixture)
            team_goals = (
                int(ctx["team_goals_home"])
                if side == "home"
                else int(ctx["team_goals_away"])
            )
            actual_cs = int(stat.get("clean_sheets") or 0)
            actual_goals = int(stat.get("goals_scored") or 0)
            actual_pts = int(row["actual_pts"])
            rows.append(
                {
                    "season": as_of.season,
                    "gw": as_of.gw,
                    "player_key": player,
                    "fixture_id": fixture,
                    "position": str(row["position"]),
                    "minutes": minutes,
                    "minutes_bucket": bucket,
                    "xpts": float(row["xpts"]),
                    "p_haul": (
                        float(row["p_haul"]) if row["p_haul"] is not None else None
                    ),
                    "actual_pts": actual_pts,
                    "hauled": float(actual_pts >= _HAUL_THRESHOLD),
                    "actual_other": float(actual_other),
                    "predicted_other": float(pred_other),
                    "other_residual": float(actual_other - pred_other),
                    "actual_bonus": float(act_bonus),
                    "actual_saves": float(act_saves),
                    "actual_defcon": float(act_defcon),
                    "team_goals": team_goals,
                    "team_goals_band": team_goals_band(team_goals),
                    "clean_sheet": int(actual_cs > 0),
                    "player_scored": int(actual_goals > 0),
                    "in_decision_pool": key in pool_set,
                }
            )
    if not rows:
        return _empty_detail()
    return pl.DataFrame(rows)


def _summarise(
    detail: pl.DataFrame,
    captain: pl.DataFrame,
    *,
    model_name: str,
) -> pl.DataFrame:
    rows: list[dict[str, object]] = []
    empty_summary = pl.DataFrame(schema=OTHER_POINTS_MEASUREMENT_SUMMARY_COLUMNS)
    if detail.is_empty():
        return _validated_summary(empty_summary)

    def add_haul(label: str, part: pl.DataFrame) -> None:
        rows.extend(_haul_metrics(model_name, label, part))

    add_haul("all", detail)
    pool = detail.filter(pl.col("in_decision_pool"))
    add_haul("decision_pool", pool)
    for position in sorted(detail["position"].unique().to_list()):
        add_haul(
            f"position={position}",
            detail.filter(pl.col("position") == position),
        )
    for bucket in ("0", "1-59", "60+"):
        add_haul(
            f"minutes_bucket={bucket}",
            detail.filter(pl.col("minutes_bucket") == bucket),
        )

    top = _top_xpts_decile(detail)
    rows.extend(_haul_metrics(model_name, "top_xpts_decile", top))
    rows.extend(_calibration_metrics(model_name, "top_xpts_decile", top))

    played = detail.filter(pl.col("minutes") > 0)
    rows.extend(_residual_metrics(model_name, "played", played))
    for band in ("0", "1", "2+"):
        rows.extend(
            _residual_metrics(
                model_name,
                f"match|team_goals={band}",
                played.filter(pl.col("team_goals_band") == band),
            )
        )
    for cs in (0, 1):
        rows.extend(
            _residual_metrics(
                model_name,
                f"match|cs={cs}",
                played.filter(pl.col("clean_sheet") == cs),
            )
        )
    for scored in (0, 1):
        rows.extend(
            _residual_metrics(
                model_name,
                f"match|scored={scored}",
                played.filter(pl.col("player_scored") == scored),
            )
        )
    long = played.filter(pl.col("minutes_bucket") == "60+")
    if not long.is_empty():
        corr = pearson_corr(
            long["other_residual"].to_numpy(),
            long["team_goals"].to_numpy().astype(np.float64),
        )
        rows.append(
            _metric_row(
                model_name,
                "other_residual_corr_team_goals",
                "minutes_bucket=60+",
                corr,
                long.height,
            )
        )

    if captain.is_empty():
        rows.append(
            _metric_row(model_name, "captain_regret_mean", "decision_pool", None, 0)
        )
    else:
        rows.append(
            _metric_row(
                model_name,
                "captain_regret_mean",
                "decision_pool",
                float(np.mean(captain["regret"].to_numpy())),
                captain.height,
            )
        )
    return _validated_summary(
        pl.DataFrame(rows, schema=OTHER_POINTS_MEASUREMENT_SUMMARY_COLUMNS)
        if rows
        else pl.DataFrame(schema=OTHER_POINTS_MEASUREMENT_SUMMARY_COLUMNS)
    )


def _haul_metrics(
    model_name: str, slice_name: str, part: pl.DataFrame
) -> list[dict[str, object]]:
    usable = part.filter(pl.col("p_haul").is_not_null())
    if usable.is_empty():
        return [
            _metric_row(model_name, "brier_haul", slice_name, None, 0),
            _metric_row(model_name, "log_haul", slice_name, None, 0),
        ]
    p = usable["p_haul"].to_numpy()
    y = usable["hauled"].to_numpy()
    return [
        _metric_row(
            model_name, "brier_haul", slice_name, brier_score(p, y), usable.height
        ),
        _metric_row(model_name, "log_haul", slice_name, log_score(p, y), usable.height),
    ]


def _calibration_metrics(
    model_name: str, slice_name: str, part: pl.DataFrame
) -> list[dict[str, object]]:
    usable = part.filter(pl.col("p_haul").is_not_null())
    if usable.is_empty():
        return [
            _metric_row(model_name, "haul_mean_p", slice_name, None, 0),
            _metric_row(model_name, "haul_mean_y", slice_name, None, 0),
            _metric_row(model_name, "haul_cal_gap", slice_name, None, 0),
        ]
    mean_p = float(np.mean(usable["p_haul"].to_numpy()))
    mean_y = float(np.mean(usable["hauled"].to_numpy()))
    return [
        _metric_row(model_name, "haul_mean_p", slice_name, mean_p, usable.height),
        _metric_row(model_name, "haul_mean_y", slice_name, mean_y, usable.height),
        _metric_row(
            model_name, "haul_cal_gap", slice_name, mean_p - mean_y, usable.height
        ),
    ]


def _residual_metrics(
    model_name: str, slice_name: str, part: pl.DataFrame
) -> list[dict[str, object]]:
    if part.is_empty():
        return [
            _metric_row(model_name, "other_mae", slice_name, None, 0),
            _metric_row(model_name, "other_mean_actual", slice_name, None, 0),
            _metric_row(model_name, "other_mean_predicted", slice_name, None, 0),
            _metric_row(model_name, "other_mean_residual", slice_name, None, 0),
        ]
    residual = part["other_residual"].to_numpy()
    return [
        _metric_row(
            model_name,
            "other_mae",
            slice_name,
            float(np.mean(np.abs(residual))),
            part.height,
        ),
        _metric_row(
            model_name,
            "other_mean_actual",
            slice_name,
            float(np.mean(part["actual_other"].to_numpy())),
            part.height,
        ),
        _metric_row(
            model_name,
            "other_mean_predicted",
            slice_name,
            float(np.mean(part["predicted_other"].to_numpy())),
            part.height,
        ),
        _metric_row(
            model_name,
            "other_mean_residual",
            slice_name,
            float(np.mean(residual)),
            part.height,
        ),
    ]


def _top_xpts_decile(detail: pl.DataFrame) -> pl.DataFrame:
    if detail.is_empty():
        return detail
    ranked = detail.with_columns(
        pl.col("xpts")
        .rank(method="average", descending=True)
        .over(["season"])
        .alias("_rank")
    )
    n = ranked.height
    cutoff = max(n / 10.0, 1.0)
    return ranked.filter(pl.col("_rank") <= cutoff).drop("_rank")


def _metric_row(
    model_name: str,
    metric: str,
    slice_name: str,
    value: float | None,
    n: int,
) -> dict[str, object]:
    return {
        "model_name": model_name,
        "metric": metric,
        "slice": slice_name,
        "value": value,
        "n": n,
    }


def _attach_actuals(preds: pl.DataFrame, store: DataStore) -> pl.DataFrame:
    stats = (
        store.read("player_fixture_stats")
        .sort("available_at")
        .unique(subset=["season", "gw", "player_key", "fixture_id"], keep="last")
        .select(
            pl.col("season").alias("target_season"),
            pl.col("gw").alias("target_gw"),
            "player_key",
            "fixture_id",
            pl.col("total_points").alias("scored_pts"),
        )
    )
    checked = (
        store.read("gameweeks")
        .filter(pl.col("data_checked"))
        .select(
            pl.col("season").alias("target_season"),
            pl.col("gw").alias("target_gw"),
        )
        .unique()
        .with_columns(pl.lit(True).alias("checked"))
    )
    return (
        preds.join(
            stats,
            on=["target_season", "target_gw", "player_key", "fixture_id"],
            how="left",
        )
        .join(checked, on=["target_season", "target_gw"], how="left")
        .with_columns(
            pl.when(pl.col("checked").fill_null(False))
            .then(pl.col("scored_pts"))
            .otherwise(None)
            .cast(pl.Int64)
            .alias("actual_pts")
        )
        .drop("checked", "scored_pts")
    )


def _stats_rows(
    store: DataStore, as_of: GWKey
) -> dict[tuple[int, int], dict[str, Any]]:
    frame = (
        store.read("player_fixture_stats")
        .filter((pl.col("season") == as_of.season) & (pl.col("gw") == as_of.gw))
        .sort("available_at")
        .unique(subset=["player_key", "fixture_id"], keep="last")
    )
    return {
        (int(row["player_key"]), int(row["fixture_id"])): row
        for row in frame.iter_rows(named=True)
    }


def _match_context(store: DataStore, as_of: GWKey) -> dict[int, dict[str, int]]:
    frame = (
        store.read("fixtures")
        .filter(
            (pl.col("season") == as_of.season)
            & (pl.col("gw") == as_of.gw)
            & pl.col("finished")
        )
        .sort("available_at")
        .unique(subset=["fixture_id"], keep="last")
        .select("fixture_id", "team_h_score", "team_a_score")
    )
    out: dict[int, dict[str, int]] = {}
    for row in frame.iter_rows(named=True):
        home = row["team_h_score"]
        away = row["team_a_score"]
        out[int(row["fixture_id"])] = {
            "team_goals_home": int(home or 0),
            "team_goals_away": int(away or 0),
        }
    return out


def _player_side(
    store: DataStore, as_of: GWKey, player_key: int, fixture_id: int
) -> str:
    players = (
        store.read("players")
        .filter(pl.col("season") == as_of.season)
        .sort("available_at")
        .unique(subset=["player_key"], keep="last")
        .select("player_key", "team_key")
    )
    fixtures = (
        store.read("fixtures")
        .filter(
            (pl.col("season") == as_of.season) & (pl.col("fixture_id") == fixture_id)
        )
        .sort("available_at")
        .unique(subset=["fixture_id"], keep="last")
        .select("fixture_id", "team_h_key", "team_a_key")
    )
    joined = players.filter(pl.col("player_key") == player_key).join(
        fixtures, how="cross"
    )
    if joined.is_empty():
        return "home"
    row = joined.row(0, named=True)
    if int(row["team_key"]) == int(row["team_h_key"]):
        return "home"
    return "away"


def _other_components(
    rules: Rules,
    position: str,
    row: dict[str, Any],
) -> tuple[float, float, float, float, float]:
    def points(**fields: int) -> float:
        return float(rules.score(MatchStats(position=position, **fields)))  # type: ignore[arg-type]

    bonus = points(bonus=int(row.get("bonus") or 0))
    saves = points(saves=int(row.get("saves") or 0))
    cards = points(
        yellow_cards=int(row.get("yellow_cards") or 0),
        red_cards=int(row.get("red_cards") or 0),
    )
    defensive = 0.0
    if rules.defensive_contributions and "defensive_contribution" in row:
        count = int(row.get("defensive_contribution") or 0)
        for rule in rules.defensive_contributions:
            if position in rule.positions and count >= rule.threshold:
                defensive = float(rule.points)
                break
    other = points(
        penalties_saved=int(row.get("penalties_saved") or 0),
        penalties_missed=int(row.get("penalties_missed") or 0),
        own_goals=int(row.get("own_goals") or 0),
    )
    return (bonus, saves, cards, defensive, other)


def _view(pit: PITData, as_of: GWKey, spec: ModelSpec) -> PITView:
    seasons = allowed_seasons(as_of, spec.train_window.seasons_back)
    return pit.view(as_of).within_seasons(seasons)


def _empty_detail() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "season": pl.Utf8(),
            "gw": pl.Int64(),
            "player_key": pl.Int64(),
            "fixture_id": pl.Int64(),
            "position": pl.Utf8(),
            "minutes": pl.Int64(),
            "minutes_bucket": pl.Utf8(),
            "xpts": pl.Float64(),
            "p_haul": pl.Float64(),
            "actual_pts": pl.Int64(),
            "hauled": pl.Float64(),
            "actual_other": pl.Float64(),
            "predicted_other": pl.Float64(),
            "other_residual": pl.Float64(),
            "actual_bonus": pl.Float64(),
            "actual_saves": pl.Float64(),
            "actual_defcon": pl.Float64(),
            "team_goals": pl.Int64(),
            "team_goals_band": pl.Utf8(),
            "clean_sheet": pl.Int64(),
            "player_scored": pl.Int64(),
            "in_decision_pool": pl.Boolean(),
        }
    )


def _validated_summary(frame: pl.DataFrame) -> pl.DataFrame:
    checked = OtherPointsMeasurementSummarySchema.validate(frame)
    checked = require_dataframe(checked)
    return checked


def _validated_captain(frame: pl.DataFrame) -> pl.DataFrame:
    checked = OtherPointsCaptainRegretSchema.validate(frame)
    checked = require_dataframe(checked)
    return checked


__all__ = [
    "OtherPointsMeasurementResult",
    "brier_score",
    "captain_regret_by_gw",
    "format_captain_regret_markdown",
    "format_summary_table",
    "log_score",
    "minutes_bucket",
    "pearson_corr",
    "run_other_points_measurement",
    "team_goals_band",
]
