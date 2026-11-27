"""Forecast skill by lead time (MAE / RMSE vs horizon).

Wave C4 / M9: measure how fast point-forecast skill decays with H=1…N so
chip windows and longer planners know which steps still carry signal.
Reuses ``compare_models`` + ``overall_summary`` (decision pool, RMSE
headline / MAE secondary) rather than inventing a second metric path.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import polars as pl

from fplt.backtest.compare import CompareMode, CompareResult, compare_models
from fplt.backtest.horizons import DEFAULT_PROJECTION_HORIZON, DEFAULT_SKILL_STEPS
from fplt.schemas.horizon_skill import HORIZON_SKILL_COLUMNS, HorizonSkillSchema
from fplt.utils.frames import require_dataframe

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HorizonSkillResult:
    """Skill curve table plus the compare output that produced it."""

    curve: pl.DataFrame
    compare: CompareResult
    output_dir: Path


def skill_curve_from_ledger(
    ledger: pl.DataFrame,
    *,
    model_name: str | None = None,
) -> pl.DataFrame:
    """Build a skill-by-horizon curve from a scored prediction ledger.

    Convenience for measuring RMSE/MAE vs lead time without re-running
    ``compare_models``. Filter to the decision-relevant pool first when
    comparing to ``fplt horizon-skill`` / promote boards.

    Args:
        ledger: Rows with ``horizon``, ``xpts``, ``actual_pts``, and
            ``model_id`` (optional ``model_name``).
        model_name: Optional label written onto every row when the ledger
            lacks ``model_name`` (defaults to ``model_id``).

    Returns:
        Validated ``HorizonSkillSchema`` frame.

    Raises:
        ValueError: Required columns are missing.
    """
    from fplt.backtest.metrics import overall_summary

    if ledger.is_empty():
        return _empty_curve()
    frame = ledger
    if "model_name" not in frame.columns:
        label = model_name
        if label is None:
            if "model_id" not in frame.columns:
                raise ValueError("ledger needs model_name or model_id")
            label = str(frame["model_id"][0])
        frame = frame.with_columns(pl.lit(label).alias("model_name"))
    summary = overall_summary(frame)
    return skill_curve_from_summary(summary)


def skill_curve_from_summary(summary: pl.DataFrame) -> pl.DataFrame:
    """Build a validated MAE/RMSE-by-horizon curve from a compare summary.

    Prefers walk-forward rows when ``mode`` is present. Ratios are relative to
    each model's own horizon-1 RMSE / MAE.

    Args:
        summary: ``overall_summary`` / compare overall frame with ``horizon``,
            ``mae``, ``rmse``, and preferably ``model_name`` / ``model_id``.

    Returns:
        ``HorizonSkillSchema`` rows sorted by model then horizon.
    """
    if summary.is_empty():
        return _empty_curve()
    frame = summary
    if "mode" in frame.columns:
        walk = frame.filter(pl.col("mode") == "walk_forward")
        if not walk.is_empty():
            frame = walk
    needed = {"horizon", "mae", "rmse"}
    missing = needed - set(frame.columns)
    if missing:
        raise ValueError(f"summary is missing {sorted(missing)}")
    if "model_name" not in frame.columns:
        if "model_id" not in frame.columns:
            raise ValueError("summary needs model_name or model_id")
        frame = frame.with_columns(pl.col("model_id").alias("model_name"))
    if "model_id" not in frame.columns:
        frame = frame.with_columns(pl.col("model_name").alias("model_id"))
    if "n" not in frame.columns:
        frame = frame.with_columns(pl.lit(0).cast(pl.Int64).alias("n"))
    slim = (
        frame.select(
            "model_name",
            "model_id",
            pl.col("horizon").cast(pl.Int64),
            pl.col("mae").cast(pl.Float64),
            pl.col("rmse").cast(pl.Float64),
            pl.col("n").cast(pl.Int64),
        )
        .unique(subset=["model_name", "model_id", "horizon"], keep="first")
        .sort("model_name", "horizon")
    )
    bases = slim.filter(pl.col("horizon") == 1).select(
        "model_id",
        pl.col("mae").alias("_mae_h1"),
        pl.col("rmse").alias("_rmse_h1"),
    )
    joined = slim.join(bases, on="model_id", how="left")
    with_ratios = joined.with_columns(
        pl.when(pl.col("_mae_h1").is_not_null() & (pl.col("_mae_h1") != 0.0))
        .then(pl.col("mae") / pl.col("_mae_h1"))
        .otherwise(None)
        .alias("mae_vs_h1"),
        pl.when(pl.col("_rmse_h1").is_not_null() & (pl.col("_rmse_h1") != 0.0))
        .then(pl.col("rmse") / pl.col("_rmse_h1"))
        .otherwise(None)
        .alias("rmse_vs_h1"),
    ).select(list(HORIZON_SKILL_COLUMNS))
    return _validated(with_ratios)


def skill_rows_for_component_eval(curve: pl.DataFrame) -> list[dict[str, object]]:
    """Map a skill curve into ``ComponentEvalSummarySchema`` row dicts.

    Uses ``component="points"``, metrics ``mae`` / ``rmse``, and
    ``slice="horizon=H"``.

    Args:
        curve: Validated ``HorizonSkillSchema`` frame.

    Returns:
        Rows suitable for concatenating into an eval-components summary.
    """
    rows: list[dict[str, object]] = []
    for row in curve.iter_rows(named=True):
        name = str(row["model_name"])
        horizon = int(row["horizon"])
        n = int(row["n"])
        slice_name = f"horizon={horizon}"
        for metric in ("rmse", "mae"):
            value = row[metric]
            rows.append(
                {
                    "component": "points",
                    "metric": metric,
                    "model_name": name,
                    "slice": slice_name,
                    "value": None if value is None else float(value),
                    "n": n,
                }
            )
    return rows


def format_skill_curve(curve: pl.DataFrame) -> str:
    """Pretty-print a skill curve for the CLI."""
    if curve.is_empty():
        return "(no horizon skill rows)"
    return str(
        curve.select(
            "model_name",
            "horizon",
            "rmse",
            "mae",
            "n",
            "rmse_vs_h1",
            "mae_vs_h1",
        ).sort("model_name", "horizon")
    )


def run_horizon_skill(
    data_dir: Path,
    season: str,
    *,
    models: list[str] | None = None,
    from_gw: int = 1,
    to_gw: int = 20,
    steps: int = DEFAULT_SKILL_STEPS,
    seasons_back: int | None = None,
    with_openfpl: bool = False,
    openfpl_model_dir: str | None = None,
    output_dir: Path | None = None,
    use_cache: bool = True,
) -> HorizonSkillResult:
    """Walk-forward compare at ``steps`` and write the skill-by-horizon curve.

    Args:
        data_dir: Root with ``staged/``.
        season: Evaluation season such as ``"2024-25"``.
        models: PointsPredictor YAML names; default is the compare core set.
        from_gw: First as-of gameweek scored.
        to_gw: Last as-of gameweek, inclusive.
        steps: Lead times to score (H=1…``steps``). Default 10.
        seasons_back: Optional train-window override for every arm.
        with_openfpl: Include OpenFPL in the default model set.
        openfpl_model_dir: Optional OpenFPL weights directory.
        output_dir: Where to write ``horizon_skill.*`` (default under
            ``data_dir/horizon_skill/``).
        use_cache: Reuse content-addressed ledgers from compare.

    Returns:
        Validated curve, the underlying ``CompareResult``, and output path.

    Raises:
        ValueError: Invalid window or empty model list (from compare).
    """
    if steps < 1:
        raise ValueError(f"steps must be positive, got {steps}")
    modes: list[CompareMode] = ["walk_forward"]
    out = output_dir or (
        data_dir / "horizon_skill" / f"{season}_gw{from_gw}-{to_gw}_h{steps}"
    )
    out.mkdir(parents=True, exist_ok=True)
    logger.info(
        "horizon_skill season=%s gw=%s-%s steps=%s models=%s out=%s",
        season,
        from_gw,
        to_gw,
        steps,
        models,
        out,
    )
    compare = compare_models(
        data_dir,
        season,
        models=models,
        test_from_gw=from_gw,
        test_to_gw=to_gw,
        modes=modes,
        steps=steps,
        seasons_back=seasons_back,
        with_openfpl=with_openfpl,
        openfpl_model_dir=openfpl_model_dir,
        output_dir=out / "compare",
        use_cache=use_cache,
    )
    curve = skill_curve_from_summary(compare.summary)
    _write_curve(out, curve, season=season, from_gw=from_gw, to_gw=to_gw, steps=steps)
    return HorizonSkillResult(curve=curve, compare=compare, output_dir=out)


def _write_curve(
    out: Path,
    curve: pl.DataFrame,
    *,
    season: str,
    from_gw: int,
    to_gw: int,
    steps: int,
) -> None:
    curve.write_parquet(out / "horizon_skill.parquet")
    payload = {
        "season": season,
        "from_gw": from_gw,
        "to_gw": to_gw,
        "steps": steps,
        "curve": curve.to_dicts(),
    }
    (out / "horizon_skill.json").write_text(json.dumps(payload, indent=2) + "\n")
    logger.info("wrote horizon skill under %s", out)


def _validated(frame: pl.DataFrame) -> pl.DataFrame:
    checked = HorizonSkillSchema.validate(frame)
    checked = require_dataframe(checked)
    return checked


def _empty_curve() -> pl.DataFrame:
    return _validated(pl.DataFrame(schema=HORIZON_SKILL_COLUMNS))


__all__ = [
    "DEFAULT_PROJECTION_HORIZON",
    "DEFAULT_SKILL_STEPS",
    "HorizonSkillResult",
    "format_skill_curve",
    "run_horizon_skill",
    "skill_curve_from_ledger",
    "skill_curve_from_summary",
    "skill_rows_for_component_eval",
]
