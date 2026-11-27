"""Scoreline rows on the component ledger.

Ranked probability score compares the home/draw/away forecast with the result.
It is recorded for the leaderboard. Tests check the formula, not which model wins.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
import polars as pl

from fplt.data.pit import PITView
from fplt.data.store import DataStore
from fplt.models.distributions import ScorelineDist
from fplt.models.spec import ModelSpec
from fplt.schemas.ledger import COMPONENT_COLUMNS, ComponentSchema
from fplt.utils.frames import require_dataframe


class _ScoreComponent(Protocol):
    """Minimal scoreline component surface used by the ledger helpers."""

    spec: ModelSpec

    def predict(self, view: PITView, fixtures: pl.DataFrame) -> ScorelineDist:
        """Joint scorelines for ``fixtures``."""
        ...


def component_path(root: Path) -> Path:
    """Parquet path of the component ledger."""
    return root / "ledger" / "components" / "part.parquet"


def empty_components() -> pl.DataFrame:
    """An empty frame that still matches ``ComponentSchema``."""
    return pl.DataFrame(schema=COMPONENT_COLUMNS)


def record_scorelines(
    model: _ScoreComponent,
    view: PITView,
    fixtures: pl.DataFrame,
    run_kind: str = "backtest",
) -> pl.DataFrame:
    """Predict ``fixtures`` and shape the rows for the component ledger.

    Args:
        model: A fitted scoreline model (Dixon-Coles, Poisson, or odds).
        view: Deadline the forecast is made at.
        fixtures: Columns ``season``, ``gw``, ``fixture_id``, ``team_h_key``,
            and ``team_a_key``.
        run_kind: ``backtest`` or ``live``.

    Returns:
        A ``ComponentSchema`` frame. Actuals and ``rps`` are null.
    """
    if fixtures.is_empty():
        return empty_components()
    dist = model.predict(view, fixtures)
    home, draw, away = _outcome_probs(dist)
    goals = np.arange(dist.pmf.shape[1], dtype=np.float64)
    frame = fixtures.select(
        pl.lit(model.spec.model_id).alias("model_id"),
        pl.lit(run_kind).alias("run_kind"),
        pl.lit(view.as_of.season).alias("as_of_season"),
        pl.lit(view.as_of.gw).alias("as_of_gw"),
        pl.col("season").alias("target_season"),
        pl.col("gw").alias("target_gw"),
        "fixture_id",
        "team_h_key",
        "team_a_key",
        pl.Series("p_home", home),
        pl.Series("p_draw", draw),
        pl.Series("p_away", away),
        pl.Series("expected_home", dist.marginal("home") @ goals),
        pl.Series("expected_away", dist.marginal("away") @ goals),
        pl.lit(None).cast(pl.Int64).alias("actual_home"),
        pl.lit(None).cast(pl.Int64).alias("actual_away"),
        pl.lit(None).cast(pl.Float64).alias("rps"),
    )
    checked = ComponentSchema.validate(frame)
    checked = require_dataframe(checked)
    return checked


def append_components(root: Path, frame: pl.DataFrame) -> None:
    """Append scoreline rows to ``ledger/components/part.parquet``."""
    if frame.is_empty():
        return
    checked = ComponentSchema.validate(frame)
    checked = require_dataframe(checked)
    path = component_path(root)
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        checked = pl.concat([pl.read_parquet(path), checked], how="vertical_relaxed")
    checked.write_parquet(path)


def read_components(root: Path) -> pl.DataFrame:
    """Read the component ledger, or an empty typed frame."""
    path = component_path(root)
    if not path.exists():
        return empty_components()
    return pl.read_parquet(path)


def fill_scoreline_actuals(root: Path, store: DataStore) -> None:
    """Fill scores and ranked probability score once the gameweek is checked.

    Unchecked gameweeks stay null even when a score is already stored.
    """
    path = component_path(root)
    if not path.exists():
        return
    ledger = pl.read_parquet(path).drop("actual_home", "actual_away", "rps")
    scores = (
        store.read("fixtures")
        .sort("available_at")
        .unique(subset=["season", "fixture_id"], keep="last")
        .select(
            pl.col("season").alias("target_season"),
            "fixture_id",
            pl.col("team_h_score").alias("scored_home"),
            pl.col("team_a_score").alias("scored_away"),
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
    joined = (
        ledger.join(scores, on=["target_season", "fixture_id"], how="left")
        .join(checked, on=["target_season", "target_gw"], how="left")
        .with_columns(pl.col("checked").fill_null(False))
    )
    scored = (
        joined["checked"]
        & joined["scored_home"].is_not_null()
        & joined["scored_away"].is_not_null()
    ).to_numpy()
    home = joined["scored_home"].cast(pl.Float64).to_numpy()
    away = joined["scored_away"].cast(pl.Float64).to_numpy()
    rps = np.full(joined.height, np.nan)
    if np.any(scored):
        rps[scored] = ranked_probability_score(
            joined["p_home"].to_numpy()[scored],
            joined["p_draw"].to_numpy()[scored],
            joined["p_away"].to_numpy()[scored],
            home[scored].astype(np.float64),
            away[scored].astype(np.float64),
        )
    filled = joined.with_columns(
        pl.when(pl.Series(scored))
        .then(pl.Series(home))
        .otherwise(None)
        .cast(pl.Int64)
        .alias("actual_home"),
        pl.when(pl.Series(scored))
        .then(pl.Series(away))
        .otherwise(None)
        .cast(pl.Int64)
        .alias("actual_away"),
        pl.Series("rps", rps),
    ).with_columns(
        pl.when(pl.col("rps").is_nan()).then(None).otherwise(pl.col("rps")).alias("rps")
    )
    validated = ComponentSchema.validate(filled.select(list(COMPONENT_COLUMNS)))
    validated = require_dataframe(validated)
    validated.write_parquet(path)


def ranked_probability_score(
    p_home: np.ndarray,
    p_draw: np.ndarray,
    p_away: np.ndarray,
    home_goals: np.ndarray,
    away_goals: np.ndarray,
) -> np.ndarray:
    """Ranked probability score of a home/draw/away forecast.

    The order is home, draw, away. The score is the mean squared error of the
    cumulative probabilities, divided by 2 so it lies in ``[0, 1]``.
    """
    cdf = np.cumsum(np.stack([p_home, p_draw, p_away], axis=1), axis=1)
    home_win = home_goals > away_goals
    draw = home_goals == away_goals
    outcome = np.where(home_win, 0, np.where(draw, 1, 2))
    observed = (np.arange(3)[None, :] >= outcome[:, None]).astype(np.float64)
    return np.sum((cdf[:, :2] - observed[:, :2]) ** 2, axis=1) / 2


def _outcome_probs(dist: ScorelineDist) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = dist.pmf.shape[1]
    goals: np.ndarray = np.arange(width, dtype=np.int64)
    home_gt = np.asarray(goals[:, None] > goals[None, :], dtype=np.float64)
    home_eq = np.asarray(goals[:, None] == goals[None, :], dtype=np.float64)
    home_lt = np.asarray(goals[:, None] < goals[None, :], dtype=np.float64)
    return (
        np.asarray((dist.pmf * home_gt).sum(axis=(1, 2)), dtype=np.float64),
        np.asarray((dist.pmf * home_eq).sum(axis=(1, 2)), dtype=np.float64),
        np.asarray((dist.pmf * home_lt).sum(axis=(1, 2)), dtype=np.float64),
    )
