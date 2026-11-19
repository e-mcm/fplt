"""Rolling mean of recently observed fantasy points."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from fplt.data.pit import PITView
from fplt.models.predictions import prediction_frame
from fplt.models.spec import ModelSpec, register_model, write_identity
from fplt.models.universe import player_fixtures
from fplt.types import GWKey


@register_model("rolling_mean")
class RollingMeanPredictor:
    """Mean of each player's last ``window`` visible appearances.

    Players with no visible appearances get zero. ``p_60`` and ``p_haul`` are
    the share of that window with at least 60 minutes and at least 10 points.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._summary = _empty_summary()

    def fit(self, view: PITView) -> None:
        """Estimate means from stats visible at ``view.as_of``."""
        window = int(self.spec.params.get("window", 5))
        stats = view.table("player_fixture_stats").collect()
        if stats.is_empty():
            self._summary = _empty_summary()
            return
        ranked = stats.sort("available_at", descending=True)
        recent = ranked.group_by("player_key").head(window)
        self._summary = recent.group_by("player_key").agg(
            pl.col("total_points").mean().alias("xpts"),
            pl.col("total_points").std(ddof=0).fill_null(0.0).alias("xpts_sd"),
            (pl.col("total_points") >= 10).mean().alias("p_haul"),
            (pl.col("minutes") >= 60).mean().alias("p_60"),
        )

    def predict(self, view: PITView, horizon: list[GWKey]) -> pl.DataFrame:
        """Predict every player-fixture in ``horizon``. Returns ``PredictionSchema``."""
        rows = player_fixtures(view, horizon)
        if rows.is_empty():
            zero = pl.lit(0.0)
            return prediction_frame(self.spec, view.as_of, rows, zero, zero, zero, zero)
        joined = rows.join(self._summary, on="player_key", how="left")
        return prediction_frame(
            self.spec,
            view.as_of,
            joined,
            pl.col("xpts").fill_null(0.0),
            pl.col("xpts_sd").fill_null(0.0),
            pl.col("p_haul").fill_null(0.0),
            pl.col("p_60").fill_null(0.0),
        )

    def save(self, path: Path) -> None:
        """Write the fitted means. Refuses to overwrite an existing artifact."""
        write_identity(path, self.spec)
        self._summary.write_parquet(path / "state.parquet")

    @classmethod
    def load(cls, path: Path, spec: ModelSpec) -> RollingMeanPredictor:
        """Load means written by ``save``."""
        model = cls(spec)
        model._summary = pl.read_parquet(path / "state.parquet")
        return model


def _empty_summary() -> pl.DataFrame:
    return pl.DataFrame(
        schema={
            "player_key": pl.Int64(),
            "xpts": pl.Float64(),
            "xpts_sd": pl.Float64(),
            "p_haul": pl.Float64(),
            "p_60": pl.Float64(),
        }
    )
