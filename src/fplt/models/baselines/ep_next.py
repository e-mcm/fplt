"""FPL ``ep_next`` copied from the latest live snapshot."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from fplt.data.pit import PITView
from fplt.models.predictions import prediction_frame
from fplt.models.spec import ModelSpec, register_model, write_identity
from fplt.models.universe import player_fixtures
from fplt.types import GWKey


@register_model("ep_next")
class EpNextPredictor:
    """Use the API's ``ep_next`` as expected points.

    Vaastav seasons have no per-deadline ``ep_next``. This predictor emits a
    row only when both the fixture and the player's status came from ``fpl_api``.
    """

    def __init__(self, spec: ModelSpec) -> None:
        self.spec = spec
        self._ep = _empty_ep()

    def fit(self, view: PITView) -> None:
        """Read ``ep_next`` from the latest ``fpl_api`` status rows."""
        status = (
            view.latest("player_status", by=["season", "player_key"])
            .filter(pl.col("source") == "fpl_api")
            .select("player_key", pl.col("ep_next").fill_null(0.0).alias("xpts"))
            .collect()
        )
        self._ep = status if not status.is_empty() else _empty_ep()

    def predict(self, view: PITView, horizon: list[GWKey]) -> pl.DataFrame:
        """Predict snapshot-backed fixtures. Returns ``PredictionSchema``."""
        rows = player_fixtures(view, horizon).filter(pl.col("source") == "fpl_api")
        if self._ep.is_empty():
            rows = rows.head(0)
        else:
            rows = rows.join(self._ep, on="player_key", how="inner")
        xpts = pl.col("xpts") if not self._ep.is_empty() else pl.lit(0.0)
        return prediction_frame(
            self.spec,
            view.as_of,
            rows,
            xpts,
            pl.lit(None),
            pl.lit(None),
            pl.lit(None),
        )

    def save(self, path: Path) -> None:
        """Write the fitted ``ep_next`` values. Refuses to overwrite."""
        write_identity(path, self.spec)
        self._ep.write_parquet(path / "state.parquet")

    @classmethod
    def load(cls, path: Path, spec: ModelSpec) -> EpNextPredictor:
        """Load values written by ``save``."""
        model = cls(spec)
        model._ep = pl.read_parquet(path / "state.parquet")
        return model


def _empty_ep() -> pl.DataFrame:
    return pl.DataFrame(schema={"player_key": pl.Int64(), "xpts": pl.Float64()})
