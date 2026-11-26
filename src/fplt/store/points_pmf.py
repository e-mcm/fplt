"""Official ``points_pmf`` / ``points_pmf_gw`` schemas and binning from draws.

Bins −5…30+; long format omits zero-probability bins (data model §3.3).
Gameweek PMFs sum fixture points per ``(player_key, target_gw, draw_id)``
before binning so doubles are not a convolution of fixture PMFs.
"""

from __future__ import annotations

from datetime import datetime
from typing import Literal

import pandera.polars as pa
import polars as pl

from fplt.schemas.l2 import SCHEMA_VERSION, L2StandardKeys

POINTS_PMF_MIN = -5
POINTS_PMF_MAX = 30


class PointsPmfSchema(L2StandardKeys):
    """P(points = k) at player–fixture grain (official runs only)."""

    points_bin: int = pa.Field(ge=POINTS_PMF_MIN, le=POINTS_PMF_MAX)
    probability: float = pa.Field(ge=0.0, le=1.0)


class PointsPmfGwSchema(pa.DataFrameModel):
    """P(points = k) at player–gameweek grain (official; DGW-correct)."""

    run_id: str
    as_of_season: str
    as_of_gw: int = pa.Field(ge=1, le=38)
    deadline_utc: datetime
    target_season: str
    target_gw: int = pa.Field(ge=1, le=38)
    horizon: int = pa.Field(ge=0)
    player_key: int
    team_key: int
    schema_version: int = pa.Field(ge=1)
    points_bin: int = pa.Field(ge=POINTS_PMF_MIN, le=POINTS_PMF_MAX)
    probability: float = pa.Field(ge=0.0, le=1.0)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True


def _clip(points: pl.Expr) -> pl.Expr:
    return (
        pl.when(points < POINTS_PMF_MIN)
        .then(pl.lit(POINTS_PMF_MIN))
        .when(points > POINTS_PMF_MAX)
        .then(pl.lit(POINTS_PMF_MAX))
        .otherwise(points)
        .cast(pl.Int64)
    )


def points_pmf_from_draws(
    draws: pl.DataFrame,
    *,
    run_id: str,
    as_of_season: str,
    as_of_gw: int,
    deadline_utc: datetime,
    grain: Literal["fixture", "gameweek"] = "fixture",
) -> pl.DataFrame:
    """Bin sim draws into fixture or gameweek PMF tables.

    ``draws`` needs ``player_key``, ``team_key``, ``target_season``,
    ``target_gw``, ``horizon``, ``draw_id``, ``points``. Fixture grain also
    needs ``fixture_id``. Blank GWs simply yield no rows for that target.
    """
    base = {
        "player_key",
        "team_key",
        "target_season",
        "target_gw",
        "horizon",
        "draw_id",
        "points",
    }
    if grain == "fixture":
        base.add("fixture_id")
    missing = base - set(draws.columns)
    if missing:
        raise ValueError(f"draws missing columns: {sorted(missing)}")

    working = draws
    if grain == "gameweek":
        working = draws.group_by(
            [
                "player_key",
                "team_key",
                "target_season",
                "target_gw",
                "horizon",
                "draw_id",
            ]
        ).agg(pl.col("points").sum().alias("points"))

    n_draws = working.select(pl.col("draw_id").n_unique()).item()
    if n_draws < 1:
        raise ValueError("draws need at least one draw_id")

    keys = [
        "player_key",
        "team_key",
        "target_season",
        "target_gw",
        "horizon",
        "points_bin",
    ]
    if grain == "fixture":
        keys.insert(2, "fixture_id")

    counts = (
        working.with_columns(_clip(pl.col("points")).alias("points_bin"))
        .group_by(keys)
        .agg(pl.len().alias("n"))
    )
    frame = counts.with_columns(
        (pl.col("n") / float(n_draws)).alias("probability"),
        pl.lit(run_id).alias("run_id"),
        pl.lit(as_of_season).alias("as_of_season"),
        pl.lit(as_of_gw).alias("as_of_gw"),
        pl.lit(deadline_utc).alias("deadline_utc"),
        pl.lit(SCHEMA_VERSION).alias("schema_version"),
    ).drop("n")
    if grain == "fixture":
        return PointsPmfSchema.validate(frame)
    return PointsPmfGwSchema.validate(frame)
