"""Canonical N1 schema: ``benchmark_projections`` (deadline benchmark archive).

One table for every external/internal projection arm. FDB-118 (FPL Review
manual ingest) must append ``source=fpl_review`` rows here — do not invent a
second schema or on-disk layout.
"""

from __future__ import annotations

from datetime import datetime

import pandera.polars as pa
import polars as pl


class BenchmarkProjectionSchema(pa.DataFrameModel):
    """One player–fixture row in ``benchmark_projections``.

    ``source`` is ``ep_next``, ``openfpl``, or ``fpl_review``. Review rows are
    reserved for a later PR after a terms-of-use check; this schema already
    accepts them so paired Tier-B boards do not need a migration.
    """

    source: str = pa.Field(isin=["ep_next", "openfpl", "fpl_review"])
    as_of_season: str
    as_of_gw: int = pa.Field(ge=1, le=38)
    target_season: str
    target_gw: int = pa.Field(ge=1, le=38)
    player_key: int
    fixture_id: int
    xpts: float = pa.Field(ge=-10, le=40)
    available_at: datetime
    quality_flag: str = pa.Field(nullable=True)

    class Config(pa.DataFrameModel.Config):
        strict = True
        coerce = True
        unique = [  # noqa: RUF012
            "source",
            "as_of_season",
            "as_of_gw",
            "player_key",
            "fixture_id",
        ]  # noqa: RUF012


BENCHMARK_COLUMNS: dict[str, pl.DataType] = {
    "source": pl.Utf8(),
    "as_of_season": pl.Utf8(),
    "as_of_gw": pl.Int64(),
    "target_season": pl.Utf8(),
    "target_gw": pl.Int64(),
    "player_key": pl.Int64(),
    "fixture_id": pl.Int64(),
    "xpts": pl.Float64(),
    "available_at": pl.Datetime(time_zone="UTC"),
    "quality_flag": pl.Utf8(),
}

SOURCE_EP_NEXT = "ep_next"
SOURCE_OPENFPL = "openfpl"
SOURCE_FPL_REVIEW = "fpl_review"
QUALITY_OPENFPL_DEGRADED = "openfpl_degraded"
