"""Official ``points_pmf`` / ``points_pmf_gw`` schemas and binning from draws.

Bins −5…30+; long format omits zero-probability bins (data model §3.3).
Gameweek PMFs sum fixture points per ``(player_key, target_gw, draw_id)``
before binning so doubles are not a convolution of fixture PMFs.

Official slate builders sample and bin **one gameweek at a time** so a
horizon-9 predict never materialises ~36M long draw rows at once (#372).
"""

from __future__ import annotations

import logging
import time
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Literal

import numpy as np
import pandera.polars as pa
import polars as pl

from fplt.schemas.l2 import SCHEMA_VERSION, L2StandardKeys
from fplt.types import GWKey

logger = logging.getLogger(__name__)

POINTS_PMF_MIN = -5
POINTS_PMF_MAX = 30

# Synthetic horizon-9 memory bench defaults (#372). Sized to stress the
# streaming path without OOMing a 16 GB runner when chunked per GW.
_BENCH_N_GWS = 9
_BENCH_FIXTURES_PER_GW = 10
_BENCH_PLAYERS = 40
_BENCH_N_SIMS = 2_000


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


def narrow_pmf_draws(draws: pl.DataFrame) -> pl.DataFrame:
    """Cast draw columns to compact dtypes before binning.

    ``points`` becomes Int16 (truncates toward zero, matching the previous
    float→int cast in ``_clip``). Keys and ``draw_id`` become Int32.

    Args:
        draws: Long draw frame with at least the PMF key columns.

    Returns:
        A frame with narrowed dtypes (same column names).
    """
    casts: list[pl.Expr] = []
    for name, dtype in (
        ("player_key", pl.Int32),
        ("team_key", pl.Int32),
        ("fixture_id", pl.Int32),
        ("draw_id", pl.Int32),
        ("horizon", pl.Int32),
        ("target_gw", pl.Int32),
        ("points", pl.Int16),
    ):
        if name in draws.columns:
            casts.append(pl.col(name).cast(dtype))
    if not casts:
        return draws
    return draws.with_columns(casts)


def prepare_pmf_draws(
    samples: pl.DataFrame,
    players: pl.DataFrame,
    *,
    as_of: GWKey,
) -> pl.DataFrame:
    """Join ``team_key``, rename ``sim`` → ``draw_id``, add horizon, narrow.

    Args:
        samples: ``PredictionSampleSchema`` rows (``sim``, ``points``, …).
        players: ``player_key`` / ``team_key`` lookup (one row per player).
        as_of: Deadline used to compute ``horizon = target_gw - as_of.gw``.

    Returns:
        Narrowed draws ready for ``points_pmf_from_draws``.

    Raises:
        ValueError: A sample row has no matching ``team_key``.
    """
    lookup = players.select("player_key", "team_key").unique(
        subset=["player_key"], keep="last"
    )
    # Select only columns needed for the join + binning so the wide sample
    # frame (minutes, etc.) is not carried into the 10k-draw join.
    slim = samples.select(
        "player_key",
        "fixture_id",
        "target_season",
        "target_gw",
        "sim",
        "points",
    )
    draws = (
        slim.join(lookup, on="player_key", how="left")
        .with_columns(
            pl.col("sim").alias("draw_id"),
            (pl.col("target_gw") - as_of.gw).cast(pl.Int64).alias("horizon"),
        )
        .select(
            "player_key",
            "team_key",
            "fixture_id",
            "target_season",
            "target_gw",
            "horizon",
            "draw_id",
            "points",
        )
    )
    if draws["team_key"].null_count() > 0:
        raise ValueError("points_pmf draws missing team_key after player join")
    return narrow_pmf_draws(draws)


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


def points_pmf_tables_from_sample(
    sample: Callable[[Sequence[GWKey]], pl.DataFrame],
    *,
    horizon_keys: Sequence[GWKey],
    players: pl.DataFrame,
    run_id: str,
    as_of: GWKey,
    deadline_utc: datetime,
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Sample and bin PMFs one gameweek at a time (DGW-safe).

    Never materialises draws for the full horizon. Each call to ``sample``
    receives a single-GW key list so DGW fixtures share ``sim`` / ``draw_id``
    within that GW; draws are dropped before the next GW is sampled.

    Args:
        sample: ``(keys) -> PredictionSampleSchema`` callable (typically a
            bound ``model.sample(view, keys, n_sims)``).
        horizon_keys: Gameweeks to cover (usually ``horizon_for(as_of, h)``).
        players: ``player_key`` / ``team_key`` lookup.
        run_id: Run id stamped onto both PMF tables.
        as_of: Predict deadline.
        deadline_utc: That deadline's UTC timestamp.

    Returns:
        ``(points_pmf, points_pmf_gw)`` frames.

    Raises:
        ValueError: Every GW returned empty samples.
    """
    fx_parts: list[pl.DataFrame] = []
    gw_parts: list[pl.DataFrame] = []
    for key in horizon_keys:
        samples = sample([key])
        if samples.is_empty():
            continue
        draws = prepare_pmf_draws(samples, players, as_of=as_of)
        del samples
        fx_parts.append(
            points_pmf_from_draws(
                draws,
                run_id=run_id,
                as_of_season=as_of.season,
                as_of_gw=as_of.gw,
                deadline_utc=deadline_utc,
                grain="fixture",
            )
        )
        gw_parts.append(
            points_pmf_from_draws(
                draws,
                run_id=run_id,
                as_of_season=as_of.season,
                as_of_gw=as_of.gw,
                deadline_utc=deadline_utc,
                grain="gameweek",
            )
        )
        del draws
    if not fx_parts:
        raise ValueError(
            f"official predict sample() returned no rows for {as_of} "
            f"horizon={len(horizon_keys)}; cannot write points_pmf"
        )
    return pl.concat(fx_parts), pl.concat(gw_parts)


def run_points_pmf_memory_bench(
    *,
    n_gws: int = _BENCH_N_GWS,
    fixtures_per_gw: int = _BENCH_FIXTURES_PER_GW,
    n_players: int = _BENCH_PLAYERS,
    n_sims: int = _BENCH_N_SIMS,
) -> dict[str, float]:
    """Synthetic horizon-N streaming PMF wall-clock bench (#372).

    Builds fake ``sample()`` draws per GW (no model / network) and times
    ``points_pmf_tables_from_sample``. Peak RSS for the phase is recorded
    via ``timed_phase("points_pmf")`` when a timing session is active.

    Args:
        n_gws: Horizon length (default 9).
        fixtures_per_gw: Fixtures synthesised per GW.
        n_players: Players per fixture.
        n_sims: Draws per player-fixture.

    Returns:
        ``{"points_pmf": seconds}`` for nightly bench tracked phases.
    """
    from fplt.utils.timing import timed_phase

    as_of = GWKey("2026-27", 1)
    keys = [GWKey(as_of.season, as_of.gw + offset) for offset in range(n_gws)]
    players = pl.DataFrame(
        {
            "player_key": list(range(n_players)),
            "team_key": [i % 20 for i in range(n_players)],
        }
    )
    deadline = datetime(2026, 8, 1, 11, 0, 0)

    def _sample(chunk: Sequence[GWKey]) -> pl.DataFrame:
        key = chunk[0]
        base_fx = (key.gw - as_of.gw) * fixtures_per_gw
        n_rows = fixtures_per_gw * n_players * n_sims
        fixture_ids = np.repeat(
            np.arange(base_fx, base_fx + fixtures_per_gw, dtype=np.int64),
            n_players * n_sims,
        )
        player_keys = np.tile(
            np.repeat(np.arange(n_players, dtype=np.int64), n_sims),
            fixtures_per_gw,
        )
        sims = np.tile(np.arange(n_sims, dtype=np.int64), fixtures_per_gw * n_players)
        points = (player_keys + sims + (fixture_ids - base_fx)) % 15
        return pl.DataFrame(
            {
                "player_key": player_keys,
                "fixture_id": fixture_ids,
                "target_season": np.full(n_rows, key.season),
                "target_gw": np.full(n_rows, key.gw, dtype=np.int64),
                "sim": sims,
                "points": points.astype(np.float64),
                "minutes": np.full(n_rows, 90, dtype=np.int64),
            }
        )

    with timed_phase("points_pmf"):
        started = time.perf_counter()
        fx, gw = points_pmf_tables_from_sample(
            _sample,
            horizon_keys=keys,
            players=players,
            run_id="bench_points_pmf",
            as_of=as_of,
            deadline_utc=deadline,
        )
        elapsed = time.perf_counter() - started
    logger.info(
        "points_pmf bench horizon=%s fixtures/gw=%s players=%s n_sims=%s "
        "fx_rows=%s gw_rows=%s wall=%.3fs",
        n_gws,
        fixtures_per_gw,
        n_players,
        n_sims,
        fx.height,
        gw.height,
        elapsed,
    )
    return {"points_pmf": float(elapsed)}
