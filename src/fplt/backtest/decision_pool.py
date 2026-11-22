"""Decision-relevant player pool for forecast metrics.

Full-squad boards (300+ rows) are dominated by 4.0m fodder zeros. Mean-seeking
compare / promote-eval / bakeoff reports score the subset an optimiser would
actually consider: priced above the floor, with prior minutes or a premium tag.

Filter (documented default):

* as-of ``price >= 45`` (£4.5m in FPL tenths), and
* either ``price >= 70`` (£7.0m premium always kept), or season minutes before
  the as-of gameweek ``>= 90``, or ``as_of_gw <= 2`` (early season, no history).

Price and minutes are PIT: ``player_status.available_at`` and
``player_fixture_stats`` rows must be visible at the as-of deadline.
"""

from __future__ import annotations

import logging
from typing import Final, Literal

import polars as pl

from fplt.data.store import DataStore

logger = logging.getLogger(__name__)

# Headline metric for mean-seeking / expected-points boards (optimiser consumes
# means; squared error is the consistent scoring rule). MAE stays secondary.
HEADLINE_METRIC: Final[Literal["rmse"]] = "rmse"
SECONDARY_POINT_METRICS: Final[tuple[Literal["mae"], ...]] = ("mae",)

DEFAULT_MIN_PRICE: Final = 45
DEFAULT_PREMIUM_PRICE: Final = 70
DEFAULT_MIN_PRIOR_MINUTES: Final = 90
DEFAULT_EARLY_SEASON_GW: Final = 2

POOL_FILTER_DESCRIPTION: Final = (
    "decision-relevant pool: as-of price >= 45 (£4.5m) and "
    "(price >= 70 or prior season minutes >= 90 or as_of_gw <= 2); "
    "PIT via available_at <= deadline"
)


def filter_decision_pool(
    ledger: pl.DataFrame,
    store: DataStore,
    *,
    min_price: int = DEFAULT_MIN_PRICE,
    premium_price: int = DEFAULT_PREMIUM_PRICE,
    min_prior_minutes: int = DEFAULT_MIN_PRIOR_MINUTES,
    early_season_gw: int = DEFAULT_EARLY_SEASON_GW,
) -> pl.DataFrame:
    """Keep ledger rows in the decision-relevant player pool.

    Args:
        ledger: Prediction ledger rows (usually after ``fill_actuals``).
        store: Staged tables with ``gameweeks``, ``player_status``, and
            ``player_fixture_stats``.
        min_price: Floor price in FPL tenths (default 45 = £4.5m).
        premium_price: Price at which players are kept without minutes.
        min_prior_minutes: Season minutes before as-of required for non-premiums.
        early_season_gw: Inclusive as-of GW that skips the minutes gate.

    Returns:
        Ledger rows whose ``(as_of_season, as_of_gw, player_key)`` passes the
        filter. Empty input stays empty. When staged tables cannot build a
        pool (missing deadlines), returns ``ledger`` unchanged and logs a
        warning.
    """
    if ledger.is_empty():
        return ledger
    keys = decision_pool_keys(
        store,
        seasons=sorted(ledger["as_of_season"].unique().to_list()),
        min_price=min_price,
        premium_price=premium_price,
        min_prior_minutes=min_prior_minutes,
        early_season_gw=early_season_gw,
    )
    if keys.is_empty():
        logger.warning(
            "decision pool empty (missing staged price/minutes); scoring full ledger"
        )
        return ledger
    return ledger.join(
        keys,
        on=["as_of_season", "as_of_gw", "player_key"],
        how="inner",
    )


def decision_pool_keys(
    store: DataStore,
    *,
    seasons: list[str],
    min_price: int = DEFAULT_MIN_PRICE,
    premium_price: int = DEFAULT_PREMIUM_PRICE,
    min_prior_minutes: int = DEFAULT_MIN_PRIOR_MINUTES,
    early_season_gw: int = DEFAULT_EARLY_SEASON_GW,
) -> pl.DataFrame:
    """Build ``(as_of_season, as_of_gw, player_key)`` keys in the pool.

    Args:
        store: Staged data root.
        seasons: Seasons to materialise (ledger ``as_of_season`` values).
        min_price: See ``filter_decision_pool``.
        premium_price: See ``filter_decision_pool``.
        min_prior_minutes: See ``filter_decision_pool``.
        early_season_gw: See ``filter_decision_pool``.

    Returns:
        Unique keys for players that pass the filter at each deadline.
    """
    if not seasons:
        return pl.DataFrame(
            schema={
                "as_of_season": pl.Utf8(),
                "as_of_gw": pl.Int64(),
                "player_key": pl.Int64(),
            }
        )
    deadlines = (
        store.read("gameweeks")
        .filter(pl.col("season").is_in(seasons))
        .sort("available_at")
        .unique(subset=["season", "gw"], keep="last")
        .select(
            pl.col("season").alias("as_of_season"),
            pl.col("gw").alias("as_of_gw"),
            pl.col("deadline_time"),
        )
    )
    if deadlines.is_empty():
        return pl.DataFrame(
            schema={
                "as_of_season": pl.Utf8(),
                "as_of_gw": pl.Int64(),
                "player_key": pl.Int64(),
            }
        )

    status = (
        store.read("player_status")
        .filter(pl.col("season").is_in(seasons))
        .select(
            pl.col("season").alias("as_of_season"),
            "player_key",
            "price",
            "available_at",
        )
    )
    # Latest status row visible at each deadline (PIT).
    priced = (
        deadlines.join(status, on="as_of_season", how="inner")
        .filter(pl.col("available_at") <= pl.col("deadline_time"))
        .sort("available_at")
        .unique(
            subset=["as_of_season", "as_of_gw", "player_key"],
            keep="last",
        )
        .select("as_of_season", "as_of_gw", "player_key", "price", "deadline_time")
    )

    stats = (
        store.read("player_fixture_stats")
        .filter(pl.col("season").is_in(seasons))
        .select(
            pl.col("season").alias("as_of_season"),
            pl.col("gw").alias("stat_gw"),
            "player_key",
            "minutes",
            "available_at",
        )
    )
    # Minutes from completed fixtures before the as-of GW, visible at deadline.
    prior = (
        priced.select("as_of_season", "as_of_gw", "player_key", "deadline_time")
        .join(stats, on=["as_of_season", "player_key"], how="left")
        .filter(
            pl.col("stat_gw").is_not_null()
            & (pl.col("stat_gw") < pl.col("as_of_gw"))
            & (pl.col("available_at") <= pl.col("deadline_time"))
        )
        .group_by(["as_of_season", "as_of_gw", "player_key"])
        .agg(pl.col("minutes").sum().alias("prior_minutes"))
    )
    joined = priced.join(
        prior, on=["as_of_season", "as_of_gw", "player_key"], how="left"
    ).with_columns(pl.col("prior_minutes").fill_null(0))

    kept = joined.filter(
        (pl.col("price") >= min_price)
        & (
            (pl.col("price") >= premium_price)
            | (pl.col("prior_minutes") >= min_prior_minutes)
            | (pl.col("as_of_gw") <= early_season_gw)
        )
    ).select("as_of_season", "as_of_gw", "player_key")
    return kept.unique()


__all__ = [
    "DEFAULT_EARLY_SEASON_GW",
    "DEFAULT_MIN_PRICE",
    "DEFAULT_MIN_PRIOR_MINUTES",
    "DEFAULT_PREMIUM_PRICE",
    "HEADLINE_METRIC",
    "POOL_FILTER_DESCRIPTION",
    "SECONDARY_POINT_METRICS",
    "decision_pool_keys",
    "filter_decision_pool",
]
