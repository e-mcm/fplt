"""Within-horizon list-price forecasts for the transfer planner.

Step 0 is always the as-of market. Later steps may use a simple
deadline-to-deadline delta estimate from PIT-visible ``player_status``
history. Transfer-volume / threshold models need staged transfer tallies
and are out of scope for v1.

Default planning leaves forecasts off (``plan_squad`` ``price_forecast=None``).
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Literal

import polars as pl

from fplt.data.pit import PITView
from fplt.schemas.plan import PriceForecastSchema

PriceForecastMethod = Literal["flat", "mean_gw_delta"]

_DEFAULT_DELTA_LOOKBACK = 3


def pit_deadline_prices(view: PITView) -> pl.DataFrame:
    """Latest list price at or before each deadline visible in ``view``.

    For every gameweek whose ``deadline_time`` is at or before the view
    cutoff, take the newest ``player_status.price`` with
    ``available_at ≤ deadline_time``. Rows after the cutoff are already
    hidden by ``PITView.table``.

    Args:
        view: Point-in-time view for the planning deadline.

    Returns:
        Columns ``player_key``, ``gw``, ``list_price``, ``deadline_time``.
    """
    season = view.as_of.season
    gameweeks = (
        view.table("gameweeks")
        .filter(pl.col("season") == season)
        .select("gw", "deadline_time")
        .collect()
    )
    status = (
        view.table("player_status")
        .filter(pl.col("season") == season)
        .select("player_key", "price", "available_at")
        .collect()
    )
    if gameweeks.is_empty() or status.is_empty():
        return pl.DataFrame(
            schema={
                "player_key": pl.Int64(),
                "gw": pl.Int64(),
                "list_price": pl.Int64(),
                "deadline_time": pl.Datetime(time_zone="UTC"),
            }
        )
    # Only deadlines that have already occurred (or are the as-of deadline).
    past = gameweeks.filter(pl.col("deadline_time") <= view.cutoff)
    if past.is_empty():
        return pl.DataFrame(
            schema={
                "player_key": pl.Int64(),
                "gw": pl.Int64(),
                "list_price": pl.Int64(),
                "deadline_time": pl.Datetime(time_zone="UTC"),
            }
        )
    joined = status.join(past, how="cross").filter(
        pl.col("available_at") <= pl.col("deadline_time")
    )
    if joined.is_empty():
        return pl.DataFrame(
            schema={
                "player_key": pl.Int64(),
                "gw": pl.Int64(),
                "list_price": pl.Int64(),
                "deadline_time": pl.Datetime(time_zone="UTC"),
            }
        )
    latest = (
        joined.sort("available_at")
        .unique(subset=["player_key", "gw"], keep="last")
        .select(
            "player_key",
            "gw",
            pl.col("price").alias("list_price"),
            "deadline_time",
        )
        .sort(["player_key", "gw"])
    )
    return latest


def forecast_list_prices(
    as_of_prices: Mapping[int, int],
    horizon: int,
    *,
    method: PriceForecastMethod = "flat",
    deadline_prices: pl.DataFrame | None = None,
    delta_lookback: int = _DEFAULT_DELTA_LOOKBACK,
) -> pl.DataFrame:
    """Build ``PriceForecastSchema`` rows for a planning horizon.

    Args:
        as_of_prices: ``player_key`` → as-of list price (step 0).
        horizon: Number of steps including the actionable week.
        method: ``flat`` keeps as-of prices; ``mean_gw_delta`` adds
            ``round(step · mean(last K GW deltas))`` per player.
        deadline_prices: Optional ``player_key``, ``gw``, ``list_price``
            from ``pit_deadline_prices`` (required for a non-zero delta).
        delta_lookback: Max prior GW deltas averaged for ``mean_gw_delta``.

    Returns:
        One row per ``player_key`` × ``step`` matching ``PriceForecastSchema``.

    Raises:
        ValueError: ``horizon < 1``, empty ``as_of_prices``, invalid method,
            or non-positive ``delta_lookback``.
    """
    if horizon < 1:
        raise ValueError("horizon must be at least 1")
    if not as_of_prices:
        raise ValueError("as_of_prices must not be empty")
    if delta_lookback < 1:
        raise ValueError("delta_lookback must be at least 1")
    if method not in ("flat", "mean_gw_delta"):
        raise ValueError(f"unknown price forecast method: {method!r}")

    deltas = (
        {}
        if method == "flat"
        else _mean_gw_deltas(deadline_prices, as_of_prices, delta_lookback)
    )
    rows: list[dict[str, int]] = []
    for key, as_of in sorted(as_of_prices.items()):
        delta = float(deltas.get(int(key), 0.0))
        for step in range(horizon):
            if method == "flat":
                predicted = int(as_of)
            else:
                predicted = _step_price(as_of, delta, step)
            rows.append(
                {
                    "player_key": int(key),
                    "step": step,
                    "list_price": predicted,
                }
            )
    frame = pl.DataFrame(rows)
    checked = PriceForecastSchema.validate(frame)
    assert isinstance(checked, pl.DataFrame)
    return checked


def forecast_list_prices_from_view(
    view: PITView,
    as_of_prices: Mapping[int, int],
    horizon: int,
    *,
    method: PriceForecastMethod = "mean_gw_delta",
    delta_lookback: int = _DEFAULT_DELTA_LOOKBACK,
) -> pl.DataFrame:
    """Forecast list prices using PIT-visible deadline history from ``view``.

    Args:
        view: Planning deadline view (filters ``available_at``).
        as_of_prices: Step-0 list prices (usually from candidates).
        horizon: Planner horizon.
        method: Forecast method (default ``mean_gw_delta``).
        delta_lookback: Lookback for mean GW deltas.

    Returns:
        ``PriceForecastSchema`` frame.
    """
    history = None if method == "flat" else pit_deadline_prices(view)
    return forecast_list_prices(
        as_of_prices,
        horizon,
        method=method,
        deadline_prices=history,
        delta_lookback=delta_lookback,
    )


def _mean_gw_deltas(
    deadline_prices: pl.DataFrame | None,
    as_of_prices: Mapping[int, int],
    lookback: int,
) -> dict[int, float]:
    if deadline_prices is None or deadline_prices.is_empty():
        return {int(key): 0.0 for key in as_of_prices}
    required = {"player_key", "gw", "list_price"}
    missing = required - set(deadline_prices.columns)
    if missing:
        raise ValueError(f"deadline_prices is missing {sorted(missing)}")
    ordered = deadline_prices.sort(["player_key", "gw"])
    with_prev = ordered.with_columns(
        pl.col("list_price").shift(1).over("player_key").alias("_prev"),
        pl.col("gw").shift(1).over("player_key").alias("_prev_gw"),
    )
    deltas = with_prev.filter(
        pl.col("_prev").is_not_null() & (pl.col("gw") == pl.col("_prev_gw") + 1)
    ).with_columns((pl.col("list_price") - pl.col("_prev")).alias("delta"))
    if deltas.is_empty():
        return {int(key): 0.0 for key in as_of_prices}
    # Keep the most recent ``lookback`` deltas per player, then average.
    ranked = deltas.with_columns(
        pl.col("gw")
        .rank(method="ordinal", descending=True)
        .over("player_key")
        .alias("_rank")
    ).filter(pl.col("_rank") <= lookback)
    means = ranked.group_by("player_key").agg(
        pl.col("delta").mean().alias("mean_delta")
    )
    out = {int(key): 0.0 for key in as_of_prices}
    for key, mean in means.iter_rows():
        out[int(key)] = float(mean)
    return out


def _step_price(as_of: int, delta: float, step: int) -> int:
    if step == 0 or delta == 0.0:
        return max(0, int(as_of))
    predicted = int(round(float(as_of) + float(step) * float(delta)))
    return max(0, predicted)
