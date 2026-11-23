"""FPL list prices and the 50% sell-on selling-price rule.

Prices are integers in tenths of a million (55 = £5.5m), matching
``player_status.price`` / FPL ``now_cost``.

Decision sims and live plans refresh selling prices from **as-of** market
prices already staged in ``player_status``. Multi-step plan horizons default
to those as-of credits; optional ``price_forecast`` rows (see
``fplt.optimise.price_forecast``) can vary list / sell prices by step.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import polars as pl

from fplt.optimise.state import OwnedPlayer, SquadState


def fpl_selling_price(purchase_price: int, current_price: int) -> int:
    """Compute FPL selling price in tenths of a million.

    Keep half of any rise (rounded down to the nearest 0.1) and take the
    full loss on any fall.

    Args:
        purchase_price: Price paid when the player joined the squad.
        current_price: Current list price (``now_cost`` / staged ``price``).

    Returns:
        Selling price credited if the player is transferred out now.

    Raises:
        ValueError: Either price is negative.
    """
    if purchase_price < 0 or current_price < 0:
        raise ValueError("prices must be >= 0")
    if current_price >= purchase_price:
        return purchase_price + (current_price - purchase_price) // 2
    return current_price


def refresh_owned_selling_prices(
    owned: list[OwnedPlayer],
    market_price_by_key: Mapping[int, int],
) -> list[OwnedPlayer]:
    """Recompute each owned selling price from purchase and market prices.

    Players missing from ``market_price_by_key`` keep their current selling
    price unchanged.

    Args:
        owned: Squad membership with sticky ``purchase_price``.
        market_price_by_key: ``player_key`` → as-of list price.

    Returns:
        A new owned list with updated ``selling_price`` values.
    """
    refreshed: list[OwnedPlayer] = []
    for player in owned:
        current = market_price_by_key.get(player.player_key)
        if current is None:
            refreshed.append(player)
            continue
        refreshed.append(
            OwnedPlayer(
                player_key=player.player_key,
                purchase_price=player.purchase_price,
                selling_price=fpl_selling_price(player.purchase_price, current),
            )
        )
    return refreshed


def apply_market_prices(
    state: SquadState,
    market_price_by_key: Mapping[int, int],
) -> SquadState:
    """Return a copy of ``state`` with owned selling prices refreshed.

    Args:
        state: Squad bank / FT / owned players.
        market_price_by_key: ``player_key`` → as-of list price from candidates
            or ``player_status``.

    Returns:
        ``state`` unchanged when nothing is owned; otherwise a copy whose
        ``owned`` selling prices follow ``fpl_selling_price``.
    """
    if not state.owned:
        return state
    return state.model_copy(
        update={"owned": refresh_owned_selling_prices(state.owned, market_price_by_key)}
    )


def market_prices_from_candidates(candidates: pl.DataFrame) -> dict[int, int]:
    """Build ``player_key`` → list price from a candidate frame.

    Args:
        candidates: Polars frame with ``player_key`` and ``price`` columns
            (``CandidateSchema``).

    Returns:
        Mapping used by ``apply_market_prices``.
    """
    return {
        int(key): int(price)
        for key, price in candidates.select("player_key", "price").iter_rows()
    }


def list_price_matrix(
    keys: Sequence[int],
    horizon: int,
    as_of_prices: Mapping[int, int],
    price_forecast: pl.DataFrame | None = None,
) -> np.ndarray:
    """Build ``(horizon, n_players)`` list prices for the MILP cash rows.

    When ``price_forecast`` is ``None``, every step copies the as-of price.
    Otherwise each ``player_key`` × ``step`` must supply ``list_price``, and
    step 0 must match ``as_of_prices``.

    Args:
        keys: Pool ``player_key`` order (column index in the matrix).
        horizon: Number of plan steps.
        as_of_prices: As-of list price per ``player_key``.
        price_forecast: Optional ``PriceForecastSchema`` frame.

    Returns:
        Float matrix of list prices shaped ``(horizon, len(keys))``.

    Raises:
        ValueError: Forecast missing a cell, or step-0 disagrees with as-of.
    """
    count = len(keys)
    matrix = np.zeros((horizon, count), dtype=np.float64)
    if price_forecast is None:
        for index, key in enumerate(keys):
            price = float(as_of_prices[int(key)])
            matrix[:, index] = price
        return matrix

    required = {"player_key", "step", "list_price"}
    missing_cols = required - set(price_forecast.columns)
    if missing_cols:
        raise ValueError(f"price_forecast is missing {sorted(missing_cols)}")
    by_key_step = {
        (int(player_key), int(step)): int(list_price)
        for player_key, step, list_price in price_forecast.select(
            "player_key", "step", "list_price"
        ).iter_rows()
    }
    for index, key in enumerate(keys):
        as_of = int(as_of_prices[int(key)])
        for step in range(horizon):
            predicted = by_key_step.get((int(key), step))
            if predicted is None:
                raise ValueError(f"price_forecast missing player_key={key} step={step}")
            if step == 0 and predicted != as_of:
                raise ValueError(
                    f"price_forecast step 0 for player_key={key} is "
                    f"{predicted}, as-of list price is {as_of}"
                )
            matrix[step, index] = float(predicted)
    return matrix


def sell_price_matrix(
    list_prices: np.ndarray,
    owned: Sequence[OwnedPlayer],
    key_index: Mapping[int, int],
) -> np.ndarray:
    """Derive sell credits from list prices and sticky purchase prices.

    Non-owned players sell at the step's list price. Owned players use
    ``fpl_selling_price(purchase, list[step])``.

    Args:
        list_prices: ``(horizon, n_players)`` list-price matrix.
        owned: Current squad with sticky ``purchase_price``.
        key_index: ``player_key`` → column index in ``list_prices``.

    Returns:
        Float matrix the same shape as ``list_prices``.
    """
    sell = np.array(list_prices, dtype=np.float64, copy=True)
    for player in owned:
        at = key_index[player.player_key]
        for step in range(sell.shape[0]):
            sell[step, at] = float(
                fpl_selling_price(player.purchase_price, int(list_prices[step, at]))
            )
    return sell
