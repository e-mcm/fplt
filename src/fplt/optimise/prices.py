"""FPL list prices and the 50% sell-on selling-price rule.

Prices are integers in tenths of a million (55 = £5.5m), matching
``player_status.price`` / FPL ``now_cost``.

Decision sims and live plans refresh selling prices from **as-of** market
prices already staged in ``player_status``. Multi-step plan horizons do
**not** forecast future rises or falls: within a horizon, sell credits stay
at the as-of selling price (honest gap vs predicting price changes).
"""

from __future__ import annotations

from collections.abc import Mapping

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
