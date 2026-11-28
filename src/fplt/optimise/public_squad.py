"""Derive planner squad state from public FPL endpoints + PIT prices.

Planning does not need ``FPLT_FPL_API_TOKEN``. Authenticated ``my-team`` is
preferred when it works; on a missing or expired token (401/403) this module
rebuilds owned players, bank, selling prices, free transfers, and chip history
from public ``entry`` / ``history`` / ``transfers`` / finished-GW ``picks``
plus staged ``player_status`` prices. Writes (``fplt apply``) still need a
valid Bearer token.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Literal, cast

import httpx
import polars as pl

from fplt.data.fpl_client import (
    CurrentTeam,
    EntryPick,
    FPLAPIError,
    FPLClient,
)
from fplt.data.pit import PITView
from fplt.optimise.prices import fpl_selling_price
from fplt.optimise.state import ChipInventory, ChipPlay, SquadState
from fplt.scoring.chips import ChipName, ChipRules, empty_remaining
from fplt.settings import Settings, get_settings, secret_value

logger = logging.getLogger(__name__)

SquadSource = Literal["yaml", "entry_json", "my_team", "public"]

_CHIP_NAMES: frozenset[str] = frozenset({"wildcard", "freehit", "bboost", "3xc"})


@dataclass(frozen=True)
class PublicSquadState:
    """Full planner input derived without an FPL Bearer token.

    ``team`` carries the fifteen, bank, selling/purchase prices, and free
    transfers (``source="public"``). ``chips`` is the season inventory implied
    by ``history.chips`` and ``ChipRules`` for the view's season.
    """

    entry_id: int
    team: CurrentTeam
    chips: ChipInventory
    last_finished: int


@dataclass(frozen=True)
class ResolvedSquad:
    """Squad state plus where owned/bank/FT came from."""

    state: SquadState
    source: SquadSource


def public_squad_state(
    entry_id: int,
    view: PITView,
    settings: Settings | None = None,
    *,
    client: httpx.Client | None = None,
    api: FPLClient | None = None,
) -> PublicSquadState:
    """Build planner squad input from public FPL endpoints and PIT prices.

    Args:
        entry_id: FPL manager entry id.
        view: Point-in-time view for the planning deadline (prices + element
            map). Purchase prices for never-transferred-in players use
            ``player_status.price`` as of the entry's first GW deadline.
        settings: Process settings (token ignored; used for HTTP client setup
            when ``api`` is omitted). Defaults to ``get_settings()``.
        client: Optional shared ``httpx.Client`` (tests).
        api: Optional pre-built ``FPLClient`` (tests).

    Returns:
        ``PublicSquadState`` with selling prices, free transfers, and chips.

    Raises:
        FPLAPIError: A required public endpoint fails or the payload is
            incomplete.
        KeyError: A pick/transfer element has no ``player_key`` / price in
            ``view``.
        ValueError: Chip or FT derivation is inconsistent with configs.
    """
    cfg = settings if settings is not None else get_settings()
    owns_api = api is None
    fpl = api or FPLClient(
        client,
        session=secret_value(cfg.fpl_session),
        api_token=None,  # public path never sends a Bearer token
    )
    try:
        summary = fpl.entry(entry_id)
        history = fpl.entry_history(entry_id)
        transfers = fpl.entry_transfers(entry_id)
        started = int(summary.get("started_event") or 1)
        current_rows = _history_current(history)
        if not current_rows:
            raise FPLAPIError(f"entry {entry_id} history has no current[] rows")
        last_finished = max(int(row["event"]) for row in current_rows)
        picks_payload = fpl.entry_picks(entry_id, last_finished)
        season = view.as_of.season
        rules = ChipRules.for_season(season)
        ft_cap = _ft_bank_cap()
        free = replay_free_transfers(
            current_rows,
            _history_chips(history),
            ft_bank_cap=ft_cap,
            rules=rules,
            started_event=started,
        )
        pending = [row for row in transfers if int(row["event"]) > last_finished]
        owned_elements, bank = _owned_after_pending(picks_payload, pending)
        if pending:
            free = _ft_after_pending(free, len(pending), ft_cap)
        element_to_key = _element_keys(view)
        purchase_by_element = _purchase_prices(
            owned_elements=list(owned_elements),
            transfers=transfers,
            view=view,
            started_event=started,
            element_to_key=element_to_key,
        )
        now_by_element = _now_prices(view, element_to_key)
        picks = _priced_picks(
            owned_elements,
            purchase_by_element=purchase_by_element,
            now_by_element=now_by_element,
        )
        team = CurrentTeam(
            entry_id=entry_id,
            event=last_finished,
            bank=bank,
            value=_optional_int(
                cast(dict[str, Any], picks_payload.get("entry_history") or {}).get(
                    "value"
                )
            ),
            free_transfers=free,
            picks=picks,
            active_chip=None,
            source="public",
        )
        chips = chips_from_history(history, rules)
        logger.info(
            "public_squad_state entry_id=%s last_finished=%s owned=%s bank=%s "
            "free_transfers=%s chips_played=%s source=public",
            entry_id,
            last_finished,
            len(picks),
            bank,
            free,
            len(chips.played),
        )
        return PublicSquadState(
            entry_id=entry_id,
            team=team,
            chips=chips,
            last_finished=last_finished,
        )
    finally:
        if owns_api:
            fpl.close()


def replay_free_transfers(
    current: list[dict[str, Any]],
    chips: list[dict[str, Any]],
    *,
    ft_bank_cap: int,
    rules: ChipRules,
    started_event: int = 1,
) -> int:
    """Replay finished-GW history into free transfers for the next deadline.

    Starts at 1 FT on the entry's first deadline, then for each finished
    gameweek applies wildcard / free-hit semantics from ``rules`` and the
    banking rule in ``next_free_transfers`` (cap from ``ft_bank_cap`` /
    ``configs/squad_state.yaml``).

    Args:
        current: ``entry/{id}/history/`` ``current[]`` rows (finished GWs).
        chips: ``entry/{id}/history/`` ``chips[]`` rows with ``name`` + ``event``.
        ft_bank_cap: Maximum banked free transfers.
        rules: Season ``ChipRules`` (``free_hit_keeps_free_transfers``, …).
        started_event: Entry's first gameweek (``started_event``).

    Returns:
        Free transfers available at the next (upcoming) deadline.

    Raises:
        ValueError: ``ft_bank_cap`` is less than 1.
    """
    if ft_bank_cap < 1:
        raise ValueError("ft_bank_cap must be >= 1")
    by_event = {
        int(row["event"]): str(row["name"])
        for row in chips
        if "event" in row and "name" in row
    }
    free = 1
    for row in sorted(current, key=lambda item: int(item["event"])):
        event = int(row["event"])
        if event < started_event:
            continue
        moves = int(row.get("event_transfers") or 0)
        chip = by_event.get(event)
        free = _next_free_after_chip(
            free,
            chip=chip,
            moves=moves,
            cap=ft_bank_cap,
            rules=rules,
        )
    return free


def chips_from_history(
    history: dict[str, Any],
    rules: ChipRules,
) -> ChipInventory:
    """Build chip inventory from public ``history.chips`` and season grants.

    Args:
        history: ``entry/{id}/history/`` body.
        rules: Season chip allotments.

    Returns:
        Remaining counts (grants minus played) and ``played`` history.

    Raises:
        ValueError: A chip name is unknown or remaining would go negative.
    """
    remaining: dict[ChipName, int] = dict(rules.remaining_counts())
    for name in empty_remaining():
        remaining.setdefault(name, 0)
    played: list[ChipPlay] = []
    for row in _history_chips(history):
        raw = str(row["name"])
        name = _normalize_chip(raw)
        if remaining.get(name, 0) <= 0:
            raise ValueError(f"history plays {name!r} but season grants are exhausted")
        remaining[name] = remaining[name] - 1
        played.append(ChipPlay(name=name, gw=int(row["event"])))
    return ChipInventory(remaining=remaining, played=played)


def compare_public_to_my_team(
    public: PublicSquadState,
    my_team: CurrentTeam,
) -> list[str]:
    """Return human-readable mismatch lines between public derivation and my-team.

    Compares bank, free transfers, and per-element selling prices. Empty list
    means agreement on those fields.

    Args:
        public: Result of ``public_squad_state``.
        my_team: Authenticated ``CurrentTeam`` (``source="my_team"``).

    Returns:
        One string per mismatched field.
    """
    mismatches: list[str] = []
    if public.team.bank != my_team.bank:
        mismatches.append(f"bank public={public.team.bank} my_team={my_team.bank}")
    pub_ft = public.team.free_transfers
    live_ft = my_team.free_transfers
    if pub_ft is not None and live_ft is not None and pub_ft != live_ft:
        mismatches.append(f"free_transfers public={pub_ft} my_team={live_ft}")
    pub_sell = {pick.element: pick.selling_price for pick in public.team.picks}
    live_sell = {pick.element: pick.selling_price for pick in my_team.picks}
    if set(pub_sell) != set(live_sell):
        mismatches.append(
            f"owned elements public={sorted(pub_sell)} my_team={sorted(live_sell)}"
        )
    else:
        for element, sell in pub_sell.items():
            other = live_sell[element]
            if sell != other:
                mismatches.append(
                    f"selling_price element={element} public={sell} my_team={other}"
                )
    return mismatches


def _next_free_after_chip(
    free: int,
    *,
    chip: str | None,
    moves: int,
    cap: int,
    rules: ChipRules,
) -> int:
    if chip == "wildcard":
        return 1
    if chip == "freehit":
        if rules.free_hit_keeps_free_transfers:
            return _banked_free_transfers(free, 0, cap)
        return 1
    return _banked_free_transfers(free, moves, cap)


def _banked_free_transfers(free: int, moves: int, cap: int) -> int:
    """Mirror ``plan.next_free_transfers`` (local copy avoids import cycles)."""
    if moves <= free:
        return min(cap, free - moves + 1)
    return 1


def _ft_after_pending(free: int, pending: int, cap: int) -> int:
    """Reduce FT for transfers already confirmed for the upcoming GW."""
    if pending <= 0:
        return free
    if pending <= free:
        return max(1, free - pending)
    # Hits already taken; planner still needs ge=1 for SquadState.
    del cap
    return 1


def _owned_after_pending(
    picks_payload: dict[str, Any],
    pending: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], int]:
    history_raw = picks_payload.get("entry_history")
    picks_raw = picks_payload.get("picks")
    if not isinstance(history_raw, dict) or not isinstance(picks_raw, list):
        raise FPLAPIError("entry picks need picks and entry_history")
    history = cast(dict[str, Any], history_raw)
    if "bank" not in history:
        raise FPLAPIError("entry picks need entry_history.bank")
    bank = int(history["bank"])
    # position -> pick row (mutate element as transfers apply)
    by_position: dict[int, dict[str, Any]] = {}
    for raw_row in cast(list[Any], picks_raw):
        row = cast(dict[str, Any], raw_row)
        by_position[int(row["position"])] = dict(row)
    owned = {int(row["element"]): int(row["position"]) for row in by_position.values()}
    for transfer in sorted(pending, key=lambda row: str(row.get("time") or "")):
        element_in = int(transfer["element_in"])
        element_out = int(transfer["element_out"])
        if element_out not in owned:
            raise FPLAPIError(
                f"pending transfer out element={element_out} not in last-finished squad"
            )
        position = owned.pop(element_out)
        owned[element_in] = position
        by_position[position] = {
            **by_position[position],
            "element": element_in,
            "element_type": by_position[position].get("element_type"),
        }
        bank += int(transfer["element_out_cost"]) - int(transfer["element_in_cost"])
    ordered = [by_position[pos] for pos in sorted(by_position)]
    return ordered, bank


def _purchase_prices(
    *,
    owned_elements: list[dict[str, Any]],
    transfers: list[dict[str, Any]],
    view: PITView,
    started_event: int,
    element_to_key: dict[int, int],
) -> dict[int, int]:
    # Most recent transfer-in wins.
    latest_in: dict[int, tuple[str, int]] = {}
    for row in transfers:
        element_in = int(row["element_in"])
        time_key = str(row.get("time") or "")
        cost = int(row["element_in_cost"])
        prior = latest_in.get(element_in)
        if prior is None or time_key >= prior[0]:
            latest_in[element_in] = (time_key, cost)
    deadline_prices = _prices_at_gw(view, started_event, element_to_key)
    purchase: dict[int, int] = {}
    for row in owned_elements:
        element = int(row["element"])
        if element in latest_in:
            purchase[element] = latest_in[element][1]
        elif element in deadline_prices:
            purchase[element] = deadline_prices[element]
        else:
            raise KeyError(
                f"no purchase price for element={element} "
                f"(no transfer-in and no player_status at GW{started_event})"
            )
    return purchase


def _prices_at_gw(
    view: PITView,
    gw: int,
    element_to_key: dict[int, int],
) -> dict[int, int]:
    """Map element id → ``player_status.price`` as of ``gw``'s deadline."""
    season = view.as_of.season
    gameweeks = (
        view.table("gameweeks")
        .filter((pl.col("season") == season) & (pl.col("gw") == gw))
        .select("deadline_time")
        .collect()
    )
    if gameweeks.is_empty():
        return {}
    deadline = gameweeks["deadline_time"][0]
    status = (
        view.table("player_status")
        .filter(pl.col("season") == season)
        .filter(pl.col("available_at") <= deadline)
        .sort("available_at")
        .unique(subset=["player_key"], keep="last")
        .select("player_key", "price")
        .collect()
    )
    key_to_price = {int(key): int(price) for key, price in status.iter_rows()}
    key_to_element = {key: element for element, key in element_to_key.items()}
    return {
        key_to_element[key]: price
        for key, price in key_to_price.items()
        if key in key_to_element
    }


def _now_prices(view: PITView, element_to_key: dict[int, int]) -> dict[int, int]:
    season = view.as_of.season
    status = (
        view.latest("player_status", ["season", "player_key"])
        .filter(pl.col("season") == season)
        .select("player_key", "price")
        .collect()
    )
    key_to_price = {int(key): int(price) for key, price in status.iter_rows()}
    out: dict[int, int] = {}
    for element, key in element_to_key.items():
        if key in key_to_price:
            out[element] = key_to_price[key]
    return out


def _priced_picks(
    owned_elements: list[dict[str, Any]],
    *,
    purchase_by_element: dict[int, int],
    now_by_element: dict[int, int],
) -> list[EntryPick]:
    picks: list[EntryPick] = []
    for row in owned_elements:
        element = int(row["element"])
        if element not in now_by_element:
            raise KeyError(f"no current price for element={element}")
        purchase = purchase_by_element[element]
        now = now_by_element[element]
        selling = fpl_selling_price(purchase, now)
        picks.append(
            EntryPick(
                element=element,
                position=int(row["position"]),
                selling_price=selling,
                purchase_price=purchase,
                multiplier=int(row.get("multiplier") or 0),
                is_captain=bool(row.get("is_captain", False)),
                is_vice_captain=bool(row.get("is_vice_captain", False)),
                element_type=_optional_int(row.get("element_type")),
            )
        )
    return picks


def _element_keys(view: PITView) -> dict[int, int]:
    frame = (
        view.latest("players", ["season", "player_key"])
        .filter(pl.col("season") == view.as_of.season)
        .select("element_id", "player_key")
        .collect()
    )
    return {int(element): int(key) for element, key in frame.iter_rows()}


def _history_current(history: dict[str, Any]) -> list[dict[str, Any]]:
    raw = history.get("current")
    if not isinstance(raw, list):
        raise FPLAPIError("entry history needs current[]")
    rows: list[dict[str, Any]] = []
    for item in cast(list[Any], raw):
        rows.append(cast(dict[str, Any], item))
    return rows


def _history_chips(history: dict[str, Any]) -> list[dict[str, Any]]:
    raw = history.get("chips")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise FPLAPIError("entry history chips must be a list")
    rows: list[dict[str, Any]] = []
    for item in cast(list[Any], raw):
        rows.append(cast(dict[str, Any], item))
    return rows


def _normalize_chip(name: str) -> ChipName:
    cleaned = name.strip().lower().replace(" ", "").replace("_", "")
    aliases = {
        "wildcard": "wildcard",
        "wc": "wildcard",
        "freehit": "freehit",
        "freehitchip": "freehit",
        "fh": "freehit",
        "bboost": "bboost",
        "benchboost": "bboost",
        "bb": "bboost",
        "3xc": "3xc",
        "triplecaptain": "3xc",
        "tc": "3xc",
    }
    mapped = aliases.get(cleaned)
    if mapped is None or mapped not in _CHIP_NAMES:
        raise ValueError(f"unknown chip name {name!r}")
    return cast(ChipName, mapped)


def _ft_bank_cap() -> int:
    from fplt.optimise.state import load_squad_state

    return load_squad_state().ft_bank_cap


def _optional_int(value: object) -> int | None:
    if value is None:
        return None
    return int(cast(Any, value))
