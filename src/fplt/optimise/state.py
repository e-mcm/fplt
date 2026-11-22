"""Squad state: cash, free transfers, chips, and who is already owned."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from fplt.scoring.chips import ChipName, empty_remaining

_CONFIG = Path(__file__).resolve().parents[3] / "configs" / "squad_state.yaml"


class OwnedPlayer(BaseModel):
    """A player already in the squad, with purchase and selling prices.

    ``purchase_price`` is sticky from the buy (or from ``my-team``). Selling
    price follows FPL's 50% sell-on rule via ``fplt.optimise.prices``. When
    YAML / payloads omit ``purchase_price``, it defaults to ``selling_price``
    (no unrealised gain tracked until the next market refresh).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    player_key: int
    selling_price: int = Field(ge=0)
    purchase_price: int = Field(ge=0)

    @model_validator(mode="before")
    @classmethod
    def _default_purchase_to_selling(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        payload = cast(dict[str, Any], data)
        if payload.get("purchase_price") is None and "selling_price" in payload:
            return {**payload, "purchase_price": payload["selling_price"]}
        return payload


class ChipPlay(BaseModel):
    """A chip that has already been played this season."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: ChipName
    gw: int = Field(ge=1, le=38)


class ChipInventory(BaseModel):
    """Remaining chip uses and the gameweeks they were already played."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    remaining: dict[ChipName, int] = Field(default_factory=empty_remaining)
    played: list[ChipPlay] = Field(default_factory=lambda: [])

    @model_validator(mode="after")
    def _non_negative_and_known(self) -> ChipInventory:
        expected = set(empty_remaining())
        keys = set(self.remaining)
        if keys != expected:
            raise ValueError(
                f"remaining must include exactly {sorted(expected)}, got {sorted(keys)}"
            )
        for name, count in self.remaining.items():
            if count < 0:
                raise ValueError(f"remaining[{name!r}] must be >= 0")
        return self


class SquadState(BaseModel):
    """What the planner is allowed to know about the current squad.

    An empty ``owned`` list is an initial draft: the bank is the whole
    budget and the fifteen picks are not transfers. ``chips`` stores
    remaining uses and play history; ``plan_squad`` consumes Bench Boost
    and Triple Captain when ``season`` and ``gameweeks`` are passed.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    bank: int = Field(ge=0)
    free_transfers: int = Field(ge=1)
    ft_bank_cap: int = Field(ge=1)
    hit_cost: int = Field(ge=0, default=4)
    decay: float = Field(gt=0, le=1, default=1.0)
    bench_weight: float = Field(ge=0, default=0.0)
    owned: list[OwnedPlayer] = Field(default_factory=lambda: [])
    chips: ChipInventory = Field(default_factory=ChipInventory)

    @model_validator(mode="after")
    def _transfers_fit_the_cap(self) -> SquadState:
        if self.free_transfers > self.ft_bank_cap:
            raise ValueError("free_transfers exceeds ft_bank_cap")
        if self.bench_weight >= 0.01:
            raise ValueError("bench_weight must stay below 0.01")
        return self


def rebuild_squad_state(state: SquadState) -> SquadState:
    """Clear ownership and fold selling prices into the bank.

    Used for a wildcard-style draft: pick fifteen from scratch with today's
    available funds (``bank + Σ selling_price``), ignoring transfer hits.

    Args:
        state: Current squad (often from ``resolve_squad_state``).

    Returns:
        A copy with ``owned`` empty and ``bank`` equal to the previous bank
        plus every owned selling price. Free transfers, chips, and rules are
        unchanged.
    """
    total = state.bank + sum(player.selling_price for player in state.owned)
    return state.model_copy(update={"owned": [], "bank": total})


def load_squad_state(path: Path | None = None) -> SquadState:
    """Load squad state from YAML.

    Args:
        path: File to read. Defaults to ``configs/squad_state.yaml``.

    Returns:
        The parsed state, including ``chips`` remaining and played entries.

    Raises:
        FileNotFoundError: The file is missing.
        ValueError: The document is not a mapping.
    """
    source = _CONFIG if path is None else path
    loaded: Any = yaml.safe_load(source.read_text(encoding="utf-8"))
    if not isinstance(loaded, dict):
        raise ValueError(f"{source.name} must contain a mapping")
    return SquadState.model_validate(loaded)


def apply_entry_picks(
    state: SquadState,
    payload: dict[str, Any],
    element_to_key: dict[int, int],
) -> SquadState:
    """Replace the owned squad and bank from an entry picks payload.

    Args:
        state: Overrides that stay in place (free transfers, decay, cap,
            chips).
        payload: Body of ``entry/{id}/event/{gw}/picks``. ``picks`` and
            ``entry_history.bank`` are read.
        element_to_key: FPL element id to ``player_key``.

    Returns:
        A copy with ``owned`` and ``bank`` taken from the payload. Chip
        inventory is left unchanged (still YAML / caller-owned).

    Raises:
        KeyError: A pick's element has no ``player_key``.
        ValueError: The payload is missing ``picks`` or the bank.
    """
    picks = payload.get("picks")
    history = payload.get("entry_history")
    if not isinstance(picks, list) or not isinstance(history, dict):
        raise ValueError("entry payload needs picks and entry_history")
    if "bank" not in history:
        raise ValueError("entry payload needs entry_history.bank")
    owned: list[OwnedPlayer] = []
    for pick in cast(list[dict[str, Any]], picks):
        element = int(pick["element"])
        if element not in element_to_key:
            raise KeyError(f"element {element} has no player_key")
        selling = int(pick["selling_price"])
        raw_purchase = pick.get("purchase_price")
        purchase = selling if raw_purchase is None else int(raw_purchase)
        owned.append(
            OwnedPlayer(
                player_key=element_to_key[element],
                selling_price=selling,
                purchase_price=purchase,
            )
        )
    bank = int(cast(dict[str, Any], history)["bank"])
    return state.model_copy(update={"owned": owned, "bank": bank})
