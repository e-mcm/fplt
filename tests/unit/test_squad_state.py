"""SquadState YAML load and chip inventory round-trip."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from fplt.optimise.state import (
    ChipInventory,
    ChipPlay,
    OwnedPlayer,
    SquadState,
    load_squad_state,
    rebuild_squad_state,
)
from fplt.scoring.chips import ChipRules


def test_default_squad_state_reads_chips() -> None:
    state = load_squad_state()
    assert state.chips.remaining == {
        "wildcard": 2,
        "freehit": 2,
        "bboost": 2,
        "3xc": 2,
    }
    assert state.chips.played == []
    season_counts = ChipRules.for_season("2025-26").remaining_counts()
    assert state.chips.remaining == season_counts


def test_squad_state_round_trip_preserves_chips(tmp_path: Path) -> None:
    original = SquadState(
        bank=120,
        free_transfers=2,
        ft_bank_cap=5,
        hit_cost=4,
        decay=0.9,
        bench_weight=0.0001,
        owned=[OwnedPlayer(player_key=1, selling_price=55)],
        chips=ChipInventory(
            remaining={
                "wildcard": 1,
                "freehit": 2,
                "bboost": 1,
                "3xc": 2,
            },
            played=[
                ChipPlay(name="wildcard", gw=8),
                ChipPlay(name="bboost", gw=15),
            ],
        ),
    )
    path = tmp_path / "squad_state.yaml"
    path.write_text(
        yaml.safe_dump(original.model_dump(mode="python"), sort_keys=False),
        encoding="utf-8",
    )
    loaded = load_squad_state(path)
    assert loaded == original
    assert loaded.chips.played[0].name == "wildcard"
    assert loaded.chips.played[1].gw == 15


def test_rebuild_keeps_chip_inventory() -> None:
    state = SquadState(
        bank=10,
        free_transfers=1,
        ft_bank_cap=5,
        owned=[OwnedPlayer(player_key=7, selling_price=40)],
        chips=ChipInventory(
            remaining={
                "wildcard": 1,
                "freehit": 2,
                "bboost": 2,
                "3xc": 2,
            },
            played=[ChipPlay(name="wildcard", gw=3)],
        ),
    )
    rebuilt = rebuild_squad_state(state)
    assert rebuilt.owned == []
    assert rebuilt.bank == 50
    assert rebuilt.chips == state.chips


def test_incomplete_remaining_is_rejected() -> None:
    with pytest.raises(ValidationError, match="remaining"):
        ChipInventory.model_validate({"remaining": {"wildcard": 1}, "played": []})


def test_negative_remaining_is_rejected() -> None:
    with pytest.raises(ValidationError, match="remaining"):
        ChipInventory.model_validate(
            {
                "remaining": {
                    "wildcard": -1,
                    "freehit": 0,
                    "bboost": 0,
                    "3xc": 0,
                },
                "played": [],
            }
        )


def test_chips_default_when_omitted() -> None:
    state = SquadState(bank=1000, free_transfers=1, ft_bank_cap=5)
    assert state.chips.remaining == {
        "wildcard": 0,
        "freehit": 0,
        "bboost": 0,
        "3xc": 0,
    }
    assert state.chips.played == []
